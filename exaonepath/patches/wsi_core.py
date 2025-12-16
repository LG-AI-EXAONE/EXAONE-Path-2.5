"""WSI patch extraction utilities.

Provides `SlideImage` with:
- tissue segmentation (`segment_regions`)
- segmentation visualization (`render_segmentation`)
- coordinate extraction to HDF5 (`write_patches`)
"""
from __future__ import annotations

import os
import math
from typing import List

import numpy as np
import cv2
import h5py
import openslide
from PIL import Image

# No CSV/DataFrame helpers in this module.


class SlideImage:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.patient_id = os.path.basename(os.path.dirname(path))
        self.wsi = openslide.open_slide(path)
        try:
            self.wsi.set_cache(openslide.OpenSlideCache(0))
        except Exception:
            pass
        # Normalize level info
        self.level_dim = self.wsi.level_dimensions
        self.level_downsamples = self._calc_level_downsamples()
        self.contours_tissue: List[np.ndarray] = []
        self.holes_tissue: List[List[np.ndarray]] = []

    def _calc_level_downsamples(self):
        """Compute per-level downsamples in a way consistent with OpenSlide metadata.

        Some slides report `wsi.level_downsamples` as scalars, while the true
        X/Y downsample inferred from dimensions can be slightly non-integer.
        We follow the same rule used in the existing pipeline:
        - if inferred (sx, sy) matches (ds, ds), use (ds, ds)
        - otherwise use the inferred pair
        """
        downs = []
        dim0 = self.wsi.level_dimensions[0]
        for ds, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            est = (dim0[0] / float(dim[0]), dim0[1] / float(dim[1]))
            if est == (ds, ds):
                downs.append((ds, ds))
            else:
                downs.append(est)
        return downs

    def get_slide(self):
        return self.wsi

    def initSegmentation(self, mask_file):
        # Optional compatibility hook if external masks are provided
        import pickle
        with open(mask_file, 'rb') as f:
            asset = pickle.load(f)
        self.holes_tissue = asset.get('holes', [])
        self.contours_tissue = asset.get('tissue', [])

    def saveSegmentation(self, mask_file):
        import pickle
        with open(mask_file, 'wb') as f:
            pickle.dump({'holes': self.holes_tissue, 'tissue': self.contours_tissue}, f)

    @staticmethod
    def _filter_contours(contours, hierarchy_2col, a_t=100, a_h=16, max_n_holes=8):
        """Filter contours by area and keep up to N holes per contour.

        `hierarchy_2col` is expected to be `hierarchy[:, 2:]` from OpenCV RETR_CCOMP,
        i.e. columns: [child, parent].
        """
        filtered_idx = []
        hole_groups = []
        # external contours have parent == -1 (col 1)
        parent_idx = np.flatnonzero(hierarchy_2col[:, 1] == -1)
        for pi in parent_idx:
            cont = contours[pi]
            # holes are contours whose parent == pi
            hole_ids = np.flatnonzero(hierarchy_2col[:, 1] == pi)
            a = cv2.contourArea(cont)
            if hole_ids.size:
                a -= np.sum([cv2.contourArea(contours[c]) for c in hole_ids])
            if a <= 0:
                continue
            if a >= a_t:
                filtered_idx.append(pi)
                # top-N holes by area
                unfiltered = [contours[c] for c in hole_ids]
                unfiltered.sort(key=cv2.contourArea, reverse=True)
                kept = [h for h in unfiltered[:max_n_holes] if cv2.contourArea(h) > a_h]
                hole_groups.append(kept)
        return [contours[i] for i in filtered_idx], hole_groups

    @staticmethod
    def _scale_contours(contours, scale):
        if contours is None:
            return []
        sx, sy = scale
        out = []
        for c in contours:
            pts = c.reshape(-1, 2)
            # Match the existing pipeline behavior: implicit truncation via int cast
            # (no rounding). This is important for exact coordinate parity.
            pts = (pts * np.array([sx, sy])).astype(np.int32)
            out.append(pts.reshape(-1, 1, 2))
        return out

    def segment_regions(self,
                      seg_level=0,
                      sthresh=20,
                      sthresh_up=255,
                      mthresh=7,
                      close=0,
                      use_otsu=False,
                      filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8},
                      ref_patch_size=512,
                      exclude_ids=None,
                      keep_ids=None,
                      seg_downsample=1.0):
        """Segment tissue regions to build foreground contours."""
        # Read level image and optionally downsample for speed
        raw = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))
        if seg_downsample is None:
            seg_downsample = 1.0
        scale_ds = float(seg_downsample) if seg_downsample and seg_downsample > 1.0 else 1.0
        if scale_ds > 1.0:
            new_w = max(1, int(raw.shape[1] / scale_ds))
            new_h = max(1, int(raw.shape[0] / scale_ds))
            img = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img = raw
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Pre-processing to suppress border/low-texture regions.
        def _draw_white_bands(rgb, thickness=20):
            h, w = rgb.shape[:2]
            cv2.rectangle(rgb, (0, 0), (w, thickness), (255, 255, 255), -1)
            cv2.rectangle(rgb, (0, h - thickness), (w, h), (255, 255, 255), -1)
            cv2.rectangle(rgb, (0, 0), (thickness, h), (255, 255, 255), -1)
            cv2.rectangle(rgb, (w - thickness, 0), (w, h), (255, 255, 255), -1)
            return rgb

        # Operate on a copy to avoid side effects
        img_rgb = _draw_white_bands(img_rgb, thickness=20)

        # Remove low-texture (monotone) regions via Laplacian threshold
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        lap_abs = cv2.convertScaleAbs(lap)
        mono_mask = lap_abs <= 15
        img_rgb[mono_mask] = (255, 255, 255)

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        ksize = int(mthresh) if int(mthresh) % 2 == 1 else int(mthresh) + 1
        s_med = cv2.medianBlur(s, ksize)

        if use_otsu:
            _, mask = cv2.threshold(s_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(s_med, int(sthresh), sthresh_up, cv2.THRESH_BINARY)

        def _post_mask(mk):
            if close and close > 0:
                kernel = np.ones((int(close), int(close)), np.uint8)
                mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, kernel)
            return mk

        mask = _post_mask(mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours or hierarchy is None:
            self.contours_tissue = []
            self.holes_tissue = []
            return

        hierarchy = np.squeeze(hierarchy, axis=0)[:, 2:]  # [child, parent]

        base_scale = self.level_downsamples[seg_level]
        # Account for additional resize from seg_downsample
        coord_scale = (base_scale[0] * scale_ds, base_scale[1] * scale_ds)
        # Scale filter thresholds relative to level scale (mask pixel area -> level-0 area)
        scaled_area = max(1, int(ref_patch_size ** 2 / (coord_scale[0] * coord_scale[1])))
        f_a_t = int(filter_params.get('a_t', 100)) * scaled_area
        f_a_h = int(filter_params.get('a_h', 16)) * scaled_area
        max_holes = int(filter_params.get('max_n_holes', 8))

        fg_contours, hole_groups = self._filter_contours(contours, hierarchy, a_t=f_a_t, a_h=f_a_h, max_n_holes=max_holes)

        # Scale back to level 0
        self.contours_tissue = self._scale_contours(fg_contours, coord_scale)
        self.holes_tissue = [self._scale_contours(hs, coord_scale) for hs in hole_groups]
        # Apply include/exclude if provided
        n = len(self.contours_tissue)
        ids = set(range(n))
        if keep_ids:
            ids = set(keep_ids) - set(exclude_ids or [])
        elif exclude_ids:
            ids = ids - set(exclude_ids)
        self.contours_tissue = [self.contours_tissue[i] for i in sorted(ids)]
        self.holes_tissue = [self.holes_tissue[i] for i in sorted(ids)]

    def render_segmentation(
        self,
        vis_level=0,
        color=(0, 255, 0),
        hole_color=(0, 0, 255),
        line_thickness=250,
        top_left=None,
        bot_right=None,
        view_slide_only=False,
        seg_display=True,
    ):
        scale = [1 / self.level_downsamples[vis_level][0], 1 / self.level_downsamples[vis_level][1]]
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0, 0)
            region_size = self.level_dim[vis_level]
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        if not view_slide_only and self.contours_tissue and seg_display:
            offset = tuple(-(np.array(top_left) * np.array(scale)).astype(int))
            lt = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            for cont in self.contours_tissue:
                contour = (cont * np.array(scale)).astype(np.int32)
                cv2.drawContours(img, [contour], -1, color, lt, lineType=cv2.LINE_8, offset=offset)

            # Draw holes (if any) to match the mask appearance
            if getattr(self, 'holes_tissue', None):
                for holes in self.holes_tissue:
                    if not holes:
                        continue
                    holes_scaled = [(h * np.array(scale)).astype(np.int32) for h in holes]
                    cv2.drawContours(img, holes_scaled, -1, hole_color, lt, lineType=cv2.LINE_8)
        return Image.fromarray(img)

    def write_patches(
        self,
        patch_level: int,
        patch_size: int,
        step_size: int,
        save_path: str,
        use_padding: bool = True,
        contour_fn: str = "four_pt",
        # Unused (kept for API stability)
        white_thresh: int = 15,
        black_thresh: int = 50,
    ) -> str:
        """Extract patch coordinates inside segmented tissue and save to HDF5.

        The output HDF5 contains:
        - coords: int32 [N, 2] (level-0 pixel coordinates)
        - contour_index: int32 [N]
        - file attrs: complete=True when finished
        """

        name = self.name
        downsample = self.level_downsamples[patch_level]
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{name}.h5")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # if file exists and complete, return
        if os.path.isfile(file_path):
            try:
                with h5py.File(file_path, 'r') as f:
                    if bool(f.attrs.get('complete', False)):
                        return file_path
            except Exception:
                pass

        # initialize/overwrite
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass

        with h5py.File(file_path, 'w') as f:
            coords = f.create_dataset('coords', shape=(0, 2), maxshape=(None, 2), chunks=(1024, 2), dtype=np.int32)
            # Store metadata on coords for downstream tools.
            coords.attrs['patch_size'] = int(patch_size)
            coords.attrs['patch_level'] = int(patch_level)
            coords.attrs['downsample'] = downsample
            coords.attrs['downsampled_level_dim'] = tuple(np.array(self.level_dim[patch_level]))
            coords.attrs['level_dim'] = tuple(np.array(self.level_dim[patch_level]))
            coords.attrs['name'] = name

            cid = f.create_dataset('contour_index', shape=(0,), maxshape=(None,), chunks=(1024,), dtype=np.int32)
            cid.attrs['name'] = name

        # nothing to do
        if not self.contours_tissue:
            with h5py.File(file_path, 'a') as f:
                f.attrs['complete'] = True
                f.attrs['n_coords'] = 0
            return file_path

        def _cont_check_fn(cont: np.ndarray, ref_patch: int):
            if contour_fn == 'four_pt':
                shift = int(ref_patch // 2 * 0.5)

                def _fn(pt):
                    cx, cy = pt[0] + ref_patch // 2, pt[1] + ref_patch // 2
                    pts = [
                        (cx - shift, cy - shift),
                        (cx + shift, cy + shift),
                        (cx + shift, cy - shift),
                        (cx - shift, cy + shift),
                    ]
                    for p in pts:
                        if cv2.pointPolygonTest(cont, (float(p[0]), float(p[1])), False) >= 0:
                            return True
                    return False

                return _fn

            if contour_fn == 'center':
                def _fn(pt):
                    cx, cy = pt[0] + ref_patch // 2, pt[1] + ref_patch // 2
                    return cv2.pointPolygonTest(cont, (float(cx), float(cy)), False) >= 0

                return _fn

            if contour_fn == 'basic':
                def _fn(pt):
                    return cv2.pointPolygonTest(cont, (float(pt[0]), float(pt[1])), False) >= 0

                return _fn

            raise ValueError(f"Unsupported contour_fn: {contour_fn}")

        def _in_holes(holes, pt, ref_patch: int) -> bool:
            if not holes:
                return False
            cx, cy = pt[0] + ref_patch / 2.0, pt[1] + ref_patch / 2.0
            for hole in holes:
                if cv2.pointPolygonTest(hole, (float(cx), float(cy)), False) > 0:
                    return True
            return False

        patch_downsample = (
            int(self.level_downsamples[patch_level][0]),
            int(self.level_downsamples[patch_level][1]),
        )
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])
        step_size_x = int(step_size * patch_downsample[0])
        step_size_y = int(step_size * patch_downsample[1])

        img_w, img_h = self.level_dim[0]

        total_written = 0
        with h5py.File(file_path, 'a') as f:
            coords_d = f['coords']
            cid_d = f['contour_index']

            for ci, cont in enumerate(self.contours_tissue):
                start_x, start_y, w, h = cv2.boundingRect(cont)
                if use_padding:
                    stop_x = start_x + w
                    stop_y = start_y + h
                else:
                    stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)
                    stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)

                checker = _cont_check_fn(cont, ref_patch_size[0])
                holes = self.holes_tissue[ci] if getattr(self, 'holes_tissue', None) else None

                xs = np.arange(start_x, stop_x, step=step_size_x)
                ys = np.arange(start_y, stop_y, step=step_size_y)

                for y in ys:
                    for x in xs:
                        pt = (int(x), int(y))
                        if not checker(pt):
                            continue
                        if holes is not None and _in_holes(holes, pt, ref_patch_size[0]):
                            continue

                        n0 = len(coords_d)
                        coords_d.resize(n0 + 1, axis=0)
                        coords_d[n0] = (pt[0], pt[1])
                        cid_d.resize(n0 + 1, axis=0)
                        cid_d[n0] = int(ci)
                        total_written += 1

            f.attrs['complete'] = True
            f.attrs['n_coords'] = int(total_written)

        return file_path

__all__ = ["SlideImage"]

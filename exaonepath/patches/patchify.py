"""Functional patch extraction API.

Primary entrypoint: `patchfy()`.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import openslide
from PIL import Image

from .wsi_core import SlideImage

Image.MAX_IMAGE_PIXELS = None


def _slide_id_from_path(wsi_path: str) -> str:
    """Return slide_id (filename without extension) from a WSI path."""
    fname = os.path.basename(wsi_path)
    slide_id, _ = os.path.splitext(fname)
    return slide_id


def _h5_is_complete(h5_path: str) -> bool:
    if not os.path.isfile(h5_path):
        return False
    import h5py

    try:
        with h5py.File(h5_path, 'r') as f:
            complete = bool(f.attrs.get('complete', False))
            has_coords = 'coords' in f and len(f['coords']) > 0
            return complete and has_coords
    except Exception:
        return False


def _ensure_slide_image(wsi: Union[str, os.PathLike, SlideImage]) -> Tuple[SlideImage, str]:
    """Normalize WSI input into a SlideImage + path string."""
    if isinstance(wsi, SlideImage):
        return wsi, wsi.path
    wsi_path = os.fspath(wsi)
    return SlideImage(wsi_path), wsi_path


def run_pipeline(
    wsi_path: str,
    out_dir: str,
    patch_size=256,
    step_size=256,
    patch_level=0,
    save_mask=False,
    save_h5=True,
    auto_skip=True,
    seg_downsample: float = 1.0,
    max_seg_pixels: float | None = 4e10,
    # MPP normalization (always enabled)
    standard_mpp: float = 0.5,
    assumed_mpp: float = 0.5,
    # Segmentation/Filtering knobs
    sthresh: int = 8,
    mthresh: int = 7,
    close: int = 4,
    use_otsu: bool = False,
    a_t: int = 1,
    a_h: int = 1,
    max_n_holes: int = 100,
    # Visualization knob (only used when save_mask=True)
    line_thickness: int = 100,
):
    """Internal pipeline for single-WSI patch extraction."""
    # Prepare output dirs
    patch_save_dir = os.path.join(out_dir, 'patches')
    mask_save_dir = os.path.join(out_dir, 'masks')
    os.makedirs(patch_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # Auto-skip existing output
    slide_id = _slide_id_from_path(wsi_path)
    canonical_path = os.path.join(patch_save_dir, slide_id + '.h5')

    if auto_skip:
        if _h5_is_complete(canonical_path):
            return None, None
        else:
            if os.path.isfile(canonical_path):
                try:
                    os.remove(canonical_path)
                except Exception:
                    pass
    else:
        # If the caller explicitly disables auto_skip, regenerate outputs.
        # This is important when parameters change (e.g., MPP-normalized patch_size/step_size).
        if os.path.isfile(canonical_path):
            try:
                os.remove(canonical_path)
            except Exception:
                pass

    WSI_object = SlideImage(wsi_path)

    # --- Always normalize patch_size/step_size by MPP (match float crop mode) ---
    wsi = WSI_object.get_slide()
    props = getattr(wsi, 'properties', {}) or {}
    this_mpp = None
    for key in (getattr(openslide, 'PROPERTY_NAME_MPP_X', 'openslide.mpp-x'), 'openslide.mpp-x', 'aperio.MPP'):
        try:
            if key in props:
                this_mpp = float(props[key])
                break
        except Exception:
            this_mpp = None
    if this_mpp is None or not np.isfinite(this_mpp) or this_mpp <= 0:
        # objective-power heuristic: 10x->1.0, 20x->0.5, 40x->0.25
        try:
            obj = props.get('openslide.objective-power')
            if obj is not None:
                obj = float(obj)
                if obj > 0:
                    this_mpp = 10.0 / obj
        except Exception:
            this_mpp = None
    if this_mpp is None or not np.isfinite(this_mpp) or this_mpp <= 0:
        this_mpp = float(assumed_mpp)

    crop_factor = float(standard_mpp) / float(this_mpp)
    # Keep behavior identical to the original float-crop branch: int() truncation
    eff_patch_size = int(int(patch_size) * crop_factor)
    eff_step_size = int(int(step_size) * crop_factor)
    if eff_patch_size <= 0:
        eff_patch_size = int(patch_size)
    if eff_step_size <= 0:
        eff_step_size = int(step_size)

    print(f"mpp: {this_mpp} (assumed_mpp={assumed_mpp}, standard_mpp={standard_mpp})")
    print(f"patch_size: {eff_patch_size}  step_size: {eff_step_size}")

    # Determine visualization/segmentation levels
    vis_level = wsi.get_best_level_for_downsample(64)
    seg_level = wsi.get_best_level_for_downsample(64)

    w, h = WSI_object.level_dim[seg_level]
    if max_seg_pixels is not None and (w * h) > float(max_seg_pixels):
        # Make the skip explicit so downstream knows this slide was not processed.
        slide_id = _slide_id_from_path(wsi_path)
        reason = (
            f"SKIP: seg_level image too large (w*h={w*h:.3g}, w={w}, h={h}, "
            f"seg_level={seg_level}, max_seg_pixels={float(max_seg_pixels):.3g})"
        )
        print(reason)
        try:
            skipped_dir = os.path.join(out_dir, 'skipped')
            os.makedirs(skipped_dir, exist_ok=True)
            with open(os.path.join(skipped_dir, slide_id + '.txt'), 'w', encoding='utf-8') as f:
                f.write(reason + "\n")
                f.write(wsi_path + "\n")
        except Exception:
            pass
        return None, None

    # Segmentation
    import time as _t

    t0 = _t.time()
    try:
        WSI_object.segment_regions(
            seg_level=seg_level,
            sthresh=sthresh,
            mthresh=mthresh,
            close=close,
            use_otsu=use_otsu,
            filter_params={'a_t': a_t, 'a_h': a_h, 'max_n_holes': max_n_holes},
            seg_downsample=seg_downsample,
        )
    except Exception:
        # fallback to a coarser level if needed
        fallback_level = min(2, len(WSI_object.level_dim) - 1)
        WSI_object.segment_regions(
            seg_level=fallback_level,
            sthresh=sthresh,
            mthresh=mthresh,
            close=close,
            use_otsu=use_otsu,
            filter_params={'a_t': a_t, 'a_h': a_h, 'max_n_holes': max_n_holes},
            seg_downsample=seg_downsample,
        )
    seg_time_elapsed = _t.time() - t0
    if len(WSI_object.contours_tissue) == 0:
        # Still allow saving a visualization (will be slide-only if no contours).
        if save_mask:
            try:
                mask = WSI_object.render_segmentation(vis_level=vis_level, line_thickness=line_thickness)
                seg_ds = float(seg_downsample or 1.0)
                if seg_ds > 1.0:
                    new_w = max(1, int(mask.width / seg_ds))
                    new_h = max(1, int(mask.height / seg_ds))
                    mask = mask.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
                mask.save(os.path.join(mask_save_dir, slide_id + '.jpg'))
            except Exception:
                pass
        return None, None

    if save_mask:
        mask = WSI_object.render_segmentation(vis_level=vis_level, line_thickness=line_thickness)
        seg_ds = float(seg_downsample or 1.0)
        if seg_ds > 1.0:
            new_w = max(1, int(mask.width / seg_ds))
            new_h = max(1, int(mask.height / seg_ds))
            mask = mask.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        mask.save(os.path.join(mask_save_dir, slide_id + '.jpg'))

    # Patch extraction
    t0 = _t.time()
    file_path = None
    if save_h5:
        file_path = WSI_object.write_patches(
            patch_level=patch_level,
            patch_size=int(eff_patch_size),
            step_size=int(eff_step_size),
            save_path=patch_save_dir,
            use_padding=True,
            contour_fn='four_pt',
        )
        # The writer returns the concrete H5 path. Ensure it's the canonical location.
        if file_path != canonical_path:
            # Don't fail hard, but keep behavior deterministic.
            file_path = canonical_path
        if not _h5_is_complete(file_path):
            print(f"warning: generated file missing completion marker: {file_path}")
    patch_time_elapsed = _t.time() - t0

    # Print elapsed times
    print(f"segmentation time: {seg_time_elapsed:.2f}s")
    print(f"patch extraction time: {patch_time_elapsed:.2f}s")

    # Return coords/contour_index from the written HDF5.
    if file_path is None:
        return None, None

    try:
        import h5py

        with h5py.File(file_path, 'r') as f:
            coords = f['coords'][...]
            contour_indices = f['contour_index'][...] if 'contour_index' in f else None
        return coords, contour_indices
    except Exception:
        return None, None


def patchfy(
    wsi: Union[str, os.PathLike, SlideImage],
    out: str,
    step_size: int = 256,
    patch_size: int = 256,
    patch_level: int = 0,
    save_mask: bool = False,
    save_h5: bool = True,
    auto_skip: bool = True,
    seg_downsample: float = 1.0,
    max_seg_pixels: Optional[float] = 4e10,
    # Segmentation/Filtering knobs
    sthresh: int = 8,
    mthresh: int = 7,
    close: int = 4,
    use_otsu: bool = False,
    a_t: int = 1,
    a_h: int = 1,
    max_n_holes: int = 100,
    # Visualization knob
    line_thickness: int = 100,
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """High-level API for single-WSI patch extraction.

    This function is designed to be imported and called from other Python code.

    Parameters
    ----------
    wsi:
        Either a WSI path (str / PathLike) or an already-opened `SlideImage`.

    out:
        Output directory. Two subfolders may be created:
        - `<out>/patches` for HDF5
        - `<out>/masks` for segmentation visualization (when `save_mask=True`)

    step_size:
        Stride between patch coordinates in level-0 pixel space.

    patch_size:
        Patch edge length in pixels (requested). The effective values written to H5
        are MPP-normalized inside `run_pipeline`.

    patch_level:
        OpenSlide pyramid level to extract patches from. 0 means full resolution.

    save_mask:
        If True, saves a segmentation visualization image to `<out>/masks/<slide_id>.jpg`.

    save_h5:
        If True, writes an HDF5 file to `<out>/patches/<slide_id>.h5`.

    auto_skip:
        If True, and a previous output HDF5 exists with `attrs['complete']=True` and non-empty coords,
        the slide is skipped.

    seg_downsample:
        Additional downsampling factor used *during segmentation only* to reduce computation.
        - 1.0 means no extra downsample.
        - >1.0 means resize segmentation image by this factor (faster, potentially less accurate).

    max_seg_pixels:
        Safety cap for segmentation image size at the chosen `seg_level`.
        If `w*h` at `seg_level` exceeds this, the slide is skipped and a reason is written to
        `<out>/skipped/<slide_id>.txt`. Set `<=0` to disable this check.

    sthresh:
        Tissue segmentation threshold on the *S channel* (saturation in HSV) used by
        `SlideImage.segment_regions`. Lower values generally mark more pixels as tissue.

    mthresh:
        Median blur kernel size applied to the saturation mask before thresholding.
        Must be an integer; even values will be made odd inside `segment_regions`.

    close:
        Morphological closing kernel size (in pixels) applied to the binary mask.
        Use 0 to disable closing.

    use_otsu:
        If True, use Otsu thresholding instead of fixed `sthresh` during saturation thresholding.

    a_t:
        Minimum tissue contour area threshold (baseline) passed via `filter_params['a_t']`.
        Note: `SlideImage.segment_regions` internally rescales this threshold based on the pyramid
        level and `seg_downsample` so this is a *baseline* knob rather than a strict level-0 area.

    a_h:
        Minimum hole area threshold (baseline) passed via `filter_params['a_h']`.
        Same rescaling behavior as `a_t`.

    max_n_holes:
        Maximum number of holes to keep per tissue contour (largest holes by area).

    line_thickness:
        Line thickness used when rendering the segmentation visualization mask.
        Only used if `save_mask=True`.

    Returns
    -------
    (h5_path, coords, contour_indices)
        If skipped/failed to segment, returns (None, None, None).

        - `h5_path`: canonical output path under `<out>/patches/<slide_id>.h5` when `save_h5=True`.
        - `coords`: Nx2 array of patch (x, y) coordinates in level-0 pixels.
        - `contour_indices`: N array with the tissue contour index for each coordinate.
    """
    os.makedirs(out, exist_ok=True)

    # Normalize disabling behavior
    if max_seg_pixels is not None and max_seg_pixels <= 0:
        max_seg_pixels = None

    WSI_object, wsi_path = _ensure_slide_image(wsi)
    slide_id = getattr(WSI_object, 'name', None) or _slide_id_from_path(wsi_path)
    patch_save_dir = os.path.join(out, 'patches')
    h5_path = os.path.join(patch_save_dir, slide_id + '.h5') if save_h5 else None

    coords, contour_indices = run_pipeline(
        wsi_path=wsi_path,
        out_dir=out,
        patch_size=patch_size,
        step_size=step_size,
        patch_level=patch_level,
        save_mask=save_mask,
        save_h5=save_h5,
        auto_skip=auto_skip,
        seg_downsample=seg_downsample,
        max_seg_pixels=max_seg_pixels,
        sthresh=sthresh,
        mthresh=mthresh,
        close=close,
        use_otsu=use_otsu,
        a_t=a_t,
        a_h=a_h,
        max_n_holes=max_n_holes,
        line_thickness=line_thickness,
    )

    if coords is None or contour_indices is None:
        return None, None, None
    return h5_path, coords, contour_indices


__all__ = [
    'patchfy',
    'run_pipeline',
    'SlideImage',
]

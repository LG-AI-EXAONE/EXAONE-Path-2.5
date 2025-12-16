"""Extract patch-level features from a single WSI.

This script reads patch coordinates from an HDF5 file (typically produced by
`exaonepath.patches.patchfy`) and runs the EXAONE Path patch encoder on the
corresponding image regions to produce patch embeddings.

Input H5 (`--coords_h5_path`) requirements
-----------------------------------------
- Dataset: `coords` (shape: [N, 2]) containing (x, y) coordinates in **level-0** pixel space.
- Attribute on `coords`: `patch_size` (int). If missing, defaults to 256.
- Optional dataset: `contour_index` (shape: [N]). If missing, all patches are treated as contour 0.

Output H5 (`--out_h5_path`) keys
--------------------------------
- `features`: [N, C] patch embeddings
- `coords`: [N, 2] coordinates (copied from input)
- `contour_index`: [N] contour indices (copied from input or synthesized)

Notes
-----
- CUDA is required.
- This implementation loads all extracted features into memory before writing.
"""

import argparse
import os

import h5py
import numpy as np
import openslide
import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast
from torchvision import transforms

from transformers import AutoModel

def _open_coords_h5(coords_h5_path):
    """Open coordinates H5 file and read coordinates."""
    if not os.path.exists(coords_h5_path):
        print(f"Coords H5 file not found: {coords_h5_path}")
        return None, None, None

    with h5py.File(coords_h5_path, "r") as f:
        if "coords" not in f:
            print(f"Invalid coords H5 (missing 'coords'): {coords_h5_path}")
            return None, None, None

        coords = np.array(f["coords"])
        patch_size = int(f["coords"].attrs.get("patch_size", 256))
        if "contour_index" in f:
            contour_index = np.array(f["contour_index"])
        else:
            # Backward-compat: if contour_index isn't present, treat all patches as one contour.
            contour_index = np.zeros((len(coords),), dtype=np.int32)

    return coords, patch_size, contour_index


def _save_slide_features(out_h5_path, feat_list, coord_list, contour_list):
    os.makedirs(os.path.dirname(os.path.abspath(out_h5_path)), exist_ok=True)
    feat_tensor = torch.cat(feat_list, dim=0).numpy()
    coords_tensor = torch.cat(coord_list, dim=0).numpy()
    contour_tensor = torch.cat(contour_list, dim=0).numpy()

    with h5py.File(out_h5_path, "w") as f:
        f.create_dataset("features", data=feat_tensor)
        f.create_dataset("coords", data=coords_tensor)
        f.create_dataset("contour_index", data=contour_tensor)


def process_single_slide(
    slide_path,
    coords_h5_path,
    out_h5_path,
    model,
    transform,
    batch_size_per_gpu=32,
):
    device = "cuda"
    print(f"Processing slide: {slide_path}")
    print(f"Coords H5: {coords_h5_path}")
    print(f"Output H5: {out_h5_path}")

    # Load coordinates
    coords, patch_size, contour_index = _open_coords_h5(coords_h5_path)
    if coords is None or contour_index is None:
        print("No coordinates loaded; aborting.")
        return

    if len(coords) == 0:
        print("No coordinates found (N=0); nothing to do.")
        return

    # Open WSI
    wsi = openslide.OpenSlide(slide_path)

    batch_images = []
    batch_coords = []
    batch_contours = []

    feat_list = []
    coord_list = []
    contour_list = []

    total_patches = len(coords)

    # Iterate through all patches
    for j, (coord, contour_val) in enumerate(zip(coords, contour_index)):
        x, y = int(coord[0]), int(coord[1])

        try:
            patch = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch = transform(patch)
        except Exception as e:
            print(
                f"Error extracting patch at index {j} (coord: {coord}) from slide {slide_path}: {e}"
            )
            continue

        batch_images.append(patch)
        batch_coords.append(coord)
        batch_contours.append(contour_val)

        # Process batch if full or last item
        if len(batch_images) >= batch_size_per_gpu or j == total_patches - 1:
            imgs_tensor = torch.stack(batch_images).to(device, non_blocking=True)

            # Inference & store results (GPU-only)
            with torch.inference_mode(), autocast("cuda", torch.bfloat16):
                features = model(imgs_tensor)

            features = features.detach().float().cpu()
            feat_list.append(features)
            coord_list.append(torch.tensor(np.array(batch_coords)))
            contour_list.append(torch.tensor(np.array(batch_contours)))

            batch_images = []
            batch_coords = []
            batch_contours = []

            # Log every ~1000 patches (use 1-based count for readability).
            if (j + 1) % 1000 == 0 or (j + 1) == total_patches:
                print(f"Processed {j + 1}/{total_patches} patches...")

    wsi.close()

    # Save results
    if feat_list:
        _save_slide_features(out_h5_path, feat_list, coord_list, contour_list)
    else:
        print(f"No features extracted for {slide_path}")


def main(args):
    # GPU-only execution (as required by the model release).
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this script, but torch.cuda.is_available() is False"
        )

    device = "cuda"
    cudnn.benchmark = True

    # Load model
    print("Loading model...")
    repo_id = "LGAI-EXAONE/EXAONE-Path-2.5"

    model = AutoModel.from_pretrained(
        repo_id,
        subfolder="patch-encoder",
        trust_remote_code=True,
    ).to(device).eval()

    # Load transform
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Process input slide
    if args.out_h5_path:
        out_h5_path = args.out_h5_path
    else:
        # Default: place next to coords file with a clear suffix.
        base, _ = os.path.splitext(args.coords_h5_path)
        out_h5_path = base + "_features.h5"

    print("Extracting patch features ...")
    process_single_slide(
        args.slide_path,
        args.coords_h5_path,
        out_h5_path,
        model,
        transform,
        args.batch_size_per_gpu,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Single WSI patch-feature extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--slide_path",
        type=str,
        required=True,
        help="Path to a whole-slide image file (.svs/.tif/.tiff/.ndpi/.mrxs/...).",
    )
    parser.add_argument(
        "--coords_h5_path",
        type=str,
        required=True,
        help=(
            "Path to the coordinates HDF5 produced by `patchfy` (must contain dataset 'coords' "
            "and ideally 'contour_index')."
        ),
    )
    parser.add_argument(
        "--out_h5_path",
        type=str,
        default="",
        help=(
            "Output HDF5 path for patch features. If empty, defaults to '<coords_h5_path>_features.h5'."
        ),
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=32,
        help="Batch size for patch encoder inference on a single GPU.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="LGAI-EXAONE/EXAONE-Path-2.5",
        help="Hugging Face repo id containing the EXAONE Path 2.5 patch encoder.",
    )

    args = parser.parse_args()
    main(args)

import argparse
import contextlib
import gc
import io
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
import sam2.sam2_video_predictor as sam2_video_predictor
import sam2.utils.misc as sam2_misc

import vot_compat as vot


def _quiet_tqdm(iterable=None, *args, **kwargs):
    return iterable if iterable is not None else []


sam2_video_predictor.tqdm = _quiet_tqdm
sam2_misc.tqdm = _quiet_tqdm


def select_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_autocast_context(device):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def determine_model_cfg(model_path):
    model_name = model_path.lower()
    if "large" in model_name:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in model_name or "b+" in model_name:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in model_name or "_s" in model_name:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in model_name or "_t" in model_name:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError(f"Cannot infer config from checkpoint path: {model_path}")


def region_to_bbox(region):
    if hasattr(region, "width") and hasattr(region, "height"):
        return [
            int(region.x),
            int(region.y),
            int(region.width),
            int(region.height),
        ]

    if hasattr(region, "points"):
        points = np.array([(p.x, p.y) for p in region.points], dtype=np.float32)
        x_min = int(np.min(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        x_max = int(np.max(points[:, 0]))
        y_max = int(np.max(points[:, 1]))
        return [x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)]

    raise TypeError(f"Unsupported VOT region type: {type(region)}")


def mask_to_bbox(mask):
    non_zero_indices = np.argwhere(mask > 0.0)
    if len(non_zero_indices) == 0:
        return [0, 0, 0, 0]

    y_min, x_min = non_zero_indices.min(axis=0).tolist()
    y_max, x_max = non_zero_indices.max(axis=0).tolist()
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to SAM2.1 checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device override. Use cpu if mps is unstable.",
    )
    return parser.parse_args()


def to_local_path(path_or_uri):
    if isinstance(path_or_uri, str) and path_or_uri.startswith("file://"):
        return unquote(urlparse(path_or_uri).path)
    return path_or_uri


def main():
    try:
        args = parse_args()
        if args.device == "auto":
            device = select_device()
        elif args.device == "cuda":
            device = "cuda:0"
        else:
            device = args.device
        handle = vot.VOT("rectangle")
        initial_region = handle.region()
        first_frame = to_local_path(handle.frame())
        if not first_frame:
            return

        first_bbox = region_to_bbox(initial_region)
        x, y, w, h = first_bbox
        init_box_xyxy = (x, y, x + w, y + h)

        sequence_path = Path(first_frame).parent
        model_cfg = determine_model_cfg(args.model_path)
        with contextlib.redirect_stdout(io.StringIO()):
            predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device)

        last_bbox = first_bbox
        expected_frame_idx = 1

        with torch.inference_mode(), get_autocast_context(device):
            with contextlib.redirect_stdout(io.StringIO()):
                state = predictor.init_state(
                    str(sequence_path),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )
                predictor.add_new_points_or_box(state, box=init_box_xyxy, frame_idx=0, obj_id=0)

            generator = predictor.propagate_in_video(state)

            while True:
                next_frame = handle.frame()
                if not next_frame:
                    break

                # Skip potential frame 0 prediction to keep sync with VOT updates.
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        frame_idx, object_ids, masks = next(generator)
                    while frame_idx < expected_frame_idx:
                        with contextlib.redirect_stdout(io.StringIO()):
                            frame_idx, object_ids, masks = next(generator)
                except StopIteration:
                    handle.report(vot.Rectangle(*last_bbox))
                    expected_frame_idx += 1
                    continue

                if len(masks) > 0:
                    mask = masks[0][0].cpu().numpy()
                    last_bbox = mask_to_bbox(mask)

                handle.report(vot.Rectangle(*last_bbox))
                expected_frame_idx += 1

        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

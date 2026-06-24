#!/usr/bin/env python
"""MEVIS MRI feature extraction — CHIMERA Agent, Task 1 (Linear WAT-32/33).

Cleaned from MEVIS's Colab export (Extract_features.ipynb). Keeps only the 3D
MRI feature-extraction path: load prostate-MRI cases (t2w + adc + hbv .mha), run
the UNICORN M3 encoder, and save candidate aggregated latents per case.

Batch by default. Point --input-dir at a folder of per-case subfolders, each
holding one *t2w*/*adc*/*hbv*.mha — cases are discovered by recursively globbing
*t2w*.mha (one case per t2w file; its adc/hbv are expected as siblings). A single
case folder (the 3 files directly inside) is just the n=1 case of the same rule.

Candidates emitted per case (the decision this script exists to inform — WAT-32):
  - mean              [D]
  - mean+max          [2D]
  - mean+max+attended [3D]   (pending: MEVIS to confirm how "most attended" is
                              derived from the grouper output — see TODO below)

Deps (pip): medicalmultitaskmodeling m3-sdk SimpleITK itk monai numpy torch

Usage:
  python inference.py --input-dir /path/to/cases --out-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --- Optional telemetry, kept from the MEVIS notebook (no-op without tokens) ---
import logfire

logfire.configure(
    send_to_logfire="if-token-present",
    sampling=logfire.SamplingOptions.level_or_duration(),
)
if "WANDB_API_KEY" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"

# --- MMM / UNICORN ---
from mmm.interactive import pipes
from mmm.api.M3Model import M3Model, M3_MODELS, UNICORN_ENCODER
from mmm.mmm_types.GroupUsage import GroupUsage
from mmm.volume3d import Tomo3DProcessor
from monai.transforms.spatial.array import ResampleToMatch


def find_modality(case_dir: Path, modality: str) -> Path:
    """Return the single *<modality>*.mha in case_dir (errors if not exactly one).

    Note: a case may also ship a *_mask.mha (prostate zone segmentation); the
    feature-extraction path runs with_segmask=False and ignores it.
    """
    matches = sorted(case_dir.glob(f"*{modality}*.mha"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *{modality}*.mha in {case_dir}, "
            f"found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]


def discover_cases(input_dir: Path) -> list[tuple[str, Path]]:
    """Find every case under input_dir, keyed by its t2w file.

    Each *t2w*.mha (found recursively) defines one case; its folder is the case
    dir and its adc/hbv are expected alongside. case_id is the t2w filename with
    the modality token stripped (PI-CAI: '<patient>_<study>'), falling back to
    the folder name. Returns [(case_id, case_dir), ...] sorted by case_id.
    """
    t2w_files = sorted(input_dir.rglob("*t2w*.mha"))
    if not t2w_files:
        raise FileNotFoundError(f"No *t2w*.mha found anywhere under {input_dir}")

    cases = []
    for t2w in t2w_files:
        cut = t2w.name.lower().rfind("t2w")
        case_id = t2w.name[:cut].rstrip("_-. ") or t2w.parent.name
        cases.append((case_id, t2w.parent))
    return sorted(cases, key=lambda c: c[0])


def load_case(case_id: str, case_dir: Path, processor: Tomo3DProcessor):
    """Load t2w/adc/hbv, resampling adc+hbv onto the t2w grid, into one case."""
    t2w_path = find_modality(case_dir, "t2w")
    adc_path = find_modality(case_dir, "adc")
    hbv_path = find_modality(case_dir, "hbv")

    t2w_image, _ = processor.image_loader(t2w_path)
    t2w_image = t2w_image.unsqueeze(0)  # C, H, W, D

    adc_image, _ = processor.image_loader(adc_path)
    adc_image = ResampleToMatch()(img=adc_image.unsqueeze(0), img_dst=t2w_image)

    hbv_image, _ = processor.image_loader(hbv_path)
    hbv_image = ResampleToMatch()(img=hbv_image.unsqueeze(0), img_dst=t2w_image)

    processed = processor(
        {"image": [t2w_image, adc_image, hbv_image], "meta": {"group_id": "maingroup"}}
    )
    return processed


def build_input_batch(processed_case, processor: Tomo3DProcessor):
    """Slice the volume, unify slice sizes, and collate into a model batch."""

    def slice_processor(x):
        x["image"] = processor.repeat_channels(x["image"])
        return x

    slices = [slice_processor(sc) for sc in processor.extract_slices(processed_case)]
    slices = pipes.UnifySizes(max_edge_len=512)(slices)
    return pipes.mtl_collate(slices)


def extract_features(model, input_batch) -> dict:
    """Run encoder -> squeezer -> grouper and aggregate across slices."""
    # Positional encoding for tomographic data (None for pathology).
    positions = [m["context"][0] for m in input_batch["meta"]]
    supercase_indices = torch.tensor([0 for _ in input_batch["meta"]])

    with torch.inference_mode():
        feature_pyramid = model["encoder"](input_batch["image"].to(model.device))
        hidden_vector = nn.Flatten(1)(model["squeezer"](feature_pyramid)[1])

        # Image transformer for 3D context across slices -> [n_slices, D].
        hidden_vector, grouper_extra = model["grouper"](
            hidden_vector, supercase_indices, GroupUsage(), positions=positions
        )

        mean_rep = torch.mean(hidden_vector, dim=0)        # [D]
        max_rep = torch.max(hidden_vector, dim=0).values   # [D]

        # TODO(most-attended): MEVIS to confirm how UNICORN's "most attended"
        # representation is computed. It is likely recoverable from the grouper's
        # second return value (`grouper_extra`, presumably attention weights) by
        # selecting the slice with the highest attention. Until confirmed we emit
        # only the mean and mean+max candidates.
        most_attended_rep = None

    features = {
        "mean": mean_rep.cpu(),                                  # [D]
        "mean_max": torch.cat([mean_rep, max_rep]).cpu(),        # [2D]
    }
    if most_attended_rep is not None:
        features["mean_max_attended"] = torch.cat(
            [mean_rep, max_rep, most_attended_rep]
        ).cpu()                                                  # [3D]
    return features


def write_platform_json(features: dict, out_path: Path) -> None:
    """Write the challenge-platform-compatible JSON.

    TODO: drop in the existing JSON-writing code reused from the pathology
    dockers so the output matches the platform's expected schema.
    """
    raise NotImplementedError("Plug in existing platform JSON writer here.")


def main():
    parser = argparse.ArgumentParser(
        description="MEVIS MRI feature extraction (CHIMERA Task 1)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder of per-case subfolders (each with t2w/adc/hbv .mha). "
        "A single case folder also works.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./mri_features"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cases whose output already exists (default: skip them, "
        "so the run is resumable).",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write platform-compatible JSON (needs write_platform_json).",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = M3Model(M3_MODELS[UNICORN_ENCODER], device_identifier=args.device)
    processor = Tomo3DProcessor(
        Tomo3DProcessor.Config(), augs_constructor=None, with_segmask=False
    )

    cases = discover_cases(args.input_dir)
    print(f"discovered {len(cases)} case(s) under {args.input_dir}")

    n_ok = n_skip = n_fail = 0
    for i, (case_id, case_dir) in enumerate(cases, 1):
        tag = f"[{i}/{len(cases)}] {case_id}"
        pt_path = args.out_dir / f"{case_id}_features.pt"
        if pt_path.exists() and not args.overwrite:
            print(f"{tag}: skip (exists)")
            n_skip += 1
            continue
        try:
            processed = load_case(case_id, case_dir, processor)
            input_batch = build_input_batch(processed, processor)
            features = extract_features(model, input_batch)
            torch.save(features, pt_path)
            shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in features.items())
            print(f"{tag}: ok -> {pt_path.name}  ({shapes})")
            if args.write_json:
                write_platform_json(features, args.out_dir / f"{case_id}_features.json")
            n_ok += 1
        except Exception as e:  # one bad case shouldn't abort a 1000-case run
            print(f"{tag}: FAIL -> {e}", file=sys.stderr)
            n_fail += 1

    print(f"done: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

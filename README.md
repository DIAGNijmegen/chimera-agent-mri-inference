# chimera-agent-mri-inference

Containerized **MRI feature extraction** from multi-parametric prostate MRI, for
the [CHIMERA Agent](https://chimera-agent.grand-challenge.org/) grand challenge
(**Task 1**).

For each case it loads the three MRI modalities (`t2w` + `adc` + `hbv`, MetaImage
`.mha`), runs the MEVIS UNICORN / M3 encoder slice by slice, and aggregates the
per-slice features into a single per-case latent. A whole folder of cases is
processed in one invocation.

The image is **portable and offline**: the model weights are baked in at build
time, so it runs with no network access or credentials at runtime.

## Pipeline

```
case dir (t2w.mha, adc.mha, hbv.mha)
  └─ load + resample adc/hbv onto the t2w grid
     └─ Tomo3DProcessor: slice the volume → UNICORN encoder → squeezer → grouper
        └─ aggregate across slices → single mean+max latent (.pt + .json per case)
```

The model needs only `t2w`, `adc`, `hbv`; any `*_mask.mha` shipped alongside is
ignored.

## Model

- **Encoder:** the MEVIS **UNICORN / M3** multi-task model
  (`medicalmultitaskmodeling`), applied per slice — encoder → squeezer → grouper,
  with the grouper supplying 3D context across slices of the resampled
  t2w/adc/hbv volume.
- **Aggregation:** the per-slice latents are reduced to one per-case vector by
  concatenating their **mean** and **max** across slices — a single `[2D]` latent
  (1024-dim with the current encoder).

The encoder is MEVIS's
[MedicalMultitaskModeling](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling)
(`mmm`) UNICORN model.

## Prerequisites

- NVIDIA GPU + driver, Docker, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`--gpus all`).
- **Network access at build time only** — the UNICORN weights are fetched during
  the build and baked in; the image then runs fully offline.
- Inputs are MetaImage volumes (`.mha`): one `t2w`, one `adc`, one `hbv` per case.

## Build

```bash
docker build -t chimera-mri-inference .
```

The build downloads the UNICORN weights from a public URL (no token or
credential) and bakes them into the image. It also accepts the MEVIS model
license via `MMM_LICENSE_ACCEPTED` (set in the Dockerfile) — review the license
before building. The download is verified to be a valid zip; re-run the build if
that check trips (the host can occasionally serve an HTML interstitial instead of
the bytes).

## Run

Batch is the default — one container processes every case under `/input`:

```bash
docker run --gpus all \
  -v /path/to/cases:/input \
  -v /path/to/output:/output \
  chimera-mri-inference
```

`/input` is a folder of per-case subfolders, each with one `*t2w*` / `*adc*` /
`*hbv*.mha`; a single case folder with the three files directly inside is just the
n=1 case. The run is **resumable** — cases whose `.pt` already exists are skipped,
and a failure on one case is logged and skipped rather than aborting the run.

- `--overwrite` *(optional)* — recompute cases whose output already exists.
- `--no-write-json` *(optional)* — emit only the `.pt`, skip the JSON companion.

To run the script directly, outside Docker:

```bash
python inference.py --input-dir /path/to/cases --out-dir /path/to/output
```

## I/O contract

**Input** — `--input-dir` is searched recursively for `*t2w*.mha`; each one
defines a case, with its `*adc*.mha` + `*hbv*.mha` in the **same folder** (exactly
one of each). `case_id` is the t2w filename with the modality token stripped
(e.g. `<patient>_<study>`), falling back to the folder name. Any `*mask*.mha` is
ignored.

**Output** — one pair of files per case, written under `/output`:

```
<output>/
├── <case_id>.pt     # deliverable: single mean+max float32 latent (1024-dim)
└── <case_id>.json   # JSON companion to the .pt
```

`<case_id>.pt` is the per-case latent as a bare `torch.Tensor`: the per-slice
features reduced to their **mean** and **max** across slices, concatenated in that
order into one 1-D tensor (`[2D]`, 1024-dim with the current encoder).
`<case_id>.json` carries the same values in the grand-challenge feature-vector
format `[{"title": "", "features": [...]}]`, for consumers that read JSON rather
than torch tensors.

## Layout

```
.
├── Dockerfile
├── inference.py        # batch MRI feature extraction: load → encode → aggregate → save
├── requirements.txt    # mmm + I/O deps (base image provides the CUDA torch stack)
└── constraints.txt     # freeze the base image's torch/torchvision/torchaudio
```

## Provenance

Standalone, offline repackaging of MEVIS's MRI feature-extraction path. The
inference script is cleaned from MEVIS's Colab export (`Extract_features.ipynb`),
trimmed to the 3D multi-parametric MRI path; the UNICORN / M3 encoder is from
[FraunhoferMEVIS/MedicalMultitaskModeling](https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling).
</content>
</invoke>

# chimera-agent-mri-inference

Containerized MRI feature extraction for the [CHIMERA Agent](https://chimera-agent.grand-challenge.org/)
grand challenge (**Task 1**, MRI-only). Wraps the UNICORN / M3 multi-task model:
for each prostate-MRI case (`t2w` + `adc` + `hbv`, MetaImage `.mha`) it runs the
encoder and saves an aggregated latent representation. Processes a whole folder
of cases in one invocation.

The image is built to be **portable and offline**: the model weights are baked
in at build time, so it can be handed to collaborators to run on their own data
without network access or credentials at runtime.

## What it does

```
case dir (t2w.mha, adc.mha, hbv.mha)
  └─ load + resample adc/hbv onto the t2w grid
     └─ Tomo3DProcessor → slices → UNICORN encoder → squeezer → grouper
        └─ aggregate across slices → latent → save (.pt + .json)
```

The mask (`*_mask.mha`) shipped with some datasets is **ignored**; the model
needs only `t2w`, `adc`, `hbv`.

## Latent format

Each case is saved as a single latent vector: the **mean + max** aggregation
across slices, concatenated into one `[2D]` tensor (1024-dim with the current
encoder).

> A third **most-attended** component — derived from the grouper's attention
> weights — is under consideration but not yet wired in, pending confirmation
> from the upstream model authors on how it is computed. See [Open items](#open-items).

## Prerequisites

- NVIDIA GPU + driver, Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`--gpus all`).
- Build host needs network access (to download the model weights at build time).

## Build

```bash
docker build -t chimera-mri-inference .
```

This downloads and bakes the UNICORN weights into the image (`ML_DATA_CACHE`),
so the resulting image is self-contained. The weights come from a **public URL**
— no token or credential is needed. Building also accepts the upstream model
license via `MMM_LICENSE_ACCEPTED` (set in the Dockerfile) — review the license
before building. The build verifies the download is a valid zip (the host can
occasionally serve an HTML interstitial for large files instead of the bytes;
re-run the build if that check trips).

## Run

Batch is the default — one container processes every case under `/input`:

```bash
docker run --gpus all \
  -v /path/to/cases:/input \
  -v /path/to/output:/output \
  chimera-mri-inference
```

`/input` is a folder of per-case subfolders (each with one `*t2w*`/`*adc*`/`*hbv*.mha`);
a single case folder with the 3 files directly inside is just the n=1 case.
Writes `/output/<case_id>.pt` and `/output/<case_id>.json` per case. The run is
**resumable** — cases whose `.pt` already exists are skipped (append `--overwrite`
to recompute), and a failure on one case is logged and skipped rather than
aborting the run.

To run the script directly (outside Docker):

```bash
python inference.py --input-dir /path/to/cases --out-dir /path/to/output
```

Pass `--no-write-json` to emit only the `.pt` and skip the JSON.

## I/O contract

- **Input:** `--input-dir` is searched recursively for `*t2w*.mha`; each one
  defines a case, and its `*adc*.mha` + `*hbv*.mha` must sit in the **same
  folder** (exactly one of each). `case_id` is the t2w filename with the modality
  token stripped (e.g. `<patient>_<study>`), falling back to the folder name.
  Any `*mask*.mha` is ignored.
- **Output:**
  - `<case_id>.pt` — the mean+max latent as a bare `torch.Tensor`.
  - `<case_id>.json` — the same vector in the challenge-platform schema
    (`[{"title": "", "features": [...]}]`), written by default.

## Open items

- [ ] **Most-attended representation** — decide whether to add it, and if so wire
  it into `extract_features`; pending upstream confirmation on how it's derived.
- [ ] **Pin remaining deps** (`pip freeze → requirements.lock.txt`) for a fully
  reproducible image.
- [x] ~~Validate by building + running~~ — validated end-to-end on debug cases
  (build + GPU run, `.pt` + `.json` produced).
- [x] ~~Built-in batch mode~~ — native: recursive case discovery, resumable,
  per-case failure isolation.
- [x] ~~Lock the latent format~~ — single mean+max `[2D]` tensor.
- [x] ~~Plug in the platform JSON writer~~ — wired in, on by default.
- [x] ~~Confirm weights are not credential-gated~~ — **public URL, no auth**;
  baked into the image at build time.
- [x] ~~Confirm a base image~~ — the upstream repo's only Dockerfile is a CPU
  dev/SSH base, **not** a runtime template; we use a `pytorch:*-cuda` base and
  freeze its torch stack via `constraints.txt`.

## Layout

```
.
├── Dockerfile
├── requirements.txt
├── constraints.txt   # freeze the base image's torch/torchvision/torchaudio
├── README.md
└── inference.py      # batch MRI feature extraction
```

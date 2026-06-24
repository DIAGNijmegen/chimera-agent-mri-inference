# chimera-agent-mri-inference

Containerized MRI feature extraction for the [CHIMERA Agent](https://chimera-agent.grand-challenge.org/)
grand challenge (**Task 1**, MRI-only). Wraps the MEVIS-provided UNICORN / M3
model: for each prostate-MRI case (`t2w` + `adc` + `hbv`, MetaImage `.mha`) it
runs the encoder and saves an aggregated latent representation. Runs over a whole
folder of cases in one invocation.

The image is built to be **portable and offline**: the model weights are baked
in at build time, so it can be handed to partners (Karolinska) to run on their
own data without network access or credentials at runtime.

> **Status: DRAFT / not yet validated.** The script and Dockerfile have not been
> run end-to-end yet (the authoring environment had no GPU + `mmm` install).
> Tracked in Linear **WAT-33**, blocked on the latent-format decision in
> **WAT-32**. See [Open items](#open-items) before relying on this.

## What it does

```
case dir (t2w.mha, adc.mha, hbv.mha)
  └─ load + resample adc/hbv onto the t2w grid
     └─ Tomo3DProcessor → slices → UNICORN encoder → squeezer → grouper
        └─ aggregate across slices → latent → save (.pt [+ .json])
```

The mask (`*_mask.mha`) shipped with some datasets (e.g. PI-CAI) is **ignored**;
the model needs only `t2w`, `adc`, `hbv`.

## Latent format

The script currently emits **candidate** aggregations side-by-side so we can pick
one for the full run (this is the WAT-32 decision):

| key                 | shape | status |
|---------------------|-------|--------|
| `mean`              | `[D]`  | ready |
| `mean_max`          | `[2D]` | ready |
| `mean_max_attended` | `[3D]` | **pending** — awaiting MEVIS on how UNICORN's "most attended" rep is derived from the grouper output |

Once the format is locked, this collapses to a single saved tensor.

## Prerequisites

- NVIDIA GPU + driver, Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`--gpus all`).
- Build host needs network access (to download the M3 weights at build time).

## Build

```bash
docker build -t chimera-mri-inference .
```

This downloads and bakes the UNICORN weights into the image (`ML_DATA_CACHE`),
so the resulting image is self-contained. The weights come from a **public
Google Drive URL** (confirmed in `mmm/api/M3Model.py`) — no token or credential
is needed. The build verifies the download is a valid zip (Google Drive can
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
Writes `/output/<case_id>_features.pt` per case. The run is **resumable** —
cases whose output already exists are skipped (append `--overwrite` to recompute),
and a failure on one case is logged and skipped rather than aborting the run.

To run the script directly (outside Docker):

```bash
python inference.py --input-dir /path/to/cases --out-dir /path/to/output
```

## I/O contract

- **Input:** `--input-dir` is searched recursively for `*t2w*.mha`; each one
  defines a case, and its `*adc*.mha` + `*hbv*.mha` must sit in the **same
  folder** (exactly one of each). `case_id` is the t2w filename with the modality
  token stripped (PI-CAI: `<patient>_<study>`), falling back to the folder name.
  Any `*mask*.mha` is ignored.
- **Output:** `<case_id>_features.pt` per case — a dict of the candidate tensors
  above. `--write-json` additionally emits platform-compatible JSON (writer not
  yet wired in; see Open items).

## Open items

- [ ] **Validate by building + running** on a GPU box with the `mmm` env (WAT-32/33).
- [ ] **Lock the latent format** (`mean` vs `mean_max` vs `mean_max_attended`) — WAT-32.
- [ ] **Most-attended representation** — awaiting MEVIS; wire into `extract_features`.
- [ ] **Pin remaining deps** (`pip freeze → requirements.lock.txt`) after first good build.
- [ ] **Plug in the platform JSON writer** (`write_platform_json`, reused from the pathology dockers).
- [x] ~~Built-in batch mode~~ — native: recursive case discovery, resumable, per-case failure isolation.
- [x] ~~Confirm weight download is not credential-gated~~ — **public Google Drive, no auth** (`mmm/api/M3Model.py`).
- [x] ~~Confirm a base image / official Dockerfile~~ — repo's only Dockerfile is a CPU dev/SSH base, **not** a runtime template; we use a `pytorch:*-cuda` base.
- [x] ~~Pin top-level deps~~ — `mmm`/`m3-sdk` pinned to `1.6.3`; `m3-sdk` is mandatory but undeclared by mmm.

## Layout

```
.
├── Dockerfile
├── requirements.txt
├── README.md
└── inference.py   # cleaned from MEVIS's Colab export; batch feature extraction
```

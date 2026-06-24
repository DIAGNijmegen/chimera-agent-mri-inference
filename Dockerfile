# MEVIS MRI feature extraction — CHIMERA Agent, Task 1.
#
# Portable, self-contained image: the UNICORN/M3 weights are baked in at BUILD
# time so the container runs fully offline..
#
# Base-image note: the official mmm repo (FraunhoferMEVIS/MedicalMultitaskModeling)
# ships only docker/mmm-base/Dockerfile, which is a CPU Ubuntu-22.04 dev/SSH box
# with NO CUDA and NO Python deps (CHANGELOG: "Stopped building Docker image with
# dependencies included"). It is NOT a runtime template — so we choose our own
# CUDA base below. mmm requires torch>=2.1.2,<3 (pyproject) and Python>=3.10;
# this base satisfies both and already ships torch 2.3.1, so pip won't upgrade it.
#
# STATUS: DRAFT — not yet validated by a real GPU build. Remaining before handoff:
#   - run a real build + in-container smoke test on the WAT-32 example case
#   - lock the latent format / wire in the most-attended rep (WAT-32, MEVIS)
#   - pin remaining deps via pip freeze after the first good build
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline \
    # logfire is a hard dep imported at module load — keep it from phoning home.
    LOGFIRE_SEND_TO_LOGFIRE=false \
    # M3Model caches weights at $ML_DATA_CACHE/models/ on first instantiation
    # (else ~/.mmm/models/). Point it inside the image so the bake step persists.
    ML_DATA_CACHE=/opt/mmm-cache

WORKDIR /app

# cv2 (pulled in via mmm/albumentations) needs these; also covers monai image IO.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Bake the UNICORN weights into the image (offline runtime) -----------------
# M3Model downloads the checkpoint on first instantiation. Confirmed against the
# repo (mmm/api/M3Model.py): M3_MODELS[UNICORN_ENCODER] is a PUBLIC Google Drive
# URL fetched via torch.hub.download_url_to_file — no auth/token, so no build
# secret needed. device_identifier='cpu' works (no forced CUDA), so this bakes on
# a GPU-less build host. Caveat: Google Drive can return an HTML interstitial for
# large files instead of the bytes — hence the zip-validity check.
RUN mkdir -p "$ML_DATA_CACHE" && \
    python -c "from mmm.api.M3Model import M3Model, M3_MODELS, UNICORN_ENCODER; M3Model(M3_MODELS[UNICORN_ENCODER], device_identifier='cpu')" && \
    python -c "import glob, os, zipfile; p = glob.glob(os.path.join(os.environ['ML_DATA_CACHE'], 'models', '*.zip'))[0]; assert zipfile.is_zipfile(p), f'Not a valid zip (Google Drive interstitial?): {p}'; print('weights OK:', p, os.path.getsize(p), 'bytes')"

COPY inference.py .

# Batch by default: mount a folder of per-case subfolders at /input and an
# output dir at /output (a single case folder also works). The run is resumable
# — existing outputs are skipped unless you append --overwrite.
#   docker run --gpus all -v /path/to/cases:/input -v /path/to/out:/output IMAGE
ENTRYPOINT ["python", "inference.py"]
CMD ["--input-dir", "/input", "--out-dir", "/output"]

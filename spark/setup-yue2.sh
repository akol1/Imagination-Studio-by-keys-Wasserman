#!/bin/bash
# YuE2 (m-a-p's song model) for the Music tab, in its own environment under $PREFIX/yue2 (Linux, NVIDIA, BF16):
#   bash spark/setup-yue2.sh            # skips what is already there at the pinned versions; safe to run twice
#   bash spark/setup-yue2.sh --verify   # checks the install, downloads nothing
# oneshot-setup-spark.sh --yue2 (and --all) runs this; the Music tab then runs spark/yue2_server.py as a persistent
# server (app/music_yue2.py). About 13 GB: the YuE2-3B checkpoint (7.3 GB, bf16), the YuE2-Vae decoder (0.5 GB) and a
# Python 3.12 environment with PyTorch for CUDA 13 (the studio's own pin, from uv's cache when it is there).
# Licences: YuE2's code is Apache 2.0 (multimodal-art-projection/YuE at YUE2_COMMIT). The weights are CC BY-NC 4.0 with
# the authors' additional permission for individual creators: personal users, content creators and musicians may
# publish, sell and license what they make, free and royalty-free; a company needs a commercial licence from the
# authors; no illegal, harmful or deceptive use. Read it: https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE
# Environment: PREFIX, YUE2_HOME (default $PREFIX/yue2), YUE2_WEIGHTS (default $YUE2_HOME/weights), YUE2_COMMIT,
# YUE2_TORCH (default the studio's torch pin).
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
OUT="${YUE2_HOME:-$PREFIX/yue2}"
WEIGHTS="${YUE2_WEIGHTS:-$OUT/weights}"
# the source (Apache 2.0): YuE2 0.1.6 and the yue2-music skill 1.2.0 (its instrumental helpers, MIT)
YUE2_COMMIT="${YUE2_COMMIT:-1647252d68b70cbead046ffd3f7caf036dde47ee}"
# the checkpoints at pinned revisions, and their weights' sha256 (each repo's weights_manifest.json says the same)
MODEL_REPO=m-a-p/YuE2-3B MODEL_REV=8f24312f187763b9854adeec87e437e2b03bbff1
MODEL_SHA=1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9 MODEL_BYTES=7261441640
VAE_REPO=m-a-p/YuE2-Vae VAE_REV=152733a19ad43aa67e367f9b5503ef8075bb5126
VAE_SHA=807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346 VAE_BYTES=530512720
# the studio's PyTorch (oneshot-setup-cuda.sh TORCH_PIN); YuE2's own other pins, from its pyproject.toml
YUE2_TORCH="${YUE2_TORCH:-torch==2.14.1}"
DEPS=(transformers==4.57.6 huggingface-hub==0.36.2 safetensors==0.7.0 tiktoken==0.12.0 numpy==2.2.6 soundfile==0.13.1 accelerate==1.13.0)
VERIFY=0
for arg in "$@"; do case "$arg" in --verify) VERIFY=1;; *) echo "unknown option $arg" >&2; exit 2;; esac; done
say() { printf '==> %s\n' "$*"; }
ok() { printf '  \342\234\223 %s\n' "$*"; }
die() { printf '  \342\234\227 %s\n' "$*" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

say "YuE2 -> $OUT"
[ "$(uname -s)" = Linux ] || die "YuE2 runs on Linux with an NVIDIA GPU (a DGX Spark, an RTX Pro 6000); a Mac uses MiniMax Music 3"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: YuE2 needs an NVIDIA GPU with BF16"
command -v git >/dev/null || die "install git first"
STAMP="$MODEL_REV $VAE_REV $YUE2_COMMIT $YUE2_TORCH"
if [ "$VERIFY" = 0 ] && [ "$(cat "$OUT/installed" 2>/dev/null)" = "$STAMP" ] && [ -x "$OUT/venv/bin/python" ] \
   && [ "$(stat -c %s "$WEIGHTS/YuE2-3B/model.safetensors" 2>/dev/null)" = "$MODEL_BYTES" ] \
   && [ "$(stat -c %s "$WEIGHTS/YuE2-Vae/model.safetensors" 2>/dev/null)" = "$VAE_BYTES" ]; then
  ok "already installed (YuE2 ${YUE2_COMMIT:0:8}, YuE2-3B ${MODEL_REV:0:8}, YuE2-Vae ${VAE_REV:0:8})"; exit 0
fi

# the source, at the pinned commit
SRC="$OUT/src"
if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$YUE2_COMMIT" ]; then
  [ "$VERIFY" = 0 ] || die "no YuE2 source at ${YUE2_COMMIT:0:8} in $SRC"
  mkdir -p "$OUT"
  [ -d "$SRC/.git" ] || git init -q "$SRC"
  git -C "$SRC" fetch -q --depth 1 https://github.com/multimodal-art-projection/YuE.git "$YUE2_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
ok "source at ${YUE2_COMMIT:0:8} (Apache 2.0)"

# its own environment: never the studio's $PREFIX/venv
PY="$OUT/venv/bin/python"
if [ "$VERIFY" = 0 ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  [ -x "$PY" ] || uv venv "$OUT/venv" --python 3.12 --python-preference only-managed --seed -q
  pipi() { UV_TORCH_BACKEND=cu130 uv pip install -q --python "$PY" "$@"; }
  "$PY" -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] == sys.argv[1] and torch.cuda.is_available() else 1)" \
    "${YUE2_TORCH#torch==}" 2>/dev/null || pipi "$YUE2_TORCH"
  pipi "${DEPS[@]}"
  # the package itself without its torch pin (2.10): the environment keeps the studio's PyTorch
  pipi --no-deps --reinstall "$SRC"
fi
[ -x "$PY" ] || die "no environment at $OUT/venv"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)" \
  || die "PyTorch in $OUT/venv does not see a GPU with BF16"
"$PY" -c "import yue2, transformers, soundfile" || die "the yue2 package is missing from $OUT/venv"
ok "environment: $("$PY" -c 'import torch, yue2; print("yue2", yue2.__version__, "· torch", torch.__version__, "on", torch.cuda.get_device_name(0))')"

# the weights, at pinned revisions, checked against their sha256 (once: a stamp keeps the size and time checked)
fetch() {  # repo revision folder sha bytes
  local repo="$1" rev="$2" dir="$WEIGHTS/$3" sha="$4" bytes="$5" file
  file="$dir/model.safetensors"
  if [ "$(cat "$dir/.checked" 2>/dev/null)" = "$rev $sha $(stat -c '%s %Y' "$file" 2>/dev/null)" ]; then ok "$3 at ${rev:0:8} (checked)"; return 0; fi
  if [ "$(stat -c %s "$file" 2>/dev/null)" != "$bytes" ]; then
    [ "$VERIFY" = 0 ] || die "no $3 weights at $dir"
    mkdir -p "$dir"
    "$PY" -I - "$repo" "$rev" "$dir" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
repo, rev, folder = sys.argv[1:4]
info = HfApi().model_info(repo, revision=rev)
if info.gated:  # not gated at the pinned revision; a later gate is said, not worked around
    sys.exit(f"{repo} is gated on Hugging Face: accept its licence there and sign in (hf auth login) first")
snapshot_download(repo, revision=rev, local_dir=folder, allow_patterns=[
    "config.json", "generation_config.json", "yue2_generation_config.json", "weights_manifest.json", "model.safetensors",
    "qwen.tiktoken", "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "licenses/*.txt"])
PY
  fi
  echo "  checking $3's sha256 ..."
  [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$sha" ] || die "$file does not match its sha256 ($sha): delete it and run again"
  echo "$rev $sha $(stat -c '%s %Y' "$file")" > "$dir/.checked"
  ok "$3 at ${rev:0:8} (sha256 ${sha:0:12})"
}
echo "  YuE2's weights are CC BY-NC 4.0 with the authors' individual creator permission: personal users, creators and"
echo "  musicians may sell and license their songs, royalty-free; a company needs the authors' commercial licence."
echo "  Read it: https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE"
fetch "$MODEL_REPO" "$MODEL_REV" YuE2-3B "$MODEL_SHA" "$MODEL_BYTES"
fetch "$VAE_REPO" "$VAE_REV" YuE2-Vae "$VAE_SHA" "$VAE_BYTES"
[ -s "$WEIGHTS/YuE2-3B/qwen.tiktoken" ] || die "no tokenizer (qwen.tiktoken) in $WEIGHTS/YuE2-3B"
cp "$SRC/MODEL_LICENSE" "$WEIGHTS/MODEL_LICENSE" 2>/dev/null || true
[ "$VERIFY" = 1 ] || echo "$STAMP" > "$OUT/installed"
ok "YuE2 for the Music tab (default engine on this machine; MiniMax Music 3 stays a choice)"

#!/bin/bash
# Build the YuE2 decode engine for CUDA (libtf_yue2.so) into $PREFIX/tf-yue2: TensorFold's CUDA runtime and music3's
# int8 / bf16 GEMVs (drowzeys/TensorFold at TF_YUE2_COMMIT, branch music3-cuda-115, TensorFold 1.1.5) with the yue2
# family in spark/tensorfold_yue2/tensorfold laid over it. The YuE2 server's PyTorch pipeline then runs each generated
# token's pass through the model on it (spark/tensorfold_yue2/__init__.py); everything else stays PyTorch.
#   bash spark/build-yue2-engine.sh            # skips the build when this commit and these sources are already built
# A build needs the CUDA toolkit's nvcc (12.9 or newer; a Spark has 13 at /usr/local/cuda) and fetches Zig 0.17.0 and
# TensorFold's source under $PREFIX. About a minute.
# Measured on a GB10 only: on another GPU it refuses unless YUE2_ANY_GPU=1 (the server then decodes in PyTorch).
# Environment: PREFIX, NVCC, TF_YUE2_COMMIT, TF_YUE2_SOURCE (an existing TensorFold checkout to lay the family over).
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
TF_YUE2_COMMIT="${TF_YUE2_COMMIT:-5505dbfaf38efce55e564062886ad74bae6f21e5}"
ZIG_VERSION=0.17.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="$HERE/tensorfold_yue2/tensorfold"
OUT="$PREFIX/tf-yue2"
# the family's own sources are part of the stamp: an edit rebuilds
STAMP="$TF_YUE2_COMMIT $(cd "$FAMILY" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
if [ -s "$OUT/libtf_yue2.so" ] && [ "$(cat "$OUT/stamp" 2>/dev/null)" = "$STAMP" ] && [ -z "${TF_YUE2_SOURCE:-}" ]; then
  echo "[tensorfold] YuE2 engine already built at ${TF_YUE2_COMMIT:0:8}"; exit 0
fi
# shellcheck disable=SC1091
. "$HERE/gpu.sh"
studio_pin_gpu
GPU="$(studio_smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
smi_i=()
[ -z "${STUDIO_GPU:-}" ] || smi_i=(-i "$STUDIO_GPU")
# the GPU's compute capability as nvcc names it: 121 on a GB10 (DGX Spark), 120 on an RTX Pro 6000 Blackwell (a
# cubin for one does not run on the other); TF_SM overrides, a comma list builds both
SM="${TF_SM:-$(nvidia-smi ${smi_i[@]+"${smi_i[@]}"} --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')}"
case "$SM" in ''|*[!0-9,]*) SM=121;; esac  # a driver without the field answers with words
case "$GPU" in *GB10*) ;; *)
  [ "${YUE2_ANY_GPU:-0}" = 1 ] || { echo "the YuE2 engine is measured on a GB10 (DGX Spark) only; this GPU is '${GPU:-none}'. YUE2_ANY_GPU=1 builds it anyway" >&2; exit 1; };;
esac
NVCC="${NVCC:-}"
[ -n "$NVCC" ] || for candidate in /usr/local/cuda/bin/nvcc "$(command -v nvcc || true)"; do
  [ -x "$candidate" ] && { NVCC="$candidate"; break; }
done
[ -n "$NVCC" ] || { echo "no nvcc: install the CUDA toolkit (12.9 or newer) or set NVCC" >&2; exit 1; }
version="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
read -r major minor <<<"$version"
[ "$major" -gt 12 ] || { [ "$major" = 12 ] && [ "$minor" -ge 9 ]; } || { echo "$NVCC is CUDA $major.$minor; the GB10 needs 12.9 or newer" >&2; exit 1; }
case "$(uname -m)" in aarch64) arch=aarch64;; x86_64) arch=x86_64;; *) echo "no Zig build for $(uname -m)" >&2; exit 1;; esac
ZIG="$PREFIX/zig-$ZIG_VERSION/zig"
if [ ! -x "$ZIG" ]; then
  echo "[tensorfold] fetching Zig $ZIG_VERSION"
  mkdir -p "$PREFIX/zig-$ZIG_VERSION"
  curl -sfL "https://ziglang.org/download/$ZIG_VERSION/zig-$arch-linux-$ZIG_VERSION.tar.xz" | tar -xJ --strip-components=1 -C "$PREFIX/zig-$ZIG_VERSION"
fi
SRC="${TF_YUE2_SOURCE:-$PREFIX/tf-yue2-src}"
if [ -z "${TF_YUE2_SOURCE:-}" ]; then
  [ -d "$SRC/.git" ] || git init -q "$SRC"
  if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$TF_YUE2_COMMIT" ]; then
    git -C "$SRC" fetch -q --depth 1 https://github.com/drowzeys/TensorFold.git "$TF_YUE2_COMMIT"
    git -C "$SRC" checkout -q -f FETCH_HEAD
  fi
fi
# the family over the checkout: its kernels, its Zig and its build step, and one line in zig/build/cuda.zig
( cd "$FAMILY" && find . -type f -print0 ) | while IFS= read -r -d '' f; do
  mkdir -p "$SRC/$(dirname "$f")"; cp "$FAMILY/$f" "$SRC/$f"
done
hook='    @import("yue2_lib.zig").library(b, target, optimize, nvcc, sms);'
anchor='    @import("h3_host.zig").library(b, target, optimize, h3_image);'
if ! grep -qF "$hook" "$SRC/zig/build/cuda.zig"; then
  grep -qF "$anchor" "$SRC/zig/build/cuda.zig" || { echo "TensorFold at ${TF_YUE2_COMMIT:0:8} has no H3 library line to hook the yue2 step after" >&2; exit 1; }
  awk -v a="$anchor" -v h="$hook" '{ print } $0 == a { print h }' "$SRC/zig/build/cuda.zig" > "$SRC/zig/build/cuda.zig.new"
  mv "$SRC/zig/build/cuda.zig.new" "$SRC/zig/build/cuda.zig"
fi
echo "[tensorfold] building the YuE2 engine with $NVCC (sm $SM)"
(cd "$SRC" && nice -n 19 "$ZIG" build tf-yue2 -j2 -Doptimize=fast -Dnvcc="$NVCC" -Dsm="$SM")
(cd "$SRC" && nice -n 19 "$ZIG" build test-yue2 -Dnvcc="$NVCC" -Dsm="$SM")
mkdir -p "$OUT"
cp "$SRC/zig-out/lib/libtf_yue2.so" "$OUT/libtf_yue2.so.new" && mv "$OUT/libtf_yue2.so.new" "$OUT/libtf_yue2.so"
echo "$STAMP" > "$OUT/stamp"
echo "[tensorfold] YuE2 engine -> $OUT/libtf_yue2.so"

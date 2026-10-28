#!/bin/bash
# The pieces behind the Library's "Swap the character" on a DGX Spark (app/swap.py, spark/spark_swap.py), about 2.1 GB
# beside MiniMax H3 Ref2VA's int8 weights (oneshot-setup-spark.sh --swap fetches those). No node pack: it runs on
# ComfyUI's own nodes.
#   bash spark/setup-swap.sh            # fetch what is missing (oneshot-setup-spark.sh --swap runs this)
#   bash spark/setup-swap.sh --verify   # check, download nothing
#   Character Swap LoRA v1.5   Akatz Labs, 155 MB    MiniMax H3 Community License (a Model Derivative of MiniMax H3)
#   MiniMax H3 Turbo, 8 steps  LightX2V, 1.96 GB     Apache-2.0
#   SAM 3.1                    Meta / Comfy-Org, 1.7 GB, only when the Library's actions have not fetched it: SAM License
# MiniMax H3 and the LoRA are not licensed for use in the European Union, the United Kingdom, South Korea or the United
# States (the MiniMax H3 Community License's Excluded Territories). Each file is checked against its SHA-256 and never
# written over: a file already there is kept.
# Environment: PREFIX, SPARK_MODELS.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
MODELS="${SPARK_MODELS:-$HOME/spark-studio-models}"
VERIFY=0
for arg in "$@"; do case "$arg" in --verify) VERIFY=1;; *) echo "usage: setup-swap.sh [--verify]" >&2; exit 2;; esac; done
bad=0

if [ ! -s "$MODELS/MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors" ] \
   && [ ! -s "$MODELS/MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" ]; then
  echo "  ! MiniMax H3 Ref2VA is not installed: bash oneshot-setup-spark.sh --swap (it fetches the int8 weights)"; bad=1
fi
[ "$VERIFY" = 1 ] || echo "  MiniMax H3 and the Character Swap LoRA: MiniMax H3 Community License, not licensed for use in the EU, the UK, South Korea or the US (https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/main/LICENSE)"

# model <repository> <revision> <file in it> <folder under $MODELS> <sha256>
model() {
  local repo="$1" rev="$2" file="$3" dir="$MODELS/$4" sum="$5" name
  name="$(basename "$file")"
  if [ -s "$dir/$name" ]; then echo "  ✓ $name"; return 0; fi
  if [ "$VERIFY" = 1 ]; then echo "  ! missing: $dir/$name"; bad=1; return 0; fi
  mkdir -p "$dir"
  curl -fL --retry 5 -C - -sS -o "$dir/$name.part" "https://huggingface.co/$repo/resolve/$rev/$file"
  echo "$sum  $dir/$name.part" | sha256sum -c --quiet - || { echo "  ! $name: checksum mismatch" >&2; rm -f "$dir/$name.part"; exit 1; }
  mv -n "$dir/$name.part" "$dir/$name"; echo "  ✓ $name (downloaded)"; [ ! -d "$PREFIX" ] || touch "$PREFIX/workflows.restart"
}
model akatz-ai/MiniMax-H3-Character-Swap-LoRA d06323b7c8e70e97cdab48b78a1e05f786c8cba9 h3_character_swap_v1.5.safetensors \
  MiniMax-H3-Comfy/loras 8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f
model lightx2v/Minimax-h3-Turbo 3ec17a324ced54151364f24f8b5fb6bf7e26414f minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors \
  MiniMax-H3-Comfy/loras 08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980
model Comfy-Org/sam3.1 7bb8374780a725b4353ed31f3a9395c9742b5621 checkpoints/sam3.1_multiplex_fp16.safetensors \
  Workflows-Comfy/checkpoints 9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03

# ComfyUI reads SAM 3.1's folder from extra_model_paths.yaml (spark/setup-workflows.sh adds it too, and the Upscale
# action's folder beside it, each looked for on its own)
YAML="$PREFIX/ComfyUI/extra_model_paths.yaml"
if [ -f "$PREFIX/ComfyUI/main.py" ] && ! grep -q "Workflows-Comfy/checkpoints" "$YAML" 2>/dev/null; then
  if [ "$VERIFY" = 1 ]; then echo "  ! $YAML does not list SAM 3.1's folder"; bad=1; else
    printf '  checkpoints: |\n    Workflows-Comfy/checkpoints\n' >> "$YAML"
    echo "  ✓ SAM 3.1's folder added to extra_model_paths.yaml"; touch "$PREFIX/workflows.restart"
  fi
fi
if [ -e "$PREFIX/workflows.restart" ] && [ -f "$(dirname "$0")/comfy.sh" ] && bash "$(dirname "$0")/comfy.sh" status | grep -q running; then
  echo "  restart ComfyUI to see the new files: bash spark/comfy.sh stop (the next render starts it)"
fi
exit $bad

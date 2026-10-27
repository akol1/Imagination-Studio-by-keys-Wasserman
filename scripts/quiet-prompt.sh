# A raw text or first-frame prompt with no spoken line is given the Studio's own ambient line.
# Sourced by scripts/video.sh and spark/video.sh. The user's sentences stay.
# app/negation.py ensure_ambient holds the rule; this only applies it.
# Left as written: a reference (REF_IMAGES, REF_VIDEOS, REF_AUDIOS, or a <Picture>, <Video> or <Audio>
# tag), REF_PLAIN=1, and REF_VIDEO_SOUND=1. PROMPT and PROMPT_FILE are the script's own.

ambient_prompt() {
  local app="$1"
  local py flags="" before after
  py="$(command -v python3 || true)"
  [ -n "$py" ] || { echo "python3 is needed to prepare a video prompt" >&2; exit 2; }
  [ "${REF_VIDEO_SOUND:-0}" = 1 ] && flags="$flags --video-sound"
  [ "${REF_PLAIN:-0}" = 1 ] && flags="$flags --plain"
  if [ "${REF_COUNT:-0}" != 0 ] || [ -n "${REF_IMAGES:-}${REF_VIDEOS:-}${REF_AUDIOS:-}" ]; then
    flags="$flags --references"
  fi
  if [ -n "${PROMPT_FILE:-}" ]; then
    [ -f "$PROMPT_FILE" ] || { echo "no prompt file at $PROMPT_FILE" >&2; exit 2; }
    before="$(cat "$PROMPT_FILE")"
    # shellcheck disable=SC2086
    after="$("$py" "$app/negation.py" $flags "$PROMPT_FILE")" || { echo "[tensorfold] prompt: could not be prepared, sent as written" >&2; return 0; }
    [ "$after" = "$before" ] && return 0
    PROMPT="$after"
    PROMPT_FILE=""
  else
    before="$PROMPT"
    # the x keeps a trailing newline, so a prompt that ends in one and is left as written stays equal to itself
    # shellcheck disable=SC2086
    after="$(printf '%s' "$PROMPT" | "$py" "$app/negation.py" $flags && printf x)" || after=""
    case "$after" in
      *x) after="${after%x}" ;;
      *) echo "[tensorfold] prompt: could not be prepared, sent as written" >&2; return 0 ;;
    esac
    [ "$after" = "$before" ] && return 0
    PROMPT="$(printf '%s' "$after")"  # a prepared prompt ends where its last field does, as compose() writes it
  fi
  echo "[tensorfold] prompt: no spoken line and no overall_soundscape heading, so the clip is told: Only ambient sound is heard."
}

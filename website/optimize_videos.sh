#!/usr/bin/env bash
#
# optimize_videos.sh — batch web-optimize the clips in videos/ for fast,
# smooth playback (the same recipe as research.nvidia.com/.../kimodo/).
#
# What it does to each .mp4:
#   • downscales to a max height (default 720p), keeping aspect (even dims)
#   • re-encodes H.264 at a quality target (CRF) with a "slow" preset
#   • -movflags +faststart  → moov atom at the front so playback can start
#     before the whole file downloads
#   • drops the audio track (-an) — the player uses muted loops anyway
#   • optionally also writes a VP9 .webm (smaller again; add a <source> for it)
#
# Output is written to a separate folder (default videos/optimized/) so your
# originals are never touched.
#
# Usage:
#   ./optimize_videos.sh                     # videos/  -> videos/optimized/
#   ./optimize_videos.sh --webm              # also emit .webm (VP9)
#   ./optimize_videos.sh --in videos --out videos/optimized --height 720 --crf 24
#   ./optimize_videos.sh --in videos/web_cases/prompt_1064/video
#
set -euo pipefail

IN_DIR="videos"
OUT_DIR=""
HEIGHT=720
CRF=24
WEBM=0
WEBM_CRF=33

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in)       IN_DIR="$2"; shift 2;;
    --out)      OUT_DIR="$2"; shift 2;;
    --height)   HEIGHT="$2"; shift 2;;
    --crf)      CRF="$2"; shift 2;;
    --webm-crf) WEBM_CRF="$2"; shift 2;;
    --webm)     WEBM=1; shift;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found on PATH." >&2; exit 1; }
[[ -d "$IN_DIR" ]] || { echo "Input dir not found: $IN_DIR" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-$IN_DIR/optimized}"
mkdir -p "$OUT_DIR"

# only downscale (never upscale); force even width via -2
VF="scale=w=-2:h='min(${HEIGHT},ih)':flags=lanczos"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1}B"; }

total_in=0; total_out=0; count=0
shopt -s nullglob
for src in "$IN_DIR"/*.mp4; do
  [[ -e "$src" ]] || continue
  base="$(basename "$src" .mp4)"
  dst="$OUT_DIR/$base.mp4"

  # skip if already optimized and up-to-date
  if [[ -f "$dst" && "$dst" -nt "$src" ]]; then
    echo "skip (up-to-date): $base.mp4"
  else
    echo "→ $base.mp4"
    ffmpeg -y -loglevel error -i "$src" \
      -vf "$VF" \
      -c:v libx264 -crf "$CRF" -preset slow -pix_fmt yuv420p \
      -movflags +faststart -an \
      "$dst"
  fi

  if [[ "$WEBM" -eq 1 ]]; then
    wdst="$OUT_DIR/$base.webm"
    if [[ -f "$wdst" && "$wdst" -nt "$src" ]]; then
      echo "  skip webm (up-to-date)"
    else
      echo "  + webm (VP9)"
      ffmpeg -y -loglevel error -i "$src" \
        -vf "$VF" \
        -c:v libvpx-vp9 -crf "$WEBM_CRF" -b:v 0 -row-mt 1 -pix_fmt yuv420p -an \
        "$wdst"
    fi
  fi

  in_sz=$(stat -c%s "$src"); out_sz=$(stat -c%s "$dst")
  total_in=$((total_in+in_sz)); total_out=$((total_out+out_sz)); count=$((count+1))
  printf "   %s → %s\n" "$(human "$in_sz")" "$(human "$out_sz")"
done

echo
echo "Done: $count file(s)"
[[ "$count" -gt 0 ]] && printf "Total mp4: %s → %s\n" "$(human "$total_in")" "$(human "$total_out")"
echo "Output:   $OUT_DIR"
echo
echo "Next: point the carousel/teaser at the optimized files, e.g."
echo "  videos/optimized/<name>.mp4   (and add a .webm <source> first if you used --webm)"

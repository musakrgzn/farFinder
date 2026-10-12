#!/usr/bin/env bash
# Run on the Ubuntu desktop; no Python or host package imports are needed.
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset. Run this script from the Ubuntu desktop terminal." >&2
    exit 1
fi
if [[ ! "$DISPLAY" =~ ^(unix)?:[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Use a local X11/XWayland DISPLAY such as :0; SSH TCP forwarding is unsupported." >&2
    exit 1
fi
if ! command -v xauth >/dev/null; then
    echo "Install the Ubuntu system utility: sudo apt install xauth" >&2
    exit 1
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
umask 077
if ! cookie="$(xauth nlist "$DISPLAY")"; then
    echo "Cannot read the X11 cookie. Check XAUTHORITY in your desktop terminal." >&2
    exit 1
fi
if [[ -z "$cookie" ]]; then
    echo "No X11 cookie for $DISPLAY. Check XAUTHORITY in your desktop terminal." >&2
    exit 1
fi
temp_auth="$(mktemp "$project_dir/.docker.xauth.XXXXXX")"
trap 'rm -f -- "$temp_auth"' EXIT
# FamilyWild permits the same cookie from a different container hostname.
printf '%s\n' "$cookie" | sed 's/^..../ffff/' | xauth -f "$temp_auth" nmerge -
chmod 600 "$temp_auth"
mv -f -- "$temp_auth" "$project_dir/.docker.xauth"
echo "Prepared .docker.xauth. Re-run after logging out or changing DISPLAY."

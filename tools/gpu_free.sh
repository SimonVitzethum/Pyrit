#!/bin/sh
# Bricht ab, wenn die GPU nicht frei ist (Auslastung > 0 % oder fremde Rechenprozesse).
# Verwendung: tools/gpu_free.sh && zig build gpu-test
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xorg|Xwayland|gnome-shell|plasmashell" | wc -l)
if [ "$util" != "0" ] || [ "$apps" != "0" ]; then
    echo "GPU belegt (Auslastung ${util} %, ${apps} Rechenprozesse) – kein GPU-Test." >&2
    exit 1
fi
echo "GPU frei."

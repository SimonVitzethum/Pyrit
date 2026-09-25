#!/bin/sh
# Startet die Pyrit-Demo: Minecraft-artige Welt, auf der GPU erzeugt,
# Zuschauermodus (W/A/S/D, Leertaste, Umschalt, Strg, Maus ziehen).
#
#   ./demo.sh                      DLSS 2x + DLSS Frame Generation 4x (mit SDK)
#   ./demo.sh --fg dlss6           andere Einstellungen werden durchgereicht
#
# Mit dem DLSS-SDK in ./DLSS wird mit DLSS gebaut, sonst mit Pyrits TAAU und
# eigener Frame Generation. Tasten 1–5 wechseln den Bildaufbau, F die
# Zwischenbilder (aus, eigene, DLSS 2x/3x/4x/6x).
set -e
cd "$(dirname "$0")"
if [ -d DLSS/include ]; then
    exec zig build demo --release=fast -Ddlss-sdk="$PWD/DLSS" -- --mode dlss --fg dlss4 --size 1920x1080 "$@"
else
    exec zig build demo --release=fast -- --mode taau --fg --size 1920x1080 "$@"
fi

# Pyrit

Ein Renderer für **Sparse Voxel DAGs** – vollständig in Zig geschrieben, von der
Host-Bibliothek über die GPU-Kernel bis zu den Tests. Keine Grafik-API: Pyrit
rechnet in reinem CUDA (NVIDIA, RT-Cores über OptiX) und gibt CUDA-Puffer aus.
Der C-Header `include/pyrit.h` beschreibt nur die ABI für andere Sprachen.

![Gestreamte Voxelwelt, 1920x1080](docs/bilder/pyrit.png)

## Was drin ist

- **Sparse Voxel DAGs**, gebaut auf der GPU; keine Meshes, keine Vertices.
- **Große Welten**, gestreamt mit **dynamischem LOD**: keine festen Stufen,
  verfeinert wird nach der Größe eines Voxels auf dem Bildschirm. Ein
  Speicherbudget regelt die Zielgröße gleitend nach.
- **Raytracing** über RT-Cores (OptiX, Anbindung selbst in Zig geschrieben) mit
  Software-Rückfallweg.
- **Globale Beleuchtung**, Schatten, Reflexionen, Transparenz mit Brechung und
  getönten Schatten, Wellen auf Wasser.
- **Motion Vectors pro Pixel** als Kernziel – Grundlage für TAA, TAAU, Frame
  Generation und DLSS.
- **Vollständige Ausgabekette**: temporale Akkumulation, varianzgeführter
  À-trous-Denoiser, TAAU (bis 4x), optional NVIDIA DLSS Super Resolution und
  Ray Reconstruction über NGX-CUDA, sowie Frame Generation in reinem CUDA.
- **GPU-Skelettanimation** mit Keyframe-Interpolation und Windschwingen.

## Bauen

Gebraucht wird Zig 0.16. Eine GPU ist zum Bauen nicht nötig.

```sh
zig build                 # Bibliothek (statisch + dynamisch), Header, PTX
zig build test            # Unit-, CPU- und ABI-Tests, AMD-Übersetzung (ohne GPU)
zig build kernel-check    # PTX mit ptxas prüfen (CUDA-Toolkit, keine GPU)
zig build gpu-test        # GPU gegen CPU-Referenz (braucht eine freie NVIDIA-GPU)
zig build --release=fast  # optimiert
```

Die GPU-Kernel entstehen aus demselben Zig-Code für `nvptx64-cuda` (nach PTX)
und `amdgcn-amdhsa` (bisher nur als Übersetzungsprüfung; ein HIP-Backend folgt).

DLSS ist optional und braucht das nicht mitgelieferte SDK:
`zig build -Ddlss-sdk=/pfad/zum/DLSS`.

## Werkzeuge

```sh
zig build render -- --world --size 1920x1080 --frames 30 --out bild.ppm
zig build view   -- --size 1280x720
```

`pyrit-view` ist ein Betrachter **ohne jede Grafik-API**: Pyrit rendert in einen
CUDA-Puffer, der Betrachter kopiert ihn in einen Shared-Memory-Puffer und zeigt
ihn über **Wayland** (xdg-shell) an. `libwayland-client` wird dynamisch geladen,
die xdg-shell-Tabellen stehen in `tools/wayland.zig`. Steuerung: W/A/S/D, Maus
dreht, Leertaste hoch, Q/E rollen die Sonne, +/− ändern die Zielgröße der Voxel.

`pyrit-render` misst mit: `--profile` schlüsselt die Frame-Zeit auf, `--flicker`
misst zeitliches Flimmern, `--turn` dreht die Kamera im Testflug, `--view` setzt
die Sichtweite.

## Dokumentation

- [`docs/API.md`](docs/API.md) – die API im Detail
- [`pyrit-plan.md`](pyrit-plan.md) – Aufbau, Entscheidungen und Messwerte

## Lizenz

Noch nicht festgelegt.

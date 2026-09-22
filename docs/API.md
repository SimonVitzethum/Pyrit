# Pyrit – API-Handbuch

Pyrit ist ein GPU-Renderer für Sparse Voxel DAGs. Geometrie besteht nur aus Voxeln, es gibt keine Meshes und keine Vertices. Die Runtime liefert Treffer, Tiefe, exakte Motion Vectors, beleuchtete Farbe und ein fertiges Bild. Alles läuft in CUDA-Puffern der Anwendung, auf Wunsch auf den RT-Cores (OptiX).

Pyrit ist komplett in Zig geschrieben. Zig-Programme nutzen die Module direkt; für C, C++, Rust, Java (FFM) usw. beschreibt `include/pyrit.h` dieselbe ABI. Ein Test prüft, dass Header und Implementierung übereinstimmen.

## Bauen und Testen

| Befehl | Zweck |
| --- | --- |
| `zig build --release=fast` | Bibliothek `libpyrit.a` / `libpyrit.so`, Header, PTX |
| `zig build test` | alle Tests ohne GPU (CPU-Referenzen, ABI, AMD-Übersetzung) |
| `zig build kernel-check` | PTX mit `ptxas` für sm_120 prüfen (CUDA-Toolkit, keine GPU) |
| `tools/gpu_free.sh && zig build gpu-test --release=fast` | GPU gegen CPU, Durchsatz – nur bei freier GPU |
| `zig build render -- --out bild.ppm [--vox datei.vox] [--size 1280x720] [--frames 64] [--no-rt] [--no-gi]` | Bild rendern |
| `zig build render -- --world [--scale 2] [--fg] [--dlss\|--rr] --size 1920x1080` | Flug über eine gestreamte Welt: Hochskalieren, Frame Generation, DLSS |
| `zig build -Ddlss-sdk=<pfad>` | mit NVIDIA DLSS (SR und Ray Reconstruction) bauen, siehe „DLSS“ |
| `zig build optix-abi-test -Doptix-include=<pfad>` | OptiX-Anbindung gegen die Original-Header prüfen |

Voraussetzungen zur Laufzeit: NVIDIA-Treiber (libcuda, für RT-Cores libnvoptix). Zum Bauen genügt Zig 0.16; ein CUDA-SDK ist nicht nötig.

## Ablauf

```zig
const pyrit = @import("pyrit");   // Host-API (dieselben Funktionen wie pyrit.h)
const api = pyrit.api;
const types = pyrit.types;

var ci = std.mem.zeroes(api.CreateInfo);
ci.struct_size = @sizeOf(api.CreateInfo);
ci.version = api.version;
var ctx: ?*anyopaque = null;
_ = pyrit.pyr_create(&ci, @ptrCast(&ctx));

// Geometrie auf der GPU bauen (Voxelliste, Reihenfolge beliebig)
var geo: api.Handle = null;
_ = pyrit.pyr_geometry_build(@ptrCast(ctx), 8, voxels.ptr, count, api.build_host_input | api.build_editable, &geo);

var inst: api.Handle = null;
_ = pyrit.pyr_instance_create(@ptrCast(ctx), geo, &inst);
var view: api.Handle = null;
_ = pyrit.pyr_view_create(@ptrCast(ctx), &view);

// pro Frame
_ = pyrit.pyr_instance_set_transform(@ptrCast(ctx), inst, &matrix);   // beliebig viele Änderungen
_ = pyrit.pyr_commit(@ptrCast(ctx), null);                             // neuer Frame
_ = pyrit.pyr_render(@ptrCast(ctx), view, &camera, &targets);          // Treffer, MV, Farbe
_ = pyrit.pyr_postprocess(@ptrCast(ctx), view, &targets, &post);       // TAA, Denoiser, Tonemapping
```

Alle GPU-Arbeit läuft asynchron auf dem Stream des Kontexts (`pyr_cuda_stream`). Eigene Kernel auf demselben Stream sind automatisch richtig geordnet.

**Wichtig bei eigenen Eingabepuffern** (Strahlen, Voxellisten): Den Stream des Kontexts erzeugt Pyrit als nicht blockierend. Er wartet also nicht auf den CUDA-Standard-Stream, und `cuMemcpyHtoD` kann zurückkehren, bevor die Daten auf der GPU sind. Daher entweder `cuMemcpyHtoDAsync(..., pyr_cuda_stream(ctx))` verwenden oder vorher synchronisieren. Alternativ gibt man beim Erzeugen einen eigenen Stream über `PyrCreateInfo.cuda_stream` mit.

## Kontext

`PyrCreateInfo` legt fest:
- Gerät, optional ein vorhandener CUDA-Kontext und Stream,
- Kapazitäten (Instanzen, Geometrien, Ansichten, Poolgrößen),
- Flags:

| Flag | Wirkung |
| --- | --- |
| `PYR_CREATE_DEBUG` | synchrone Fehlerprüfung nach jedem Kernel, OptiX-Validierung |
| `PYR_CREATE_NO_RT` | RT-Cores nie verwenden |
| `PYR_CREATE_FORCE_RT` | RT-Cores immer verwenden |

Ohne Flag entscheidet Pyrit selbst:
- **Mit Shading** (Farbe, Normale oder Albedo angefordert) laufen `pyr_render` und `pyr_trace` immer auf den RT-Cores.
- **Ohne Shading** erst ab 16 Instanzen; darunter ist die CUDA-Traversierung schneller.

`pyr_features` meldet `PYR_FEATURE_RT_CORES`, `pyr_get_stats` Poolbelegung und Anzahlen. Fehler liefern einen `PyrResult`, die Details stehen in `pyr_error_message()` (thread-lokal).

## Geometrie

Eine Geometrie ist ein Würfel aus 2^log2_size Voxeln pro Kante, mit log2_size zwischen 3 und 20. Im Objektraum ist ein Voxel 1 groß.

| Weg | Funktion | Wo |
| --- | --- | --- |
| Voxelliste → DAG | `pyr_geometry_build` | **GPU**: Sortieren, Deduplizieren, Knotenebenen, RT-Primitive |
| Voxel ändern | `pyr_geometry_edit` (Attribut 0 = entfernen) | **GPU**: Neubau aus der gespeicherten Voxelliste |
| DAG auf der CPU bauen | `pyr_dag_build_dense/points/fn` + `pyr_geometry_create` | CPU, dann Upload |
| Datei laden | `pyr_vox_parse` (MagicaVoxel), `pyr_dag_load` | Datei auf der CPU, Bau auf der GPU |
| Speichern | `pyr_geometry_download` + `pyr_dag_save` | |

- **Voxelformat:** `PyrVoxel {x, y, z, attribute}`. Doppelte Koordinaten sind erlaubt, der letzte Eintrag gewinnt.
- **Eingabe:** `pyr_geometry_build` erwartet einen Gerätezeiger. Mit `PYR_BUILD_HOST_INPUT` darf es ein Host-Zeiger sein. `PYR_BUILD_EDITABLE` behält die Voxelliste auf der GPU, damit später `pyr_geometry_edit` möglich ist.
- **Messung** (128³-Szene, 312 000 Voxel, RTX 5070 Laptop, GPU ohne Last): 2,7 ms für den Bau inklusive Upload und GAS; eine Änderung mit 90 000 Einträgen dauert 2,1 ms.
- **Zwischenspeicher:** Der Bau braucht rund 115 Byte je Voxel, nur während des Baus. Er kommt stream-geordnet aus dem CUDA-Speicherpool (`cuMemAllocAsync`) und hält nichts dauerhaft fest.
- **LOD:** `pyr_geometry_downsample(ctx, geo, shift, flags, &lod)` baut auf der GPU eine Fassung mit 2^shift-fach gröberer Auflösung; je grober Zelle bleibt das oberste Voxel. Für dieselbe Weltgröße skaliert man die Instanz um 2^shift. Große Welten entstehen so: Chunks als Geometrien, Instanzen auf einem Gitter, und je nach Entfernung wechselt man mit `pyr_instance_set_geometry` zwischen den Detailstufen.
- **Freigabe:** Geänderte oder gelöschte Geometrien werden erst nach dem nächsten `pyr_commit` freigegeben, weil der alte Zustand bis dahin noch gerendert werden kann.

## Instanzen und Frames

Eine Instanz ist eine Geometrie mit Transformation (3×4, Objekt → Welt), Maske und Benutzerwert. Ihr Index ist über ihre ganze Lebensdauer stabil, er erscheint in `PyrHit.instance`.

- Änderungen werden mit `pyr_commit` aktiv. Der alte Zustand wird dann zum Vorframe und liefert die Motion Vectors.
- `pyr_instance_reset_history` markiert einen Teleport. Die Treffer tragen dann `PYR_HIT_NEW`, und der Motion Vector kommt nur aus der Kamerabewegung.
- **Masken:** Auf den RT-Cores wirken nur die unteren 8 Bits.

## Rendern

`pyr_render(ctx, view, camera, targets)`: Die Ansicht merkt sich ihre Kamera vom Vorframe.

| Ziel in `PyrTargets` | Inhalt |
| --- | --- |
| `hits` | `PyrHit` (t, Instanz, Attribut, Fläche und Flags) |
| `depth` | lineare Tiefe entlang der Blickachse |
| `motion` | Motion Vector in Pixeln: Position im Vorframe − jetzt, ohne Jitter, exakt pro Pixel |
| `color` | lineare HDR-Farbe; a = 1 bei Treffer, 0 bei Himmel |
| `normal` | Weltnormale, w = lineare Tiefe |
| `albedo` | Albedo |
| `transparent_mask` | Instanzmaske der transparenten Ebene (Wasser, Glas), siehe unten |

Nicht benötigte Ziele bleiben 0. Shading wird nur berechnet, wenn `color`, `normal` oder `albedo` gesetzt ist.

**Transparenz:** Durchsichtig ist, was entweder über die Instanzmaske (`(mask & transparent_mask) != 0`) oder über das Material (`PYR_MATERIAL_TRANSPARENT`) gekennzeichnet ist. Die Material-Variante braucht keine eigene Instanz: Wasser kann in derselben Chunk-Geometrie wie der Boden liegen – die Traversierung überspringt solche Voxel für Primär- und Schattenstrahlen (`PYR_TRACE_SKIP_TRANSPARENT`). Bis zu `PYR_MAX_TRANSPARENT_LAYERS` (4) Körper hintereinander werden je Pixel verfolgt:
- **Eintritt:** Fresnel-Reflexion (`ior`), Deckkraft der Oberfläche (`opacity`), mit `PYR_MATERIAL_REFRACT` Brechung nach Snell.
- **Im Körper:** Absorption `base_color` hoch `density · Strecke` bis zum echten Austritt (die Traversierung findet den Übergang in leere Voxel).
- **Austritt:** Fresnel-Transmission und Rückbrechung; eine Glasscheibe versetzt den Strahl also parallel. Liegt ein Körper auf dem Untergrund (Wasser auf Boden), endet das Medium dort.
- Ohne `PYR_MATERIAL_REFRACT` läuft der Strahl gerade hindurch (dünne Scheiben, Laub).
- **Medium:** Eine eigene Instanz endet dort, wo ihre Voxel enden; ein nur über das Material markierter Körper reicht bis zum Untergrund – genau richtig für Wasser über Grund.
- **Schatten:** Durchsichtige Körper blockieren Schattenstrahlen nicht, sondern tönen sie (Deckkraft und Absorption). Unter Wasser liegt damit blaues Licht statt Schwarz.
- **Wellen:** `PYR_MATERIAL_WAVES` stört die Normale zeitabhängig (`wave_height`, `wave_length`, `wave_speed`). Die Geometrie bleibt stehen, Treffer, Tiefe und Motion Vectors bleiben exakt.
- Treffer, Tiefe, Motion Vectors und Denoiser-Puffer beziehen sich auf den undurchsichtigen Untergrund. Schatten- und GI-Strahlen ignorieren transparente Körper.

**Kamera:** `pyr_camera_look_at`, `pyr_camera_perspective`, `pyr_camera_orthographic`. Für TAA liefert `pyr_jitter_halton(frame)` den Subpixel-Versatz, der in `camera.jitter` gehört.

## Materialien und Licht

**Voxelattribut:** Bits 0..7 sind der Materialindex, Bits 8..31 die Farbe `0xRRGGBB` (sRGB).
- Erzeugen mit `pyr_voxel_attribute(material, r, g, b)` oder `PYR_VOXEL(...)`.
- Attribut 0 heißt „leer“.

**Materialien:** 256 Stück über `pyr_material_set`.
- Felder: Grundfarbe, Rauheit, Metall, Emission.
- Mit `PYR_MATERIAL_VOXEL_COLOR` wird die Grundfarbe mit der Voxelfarbe multipliziert; das ist der Standard.

**Licht:** `pyr_set_lighting`, Voreinstellung über `pyr_lighting_default`.
- Sonne mit Winkelradius für weiche Schatten
- Himmel: Zenit, Horizont, Boden
- bis zu 16 Punktlichter als Kugeln mit Radius
- Flags `PYR_LIGHTING_SHADOWS`, `_GI` (eine indirekte Reflexion), `_AO` (günstigere Alternative zu GI), `_SUN_DISK`, `_REFLECTIONS` (Reflexionsstrahlen für Oberflächen mit Rauheit < 0,5, gestreut nach Rauheit)

- `gi_distance` begrenzt die Reichweite der indirekten Strahlen (0 = unbegrenzt); darüber zählt der Himmel: 64 Voxel sparen rund 20 % Renderzeit bei etwa 1 % Bildunterschied.
- `PYR_LIGHTING_GI_HALF` rechnet die indirekte Beleuchtung in halber Auflösung (ein Strahl je 2x2-Block, wandernder Abtastpunkt) und skaliert sie kantenbewusst hoch: **halbe Renderzeit, aber deutlich mehr Flimmern**. Ein Strahl versorgt vier Pixel, das Rauschen ist damit über den 2x2-Block korreliert – räumlich kaum wegzufiltern und zeitlich nur langsam wegzumitteln. Gemessen bei *stehender* Kamera als mittlerer Unterschied aufeinanderfolgender Bilder: 0,89 gegen 0,28 bei voller Auflösung (Faktor 3,2). In einem Standbild sieht man davon nichts, im laufenden Bild sehr wohl. Standard im Betrachter ist daher volle Auflösung; halbe Auflösung lohnt, wo Renderzeit wichtiger ist als Ruhe im Bild. Braucht die Ziele `color`, `normal`, `albedo` und `hits`.
- `secondary_bias` versetzt Schatten-, GI- und Reflexionsstrahlen entlang der Normale, anteilig zur Trefferentfernung – nötig bei `PyrTargets.secondary_mask`.

Das Shading nutzt Lambert und GGX. Schatten- und GI-Strahlen laufen über dieselbe Strahlverfolgung wie die Primärstrahlen.

### Materialien über Farbe und Rauheit hinaus

- **Texturen** (`pyr_texture_create`, RGBA8): Index 1-basiert in `material.texture` bzw. `normal_texture`, Kachelgröße über `texture_scale` in Welteinheiten. Voxelflächen sind achsenparallel, deshalb wird genau eine Ebene projiziert – Triplanar-Mischen wäre hier Verschwendung. Gefiltert wird von Hand (bilinear), ohne Texturhardware, damit derselbe Code später auf AMD läuft.
- **Detailnormale** ohne Textur: `normal_strength` und `normal_scale` erzeugen sie aus Wertrauschen. Beleuchtet wird mit der gestörten, weiterverfolgt mit der geometrischen Normale; im Ziel `normal` steht die geometrische, damit Denoiser und Reprojektion stabil bleiben.
- **Klarlack** (`clearcoat`, `clearcoat_roughness`): eine zweite, glatte Schicht. Was sie reflektiert, fehlt darunter.
- **Unterflächenstreuung** (`subsurface`, `subsurface_color`): Licht von hinten kommt getönt durch (Laub, Haut, Wachs). Getrennt vom BRDF gerechnet, sonst zählt es doppelt.

### Lichtquellen und Umgebung

- `PyrLight.kind`: `PYR_LIGHT_SPHERE` (Punkt mit Radius), `PYR_LIGHT_RECT` (Flächenlicht, `normal` und halbe Kanten in `size`) und `PYR_LIGHT_SPOT` (Kegel, `size` = cos innen/außen). Bis zu 64 Stück.
- **Umgebungskarte** (`pyr_environment_set`, equirektangulär, 4 Floats je Texel): Pyrit baut daraus eine Verteilung (Summenfunktion je Zeile plus eine über die Zeilen, mit sin θ gewichtet) und tastet sie nach Helligkeit ab. Der GI-Strahl tastet dieselbe Karte über den Cosinus-Lappen ab; beide Anteile werden nach der Potenz-Heuristik gewichtet (MIS), sonst zählt die Karte doppelt. Gemessen an einer Karte mit 0,03 rad großer, 4000× heller Sonne: **ohne** Importance-Sampling liegt der Boden bei Helligkeit 46 statt 114 – der Cosinus-Strahl trifft die Scheibe praktisch nie, das Licht fehlt schlicht.
- **Mehrere Reflexionen** (`gi_bounces`): ab der zweiten entscheidet russisches Roulette, der Erwartungswert bleibt richtig. Gemessen (1 → 3): dunkle Bereiche 43,1 → 46,6, Renderzeit 11,6 → 14,1 ms.
- **Teilnehmendes Medium** (`fog_density`, `fog_color`, `fog_height`, `fog_falloff`, `fog_anisotropy`, `fog_steps`): Strahlmarschierung entlang des Sichtstrahls mit Sonnenabtastung je Schritt – das ergibt die Lichtschächte. Die Schrittlage wird je Pixel verschoben, das Rauschen daraus nimmt der zeitliche Filter weg.

### Kamera- und Bildeffekte

`PyrPostInfo.fx` zeigt auf `PyrPostFx` und schaltet Bloom, Tiefenschärfe, Bewegungsunschärfe, Belichtungsautomatik und Farbkorrektur zu. Die Kette läuft auf dem fertigen HDR-Bild in Ausgabeauflösung; Bewegung und Tiefe liefert TAAU ohnehin schon. Der Autofokus liest die Tiefe in der Bildmitte direkt auf der GPU, es gibt keinen Rückkanal zum Host. Die Belichtungsautomatik misst die mittlere Log-Helligkeit über jedes 16. Pixel mit Festkomma-Atomics und führt sie gedämpft nach; ihr Zustand bleibt auf der GPU.

## Nachbearbeitung, Hochskalieren, DLSS, Frame Generation

`pyr_postprocess(ctx, view, targets, post)` läuft vollständig auf der GPU:
1. **Temporal:** Die Beleuchtung wird mit den Motion Vectors reprojiziert und über Normale und Tiefe auf Gültigkeit geprüft (Rauschreduktion für GI). `clamp_sigma` (typisch 1,5) begrenzt Geisterbilder.
2. **À-trous-Denoiser, varianzgeführt:** `denoise_iterations` Schritte (typisch 3–5), kantenerhaltend über Normale, Tiefe und Helligkeit. Die Helligkeitstoleranz kommt aus der *gemessenen* Varianz: die Akkumulation führt die Momente der Helligkeit mit (zeitlich, in den ersten Frames räumlich geschätzt), der Filter glättet sie 3x3 und filtert sie mit quadrierten Gewichten mit. Dadurch wird verrauschtes Gebiet geglättet, statt sein Rauschen für Kanten zu halten – vor allem auf dunklen, indirekt beleuchteten Flächen und bei bewegter Kamera, wo nur wenige Frames akkumuliert sind. `denoise_phi` steuert die Stärke (0 = 4; kleiner = glatter, größer = mehr Details und mehr Rauschen). Gefiltert wird die Beleuchtung ohne Albedo, damit Voxelfarben scharf bleiben.
   Gemessen (CPU-Test, 5 akkumulierte Frames, dunkle Fläche mit Einzelsample-Rauschen): Restrauschen 0,0026 → 0,0013 bei zugleich besser erhaltener Kante (Sprunghöhe 0,60 → 0,90 der echten Kante).
3. **Upscaler** (`post.upscaler`), Ausgabe in `output_width × output_height` (0 = Renderauflösung):

| Upscaler | Verfahren |
| --- | --- |
| `PYR_UPSCALER_AUTO` / `_TAAU` | eigenes TAAU: gejitterte Samples an ihrer echten Subpixelposition rekonstruiert, Verlauf per Catmull-Rom, Varianzbegrenzung in YCoCg, MV des vordersten Nachbarn. Bei Faktor 1 gewöhnliches TAA. Bis 4× |
| `PYR_UPSCALER_DLSS` | NVIDIA DLSS Super Resolution hinter dem eigenen Denoiser |
| `PYR_UPSCALER_DLSS_RR` | NVIDIA DLSS Ray Reconstruction: ersetzt Denoiser und TAA; bekommt verrauschte Farbe, Albedo, Normalen, Tiefe, MVs sowie Rauheit und spiegelnde Albedo (dafür das Ziel `material` setzen) |
| `PYR_UPSCALER_NONE` | nur Renderauflösung, ohne TAAU |

4. **Tonemapping:** ACES, Reinhard oder keines. Ausgabe als HDR (`output_hdr`) und/oder RGBA8 sRGB (`output_ldr`).

### Überlappung mit dem nächsten Frame

Die Nachbearbeitung läuft auf einem eigenen CUDA-Stream, angebunden an das Rendern *dieses* Frames über ein Ereignis. DLSS und TAA arbeiten damit auf den Tensorkernen, während die Shader- und RT-Einheiten schon den nächsten Frame rechnen können. Voraussetzung ist `PYR_CREATE_ASYNC_POST`: ohne das Flag wartet der nächste Frame auf die Nachbearbeitung des vorigen, weil er sonst in dieselben Ziele schreiben würde, aus denen noch gelesen wird.

Mit dem Flag muss die Anwendung **zwei Sätze von `PyrTargets` abwechselnd** benutzen und darf nicht nach jedem Frame synchronisieren – sonst gibt es nichts zu überlappen. `pyr_synchronize` wartet auf beide Ströme.

Nachgemessen: das überlappende Bild ist **pixelgleich** zum seriellen (RMSE 0,000), die Synchronisation stimmt also. Ein Geschwindigkeitsgewinn ließ sich bisher nicht messen, weil die Test-GPU zu 99 % von einem anderen Prozess belegt war – ohne freie Einheiten kann Überlappung nichts gewinnen. Die Zahl steht noch aus.

Für TAAU und DLSS gehört pro Frame ein Jitter in die Kamera (`pyr_jitter_halton(frame)` → `camera.jitter`). Die Verlaufspuffer gehören der Ansicht. `PYR_POST_RESET` verwirft den Verlauf, etwa bei einem Kameraschnitt.

**Frame Generation** (`pyr_frame_generate(ctx, view, &info)`), reines CUDA: Nach `pyr_postprocess` von Frame N entsteht ein Zwischenbild zwischen N−1 und N (`info.t`, Standard 0,5; für mehrere Zwischenbilder mehrfach mit 1/3, 2/3 …). Grundlage sind die exakten Motion Vectors und die Tiefe: Jedes Pixel wirft seinen Bewegungsvektor in das Zwischenbild (die kleinste Tiefe gewinnt, also die richtige Verdeckung); wo nichts ankommt, sucht ein Fixpunktverfahren. Das Zwischenbild wird vor Frame N angezeigt. Eigene Verfahren hängt man über `info.generate` ein: Pyrit ruft die Funktion mit allen Eingaben (`PyrFrameGenParams`: beide Frames in HDR, MV und Tiefe in Ausgabeauflösung) und dem Stream auf. Funktioniert hinter TAAU und hinter DLSS. DLSS-FG selbst ist ohne D3D12/Vulkan nicht nutzbar.

**DLSS einbinden:** Das SDK (github.com/NVIDIA/DLSS) wird nicht mitgeliefert. `zig build -Ddlss-sdk=<pfad>` übersetzt die NGX-Header beim Bauen und linkt `libnvsdk_ngx.a` (braucht libstdc++ des Systems). Zur Laufzeit sucht NGX `libnvidia-ngx-dlss*.so` im SDK-Verzeichnis `lib/Linux_x86_64/rel`, in `$PYRIT_DLSS_PATH` oder im Programmverzeichnis; für die Auslieferung gelten die Lizenzbedingungen des SDK. Ohne DLSS-Build liefern die DLSS-Upscaler `PYR_ERROR_NOT_FOUND`.

**Messung** (RTX 5070 Laptop, Flug über die gestreamte Welt mit Schatten + GI, Ausgabe 1920×1080):

| Einstellung | ms pro Frame |
| --- | --- |
| nativ 1080p, TAA | 14,8 |
| nativ mit `PYR_LIGHTING_GI_HALF` | 9,6 |
| 960×540 → 1080p, TAAU | 5,3 |
| dazu ein Zwischenbild je Frame (FG) | 6,1 für 2 angezeigte Bilder |
| 960×540 → 1080p, DLSS SR | 8,9 |
| 960×540 → 1080p, DLSS Ray Reconstruction | 12,0 |

## Große Welten

Eine Welt streamt Chunks um die Kamera: nah fein, fern grob (Octree über LOD-Stufen). Jeder Chunk hat 2^`chunk_log2` Voxel pro Kante; auf Stufe l ist ein Voxel 2^l Grundvoxel groß.

```zig
var wi = std.mem.zeroes(api.WorldInfo);   // Standard: 32³-Chunks, 8 Stufen, eingebautes Gelände
var world: ?*anyopaque = null;
_ = pyrit.pyr_world_create(ctx, &wi, &world);
// pro Frame
_ = pyrit.pyr_world_update(ctx, world, &kamera_welt, &origin);   // wartet nie auf die GPU
_ = pyrit.pyr_commit(ctx, &.{ .time = t, .origin = origin });
```

- **Erzeugung auf der GPU:** Ein Generator-Kernel schreibt die Voxel eines Chunks direkt in der Auflösung seiner Stufe und nur die sichtbare Haut, nie ein volles Volumen. Eingebaut ist ein Höhenfeld-Gelände (`PyrTerrainInfo`, Höhe abfragbar mit `pyr_terrain_height`) mit Wasser bis `sea_level` (`attr_water`, Material mit `PYR_MATERIAL_TRANSPARENT`) und Bäumen (`attr_leaves`, `attr_wood`, `tree_density`). Eigene Generatoren setzen `info.generate`: Pyrit übergibt je Batch die Chunk-Schlüssel und Ausgabepuffer (`PyrWorldGenParams`) und den Stream.
- **Bau:** Bis zu `chunks_per_update` Chunks entstehen in einem GPU-DAG-Bau (Teilbäume batchweit dedupliziert), alle GAS ohne Synchronisation. Erzeugen und Bauen laufen auf einem eigenen Thread und CUDA-Stream.
- **Übergänge:** Grobe Chunks bleiben sichtbar, bis alle feineren fertig sind; es entstehen keine Löcher. Neue Chunks übernehmen den Verlauf von TAA und Denoiser (`PYR_INSTANCE_KEEP_HISTORY`).
- **Speicher:** Nicht mehr gebrauchte Chunks werden nach `keep_frames` freigegeben. Leere Bereiche kosten nichts. Das Standardgelände mit etwa 20 000 Voxeln Sichtweite belegt rund 11 MiB.
- **Genauigkeit:** Die Welt rechnet in f64 und legt Instanzen relativ zum Render-Ursprung ab. Diesen Ursprung in großen Schritten mitführen, zum Beispiel alle 1024 Voxel.
- **Überlauf gibt es nicht:** Passt ein Chunk nicht in `chunk_capacity`, wird die Kapazität erhöht und der Auftrag wiederholt, statt Voxel abzuschneiden. Der Generatorpuffer wächst dabei mit, solange Grafikspeicher frei ist (ein Achtel des Gesamtspeichers, mindestens 128 MiB, bleiben unangetastet); erst wenn es eng wird, sinkt stattdessen die Zahl der Chunks je Auftrag, im Grenzfall auf einen. Abgebrochen wird nie. Nachgemessen mit `chunk_capacity = 256`: die Kapazität wächst in vier Schritten auf 6672, die Chunkzahl je Auftrag bleibt bei 64, und das fertig geladene Bild ist **pixelgleich** zu dem mit der Vorgabe. Ohne mitwachsenden Puffer bräuchte derselbe Aufbau 1867 statt 61 Updates (9,9 s statt 0,6 s). `PyrWorldStats.overflow_chunks` zählt die Vergrößerungen.
- `pyr_world_wait` wartet auf den laufenden Batch (Ladebildschirm, Teleport). `PYR_WORLD_SYNC` baut ohne Thread.
- **Messung:** 1063 Chunks in 114 ms aufgebaut. Im Flug (1080p, 512 Minecraft-Chunks Sichtweite, 6500 Chunks resident) kostet `pyr_world_update` auf dem Hauptthread im Mittel 5,3 ms, aufgeteilt in Planer 1,8 ms, Auftrag 1,3 ms, Sichtbarkeit 0,7 ms, Übernahme 0,04 ms, Verdrängung 0,1 ms. (Vorher 25,1 ms: der Planer schlug jeden Knoten neunmal in der Hashtabelle nach – acht Kinder prüfen, dieselben acht beim Absteigen erneut – und benutzte die allgemeine, bytefweise Streuung. Beides behoben: 8,4 → 1,8 ms.)

### Die Welt verändern

`pyr_world_edit(ctx, world, edits, count)` setzt oder entfernt einzelne Grundvoxel:

```c
PyrWorldEdit e[2] = {
    { .x = 100, .y = 70, .z = -3, .attribute = PYR_VOXEL(0, 220, 40, 40) },  /* setzen */
    { .x = 101, .y = 70, .z = -3, .attribute = 0 },                          /* entfernen */
};
pyr_world_edit(ctx, world, e, 2);
```

- Koordinaten sind **Grundvoxel**, unabhängig davon, in welcher Stufe der Chunk gerade vorliegt.
- Die Änderungen liegen als **Überlagerung** über dem Generator und sind die Wahrheit: betroffene Chunks werden sofort neu gebaut (mit Vorrang vor dem Nachladen, und der alte Chunk bleibt sichtbar, bis der neue fertig ist – kein Loch), und jede spätere Neuerzeugung trägt sie wieder auf. Damit überleben sie Verdrängung und LOD-Wechsel, ohne dass ein Chunk-Cache nötig wäre.
- **Auf gröberen Stufen** füllt Hinzufügen die Zelle immer (eine grobe Zelle gilt als belegt, sobald irgendetwas darin liegt). Entfernen wirkt erst, wenn *alle* Grundvoxel der Zelle entfernt sind – sonst risse ein einzelnes abgebautes Voxel in der Ferne ein ganzes Loch. Bis Stufe 8 wird mitgezählt, darüber sind es über 2^24 Grundvoxel je Zelle.
- **Kosten:** `pyr_world_edit` kostet auf dem Hauptthread **0,025 ms** je Aufruf – es trägt die Änderung nur in die Überlagerung ein und merkt die betroffenen Chunks vor; erzeugt und gebaut wird auf dem Hintergrund-Thread. Eine Entfernung merkt nur die Stufen vor, auf denen sie wirklich wirkt. Geänderte Chunks bekommen höchstens die Hälfte eines Auftrags, damit ein Strom von Änderungen das Nachladen nicht aushungert. Gemessen mit einer Änderung pro Frame auf einer fertig geladenen Welt: `pyr_world_update` 1,54 ms, Welt bleibt vollständig geladen.
- **Ablauf auf der GPU:** Nach dem Generator bekommt jeder Chunk seine Änderungsliste bereits in seiner Stufenauflösung. Drei Durchgänge: vorhandene Voxel überschreiben oder als entfernt markieren, überlebende dicht in einen zweiten Puffer schreiben (dabei entstehen die Zähler neu, die Präfixsummen bleiben exakt), unverbrauchte Änderungen anhängen.

`pyr_world_edits_bytes` / `pyr_world_edits_save` / `pyr_world_edits_load` sichern die Überlagerung in einen Puffer und zurück. Pyrit fasst keine Dateien an – wohin der Puffer geht, entscheidet der Aufrufer. Gespeichert werden nur die Änderungen, nicht das Gelände: das erzeugt der Generator jederzeit wieder. Nachgemessen: gesichert, in eine **frische** Welt geladen und gerendert ergibt ein pixelgleiches Bild.

## Animation

**Starre Bewegung:** `pyr_instance_set_transform` genügt. Die Motion Vectors sind exakt.

**Wasser und Laub** bewegen sich über `PYR_MATERIAL_WAVES` (zeitabhängige Normale), ohne dass sich Geometrie ändert.

**Skelette mit Voxel-Teilen** laufen vollständig auf der GPU:

```zig
_ = pyrit.pyr_skeleton_create(ctx, &bones, n, &skel);     // Knochen: Eltern, Voxel-Teil, Ruhelage
_ = pyrit.pyr_clip_create(ctx, skel, &keys, k, 2.0, PYR_CLIP_LOOP, &clip);
_ = pyrit.pyr_actor_create(ctx, skel, &.{ .root = lage, .clip = clip, .speed = 1, ... }, &actor);
// danach kein Aufruf pro Frame nötig: pyr_commit spielt ab
```

- Bei jedem `pyr_commit` rechnet ein Kernel für jeden Knochen aller Akteure die Lage aus der Szenenzeit. Dabei werden Keyframes interpoliert (Slerp), zwei Clips überblendet (`blend_clip`, `blend`) und die Kette bis zur Wurzel multipliziert. Das Ergebnis (Transformation, Inverse, AABB) schreibt der Kernel direkt in die Instanzen.
- Der Vorframe liegt im zweiten Instanzpuffer, deshalb sind die Motion Vectors exakt. Auf den RT-Cores wird die IAS nur nachgeführt (Refit).
- `pyr_actor_set` wechselt Clip, Überblendung, Lage, Maske oder Benutzerwert; `pyr_skeleton_destroy` und `pyr_clip_destroy` geben sie frei, sobald keine Akteure mehr darauf laufen.
- **Messung:** 2000 Akteure mit je 3 Teilen kosten 0,09 ms pro `pyr_commit`, einschließlich IAS-Refit.

**Voxel ändern:** `pyr_geometry_edit` baut auf der GPU neu. Mit dem Wechsel der Geometrie ändert sich die Vorgeschichte. Für Daumenkino-Animation schaltet `pyr_instance_set_geometry` zwischen Geometrien um.

## Strahlen

`pyr_trace(ctx, rays, hits, count, mask, flags)` nimmt `PyrRay[count]` als Gerätezeiger.

| Flag | Wirkung |
| --- | --- |
| `PYR_TRACE_ANY_HIT` | der erste gefundene Treffer genügt (Schatten, Sichtbarkeit) |
| `PYR_TRACE_NO_ATTRIBUTE` | spart die Attributbestimmung |
| `PYR_TRACE_EXTENDED` | schreibt `PyrHitEx` mit Voxelkoordinate, Weltposition und Weltnormale, z. B. für Picking im Editor |

## Eigene GPU-Kernel (Zig)

Das Modul `pyrit_device` enthält Traversierung, Szene, Shading-Bausteine und Nachbearbeitung. Derselbe Code läuft auf CPU, NVIDIA (nvptx64) und AMD (amdgcn).

```zig
const pyr = @import("pyrit_device");

export fn schatten(scene: *const pyr.Scene, ...) callconv(.nvptx_kernel) void {
    const hit = pyr.trace(scene, ray, 0xFFFF_FFFF, pyr.trace_any_hit | pyr.trace_no_attribute);
    const lit = hit.instance == pyr.no_hit;
}
```

Den Szenenzeiger liefert `pyr_scene_device`. Er bleibt über alle Frames gleich, sein Inhalt wird bei `pyr_commit` in Stream-Reihenfolge aktualisiert. In eigenen Kerneln läuft die Traversierung auf den CUDA-Kernen.

## Leistung

Gemessen mit RTX 5070 Laptop bei 1920×1080, eine Geometrie mit 128³ Voxeln:

| Szene | Nur Primärstrahlen | Volle Pipeline (Schatten, GI, TAA, Denoiser) |
| --- | --- | --- |
| 3 Instanzen | 0,8 ms | 10 ms |
| 1000 Instanzen | 6,1 ms | 31 ms |

Die Messungen der vollen Pipeline stammen aus einem Lauf ohne fremde GPU-Last. Die RT-Cores sind bei vielen Instanzen und bei Sekundärstrahlen deutlich schneller. Sie werden automatisch gewählt, siehe Abschnitt „Kontext“.

**Flug über die gestreamte Welt, 1920×1080, Schatten + GI, 8 px je Voxel** (Anteile einzeln gemessen):

| Einstellung | Rendern | Nachbearbeitung | gesamt je Frame |
| --- | --- | --- | --- |
| Standard | 10,7 ms | 2,8 ms | 14,8 ms |
| `gi_distance = 64` | 8,4 ms | 2,8 ms | 12,5 ms |
| zusätzlich grobe Fassung für Sekundärstrahlen | 7,6 ms | 2,8 ms | 11,7 ms |
| 960×540 → 1080p (TAAU) | – | – | 5,3 ms |

Der GI-Bounce macht etwa die Hälfte der Renderzeit aus (ohne GI: 5,4 ms). Die Nachbearbeitung besteht aus temporaler Akkumulation (~1,2 ms), À-trous (0,67 ms je Schritt) und TAAU mit Tonemapping.

## Grenzen

- **Transparenz:** Schatten- und GI-Strahlen sehen transparente Körper nicht; innere Totalreflexion wird angenähert (der Strahl läuft gerade weiter).
- **Animation:** Knochen sind starr; weiche Verformung gibt es nur als Normalenstörung (`PYR_MATERIAL_WAVES`), nicht als echte Voxelverschiebung.
- **DLSS Ray Reconstruction:** Jitter- und Matrixkonventionen sind nicht gegen eine NVIDIA-Referenz geprüft.
- **DLSS Frame Generation:** braucht D3D12 oder Vulkan, daher die eigene CUDA-Frame-Generation.
- **ROCm/HIP:** Die Kernel übersetzen bereits, die Laufzeitseite fehlt.

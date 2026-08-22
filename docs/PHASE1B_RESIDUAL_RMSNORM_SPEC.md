# Phase 1B — vorregistrierter statischer Residual-Add+RMSNorm-Kandidat

**Status:** eingefroren am 22.08.2026, einschließlich des dokumentierten
Darwin-Limit-Preflights, bevor der Custom-Kernel konstruiert, kompiliert oder
ausgeführt wurde.

**Contract:** `phase1b-residual-rmsnorm-v1`

**Forschungsfrage:** Kann genau ein statischer, sicherer MLX-Custom-Metal-Kernel
auf dem vorhandenen Apple M1 Max die feste FP16-Operation
`residual_add + RMSNorm` gegenüber der schnellsten korrekten MLX-Baseline um
mindestens 5 % beschleunigen, ohne den eingefrorenen numerischen oder
Speichervertrag zu verletzen?

Ein negatives, instabiles oder nicht reproduzierbares Ergebnis ist terminal und
gültig. Diese Vorregistrierung erlaubt keine Kandidatenmutation, keine adaptive
Suche, keine Modellinferenz, keine Produktionsintegration und keine Aussage für
andere Shapes, Dtypes, Geräte oder vollständige Transformer.

## 1. Autorisierter Scope und feste Identitäten

Der Nutzer hat den isolierten Kernelversuch und die Nutzung vorhandener CPU/GPU
freigegeben. Der Versuch benötigt weder Download noch Installation und lädt kein
Modell. Die Gemma-Nähe besteht ausschließlich darin, dass der bereits lokal
belegte Gemma-3-4B-Config-Snapshot `hidden_size=2560` ausweist.

| Feld | Eingefrorener Wert |
|---|---|
| Workload-ID | `residual_rmsnorm:r1024:h2560:f16:eps1e-6:v1` |
| Kandidat-ID | `tg256_halfcache_fp32reduce_safe_v1` |
| Qualification-Run-ID | `phase1b-qualify-20260822-01` |
| Benchmark-Run-ID | `phase1b-benchmark-20260822-01` |
| History-DB | `.friday-data/phase1b-rmsnorm.sqlite3` |
| Dashboard | read-only `127.0.0.1:8774` |
| Rows | exakt `1024` |
| Hidden Size | exakt `2560` |
| `x`, `residual`, `weight`, `y` | exakt `float16` |
| `x`, `residual`, `y` Shape | exakt `(1024, 2560)` |
| `weight` Shape | exakt `(2560,)` |
| Epsilon | exakt `1e-6` |
| Layout | feste C-kontiguierliche Host-Fixtures; MLX `ensure_row_contiguous=True` |
| Math Mode | MLX Metal `safe` |
| Threadgroup | exakt `(256, 1, 1)` |
| Grid | exakt `(262144, 1, 1)` Dispatch-Threads |
| Threadgroups | exakt `1024`, eine pro Zeile |

Die beiden Run-IDs sind once-only. Ein vorhandener terminaler Record sperrt die
Wiederholung. Qualification und Benchmark müssen denselben sauberen Git-Commit,
Codehash, Spezifikationshash, Quellenhash und Hardware-/Softwarefingerprint binden.

## 2. Eingefrorene Semantik

Die Operation ist elementweise beziehungsweise pro letzter Achse:

```text
z_f16 = fp16(fp32(x) + fp32(residual))
mean_square = fp32_reduce_sum(fp32(z_f16)^2) / 2560
y_f16 = fp16(fp32(z_f16) * rsqrt(mean_square + 1e-6) * fp32(weight))
```

FP16-Materialisierung von `z`, FP32-Reduktion, Safe-Math und FP16-Output sind
Semantik, keine frei veränderbaren Optimierungsparameter. Die Reduktionsreihenfolge
darf von der Host-Referenz abweichen, aber nur innerhalb der festen Accuracy-Gates.
NaN oder Inf in Input oder Output macht den Fall und damit den Lauf ungültig.

Der unabhängige Host-Oracle erzeugt `z_f16` exakt wie oben, akkumuliert dessen
Quadrate dann in NumPy-FP64, führt Normierung und Gewichtung in FP64 aus und rundet
erst das Ergebnis auf FP16. Er verwendet nie die ursprünglichen, noch nicht auf
FP16 gerundeten Zufallswerte als Tensorinput.

## 3. Genau ein Kandidat

Die einzige erlaubte Metal-Quelle ist die Konstante
`friday_phase1b.kernel_source.KERNEL_SOURCE`. Ihr SHA-256 ist eingefroren als:

`33b626c16c79819d6995d6bb78745eb1fd81face648b59f505a924d3125da6f6`

Der einzige erlaubte Kernelname lautet:

`friday_rrms_f16_r1024_h2560_33b626c16c79`

Der Quellenhash ist Teil des Namens. Das schließt die für die lokal verwendete
MLX-Version 0.32.0 dokumentierte Same-Name/Stale-Source-Kollision aus. Quelle,
Header, Compile-Option, Templateparameter, Shapes, Dtypes, Grid und Threadgroup
sind Konstanten; Worker, CLI, DB und Environment dürfen keinen Sourcecode,
Kernelparameter oder Compilerflag annehmen.

MLX 0.32.0 exponiert für `mx.array` kein öffentliches `strides`-Attribut. Daher
prüft der Worker die C-Kontiguität der einzigen erlaubten NumPy-Hostfixtures vor
der Konvertierung und verwendet zusätzlich `ensure_row_contiguous=True`. Es gibt
keinen externen Tensorinput. Eine eventuell durch fremde Inputs ausgelöste
Layoutkopie liegt damit außerhalb des erreichbaren Messpfads.

Der Kandidat speichert die 2560 FP16-Residualwerte einer Zeile in `5156` Bytes
statischem Threadgroup-Speicher (`5120` Bytes Werte, `32` Bytes acht FP32-
Partialsummen und `4` Bytes inverse RMS), reduziert erst innerhalb von acht
32er-SIMD-Gruppen und danach in der ersten SIMD-Gruppe. Jede Spalte wird genau
einmal geschrieben; die Hostvalidierung garantiert alle Grenzen. Guardregions
oder harte GPU-Speicherisolation werden ausdrücklich nicht behauptet.

Referenzen für den verwendeten öffentlichen Pfad sind die
[MLX-Custom-Metal-Dokumentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html),
die [`mx.fast.rms_norm`-API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html)
und der dokumentierte [MLX-0.32.0-Namenskollisionsfehler](https://github.com/ml-explore/mlx/issues/3832).

## 4. Korrektheitsmatrix

Alle Zufallsfixtures entstehen mit NumPy `PCG64` zuerst in Host-FP32 und werden
dann deterministisch auf FP16 gerundet. Jeder Fall verwendet die volle feste
Shape; Fixtures, Metadaten und tatsächliche FP16-Bytes erhalten SHA-256-Digests.

| Fall | Seed | Definition | Zusatzgate |
|---|---:|---|---|
| `zeros` | — | alle Eingaben null, Gewicht eins | Output bitgenau null |
| `cancellation` | `0xB16B0001` | `residual=-x`, Gewicht in `[0.75,1.25)` | Output bitgenau null |
| `constant` | — | `x=0.5`, `residual=0.25`, Gewicht eins | alle Zeilen identisch |
| `visible_normal` | `0xB16B0002` | Normalverteilung, je `sigma=0.5`; Gewicht `[0.75,1.25)` | allgemeines Gate |
| `visible_bounded` | `0xB16B0003` | beide Inputs uniform `[-4,4)`; Gewicht `[0.5,1.5)` | allgemeines Gate |
| `holdout_normal` | `0xB16B4001` | Normalverteilung, je `sigma=1.25`; Gewicht `[0.5,1.5)` | allgemeines Gate |

Der Holdout wird im gleichen Qualification-Prozess erst nach allen sichtbaren
Fällen ausgeführt. Weil es keine nachträgliche Mutation oder zweite Qualification
gibt, kann der Kandidat darauf nicht angepasst werden.

Für Kandidat und jede Baseline gelten unabhängig:

- Shape und Dtype exakt wie Abschnitt 1;
- alle Werte endlich;
- `abs_max <= 0.0078125` gegen den FP64-Oracle;
- `rel_q99 <= 0.005` für Oracleelemente mit `abs(oracle) >= 0.125`;
- `normalized_l2 <= 0.0015`;
- `zeros` und `cancellation` bitgenau null;
- Kandidat und stärkste korrekte Baseline zusätzlich elementweise
  `allclose(atol=0.00390625, rtol=0.001953125)`.

Quantile, Maximum, normierter L2, Finite-Status, Fixturehash und Pass/Fail werden
vollständig gespeichert. Toleranzen dürfen nach Compilation nicht gelockert
werden. Ein Correctnessfehler beendet den Pfad vor jeder Kandidatenzeitmessung.

## 5. Starke MLX-Baselines

Die Baselineauswahl enthält genau diese semantisch gleichen Varianten:

1. `eager_transparent`: transparente MLX-Komposition gemäß Abschnitt 2;
2. `compiled_transparent`: exakt dieselbe Funktion mit
   `mx.compile(..., shapeless=False)`;
3. `fast_rms_norm`: `mx.fast.rms_norm(x + residual, weight, 1e-6)`;
4. `compiled_fast_rms_norm`: derselbe Fast-Pfad in
   `mx.compile(..., shapeless=False)`.

Alle vier müssen zuerst das Accuracy-Gate bestehen. Drei frische
Charakterisierungsworker messen pro Variante dieselben gepaarten Fixtures. Die
Variante mit dem kleinsten hierarchischen geometrischen Median wird fest zur
Benchmarkbaseline; bei einem Unterschied von höchstens 0.5 % entscheidet die
Präzedenz `fast_rms_norm`, `compiled_fast_rms_norm`,
`compiled_transparent`, `eager_transparent`, um einen Rauschentscheid zu vermeiden.
Die nachfolgenden A/A- und A/B-Fixtures sind unabhängig von der Auswahl.

PyTorch MPS ist nicht Teil des Promotion-Gates: Phase 1B bewertet eine MLX-
Runtime-Erweiterung mit identischer MLX-Evaluation und Synchronisation. Deshalb
entsteht ausdrücklich keine Aussage, der Kandidat sei die schnellste
frameworkübergreifende Apple-GPU-Implementierung.

## 6. Messplan

### 6.1 Qualification — genau einmal

Ein frischer Worker konstruiert den Kandidaten genau einmal, löst dessen erste
Evaluation und damit die JIT-Compilation aus, speichert Compile-plus-first-eval
als diagnostische Kaltstartzeit und führt danach die vollständige Matrix aus
Abschnitt 4 aus. Diese Zeit ist kein Runtimewert und wegen möglicher MLX-/Metal-
Caches kein belastbarer Cold-Compile-Claim.

Nur `qualification=passed` erlaubt den Benchmark. Compilefehler, Timeout, Crash,
falscher Hash oder Correctnessfehler erzeugen terminal `invalid` und
`baseline_fallback`; es gibt keinen Retry.

### 6.2 Baselinecharakterisierung

Es laufen genau drei frische Worker mit Fixture-Seeds
`0xB16B1000`, `0xB16B1001`, `0xB16B1002` und Reihenfolge-Seeds
`0xB16B1100`, `0xB16B1101`, `0xB16B1102`.

Jeder Worker nutzt pro Arm `20` nicht getimte Warmups, anschließend `15`
randomisiert balancierte Blöcke mit je `50` frischen Operationen pro Arm. Pro
Operation werden ein neuer Lazy-Output erzeugt, `mx.eval(output)` und
`mx.synchronize()` innerhalb des Zeitfensters ausgeführt. Gemessen wird mit
`time.perf_counter_ns`; ein Blockwert ist `block_ns / 50`.

### 6.3 A/A-Nullkontrolle

Nach Baselineauswahl laufen genau drei neue Worker mit Fixture-Seeds
`0xB16B2000..0xB16B2002` und Reihenfolge-Seeds `0xB16B2100..0xB16B2102`.
Jeder misst zwei separat erzeugte Callables derselben ausgewählten Baseline:
`20` Warmups je Arm, `31` balancierte Blöcke, `50` Operationen je Arm.

Für `R = T_B / T_A` wird aus Session- und Block-Logratios ein hierarchisches
10.000er-Bootstrap mit Seed `0xB16B2AA0` berechnet. A/A besteht nur, wenn:

- Punktschätzer und gesamtes 95-%-KI in `[0.98,1.02]` liegen;
- das KI `1.0` enthält;
- jede Sessionratio in `[0.95,1.05]` liegt;
- alle Rohzeiten positiv und vollständig sind.

Die 2,5-%- und 97,5-%-Grenzen verwenden lineare Interpolation auf der sortierten
Bootstrapverteilung mit Position `(n-1)×p`.

Ein A/A-Fehler stoppt vor A/B und wird terminal als Messsystem-NO-GO gespeichert.

### 6.4 A/B-Bestätigung

Nur nach bestandenem A/A laufen genau drei neue Worker mit Holdout-Fixture-Seeds
`0xB16B3000..0xB16B3002` und Reihenfolge-Seeds `0xB16B3100..0xB16B3102`.
Jeder prüft Kandidat und Baseline vor dem Timing erneut gegen den Oracle und misst
danach `20` Warmups je Arm sowie `31` balancierte Blöcke mit `50` frischen
Operationen je Arm.

Für `R = T_candidate / T_baseline` wird dasselbe hierarchische 10.000er-Bootstrap
mit Seed `0xB16B3AB0` verwendet. Eine Performancepromotion erfordert gemeinsam:

- `R <= 0.95` (mindestens 5 % Punktschätzer-Gewinn);
- obere 95-%-KI-Grenze `< 1.0`;
- keine einzelne Sessionratio `> 1.05`;
- alle Qualification-, Correctness-, A/A-, Ressourcen- und Provenienzgates grün.

Sonst lautet die terminale Aktion `baseline_fallback`; `tie`, `inconclusive` und
`regression` bleiben gültige Ergebnisse. Aus genau einem Prozess oder Einzelwert
wird keine Performanceaussage abgeleitet.

## 7. Speicher- und Prozessvertrag

Jeder Worker läuft in einer neuen Prozesssession und einem frischen privaten
temporären Arbeitsverzeichnis. Der Parent verwendet einen monotonic Watchdog,
drainiert stdout/stderr begrenzt und beendet bei Fehler/Timeout zuerst die ganze
Prozessgruppe, dann begrenzt per `SIGKILL`. Feste Limits:

| Grenze | Wert |
|---|---:|
| Worker Wall Clock | `120 s` |
| Controller gesamt | `900 s` |
| `RLIMIT_CPU` | `90 s` |
| `RLIMIT_AS` / `RLIMIT_DATA` | auf diesem Darwin/Python nicht absenkbar; kein behauptetes Gate |
| `RLIMIT_CORE` | `0` |
| `RLIMIT_FSIZE` | `16 MiB` |
| `RLIMIT_NOFILE` | `64` |
| MLX Memory Guideline | `512 MiB` |
| MLX Cache | `64 MiB` |
| Parent-beobachtetes Worker-RSS-Abbruchziel | `2 GiB` |
| stdout / Result-JSON | ein kanonisches Objekt, `2 MiB` |
| stderr | `256 KiB` |

Ein separater Pre-Compilation-Prozess bestätigte, dass Darwin sowohl
`RLIMIT_AS` als auch `RLIMIT_DATA` beim Absenken von `RLIM_INFINITY` mit
`ValueError: current limit exceeds maximum limit` ablehnt. Dieser reproduzierbare
Plattformbefund wird im Ergebnis als `unsupported` gespeichert. Er wird nicht
als vorhandene Adressraumisolation ausgegeben; die ausführbaren Grenzen sind
stattdessen die festen kleinen Shapes, MLX-Guideline, Parent-RSS-Watchdog,
CPU-/Datei-/FD-Limits und der Wall-Clock-Prozessgruppenabbruch. Würde der Versuch
eine harte Adressraum- oder Unified-Memory-Garantie erfordern, ist er NO-GO.

Vor dem Spawn werden nur feste Environmentwerte weitergegeben; generische
Credential-, Token- und Proxyvariablen fehlen. Ein Python-Audit-Hook verweigert
Socket-Connect/Bind/Listen. Das ist Defense-in-Depth, keine harte native Netzwerk-
oder Dateisystemsandbox. Ebenso sind `RLIMIT_AS`, das RSS-Polling und
`mx.set_memory_limit` auf macOS keine bewiesene harte Grenze für Apple Unified
Memory oder bereits eingereichte GPU-Arbeit.

Nach den Timings wird pro Arm in kontrollierter Reihenfolge Cache geleert, Peak
zurückgesetzt und ein fester 20-Operations-Speicherprobe ausgeführt. Promotion
erfordert je Session:

- MLX-Peak des Kandidaten `<= 512 MiB`;
- beobachtetes Worker-Peak-RSS `< 2 GiB`;
- Kandidaten-MLX-Peak höchstens `16 MiB` über der Baseline.

Active-, Cache-, Peak- und Prozess-RSS bleiben getrennte Messfelder. Keine Größe
wird als exakter physischer GPU-Speicherverbrauch ausgegeben.

## 8. Provenienz, Speicherung und UI

Vor Live-Ausführung müssen Worktree und Index sauber sein. Der Controller bindet
Git-Commit, Code-/Spec-/Sourcehash, MLX/Python/macOS/Xcode, Hardwareinformationen,
alle Seeds, Fixtures, Reihenfolgen, Rohblöcke, Compile-/Wall-/CPU-Zeiten,
Correctness, Speicher und Entscheidung in kanonisches JSON. Die SQLite-Historie
ist append-only, verwendet Transaktionen, eine Hashkette und Dateimodus `0600`.

Das Dashboard öffnet nur die feste DB read-only, bindet ausschließlich Loopback,
liefert keine externen Assets und akzeptiert keine Datei-, SQL-, Source- oder
Ausführungsparameter. Es zeigt Qualification und Benchmarkhistorie einschließlich
Baselinewahl, A/A, A/B, Correctness, Speicher, Fehler und Fallback.

## 9. Abbruch- und Geltungsregeln

Sofortiger terminaler Abbruch gilt bei unsauberer oder wechselnder Provenienz,
unerwarteter Version/Hardware, Source-/Namenshashfehler, unbekanntem Manifestfeld,
Compilefehler, Crash, Timeout, unvollständigem Resultat, nicht endlichen Daten,
Correctnessfehler, A/A-Fehler oder Ressourcenüberschreitung. Keine automatische
Wiederholung und keine nachträgliche Schwellen-, Seed-, Shape-, Dtype-, Kernel-
oder Baselineänderung ist zulässig.

Selbst bei `promoted` ist nur bewiesen, dass der feste Kandidat in dieser
vorregistrierten Mehrprozessmessung auf genau diesem Fingerprint und Scope die
gewählte MLX-Baseline schlug. Runtime-Routing, Modelle, andere Zeilenzahlen,
Training/Backward, adaptive Optimierung, Cross-Device-Generalisation und ein
Vergleich mit NVIDIA AVO bleiben außerhalb der Aussage.

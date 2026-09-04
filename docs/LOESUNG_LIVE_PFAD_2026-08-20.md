# Lösung Live-Pfad: Befund und W1v3-Umsetzung (20.08.2026)

Analyse auf ausdrückliche Nutzeranfrage: „Suche den Fehler, halte die Lösung fest, damit
das funktioniert.“ Die jüngste Nutzeranweisung autorisierte die begrenzte W1v3-/Output-
Umsetzung und genau einen anschließenden `eager_baseline`-Canary. Dieses Dokument hält den
technischen Vertrag und den abgeschlossenen Run22 fest. `aa_gpu` bleibt nicht freigegeben;
Installationen und Downloads fanden nicht statt.

## 1. Befund: Der historische Canary-Fehler (Interpreterauflösung) ist BEHOBEN

Empirisch heute verifiziert (Reproduktion mit exakt der Supervisor-Umgebung
`PATH=/usr/bin:/bin`, `PYTHONPATH=<root>`, `PYTHONSAFEPATH=1`, Flags `-P -s -B`):

| Launcher an `Popen` | Ergebnis |
|---|---|
| lexikalisch `.venv/bin/python` (aktueller Code, `supervisor.py` → `argv = (executable.lexical, …)`) | `numpy OK 2.5.2`, Exit 0 |
| aufgelöst `/opt/homebrew/…/python3.12` (alter Canary-Zustand) | `ModuleNotFoundError: No module named 'numpy'`, Exit 1 |

Der früher in `PROJECT_STATUS.md` als „AWAITING USER APPROVAL" geführte Launcher-Fix
ist im Code enthalten und wirksam; Run18–21 liefen nachweislich am NumPy-Import vorbei
bis in die Warmup-Phase. Der Statuskopf ist im Rahmen der Run22-Dokumentation auf den
aktuellen Stand mit `22` Runs und abgeschlossenem Run22 gezogen worden.

## 2. Historischer Blocker: Run21 `warmup_unstable` hatte eine Design-Ursache

Run21-Zahlen (Journal): last5 Median `2 155 792 ns`, MAD `87 876 ns` (≈ 4,1 % des
Medians), Min `2 067 916 ns`, Max `2 677 583 ns` (+24 %); all-Median
`2 391 354,5 ns` > last5-Median, d. h. die Zeiten fielen bei Abbruch nach 16
Warmups noch (GPU-Clock-Ramp/Cache-Aufwärmung nicht abgeschlossen).

Der Vertrag prüft die ±5-%-Stabilität auf **Einzel-Evals von ~2,2 ms**. Dieselbe
Spezifikation verlangt für die eigentliche Messung ausdrücklich 50–200-ms-Batches,
weil Einzel-Evals dieser Größenordnung dispatch-/scheduler-dominiert streuen. Das
Warmup-Gate ist damit statistisch strenger als die Messung, die es schützen soll:
Ein einzelner OS-Scheduling-Ausreißer (+24 %) invalidiert den Lauf. Das ist kein
Codefehler relativ zur Spec — der Code implementiert die Spec exakt — sondern ein
Spec-Designfehler. „OS/Thermik/MLX unklar" bleibt als Auslöser des einzelnen
Ausreißers plausibel; die Empfindlichkeit dagegen ist aber implementiert.

### Fix W1v3: zeitgebundene Warmup-Blöcke mit Batch-Gate

Der Warmup-Block läuft bis `block_ns >= 50_000_000` ns. Er darf höchstens 4096
Auswertungen enthalten; wird die Mindestdauer bis dahin nicht erreicht, endet der Lauf
fail-closed mit `repetition_window_unreachable`. Der Gate-Wert ist exakt
`round(block_ns / evaluations)` als ganzzahliger Wert. Es gibt acht initiale Blöcke,
maximal 16 Blöcke insgesamt, und die letzten fünf Gate-Werte müssen innerhalb von ±5 %
ihres Medians liegen. Ein neuer Block wird nur bis zur Stabilitätsentscheidung oder zur
Maximalzahl angefordert.

```python
while block_ns < 50_000_000:
    measure_one_eval()
    block_ns = elapsed_since_block_start()
    if block_ns >= 50_000_000:
        break
    if evaluations >= 4096:
        raise BenchmarkError("repetition_window_unreachable")
per_eval_ns = round(block_ns / evaluations)
gate_samples.append(per_eval_ns)       # höchstens ein Wert pro Block
block_summaries.append({                # ein geschlossener Summary pro Block
    "block_index": index,
    "evaluations": evaluations,
    "block_ns": block_ns,
    "per_eval_ns": per_eval_ns,
    "median_eval_ns": median(eval_times),
    "min_eval_ns": min(eval_times),
    "max_eval_ns": max(eval_times),
})
```

`eval_times` bleibt eine interne, gebundene Arbeitsstruktur für den Summary; sie wird
nicht als ungebundene per-Eval-Diagnoseliste persistiert. Die neue
`warmup_unstable`-Diagnose erzeugt strikt Schema v2 mit den Block-Summaries und den
zugehörigen Gate-Samples. Historische Schema-v1-Diagnosen bleiben beim Readback
kompatibel.

Warum nicht einfach Block-Medianen als Gate? Ein Block-Median der einzelnen Eval-Zeiten
ist robust gegen einen einzelnen Ausreißer, kann aber einen echten batchweiten
Throughput-Abfall maskieren: Wenn sich die Gesamtdauer durch Dispatch-, Synchronisations-
oder andere Blockkosten erhöht, während viele Einzelevals nahe am alten Median bleiben,
bleibt der Block-Median unauffällig. `block_ns / evaluations` erfasst dagegen die gesamte
äußere Batchdauer pro Auswertung und ist damit die korrekte Gate-Statistik für den
produktiven Batchpfad. Die Einzel-Eval-Median-, Min- und Max-Werte bleiben als geschlossene
Diagnose erhalten.

Die alte 64er-Obergrenze ist damit ersetzt: W1v3 verwendet 4096 Auswertungen und bricht
bei Nichterreichen der 50-ms-Mindestdauer mit `repetition_window_unreachable` ab. Das
vermeidet sowohl eine shapespezifische Eval-Annahme als auch ein stilles Weiterlaufen.

Eine W2-Variante (nur `H0_MAX_WARMUPS` auf 64) wird nicht verfolgt. Sie würde das
eigentliche Gate-Designproblem nicht beheben und ist durch die jüngste W1v3-Anweisung
überholt.

## 3. Zusatzbefund Messqualität: totes Output-Retain in `_batch`

`_measure_once` gibt `_Timed(…, output)` mit dem 2048²-FP16-Device-Array (8 MiB)
zurück; **kein Aufrufer liest `.output`**. `_batch` hält dadurch während des
laufenden Zeitfensters bis zu `repetitions × 8 MiB` Device-Arrays gleichzeitig
lebendig (bei 64 Wiederholungen 512 MiB) — direkt unter dem gesetzten
`set_memory_limit(1 GiB)`. Allokationsdruck im Messfenster ist eine unnötige
Rauschquelle und Speicher-Gate-Nähe ohne Messwert-Nutzen.

Umgesetzt: Das `output`-Feld aus `_Timed` ist entfernt; Correctness nutzt den separaten
Pfad `_run_backend_matmul` und ist nicht betroffen. Die Retention-Nachprobe erzeugte
`67108864 B`, hielt aber `0` Payloads/`0 B` live; `_Timed` enthält nur die drei
Zeitfelder.

```python
@dataclass(frozen=True)
class _Timed:
    duration_ns: int
    evaluation_ns: int
    synchronize_ns: int
# _measure_once/_measure_existing_output: return _Timed(duration_ns, evaluation_ns, synchronize_ns)
```

## 4. Empfohlene Reihenfolge

1. W1v3 und den `_Timed.output`-Fix sind umgesetzt und offline geprüft; anschließend
   wurden Code-/Spec-/Environment-Hashes neu eingefroren.
2. Genau ein `eager_baseline`-Canary wurde unter diesem Freeze ausgeführt. Es gab keinen
   Retry und kein `aa_gpu`.
3. Eine neue ausdrückliche Nutzerfreigabe wäre für `aa_gpu` 3+3 oder weitere Runs nötig.

Status: **Run22 abgeschlossen; H0-Baseline-Reference vorhanden**. Das Ergebnis ist keine
vergleichende Performanceaussage und beweist weder A/A noch Optimierung oder
Self-Optimization.

## 5. Verifikationsprotokoll und Run22-Abschluss (20.08.2026)

- Vorher-Baseline vor der W1v3-/Output-Umsetzung: `206 passed, 47 subtests`, DB `21`
  Runs; Retention `64` Payloads bzw. `67108864 B` lebend. Vorherige Hashes: Code
  `aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c`, Spec
  `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.
- CLI-Lock intakt: `mlx-run --mode eager_baseline --process-set characterization
  --process-index 0` ohne `--execute` → `state=not_released`, Exit `78`.
- Launcher-Repro (Supervisor-Umgebung `PATH=/usr/bin:/bin`, `PYTHONPATH=<root>`,
  `PYTHONSAFEPATH=1`, Flags `-P -s -B`): lexikalisch `.venv/bin/python` →
  `numpy OK 2.5.2`, Exit 0; aufgelöst `/opt/homebrew/…/python3.12` →
  `ModuleNotFoundError: No module named 'numpy'`, Exit 1.
- Der engere Unittest-Lauf wurde nach `30.018 s` mit Exit `124` absichtlich gestoppt,
  nachdem `103` Marker grün und keine Fehler/Fehlschläge sichtbar waren; er wird nicht
  als Fehler verschwiegen. Der anschließende vollständige einmalige Pytest-Lauf endete
  mit Exit `0`, Wall `66.837 s`: `228 passed`, `2211` Subtests in `66.24 s`.
- `.venv/bin/python` → Symlink auf `/opt/homebrew/opt/python@3.12/bin/python3.12`,
  `pyvenv.cfg` mit `include-system-site-packages = false`; erklärt, warum nur der
  lexikalische Launcher die venv-Paketsuche behält.
- CLI-Lock: ohne `--execute` Exit `78` mit `state=not_released`; fehlende Pflichtargumente
  Exit `64`. `xcodebuild -checkFirstLaunchStatus` Exit `0`. ProjectAtlas `0.4.5-rc1`;
  Umgebung Python `3.12.13`, NumPy `2.5.2`, MLX `0.32.0`, macOS `26.5.2 arm64`.
  Der sandboxed Import meldete kein Metal-Gerät; ohne MLX-Operation.
- W1v3-/Output-Umsetzung in Benchmark, Worker, Runner, Aggregation und den zugehörigen
  Tests: äußere Warmup-Blöcke mindestens `50 ms`, maximal `4096` Evals
  fail-closed mit `repetition_window_unreachable`, Gate `round(block_ns/evals)`,
  `8..16` Blöcke, ±`5 %` für die letzten fünf Gate-Werte, bounded geschlossene
  Block-Summaries, Failure-Schema-v2 mit v1-Readback; `_Timed.output` entfernt.
- Genau ein Live-Befehl wurde ausgeführt: Run22
  `h0-eager_baseline-characterization-0-14d435dcc2170feec70d8baaa712860e59a6148ca3f211aad98eff1c9d7cf0ff`,
  äußerer `real=3.79 s`, Exit `10`, DB `22`. Der Wrapper
  `completed/measurement_complete/baseline_fallback` mit `error=null` ist gemäß
  Worker-/Runner-Vertrag neutral; verschachtelt gelten
  `baseline_reference/not_run/aggregation_required=false` als erfolgreiche eager-
  baseline-reference. Die anfängliche Operator-Deutung von `baseline_fallback` als Fail
  wurde korrigiert; kein Produktfehler.
- Run22: `8` stabile Warmups; Gate-Werte
  `[2566556,2179783,2188775,2143891,2155069,2174895,2195533,2192185]`, Median der
  letzten fünf `2174895 ns`; `30` Blöcke mit Reps `32`; Calibration `68155792 ns`;
  Baseline-Median `2138574.859375 ns`, MAD `17041.671875 ns`, IQR `35343.0859375 ns`,
  Min `2105915.34375 ns`, Max `2210087.25 ns`.
- Correctness-Gate bestanden: `9/9` Cases, `86/86` Metrics; `abs_max=0.0310508173`,
  `normalized_l2=0.0002074681`, `abs_q99=0.0110023008`,
  `rel_q99_abs_oracle_ge_1=0.0004333980`.
- Speicher: active `16777216 B`, peak `25165824 B`, cache `8422698 B`, RSS-Peak
  `369655808 B`; Memory-Gate `not_evaluable_missing_required_metric`, `hard_limit=false`.
  Retention nach Fix: created `67108864 B`, retained `0` Payloads/`0 B`; `_Timed` nur
  `duration_ns`, `evaluation_ns`, `synchronize_ns`.
- Freeze: Code `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`,
  Spec `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; kein Git-Root.
  Manifest `73058165244fe505035182f0044dc5ab8bd16ef523ebfc44b44d5b6f616e239e`, Result
  `bda3d23d56e49c2d26bf7c3e73d52b61c3ea022c3fb61ab0719bfedef58a6d09`, Evidence
  `edaf6cae5a98185f183fd368189a8be3a56c194540e4f64300903cff42d1a6a0`, Projection
  `a51aa4b3cadf00dc5338eee199206b2b8f876c4fd3aeaaa2d5261364254ed790`, Bundle
  `a566c912032efab919dddf5ca7f67b986f29464a655abf15617733aeb6947c49`.
- Dashboard-Snapshot war socketfrei: `snapshot_id=325afcc9a45311ba716f64a51e7395cd7f2cf1c872c9a3f349c6daf9361398de`,
  `source_revision=7cdad7edcb6099894d588bb9927de322bd4f7ce02d256673768647db54131c73`,
  `run_count=22`, `completed`.

## 6. Machbarkeitseinschätzung des Gesamtprojekts

Frage: Ist „Hardware-Aware Self-Optimizing AI Runtime" auf M1 Max umsetzbar?
Antwort: **Ja, als Forschungsprototyp klar umsetzbar** — mit folgender
Risikostaffelung:

| Phase | Machbarkeit | Einschätzung |
|---|---|---|
| H0 (Messsystem + A/A) | Hoch | Reine Benchmark-Methodik. Infrastruktur, W1v3 und Output-Fix sind umgesetzt; Run22 liefert eine gültige eager-baseline-reference. Das ist keine A/A-, Optimierungs- oder Self-Optimization-Entscheidung. |
| H1 (template-constrained tuning) | Machbar, Gewinne klein | Präzedenz: AutoTVM, Halide-Autoscheduler, TorchInductor-Autotuning. Aber: MLX-Matmul 2048² FP16 ist nahe Roofline — auf genau dieser Workload ~0 % Headroom zu erwarten. Tuning-Fläche ohne Custom-Metal schmal (compile vs. eager, Layouts, Streams). |
| Phase 1B+ (Custom-Metal-Kernel) | Hart, echter Aufwandsschwerpunkt | MLX-GEMM zu schlagen: marginal (0–15 % auf Spezial-Shapes). Reales Headroom liegt bei quantisierten Matmuls, Attention-Varianten, Fusion, kleinen/schiefen Shapes. |
| H2 (Modelle) | Baubar, Forschungswette | Ob Self-Optimization handgetunte Stacks schlägt, ist offen; die Infrastruktur dafür ist baubar (MLX-LM vorhanden). |

Zwei Kernwarnungen:

1. **Workload-Wahl:** Die kanonische 2048²-GEMM wird als H1-Ziel vermutlich
   überall „tie" liefern. Für belegbare Gewinne braucht H1 Ops mit Headroom
   (quantisierte Matmul, Attention, Fusion, kleine Shapes) — bei der H1-Planung
   berücksichtigen, nicht erst bei H2.
2. **Prozesskosten:** Preregistrierung, Hash-Freezes und fail-closed-Disziplin
   kosten Tempo, sind aber der Grund, warum Ergebnisse belastbar sein werden.
   Realistisches Endergebnis: vertrauenswürdiges Messsystem plus moderate, sauber
   belegte Gewinne auf ausgewählten Ops — kein genereller
   „self-optimizing runtime"-Durchbruch.

## 7. Restpunkte

1. H0-Baseline ist ausführbar und durch Run22 referenziert; daraus folgt keine
   vergleichende Performanceaussage.
2. A/A, Optimierung und Self-Optimization sind nicht bewiesen. `aa_gpu` und weitere
   Runs bleiben bis zu einer neuen ausdrücklichen Freigabe gesperrt.
3. Die eingefrorene Spec wurde nicht geändert; ihr Hash ist im Run22 eingefroren.

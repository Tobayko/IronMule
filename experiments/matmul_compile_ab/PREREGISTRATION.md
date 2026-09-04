# Zyklus 16 – `fixed_cache_compiled_decode_v1`

## Versiegelung und Geltungsbereich

Diese Mini-Vorregistrierung gehört zur Studie
`matmul-compile-ab-20260824-01` (Messlauf
`matmul-compile-validation-20260824-01`). Der Kandidat ist
`fixed_cache_compiled_decode_v1`. Der Anspruch bleibt immer
`formal_claim=false`.

Gemessen wird nur eine lokale, hardwaregebundene Runtime-Änderung im bereits
vorhandenen MLX-Pfad des Modells `mlx-community/gemma-3-4b-it-4bit`, Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`. Gewichte, Quantisierung,
Tokenizer und Modellparameter werden nicht geändert. Matmul bleibt in allen
drei Armen aktiv. Das umgangssprachliche „Matmul an/aus“ bedeutet hier
„Runtime-Kompilierung und fester KV-Cache an/aus“; es gibt keinen Arm, der
Matmul entfernt oder durch eine andere Mathematik ersetzt.

Die alte, ungültige −7,4-%-/−23,8-%-Notiz aus früheren Experimenten ist keine
Evidenz dieser Studie und darf weder als Baseline noch als Erfolg verwendet
werden.

## Hypothese

Ein fester, bereits gepufferter KV-Cache hält die Formen während des Decodes
konstant. `mx.compile` kann dadurch wiederverwendbare MLX-Kernels erzeugen.
`fixed_compiled` soll die 31 Decode-Forwards schneller ausführen als sowohl
`standard_eager` als auch `fixed_eager`, während alle Token byte- und
reihenfolgenidentisch bleiben. `fixed_eager` trennt den Effekt des festen
Caches vom Effekt der Kompilierung.

Der primäre Erfolg ist nur dann erreicht, wenn beide gepaarten
Laufzeitvergleiche zugunsten des Kandidaten ausfallen. Die Ausgabequalität ist
kein Ersatz für Tokenidentität: Ein einziger Token- oder Text-Mismatch ist ein
terminaler Korrektheitsfehler.

## Unveränderlicher Workload

- Apple M1 Max, 32 GB Unified Memory, Netzbetrieb/AC.
- Modell: exakt die lokale Snapshot-Revision oben; keine Netzwerkzugriffe.
- Der Prompt ist bytegleich mit Zyklus 14/15, ohne zusätzliche Zeile oder
  Formatierung. Raw-Prompt-SHA-256:
  `c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b`.
- Chat-Template, gerenderte Promptbytes und Prompt-Tokenbytes werden je Lauf
  gespeichert und zwischen Armen geprüft. Jeder abgeschlossene Arm speichert
  zusätzlich den Raw-Prompt-Hash, den Hash der Prompt-Token-IDs und den Hash
  der gerenderten Promptbytes; die vollständigen Tokenbytes bleiben nur im
  Blockereignis.
- Greedy, `temperature=0`, keine Sampling-Varianz.
- Genau 32 Ausgabetoken: der erste Token wird aus dem unveränderten Prefill
  gewählt, danach exakt 31 Decode-Forwards. EOS beendet den Messworkload
  nicht vorzeitig.
- Erwartete Promptlänge: 322 Token; Gate `prompt_tokens + 31 <= 512`.
- Fixed-Cache-Kapazität: 512 Positionen. Der Standardcache wird nach dem
  Prefill ohne Inhaltsänderung auf diese Kapazität gepaddet. Ungültige Plätze
  werden über eine feste tensorbasierte globale/sliding Bool-Maske
  ausgeschlossen.

## Arme und Ablauf

Jeder der sechs Blöcke ist ein frischer Python-Prozess mit genau einem
Modell-Ladevorgang. Das Modell wird nie parallel mit einem anderen Prozess
geladen. Jeder Arm hat einen unabhängigen Cachezustand und erhält acht
Warmup-Decode-Schritte; anschließend wird der Zustand zurückgesetzt und eine
volle 32-Token-Messgenerierung ausgeführt.

Jeder abgeschlossene Arm muss `finish_reason=fixed_steps`,
`decode_forwards=31`, `warmup_forwards=8` sowie genau acht Warmup-Zeitwerte
tragen. `compile_wrapper_ns` und `compile_cold_ns` sind für
`fixed_compiled` Zahlen; für die beiden Eager-Arme ist der Wrapper null und
die Kaltzeit nullwertig. Fehlende oder unerwartete Längen/Feldtypen machen den
Arm ungültig.

- `standard_eager`: bestehende MLX-Standardcaches, keine Kompilierung;
  Referenzarm.
- `fixed_eager`: derselbe neue Fixed-Cache wie beim Kandidaten, aber ohne
  `mx.compile`; Attributionarm.
- `fixed_compiled`: derselbe Fixed-Cache und dieselbe Mathematik mit
  `mx.compile(body, shapeless=False)`. `body(input_ids, state)` erhält den
  vollständigen Cachezustand als expliziten Argumentbaum und gibt
  `(logits, new_state)` als expliziten Rückgabebaum zurück; es gibt keinerlei
  `inputs=`-/`outputs=`-Capture. Einziger Kandidat.

Die vorab festgelegte Reihenfolge sind alle sechs Permutationen, je einmal:

1. `standard_eager → fixed_eager → fixed_compiled`
2. `standard_eager → fixed_compiled → fixed_eager`
3. `fixed_eager → standard_eager → fixed_compiled`
4. `fixed_eager → fixed_compiled → standard_eager`
5. `fixed_compiled → standard_eager → fixed_eager`
6. `fixed_compiled → fixed_eager → standard_eager`

Es gibt keinen Retry. Ein fehlender Snapshot, falsche Hardware, API-/Compile-
Fehler, Timeout, Speicher-/Swap-Grenzwert oder Budgetfehler stoppt sicher und
bewahrt Teilergebnisse.

## Technische Invarianten des Fixed-Cache

Der Cache besteht aus dict/list-gebundenen MLX-Arrays: feste `keys` und
`values` mit Position 512 sowie ein tensorförmiger globaler Offset. Decode-
Updates verwenden `mx.slice_update` mit dynamischem Offset und geben immer
dieselbe Form zurück. Der Fixed-Pfad benutzt keine dynamische
Zusammenfügung, keine wachsenden Arrays und keine Kontextkürzung. Die
Standardcaches werden nur beim einmaligen Konvertieren nach dem Prefill
gelesen. Kann die lokale MLX-/mlx-lm-API den gebundenen State nicht korrekt
kompilieren, ist der Kandidat `candidate_not_runnable`; ein Scheinvergleich
ist verboten.

Der Offset wird ausschließlich im äußeren vollständigen Forward genau einmal
erhöht. Jede Cache-Schicht schreibt nur an den übergebenen Offset. Der
Geltungsbereich bleibt eng: Fixed-Cache-Länge 512, Prompt 322 Token und 31
Decode-Forwards; es wird kein allgemeiner Matmul-Kernel entfernt oder ersetzt.

## Primärmetriken und Statistik

Pro Arm werden gespeichert: Prefill, TTFT, Cachekonversion, Compile-Wrapper,
erster synchronisierter Compile-Aufruf, Warmupzeiten, 31 einzelne
Decode-Forward-Zeiten, Inter-Token-p50/p95/p99, Decode-Gesamtzeit,
Modellarbeit, Prozess-Wallzeit, Tokenrate, Peak-RSS, MLX-Peak, Swap sowie
Token-/Text-SHA-256.

Die Evidenz trennt pro Arm `observed_model_work_ns` (gestoppene beobachtete
Armzeit), `charged_model_work_ns` (vom BudgetGuard akzeptierte Zeit),
`guard_recorded_model_work_ns` (tatsächlicher Zuwachs in der Guard-Buchung,
auch wenn ein Gate danach ablehnt), `charge_accepted` sowie die Guard-Werte
vor/nach `record_gpu`. `duty_formula_break_seconds` und
`required_break_blocks` sind die aus der beobachteten Armzeit berechnete
theoretische Pflichtpause; sie sind keine Behauptung, dass eine Pause bereits
ausgeführt wurde. Eine Pause wird nur nach akzeptierter Charge ausgeführt.
Beide Summen werden zusätzlich auf Blockebene
gespeichert. Vollständige Blöcke dürfen nur akzeptierte Arme enthalten. Wird
ein Arm vom BudgetGuard abgelehnt, bleiben beobachtete Zeit und Ressourcen-
Evidenz in einem terminalen `resource_or_budget_failed`-Teilereignis erhalten,
ohne eine falsche Charge zu behaupten.

Für jeden Arm werden zusätzlich TTFT, Prefill, Arm-Modellzeit, Arm-Walltime,
Prozess-Walltime, Tokenrate sowie RSS-, MLX- und Swap-Werte mit Werten, Median
und MAD gespeichert. Aus den sechs Blöcken werden Median und MAD gebildet.
Die einzige
entscheidungsrelevante Primärmetrik ist die gemessene
`decode_total`-Gesamtzeit. Primäre gepaarte Ratios sind
`fixed_compiled / standard_eager` und `fixed_compiled / fixed_eager` für diese
Gesamtzeit. Das Kandidatengate lautet je Vergleich: Median ≤ 0,95 und obere
Grenze des gepaarten 95-%-Bootstrap-KI < 1,0. Inter-Token-p50/p95/p99 werden
weiterhin gemessen und berichtet, sind aber ausdrücklich report-only und
ändern die Entscheidung nicht. Der Bootstrap ist gepaart, Perzentil-95-%-KI,
Seed `20260824`, 10.000 Resamples. Es werden keine Ausreißer entfernt.
MDE-Orientierung: 5 %; sie ändert die Entscheidungstabelle nicht.

Zusätzlich werden rein berechnete Werte getrennt ausgewiesen. Die warme
End-to-End-Projektion lautet
`(fixed prefill + cache conversion + 31 decode) / (standard prefill + 31
decode)`. Die konservative kalte Projektion addiert zum Fixed-Zähler
`compile_wrapper_ns + compile_cold_ns`; `compile_cold_ns` bleibt separat
gemessen. Der Break-even teilt dieses tatsächliche kalte Setup durch die
gemessene Einsparung pro Decode-Forward gegenüber `standard_eager`. Alle drei
Werte sind berechnete Obergrenzen/Projektionen, keine zusätzlichen Messungen;
der Wrapper allein wird nicht als kalter Aufwand verwendet.

## Vorab festgelegte Entscheidungstabelle

| Priorität | Bedingung | Entscheidung |
|---|---|---|
| 1 | Ressourcen-, Swap-, Budget-, Hardware-, Snapshot- oder Integritätsgate fehlgeschlagen | `resource_or_budget_failed` |
| 2 | Token/Text/Promptidentität oder modellinterne Deterministik fehlgeschlagen | `correctness_failed` |
| 3 | Fixed-Cache-/Compile-API nicht korrekt ausführbar | `candidate_not_runnable` |
| 4 | Beide `decode_total`-Ratio-Mediane ≤ 0,95 und beide oberen 95-%-KI-Grenzen < 1,0, alle Gates bestanden | `runtime_compile_wins_exact_scope` |
| 5 | Nur Vergleich gegen `fixed_eager` gewinnt, anhand `decode_total` | `compile_gain_no_system_gain` |
| 6 | Nur Vergleich gegen `standard_eager` gewinnt, anhand `decode_total` | `fixed_cache_gain_not_compile_gain` |
| 7 | Korrekt, aber `decode_total`-Ratios unklar | `no_clear_speedup_baseline_retained` |
| 8 | Klare `decode_total`-Regression nur wenn mindestens eine untere gepaarte Bootstrap-95-%-KI-Grenze > 1,0 ist | `compile_regression_baseline_retained` |

Liegt die untere Bootstrap-Grenze nicht über 1,0, bleibt das Ergebnis trotz
eines Medianwerts über 1,0 unklar und fällt unter Zeile 7. Diese Regel ist vor
der Messung festgelegt und wird danach nicht verändert.

Keine Entscheidung erlaubt Kandidatenausführung, Produktaktivierung,
Gewichtsänderung oder die Aussage eines allgemeinen Gemma-Speedups.

## Ressourcen und Protokoll

- AC-Pflicht; Offline-Umgebung: `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `PYTHONNOUSERSITE=1`.
- Im Worker wird erst nach Autorisierung AC geprüft und ein eigener
  `BudgetGuard` mit `duty_cycle_limit=0.15`, `continuous_gpu_limit_s=6`,
  `gpu_work_limit_s=120`, `wall_limit_s=1200`, `required_break_s=4` und
  `candidate_cooldown_s=0` aktiviert. Pro Arm umfasst die gestoppte
  Wall-clock-Obergrenze beide Prefills, Compile-Kaltstart, Warmup und die
  Messdekodierung. Die Uhr stoppt vor `record_gpu`; danach werden RSS/MLX/Swap
  sofort geprüft und anschließend mindestens `13` 4-s-`required_break()`-Blöcke
  (52 s) nach jedem akzeptierten Arm ausgeführt; zusätzlich gilt die Formel
  `ceil(seconds * (1 - 0.15) / (0.15 * 4))`. Damit liegen mindestens 51 s
  zwischen Armenden auch über frische Worker hinweg. Pausen gehören
  weder zur Armzeit noch zur Worker-Modellarbeit. `BudgetError`, insbesondere
  >6 s zusammenhängende Arbeit, ergibt terminal `resource_or_budget_failed`.
  `observed_model_work_ns` ist die Summe der gestoppten Armzeiten; die
  Budgetsumme ist ausschließlich die tatsächlich akzeptierte Charge;
  das 6-s-Limit gilt je Arm und als maximales Kind-Kontinuum, nicht als Summe
  der drei Arme. Der Parent-BudgetGuard erfasst keine GPU-Arbeit und schläft
  nicht: Er prüft ausschließlich die Gesamt-Walltime. Die Parent-Auswertung
  aggregiert die drei Kind-BudgetGuard-Summaries numerisch (Gesamtarbeit ≤120 s,
  maximales Kind-Kontinuum ≤6 s, Duty-Faktor je Kind exakt 0,15) und verwirft
  jede doppelte Abrechnung oder Pause.
- Peak-RSS höchstens 6 GiB, MLX-Peak höchstens 5 GiB, Swap-Delta 0.
- Worker-Timeout höchstens 300 s und höchstens bis zur noch verbleibenden
  registrierten Gesamt-Walltime abzüglich einer vorab reservierten
  Finalisierungsreserve von 15 s. Eine gemeinsame monotone Deadline gilt für
  Worker-Wait, Prozessgruppenabbruch, Outputreader und Join; nach ihrem Ablauf
  folgt sofort SIGKILL/fail-closed, ohne zusätzliche feste Wartezeiten.
  Live-stdout-Limit,
  genau ein striktes JSON-Ereignis und bereinigte Umgebung.
- Ein unerwartetes minimales Worker-`event:error` wird als bereinigtes
  Terminalereignis im Fail-safe-Ergebnis erhalten, zählt aber nicht als
  abgeschlossener Hardwareblock.
- Privater Startmarker: Verzeichnis Modus 0700, Datei 0600. Existierender
  Marker oder `results.json` verhindert jeden erneuten Hardwarelauf.
- Der lokale Snapshot-Resolver bindet Modell-ID, Revision, Snapshot- und
  Gewichtshashes. Ergebnis, Marker, Git-Revision/Dirty-State,
  Code-/Spezifikations-/Prompt-/Umgebungsfingerprints, Hardware, AC, PID,
  Load-Zähler, Budget, Ressourcen, Fehler und Abbruchgrund werden gespeichert.

## Grenzen

Diese Studie misst genau einen lokalen 4B-Decode-Workload und eine
Runtime-Cache-/Compile-Variante. Sie beweist weder allgemeine Textqualität,
allgemeine Modellleistung, Multi-Turn-Verhalten, parallele Requests,
selbstlernende Optimierung, andere Modelle, andere Promptgrößen noch eine
allgemeine „Matmul aus“-Eigenschaft. Das ursprüngliche Matmul bleibt in allen
Armen aktiv; die Begriffe `an/aus` sind daher nur eine verständliche Kurzform
für Compile-/Cache-Umgebung.

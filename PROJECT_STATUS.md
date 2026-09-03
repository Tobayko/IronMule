# Projektstatus

**Stand:** 1. September 2026 (Kurzfassung; vollständige Historie im Arbeitsjournal)
**Zielgerät:** Apple M1 Max, 32 GB Unified Memory, 10-Core CPU, 32-Core GPU

Diese Datei ist der kompakte Einstieg: Gate-/Entscheidstabellen und Verweise.
Alle Rohwerte, Hashes, Preflights und Audits stehen unverändert in
[`docs/ARBEITSJOURNAL.md`](docs/ARBEITSJOURNAL.md).

## Auditierter aktueller Stand

| Bereich | Verifizierbarer Stand | Zulässige Aussage |
| --- | --- | --- |
| Root-Provenienz | Root-Git-Repository; Zyklus 15 auf Seal-Commit `23cff19` (Ergebnis-/Dokumentationsabschluss folgt getrennt), Zyklus 14 auf Vor-Hardware-Commit `8067dc6` und separatem Ergebnis-/Dokumentations-Commit `8923467`, Zyklus 13 auf `f954617`, Zyklus 11 auf `1dff9a5`, formale Head-Skip-Studie auf `9466bb9`, Phase 1B auf `ea8f959`, N8/N10-Shadow-Router auf `70bc451`, N10-Runtime auf `5eaad38`, N10-v2 auf `959df09`, N10-v1 auf `c3e582c`, formaler H1-v2-Code auf `1fbe73c`; `ProjectAtlas/` als gepinntes, unverändertes Gitlink | formale und native Läufe sind an konkrete Root-Revisionen gebunden |
| H0 | `.friday-data/h0.sqlite3` mit `28` Runs, darunter `9` `aa_gpu`-Runs | H0-Rohhistorie vorhanden; **kein** formal geschlossenes A/A-Gate |
| H0.1 | `3` Legacy-Beobachtungen, `6` Paced-Sessions, `1` Study mit `h01_complete_unresolved` | replizierte Stationarität nicht unterstützt; gültiger Negativbefund |
| H1/H2 historisch | zehn rekonstruierbare Zusammenfassungen, keine Rohblöcke und keine vollständige historische Provenienz | ausschließlich `legacy_summary`; formale H1/H2-Claims `false` |
| H1/H2 künftig | SQLite-v1-Evidenz, saubere Git-/Code-/Spec-/Environment-Bindung, gemeinsame Budgets und read-only Historien-UI implementiert; vier native Ereignisse vorhanden | prospektive Exploration möglich; formale Claims bleiben in v1 ausdrücklich `false` |
| H1-v2 formal | terminale 16-Record-Historie: versiegelte Präregistrierung, sechs bestandene A/A-Sessions, MDE `5 %`, sechs frische A/B-Sessions und Split-Entscheid `h1_gain_confirmed` | für genau ein Gerät, FP16-`2048²`, acht Matmuls und den Batch-Dispatch-Plan ist der Gain jenseits der MDE formal bestätigt; kein Modell-/Cross-Device-Claim |
| Begrenzte Runtime | exakte H1-Bindung, tensorbasierte Scope-Prüfung, serieller Fallback, Circuit Breaker, Hash-Ketten-Historie und read-only UI; CPU- und MLX/GPU-Gates auf sauberem Commit bestanden | Batch ist nur für den exakt registrierten Workload freigegeben; Policy-/Runtime-Befund ist Engineering-Validierung, kein neuer formaler oder Modell-Claim |
| N10-Runtime / Runtime-lite | getrenntes Paket `friday_runtime_n10/` mit exaktem 16-Record-/DB-/Snapshot-Claim, N=10-Allowlist, Cold-Load-Gate, Circuit Breaker und eigener DB/UI; CPU- und MLX/GPU-Gates auf sauberem Commit bestanden | **Engineering-GO nur im exakten N10-Scope**; jede Evidenz-, Code-, Spec-, Umgebungs-, Hardware- oder Workload-Abweichung fällt seriell zurück; N8 bleibt unverändert |
| N8/N10-Shadow-Router | getrenntes Paket `friday_avo_router/`; beide versiegelten Policies müssen autorisieren, reale Tensor-Metadaten bestimmen die Empfehlung, erzwungener Plan bleibt `serial_shadow_only`; CPU-, MLX-Metadaten-, History-, UI- und Security-Gates auf sauberem Commit bestanden | **Shadow-GO** nur für Beobachtung und als Gate zur getrennten Ein-Kernel-Vorregistrierung; keine optimierte Ausführung, keine produktive Integration und kein neuer formaler Claim |
| Phase 1B Residual+RMSNorm | statischer, quellhashgebundener Custom-Metal-Kandidat auf Commit `ea8f959`; Security-Diff, 566er Vollsuite, Qualification, A/A, A/B, Persistenz und UI abgeschlossen | Correctness und Messsystem bestanden; nur `1,870 %` Gain bei vorregistrierten `5 %` Mindestgewinn: **keine Promotion**, `baseline_fallback`, gültiger Negativbefund |
| H2 Gemma-Minimallauf | eine offline erzwungene Gemma-4B-Runde schlug `N=3,10,16` vor; Harness bestätigte explorativ `N=10` mit frischen drei Replikaten | nützliche Modellselektion beobachtet, aber Schema v1 bleibt `formal_claim=false`; keine Runtime-Erweiterung und keine zweite Runde |
| Prefill-Head-Skip formal | terminale 16-Record-Historie: versiegelte Präregistrierung, sechs bestandene A/A-Sessions, eingefrorene MDE `5 %`, sechs frische A/B-Sessions, `R=0,846385`, Gesamt-KI `[0,843147; 0,851284]`, `12/12` Tokenidentitätsgates | Gain ist nur für ein Gerät, einen gebundenen Gemma-4B-Snapshot, einen 897-Token-Prompt, Chunk `256`, Batch `1` und greedy ohne Prompt-Logprobs formal bestätigt; Integration bleibt freigabepflichtig |
| Persistenter Modellprozess | prospektiver Zyklus 13 mit zwei bestandenen A/A-Paaren, drei Charakterisierungs- und drei Validierungspaaren; Gesamtmedian `R=0,346968`, `6/6` exakte Tokenpaare, kein RSS-/Swap-Wachstum | **Engineering-Gain nur im gemessenen Scope**, gerechnet `−65,3032 %`, `formal_claim=false`; normaler Dienstpfad und automatische Aktivierung bleiben freigabepflichtig |
| Gemma-4B-Planer | prospektiver Zyklus 14 mit drei frischen Prozessen; `3/3` exakt gleiche greedy Antworten und richtige ID im Rohtext, aber `0/3` Antworten ohne Markdown-Rahmen | **`planner_contract_failed`**; keine Selbstoptimierung, kein Geschwindigkeitsgewinn und keine Aktivierung, `formal_claim=false` |
| Zwei-Modell-Planner | prospektiver Zyklus 15 mit sechs balancierten Paaren und zwölf frischen, seriellen Prozessen; 1B und 4B jeweils deterministisch `6/6`, aber strict contract/parser/candidate `0/6` | **`no_planner_qualified`**; 1B: Markdown, falscher Schlüssel `persistent_service_id`, End-of-turn-Trailer; 4B: Markdown-Codeblock trotz richtiger ID; `formal_claim=false` |
| N10-v1 / N10-v2 formal | V1 auf `c3e582c` vor Timing terminal; V2 auf `959df09` mit registrierter Fixture-Identität versiegelt und mit 6 A/A- plus 6 A/B-Sessions terminal abgeschlossen | `N=10`-Batch-Dispatch ist für genau ein Gerät, FP16-`2048²`, zehn Matmuls und den festen Plan jenseits 5 % bestätigt; nur ein begrenzter N10-Runtime-Prototyp ist freigegeben |

## Spekulation im Auslieferungspfad — entschieden am 2026-09-02

| Bereich | Ergebnis | Zulässige Aussage |
| --- | --- | --- |
| H1.0 Umschaltpunkt | 4B, `897`-Token-Prompt, Längen `32`–`128`, Breiten `1`–`3`, gepaart gegen den Auslieferungspfad: Verlust in **jeder** Zelle, monoton wachsend mit Breite und Länge (`−12,23 %` bis `−75,35 %`) | es gibt keinen Umschaltpunkt unterhalb von `128` generierten Token; **der `(K,N)`-Dispatcher wird nicht gebaut**, `speculate_k` bleibt im Auslieferungspfad `0` |
| H1.0 Amendment | gepatchter Spekulationspfad ohne die redundante Barriere: `−26,16 %` gegen `−26,46 %` (Breite `1`), `−51,29 %` gegen `−51,69 %` (Breite `2`) | der Rückstand ist **kein Defekt** des Spekulationspfades: das Entfernen der Barriere bewegt `0,3`–`0,4` Punkte |
| Ursache, gemessen (S3) | `Engine.generate()['acceptance']` auf der versiegelten Workload: **`0,0` bei `k = 1, 2, 3`**; Decode `2,398` / `3,273` / `4,054 s` gegen rund `1,9 s` der Baseline | die tragende Ursache ist **Zusatzarbeit ohne jeden Gegenwert**: der `3`-Gramm-Lookup findet in diesem Prompt nichts, jede Iteration rechnet `k+1` Positionen und liefert ein Token. Die Kosten wachsen **linear in `k`** (rund `0,8 s` je Entwurfsplatz), **nicht** proportional zur Breite — Breite `4` kostet das `2,13`-fache, nicht das Vierfache, weil Decode bandbreitengebunden ist. Der Hebel des Verfahrens ist damit intakt; die Workload liefert ihm nichts. Korroboration über zwei Wege: `2,398/1,9 = 1,262` gegen den gepaarten `ratio_median 1,2616` bei `96`/w1 (`1,9` gerundet). Die Readback-Asymmetrie ist der kleinere Rest |
| D4 Serving-Latte | Nutzerentscheidung vom 2026-09-02, übermittelt über die Mentor-Session, wörtlich „bundled_readback bleibt drin, D4 so entscheiden" | entschieden ist **genau ein Knopf**: `bundled_readback` bleibt im Auslieferungspfad, unter der schwächeren Serving-Latte (Intervall vollständig unter `1,0` plus Tokenidentität) und ohne jede Studienpromotion zu ändern. Ob diese Latte **allgemein** für künftige Serving-Knöpfe gilt, ist damit **nicht** entschieden und bleibt offene Nutzerfrage |
| H1.0 zweites Modell | `gemma-3-1b-it-4bit`, `897`-Token-Prompt, `32` Token, Breite `1`, `24` Paare gegen den Auslieferungspfad: `−15,59 %`, KI `[1,1381; 1,2353]`, A/A `11,77 %` | bestätigte Verschlechterung auch auf dem 1B, und zwar an der schärfsten Stelle (kürzeste Länge, schmalste Breite). Auflösungsziel `3 %` in diesem Regime **nicht** erreicht; die übrigen 1B-Zellen sind **nicht gemessen** |
| Tokenidentität der Spekulation | `4B`/`128`/Breite `2`: Bruch reproduziert sofort in Paar `0`, erster divergierender Index `10`, ungegattertes Token (`j = 0`, Entwurf leer); Logits `75,0` gegen `74,5` = **exakt ein bf16-ULP** (Raster ablesbar in `experiments/identity_forensics/logit_gap.json`: alle Werte in `[32,64)` Vielfache von `0,25`, in `[64,128)` von `0,5`) | **Ursache gemessen** (`BACKLOG.md` S3): ein Forward der Breite `3` statt `1` kippt eine Ein-ULP-Entscheidung. Die Identitätsbehauptung „per Konstruktion" von `_decode_speculative` hält damit **nicht** — dauerhaft, nicht bis zur Klärung. Spekulation darf im Auslieferungspfad nie als tokenidentisch geführt werden; `friday_serve/speculation.py` verliert seine Begründung „braucht kein Promotionsgate, weil identisch per Konstruktion" |

**Gemessene 4B-Tabelle**, versiegelter `897`-Token-Prompt, gepaart, `6` Paare je
Zelle, Baselinearm = Auslieferungspfad:

| Token | Breite `1` | Breite `2` | Breite `3` | A/A des Regimes |
| --- | --- | --- | --- | --- |
| `32` | `−12,23 %` | `−22,61 %` | `−32,00 %` | `1,13 %` |
| `48` | `−15,98 %` | `−28,99 %` | `−43,58 %` | `1,04 %` |
| `64` | `−19,46 %` | `−38,95 %` | `−58,01 %` | `2,22 %` |
| `96` | `−26,46 %` | `−51,69 %` | `−75,35 %` | `0,52 %` |
| `128` | `−29,88 %` | **`identity_break`** | nicht gemessen | `0,70 %` |

Die `128`er Zeile ist zu lesen wie sie dasteht: Breite `1` ist mit `−29,88 %`
gemessen, Breite `2` brach die Tokenidentität in Paar `0` und beendete die
Studie nach Vorregistrierung, Breite `3` wurde deshalb **nicht** gemessen. Eine
ungemessene Zelle kann keinen Umschaltpunkt vortäuschen, aber sie kann einen
verdecken.

**Der Satz, der daraus folgt, mit seiner Bedingung:** der Auslieferungspfad
bündelt den Readback (D4, 2026-09-02); gegen **diesen** Pfad verliert
Prompt-Lookup-Spekulation auf `gemma-3-4b-it-4bit` bei jeder gemessenen Länge
und Breite, und auf `gemma-3-1b-it-4bit` an der schärfsten gemessenen Stelle
(`32` Token, Breite `1`) ebenfalls, dort als statistisch bestätigte
Verschlechterung. Der `(K,N)`-Dispatcher wird nicht gebaut, H1.1 und H1.2 des
Masterplans entfallen. Wird D4 je revidiert, verliert der Baselinearm seinen
strukturellen Vorteil und die Frage ist neu zu stellen.

Was dabei **nicht** gemessen ist und auch nicht projiziert wird: die übrigen
1B-Zellen (Längen `48`–`128`, Breiten `2` und `3`). Der Kill stützt sich auf die
vollständige 4B-Reihe plus den schärfsten 1B-Punkt, nicht auf eine Fortschreibung.

**Die Reichweite ist workloadbedingt, nicht modellbedingt.** S3 hat gemessen,
dass die Annahmequote auf der versiegelten Workload `0,0` ist — der Lookup
trifft dort nie. Der Kill gilt für **diese** Auslieferungsworkload, nicht für
Prompt-Lookup-Spekulation an sich. Auf einer wiederholungsreichen Workload
(`journal.txt`) misst S1 mit derselben Technik `3`–`6 %` Gewinn. Zwei Workloads,
zwei Vorzeichen, eine Ursache: die Trefferquote des Lookups. Wird D4 je revidiert, verliert der
Baselinearm seinen strukturellen Vorteil und die Frage ist neu zu stellen.

Der gepatchte IronMule-Arbeitsbaum wurde nach Prüfung von `patch.diff` und
`runtime_py_sha256` gegen die Dateien auf der Platte zurückgesetzt
(`9d30965e…`, `git status` leer, HEAD `03e884cb`); diese
Dokumentationsabschnitte entstanden **danach**.

## Stand nach Zyklus 17 bis 21 und Optimizer-Arbeiten (Kurzfassung)

> **Korrektur 2026-09-03 (Codex-Review).** Die mit `†` markierten Zeilen (Gemma
> Multi-Modell-Benchmark, Combinatorial Sweep, sowie Teile von D5) stammen aus
> Messharnessen mit methodischen Fehlern (blockweise statt gepaart, lazy-MLX-Graphen
> einmal evaluiert, Baseline auf Kandidatenlänge gekürzt) oder aus dem gelöschten
> `tools/autotune.py`, der erfundene `pairs=6`-Statistik in die versiegelte DB schrieb.
> Diese Zahlen sind **zurückgezogen**: `formal_claim=false`, keine Aktivierung,
> Wiederholungsmessung mit den reparierten Harnessen ausstehend
> (`docs/ARBEITSJOURNAL.md`, Eintrag 2026-09-03). Der RL-Controller läuft nur noch
> im Shadow-Modus. Unberührt bleibt die echte D4b-Kalibrierung
> (`…20260902-203442`: `head_skip` + `fixed_compiled` verified) und die
> vorregistrierten Studien F1, D2, P2.

| Bereich | Ergebnis | Zulässige Aussage |
| --- | --- | --- |
| Zyklus 16 Matmul-Umgebungs-A/B | `fixed_compiled` gegen `standard_eager` Ratio `0,9295921887`, 18/18 tokenidentisch | rechnerisch rund 7 % schnellere Decode-Phase, nur Laufzeitumgebung; `formal_claim=false`, keine Aktivierung |
| Zyklus 17 gebündelter Readback | Ratio-Median `0,9581074518`, 5-%-Schwelle verfehlt | `no_clear_speedup_baseline_retained`; gültiger Negativbefund |
| Zyklen 18–20 Fused-Greedy-Vorläufe | drei terminale fail-closed Abbrüche vor Modellload | Fehlerkette dokumentiert, keine Performance-Evidenz |
| Zyklus 21 Fused-Greedy-Compile | Ratio-Median `1,000510010` | `fused_greedy_compile_inconclusive`; Baseline bleibt |
| IronMule R6/R7 (2026-08-27) | Fixes verifiziert, Suiten grün, Wheel gebaut aber nicht installiert | R2/R3/R6-Architekturblocker bleiben offen |
| H0-Signal-Board (2026-08-28) | interaktives read-only Dashboard über `28` Runs | kein Performanceclaim aus Nachmessungen |
| L1 Optimizer (2026-08-30) | `friday_optimizer/`-Control-Plane offline implementiert; Memory `401` Records, Dataset `392` Records | `smoke_only/no_learning_claim` (`train=2`, `val=0`, `holdout=0`); Hardware, Learned Ranking und Promotion bleiben gesperrt |
| Q2 Readiness (2026-08-30) | erster Readiness-Versuch blockiert (`foreign_load`, Last, Speicher) | `model_started=false`, `session_consumed=false` |
| R0/R1 RL-Vorstufen (2026-09-01) | Entscheidungslogging mit Propensity und Replay-/OPE-Environment offline implementiert; Vollsuite `1459 passed` | Korpus ist leer, jeder Schätzer meldet `insufficient_data`; **kein** Lernclaim, RL bleibt NO-GO bis R2 |
| P2 Identitäts-Tie (2026-09-02) | realer gegateter Lauf: beide Chunkings kippen an Position `10`, dort kleinster Top-2-Abstand `0,500` bei Median `4,0`, Störung `2,25`–`2,50`; `tie_hypothesis_supported` | Prefill-Klasse ist **nicht mechanisch defekt**; Tokenidentität bleibt gebrochen und das Gate hat korrekt ausgelöst; `formal_claim=false` |
| **D5 `friday_serve` mit Knöpfen (2026-09-02)** | erste Messung des Auslieferungspfads: 4B `897`/`32` Ratio `0,8439`, KI `[0,8417; 0,8566]`, A/A `3,69 %`; 4B `897`/`256` `0,8590`; 1B `897`/`32` `0,6960`, A/A `14,25 %`. Tokenidentität auf allen Armen gehalten | **`15,61 %`** end-to-end kurz, `14,10 %` bei `256` Token, `30,40 %` auf dem 1B. Spekulation verschlechtert ab Breite `2` auf beiden Modellen. Antwortlängen über rund `287` Token sind wegen `continuous_gpu_limit_s = 6,0` nicht messbar. `formal_claim=false`, keine Aktivierung |
| D2 Serve-Äquivalenz (2026-09-02) | `friday_serve` gegen `mlx_lm.stream_generate`, drei Promptfamilien, alle Knöpfe aus | `equivalent`, Tokenfolgen identisch; Voraussetzung für D5 |
| Phase 3 Bandit (2026-09-02) | Prämisse widerlegt: Spekulation verliert auf `journal`/`tests` nicht reproduzierbar; Breiten unterscheiden sich um `0,016` | **terminaler Negativbefund** — Thompson Sampling entfällt, feste Entwurfsbreite; damit ist auch R1bs Kill-Kriterium erfüllt und R2s Kampagne bleibt der einzige Korpusweg |
| Kandidat 5 Prefill-Schrittweite (2026-09-02) | Offline-Vorfilter über die gesamte Gap-Evidenz meldet dreimal `degenerate`, acht von sechzehn Positionen | **terminaler Negativbefund** — die Blockstruktur-Klasse bleibt geschlossen, keine Hardwarezeit |
| **F1 warmer Arm (2026-09-02)** | erste End-to-End-Messung: `6` Paare, Tokenidentität `6/6`, Ratio-Median `0,8600567`, KI `[0,853444; 0,873056]`, Rauschen aus A/A `0,612 %` | **`qualified`** gegen vorregistrierte Schwelle `10 %`; Gewinn `13,99 %` end-to-end für ein Gerät, den Snapshot `93724907…`, `897`-Token-Prompt und `32` generierte Token; Projektion `13,68 %` bestätigt; `formal_claim=false`, keine Aktivierung |
| **D4b Kalibrierung & Profil (2026-09-02)** | Echte GPU-Kalibrierung auf M1 Max (`tools/run_calibration.py run --pairs 2 --execute`), MDE `0,34 %`. Knöpfe: `head_skip` (Ratio `0,8761`, KI `[0,8686; 0,8836]`, `verified`), `fixed_compiled` (Ratio `0,9854`, KI `[0,9824; 0,9883]`, `verified`), `prefill_step_size` (`not_applicable`). Persistiert in `.friday-data/device-profile.sqlite3`. | **`device_profile_dispatch`** aktiv im Serving-Pfad (`friday_serve status`). Reale Generierung mit verifizierten Knöpfen tokenidentisch bei `87,9 tok/s` Decode. |
| `†` Gemma Multi-Modell-Benchmark (2026-09-02) | ~~Gepaarter Benchmark 1B/4B/12B~~ — Harness maß blockweise statt gepaart und prüfte nur die letzte Tokenfolge; das „Bandbreite"-Feld war `model_gb × tps`, keine unabhängige Messung | **zurückgezogen**; Nachmessung mit `benchmark_gemma_family.py` (repariert) ausstehend |
| `†` Systematischer Combinatorial Sweep & Startup-Ablation (2026-09-03) | ~~Core Triad gewinnt +13,76 %/+16,47 %~~ — dieselben Harness-Fehler; `fuse_projections`-Sperre wegen `candidate_correctness_failed` bleibt korrekt | **zurückgezogen**; Nachmessung ausstehend. `fuse_projections` bleibt gesperrt (Korrektheit, nicht Lattenhöhe) |

## Geltende Entscheide und Grenzen

- **GO im exakten Scope:** begrenzter N8-Runtime-Prototyp, N10-Runtime-Prototyp,
  N8/N10-Shadow-Router (nur Shadow), Head-Skip-Runtime (Engineering-GO).
- **Negativ/abgelehnt:** Phase 1B Residual+RMSNorm (`baseline_fallback`),
  Gemma-Planer 1B/4B (`no_planner_qualified`), gebündelter Readback
  (abgelehnt als **Studienpromotion**, Zyklus 17, `0,9581` gegen die
  vorregistrierte `0,95`; im **Auslieferungspfad behalten** durch
  Nutzerentscheidung D4 vom 2026-09-02 — der H1.0-Befund hängt daran, dass der
  Auslieferungspfad bündelt), Fused-Greedy-Compile (inconclusive).
- **NO-GO:** produktive Phase 1B, adaptive Kernelsuche, breiterer Live-Suchraum,
  weitere Modellrunden ohne neue Freigabe, automatische Produktaktivierung,
  Downloads und Installationen.
- **NO-CLAIM:** Cross-Device, Cross-Model, allgemeine Modellqualität,
  Self-Learning.
- Reale Hardwareläufe nur manuell, AC-only, fremdlastfrei, maximal 30 Minuten,
  je Lauf einzeln freigegeben.

## Vollständige Historie

Der vollständige frühere Inhalt dieser Datei (alle Zyklen, Hashes, Preflights,
Audits und Rohwerte) steht unverändert in
[`docs/ARBEITSJOURNAL.md`](docs/ARBEITSJOURNAL.md) unter
„Archiv — vollständiger PROJECT_STATUS.md-Stand bis 2026-08-30".
Weitere Einstiege: [`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md),
[`BACKLOG.md`](BACKLOG.md), [`docs/KANDIDATENLISTE.md`](docs/KANDIDATENLISTE.md),
Studienakten unter `docs/*SPEC*.md` und `docs/*VORREGISTRIERUNG*.md`.

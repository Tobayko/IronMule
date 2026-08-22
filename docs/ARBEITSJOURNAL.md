# Arbeitsjournal

## 2026-08-19

- Benutzerziel: Sol ausschließlich als Orchestrator verwenden; Implementierung, Refactoring,
  Tests und operative Aufgaben ausschließlich durch Luna (`gpt-5.6-luna`); vor Installationen oder
  Downloads lokaler KI, Modelle oder Software Freigabe einholen; Änderungen und Messungen
  dokumentieren und Messwerte später in einer lokalen Historien-UI darstellen.
- Atlas-Purpose: ProjectAtlas bleibt für Navigation und Verifikation verbindlich; die Änderung
  ergänzt die bindenden Nutzerregeln in `AGENTS.md`.
- Verifikation: erster Sandbox-Lauf Exit 1 (`RuntimeError: No Metal device available`) wegen
  fehlendem Sandbox-GPU-Zugriff; genehmigter Lauf außerhalb der Sandbox Exit 0, Tool-Walltime
  1.741108708 s.
- Audit: Xcode 26.6, ProjectAtlas 0.4.5-rc1, MLX 0.32.0 Metal auf Apple M1 Max mit
  34,359,738,368 Bytes Device Memory, PyTorch 2.13.0 MPS Correctness ok und drei MCP-
  Konfigurationen ok.
- Keine Installationen und Downloads ausgeführt. Es gibt noch keinen PoC, Benchmark, Rohdaten-
  bestand oder Dashboard; die Architekturwahl steht aus.
- Atlas-Purpose-Mutation: Root `.` wurde nach einem stale/conflict Review per `atlas_purpose_set`
  auf `approved` / `source: agent` / `agent_reviewed: true` gesetzt. Exakter Zweck: "Research
  project for a Hardware-Aware Self-Optimizing AI Runtime, including its proof of concept,
  experiments, tests, and project documentation."
- Korrektur dieses Journaleintrags: Die Ergänzungen in `AGENTS.md` (dauerhafte Fehler-/Ursachen-/
  Lösungsdokumentation, Baseline-Vorher-/Nachher-Messungen und `docs/ARBEITSJOURNAL.md` in der
  Projektstruktur) sowie diese Purpose-Dokumentation sind hiermit selbst erfasst.
- Dateiänderungen dieses Eintrags: `AGENTS.md`, `PROJECT_STATUS.md` und diese Datei
  (`docs/ARBEITSJOURNAL.md`) sind hiermit selbst dokumentiert.

## 2026-08-19 — Phase-1-Vorregistrierung

- Entscheidung: `docs/PHASE1_MATMUL_SPEC.md` registriert den kleinsten Versuch als feste
  FP16-C-contiguous-Matmul (`2048²`, `Y=mx.matmul`), ohne Customcode. Die Spec ist
  speicher-/UI-neutral; JSONL-vs.-SQLite und das Loopback-Dashboard bleiben offen.
- Messvertrag: PCG64 Seed `0xF17A2026`, Host-FP32→FP16, SHA-256, FP64-Oracle aus den
  exakten FP16-Werten, 17.179869184 GFLOP und 25,165,824 Bytes Nutzdaten. Eager und
  `mx.compile(..., shapeless=False)` werden getrennt nach Compile/First-eval/Cold/Warm
  gemessen; Zweierpotenz-Fenster, Warmups, 3+3 Prozesse, 30 gepaarte Blöcke und
  hierarchisches Bootstrap sind vorab eingefroren.
- Gate-Entscheidung: Win nur bei Ratio `<=0.95`, oberem 95-%-KI `<1`, keiner Session
  `>=1.05` sowie grünen Holdout-, Correctness-, Memory- und Safety-Gates; sonst
  Regression/Unentschieden/Baselinefallback. Fehlende Metriken sind `null` plus Grund.
- MLX-0.32-Introspection/API-Verfügbarkeit: Der aktuelle Luna-Read-only-Introspektionslauf
  im bestehenden `.venv` bestätigte MLX 0.32.0 sowie `mx.matmul`, `mx.eval`,
  `mx.synchronize`, `mx.compile` und
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`. Die Sandbox hatte kein Metal; daher fand in diesem Lauf
  kein GPU-Lauf statt.
- ProjectAtlas: Runtime `0.4.5-rc1` und fokussierter Session-Brief wurden vor der
  Dokumentation abgerufen. Der Brief zeigte stale/conflict-Purpose-Hinweise; diese
  wurden bewusst nicht an `ProjectAtlas/` mutiert. Die Purpose-Zeile für den Root bleibt
  als bereits geprüfte, absichtlich dokumentierte Entscheidung bestehen.
- Scope-/Sicherheitsbefund: Root ist kein Git-Repository (`revision: null` mit Grund);
  Code-/Manifest-/Spec-/Umgebungs-Hashes sind für spätere Läufe vorgesehen. Es wird
  keine Netzisolation behauptet. Worker-, Timeout- und Speicherlimit-Durchsetzung bleibt
  eine separate Sicherheits-/Architekturentscheidung und wurde nicht implementiert.
- Verifikation: Spec mit `wc -l` auf 257 Zeilen geprüft (Ziel 180–260). Keine Tests,
  Downloads, Installationen oder GPU-Tests ausgeführt; keine Messwerte oder Rohdaten
  erzeugt. Geänderte Forschungsdateien: `docs/PHASE1_MATMUL_SPEC.md`, `AGENTS.md`,
  `PROJECT_STATUS.md`, `docs/ARBEITSJOURNAL.md`; `ProjectAtlas/` blieb unverändert.

## 2026-08-19 — Korrektur der Phase-1-Spezifikation und Introspection-Dokumentation

- ProjectAtlas: Die Queue-Zeile war exakt der Ordner `ProjectAtlas`; der Dry-Review
  meldete `stale/conflicts=1`. Wegen der Upstream-Regel wurde `ProjectAtlas/` nicht
  mutiert.
- Entscheidung/Korrektur: Reine `mx.compile`-Wrapperzeit heißt nun
  `compile_wrapper_setup_ns`; nur `first_eval_compile_inclusive_ns` darf die tatsächliche
  erste Kompilierung einschließen. Bei einem 4096er-Fenster außerhalb 50–200 ms wird die
  Session `invalid` und es gibt keine Performanceaussage. Warmup, Manifest-Seeds und
  Baseline-only-Correctness-Envelope sind nun exakt festgelegt.
- Befund: Der aktuelle Luna-Read-only-Introspektionslauf im bestehenden `.venv` bestätigte
  MLX 0.32.0 und die dokumentierten Matmul-, Eval-, Synchronisations-, Compile- und
  Memory-APIs. Die Sandbox hatte kein Metal; es wurde daher kein GPU-Lauf ausgeführt.
- Änderung/Verifikation: Nur `docs/PHASE1_MATMUL_SPEC.md`, `PROJECT_STATUS.md` und
  `docs/ARBEITSJOURNAL.md` wurden geändert; keine Installation, kein Download, kein Test
  und kein GPU-Lauf wurde ausgeführt. Anschließend wird ein Atlas-Watcher-Lauf und eine
  erneute Prüfung der exakten Dokument-Slices durchgeführt.
- Korrektur: Die hierarchischen Bootstrap-Resamples sind nun getrennt und fest
  vorregistriert: Charakterisierung `0xB0052026`, Holdout `0xB0052126`, jeweils 10.000.
  Die Ratio-, KI- und Session-Gates gelten in beiden Datensätzen separat.
- Korrektur: Das Akzeptanzkriterium nennt nun explizit
  `compile_wrapper_setup_ns` und `first_eval_compile_inclusive_ns`.
- Historie: Die frühere 257-Zeilen-Prüfung bezog sich auf den Stand vor der Korrektur;
  der aktuelle Stand vor dieser letzten Miniänderung war 276 Zeilen.
- Verifikation: `atlas_watch_once` erfolgreich; die Watch-Ausgabe enthielt keine
  Generationsnummer. Exakte Slices wurden geprüft.

## 2026-08-19 — Unabhängiger Luna-Audit: Phase-1A-Scope und Correctness

- Auditbefund P1 (Scope): Die Spezifikation war als Phase 1 bezeichnet, obwohl sie den
  vollständigen DoD aus `IMPLEMENTIERUNGSPLAN.md` nicht erfüllt. Ursache: Harness-Preflight,
  sichere `mx.compile`-Negativkontrolle und der spätere Custom-MLX-Metal-Kandidat waren nicht
  klar getrennt. Lösung: `docs/PHASE1_MATMUL_SPEC.md` ist nun eindeutig `Phase 1A —
  Matmul-Harness-Preflight`; alle Akzeptanzkriterien gelten nur Phase 1A. Phase 1B mit
  begrenztem Custom-MLX-Metal-Kandidaten und isoliertem Worker ist ausdrücklich offen,
  nicht erfüllt und braucht separate Sicherheits-/Architekturfreigabe.
- Auditbefund P2 (Correctness): Eine eigenständige, nicht getimte Correctness-Matrix mit
  festen sichtbaren und Holdout-Seeds sowie der Zero-RHS- und Sign-Invariante fehlte. Lösung:
  Matrix und Gates wurden ergänzt; alle Oracles verwenden exakt konvertierte FP16-Werte,
  und Correctness-Fälle bleiben vollständig außerhalb der Performanceaggregation.
- Mess-/API-Korrektur: Der Timing-Contract schreibt nun pro Output `mx.eval(out)` und vor
  Zeitfensterende `mx.synchronize()` vor; `time.perf_counter_ns`, `mx.eval` und
  `mx.synchronize` werden im Manifest festgehalten. `mx.compile` ist nur die sichere bekannte
  Negativkontrolle gemäß `CODEX_START.md`.
- ProjectAtlas: Der unabhängige Read-only-Audit meldete das verschachtelte `ProjectAtlas/`
  als git-clean; das Upstream-Repository wurde nicht verändert. Nach den Dokumentänderungen
  wurde ein `atlas_watch_once` ausgeführt und die exakten Slices der drei freigegebenen
  Dokumente erneut geprüft.
- Verifikation: Keine Tests, GPU-Läufe, Installationen oder Downloads ausgeführt; keine
  Messwerte oder Rohdaten erzeugt. Geändert wurden ausschließlich `docs/PHASE1_MATMUL_SPEC.md`,
  `PROJECT_STATUS.md` und `docs/ARBEITSJOURNAL.md`.

## 2026-08-19 — Phase-1A-Architekturvorschlag und Worker-Review

- Entscheidungsvorschlag (nicht freigegeben): Sol empfiehlt als Paket SQLite aus der vorhandenen
  Python-Standardbibliothek (`sqlite3`), Migration v1, Transaktionen und vollständige Rohsamples;
  JSONL bleibt als Alternative möglich. Dazu gehört ein read-only Python-Stdlib-Dashboard nur auf
  `127.0.0.1`, ohne externe Assets, mit festen Result-/Zeitlimits und Historie.
- Worker-Option A wurde ausschließlich dokumentiert: fester `python -m`-Entrypoint, geschlossenes
  doppelt validiertes Manifest ohne unbekannte Felder/Pfade/Source/Flags/Module, `start_new_session`,
  monotonic Watchdog, `killpg`, bounded wait, aktiv drainendes stdout/stderr mit Bytebudget,
  begrenztes Ergebnis, feste `mx.matmul`/`mx.compile`-Konfiguration und Baselinefallback.
- Sicherheitsabgrenzung: `mx.metal.set_memory_limit(1GiB)`, Parent-RSS-Ziel 2 GiB/Polling und
  bereinigtes env/cwd/`close_fds` sind nur Best Effort. Netzwerk-/Filesystemisolation, harte
  Unified-Memory-Grenze, Schutz vor GPU-/Driver-Hang und Parent-death-Garantie werden nicht behauptet.
- Worker-Review-Befund: Python 3.12.13 auf Darwin bietet `setsid`/`killpg`; `RLIMIT_AS` und
  `RLIMIT_RSS` sind beide Resource-ID 5, RSS daher nur Präferenz; `ru_maxrss` erst nach Ende;
  MLX-Memory-Limit nicht als harte Unified-Memory-Grenze belegt. Stop-Kriterien wurden dokumentiert.
- Phase 1B Custom Metal bleibt nicht freigegeben. Option B (signierter App-Sandbox-Helper) sowie
  Entitlements, Deployment und Installation benötigen eine separate Sicherheits-/Architekturfreigabe.
- Verifikation: keine Tests, GPU-Läufe, Downloads oder Installationen; keine Messwerte/Rohdaten
  erzeugt. Geändert wurden ausschließlich `docs/PHASE1A_ARCHITEKTURFREIGABE.md`,
  `PROJECT_STATUS.md`, `AGENTS.md` und dieses Journal. ProjectAtlas/ blieb unverändert.
- Fehler erkannt: Patch-Anker platzierte neuen Eintrag am Dateikopf; Ursache war nicht-EOF-Anker;
  Lösung: Block ans EOF verschoben, künftige Einträge ausschließlich EOF; keine historischen Inhalte sonst geändert.

## 2026-08-19 — Kritische wissenschaftliche Neubewertung und Forschungspivot

- ProjectAtlas-first erneut ausgeführt: vollständiger ProjectAtlas-Skill gelesen, Runtime
  `0.4.5-rc1` und fokussierter Session-Context abgerufen; ProjectAtlas/ blieb unverändert.
- Entscheidung dokumentiert: Phase 1A ist ein H0-Messsystem-Preflight, nicht der Nachweis
  von Self-Optimization, allgemeiner Hardware-Awareness, LLM-Nutzen oder Generalisation.
  Ein Forschungspivot auf H1 deterministic template-constrained tuning wird empfohlen;
  Architektur und Umsetzung bleiben nicht freigegeben.
- Methodische Korrekturen festgehalten: `mx.compile` ist eine Framework-
  Vergleichsvariante, A/A der echte Nullpfad; die bisherigen drei Bestätigungsprozesse
  sind Replikate derselben `2048²`-Workload und kein unbekannter Performance-Holdout.
  Drei Prozesscluster sind nur ein Engineering-Gate; H1 benötigt nach dem A/A-Pilot eine
  cluster-level Powerplanung.
- Unabhängige FP64-Hard-Caps bleiben das Semantik-Gate. Baseline-abgeleitete Faktoren sind
  nur zusätzliche Regressionsdiagnose und keine Zulässigkeitsdefinition. H0-Kontrollen für
  A/A, absichtlich langsam/falsch, Missing Data sowie Timeout/Crash nach Workerfreigabe
  wurden als Gates vorgeschrieben.
- Literaturmesswerte und Ursachen dokumentiert: Metal-Sci ist direkte Prior Art für
  Apple-Silicon-Metal-/LLM-Kernelsuche; `Gaming Without an Attacker` berichtet `16/53 = 30 %`
  Transfer-Failures durch Benchmark-Fingerprinting; KernelBench-Verified zeigt die
  Notwendigkeit starker Baselines, verborgener Testverteilungen und Memory-Metriken.
  BaseRT wurde nur als Preprint-/Autorenclaim und nicht als bestätigte Baseline markiert.
- Erfolgreiche Lösung: neue Entscheidungsvorlage `docs/KRITISCHE_NEUBEWERTUNG_2026-08-19.md`
  mit Claim-Ledger, H0/H1/H2, Split-/Overfitting-Regeln, Amortisationsformel,
  Metriken, Kill-/Pivot-Kriterien, Dashboard-Feldern und Live-Modell-Gate.
- Geändert wurden ausschließlich `docs/KRITISCHE_NEUBEWERTUNG_2026-08-19.md`,
  `docs/TECHNISCHES_KONZEPT.md`, `docs/PHASE1_MATMUL_SPEC.md`, `PROJECT_STATUS.md` und
  dieses Journal. `IMPLEMENTIERUNGSPLAN.md`, `CODEX_START.md`,
  `docs/PHASE1A_ARCHITEKTURFREIGABE.md`, `AGENTS.md` und `ProjectAtlas/` blieben unverändert.
- Verifikation in diesem Dokumentationsschritt: keine Tests, GPU-/Worker-Läufe, Downloads,
  Installationen, Modelltests oder Benchmark-Rohdaten; der lokale Modelltest bleibt bis zu
  ausdrücklicher Freigabe gesperrt.

## 2026-08-19 — Post-Edit-P1-Review und formale H1-Sperre

- Unabhängiges Luna-Review erneut ProjectAtlas-first ausgeführt: Skill vollständig gelesen,
  Runtime `0.4.5-rc1`, Session-Brief und exakte Dokument-Slices abgerufen. Der Post-Edit-
  Atlasstand war vor diesem Patch Generation `21`, `549 Dateien`, `257 Ordner` und der
  Session-Brief `805` Blocker; diese Metriken bleiben getrennt.
- P1-Ursache: Die erste H0-Kontrollbeschreibung ließ A/A, Analyse-Fixtures und geplante
  Worker-Controls semantisch zusammenfallen. Lösung: GPU-A/A mit exakt drei Charakterisie-
  rungs- und drei Bestätigungsprozessen, je 30 Paaren, festen `AA1A`-/`AA0D`-Seeds und
  explizitem `tie`-Gate; deterministische Nicht-GPU-Analyse-Fixtures und gesperrte
  `control_timeout`-/`control_exit_70`-Modi getrennt dokumentiert.
- Analyse-/Correctness-/Missing-Data-Lösung: Baseline-Formel mit `p,b`, Slow `1.10x`,
  optionaler Known-Win `0.90x` nur analytisch, Falsch-Fixture `64²`/Seed `0xBAD02026`,
  fehlendes `rss_peak_bytes` ohne `missing_reason` und identischer Replay-Decision-Hash.
  Sleep- oder GPU-Zeit ist für diese Kontrollen ausdrücklich ausgeschlossen.
- H1 wurde formal gesperrt: A/A schätzt ausschließlich die SD der Session-Log-Ratios;
  danach müssen Mindestwirkung `5 %`, `alpha 0.05`, Power `0.80`, feste Familien-/Cluster-
  zahl, mindestens fünf unabhängige Sessions je Arm/Familie und eine obere Machbarkeits-
  grenze (empfohlen `20`) vor Kandidatensichtung registriert werden. Überschreitung führt
  zu `infeasible/no claim`; Pilotdaten gehen nicht in H1 ein.
- Versiegelte Testsets verwenden pro Hypothese/Revision frische nicht-enumerierbare
  Shape-/Value-/Layout-Verteilungen, einen 256-bit-Seed außerhalb des Repositories und
  vorab nur dessen kryptographischen Commit-Hash. H2 darf kein durch H1 geöffnetes Set
  wiederverwenden.
- H1-/H2-Metrikursache korrigiert: H1 nutzt `R=T_candidate/T_strongest_baseline`,
  Familien-Sessionmediane, geometrisches Mittel auf Log-Skala, `R<=0.95`, obere Cluster-
  KI `<1.0`, Guardrails und separates `N_break_even`. H2 hat genau eine Primärmetrik,
  finale Best-Valid-Sealed-Test-Ratio nach Trialbudget `B`; übrige Werte sind sekundär.
- Dokument-Line-Messungen dieses Reviewstands: Neubewertung `314` neu; Konzept `1504 →
  1536`; Spezifikation `328 → 369` vor diesem P1-Patch; Status `89 → 100`; Journal `135 →
  168` vor diesem Append.
- Journal-Provenienz: Vorheriger SHA-256 `5c55115910587bd990b6afa058897385e832a07214b0f9e284c6c42862625ab5`,
  Vorhergröße `12623` Bytes. Künftig wird vor jedem EOF-Append derselbe Hash-/Bytecount-
  Schritt durchgeführt; der Post-Append-Hash wird nicht erneut ins Journal geschrieben,
  um Rekursion zu vermeiden.
- Geändert wurden in diesem Review ausschließlich `docs/PHASE1_MATMUL_SPEC.md`,
  `docs/KRITISCHE_NEUBEWERTUNG_2026-08-19.md`, `PROJECT_STATUS.md` und dieses Journal.
  Keine Runtime-/GPU-/Worker-/Modelltests, Downloads oder Installationen fanden statt.

## 2026-08-19 — P1-Restprüfung: A/A-Schätzer, Break-even und Atlas-Status

- Erneutes unabhängiges Luna-Review nach ProjectAtlas-First: Skill vollständig gelesen,
  Runtime `0.4.5-rc1` und fokussierter Session-Brief abgerufen; der Atlas-Index war verfügbar
  mit 549 Dateien und 257 Ordnern. Vor der Änderung wurden nur die exakten Slices der vier
  betroffenen Dokumente gelesen.
- P1-Restursache im A/A-Gate: Das frühere Engineering-Gate band weder die Session- und
  Set-Schätzer noch das hierarchische Bootstrap formal an. Lösung in der Spec:
  `R_s=exp(median_b(log(t_B/t_A)))`, `R_AA=exp(median_s(log(R_s)))`, hierarchisches
  10.000er-Perzentil-Bootstrap über Sessions und Blöcke, feste Seeds `0xAA052026` und
  `0xAA052126`, vollständiges 95-%-KI und Engineering-Band `[0.98,1.02]` sowie Sessionband
  `[0.95,1.05]`. Der Text kennzeichnet dies ausdrücklich als Engineering-Äquivalenzgate,
  nicht als wissenschaftlichen Äquivalenznachweis.
- P1-Restursache beim Break-even: stärkste Baseline, Scope und nichtpositive/fehlende Zeiten
  waren nicht entscheidungsfest. Lösung in Neubewertung und Spec: stärkste Baseline als
  `T_strongest_baseline`, nichtnegative Zeiten in gleicher Einheit und demselben registrierten
  Workload-Mix/Scope, `no_break_even`/unendlich bei nichtpositivem Nenner und sonst die
  aufgerundete Formel `ceil((T_tune+T_compile)/(T_strongest_baseline-T_candidate))` mit
  Hard-Gate `<= N`.
- P1-Restursache im Status: historische `543 Dateien / 257 Ordner / 281 Purpose-Hinweise`,
  aktueller Post-Edit-Atlas-Stand und Session-Brief wurden nicht vollständig getrennt.
  Lösung: historische Verifizierungs-Bullet explizit markiert, Generation `22` mit `549/257`,
  aktuelle Atlas-Overview mit `280` fehlenden Purposes und Session-Brief `805` als getrennte
  Metriken dokumentiert.
- Messprotokoll vor diesem EOF-Append: Journal SHA-256
  `b5d5db3c52c64feb19608a83c1f953f504a419f1bfa454eb90ba32e5a8e8dd1a`, `15621` Bytes,
  `207` Zeilen. Vor jedem weiteren EOF-Append ist derselbe Hash-/Bytecount-Schritt erneut
  auszuführen; der Post-Append-Hash wird nicht wieder in das Journal geschrieben, um
  Rekursion zu vermeiden.
- Geändert wurden in diesem Restpatch ausschließlich `docs/PHASE1_MATMUL_SPEC.md`,
  `docs/KRITISCHE_NEUBEWERTUNG_2026-08-19.md`, `PROJECT_STATUS.md` und dieses Journal.
  Keine Tests, Downloads, Installationen, Modelltests, Runtime-, Worker- oder GPU-Läufe fanden
  statt.

## 2026-08-19 — Root-Git-Statusfehler und korrekte Sauberkeitsprüfung

- Der gerade beobachtete Root-Git-Check `git status --short` im Verzeichnis
  `/Users/tobiasburandt/Project_Friday` endete mit Exit `128`; `git diff --stat` im selben
  Root endete mit Exit `129`. Ursache ist, dass der Project-Friday-Root kein Git-Worktree ist.
  Ausschließlich das verschachtelte `/Users/tobiasburandt/Project_Friday/ProjectAtlas/`
  ist versioniert.
- Erfolgreiche Handhabung: Git-Sauberkeit wird ausschließlich mit
  `git -C ProjectAtlas status --short` geprüft. Root-Dokumentänderungen werden unabhängig
  davon über explizite Datei-/Hash-/Byte-/Zeilenprüfungen und ProjectAtlas-Summaries/-Slices
  verifiziert. Das vermeidet die falsche Schlussfolgerung, der Root sei wegen Exit 128/129
  schmutzig oder beschädigt.
- Messprotokoll vor diesem EOF-Append: Journal SHA-256
  `35a2649aa8d98e28896fd2dbf7d44bded84631d7eae6fba6f3f139917d6b52e6`, `18073` Bytes,
  `241` Zeilen. Keine Tests, Downloads, Installationen, Modelltests, Runtime- oder GPU-Läufe
  wurden ausgeführt.

## 2026-08-19 — Freigegebener H0-Offlinestand und adversariales Abschlussreview

- ProjectAtlas wurde für diesen Dokumentationspass nicht erneut aufgerufen: Das bereits
  dokumentierte Nutzungslimit blockiert den Refresh. Verwendet wurden ausschließlich der
  zuvor gelesene vollständige Skill, der fokussierte Atlas-Kontext und die bekannten
  Abschlussberichte. Der letzte bekannte verschachtelte `ProjectAtlas/`-Gitstatus war sauber;
  der Project-Friday-Root ist kein Git-Worktree. Es gab keinen Download, keine Installation,
  keinen Modelltest, keinen MLX-/GPU-/Metal-Lauf und in diesem Dokumentationspass keinen
  Test- oder Prozessaufruf.
- Der Nutzerentscheid vom 19.08.2026 wurde in allen aktuellen Übergabedokumenten exakt
  festgehalten: `JA — Ich gebe den Forschungspivot H0 → H1 → H2 und die Implementierung von
  Phase 1A/H0 mit SQLite v1, read-only Loopback-Dashboard und festem Worker Option A frei.
  Keine Downloads, Installationen, Custom-Metal-Kernels oder Modellgewichte.` Phase 1A ist
  damit `approved/implemented-offline`; `mlx-run` bleibt Exit `78`/`not_released`.
- H0 ist ausdrücklich nur eine einzelne FP16-`2048²`-Matmul, kein Modelltest und kein
  Nachweis von Self-Optimization, Hardware-Generalisation, realer Performance,
  Correctness-, Memory- oder Safety-Gates. Der Offline-Unterbau bindet SQLite v1 an
  `.friday-data/h0.sqlite3`, Worker Option A an einen geschlossenen Vertrag und das
  Dashboard read-only an `127.0.0.1`. Die nächste Reihenfolge wurde festgelegt:
  finaler Gesamttest → Offline-Control-Historie/UI → separat angekündigter MLX-H0-Go/No-Go-
  Lauf → A/A-3+3-Aggregation → erst danach H1-Planung; H2-Modelle und Custom Metal sind
  spätere, separat zu prüfende Schritte.
- `docs/H0_ADVERSARIAL_REVIEW.md` bündelt die ursprünglichen P0/P1-Findings und die
  Kontrollen: keine Einzelprozesspromotion; tatsächlich gepaarte A/A-Reihenfolge;
  getrennte Memory-/RSS-/Missing-Felder; exakte A/A-Formeln, Seeds und Gates;
  FP64-Correctness-Oracle außerhalb der Performanceaggregation; vollständiger Adapter-
  und RSS-Nachweis; Workload-/Shape-Familie statt Timingblock als Inferenz-Einheit;
  Snapshot-/Sample-Count-Trennung; SQLite-Atomicity und Replay; neutraler Common Result;
  sowie Seed-Konflikttrennung (`AA05` A/A, F17A/B10C Session, B005 späterer H1-Kontext).
  Die verbleibenden Stop-Gates sind dort ausdrücklich als nicht-Hardware-Evidenz markiert.
- `docs/PHASE1_MATMUL_SPEC.md` wurde von offener JSONL-/SQLite- und Dashboardwahl auf den
  implementierten SQLite-/read-only-Vertrag aktualisiert. Die A/A-Bootstrap-Seeds
  `0xAA052026`/`0xAA052126` sind ausschließlich A/A und jetzt manifestgebunden; der
  rückwärtsinkompatible Manifest-v1-Contractfix bleibt vor dem echten A/A-Lauf zu
  verifizieren. `0xB0052026`/`0xB0052126` sind ausdrücklich spätere Kandidaten-/H1-Seeds.
  Timeout/Crash sind geschlossene Offline-Control-Fixtures; ein realer GPU-Control-Lauf
  bleibt bis zum separaten Go/No-Go gesperrt. Statistikformeln wurden nicht verändert.
- `PROJECT_STATUS.md`, `IMPLEMENTIERUNGSPLAN.md`, `CODEX_START.md` und
  `experiments/README.md` wurden auf den Freigabestand, H0-Offline-Harness, feste DB,
  nächste Reihenfolge und fehlende produktive Rohdaten synchronisiert. README und Status
  behaupten keine erfundene Historie; Dashboardtests nutzen nur temporäre DBs.

### Gemeldete Offline-Test- und Messwerte

- Konsolidierter erster Lauf: `129 collected`, `106 pass`, `7` veraltete Seed-Fixtures und
  `16` Sandbox-Socketfehler. Die Scope-/Umgebungsunterschiede werden nicht als Regression
  interpretiert. Nach dem Fixturefix ausschließlich in `tests/test_protocol.py` liefen
  Protocol/Worker/Supervisor `24/24`, Wall `0.823154 s`, Self U/S `0.257013/0.080812 s`,
  RSS `46,301,184 B`.
- Gesamte Suite ohne Dashboard: `113/113`, Wall `20.778518 s`, Self U/S
  `19.526546/0.333315 s`, Self-RSS `64,405,504 B`, Child-RSS `23,609,344 B`. Diese Suite
  ist wegen unterschiedlicher Scopes/Umgebungen nicht mit dem ersten Lauf als Regression
  vergleichbar.
- Foundation-Korrekturrunden: zunächst `24` Tests mit einem Fehler durch `/var` versus
  `/private/var`; nach `.resolve()` `24/24` grün bei ca. `0.424 s`; danach Owner/Mode
  `25/25` bei ca. `0.410 s`, private-Limits `26/26` bei ca. `0.397 s`, und nach der
  Scalar-Grenze `27/27` bei `0.385077 s`, Self-MaxRSS `37,666,816 B`, Child-MaxRSS
  `27,492,352 B`. Diese Messwerte stammen aus priorer Foundation-Verifikation.
- Benchmark/Worker/Protocol/Storage/Runner/Provenance fokussierte Werte wurden in den
  jeweiligen Runden nicht immer separat gemeldet; fehlende Einzelwerte bleiben deshalb
  unbekannt und werden nicht aus Suitewerten abgeleitet. Dashboard: zuvor `13/13` HTTP-
  Einheiten und später `3/3` finite/cleanup-Einheiten; eine konsolidierte Wall-/RSS-Metrik
  für diese Teilmengen wurde nicht berichtet. In der Sandbox schlugen zusätzlich `16`
  Loopback-Socketversuche mit `PermissionError` fehl; dies ist keine Dashboard-
  Funktionsaussage außerhalb der Sandbox.
- Offline-Historie: DB-Init Exit `0`, `0.106884 s`; fünf Runs plus ein idempotenter Replay.
  Slow `0.195217 s`, Exit `10`, `regression`; Known-Win `0.164255 s`, Exit `0`, nur
  analytisch `promoted`; Wrong `0.151175 s`, Exit `10`, `correctness`; Missing
  `0.158439 s`, Exit `10`, `missing`; Exit70 `0.141642 s`, Exit `10`, `worker_exit`;
  Replay `0.153464 s`, `idempotent`. Sequenz `0.964355 s`, Self-RSS `15,859,712 B`,
  Child-RSS `28,524,544 B`. DB `118,784 B`, Modus `0600`, UID `501`, Elternverzeichnis
  `0700`, `application_id=1179797552`, `user_version=1`, fünf Runs, je ein Common Result,
  keine Rohsamples, Snapshot verfügbar/nicht abgeschnitten, `run_count=5`, stabile
  Revisions-/Identitätshashes (`e7ca2fed…`, `d9c1bdfa…`). Kein GPU-Lauf.
- `xcodebuild -checkFirstLaunchStatus` endete mit Exit `0`. Für A/A ist der
  Manifest-Bootstrap-Seedvertrag nun als `AA05` festgelegt; der Offline-Status meldet
  `aggregation_contract_ready=true`, `live_execution_authorized=false`.

### Fehler, Ursachen und erfolgreiche Lösungen

- Non-ASCII-Bytes-Literal: ein Python-Bytesliteral enthielt nicht-ASCII-Zeichen; Ursache
  war Quelltextkodierung im Testfixture. Lösung: kanonische ASCII-/UTF-8-Grenze und
  explizite Bytesrepräsentation.
- `/usr/bin/time -l`-Clockrate: die externe Wall-/Clockrate-Ausgabe war umgebungsabhängig.
  Lösung: reproduzierbare Python-Wallzeit und `resource.getrusage`; kein behaupteter
  Cross-System-RSS-Vergleich.
- ProjectAtlas-Slices, Concurrent-Generation, DB-Lock und Usage-Limit: Atlas-Slices wurden
  wegen Refresh-/Dependency-Limit bzw. Lock/Concurrent-Generation nicht zuverlässig
  fortgesetzt. Ursache ist externer Atlas-Zustand; Lösung war die dokumentierte Nutzung des
  bereits gelesenen fokussierten Kontextes, kein breiter Fallback und keine Atlasänderung.
- `tempfile`-Import: ein Testpfad verdeckte/fehlte den Standardbibliotheksimport. Lösung:
  Import-/Temp-Root-Verwendung korrigiert und private Test-Root-Grenze beibehalten.
- Process-Set: freie oder inkonsistente Set-/Indexwerte konnten den Vertrag verletzen.
  Lösung: geschlossene Set-/Index-/Seed-Allowlist und fail-closed Validierung.
- Cleanup-Deadline: Dashboard-/Worker-Aufräumen durfte nicht unbounded blockieren. Lösung:
  feste Cleanup-/Deadline-Behandlung und explizite Fehlerklasse.
- Interpreter-Pfad: Testläufe waren zwischen System-Python, `.venv` und `python3` nicht
  vergleichbar. Lösung: den tatsächlich verwendeten vorhandenen Interpreter jeweils
  protokollieren; keine Installation.
- Dashboard `414`, `source_busy` und Socket-Sandbox: übergroße Requests bzw. belegte
  Quelle werden bounded abgewiesen; Socket-Bindung bleibt in der Sandbox durch
  `PermissionError` eingeschränkt. Keine Sicherheits- oder Live-Erreichbarkeitsgarantie
  daraus ableiten.
- Benchmark-Mock-Rekursion und Top-Level-Classification: Fake-Backend-/Callable-Rekursion
  und eine unzulässige Top-Level-Klasse wurden erkannt. Lösung: injizierbare, getrennte
  Offline-Callables und neutrale `measurement_complete`-Semantik ohne Promotion.
- Storage Overflow/Leaks: zu große Integer/JSON-Hüllen und temporäre Ressourcen konnten
  Grenzwerte bzw. Cleanup verletzen. Lösung: signed-64-/finite-/depth-/byte-Limits,
  normalisierte `StorageError`-Pfade, transaktionaler Rollback und Replayprüfung.
- Runner Temp-Root/Path-Resolve/Argparse/Dashboard-Exit: Pfadvergleich `/var` vs.
  `/private/var`, Test-Root-/Symlinkgrenzen sowie unbounded Argumentfehler wurden gefunden.
  Lösung: `.resolve()`, private tokengebundene Testroots, `O_NOFOLLOW`-Best-Effort,
  statische Exit-64-Ausgabe, `mlx-run` Exit 78 und kein CLI-Test-Seam.

### Ehrliche Grenzen und Prozessabweichung

- RSS und Memory-Limit bleiben best effort; es gibt keine harte Unified-Memory-, Netzwerk-,
  Dateisystem-, Treiber-Hang- oder Parent-Death-Garantie. Python-`sqlite3` ist nicht
  vollständig fd-bound TOCTOU-frei. Hash-Domain-Separation wurde bewusst nicht still
  geändert. Der Root ist kein Git-Worktree; Git-Sauberkeit gilt nur für `ProjectAtlas/`.
- Ein read-only Dokumentations-/Reviewauftrag führte entgegen seinem Scope in einer
  früheren Korrekturrunde zu einem lokalen Testaufruf. Dieser war sicher und ohne MLX/GPU,
  wird aber als Prozessabweichung dokumentiert. Der vorliegende Doku-Pass selbst führte
  keine Tests oder Prozesse aus.
- Vor diesem EOF-Append tatsächlich gemessen: Journal SHA-256
  `8c6e69daf5838d0264497c0ba37fa1b376d0559a051ed2878746896b30c02ad4`, `19150` Bytes,
  `258` Zeilen. Ein Post-Append-Hash wird nur im Abschlussbericht genannt und nicht erneut
  in dieses Journal geschrieben, um Append-Rekursion zu vermeiden.

## 2026-08-19 — Erratum: A/A-Contractfix und Dashboard-HTTP-Evidenz

- Eine historische Aussage im vorherigen Eintrag, wonach der rückwärtsinkompatible
  Manifest-v1-Contractfix für die A/A-Bootstrap-Seeds noch zu verifizieren sei, ist mit
  diesem Erratum superseded. Tatsächlicher Stand: `aa_gpu`-Manifeste binden die
  `AA05`-Seeds (`0xAA052026`/`0xAA052126`) an Set und Index; der Contractfix ist
  abgeschlossen, `aggregation_contract_ready=true`. `live_execution_authorized=false`
  bleibt bewusst gesetzt: Das sperrt die echte Live-/GPU-Ausführung und ist kein offener
  Contractfix.
- Dashboard-Evidenz ist präzisiert: Vor der finalen Finite-/Cleanup-Härtung bestand eine
  autorisierte HTTP-Verifikation mit `13/13`; danach liefen `3/3` reine Finite-/Cleanup-
  Units. Der letzte `16`-er HTTP-Scope war wegen Sandbox-/Usage-Limit nicht wiederholbar.
  Es wird kein grüner finaler HTTP-Gesamtlauf behauptet. Die Offline-Historie bleibt
  unverändert: fünf Runs plus ein idempotenter Replay.
- Geändert wurden in diesem Korrekturpass nur `docs/PHASE1_MATMUL_SPEC.md`,
  `docs/H0_ADVERSARIAL_REVIEW.md` und dieses Journal. Keine Codeänderung, kein Test,
  kein Prozess, kein Atlas-Aufruf, kein Download und keine Installation.
- Vor diesem EOF-Append tatsächlich gemessen: Journal SHA-256
  `6ec6c28f14f90ca4722ffbf4c1a174d879dd93f72da38a900710aaf7a5dcd9e8`, `28872` Bytes,
  `387` Zeilen. Der Post-Append-Hash wird nur im Abschlussbericht genannt und nicht
  erneut in das Journal geschrieben.

## 2026-08-19 — Erratum zur Hash-Transkription

- Korrektur des unmittelbar vorherigen Erratum-Eintrags: Der dort angegebene Vorher-
  Hash enthielt einen Transkriptionsfehler. Tatsächlich vor dem Append gemessen waren
  SHA-256 `6ec6c28f14f90ca472fbf4c1a174d879dd93f72da38a900710aaf7a5dcd9e8`, `28872`
  Bytes und `387` Zeilen. Der Korrekturhinweis ändert keine Projekt- oder Testdaten.
- Vor diesem weiteren EOF-Append tatsächlich gemessen: Journal SHA-256
  `6fa51c5d318112e95dc778ee19dee739b815d6ae33b2c272ee275d0d06ee5d54`, `30389` Bytes,
  `409` Zeilen. Der Post-Append-Hash wird nur im Abschlussbericht genannt und nicht
  erneut in das Journal geschrieben.

## 2026-08-19 — Finaler Offline-Pre-Live-Adapter und dokumentarischer Abschluss

### Entscheidung und geschlossener Adapter-Scope

- Der Offline-Pre-Live-Adapter ist **GO**. Ein echter MLX-/GPU-Lauf bleibt **NO-GO**,
  bis er separat angekündigt, vom Nutzer freigegeben und auf dem Zielgerät validiert
  wird. `mlx-run` bleibt `EXIT_MLX_LOCKED=78`/`not_released`; beim Lockpfad werden Runner
  und Worker nicht importiert. Es wurden keine Modelle, Downloads, Installationen oder
  Custom-Metal-Kernels freigegeben oder verwendet.
- Das finale Adapterreview schloss die vollständige Evidence-Bindung über
  `source_evidence_sha256`, vollständigen Result-Hash, Manifest-Hash, versionierten
  Projection-Hash und genau ein deklaratives Projection-Artifact. Zusätzlich geschlossen
  wurden der exakte Adaptervertrag und die Correctness-Verlinkung, der read-only Storage-
  Verifier samt Child-Rows, positive Timingwerte, Warmup-/Ratio-Rekonstruktion,
  `measured_at > 0` sowie Median-/Probe-Bindung. Im abgegrenzten finalen Offline-Adapter-
  Scope bestehen keine offenen P0/P1/P2. Das Live-Gate bleibt eine separate Hardware-
  und Nutzerentscheidung, keine Adapterlücke.

### Fehler, Ursachen und erfolgreiche Lösungen

- Die Adapteraufgabe wurde wegen ausbleibender Statusmeldung zweimal kontrolliert
  unterbrochen; der jeweils sichtbare Teilstand wurde anschließend gezielt triagiert,
  ohne den Scope zu erweitern.
- Beim ersten Adapterabschluss sank der fokussierte Fehlerstand über die Korrekturrunden
  `7 → 6 → 5 → 2`. Zu den konkreten Ursachen gehörten die inkonsistente Feldbenennung
  `probes` versus `probe_raw` und ein Einrückungsfehler. Die Lösung war eine exakte
  Bindung an die erzeugte Evidence-Form sowie die lokale Einrückungskorrektur; danach
  waren alle Adapterbefunde geschlossen.
- Beim finalen DB-Reporting wurde `query_only` zunächst erst nach dem Schließen der
  Verbindung abgefragt. Ursache war die falsche Lebenszyklusreihenfolge im reinen
  Reportingpfad. Der unveränderte Read-only-Check wurde vor `close` wiederholt und
  bestätigte erfolgreich `query_only=1`; es entstand keine Datenänderung.

### Definitive Offline-Test- und Messwerte

- Hauptsuite ohne Dashboard: `177` bestanden, `3` Windows-Skips, `12` Subtests; Wall
  `26.034290 s`, Total User/System `23.373336/1.227233 s`, Self-Peak-RSS
  `15,499,264 B`, Child-Peak-RSS `74,186,752 B`.
- Socketfreie Dashboard-Prüfung: `4/4` plus `3` Setup-Subbranches; Wall `0.002041 s`,
  RSS `31,260,672 B`. Die frühere autorisierte Dashboard-HTTP-Prüfung bestand `13/13`.
  Der spätere `16`-er HTTP-Scope wurde nach der Finite-/Cleanup-Härtung wegen Sandbox-/
  Usage-Limit nicht final wiederholt; ein grüner finaler `16`-er HTTP-Lauf wird nicht
  behauptet.
- Dritte und finale Offline-Control-Generation:

  | Lauf | Wall | Exit | Ergebnis |
  |---|---:|---:|---|
  | slow | `0.191745 s` | `10` | `regression` |
  | known | `0.156192 s` | `0` | nur synthetisch |
  | wrong | `0.157021 s` | `10` | `correctness` |
  | missing | `0.157268 s` | `10` | `missing` |
  | exit70 | `0.145334 s` | `10` | `worker_exit` |
  | replay | `0.159542 s` | — | `idempotent` |

  Die Sequenz dauerte `0.967681 s`; Self-Peak-RSS `16,334,848 B`, Child-Peak-RSS
  `28,819,456 B`. Provenance: `5745e93f…39d57`; Replay-Bundle:
  `6ae4a453…b7335`.
- Finale DB: `15` Runs in drei Generationen zu je fünf Controls; jeder Run enthält
  genau ein verifiziertes `common_result`, und die älteren `10` Runs blieben unverändert.
  Snapshot `source_revision=3b70324f…ab658d`, `id=512934c9…b5b52`, `run_count=15`,
  nicht abgeschnitten; DB-Größe `229,376 B`, Datei `0600`, Verzeichnis `0700`,
  `query_only=1`. Sämtliche Werte sind Offline-Control-Evidenz, keine reale H0-MLX-/GPU-
  Performance-, Correctness-, Memory- oder Safety-Messung.

### Dokumentationsänderungen und Grenzen dieses Passes

- Aktualisiert wurden `PROJECT_STATUS.md`, `IMPLEMENTIERUNGSPLAN.md`,
  `docs/PHASE1A_ARCHITEKTURFREIGABE.md`, `docs/H0_ADVERSARIAL_REVIEW.md`, dieses
  append-only Journal und `experiments/README.md`. Die Phase-1-Spezifikation blieb
  unverändert, weil sie keine falsche Pending-Aussage zum abgeschlossenen A/A-Contractfix
  mehr enthält.
- `experiments/` enthält weiterhin keine MLX-/GPU-Rohdaten. Die lokale
  `.friday-data/h0.sqlite3` enthält die `15` Controls. Das read-only Dashboard kann mit
  `./.venv/bin/python -m friday_h0.cli dashboard --port 8765` ausschließlich auf
  `127.0.0.1` gestartet werden; es wird nicht behauptet, dass der Server aktuell läuft.
- Dieser Abschluss übernahm die bereits gemeldete Test- und DB-Evidenz; in diesem
  Dokumentationspass liefen keine Projekt-, Unit-, Worker-, Dashboard-, MLX-, GPU- oder
  Modelltests und kein ProjectAtlas-Refresh. Es erfolgten nur eng begrenzte Dokument-/
  Hashprüfungen. ProjectAtlas bleibt usage-limited; der Project-Friday-Root ist kein
  Git-Worktree, der letzte bekannte verschachtelte ProjectAtlas-Status war sauber.
- Vor diesem EOF-Append tatsächlich gemessen: Journal SHA-256
  `bd213e84aa6a5ab5fdc7895b8f359619075934a437a3080bb50f30191bc68a16`, `31060` Bytes,
  `420` Zeilen. Der Post-Append-Hash wird nur im Abschlussbericht genannt und nicht erneut
  in das Journal geschrieben, um Append-Rekursion zu vermeiden.

## 2026-08-20 — Live-H0-Freigabe, fail-closed Canary und Security-Stop

### Entscheidung und Scope

- Der Nutzer gab den zuvor angekündigten echten lokalen H0-MLX-Lauf frei. Die zugleich
  genannte Präferenz für Qwen 3.8 27B gilt für einen späteren echten lokalen Modelltest;
  Qwen wurde weder heruntergeladen noch verwendet. Die Freigabe wird nicht nachträglich
  auf „nur ein Canary“ verengt: Sol begrenzte die operative Ausführung anschließend aus
  Sicherheits- und Wissenschaftsgründen selbst auf genau einen `eager_baseline`-Canary
  und stoppte nach dessen **NO-GO** vor `aa_gpu`.
- Implementiert sind `run_mlx` und der statische CLI-Schalter `mlx-run --execute`.
  Ohne `--execute` bleibt Exit `78`/`not_released` vor Runner-, Worker-, Benchmark- und
  MLX-Import. Es gab keine freie Mode-/Set-/Index-/Seed-/Shape-/Backendwahl.
- Der Offline-Pre-Live-Adapter bleibt **GO**. Der Canary ist **NO-GO** und keine H0-
  Hardwaremessung. Weitere Live-Ausführung ist bis zur Sicherheits-/Architekturentscheidung
  über den minimalen Launcher-Fix gesperrt: **AWAITING USER APPROVAL**.
- Dieser Dokumentationspass änderte keine Source-, Test-, Spezifikations- oder DB-Datei,
  führte keinen Worker-, MLX-, GPU-, Modell- oder Testlauf aus und installierte bzw.
  lud nichts herunter. ProjectAtlas wurde zuerst mit dem fokussierten Kontext verwendet;
  Runtime `0.4.5-rc1`, Session-Brief `586 Dateien`, `261 Ordner`, `846 Blocker`. Danach
  wurden nur die sechs freigegebenen Dokumente bearbeitet bzw. read-only geprüft.

### Implementierungs- und Testevidenz vor dem Canary

- Live-Pfad-Scope: `45/45 OK`, Wall `4.022908 s`, User `3.149974 s`, System
  `0.200018 s`, gemeldeter Peak-RSS `42,139,648 B`. Eine Self-/Child-Aufteilung ist
  nicht belegt und wird nicht abgeleitet.
- Cache-API-Fokusscope nach Korrektur auf die tatsächlich vorhandene MLX-Metal-
  `get_cache_memory`-API: `16/16 OK`, Wall `0.086906 s`, User `0.140900 s`, System
  `0.054489 s`, Peak-RSS `49,938,432 B`; auch hier ist keine Self-/Child-Aufteilung
  belegt.
- Aktuelle vollständige Nicht-Live-Suite ohne Dashboard: `133/133`, Wall
  `23.720160 s`, User/System `22.722187/0.559409 s`, Self-/Child-Peak-RSS
  `71,368,704/23,642,112 B`. Unabhängiger Replay desselben Scopes: `133/133`, Wall
  `23.588426 s`, User/System `22.769535/0.504137 s`, Self-/Child-Peak-RSS
  `60,342,272/23,707,648 B`. Ein echter Importguard belegte, dass dabei keine MLX-
  Matmul-/GPU-Workload lief.
- Socketfreie Dashboardprüfung: `4/4` plus `3` Setup-Subtests, Wall `0.001793 s`,
  User/System `0.001437/0.000137 s`, Self-/Child-Peak-RSS `31,457,280/0 B`; der
  Socketkonstruktor war blockiert und wurde `0`-mal aufgerufen. Die frühere autorisierte
  HTTP-Prüfung `13/13` bleibt die letzte vollständige HTTP-Evidenz. Spätere Sandbox-
  Bindefehler und der nicht erneut ausführbare `16`-er HTTP-Scope sind kein Produktfehler
  und kein erfundener grüner Endstand.
- Der historische `177`-er Offline-Scope (`3` Windows-Skips, `12` Subtests, Wall
  `26.034290 s`, Total U/S `23.373336/1.227233 s`, Self-/Child-RSS
  `15,499,264/74,186,752 B`) ist anders enumeriert und nicht als Regression oder Zuwachs
  gegenüber den aktuellen `133` Tests zu lesen.

### Preflight, Canary und gespeicherte Evidenz

- Der erste Preflight in der Sandbox endete wegen `RuntimeError: No Metal device
  available`; Ursache war der fehlende Metal-Zugriff der Sandbox. Lösung war der
  ausdrücklich autorisierte Zielgeräte-Smoke außerhalb der Sandbox. Er bestätigte MLX
  `0.32.0` und eine 1-Element-Operation in Tool-Wall `1.741108708 s`, aber keine Matmul
  und kein H0-Ergebnis. Eine spätere reine API-Prüfung bestätigte `get_cache_memory`,
  ohne diese API oder eine GPU-Workload aufzurufen.
- Canary: äußerer Wall `0.166578416 s`; Child User `0.106607 s`, Child System
  `0.040468 s`, Child-Peak-RSS `28,442,624 B`, gespeicherter Worker-RSS
  `23,150,592 B`. Äußere Self-User/System/RSS wurden nicht separat gemessen; daraus wird
  nichts abgeleitet.
- Gespeichertes Common Result: `invalid/runtime_unavailable/baseline_fallback`, Fehler
  `NumPy import unavailable: ModuleNotFoundError`. Child-Rows: ein terminales Event,
  `0` Rohsamples, `0` Correctness-Metriken, `3` Supervisor-Scalars und `1`
  `normalization_projection_v1`-Artifact. Es fehlen deshalb Performancezeiten,
  Ratio/KI, Warmup/Repetitions, Correctness sowie MLX-active/peak/cache-Memory. Kein
  `aa_gpu`, keine Promotion und keine Hardwareaussage.
- Canary-Run-ID:
  `h0-eager_baseline-characterization-0-962b15521ae3b8e6e7bbec401b949cb005a26dc31a4e44c9b19a5a7ae2d23a2f`.
  Code-Hash `246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b`,
  Spec-Hash `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`,
  Environment `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`,
  Manifest `11ac87fb704169e58ac506eda5d0549a91ad19e8ff52b43c5bb7f28e61d982c1`,
  Result `cb97e223fd26c87aa1f1e3a87e56b4c61c76c5b69e7d0420721392727e31aa02`,
  Source-Evidence `406c42b4a99f72703b9623fd8ba5e5c0e68c46495f5a7bd0db1cef1674e0499d`,
  Projection `d9071855d3b1dc6318aa8c832c66c368314ef9ce4ff790911dd4a96939fdaf24`,
  Status-Event-Payload `23a5775546b8d90aa66627f6a0fc9ee8718c66acbf899fe52fc619f6d20cd33d`
  und Bundle `1de0c11763c38462420bb74277d8018b2db1517f9eb17e234938b27681a8c41b`.
  `stdout` und `stderr` waren leer und tragen jeweils SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- DB nach dem Canary: `16` Runs (`15` unveränderte Offline-Controls plus Canary).
  Zwei read-only Dashboard-Snapshots waren stabil:
  `snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
  `source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
  `run_count=16`, `available_count=16`, `returned_count=16`, `truncated=false`,
  `query_only=1`. Die lokale UI kann damit die Historie ohne Schreibzugriff darstellen;
  der Serverbetrieb wird nicht behauptet.

### Fehler, Ursachen und erfolgreiche Lösungen

- Hauptursache des Canary-NO-GO: `Path(sys.executable).resolve()` wandelt den lexikalischen
  venv-Launcher `/Users/tobiasburandt/Project_Friday/.venv/bin/python` in den
  Basisinterpreter `/opt/homebrew/Cellar/python@3.12/3.12.13_2/Frameworks/Python.framework/Versions/3.12/bin/python3.12`
  um (`same=false`). Die bereinigte Worker-Umgebung verliert dadurch die venv-
  Paketsuche. Erfolgreiche Handhabung war fail-closed Persistenz und Stopp vor A/A.
- Minimaler Lösungsvorschlag: den fest erwarteten absoluten, lexikalischen Launcher an
  `Popen` übergeben und Launcher, Parent sowie aufgelöstes Ziel unmittelbar vor/nach
  Spawn über Owner, Modus, Typ, Device und Inode prüfen. Restgrenze: Python-
  `Popen(path)` ist nicht fd-gebunden; ein schreibberechtigter Angreifer könnte zwischen
  Prüfung und Exec austauschen. Vollständige fd-Bindung würde Helper/`fexecve` und eine
  neue Architekturentscheidung verlangen. Deshalb keine stille Änderung:
  **AWAITING USER APPROVAL**.
- Ein erster MetaPath-Importguard meldete `find_spec("mlx")`-Metadatenprobes als falschen
  MLX-Import. Ursache war die zu breite Beobachtung; Lösung war ein Guard auf tatsächliche
  `builtins.__import__`-/`importlib.import_module`-Aufrufe. Danach liefen `133/133` ohne
  MLX-Import.
- Eine historische Spec-Assertion gehörte zu einer früheren Provenienzgeneration.
  Lösung: alte und aktuelle Generation/Hashes getrennt ausweisen, nicht historische
  Evidenz umschreiben.
- Beim read-only Dashboard-Detail wurde zunächst der UI-Missing-Wrapper statt dessen
  `run_id.value` an SQLite gebunden; SQLite meldete `ProgrammingError` für einen Dict-
  Parameter. Lösung: den geschlossenen Wrapper auslesen und ausschließlich den Stringwert
  übergeben; der nächste Detail-Readback lieferte Status `200`.
- Ein früherer Reportingversuch fragte `query_only` nach dem Schließen der Verbindung ab.
  Ursache war die Reihenfolge; der read-only Retry ermittelte `query_only=1` vor `close`.
- Der erste kombinierte Dokumentpatch dieses Passes wurde atomar abgelehnt, weil ein
  Endkontext im Review nicht exakt übereinstimmte. Es wurde nichts teilweise verändert;
  Lösung waren kleine, dateispezifische `apply_patch`-Änderungen.

### Dokumentänderungen und Provenienz

- Aktualisiert: `PROJECT_STATUS.md`, `IMPLEMENTIERUNGSPLAN.md`,
  `docs/PHASE1A_ARCHITEKTURFREIGABE.md`, `docs/H0_ADVERSARIAL_REVIEW.md`,
  `experiments/README.md`; dieses Journal wurde ausschließlich am EOF erweitert.
- Code und `docs/PHASE1_MATMUL_SPEC.md` blieben eingefroren. Die dokumentierten Freeze-
  Hashes sind Code `246eb77f…fb333b` und Spec `a713d633…b47bac`; die abschließende
  read-only Prüfung wird im Übergabebericht genannt. Der Project-Friday-Root ist kein
  Git-Worktree; ProjectAtlas selbst wurde nicht verändert.
- Vor diesem EOF-Append tatsächlich gemessen: Journal SHA-256
  `51e5896a4e9bd7572d0256e8a88a75e6a50716f4d6070bcfa8ddce69490dce4f`, `36,361`
  Bytes, `505` Zeilen. Der Post-Append-Hash wird nur im Abschlussbericht genannt, damit
  keine Hash-Rekursion entsteht.

### Abschlussverifikation dieses Dokumentationspasses

- Der erste read-only Provenienz-Readback behandelte das `Provenance`-Dataclassobjekt
  irrtümlich wie ein Dictionary und endete mit `TypeError: 'Provenance' object is not
  subscriptable`. Die korrigierte Attributabfrage bestätigte unverändert Code-SHA-256
  `246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b` und Spec-SHA-256
  `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`.
- Der Snapshot-Readback fragte zunächst einen nicht zur Snapshot-Hülle gehörenden Key
  `query_only` ab und endete nach erfolgreicher Ausgabe der übrigen Felder mit `KeyError`.
  Die verifizierten Snapshotwerte blieben stabil: ID `aaddbae8…019b`, Source-Revision
  `f5e2d328…ee58`, `16` Runs, `16` zurückgegeben, nicht abgeschnitten. `query_only=1`
  ist die separat erhobene Dashboard-Verbindungsevidenz; eine nackte SQLite-`mode=ro`-
  Kontrollverbindung meldete erwartungsgemäß nicht denselben zusätzlichen PRAGMA-Zustand
  und wird nicht als Dashboardevidenz ausgegeben.
- `git -C ProjectAtlas status --short` blieb leer. Vor diesem zweiten EOF-Append:
  Journal SHA-256 `bae8858ef81f7e15933eb166fa00c8234fd71e7ecd5e3df9847af392d6b1b40c`,
  `45,483` Bytes, `637` Zeilen. Der finale Hash folgt ausschließlich im Abschlussbericht.

## P2-Int64-Grenzen Aggregation (2026-08-20)

- ProjectAtlas-Orientierung: `friday_h0/aggregation.py` als Implementierungsquelle,
  `tests/test_aggregation.py` als gezielte Testquelle; ProjectAtlas-Runtime `0.4.5-rc1`,
  Index verfügbar. ProjectAtlas selbst blieb unverändert.
- Änderung: `_finite` weist Integer vor Float-Konvertierung außerhalb
  `0..(2**63-1)` zurück; damit werden `2**63`, negative Integer und `bool` für
  integer-valued Timing/Metric-Felder fail-closed behandelt. Endliche Float-Metriken
  (z. B. Ratio/CV) behalten die bisherige Prüfung. `measured_at_ns` akzeptiert nun
  ebenfalls exakt `1..(2**63-1)`.
- Messwerte: `MAX_INT64=9223372036854775807`, obere Ablehnung `9223372036854775808`,
  untere Ablehnung `-1`; Float-Kontrolle `1.0e300` bleibt zulässig. Der neue Aggregations-
  Grenztest bestätigte Maximum akzeptiert sowie obere Grenze, negative Grenze und `bool`
  abgelehnt; Maximum für `total_elapsed_ns` und `measured_at_ns` aggregiert als `h0_valid`.
- Verifikation: `py_compile` für Aggregation/Test erfolgreich; gezielter
  `tests.test_aggregation.AggregationTests.test_integer_boundaries_match_runner_worker_without_capping_float_metrics`
  erfolgreich (`1/1`, `1.548 s`); relevante Runner-Paritätstests erfolgreich (`2/2`, `0.134 s`)
  ohne MLX-Ausführung. Der vollständige Aggregationslauf wurde nach `30.2 s` mit fortlaufender
  Ausgabe begrenzt; kein Fehler-Traceback wurde beobachtet.

## P1-Int64-Grenzen Runner/Worker (2026-08-20)

- Ursache: Der Runner validierte `memory[].measured_at_ns` nur als positiven Python-
  Integer und ließ dadurch `2**63` passieren; die Aggregation begrenzte bereits auf
  den signierten Int64-Bereich. Dadurch war die Protokoll-Parität an der oberen Grenze
  nicht vollständig abgesichert.
- Lösung: Runner-Prüfung an `MEMORY_MAX_INT` gebunden und in beiden Runner-
  Validierungsschichten auf exakt `1..(2**63-1)` mit expliziter `bool`-Ablehnung
  vereinheitlicht. Relevante Boundary-Tests ergänzen Maximum-Akzeptanz und Ablehnung
  von `2**63`, `0`, `-1`, `bool`; der Worker-Helfer `_diagnostic_positive_int` wird
  mit denselben vier Ablehnungen und `MAX_INT64` geprüft. Keine MLX-Ausführung,
  keine Downloads und keine Datenbankänderung.
- Messung/Verifikation: `MAX_INT64=9223372036854775807`, `2**63=9223372036854775808`;
  `py_compile` für Runner/Aggregation/Worker und relevante Tests erfolgreich. Gezielt
  `3/3` Tests erfolgreich, `failures=0`, `errors=0`, `elapsed_s=1.798791`,
  `maxrss=44236800` Bytes. Ein erster Lauf referenzierte einen nicht vorhandenen
  Testnamen (`AttributeError`); Ursache war nur die falsche Testauswahl, gelöst durch
  den tatsächlich vorhandenen Testnamen. `/usr/bin/time -l` meldete zusätzlich eine
  macOS-`sysctl`-Berechtigungswarnung; der exakte RSS-Wert stammt daher aus
  `resource.getrusage`.
- Nachmessung nach Ergänzung der `0`-Ablehnung im Aggregationstest: weiterhin `3/3`
  erfolgreich, `failures=0`, `errors=0`, `elapsed_s=1.935360`, `maxrss=44269568` Bytes.

## Finaler Contract, Offline-Evidenz und Run21-NO-GO (2026-08-20)

### Orientierung und finaler Vertrag

- Vor dem Dokumentationspass wurde ProjectAtlas verwendet: Runtime `0.4.5-rc1`,
  Project-Friday-Index verfügbar. Dieser Pass änderte weder ProjectAtlas noch Code,
  Tests, Datenbank, Live-/GPU-Zustand oder Installationen.
- Der finale Contract-Stand ist mit Core `175/0`, Dashboard `4/4` und `0` offline MLX-
  Imports belegt. Provenienz `575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`;
  Code `aae3245e…` (im übergebenen Satz nur als Präfix), Spec
  `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

### Run21 und wissenschaftliche Einordnung

- Run21 wurde exakt einmal gestartet: Exit `10`, Wall `1.14 s`, User `0.98 s`, System
  `0.16 s`, Peak-RSS `369,573,888 B`. Der Persistenzbefund lautet exakt
  `invalid/invalid/baseline_fallback`; die Diagnose ist `warmup_unstable` nach `16`.
- Agent-Statistik: `all` Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR `582,260.25 ns`;
  `last5` Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`, Minimum
  `2,067,916 ns`, Maximum `2,677,583 ns`, Stabilität `false`.
- Persistenzzählung: Rohsamples `0`, Correctness-Zeilen `0`, Scalars `3`, Artifact `1`.
  Es gab kein `aa_gpu`; Run21 ist weder Performance- noch Correctness-Aussage.
- DB vor Run20: `c9a521…`; Run21-DB: `420b7c…`; Bundle `027908…`; Result `ac4a82…`;
  Payload `cd409d…`; Evidence `837841…`. Diese Werte wurden nur in verkürzter Form
  übergeben; die Ellipsen markieren fehlende Hash-Suffixe und werden nicht ergänzt.
- Wissenschaftliche Entscheidung: Der eingefrorene Vertrag `8 → maximal 16` Warmups
  und letzte fünf Werte innerhalb `±5 %` entspricht dem Code. Kein
  Implementierungsdefekt wurde gefunden. Die Ursache bleibt OS-/Thermik-/MLX-unklar.
  Es gab keine post-hoc Threshold-Änderung und keinen Retry.

### Dashboard- und Harness-Abgrenzung

- Die statische Prüfung von `friday_h0/dashboard.py` bestätigt: read-only SQLite-Open,
  automatische Run-Historienabfrage und Statusübernahme durch `_status`; ein gespeicherter
  `invalid`-Status bleibt damit sichtbar. In diesem Pass wurden Dashboard-Server und
  Sockets nicht gestartet.
- Der `python`-Aliasfehler und der Dashboard-`self.path`-Fehler sind Harnessfehler, nicht
  Projektfehler. Konvergenzregel: Ein Harnessbefund wird erst nach reproduzierbarer
  Wiederholung und unabhängigem Readback bewertet; er darf weder wissenschaftliche
  Schwellen nachträglich ändern noch einen Canary-Retry begründen.

## H0.1 Design A — separater stdlib-only Core (2026-08-20)

### Orientierung und Architekturentscheidung

- ProjectAtlas wurde vor Quellarbeit verwendet. Der erste Session-Brief meldete wegen
  `58` bereits vorhandener neuer Gradle-Fixture-Pfade `refresh_required`; der empfohlene
  lokale One-shot-Watch wurde ausgeführt (`613` Kandidaten, `593` indexierte Dateien,
  `103` aktualisierte Summaries). Danach war der Index verfügbar; die erste Empfehlung
  `tests/test_worker.py` wurde per File-Summary eingeordnet. ProjectAtlas-Code wurde nicht
  geändert. Die weiterhin unversionierten `.gradle/`-Verzeichnisse unter den Groovy-/
  Kotlin-Fixtures waren bereits Anlass des initialen Refresh-Hinweises und blieben
  unangetastet.
- H0.1 ist als neue, eigenständige Studie `paced_trajectory_design_a` implementiert.
  Keine Datei unter `friday_h0/`, keine H0-Spezifikation, kein H0-Ergebnis und keine
  SQLite-v1-Datenbank wurden verändert. H0.1 kann weder H0 reklassifizieren noch eine
  Promotion auslösen; alle Resultate tragen `no_h0_conclusion`.
- Vor Daten wurde `sha256_fisher_yates_v1` mit SHA-256-Counter-RNG und Rejection
  Sampling statt PCG64 registriert. Die sechs disjunkten Sessions `C0,V0,C1,V1,C2,V2`
  besitzen feste Int64-Seeds, materialisierte Schedules und gepinnte Schedule-Hashes.
- Der neue Core `friday_h01/` verwendet ausschließlich die Python-Standardbibliothek.
  Manifest, Trace und Resultat haben geschlossene exakte Schemas, kanonische JSON-/
  SHA-256-Bindungen, explizite signed-Int64- und Bool-Ablehnung, vollständige Parent-/
  Fixture-/Environment-Provenienz und fail-closed Telemetrie-XOR-Felder.
- Der Messvertrag ist fest: `32` aufgezeichnete Burn-in-Samples, mindestens
  `20,000,000,000 ns` beobachteter Cooldown nach exakt gleicher Anforderung, `80`
  Main-Samples in `20` Viererblöcken mit je zweimal `50 ms` und zweimal `750 ms`.
  Es gibt keinen adaptiven Stopp und keine Ausreißerlöschung.
- Die registrierte Analyse verwendet Log-Dauern, pacing-stratifizierte Residuen,
  Theil-Sen, maximale Median-Changepoints für Splits `8..72`, Spearman-ACF Lags
  `1..4` plus ESS, Pacing-Effekt, Tail-Ratio sowie `8192` deterministische zirkuläre
  Moving-Block-Bootstraps mit Blocklänge `8`. Wissenschaftliche Gate-Fehler ergeben
  `complete_unresolved`; Vertragsfehler ergeben getrennt `invalid`.

### Dateien und reproduzierbare Tests

- Neu: `docs/H01_PACED_TRAJECTORY_SPEC.md`, `friday_h01/__init__.py`,
  `friday_h01/constants.py`, `friday_h01/canonical.py`, `friday_h01/schedule.py`,
  `friday_h01/protocol.py`, `friday_h01/analysis.py` sowie die drei fokussierten
  Module `tests/test_h01_schedule.py`, `tests/test_h01_protocol.py` und
  `tests/test_h01_analysis.py`. Storage, Dashboard, Worker und Live-Ausführung sind
  nicht Bestandteil dieser Phase.
- Die Tests pinnen alle sechs Schedule-Hashes und prüfen Determinismus, Blockbalance,
  Schemaabschluss, Delete/Duplicate/Reorder/Timestamp/Digest-Mutationen, Provenienz,
  Cooldown, Telemetrie, Bool/Int64/Nichtendlichkeit, Bootstrap-/Decision-Replay sowie
  stabile, `6 %` Drift-, `7 %` Step-, hochpersistente AR(1)-, `30 %` Spike- und
  `4 %` Pacing-Fixtures.
- Erste Syntaxmessung: Exit `0`; Wall `0.24 s`, User `0.09 s`, System `0.02 s`.
  Erster Importguard-Testlauf: `15` Tests, `25` Subtests, `1` Failure, `1` Error,
  `0` Skips, interne Laufzeit `1.5502199169714004 s`, Self-User `1.622630 s`,
  Self-System `0.028783 s`, Child-User/System jeweils `0`, Peak-RSS `30,982,144 B`,
  NumPy/MLX-Import-/Discovery-Versuche `0`, geladene blockierte Module danach `0`.
- Ursachen: `build_trace` wies den Bool korrekt zurück, ließ dabei jedoch den internen
  `CanonicalError` statt des öffentlichen `ProtocolError` passieren. Die erste AR(1)-
  Fixture war nicht persistent genug, um neben dem ACF- auch das ESS-Gate zu reißen.
  Lösung in genau einer Fixrunde: öffentliche Exception-Übersetzung und eine eindeutig
  persistente, deterministische AR(1)-Fixture (`rho=0.97`). Schwellen oder Analysecode
  wurden nicht post-hoc an das Ergebnis angepasst.
- Finale Syntaxmessung: Exit `0`; Wall `0.06 s`, User `0.02 s`, System `0.01 s`.
  Finaler harter Importguard-Lauf: Exit `0`, `15/15` Tests und `25/25` Subtests,
  Failures `0`, Errors `0`, Skips `0`, Expected-Failures `0`, Unexpected-Successes
  `0`; interne Laufzeit `1.5073861659038812 s`, externe Wall `1.57 s`, User
  `1.53 s`, System `0.01 s`, Self-User `1.532909 s`, Self-System `0.017363 s`,
  Child-User/System jeweils `0`, Peak-RSS `22,986,752 B`. Import-/Discovery-Versuche
  für NumPy oder MLX `0`; entsprechende Einträge in `sys.modules` vorher/nachher `0`.
  `/usr/bin/time -lp` meldete in der Sandbox zusätzlich die bekannte
  `sysctl kern.clockrate`-Berechtigungswarnung; Exit, interne Zeiten und RSS stammen
  davon unabhängig aus Prozessstatus und `resource.getrusage`.
- Es gab keine Downloads, Installationen, Modellgewichte, Custom-Kernels, GPU-/MLX-
  Ausführung, Live-Messung, Produktions-DB-Schreibzugriffe oder Dashboard-/Socket-
  Aktivität.

### Systematische Bool/Integer-Härtung nach kritischem Review

- Ein read-only Nachreview des zunächst grünen Cores zeigte eine Python-spezifische
  Vertragslücke: Dictionary-Gleichheit behandelt `True == 1` und `False == 0` als
  wahr. `validate_schedule` verglich einzelne numerische Schedule-Felder noch nicht
  vor dem vollständigen Materialisierungsvergleich; außerdem waren die
  `schema_version`-Felder von Manifest, Trace und Resultat nur über feste Gleichheit
  gebunden. Damit konnten typfalsche Bool-Werte theoretisch durch eine Gleichheit
  maskiert werden. Der zunächst grüne Bool-Test traf andere Felder und belegte diese
  Lücke daher nicht. Der Befund wurde vor weiterer Änderung als NO-GO gemeldet.
- Nach ausdrücklich autorisiertem Strategiewechsel wurde ProjectAtlas erneut zuerst
  aktualisiert: One-shot-Watch mit `624` Textkandidaten, `604` indexierten Dateien,
  `20` Skips, `399` Strukturkandidaten, `105` Summaries sowie `7` neu geparsten und
  `465` unveränderten Symbolquellen. Der anschließende Brief meldete den Index als
  verfügbar; die erste Empfehlung `tests/test_h01_analysis.py` wurde per File-Summary
  gelesen.
- Lösung: `friday_h01/canonical.py` zentralisiert nun neben `int64` auch
  `nonnegative_int64`, `positive_int64` und `exact_int64`. Alle Helfer lehnen `bool`
  ab, bevor Bereich oder registrierter Wert verglichen werden. Schedule prüft
  `schema_version`, Seed und alle Sample-/Phase-/Block-/Position-/Gap-Integer vor dem
  Materialisierungsvergleich. Manifest prüft Schema, Session, eingebettete Schedule
  und sämtliche Budgets vor Run-ID-Rekonstruktion. Trace prüft Schema, Cooldown und
  alle Sample-/Gap-/Start-/Duration-Integer vor Bindungsvergleichen. Result prüft
  Schema, Sample-Bilanz, Changepoint-Split und Bootstrap-Budgets vor Decision-Hash.
  Öffentliche Fehler bewahren den exakten betroffenen Feldpfad.
- Die Absicherung ist metamorph statt handverlesen: Aus jeweils einem gültigen
  Schedule, Manifest, Trace und Resultat wurden alle integerwertigen Leaves rekursiv
  bestimmt. Abhängige Schedule-, Run-ID- und Decision-Hashes wurden nach jeder
  Mutation neu gebunden, damit kein Hash-Mismatch die Typprüfung maskiert. Exakt
  `562 + 576 + 787 + 10 = 1,935` Leaves wurden einzeln zu `True` mutiert und mussten
  an der jeweiligen öffentlichen Fehlergrenze mit korrektem Pfad scheitern. Zusätzlich
  wurden repräsentative `False`- und `1.0`-Mutationen für alle vier Objekttypen geprüft.
- Finale Syntaxprüfung nach der Härtung: Exit `0`, Wall `0.05 s`, User `0.03 s`,
  System `0.01 s`. Ein einziger finaler H0.1-Importguard-Lauf: Exit `0`, `19/19`
  Tests, `1,960/1,960` Subtests, Failures `0`, Errors `0`, Skips `0`,
  Expected-Failures `0`, Unexpected-Successes `0`; interne Laufzeit
  `7.276949458988383 s`, externe Wall `7.36 s`, User `7.27 s`, System `0.03 s`,
  Self-User `7.274503 s`, Self-System `0.037930 s`, Child-User/System jeweils `0`,
  Peak-RSS `23,642,112 B`. NumPy/MLX-Import-/Discovery-Versuche und entsprechende
  `sys.modules`-Einträge blieben jeweils `0`.
- Der geforderte statische Abschluss-Sweep fand per Textsuche nur den fachlich
  korrekten Float-Varianzvergleich `left_ss/right_ss == 0.0`. Ein zusätzlicher
  AST-Sweep über `Eq`/`NotEq` mit echten Integer-Literalen (`type(value) is int`,
  Bool ausgeschlossen) ergab im gesamten `friday_h01`-Package exakt `[]`.
- H0, Storage, Dashboard, Live-/GPU-/MLX-Ausführung, Modelle, Installationen und
  Produktionsdatenbanken blieben weiterhin unberührt.

## H0.1 Design A v2 — replizierter Engineering-Envelope (2026-08-20)

### V1-NO-GO und vorregistrierter Strategiewechsel

- Der zuvor technisch grüne v1-Core wird wissenschaftlich als **NO-GO** verworfen.
  Ursachen sind keine gemessenen Grenzwertverletzungen, sondern drei vor Live-Daten
  erkannte Designfehler: Ein geliefertes Resultat konnte bei gemeinsam veränderten
  Metriken und Decision-Hash ohne vollständige Neuberechnung aus der Trace plausibel
  bleiben; die Trendachse verwendete Sampleindex statt reale Zeit; außerdem waren
  Bootstrap-/Einzelsession-Interpretationen für den deterministischen Engineering-
  Envelope nicht gerechtfertigt. Es existieren keine H0.1-v1-Live-Daten. Schwellen
  wurden nicht anhand von Daten verändert.
- Vor den v2-Änderungen wurde ProjectAtlas verwendet. Der One-shot-Watch bestätigte
  einen frischen Index (`0` geänderte Kandidaten/Dateien/Summaries); der fokussierte
  Brief meldete `624` Dateien, `294` Ordner und einen verfügbaren Index. Die erste
  Empfehlung `tests/test_h01_analysis.py` wurde per Source-Summary gelesen.
- `docs/H01_PACED_TRAJECTORY_SPEC.md` registriert v2 vor jeder Live-Erhebung als
  deterministic replicated engineering envelope. C/V sind nur symmetrische,
  vorregistrierte Replikationslabels. Sessionstatus ist ausschließlich
  `h01_session_complete` oder `h01_invalid`; eine gültige Session bleibt auch bei
  Gate-Fail vollständig und charakterisiert.
- Exakte terminale Study-Statusmenge: `h01_stationarity_supported` nur wenn alle
  sechs vollständigen Sessions jedes Gate bestehen; `h01_complete_unresolved` bei
  mindestens einem Gate-Fail; `h01_invalid` bei jedem Contract-, Safety-,
  Correctness-, Replay-, Provenienz-, Reihenfolge- oder Vollständigkeitsfehler.
  Es gibt keine C-/V-Sonderauswertung und keine weitere Zwischenklasse. H0-
  Reklassifikation und Promotion bleiben stets ausgeschlossen.

### Implementierter v2-Vertrag

- Schema wurde wegen des absichtlichen Breaking Changes auf `2` erhöht. Manifeste
  binden nun zusätzlich Study-Spec- und Code-SHA-256; Run-IDs binden weiterhin
  Session, materialisierten Schedule, Budgets, Gates, Environment, vollständige
  Fixture und H0-Parent-Lineage.
- Schedule-/Trace-Feld `gap_ns` wurde durch `requested_gap_ns` ersetzt. Jedes Sample
  bindet außerdem `gap_start_ns`, `gap_end_ns`, `start_ns` und `duration_ns`.
  `actual_gap_ns` wird nur als Differenz abgeleitet; `gap_end_ns == start_ns`, exakte
  Kontinuität zum vorigen Sample bzw. nach Cooldown sowie strikt monotone Starts
  werden fail-closed geprüft.
- Vor Live-Daten ist `MAX_GAP_OVERSHOOT_NS = 250,000,000` registriert. Actual Gap
  muss einschließlich zwischen Requested Gap und Requested Gap plus `250 ms` liegen.
  Extra Pause, Rebound, falsche Gap-Grenze oder Timestamp machen die Session ungültig;
  Pacing-Labels gelangen nur nach bestandener Adherence in die Analyse.
- Theil-Sen verwendet jetzt die realen Main-Startzeiten in Sekunden relativ zum
  ersten Main-Sample. Effekt ist `expm1(slope_per_second * observed_span_seconds)`.
  Changepoint, ACF/ESS, Pacing-Effekt und Tail-Ratio bleiben deterministische,
  vorregistrierte Effekt-/Envelope-Metriken. Bootstrap-Code, -Konstanten, -Gates und
  -Resultfelder wurden vollständig entfernt; es werden keine p-Werte oder
  probabilistischen Konfidenzintervalle erzeugt.
- `validate_result(value, manifest, trace)` validiert zunächst die geschlossene
  Resulthülle, führt dann `analyze_trace(manifest, trace)` selbst aus und verlangt
  kanonische Bytegleichheit des vollständigen Resultats. Geteilte Metric-/Gate-/
  Status-/Decision-Co-Mutationen bei unveränderter Trace werden damit verworfen.
- Neu ist `friday_h01/study.py`: genau die Reihenfolge `C0,V0,C1,V1,C2,V2`, keine
  fehlende, doppelte, vertauschte oder selektiv gestoppte Session. Manifest, Trace
  und Resultat jeder Session werden validiert und replayt. Spec, Code, Environment,
  Fixture und Parent-Lineage müssen identisch sein. Session-, Gate-, Trace-, Result-
  und Study-Bindungen erhalten kanonische SHA-256; der Study-Decision-Hash ist exakt
  replaybar.

### Tests und Messungen

- Aktualisierte Tests decken Determinismus, reale Gap-Adherence, Cooldown-Kontinuität,
  Extra-Pause/Rebound, Timestamp-/Digestmutation, volle Result-Co-Mutation,
  Missing/Duplicate/Reorder/Mixed-Provenance/Selective-Stop, symmetrische Study-
  Statussemantik und replaybare Study-Entscheidungen ab. Die positive Fixture ist
  nicht konstant, sondern deterministisches SHA-256-Rauschen. Die Abhängigkeits-
  Negativfixture ist ein nach langer Burn-in-Phase gesampelter AR(1)-Prozess mit
  `rho=0.97` und deterministischen SHA-256-Innovationen; sie verletzt ACF und ESS.
  Drift `6 %`, Step `7 %`, Spike `30 %` und Pacing `4 %` bleiben als getrennte
  Envelope-Negativtests bestehen.
- Metamorphe signed-Integer-Absicherung wurde an v2 angepasst: `562` Schedule-,
  `575` Manifest-, `1,011` Trace-, `8` Session-Result- und `3` Study-Result-Leaves.
  Jede Integerposition wird einzeln mit `True` mutiert; abhängige Hashes werden neu
  gebunden, damit Typprüfung nicht durch einen Hash-Mismatch maskiert wird. Zusätzlich
  werden `False`, Float-Impostoren und integerwertige Bool-Feld-Impostoren geprüft.
- Syntaxprüfung: Exit `0`; Wall `0.15 s`, User `0.10 s`, System `0.02 s`.
  Einziger H0.1-v2-Importguard-Lauf: Exit `0`, `29/29` Tests, `2,189/2,189`
  Subtests, Failures `0`, Errors `0`, Skips `0`, Expected-Failures `0`,
  Unexpected-Successes `0`; interne Laufzeit `12.396627290872857 s`, externe Wall
  `12.55 s`, User `12.34 s`, System `0.06 s`, Self-User `12.345237 s`, Self-System
  `0.067497 s`, Child-User/System jeweils `0`, Peak-RSS `32,145,408 B`.
  NumPy/MLX-Import-/Discovery-Versuche und entsprechende `sys.modules`-Einträge
  vorher/nachher: jeweils `0`.
- Statische Terra-P0-Checkliste: Im Package keine Bootstrap-/`p_cp`-/alten Session-
  Status-/indexbasierten Slope-/alten `gap_ns`-Reste; genau drei Study-Terminalstatus;
  `analyze_trace`-Replay in der öffentlichen Resultvalidierung; reale Gap-Felder,
  `250 ms`-Grenze und exakte `SESSION_ORDER`-Verwendung vorhanden; keine NumPy-/MLX-
  Imports. Die Spezifikation erwähnt Bootstrap/`p_cp` ausschließlich, um deren
  Nichtverwendung normativ festzuhalten.
- Geänderte/neu hinzugefügte v2-Dateien: `docs/H01_PACED_TRAJECTORY_SPEC.md`,
  `friday_h01/__init__.py`, `friday_h01/constants.py`, `friday_h01/schedule.py`,
  `friday_h01/protocol.py`, `friday_h01/analysis.py`, neu `friday_h01/study.py`,
  `tests/test_h01_schedule.py`, `tests/test_h01_protocol.py`,
  `tests/test_h01_analysis.py` und neu `tests/test_h01_study.py`.
- Keine Datei unter `friday_h0/`, keine H0-Spezifikation, keine Datenbank, kein
  Storage/Dashboard, keine Live-/GPU-/MLX-Ausführung, keine Modelle, Downloads oder
  Installationen wurden berührt.

## H0.1 SQLite-v1 und read-only Dashboard-Slice (2026-08-20)

### Entscheidung und Implementierung

- ProjectAtlas wurde vor jeder breiten Dateiarbeit verwendet. Der initiale One-shot-
  Watch erfasste `626` Textkandidaten, indexierte `606`, übersprang `20`, erzeugte
  `105` strukturelle Summaries und meldete `254` Symbole/`1,458` Relationen. Der
  fokussierte Brief und die erste Empfehlung (`ProjectAtlas/.../schema.rs`) bestätigten
  die vorhandenen Exact-Schema-, Preflight- und Read-only-Muster. Nach den Änderungen
  erfasste der Refresh `632` Kandidaten/`612` indexierte Dateien/`20` Skips,
  `104` strukturelle Summaries sowie `6` geparste/`474` unveränderte Symbolquellen
  mit `147` Symbolen und `776` Relationen.
- Vor jeder H0.1-Persistenz wurde `docs/H01_STORAGE_DASHBOARD_SPEC.md` registriert.
  H0.1 verwendet ausschließlich die eigene, in diesem Slice noch **nicht erzeugte**
  Produktivdatei `.friday-data/h01.sqlite3`. Es gibt keinen Cross-DB-Snapshot und
  keine Änderung oder Umdeutung von H0.
- SQLite v1 ist bewusst kompakt: eine kanonische Tabelle `bundles`, zwei exakte
  Abfrageindizes und je ein `BEFORE UPDATE`-/`BEFORE DELETE`-Abbruchtrigger. Exakt
  geprüft werden `application_id = 0x48303131`, `user_version = 1`,
  `integrity_check`, relevante `sqlite_master`-DDL, `table_xinfo`, `index_list` und
  `index_xinfo`. Extra/missing/changed Column, Index oder Trigger wird fail-closed
  abgelehnt.
- Bundle-Persistenz ist eine `BEGIN IMMEDIATE`-Transaktion. Manifest, Trace, Resultat,
  H0-Lineage und das vollständige Bundle haben getrennte SHA-256-Bindungen und werden
  aus kanonischen Bytes replayt. Dieselbe ID plus dieselben Bytes ist idempotent;
  dieselbe ID plus andere Bytes ist ein Konflikt. Jede Integergrenze lehnt Bool ab,
  signed-int64 und 1-MiB-Kanonikgrenzen gelten. Paced Sessions und Studies werden vor
  Persistenz durch ihre bestehenden wissenschaftlichen Replay-Validatoren geprüft.
  Legacy-H0-Warmup-Beobachtungen bleiben eigener Kind/Status
  `legacy_h0_warmup_observation`/`legacy_observation` und können weder Stationarität,
  Promotion noch eine H0-Schlussfolgerung beanspruchen.
- `DashboardService` öffnet bei jedem Aufruf ausschließlich eine intern erzeugte
  percent-encodierte SQLite-URI mit `mode=ro` und geprüftem `query_only=1`. Snapshot
  und Detail laufen jeweils in einer einzelnen H0.1-Lesetransaktion. Revision,
  Totals, Kind-/Statuszähler und Historie sind vorhanden; Detail-Records und
  Trace-Punkte sind auf je `200` begrenzt, H0-Elternlinie wird separat ausgegeben.
  Der optionale Adapter bindet fest an `127.0.0.1`, implementiert nur GET/HEAD,
  weist schreibende/sonstige Methoden mit `405` ab und setzt CSP, No-Store,
  Nosniff, Frame-, Referrer- und Cross-Origin-Header. Kein echter Socket wurde im
  Slice gestartet.

### Fehler, Ursachen und erfolgreiche Lösungen

- Der erste PyCompile-Messwrapper lieferte nach diagnostikfreier Kindausführung
  Exit `1`, weil `/usr/bin/time -lp` im Sandboxprofil `sysctl kern.clockrate` nicht
  lesen durfte (`real 0.07 s`, User `0.04 s`, System `0.01 s`). Das war kein
  Syntaxfehler. Die abschließende direkte, unveränderte `python -m py_compile`-
  Prüfung aller H0.1-Quellen und -Tests lieferte Exit `0` ohne Diagnose.
- Zwei Testharness-Versuche stoppten jeweils **vor Test 1**: zunächst verlangte
  `unittest.discover` ein hier nicht vorhandenes `tests/__init__.py`; danach machte
  das Ersetzen der Klasse `socket.socket` durch eine Funktion den beim Import
  definierten `ssl.SSLSocket` unmöglich. Erfolgreiche Lösung: vorhandene sechs
  Testmodule explizit laden und echte Konstruktion über den CPython-Audit-Event
  `socket.__new__` zählen/blockieren. So bleibt der Socket-Typ für stdlib-Imports
  intakt, während jeder reale Konstruktionsversuch fail-closed wäre. Diese beiden
  Setup-Abbrüche führten jeweils `0` Tests aus und sind keine Testwiederholungen.

### Reproduzierbare Verifikation und Messwerte

- Der eine tatsächlich ausgeführte komplette H0.1-Lauf unter hartem
  NumPy-/MLX-Import-, `find_spec`-, `sys.modules`- und Socket-Auditguard: Exit `0`,
  `40/40` Tests, `2,214/2,214` Subtests, Failures `0`, Errors `0`, Skips `0`,
  Expected-Failures `0`, Unexpected-Successes `0`; Wall `15.393831 s`, Self-User
  `15.289372 s`, Self-System `0.063610 s`, Child-User/System jeweils `0`, Peak-RSS
  `39,141,376 B`. Socket-Konstruktionsversuche, NumPy-/MLX-Import-/Discoveryversuche
  und verbotene `sys.modules`-Einträge: jeweils exakt `0`.
- Gegenüber dem vorigen reinen H0.1-v2-Core-Verifikationslauf sind `11` Tests und
  `25` Subtests hinzugekommen. Die Wall-Zunahme beträgt `2.997204 s`, Peak-RSS plus
  `6,995,968 B`; das ist die größere Verifikationslast inklusive SQLite-/Study-
  Replay, keine Produktiv- oder Performanceaussage.
- Neue Negativbelege: Extra-/fehlende/veränderte Tabelle/Spalte/Index/Trigger;
  JSON-/Hash-Tamper; Update/Delete; ID-Konflikt; vom SQLite-Authorizer erzwungener
  atomarer Insert-Abbruch; Read-only-Schreibversuch; URI-/Missing-Path; malformed,
  non-finite, zu groß, Bool/Float/Int64-Overflow; Legacy-Stationarity/Promotion;
  Query-/Pfadgrenzen sowie POST. Datei-SHA vor/nach Read-only Storage- und
  Dashboard-Aufrufen blieb jeweils exakt gleich.
- Statischer Scope-Audit: keine NumPy-/MLX-Referenz in `friday_h01/storage.py` oder
  `dashboard.py`; Socketfläche nur im expliziten Loopback-Serverkonstruktor; keine
  `.friday-data/h01.sqlite3` vorhanden. Es wurde keine Datei unter `friday_h0/`,
  keine H0-DB, kein Live-/GPU-/MLX-Pfad, kein Modell, Download oder Installer berührt.
  Das verschachtelte ProjectAtlas-Repository enthält weiterhin zwei vorgefundene,
  nicht von diesem Slice erzeugte untracked Gradle-Caches unter Groovy/Kotlin-
  Fixtures; sie wurden weder verändert noch gelöscht.
- Neu: `docs/H01_STORAGE_DASHBOARD_SPEC.md`,
  `friday_h01/migrations/0001_initial.sql`, `friday_h01/storage.py`,
  `friday_h01/dashboard.py`, `tests/test_h01_storage.py` und
  `tests/test_h01_dashboard.py`; dieser Journaleintrag wurde append-only ergänzt.

## H0.1 Storage/UI — systematische P1-Härtung (2026-08-20)

### Auditentscheidung und geschlossene Befunde

- Der unabhängige Audit setzte den ersten Storage/UI-Stand auf **NO-GO**: Direkte
  `UPDATE`-/`DELETE`-Trigger schlossen `INSERT OR REPLACE` bei ausgeschalteten
  rekursiven Triggern nicht hinreichend; Dashboard-Snapshots zählten ungeprüfte
  Zeilen; die Revision band nur Count/Maxima; und Schema-/Pfadprüfung war nicht an
  dieselbe explizite DB-Transaktion und Dateiidentität gebunden. Als einfache P2-
  Lücken wurden beliebige HTTP-Methoden, ein Response-Bytecap, vollständige
  Schema-Drifttests und doppelt exponierte H0-Lineage dokumentiert.
- Vor Änderungen wurde ProjectAtlas erneut verwendet. Der initiale Watch war bereits
  frisch (`0` Kandidaten/Änderungen); Brief und erste Empfehlung führten zum
  ProjectAtlas-Schema-Preflight als Referenz für gebundene read-only Snapshots. Der
  abschließende Refresh erfasste `632` Textkandidaten, indexierte `612`, übersprang
  `20`, erzeugte `104` strukturelle Summaries und meldete `6` geparste/`474`
  unveränderte Symbolquellen, `177` Symbole und `1,010` Relationen.
- Der Audit fand die Lücken vor jeder H0.1-Produktivdatenbank und vor Live-Daten.
  Deshalb bleibt die bewusst separate Schema-Bezeichnung v1 erhalten; es existiert
  keine Altdatei, die still migriert oder neu interpretiert werden könnte.

### Implementierte P1-/P2-Grenzen

- Die registrierte Migration enthält nun zusätzlich `bundles_no_reinsert` als
  `BEFORE INSERT`-Trigger. Sein `EXISTS` bindet sowohl `entity_id` als auch
  `bundle_sha256`, sodass `INSERT OR REPLACE` bei gleicher ID oder kollidierendem
  Bundle-Hash vor der Konfliktauflösung scheitert. Jede Verbindung setzt und prüft
  `recursive_triggers=1`; die Anwendung prüft Idempotenz/Kollision erst innerhalb
  `BEGIN IMMEDIATE` und führt bei vorhandener ID keinen Insert aus.
- Writable Open und Persistenz beginnen `BEGIN IMMEDIATE`, bevor Schema und vorhandene
  Daten verifiziert werden. Persistenz replayt innerhalb derselben Transaktion alle
  Altzeilen, entscheidet dann Idempotenz/Kollision, replayt einen neuen Insert erneut
  und prüft die Dateiidentität unmittelbar vor Commit. Ein vorhandenes leeres oder
  inkompatibles File wird nicht synthetisiert oder repariert; nur ein vor Connect
  nachweislich fehlender, ausdrücklich gewählter Pfad erhält die registrierte v1-DDL.
- Pfadbindung verwendet keine symlinkauflösende Normalisierung des angeforderten
  Finalpfads. Direkter Eltern- und Finalpfad müssen Directory bzw. Regular File und
  keine Symlinks sein. Device, Inode, UID und Mode werden vor/nach Connect und nach
  `BEGIN` verglichen; `PRAGMA database_list` muss genau einen `main`-Eintrag auf
  dieselbe aufgelöste Dateiidentität binden. Ausschließlich die SQLite-interne
  `temp`-Zeile mit leerem Pfad ist zusätzlich erlaubt; ATTACH oder andere Einträge
  werden abgelehnt. Die dokumentierte Zusicherung beschränkt sich auf diese belegten
  Invarianten und behauptet keine universelle TOCTOU-Elimination.
- Dashboard-Snapshot und Detail laufen jetzt über `read_transaction()`: nach `BEGIN`
  folgen Datei-, Profil-, Schema- und Integritätsprüfung. Der Snapshot liest danach
  jede Zeile einschließlich aller JSON-/Teil-/Bundle-Hashes und wissenschaftlichem
  Result-Replay; erst aus vollständig verifizierten Bundles entstehen Totals,
  Status-/Kind-Zähler und Recent-Historie. Eine beschädigte Zeile verwirft den ganzen
  Snapshot. Revision ist kanonischer SHA-256 über Schema-Fingerprint sowie die strikt
  nach Row-ID geordneten Identitäten aus Row-ID, Entity-ID, Bundle-Hash und Zeit.
- Dashboard-Views verwenden explizite Feldlisten. H0-`source` wird aus Session-
  Manifest/Trace und Study-Provenienz entfernt; Study-Records sind die gebundenen
  Session-Summaries, Legacy-Records nur Index plus Observation-Hash. Die signierten
  gespeicherten Bytes bleiben unverändert und die vollständige Elternlinie steht
  separat unter `parent_h0_lineage`.
- `__getattr__` leitet jede unbekannte `do_*`-Methode auf `405` um. Fertig
  serialisierte HTML-/JSON-Responses sind vor dem ersten Header auf `1 MiB` begrenzt.
  Das exakte Schema prüft nun ausdrücklich alle `sqlite_master`-Einträge einschließlich
  Autoindizes; Negativfixtures umfassen zusätzlich View, Index, Trigger, Triggerbody
  und Autoindex-Änderungen.

### Fehler, Ursache und erfolgreiche Fixrunde

- Der erste kohärente Guard-Lauf war **NO-GO**: `38` Tests/`2,207` Subtests,
  Failures `0`, Errors `8`, Wall `13.840453 s`, Self-User `13.750191 s`, Self-System
  `0.052598 s`, Child-User/System `0`, Peak-RSS `32,620,544 B`; alle Guards blieben
  bei `0`. Alle acht Errors hatten exakt dieselbe Ursache vor fachlicher
  Storage-/Dashboard-Ausführung: Die erste `database_list`-Grenze verlangte insgesamt
  eine Zeile, diese SQLite-Laufzeit materialisiert aber neben `main` auch die lokale
  `temp`-Zeile.
- Die einzige systematische Fixrunde verlangt nun exakt einen `main`-Eintrag und
  erlaubt daneben ausschließlich `temp` mit leerem Pfad. Dadurch bleibt fremdes
  ATTACH fail-closed, ohne korrektes SQLite-Laufzeitverhalten abzulehnen.

### Finale Verifikation und Messwerte

- Finale PyCompile-Prüfung der geänderten H0.1-Quellen/Tests: Exit `0`, keine Diagnose.
- Vollständiger H0.1-Lauf nach der einzigen Fixrunde unter hartem NumPy-/MLX-Import-,
  `find_spec`-, `sys.modules`- und Socket-Auditguard: Exit `0`, `44/44` Tests,
  `2,228/2,228` Subtests, Failures `0`, Errors `0`, Skips `0`, Expected-Failures `0`,
  Unexpected-Successes `0`; Wall `19.340058 s`, Self-User `18.428778 s`, Self-System
  `0.130453 s`, Child-User/System jeweils `0`, Peak-RSS `38,371,328 B`.
  Socket-Konstruktions-, NumPy-/MLX-Import-/Discoveryversuche und verbotene geladene
  Module: jeweils exakt `0`.
- Gegenüber dem vorigen grünen Storage/UI-Lauf: `+4` Tests, `+14` Subtests,
  Wall `+3.946227 s`, Peak-RSS `-770,048 B`. Die zusätzliche Laufzeit umfasst
  vollständigen Snapshot-/Altzeilen-Replay, Dateiidentitäts- und erweiterte
  Schema-Negativtests; sie ist keine Modell- oder Produktionsperformanceaussage.
- Neue Belege: `INSERT OR REPLACE` mit gleichen, anderen und Bundle-Hash-kollidierenden
  Bytes jeweils bei `recursive_triggers=0/1`; zwei interleaved Storage-Verbindungen;
  symlinkfreier Finalpfad; deterministischer Swap nach Connect und nach `BEGIN`;
  schema-before-row-Reihenfolge innerhalb aktiver Transaktion; Snapshot-Tamper;
  revisionsverschiedene gültige DBs mit gleichem Count/Max-Zeitpunkt; unbekanntes
  `PROPFIND -> 405`; Response über 1 MiB vor Header; getrennte Lineageprojektion.
- Statische P1-Checkliste bestätigte: Dashboard verwendet keine rohen Count-/Recent-
  SQL-Ergebnisse; alle Snapshotwerte stammen aus `verified_rows`; Revision bindet
  Schema plus geordnete Content-Identitäten; Writer-Schemacheck liegt nach
  `BEGIN IMMEDIATE`; kein NumPy-/MLX-/H0-Import in H0.1-Storage/Dashboard; keine
  `.friday-data/h01.sqlite3` vorhanden. Keine Datei unter `friday_h0/`, keine H0-DB,
  kein Live-/GPU-/MLX-Pfad, Modell, Download oder Installer wurde berührt.
- Das verschachtelte ProjectAtlas-Repository weist weiterhin ausschließlich die
  vorgefundenen untracked Gradle-Caches in den Groovy-/Kotlin-Fixtures aus; sie wurden
  nicht verändert oder entfernt.

## 2026-08-20 — Run22 abgeschlossen; Dokumentationsabschluss

### Ziel und Freigabe

- Ziel dieses Eintrags ist ausschließlich die nachvollziehbare Dokumentation des bereits
  abgeschlossenen Run22 in `PROJECT_STATUS.md`,
  `docs/LOESUNG_LIVE_PFAD_2026-08-20.md` und diesem Journal.
- Der Nutzer autorisierte eine begrenzte W1v3-/Output-Fix-Umsetzung und genau einen
  `eager_baseline`-Canary. `aa_gpu`, weitere Runs, Downloads und Installationen waren
  nicht freigegeben. In diesem Dokumentationsschritt wurden Code, Spec, Tests, DB und
  Live-Zustand nicht verändert; `docs/PHASE1_MATMUL_SPEC.md` blieb unverändert, weil
  ihr Hash im Run eingefroren ist.

### Befund, Ursache und Lösung

- Der historische Launcher-Fehler ist behoben: der lexikalische venv-Launcher wird an
  `Popen` übergeben; die Reproduktion ergab lexikalisch `numpy OK 2.5.2` (Exit 0) und
  beim aufgelösten Basisinterpreter den früheren `ModuleNotFoundError` (Exit 1).
- Run21 war wegen des damaligen Einzel-Eval-Gates `warmup_unstable`; die Ursache war ein
  Designproblem des Gates, nicht ein Codefehler relativ zur damaligen Spec. W1v3 setzt
  nun äußere Warmup-Blöcke von mindestens `50 ms`, maximal `4096` Evals und fail-closed
  `repetition_window_unreachable`, falls die Mindestdauer bis zum Cap nicht erreicht
  wird. Der Gate-Wert ist `round(block_ns/evaluations)`, es gibt `8..16` Blöcke und die
  letzten fünf Gate-Werte müssen innerhalb ±`5 %` liegen. Block-Summaries sind bounded
  und geschlossen; Warmup-Fehler verwenden Schema v2, wobei Schema-v1-Readback erhalten
  bleibt. Das tote `_Timed.output`-Feld wurde entfernt.
- Die Pseudocode-Reihenfolge ist bindend: nach jeder Eval zuerst die äußere verstrichene
  Blockzeit prüfen; nur wenn sie weiterhin unter `50 ms` liegt und `evaluations >= 4096`
  ist, mit `repetition_window_unreachable` abbrechen.
- Die anfängliche Operator-Deutung von `baseline_fallback` als Fail wurde anhand des
  Worker-/Runner-Vertrags korrigiert. Der Common-Wrapper
  `completed/measurement_complete/baseline_fallback` mit `error=null` ist neutral;
  verschachtelt bedeuten `baseline_reference`, `not_run` und
  `aggregation_required=false` eine erfolgreiche eager-baseline-reference. Das ist kein
  Produktfehler.

### Vorher/Nachher und Verifikation

- Vorher: `206 passed + 47 subtests`, DB `21` Runs, Retention `64` Payloads bzw.
  `67,108,864 B` lebend. Vorherige Freeze-Werte: Code
  `aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c`, Spec
  `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.
- Der enge Unittest-Lauf wurde absichtlich nach `30.018 s` mit Exit `124` gestoppt,
  nachdem `103` Marker grün und keine Fehler/Fehlschläge sichtbar waren. Er wird als
  abgebrochener Lauf dokumentiert, nicht als Fehler verschwiegen. Der anschließende
  vollständige einmalige Pytest-Lauf endete mit Exit `0`, Wall `66.837 s`,
  `228 passed` und `2211` Subtests in `66.24 s`.
- CLI-Lock: Exit `78`; Usage-Fehler: Exit `64`; `xcodebuild -checkFirstLaunchStatus`:
  Exit `0`. ProjectAtlas `0.4.5-rc1`. Umgebung: Python `3.12.13`, NumPy `2.5.2`,
  MLX `0.32.0`, macOS `26.5.2 arm64`. Der sandboxed Import meldete kein Metal-Gerät;
  es wurde dort keine MLX-Operation ausgeführt.
- Retention nach dem Fix: `67,108,864 B` erzeugt, `0` Payloads und `0 B` retained;
  `_Timed` enthält nur `duration_ns`, `evaluation_ns` und `synchronize_ns`.

### Run22 — ein einziger Live-Lauf

- Es wurde genau ein Live-Befehl ausgeführt, ohne Retry und ohne `aa_gpu`. Run-ID:
  `h0-eager_baseline-characterization-0-14d435dcc2170feec70d8baaa712860e59a6148ca3f211aad98eff1c9d7cf0ff`.
  Äußerer `real=3.79 s`, Exit `10`, DB danach `22` Runs.
- Run22: `8` Warmups, stabil; Gate-Werte
  `[2566556,2179783,2188775,2143891,2155069,2174895,2195533,2192185]`, Median der
  letzten fünf `2174895 ns`. Es gab `30` Blöcke mit `32` Reps; Calibration
  `68155792 ns`. Baseline-Median `2138574.859375 ns`, MAD `17041.671875 ns`, IQR
  `35343.0859375 ns`, Minimum `2105915.34375 ns`, Maximum `2210087.25 ns`.
- Correctness-Gate bestanden: `9/9` Cases, `86/86` Metrics; `abs_max=0.0310508173`,
  `normalized_l2=0.0002074681`, `abs_q99=0.0110023008`,
  `rel_q99_abs_oracle_ge_1=0.0004333980`.
- Memory: active `16,777,216 B`, peak `25,165,824 B`, cache `8,422,698 B`, RSS-Peak
  `369,655,808 B`; Memory-Gate `not_evaluable_missing_required_metric`,
  `hard_limit=false`.
- Freeze: Code `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`,
  Spec `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; kein Git-Root.
  Artefakte: Manifest `73058165244fe505035182f0044dc5ab8bd16ef523ebfc44b44d5b6f616e239e`,
  Result `bda3...`, Evidence `edaf...`, Projection `a51...`, Bundle
  `a566c912032efab919dddf5ca7f67b986f29464a655abf15617733aeb6947c49`.
- Dashboard-Snapshot war socketfrei: `snapshot_id=325afcc9a45311ba716f64a51e7395cd7f2cf1c872c9a3f349c6daf9361398de`,
  `source_revision=7cdad7edcb6099894d588bb9927de322bd4f7ce02d256673768647db54131c73`,
  `run_count=22`, Status `completed`.

### Grenzen und nächster Schritt

- Run22 ist ein einzelner Baseline-Lauf und daher keine vergleichende
  Performanceaussage. H0-Baseline ist ausführbar und referenziert; A/A, Optimierung und
  Self-Optimization sind nicht bewiesen.
- `aa_gpu` und weitere Runs benötigen eine neue ausdrückliche Nutzerfreigabe.

### Ergänzung — vollständige Run22-Hashes und Dashboard-Server

- Vollständiger Result-Hash:
  `bda3d23d56e49c2d26bf7c3e73d52b61c3ea022c3fb61ab0719bfedef58a6d09`.
  Vollständiger Evidence-Hash:
  `edaf6cae5a98185f183fd368189a8be3a56c194540e4f64300903cff42d1a6a0`.
  Vollständiger Projection-Hash:
  `a51aa4b3cadf00dc5338eee199206b2b8f876c4fd3aeaaa2d5261364254ed790`.
- Nach Abschluss wurde der read-only Dashboard-Server erneut auf
  `http://127.0.0.1:8765/` gestartet: `state=serving`, Session `4414`. Die integrierte
  Browseransicht traf vor dem Serverstart zunächst `connection refused` und blockierte
  danach den lokalen Reload per Browser-URL-Policy. Es gab keine Umgehung und keine
  Datenänderung. Der socketfreie Snapshot bleibt die verifizierte UI-Evidenz; der direkte
  lokale Link ist verfügbar.

## 2026-08-20 — H0→H0.1 Legacy-Warmup-Importer, nur Dry-run/Temp-E2E

### Vorregistrierung und Architekturentscheidung

- ProjectAtlas wurde vor der Quellarbeit verwendet; der erste Watch war unverändert und
  der fokussierte Brief führte zuerst zur H0-Storage-Vertrauensgrenze. Vor jeglicher
  Importer-Ausführung wurde `docs/H01_LEGACY_IMPORT_SPEC.md` angelegt. Der Import ist
  rein beschreibend: `legacy_h0_warmup_observation`, `legacy_observation`,
  `no_h0_conclusion`; Stationarität, Paced-Gate, Promotion und H0-Reklassifikation sind
  jeweils exakt `false`. `analyze_trace` und `analyze_study` werden nicht aufgerufen.
- Deterministische Auswahl: alle H0-v1-`common_result`-Bundles mit
  `mode=eager_baseline`, sortiert nach `created_at_unix_ns, run_id`. Jeder Kandidat wird
  über H0 `Storage.open(..., read_only=True)` mit exaktem Schema sowie
  `verify_common_result_bundle` vollständig einschließlich Wrapper-/Child-Hashes
  replayt. Importierbar sind ausschließlich der completed eager Warmup-Adapter und der
  invalid-`warmup_unstable`-Adapter; fehlende/unterstützungsfremde Evidenz erscheint mit
  geschlossenem Ausschlussgrund im Report. Ein behaupteter registrierter Adapter mit
  unbekanntem, malformed, booleschem, zu großem oder Int64-fremdem Wert verwirft den
  gesamten Audit fail-closed.
- Die Quelldatenbank wird als reguläre Nicht-Symlink-Datei gehasht, per URI `mode=ro`
  und `query_only=1` innerhalb einer Lesetransaktion geprüft und nach Close erneut über
  Eltern-/Dateiidentität und SHA-256 gebunden. Das ist die dokumentierte getestete
  Invariante, kein allgemeiner TOCTOU-Anspruch. Dry-run ist Standard; Execute verlangt
  Zielpfad plus `execute=True`. In diesem Slice wurde Execute nur gegen temporäre
  Testdatenbanken verwendet, niemals gegen `.friday-data/h01.sqlite3`.
- Die geordnete Sequenz bleibt vollständig erhalten. Median, MAD und linear
  interpolierter IQR werden ohne Binär-Float als reduzierte rationale Zahlen
  `{numerator,denominator}` rekonstruiert; Count, Last-5, Min und Max sind ebenfalls
  gebunden. Die Legacy-Lineage enthält Parent-Run, Manifest, Result, gesamte Evidence,
  Bundle, Code, Spec, Environment und Quelldatenbank-SHA-256. Entity-ID und kanonischer
  Importreport sind deterministische SHA-256-Ableitungen.

### Änderungen

- Neu: `friday_h01/import_h0.py`, `tests/test_h01_import_h0.py`,
  `docs/H01_LEGACY_IMPORT_SPEC.md`, `tools/run_h01_guard.py`.
- Geändert: `friday_h01/storage.py` (exakter Legacy-v1-Payload-Replay),
  `friday_h01/dashboard.py` (gebundene Warmup-Statistik-/Historienprojektion),
  `tests/test_h01_storage.py`, `tests/test_h01_dashboard.py` und
  `docs/H01_STORAGE_DASHBOARD_SPEC.md`. Die SQLite-v1-Migration blieb unverändert.
- Es wurde keine Datei unter `friday_h0/` per Patch verändert, kein H0-Bundle oder
  Produktions-DB geschrieben, kein Modell geladen, keine Software installiert und kein
  Live-/GPU-/MLX-Pfad ausgeführt. `.friday-data/h01.sqlite3` war nach allen Prüfungen
  weiterhin nicht vorhanden.

### Messungen und Tests

- Erster fokussierter Importer-Guardlauf: `5/5` Tests, `6/6` Subtests, keine
  Failures/Errors/Skips; interne Wall `0.606283583 s`, Self-User `0.514975 s`,
  Self-System `0.062231 s`, Peak-RSS `38,731,776 B`; extern
  `real=0.69 s`, `user=0.52 s`, `sys=0.06 s`. NumPy-/MLX-Imports und geladene Module
  jeweils `0`, Socketkonstruktionen `0`.
- Ein erfolgreicher kombinierter Zwischenlauf belegte `52/52` Tests und `2234/2234`
  Subtests (vollständiges H0.1 plus drei fokussierte H0-Storage-Tests), intern
  `17.662442583 s`, Self-User `17.484975 s`, Self-System `0.142724 s`, Peak-RSS
  `44,695,552 B`, extern `real=17.81 s`; Imports/Sockets jeweils `0`. Er bleibt ein
  Zwischenbeleg, weil der Inline-Harness anschließend durch einen reproduzierbaren
  Projekt-Runner ersetzt wurde.
- Finaler persistenter H0.1-Guardrunner
  `/Users/tobiasburandt/Project_Friday/.venv/bin/python -m tools.run_h01_guard`:
  Exit `0`, `49/49` Tests, `2234/2234` Subtests, `0` Failures, `0` Errors, `0` Skips,
  interne Wall `19.018361374968663 s`, Self-User `18.890795 s`, Self-System
  `0.156917 s`, Child-User/System jeweils `0`, Peak-RSS `44,924,928 B`; extern
  `real=19.16 s`, `user=18.90 s`, `sys=0.15 s`. Sieben explizite Testmodule wurden
  vorab exakt im Project-Root aufgelöst; NumPy/MLX preloaded/attempted/loaded jeweils
  `0`, `socket.__new__`-Versuche `0`.
- Separater H0-Storage-Regressionlauf ohne Socket-Patch: `3/3` Tests, `0.038 s`
  intern, extern `real=0.11 s`, `user=0.06 s`, `sys=0.03 s`; vollständiger
  read-only Bundle-Replay, Tamper-Ablehnung und exaktes v1-Schema bestanden.
- PyCompile vor dem finalen Runner: alle H0.1-Module und -Tests Exit `0`, extern
  `real=0.08 s`, `user=0.05 s`, `sys=0.01 s`. Nach dem Runner-Rootfix wurden Runner,
  Importer, Storage, Dashboard und betroffene Tests erneut kompiliert: Exit `0`,
  `real=0.05 s`, `user=0.04 s`, `sys=0.01 s`.

### Erkannte Fehler, Ursachen und dauerhafte Lösungen

- Zwei frühe Inline-Harnesses erreichten keine Tests: zuerst enthielt ein Shell-f-string
  eine ungültige Escape-Sequenz; danach wurde `socket.socket` vor `ssl`-Import ersetzt,
  sodass `SSLSocket` eine Funktion subklassifizieren sollte. Ursache war nicht
  Projektcode, sondern nicht reproduzierbare Inline-Orchestrierung. Dauerhafte Lösung:
  `tools/run_h01_guard.py` lädt HTTP/SSL normal, setzt einen MetaPath-`find_spec`-Guard
  für NumPy/MLX vor den Projektimports und blockt reale Socketkonstruktion während der
  Tests über `sys.addaudithook`/`socket.__new__`, ohne Socketklassen zu monkeypatchen.
- Der erste Aufruf des neuen Skripts als Dateipfad führte `7` `_FailedTest` aus, weil
  Python `sys.path[0]` auf `tools/` statt den Projektroot setzte (interne Wall
  `0.000326750 s`, Self-User `0.052975 s`, Self-System `0.019290 s`, RSS
  `25,722,880 B`). Die dauerhafte Korrektur pinnt
  `Path(__file__).resolve().parents[1]` an `sys.path[0]`, wechselt in diesen Root,
  prüft jede Modul-Spec samt exaktem Ursprung fail-closed und wird als `-m`-Modul
  gestartet. Der danach einmalig autorisierte korrigierte Lauf ist der oben
  dokumentierte finale grüne Nachweis.

### Finaler statischer Scope- und Atlas-Audit

- Der abschließende ProjectAtlas-One-shot-Refresh war erfolgreich: `636` Textkandidaten,
  `616` indexiert, `20` Skips; `401` strukturelle Kandidaten, `106` Summaries;
  Symbolpass `484` Kandidaten, `10` neu geparst, `474` unverändert, `0` Timeouts,
  `338` Symbole und `1565` Relationen. Der anschließende Brief meldete den Index als
  `available` und rankte den neuen Importer sowie die H0.1-Storage-/Dashboardtests.
  Ein letzter Refresh nach Journal-Synchronisierung bestätigte erneut `636/616/20`
  Textkandidaten/indexierte/übersprungene Pfade und `0` Symbol-Timeouts; dabei wurden
  `1` Datei neu geparst und `483` als unverändert erkannt.
- Read-only H0-Provenienz nach Abschluss entspricht weiterhin exakt dem dokumentierten
  Run22-Freeze: Code
  `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`, Spec
  `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.
  Der statische Importer-Sweep fand keinen Design-A-Analyseaufruf und keine NumPy-/MLX-
  Imports; Execute-Aufrufe existieren nur in Temp-E2E-Tests. Die Produktionsdatei
  `.friday-data/h01.sqlite3` ist nicht vorhanden.
- Das verschachtelte ProjectAtlas-Git-Repository ist weiterhin nur durch die bereits
  bekannten untracked `.gradle/`-Fixture-Caches unter Groovy/Kotlin unclean. Kein
  ProjectAtlas-Quellpfad wurde geändert.

## 2026-08-20 — Versioniertes H0-Generationsinventar, keine Extraktion

### Ursache und Strategiewechsel vor Daten

- Der einmalige produktionsnahe Legacy-Dry-run war ein NO-GO: Das vollständig über
  H0-Storage verifizierte Completed-Bundle
  `h0-eager_baseline-characterization-0-618a0e350c3e2a8f29821eb5a1a4e550836e2d267a67a196b48ec77d1d01c2d5`
  ließ sich nicht durch den aktuellen `normalize_mlx_common_result` projizieren;
  `arms.baseline.warmup` hatte für diesen aktuellen Normalizer unbekannte oder
  fehlende Felder. Ursache war eine unbelegte generationenübergreifende Kopplung,
  nicht ein beschädigtes gespeichertes Bundle. Es gab keinen Retry dieses Dry-runs.
- Vor dem Produktionsinventar wurde deshalb
  `docs/H01_LEGACY_IMPORT_SPEC.md` auf einen Inventory-only-Vertrag umgestellt. Der
  aktuelle Normalizer wurde vollständig aus dem generischen Legacy-Pfad entfernt.
  Die statische Registry ist absichtlich leer; insbesondere existiert kein
  Completed-Descriptor. Legacy-Execute ist vor jedem Zielzugriff geschlossen.
- Der value-unabhängige Algorithmus
  `sha256_recursive_json_structure_v1` bindet rekursive Objekt-Keys,
  Containerarten, Listenlängen/-reihenfolge und die getrennten Scalar-Klassen
  `null/bool/int/float/string`, aber keine Scalar-Werte oder Warmup-Dauern.
  Deklarierte `schema_version`-Pfade/Werte sind separat über
  `recursive_schema_version_paths_v1` gebunden. Registry-Matching unterscheidet
  exakt `matched`, `unsupported_generation` und `claimed_known_malformed`.
- Descriptor- und Entity-ID-Scaffolding bindet Adapter-ID, kompletten
  Descriptor-/Selector-Hash, Parent-Manifest/Result/Evidence/Bundle sowie nur den
  zukünftigen kanonischen Warmup-Sequenzhash. Es erzeugt oder persistiert kein
  H0.1-Entity. Ein archiviertes Produktionsfixture wurde ausdrücklich nicht kopiert.

### Änderungen und erkannte Arbeitsfehler

- Geändert wurden nur `friday_h01/import_h0.py`,
  `tests/test_h01_import_h0.py`, `docs/H01_LEGACY_IMPORT_SPEC.md` und dieses
  append-only Journal. Keine Datei unter `friday_h0/`, keine SQLite-Migration und
  keine Produktionsdatenbank wurden verändert.
- Der erste kombinierte Delete/Add-`apply_patch` wurde vom Patch-Tool vor Anwendung
  abgelehnt, weil zwei Operationen dasselbe Ziel nannten. Es entstand dabei keine
  Dateiänderung. Die reproduzierbare Lösung waren zwei explizite
  `apply_patch`-Operationen: erst Delete, dann vollständiges Add.
- Ein PyCompile-Aufruf unter `/usr/bin/time -lp` meldete ausschließlich
  `sysctl kern.clockrate: Operation not permitted` und Wrapper-Exit `1`, ohne
  Python-Trace. Der direkte identische PyCompile-Aufruf ohne diesen nicht
  sandboxkompatiblen macOS-Wrapper endete mit Exit `0`; der Fehler lag im
  Messharness, nicht im Python-Code.
- Ein unabhängiger Luna-Static-Review der drei fachlichen Dateien fand keine
  P0/P1-Abweichung von Replay-before-fingerprint, leerer Registry,
  Typfingerprint, Matchzuständen, Entity-Bindung oder geschlossener Execute-Grenze.

### Reproduzierbare Tests und Messung

- ProjectAtlas wurde vor der Arbeit verwendet; nach den Edits aktualisierte ein
  One-shot-Watch `636` Textkandidaten (`616` indexiert, `20` übersprungen),
  `401` strukturelle Kandidaten (`103` Summaries, `13` bereinigt) und
  `484` Symbolkandidaten (`2` geparst, `482` unverändert, `0` Timeouts).
- PyCompile für Importer und Importer-Test: Exit `0`.
- Der vollständige persistente H0.1-Core-Guard wurde danach genau einmal gestartet:
  Exit `0`, `48/48` Tests, `2233/2233` Subtests, `0` Failures, `0` Errors,
  `0` Skips; interne Wall `18.490617249859497 s`, Self-User `18.425695 s`,
  Self-System `0.14274199999999998 s`, Child-User/System jeweils `0`, Peak-RSS
  `47,185,920 B`. NumPy/MLX preloaded/attempted/loaded jeweils `0`, reale
  Socketkonstruktionen `0`. Äußere Tool-Wall war `18.606849833 s`.
- Fingerprinttests belegen Determinismus und gleiche Digests bei reinen
  Scalar-Wertänderungen sowie andere Digests bei Key-, Typ- und Listenlängenänderung;
  `bool` und `int` sind verschieden. Bundle-Tamper und Schema-Drift brechen vor dem
  ersten Fingerprint ab. Registrytests belegen matched/claimed-malformed/unsupported
  und die Descriptor-/Entity-ID-Hashbindung.

### Einmaliges Produktionsinventar, read-only

- Preflight: `/Users/tobiasburandt/Project_Friday/.friday-data/h0.sqlite3`, regulär,
  kein Symlink, Device `16777229`, Inode `229166267`, UID `501`, Mode `0600`, Größe
  `1,781,760 B`, SHA-256
  `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`;
  kein WAL/SHM und keine `.friday-data/h01.sqlite3`. Der vollständige H0-Quellbaum
  hatte `19` Dateien und SHA-256
  `8e2c15450fa92050ce51b33e3d7891fe4e8c587e9bae4e87199eb9bb426b0753`.
- `inventory_h0_generations` wurde auf der Produktions-H0-DB exakt einmal
  ausgeführt: Exit `0`, Wall `0.12937262514606118 s`, User `0.125044 s`, System
  `0.0031420000000000024 s`, Peak-RSS `33,816,576 B`; NumPy/MLX-Guard sauber.
  Alle vier Kandidaten waren `full_bundle_verification=verified`; Counts:
  `eligible=4`, `matched=0`, `unsupported_generation=4`,
  `claimed_known_malformed=0`. Registry-SHA-256:
  `3b5839ad468db384682407a350412f477f5a2655910f5038c011702df52f7f66`.
- Kanonisches Inventar: `10,885 B`, Bytes-SHA-256
  `75ed25aa993152cd82bb9586975e090f898366d8a1d4e6f553afd0b4d3c6e0f2`,
  Body-/Inventory-SHA-256
  `798183b46c5cecd3daa90e0e06190b58895346206400f4e9c207c164bb29e1d5`;
  der Body-Hash-Replay war exakt wahr.

### Exakte Kandidaten in gespeicherter Reihenfolge

1. `962b15521ae3b8e6e7bbec401b949cb005a26dc31a4e44c9b19a5a7ae2d23a2f`,
   erstellt `1787203943939913000`: `invalid/runtime_unavailable/baseline_fallback`,
   Fehler `runtime_unavailable`; Code `246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b`,
   Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`,
   Environment `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`,
   Manifest `11ac87fb704169e58ac506eda5d0549a91ad19e8ff52b43c5bb7f28e61d982c1`,
   Result `cb97e223fd26c87aa1f1e3a87e56b4c61c76c5b69e7d0420721392727e31aa02`,
   Evidence `406c42b4a99f72703b9623fd8ba5e5c0e68c46495f5a7bd0db1cef1674e0499d`,
   Bundle `1de0c11763c38462420bb74277d8018b2db1517f9eb17e234938b27681a8c41b`,
   Wrapper `23a5775546b8d90aa66627f6a0fc9ee8718c66acbf899fe52fc619f6d20cd33d`;
   Struktur `5bd47782958e02d186b8daf166364e3c39d2588e63928750ba614e2a91164ecf`
   (`36` Nodes, Tiefe `4`), kein Diagnostic.
2. `618a0e350c3e2a8f29821eb5a1a4e550836e2d267a67a196b48ec77d1d01c2d5`,
   erstellt `1787210431045549000`: `completed/measurement_complete/baseline_fallback`,
   kein Fehler; Code `5f62c419bac782ecc89fd5056b9070ab4789ea3b336f72c1ff7d351c5c5cc055`,
   Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`,
   Environment `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`,
   Manifest `c2f2893332ae46b499509528c2594dd62ffe8c1cd0cbbeb55363e5ba592615d4`,
   Result `88f116d4ac2d934a02674e16fea9372acc3dec46efe02974800b45b143a228c4`,
   Evidence `3cf4e566b51cceb6513b7972e283b8d085e34fbd65d7bb0a655f67b4ae2ded69`,
   Bundle `a88b985afeee3cad1c9eb2daa4694094f8ac403781e47236c033ed38d8a6c000`,
   Wrapper `108eaa4adae17631959a8d900c7c542913689568fe7869ed5994e48ede6aa259`;
   Struktur `397824478c7e9cbdc319387867d7d6623ac24a0e7671389fa54511fbba9fd658`
   (`5262` Nodes, Tiefe `8`), kein Diagnostic.
3. `575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`,
   erstellt `1787222469942453000`: `invalid/invalid/baseline_fallback`, Fehler
   `warmup_unstable`; Code `aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c`,
   Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`,
   Environment `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`,
   Manifest `df44a13e11732bfe75559014fd664fe35330ff15caf283c807dd6b5a163cb6bf`,
   Result `ac4a82c0e3eee350a56345b6269287bff5f86693908708878d762c9a9e056070`,
   Evidence `837841986a7d957ab7762c62d1197bad9b60f4c638d6b2a13ce03f6ea45d755f`,
   Bundle `0279080523fab0df7c8c017a9d460845cb165a3a9dc762410130bac40436f23e`,
   Wrapper `cd409d4d48ac58b2e3a7e02bf906798c5b22832fdf1e8303ccfbb5bd8ec3d7e4`;
   Resultstruktur `0747fd2dc5ee69a6acacca5b9188426b9957f5262853f749edbbfdab7c059857`
   (`57` Nodes, Tiefe `6`), Diagnostic-Struktur
   `6d76f95584ab9c75d1bef8bd4686e418075dd0877beebf06c94400b7eb4c999a`
   (`21` Nodes, Tiefe `3`, deklarierte Schema-Version `1`).
4. `14d435dcc2170feec70d8baaa712860e59a6148ca3f211aad98eff1c9d7cf0ff`,
   erstellt `1787228140625499000`: `completed/measurement_complete/baseline_fallback`,
   kein Fehler; Code `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`,
   Spec `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`,
   Environment `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`,
   Manifest `73058165244fe505035182f0044dc5ab8bd16ef523ebfc44b44d5b6f616e239e`,
   Result `bda3d23d56e49c2d26bf7c3e73d52b61c3ea022c3fb61ab0719bfedef58a6d09`,
   Evidence `edaf6cae5a98185f183fd368189a8be3a56c194540e4f64300903cff42d1a6a0`,
   Bundle `a566c912032efab919dddf5ca7f67b986f29464a655abf15617733aeb6947c49`,
   Wrapper `ce43b5c4c27c56355d4f5898783ec538cd4b07203076173860660e56382b77da`;
   Struktur `ef39c352ebf66e16da9c2e331cbf528444e2e8cfea871d04cb020ea23809a249`
   (`5203` Nodes, Tiefe `8`), kein Diagnostic.

### Nachzustand und Entscheidung

- Postflight entsprach dem Preflight byte- und identitätsgenau: H0-DB-Hash,
  Device/Inode/UID/Mode/Größe, H0-Quellbaum sowie Importer-/Spec-/Test-Hashes waren
  unverändert; kein WAL/SHM wurde erzeugt und `.friday-data/h01.sqlite3` blieb
  abwesend. Importer/Spec/Test hatten während des Laufs SHA-256
  `4576a6937532b43b38991bb2ceeef6c331b7d174d691705bc244047587e91a1c`,
  `f8ce6bd8e9345b14e57d8ef1ace8b96e0a7de970407b0106a94e3b235bb965d3`
  und `2a0b3aa8d6692a88558b0360b74e362fd4fdab9700affdf70696a365179a1e0c`.
- GO für das reine, reproduzierbare Generationsinventar und die leere versionierte
  Registry. NO-GO für jede Legacy-Extraktion oder Persistenz: alle vier Generationen
  sind bewusst unsupported, und kein Produktionsbundle wurde kopiert, extrahiert oder
  reklassifiziert.

## 2026-08-20 — H0.1: eingefrorene Generationsadapter A–D

### Ziel, Freigabegrenzen und Entscheidungen

- Auf Basis des zuvor eingefrorenen Produktionsinventars wurden vier ausschließlich
  provenance-/strukturgebundene Registry-Deskriptoren registriert. Es gab keinen
  Produktions-Dry-run, keinen Import-Execute, keine Erstellung einer Produktions-H01-DB,
  keinen Live-/GPU-/MLX-Lauf und keinen Download oder Installation. H0-Quellen und die
  Produktions-H0-DB wurden nicht verändert.
- Matching geschieht vor jedem Zugriff auf Warmupwerte über vollständige gespeicherte
  Code-/Spec-/Environment-Hashes, Result-Struktur und Schema-Tags sowie bei C zusätzlich
  Diagnostic-Struktur und -Tags. Bekannte Provenance mit abweichender Form ist ein harter
  `claimed_known_malformed`-Fehler; es gibt weder generischen Fallback noch Aufruf des
  aktuellen H0-Normalizers und keinen Run-ID-/Messwertselektor.
- Adapter A `no_warmup_runtime_unavailable_v1` ist eine erkannte Exklusion und erzeugt
  kein Bundle. B `completed_eager_warmup_v1` liest exakt 11 historische Einzelwarmups,
  C `warmup_unstable_diagnostic_v1` exakt 16 Werte aus dem Schema-1-Diagnostic und D
  `completed_eager_warmup_w1v3` exakt acht W1v3-Blöcke/-Werte. Bool, Nicht-Integer,
  signed-int64-Überlauf, inkonsistente Samples/Blöcke/Median und falsche Zustände werden
  geschlossen abgelehnt.
- Registry-Version, Registry-, Descriptor-, Selector- und Rohwarmup-SHA sowie Parser-ID
  sind in Entity-ID, Manifest, Observation, Lineage und kanonischem Dry-run-Report
  gebunden. Legacy-Ergebnisse bleiben rein deskriptiv: `no_h0_conclusion`, keine
  Stationaritätsaussage, Promotion oder H0-Reklassifikation.

### Archivierte, verifizierte Testevidenz

- Nach öffentlichem H0-`verify_common_result_bundle`-Replay über den read-only geöffneten
  Ursprung wurden vier kanonische Fixtures ohne Pfad-/Secret-Felder unter
  `tests/fixtures/h01/` gespeichert. Bytes/SHA-256: A `5,918 B` /
  `7f2734234984994e4625f8246da8f7c80f05280ced895167637a4aca14d74813`, B
  `202,984 B` / `22876adef92c0ccd93e50be82e50db349e9e261e1111b573a49bc1c4d1802f0c`,
  C `6,303 B` / `f5058482ded55742eab50ef5b29ab9e82d8b9bac256e36eb4f70abfc74967f7d`
  und D `197,499 B` /
  `2f60fb80eb8610c2fec658d6495cfb3eba2ab9407caae951ee96a4dc0a1ee82a`.
- Finale statische Registry-SHA-256:
  `eb07449bf99c622fb06e26347e6760050c4de2dd60a84fbfab0ca9fa3373c745`.
  Descriptor-SHA-256: B
  `062783bc0039fe9facb167dcdd785e9cae14556d7fc0787e67991c9063c89f78`, D
  `aa8cec6e7854ab70c0c7e54956fdda2285128575fc43d449d2396eb2ec133680`, A
  `e56f468f440d3b0c9e511da1fa999c4a40096b04f4e51aa3f32b1d8fb9b46f93`, C
  `1123ad870e23ea53e7792e1e84c9260344aa401bc5793d5017c59314ed3d3781`.

### Änderungen und reproduzierbare Verifikation

- Geändert wurden `friday_h01/import_h0.py`, `friday_h01/storage.py`,
  `friday_h01/dashboard.py`, `tests/test_h01_import_h0.py`,
  `tests/test_h01_storage.py`, `tests/test_h01_dashboard.py`,
  `docs/H01_LEGACY_IMPORT_SPEC.md`, dieses Journal und die vier Fixture-Dateien.
- ProjectAtlas wurde vor der Implementierung verwendet. Der Refresh nach den Edits
  erfasste `640` Textkandidaten (`620` indexiert, `20` übersprungen), `405`
  strukturelle Kandidaten (`108` zusammengefasst, `13` bereinigt) und `484`
  Symbolkandidaten (`7` geparst, `477` unverändert, `0` Timeouts).
- `py_compile` für H01-Core, Importer, Storage/Dashboard-Tests und Guard: Exit `0`,
  Real `0.11 s`, User `0.05 s`, System `0.02 s`.
- Der persistente H01-Guard wurde exakt einmal vollständig ausgeführt: `54` Tests,
  `2,241` Subtests, `1` Failure, `0` Errors/Skips, interne Wall
  `19.514408542076126 s`, Self-User `19.303294 s`, Self-System `0.196176 s`,
  Child-User/System `0`, Peak-RSS `45,694,976 B`; NumPy-/MLX-Importversuche,
  vor-/nachgeladene Module und reale Socketkonstruktionen jeweils `0`.
- Fehler: Ein bereits angepasstes Dashboard-Fixture nutzt nun korrekt B mit 11 Werten,
  die zugehörige Assertion erwartete noch 8. Ursache war ausschließlich eine veraltete
  Testkonstante, nicht Parser, Storage oder Dashboard. Erfolgreiche Lösung: Erwartung
  auf den registrierten B-Vertrag 11 korrigiert; der gezielte Nachlauf
  `tests.test_h01_dashboard` bestand `6/6` Tests in `4.272 s` intern (`4.42 s` real,
  `4.23 s` user, `0.06 s` system). Es wurde kein zweiter Full-Guard gestartet.
- Die separate H0-Storage-Regression bestand `24/24` Tests in `0.245 s` intern
  (`0.32 s` real, `0.19 s` user, `0.09 s` system). Die Produktions-H0-DB blieb auf
  SHA-256 `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`;
  Produktions-H01, H01-WAL und H01-SHM blieben abwesend.
- Der verschachtelte ProjectAtlas-Gitstatus ist nicht sauber: bereits beobachtbar sind
  zwei untracked Gradle-Cacheverzeichnisse unter den Groovy-/Kotlin-Fixtures. Sie wurden
  weder verändert noch entfernt, da das außerhalb dieses H01-Slices liegt.

### Entscheidung

- **GO für den separaten, reversiblen Generation-Adapter-/Fixture-/Detached-Bundle-Slice.**
  Die geforderten positiven B/C/D-, A-Exklusions-, Reihenfolge-, Shape-, Value-, Hash-,
  Tamper-, Collision- und No-Reclassification-Pfade liefen im einmaligen Guard bis auf
  die isolierte veraltete Dashboard-Erwartung erfolgreich; deren gezielter Nachlauf ist
  grün. **Weiterhin NO-GO für Produktions-Dry-run oder Import-Execute**, bis diese jeweils
  separat freigegeben und mit unverändertem H0-Pre-/Postflight durchgeführt werden.

## 2026-08-20 — Solo-Übernahme, Root-Guard und autorisierter Adapter-Dry-run

### Entscheidung und Arbeitsgrenze

- Der Nutzer verlangte ausdrücklich: `Verwende luna nicht mehr mach nur noch du das bitte`.
  Seit dieser Anweisung arbeitet der Root-Agent selbst; alle noch aktiven Luna-Arbeiten
  wurden beendet und es wurden keine weiteren Subagenten eingesetzt.
- Die frühere Nutzerfreigabe `Ich gebe alles frei mach dein ding :)` wird für den bereits
  spezifizierten H0→H0.1-Adapter-Dry-run und den zuvor freigegebenen SQLite-v1-Scope
  herangezogen. Sie autorisiert weiterhin keine Downloads, Installationen, Modellgewichte,
  Custom-Metal-Kernels oder eine H0-Reklassifikation. Keine dieser Aktionen fand statt.
- ProjectAtlas wurde vor den Root-eigenen Prüfungen und erneut vor der Planung des
  Execute-Slices verwendet. Der jüngste Refresh fand keine seit dem unmittelbar
  vorausgehenden Indexstand noch nicht erfassten Änderungen (`0` neue Kandidaten);
  der fokussierte Brief bestätigte `friday_h01/import_h0.py`, `friday_h01/storage.py`
  und `tests/test_h01_import_h0.py` als kleinsten relevanten Scope.

### Root-eigene Reproduktion

- Der Root-Agent führte `tools/run_h01_guard.py` selbst aus: Exit `0`, `54/54` Tests,
  `2,244/2,244` Subtests, `0` Failures, `0` Errors, `0` Skips; interne Wall
  `20.552923833 s`, Self-User `20.044671 s`, Self-System `0.251724 s`, Child-User/System
  jeweils `0`, Peak-RSS `44,072,960 B`. NumPy-/MLX-Importversuche, geladene Module und
  reale Socketkonstruktionen waren jeweils `0`.
- Der exakt einmalige Produktions-Dry-run materialisierte intern den vollständigen
  `LegacyImportOutcome`, schrieb aber nichts. Erst die nachgelagerte, nicht zum Importer
  gehörende ad-hoc-Zusammenfassung griff auf ein nicht existentes Feld `kind` zu und
  beendete den Kommando-Harness mit `KeyError`. Ursache: erfundene Projektionsschlüssel
  statt Ausgabe des kanonischen Reports. Es gab keinen Retry auf der Produktionsquelle.
- Strategiewechsel gegen eine Wiederholungsschleife: Eine byteidentische temporäre Kopie
  der H0-DB wurde offline replayt. Ein erster abgeleiteter Printer wiederholte denselben
  Fehlerklassentyp mit dem falschen Feld `warmup_count`; danach wurde keine weitere
  abgeleitete Projektion gebaut, sondern ausschließlich der kanonische Report ausgegeben.
  Dieser Replay war erfolgreich. Beide Fehler waren Reporting-Harnessfehler nach Rückkehr
  des Importers, keine Parser-, Registry-, Storage- oder Quelldatenfehler.

### Kanonisches Dry-run-Ergebnis

- Report-Schema `friday_h01.legacy_h0_import_report.v2`, Modus
  `adapter_dry_run`, Report-SHA-256
  `bc3432508e0a64ed26202289a073633db8b2fb84ed5ed9faee3eda974943f7d9`,
  Registry-SHA-256
  `eb07449bf99c622fb06e26347e6760050c4de2dd60a84fbfab0ca9fa3373c745`.
  Counts: `eligible=4`, `importable=3`, `excluded=1`.
- A wurde als `recognized_no_warmup_runtime_unavailable` ausgeschlossen. B erzeugte
  Entity `legacy-h0-warmup-fe3736c2b921881cba5b96cd791b6bf251ebbf6f3fc160b84d62e84b5baa7586`,
  Bundle `fe666f3b5351fb3d0f7502711bf8024700a253b0d88d008b66c90e216885e16c`
  und Raw-Warmup-SHA `29cabf4675b815d99b5c2ee14de1f6b621d2c91671b6d1800fc4a22032391304`
  über `11` Werte; Median `2,133,334 ns`, MAD `84,459 ns`, IQR `691,312 ns`.
- C erzeugte Entity `legacy-h0-warmup-aa545f64c56191be9f1440417490269d5c5fc72ea179bc2aada75e7eaac72222`,
  Bundle `9ce18778b498a6fcd4910a1521704cbe2998514159981f07cce724820d369b5d`
  und Raw-Warmup-SHA `6b2e8bf031c4221bf9887bbb9f54cb9f78caf9150f6a0db4c7c5010d05f661d8`
  über `16` Werte; Median `4,782,709/2 ns`, MAD `287,125 ns`, IQR
  `2,329,041/4 ns`.
- D erzeugte Entity `legacy-h0-warmup-bfdfc971a50d9a2074ed7de77b88683f44ef9274cd58a60e2229fefb6dad1c88`,
  Bundle `130675576d74b798db2afeab62aee4574021c55b08d16cbadcda9166250ff2a3`
  und Raw-Warmup-SHA `1e49dd0b2a4dffa12403afab54dac6eebaac7f84f5d86aafdfa9106e5fd34213`
  über `8` Werte; Median `2,184,279 ns`, MAD `10,319 ns`, IQR `46,167/2 ns`.
- Die Produktions-H0-DB blieb vor und nach dem Lauf identisch: SHA-256
  `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`,
  Device `16777229`, Inode `229166267`, UID `501`, Mode `0600`, Größe
  `1,781,760 B`; kein H0-WAL/SHM und keine Produktions-H01-DB entstanden.
  Die temporäre Kopie und anschließend ihr leeres Verzeichnis
  `/private/tmp/project_friday_h01_replay.R9BG3v` wurden vollständig entfernt.

## 2026-08-20 — Atomarer H0→H0.1-Import und private Produktions-H01

### Vorregistrierte Entscheidung und Änderungen

- Vor der Implementierung wurde `docs/H01_LEGACY_IMPORT_SPEC.md` von der
  Dry-run-Grenze auf einen expliziten Execute-Vertrag erweitert: kein Defaultziel,
  vollständiger H0-Replay vor Target-Open, Source/Target-Alias-Reject, ein atomarer
  Batch, idempotenter Wiederanlauf, read-only Target-Replay und H0-Identitäts-/Hash-
  Postflight. `docs/H01_STORAGE_DASHBOARD_SPEC.md` registriert denselben Batchvertrag.
- `Storage.persist_bundles` rebuildet höchstens `200` Bundles vor
  `BEGIN IMMEDIATE`, lehnt doppelte IDs ab, verifiziert im Transaction-Snapshot Schema,
  Datei und alle vorhandenen Zeilen, prüft sämtliche Konflikte vor dem ersten Insert
  und replayt jeden Insert vor Commit. `persist_bundle` delegiert auf diesen Pfad.
- `audit_h0_legacy_warmups(..., execute=True, target=...)` persistiert nur den zuvor
  kanonisch gebauten vollständigen Importable-Satz, verifiziert geordnete Outcomes und
  öffnet den Target anschließend erneut `mode=ro/query_only=1`. Der Report bleibt vom
  Transaktionszustand getrennt und trägt im Execute-Modus `adapter_execute`.
- Geändert wurden `friday_h01/storage.py`, `friday_h01/import_h0.py`,
  `tests/test_h01_storage.py`, `tests/test_h01_import_h0.py`, die beiden genannten
  Specs und dieses append-only Journal. H0-Code, H0-Schema und H0-Daten wurden nicht
  verändert. Keine Installation, kein Download, kein Modell, kein MLX-/GPU-Lauf und
  kein Socketserver waren Bestandteil dieses Slices.

### Offline-Verifikation vor Produktions-Execute

- PyCompile der vier geänderten Python-Dateien: Exit `0`.
- Fokussierter Storage-/Importer-Lauf: `21/21` Tests, `0` Fehler, intern
  `2.657 s`; äußerer Total `2.830 s`, User `2.58 s`, System `0.13 s`.
- Erster vollständiger Post-Execute-Code-Guard: `56/56` Tests,
  `2,244/2,244` Subtests, `0` Failures/Errors/Skips; interne Wall
  `20.875529707875103 s`, Self-User `20.542698 s`, Self-System
  `0.20074699999999998 s`, Child-User/System `0`, Peak-RSS `46,219,264 B`.
  NumPy-/MLX-Importversuche, vor-/nachgeladene Module und Socketkonstruktionen jeweils
  `0`; äußerer Total `20.996 s`.
- ProjectAtlas-Refresh nach den Edits: `640` Textkandidaten (`620` indexiert,
  `20` übersprungen), `405` strukturelle Kandidaten (`110` Summaries, `13`
  bereinigt) und `484` Symbolkandidaten (`7` geparst, `477` unverändert,
  `0` Timeouts).

### Genau ein Produktions-Execute

- Preflight: ausschließlich `.friday-data/h0.sqlite3`, Device `16777229`, Inode
  `229166267`, UID `501`, Mode `0600`, Größe `1,781,760 B`, SHA-256
  `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`;
  Produktions-H01 und Sidecars waren abwesend.
- Der Execute wurde exakt einmal auf H0 und Ziel `.friday-data/h01.sqlite3`
  aufgerufen: Exit `0`, Report-SHA-256
  `4e73ab2d7b0aa0bf0cb7e559550de254ddadfa41c0f31ee86b92b9203bef788f`,
  Counts `eligible=4/importable=3/excluded=1`. Alle drei geplanten B/C/D-Entities
  wurden in Quellreihenfolge mit Zustand `inserted` und den zuvor dry-run-verifizierten
  Bundle-SHAs `fe666f3b…e16c`, `9ce18778…9b5d` und `13067557…2a3` bestätigt.
- Execute-Messung: Wall `0.20046620815992355 s`, Self-User `0.21628899999999998 s`,
  Self-System `0.036427999999999995 s`, Peak-RSS `34,422,784 B`. Diese Laufzeit ist
  eine Importmessung, keine MLX-/Matmul- oder Optimierungsmetrik.
- Postflight: H0 blieb identitäts- und bytegleich bei obigem Hash. H01: Device
  `16777229`, Inode `229791947`, UID `501`, Größe `53,248 B`, Inhalt-SHA-256
  `fd2c6e56d5f108d6670745a338930d6050c38b03eac8cc050170a466818d9d57`.
  Es entstanden keine WAL-/SHM-Dateien.
- Erster read-only Dashboard-Replay: `3` Legacy-Observations, keine Paced-Session/
  Study, Revision `d9bc6e5ab430b68e16c9b9dfa62463896c9ad9d64ef003a4a862460378b2af3f`;
  Snapshot plus drei Details Wall `0.02583829080685973 s`, Self-User/System
  `0.095112/0.039071999999999996 s`, Peak-RSS `29,327,360 B`.

### Sicherheitsbefund 0644 und einmalige Hardening-Runde

- Der Postflight fand, dass SQLite die neue H01-Datei unter dem lokalen Umask mit
  Mode `0644` angelegt hatte. Die Daten waren korrekt, aber die Rechte waren weniger
  restriktiv als H0 (`0600`). Ursache war fehlende explizite Mode-Setzung bei der
  Neuanlage, nicht SQLite-Schema oder Importdaten.
- Lösung: Neue DB-Dateien werden unmittelbar nach SQLite-Create über einen
  `O_NOFOLLOW`-Dateideskriptor auf exakt `0600` gesetzt und geprüft. Jede bestehende
  Datenbank mit Gruppen-/Sonstigen-Rechten wird fail-closed abgelehnt; read-only
  benötigt Owner-Read, schreibbar zusätzlich Owner-Write.
- Erster fokussierter Hardening-Lauf: `11` Testmethoden, `0` Failures, `11` Errors
  in den Schema-Drift-Subtests, intern `0.753 s`, äußerer Total `0.880 s`, User/System
  `0.75/0.07 s`. Gemeinsame Ursache: Diese Testfixtures erzeugten absichtlich
  manipulierte DBs direkt mit SQLite/Mode `0644` und erreichten daher korrekt zuerst
  die neue Rechte-Grenze statt der erwarteten `SchemaError`-Grenze.
- Einzige Fixrunde: Die manipulierten Fixtures werden vor dem Schema-Test auf `0600`
  gesetzt. Nachlauf `11/11` grün, intern `0.723 s`, äußerer Total `0.858 s`,
  User/System `0.74/0.07 s`.
- Abschließender kompletter Guard: `57/57` Tests, `2,244/2,244` Subtests,
  `0` Failures/Errors/Skips; interne Wall `25.714773458894342 s`, Self-User
  `22.572449 s`, Self-System `0.502031 s`, Child-User/System `0`, Peak-RSS
  `43,368,448 B`; NumPy/MLX/Socket jeweils `0`. Äußerer Total `26.231 s`.
- Die bereits erzeugte Produktions-H01 wurde einmalig nur per `chmod` von `0644` auf
  `0600` gehärtet. Inode, UID, Größe und Inhalt-SHA blieben exakt unverändert. Der
  finale read-only Dashboard-Snapshot blieb bei Revision `d9bc6e5a…af3f`, Total `3`;
  Wall `0.04000666690990329 s`, Self-User/System `0.099158/0.044884 s`, Peak-RSS
  `29,261,824 B`. Die Differenz zum ersten Snapshot ist keine Performanceaussage.
- Der erste echte Serverstart in der Standardsandbox scheiterte erwartungsgemäß vor
  Bind mit `PermissionError: [Errno 1] Operation not permitted`; kein Listener blieb
  zurück. Nach der expliziten Loopback-Eskalationsfreigabe startete derselbe unveränderte
  read-only Server auf `127.0.0.1:8766`, Session `40690`.
- Der erste Curl-Harness enthielt ein ungequotetes `?` und wurde von zsh mit
  `no matches found` abgelehnt. Der danach korrekt gequotete Sandbox-Curl konnte den
  außerhalb der Sandbox laufenden Listener nicht erreichen (`curl (7)`). Der identische
  freigegebene Loopback-Abruf außerhalb der Sandbox lieferte Exit `0` und den exakten
  API-Snapshot mit Total `3`/Revision `d9bc6e5a…af3f`; der separate HTML-Root-Abruf
  lieferte Exit `0` und die sichtbare Tabelle mit allen drei Entities. Das sind
  Harness-/Isolationseffekte, keine Dashboard- oder Datenfehler.

### Forschungsentscheid

- **GO** für den versionierten, atomaren, idempotenten Import der drei historischen
  Warmup-Beobachtungen und für deren lokale read-only Historienprojektion.
- **Weiterhin kein H0.1-Stationaritätsbefund:** Die importierten B/C/D-Daten stammen
  aus unterschiedlichen historischen Verträgen und sind ausschließlich deskriptive
  Ausgangsevidenz. Es existieren `0` Paced-Sessions und `0` Paced-Studies; H0 bleibt
  unverändert und weder Performancegewinn, Self-Optimization noch Generalisation ist
  belegt.

## 2026-08-21

### Namensentscheid Produktname

- Der Nutzer hat den Produktnamen **Matmole** freigegeben. Zuvor geprüfte Alternative
  `Mole` (ohne `Mat`-Teil) wurde verworfen.
- Begründung für `Mat`: Der Bestandteil ist kein H0-Artefakt. Matrixmultiplikation bleibt
  auch in H2 der dominierende Rechenkern einer Transformer-Inferenz, der Namensteil trägt
  also über alle Phasen. Zusätzlich wirkt `Mat` als Disambiguator gegenüber den
  Nebenbedeutungen von `Mole` (deutsch Hafendamm, englisch Spion/Muttermal,
  Whac-A-Mole-Idiom) und gegenüber der bereits breit belegten Namensnutzung von `mole`
  als Paket-, Domain- und Repositoryname.
- Der Name ist eine reine Benennungsentscheidung. Es gibt daraus **keine** Änderung an
  Code, Paketnamen, Modulpfaden, Datenbanken oder Verträgen; `friday_h0`, `friday_h01`
  und `.friday-data/` bleiben unverändert. Ein späteres Umbenennen ist gesondert zu
  beauftragen und zu verifizieren.
- Es wurde nichts installiert, heruntergeladen oder ausgeführt. Der Forschungsstand ist
  unberührt: H0 Baseline, H0.1 weiterhin `unresolved` mit `0` Paced-Sessions.

### Modellwahl für den späteren lokalen Modelltest

- Der Nutzer hat **Gemma 3 4B** als Modell für den späteren lokalen Test bestimmt. Diese
  Wahl ersetzt die zuvor nur als Präferenz dokumentierte, nie verifizierte Nutzer-
  bezeichnung `qwen 3.8 27b` (siehe `IMPLEMENTIERUNGSPLAN.md`). Die konkrete
  MLX-Repository-ID ist noch **nicht** verifiziert und vor jedem Bezug gegen die
  tatsächlich existierende Hugging-Face-ID zu prüfen.
- Bekannte Eigenschaft mit direkter Messrelevanz: Gemma 3 4B ist multimodal. Der
  SigLIP-Vision-Tower belegt bei reiner Text-Inferenz Speicher, ohne an der Berechnung
  teilzunehmen. Da Peak-RSS in diesem Projekt in jedem Lauf protokolliert wird, ist
  dieser Anteil ein konstanter Offset, der vor dem ersten Lauf im Vertrag festzuschreiben
  und in jeder Auswertung getrennt auszuweisen ist. Die Rechenzeit ist davon nicht
  betroffen.
- Der Nutzer hat die zweistufige Reihenfolge entschieden: **Stufe 1 Gemma 3 1B**
  (text-only), **Stufe 2 Gemma 3 4B**. Stufe 1 schließt den Pfad
  Laden → Messen → Aggregieren → Replay einmal vollständig; erst danach folgt 4B.
  Begründung ist die Regel des kleinsten ersten Versuchs in `AGENTS.md` sowie eine
  unverfälschte Peak-RSS-Referenz ohne Vision-Tower.
- Aus der Stufung folgt eine verwertbare Messeigenschaft: Da Stufe 1 text-only ist und
  Stufe 2 denselben Pfad mit zusätzlichem SigLIP-Vision-Tower durchläuft, ist die
  Peak-RSS-Differenz beider Stufen die direkte Messung des Tower-Offsets. Der Offset ist
  damit nicht zu schätzen, sondern zu messen. Ein Vergleich der Rechenzeiten zwischen
  Stufe 1 und Stufe 2 ist davon ausdrücklich **nicht** gedeckt, da sich Parameterzahl und
  Architektur zwischen 1B und 4B ebenfalls unterscheiden.
- Auch für Stufe 1 ist die MLX-Repository-ID vor dem Bezug zu verifizieren; sie wird hier
  nicht als bekannt behauptet.
- Lizenz: Gemma steht unter der Gemma Terms of Use mit Nutzungsbeschränkungen, nicht
  unter Apache-2.0 oder MIT. Für die lokale Forschung unkritisch, vor einer
  Veröffentlichung von Ergebnissen oder Artefakten jedoch zu prüfen.
- Verifizierter Umgebungsstand zum Zeitpunkt des Entscheids, ohne jede Änderung:
  `mlx` und `numpy` sind im `.venv` vorhanden; `mlx_lm`, `mlx_vlm` und `transformers`
  fehlen. Der Hugging-Face-Hub-Cache unter `~/.cache/huggingface/hub` existiert nicht
  bzw. ist leer; es liegt kein Modellgewicht lokal vor.
- Ein Modelltest setzt damit sowohl eine Installation (`mlx_lm`) als auch einen Download
  (Gewichte) voraus. Beides bleibt gemäß den bindenden Nutzerregeln bis zu einer
  ausdrücklichen Freigabe gesperrt. Es wurde nichts installiert, heruntergeladen oder
  ausgeführt.
- Reihenfolge unverändert: Der Modelltest gehört zu H2. Vorrang hat weiterhin der
  H0.1-Sechs-Session-Vertrag `C0,V0,C1,V1,C2,V2`; H0.1 bleibt `unresolved`.

### H0.1 — kontrollierter Ausführungspfad gebaut und sechs Paced-Sessions ausgeführt

Nutzerfreigabe: ausdrückliche Freigabe für den GPU-Lauf. Es wurde **nichts**
installiert und **nichts** heruntergeladen; insbesondere wurde kein LLM geladen.
Die Anfrage nach einer Modellinstallation betraf H2 und ist für H0.1 gegenstandslos:
H0.1 misst die feste `2048²`-FP16-Matmul, nicht ein Sprachmodell.

**Ausgangslage.** `friday_h01` besaß Schedule, Protocol, Analysis, Study, Storage,
Dashboard und Legacy-Import, aber keinen Ausführungspfad. `build_trace` nahm bereits
aufgezeichnete `durations_ns` entgegen; niemand erzeugte sie. Damit war der im
Implementierungsplan geforderte „kleinste kontrollierte H0.1-Ausführungspfad" die
tatsächliche Lücke.

**Neu implementiert.**

- `friday_h01/provenance.py`: geschlossene Code-Dateiliste, längenpräfigiertes
  Framing, Environment-Allowlist, Spec-Digest. Eigene Liste statt Erweiterung der
  H0-Liste, weil eine Änderung an H0s geschlossener Liste jede H0-Run-Identität
  verändert hätte.
- `friday_h01/runner.py`: Parent-Bindung über den öffentlichen H0-Bundle-Verifier
  (read-only), Preflight, Telemetrie, Pacing, Messung, Trace-Aufbau, Analyse,
  Bundle-Persistenz und Study-Replay. Backend, Sleeper und Uhr sind injizierbar.
- `friday_h01/cli.py`: `preflight`, `session`, `study`, `run-all` hinter demselben
  `--execute`-Release-Gate wie H0; ohne Flag `state=not_released`, Exit `78`, vor
  jedem Runner-, Storage-, NumPy- und MLX-Import.
- `tests/test_h01_runner.py`: 16 Tests, die den vollständigen Pfad einschließlich
  Sechs-Session-Study offline über eine virtuelle Uhr ohne MLX, Metal oder GPU
  durchlaufen.

**Zwei Befunde beim Bauen, beide vertragsseitig korrekt.**

1. H0.1 definiert `fixture_sha256` als kanonischen Hash über die drei
   Fixture-Komponenten und **nicht** als H0s Rohbyte-Digest. Die Komponenten
   `a_sha256`, `b_sha256` und `metadata_sha256` werden unverändert aus der
   vertrauenswürdigen H0-Registry übernommen, das Aggregat ist H0.1-eigen.
2. Eine bereits aufgezeichnete Session lässt sich nicht erneut messen. Die `run_id`
   ist deterministisch aus Provenienz abgeleitet und damit bei jeder Wiederholung
   gleich, die Messdaten sind es nicht; der append-only Store antwortet korrekt mit
   `StorageConflict`. Ein Abbruch mitten in der Study macht die bereits
   aufgezeichneten Sessions dieser Provenienz unbrauchbar — ein Wiederholungslauf
   ist eine neue Study, kein Patch. Das ist als Betriebsrisiko dokumentiert.

**Preflight auf dem Zielgerät.** `state=preflight_ok`, Parent Run22
(`h0-eager_baseline-characterization-0-14d435dc…`), `fixture_seed=4051312678`,
`paced_sessions=0`, `paced_studies=0`, Netzbetrieb (`power_source=ac_power`),
`thermal_state` als registriertes `api_unavailable` (keine stdlib-Bindung an
`ProcessInfo.thermalState`). Drei Proben `24.60 / 4.80 / 3.86 ms` mit sichtbarem
Warmup. Eingefrorener `code_sha256=f66e4b5a2444643fb375a098398bbd3829d717a7b956e62f46a6a54617986e94`.

**Lauf.** Sechs getrennte Prozesse in der registrierten Reihenfolge
`C0,V0,C1,V1,C2,V2`, danach der Study-Schritt. Alle sechs Sessions sind
`h01_session_complete`, `state=inserted`, Wall je Session `66.00`–`66.37 s`.

| Session | Trend | Changepoint | ACF | ESS | Pacing | Tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | -0,1405 | -0,2344 | 0,142 | 49,9 | 0,1048 | 2,748 |
| V0 | 0,3033 | 0,2917 | 0,249 | 42,5 | 0,0409 | 3,132 |
| C1 | -0,0982 | -0,3026 | 0,144 | 47,3 | 0,0382 | 2,575 |
| V1 | -0,3960 | -0,3616 | 0,365 | 25,0 | 0,1955 | 2,532 |
| C2 | 0,0171 | 0,1297 | 0,169 | 59,8 | -0,1016 | 2,604 |
| V2 | -0,2667 | -0,3253 | 0,135 | 63,0 | -0,0159 | 2,572 |
| **Grenze** | ±0,05 | ±0,05 | 0,50 | 40 | ±0,03 | 1,20 |

Gate-Bilanz über alle sechs Sessions: `changepoint` 6/6 fail, `tail` 6/6 fail,
`trend` 5/6 fail, `pacing` 5/6 fail, `ess` 1/6 fail, `acf` 0/6 fail. Summe `23`.

**Study-Entscheid.**
`study_id=h01-study-1812a894b41e795b47f537d9cf186a402fed7e669fc5d29e044085add76c39ca`,
`status=h01_complete_unresolved`, `conclusion=replicated_stationarity_not_supported`,
`session_count=6`, `failed_gate_count=23`, `action=no_h0_conclusion`,
`h0_reclassification=false`, `promotion_applicable=false`.

**Forschungsergebnis.** Die replizierte Stationarität ist **nicht** unterstützt. Das
ist ein gültiges negatives Ergebnis, kein Produktfehler und kein fehlgeschlagener
Lauf: sechs vollständige, unabhängig replaybare Sessions liegen vor, und der
vorregistrierte Engineering-Envelope wird deutlich und nicht knapp verfehlt.

Beschreibend, ohne Ursachenbehauptung: Die dominierende Verletzung ist die
Tail-Ratio von `2,53`–`3,13` bei Grenze `1,20`. `acf` besteht in allen sechs
Sessions, und das Vorzeichen des Trends wechselt zwischen Sessions. Die Daten sind
damit nicht durch einen gerichteten Drift geprägt, sondern durch sporadische
Ausreißer. In C0 liegt der Main-Median bei `6,97 ms` bei `min 3,60 ms` und
`max 18,02 ms`; das erste Main-Sample direkt nach dem 20-Sekunden-Cooldown ist in
beiden geprüften Sessions ein Ausreißer (`14,41 ms` bzw. `13,95 ms`).

Ein methodisch relevanter Vergleich am gleichen Messpunkt: Run22 misst
`evaluation_ns` Median `2,032 ms` plus `synchronize_ns` Median `0,035 ms`, also
rund `2,07 ms` je Matmul innerhalb eines dichten 32er-Batches ohne Pause. H0.1
misst dieselbe Operation auf demselben Gerät am selben Messpunkt, aber einzeln mit
`50 ms`/`750 ms` Pacing, bei einem Median um `6,97 ms`. Die Ursache ist **nicht**
gemessen; Taktung, Kernel-Cache-Zustand und Scheduling sind unbelegte Hypothesen.
Festhaltbar ist ausschließlich, dass die H0-Baseline einen dicht gepackten
Batch-Zustand charakterisiert und nicht die isoliert gepacte Einzeloperation.

**Verifikation nach dem Lauf.** Alle sechs Sessions über `analyze_trace` aus
unverändertem Manifest und Trace neu berechnet und bytegleich zum gespeicherten
Resultat; Study über `analyze_study` neu berechnet und über `validate_study_result`
validiert, ebenfalls bytegleich. Gesamtsuite `266` Tests und `2.265` Subtests grün
(`86,40 s`), Dashboard-Suite zusätzlich `6/13` grün. Socketfreier
Dashboard-Snapshot: Total `10`, `legacy_h0_warmup_observation=3`,
`paced_session=6`, `paced_study=1`, Status `h01_session_complete=6`,
`h01_complete_unresolved=1`, `legacy_observation=3`, Revision
`a2d1b2469e21f01de04e03b747ac897bb602059d5ec5ceeb098dcba5b03b4e1b`.

**Grenze.** H0 bleibt unverändert. Es gibt keine Reklassifikation, keine Promotion,
keine Performanceaussage und keine Aussage über Self-Optimization oder
Hardware-Generalisation. `aa_gpu`, H1 und H2 bleiben unberührt und weiterhin
freigabepflichtig.

### H0.1 — Nachanalyse der 480 Main-Samples (21.08.2026)

Rein deskriptive Read-only-Auswertung der bereits aufgezeichneten sechs Sessions.
Es wurde nichts neu gemessen, keine Schwelle verändert und kein Bundle geschrieben.
Die Study bleibt `h01_complete_unresolved`.

- **Der Cooldown-Effekt ist der einzige systematische Befund.** Das erste
  Main-Sample nach den 20 Sekunden Cooldown liegt in **allen sechs** Sessions bei
  `11,4`–`15,2 ms` gegen einen Median von `7,09 ms` der übrigen `474` Samples. Das ist
  reproduzierbar über alle Replikate. Die Ursache ist nicht gemessen.
- **Es gibt keinen Drift.** Über die vier Sessionviertel liegt der Median bei
  `7,33 / 7,28 / 7,13 / 6,92 ms`. Der `trend`-Gate-Fail in 5/6 Sessions ist damit kein
  gerichteter Drift, sondern die Reaktion des Theil-Sen-Schätzers auf eine stark
  ausreißerbehaftete Residuenverteilung.
- **Die Streuung ist der Befund, nicht einzelne Ausreißer.** `80` von `480`
  Main-Samples (`16,7 %`) liegen über dem `1,5`-fachen Median; `p90/Median` ist über
  alle sechs Sessions konsistent `1,52`–`1,86`. Position im Viererblock und
  Pacing-Label erklären das nicht: Blockpositionen liegen bei `6,94`–`7,38 ms`,
  `short_50ms` bei `6,95 ms` gegen `long_750ms` bei `7,36 ms`.
- **Aggregation auf Viererblöcke rettet die Stabilität nicht.** Die 20 Blockmediane
  einer Session streuen weiterhin mit einer Spanne von `62`–`78 %` ihres Medians.
- **Between-Session-Varianz.** Session-Mediane `6,34 / 6,71 / 6,91 / 6,97 / 8,42 /
  8,67 ms`, Mittel `7,34 ms`, Standardabweichung `0,97 ms`, Variationskoeffizient
  `13,2 %`, Spanne `37 %`.

**Abgeleitete Nachweisgrenze für H1.** Bei einem Vergleich von drei gegen drei
Sessions im gepacten Modus beträgt der Standardfehler der Differenz rund `0,79 ms`.
Ein Effekt muss damit grob `1,6 ms` überschreiten, also etwa **`21 %` des Medians**,
um vom Untergrundrauschen unterscheidbar zu sein. Das ist eine grobe Näherung aus
sechs Sessions und kein Signifikanztest; als Größenordnung ist es belastbar.

Zum Vergleich derselbe Messpunkt im dichten 32er-Batch von Run22: Median `2,032 ms`
bei Standardabweichung `0,142 ms`, Variationskoeffizient `7,0 %`. Der Batch-Modus ist
also rund achtfach stabiler als die isoliert gepacte Einzeloperation, charakterisiert
aber einen Dauerlastzustand statt verteilter Nutzung.

### H0.1-Guard — Abdeckungslücke erkannt und geschlossen (21.08.2026)

Ausgangspunkt war die Meldung, der H0.1-Guard sei erfolgreich. Das ist zutreffend
und war auch nach dem Bau des Ausführungspfads reproduzierbar: `status=pass`,
`57` Tests, `2.244` Subtests, `0` blockierte Importe, `0` Socketkonstruktionen,
Exit `0`.

**Die Lücke.** `tools/run_h01_guard.py` fuhr sieben fest gelistete Testmodule.
Das neue `tests/test_h01_runner.py` war in dieser Liste nicht enthalten. Ein grüner
Guard sagte damit nichts über den neu gebauten Ausführungspfad aus, las sich aber
wie eine vollständige Freigabe.

**Warum das Modul nicht einfach aufgenommen werden kann.** Der Guard wertet
`not finder.attempts` als Teil seiner Pass-Bedingung: bereits ein blockierter
Import-*Versuch* macht ihn rot. `friday_h01/provenance.py` fragt über
`importlib.util.find_spec` gezielt nach NumPy und MLX, um deren Versionen in die
Environment-Identität aufzunehmen. Unter dem Guard wird genau diese Anfrage
blockiert, was `42` Versuche erzeugt. Das ist kein Vertragsbruch des Runners,
sondern die notwendige Folge davon, dass die Umgebungsidentität die installierten
Paketversionen enthalten muss.

**Manuell nachgestellte Guard-Bedingungen für den Runner.** `tests.test_h01_runner`
wurde unter identischem MetaPath-Blocker und Socket-Audit ausgeführt:
`16/16` Tests grün, `0` Socketkonstruktionen, NumPy und MLX zu keinem Zeitpunkt in
`sys.modules`. Auch der reine Import von `friday_h01.runner`, `friday_h01.cli` und
`friday_h01.provenance` läuft unter dem Blocker ohne einen einzigen Versuch durch;
die MLX-/NumPy-Importe sind ausschließlich lazy im Messpfad.

**Härtung.** `tools/run_h01_guard.py` führt jetzt eine explizite
`H01_EXCLUDED_TEST_MODULES`-Liste mit Begründung je Modul, meldet sie als
`excluded_test_modules` im JSON-Report, und bricht mit
`status=configuration_error` und Exit `2` ab, sobald ein `tests/test_h01_*.py`
existiert, das weder geführt noch begründet ausgeschlossen ist. Verifiziert: eine
angelegte Probedatei erzeugt exakt diesen `configuration_error`; nach Entfernen ist
der Guard wieder `pass`.

**Nebenbefund mit Identitätswirkung.** `environment_sha256` hängt davon ab, ob
NumPy und MLX auffindbar sind. Unter Guard-Bedingungen kippt `available` auf
`false` und der Digest ändert sich. Das ist vertragskonform, weil eine andere
Umgebung eine andere Identität haben muss, bedeutet aber: Ein produktiver
Sessionlauf darf niemals unter dem Import-Blocker stattfinden. Praktisch besteht
kein Risiko, da der Guard ausschließlich Tests fährt und keine Session aufzeichnet.

`tools/run_h01_guard.py` gehört nicht zur geschlossenen `_CODE_FILES`-Liste von
H0.1. Der `code_sha256` blieb daher bei
`f66e4b5a2444643fb375a098398bbd3829d717a7b956e62f46a6a54617986e94`; die sechs
Sessions und die Study bleiben unverändert gültig. Gesamtsuite nach der Härtung:
`272` Tests und `2.278` Subtests grün.

### H1-Vorregistrierung als Entwurf angelegt (21.08.2026)

Nutzerauftrag: fortfahren, zusätzlich ausdrücklich **Hardware schonen**. Neu ist
`docs/H1_VORREGISTRIERUNG_ENTWURF.md`. Es ist ein Entwurf, autorisiert keinen Lauf
und keine Installation und benötigt eine Nutzerfreigabe.

Ausdrücklich **nicht** getan: keine H0.1-Schwelle, kein Seed, keine Samplezahl und
kein Gate verändert. Die Study bleibt `h01_complete_unresolved`. Ein Anpassen der
Grenzen nach Datensicht ist durch `docs/H01_PACED_TRAJECTORY_SPEC.md` untersagt und
hätte das Ergebnis wertlos gemacht.

**Inhaltliche Kernpunkte des Entwurfs.**

- Aus H0.1 feststehend: Nachweisgrenze im gepacten Modus rund `21 %`. Damit ist
  dieser Modus für H1 als Primärentscheidung untauglich, weil übliche
  Kernel-Optimierungen darunter liegen.
- Ausdrücklich **nicht** feststehend: eine Nachweisgrenze für den dichten
  Batch-Modus. Das gemessene `CV = 7,0 %` ist Within-Batch-Präzision, nicht
  Between-Run-Reproduzierbarkeit. Die vier vorhandenen `eager_baseline`-Runs haben
  unterschiedliche `code_sha256` und taugen nicht als Replikate. Diese Unterscheidung
  ist der Grund für den zweistufigen Ablauf.
- Fester Ablauf **A/A vor A/B**: Stufe 1 misst ohne jede Optimierung die
  Between-Run-Streuung und leitet daraus `MDE = 2 × s_between × sqrt(2/k)` ab. Erst
  nach Eintragen dieser Zahl darf Stufe 2 einen Kandidaten prüfen. Ein Effekt unter
  `MDE` ist ein Nullbefund, kein tendenzieller Gewinn. Die Kandidatenzahl steht vor
  Stufe 2 fest; eine Erhöhung ist eine neue Vorregistrierung.
- Der Cooldown-Effekt aus H0.1 ist als bekannte Störgröße geführt: feste, vorab
  registrierte Warmup-Verwerfung nach jeder Pause, für Baseline und Kandidat
  identisch. Asymmetrisches Warmup macht den Lauf ungültig.

**Hardwareschonung, quantitativ begründet.** Gemessene Belastung des H0.1-Laufs:
`5,26 s` GPU-Arbeit über `396,9 s` Wall, Duty-Cycle `1,33 %`, längste
ununterbrochene Last `26,8 ms`. Run22 zum Vergleich: `2,06 s` GPU-Arbeit in `30`
Blöcken, innerhalb der Blöcke ohne Pausen. H1 kann durch Kandidatensuche deutlich
mehr Last erzeugen, deshalb sind harte Budgets registriert: GPU-Arbeit `≤ 120 s` je
Lauf, ununterbrochene Last `≤ 2 s` mit Pflichtpause `≥ 4 s`, Duty-Cycle `≤ 25 %` über
jedes `60 s`-Fenster, Wall `≤ 20 min` je Lauf, Cooldown `≥ 60 s` zwischen Kandidaten,
Netzbetrieb verpflichtend. Jede Überschreitung bricht fail-closed ab und verwirft den
Lauf, statt ihn zu kürzen.

Netzbetrieb ist dabei nicht nur Schonung, sondern Messanforderung: Auf Akku begrenzt
macOS das GPU-Power-Budget, ein Batterielauf wäre nicht vergleichbar. Eine
Temperaturschwelle ist bewusst **nicht** registriert, weil sie nicht messbar ist —
`ProcessInfo.thermalState` hat keine stdlib-Bindung und `powermetrics` benötigt
erhöhte Rechte. Die Budgets begrenzen stattdessen die Ursache des Wärmeeintrags.

Verifikation nach der Änderung: Gesamtsuite `272` Tests und `2.278` Subtests grün,
H0.1-Guard `pass` mit Exit `0`, `code_sha256` unverändert bei `f66e4b5a…6e94`. Das
Dokument liegt außerhalb der geschlossenen `_CODE_FILES`-Liste; die sechs Sessions
und die Study bleiben gültig.

### Launcher-Sicherheitsentscheidung und Start der A/A-Kalibrierung (21.08.2026)

**Sicherheitsentscheidung.** Das Projekt führte seit dem 20.08.2026 einen offenen
Punkt `AWAITING USER APPROVAL` für den Launcher. Der Sachstand wurde vor der Frage
vollständig geprüft: Der eigentliche Fix ist bereits implementiert — der lexikalische
venv-Launcher wird an `Popen` übergeben und Owner, Modus, Typ, Device und Inode
werden unmittelbar vor dem Spawn geprüft (`friday_h0/supervisor.py`). Offen war
ausschließlich die **TOCTOU-Restlücke**: `Popen` ist auf Darwin nicht fd-gebunden,
sodass zwischen Prüfung und `exec` ein schreibberechtigter Angreifer die
Interpreter-Datei austauschen könnte. Eine vollständige Absicherung verlangt
`fexecve` über einen C-Helper und wäre eine neue Architekturentscheidung.

Dem Nutzer wurde die Lage mit beiden Optionen vorgelegt. **Entscheid: Restrisiko
akzeptiert.** Begründung im Bedrohungsmodell: Wer Schreibrechte auf
`.venv/bin/python` besitzt, kann ohnehin beliebigen Code als dieser Benutzer
ausführen; auf einem lokalen Einzelnutzergerät fügt die Lücke praktisch keine
Angriffsfläche hinzu. Der Punkt ist damit entschieden und nicht mehr offen.

Bei dieser Gelegenheit offengelegt: Der bereits ausgeführte H0.1-Lauf verwendete
denselben Mechanismus über `subprocess.run([sys.executable, ...])` in
`friday_h01/cli.py`, dort sogar **ohne** die Identitätsprüfung, die H0 durchführt.
H0 ist an dieser Stelle strenger als der H0.1-Runner. Die Frage war also nicht
spezifisch für `aa_gpu`, sondern nur bis dahin nicht gestellt worden.

**Zwei Korrekturen am H1-Entwurf, beide vor jeder H1-Messung.**

1. `k` ist **nicht** frei wählbar. Der A/A-Nullpfad ist in
   `docs/PHASE1_MATMUL_SPEC.md` Abschnitt 5.3.1 bereits vorregistriert: exakt drei
   Charakterisierungs- und drei Bestätigungsprozesse, je 30 gepaarte Blöcke, feste
   Fixture-/Order-Seeds `0xAA1A2026+i` bzw. `0xAA0D2026+i` und Bootstrap-Seeds
   `0xAA052026`/`0xAA052126`. Damit gilt `k = 3` je Set. Die ursprüngliche Frage nach
   `k = 3` gegen `k = 5` war gegenstandslos; eine Änderung wäre ein Bruch der
   bestehenden H0-Vorregistrierung gewesen. Es wurde daher nichts neu gebaut.
2. Das Budget „ununterbrochene GPU-Last" wurde von `2 s` auf `6 s` korrigiert. Ein
   A/A-Prozess erzeugt strukturbedingt rund `4,1 s` zusammenhängende Last (30 Blöcke
   à `68 ms` über zwei Arme). Der ursprüngliche Wert war ohne Kenntnis dieser
   Blockstruktur gewählt. Die Korrektur erfolgte vor jeder H1-Messung und ohne
   Sichtung von H1-Daten; nach dem ersten Lauf wäre sie unzulässiges Tuning gewesen.

**Neu: `tools/run_h0_aa.py`.** Ein reiner Sequenzer, der die sechs vorregistrierten
Prozesse nacheinander startet, `≥ 60 s` Cooldown zwischen ihnen einhält, die
Wall-Obergrenze von `20 min` prüft und den Lauf ablehnt, sobald das Gerät nicht am
Netz hängt. Er fügt dem A/A-Vertrag nichts hinzu und zeichnet kein eigenes Ergebnis
auf. Er liegt außerhalb der geschlossenen Code-Listen von H0 und H0.1; die
`code_sha256` beider Phasen bleiben unverändert (H0 `101cdadf…`, H0.1 `f66e4b5a…`).

**Vorabprüfung.** H0-Provenienz identisch mit Run22 (`code_sha256=101cdadf…`), die
A/A-Läufe sind damit echte Replikate derselben Codebasis. Netzbetrieb bestätigt.
Erwartete Belastung: rund `25 s` GPU-Arbeit gesamt gegen ein Budget von `≤ 120 s`.

### A/A-Kalibrierung ausgeführt — Ergebnis und zwei eigene Fehler (21.08.2026)

Nutzerfreigabe für Stufe 1 erteilt. Ausgeführt wurde der bereits vorregistrierte
H0-A/A-Nullpfad aus `docs/PHASE1_MATMUL_SPEC.md` 5.3.1 über den neuen Sequenzer
`tools/run_h0_aa.py`. Keine Installation, kein Download, keine Vertragsänderung an
H0 oder H0.1. `code_sha256` H0 `101cdadf…`, H0.1 `f66e4b5a…`, beide unverändert.

**Zwei Fehler im Sequenzer, beide von mir, beide korrigiert.**

1. Erster Abbruch nach `characterization[0]`: Der Sequenzer wertete jeden
   Exit-Code `!= 0` als Fehler. `friday_h0.runner.result_exit_code` liefert jedoch
   `0` nur bei `action=promoted`; ein einzelner A/A-Prozess kann per Design nicht
   promoted sein, weil die Entscheidung erst aus der Aggregation entsteht. Exit `10`
   ist dort der Normalfall. Der Prozess selbst war gültig
   (`status=completed`, `classification=measurement_complete`, `error=null`, beide
   Arme mit je `30` Blöcken). Korrigiert: Erfolg wird am aufgezeichneten Status
   geprüft, nicht am Exit-Code.
2. Zweiter Abbruch vor jeder Messung: `ModuleNotFoundError: friday_h0`, weil das
   Skript aus `tools/` ohne Projektwurzel im `sys.path` startete. Es wurde nichts
   gemessen. Korrigiert.

Zusätzlich erhielt der Sequenzer eine Resume-Funktion: bereits aufgezeichnete
Prozesstupel werden übersprungen statt wiederholt. Das ist zwingend, weil die
`run_id` deterministisch aus der Provenienz folgt und der Store append-only ist.

**Ein echter fail-closed Befund.** `characterization[2]` endete nach `1,98 s` mit
`status=invalid`, `error.code=warmup_unstable`: „last five warmup block gate values
are not stable after 16 blocks". Die letzten fünf Warmup-Gate-Werte lagen bei
`4,31 / 5,93 / 5,98 / 7,14 / 5,62 ms`, eine Spanne von `47,8 %` gegen die
registrierte Grenze von `5 %`. Das ist derselbe Befund wie historisch bei Run21 und
kein Implementierungsdefekt; die Ursache im Gerät ist nicht gemessen. Der Prozess
ist nicht wiederholbar, da `run_id` und Store dies fail-closed verhindern.

**Ergebnis über die fünf gültigen Prozesse.**

| Prozess | Status | `R_s` | im Band `[0.95,1.05]` | Baseline-Median |
| --- | --- | ---: | :---: | ---: |
| characterization[0] | completed | `0,9945` | ja | `2,759 ms` |
| characterization[1] | completed | `0,9956` | ja | `4,588 ms` |
| characterization[2] | **invalid** | – | – | – |
| confirmation[0] | completed | `1,0210` | ja | `3,190 ms` |
| confirmation[1] | completed | `1,0099` | ja | `3,109 ms` |
| confirmation[2] | completed | `0,9886` | ja | `3,403 ms` |

- Gepaarte Session-Ratios `R_s`: Mittel `1,00192`, Standardabweichung `0,01324`,
  **CV `1,32 %`**. Alle fünf liegen im registrierten Band.
- Ungepaarte Baseline-Mediane: Mittel `3,410 ms`, Standardabweichung `0,698 ms`,
  **CV `20,5 %`**.

**Der zentrale Befund.** Aus denselben Läufen folgt je nach Schätzer
`MDE = 2,16 %` (gepaart) oder `MDE = 33,4 %` (ungepaart) — ein Faktor von rund `15`.
Beide Arme eines Blocks erleben denselben Störuntergrund, der sich im Quotienten
herauskürzt; ungepaart bleibt er vollständig stehen. Das erklärt zugleich
rückblickend das H0.1-Ergebnis: H0.1 misst eine einzelne Operation ohne Vergleichsarm
und ist damit strukturell ungepaart, weshalb dort das Rauschen dominierte.

**Eigener Fehler im H1-Entwurf, korrigiert.** Die erste Fassung definierte
`MDE` über die Standardabweichung der *ungepaarten Lauf-Mediane*. Das war falsch.
Der gepaarte Schätzer `R_s = exp(median_b(log(t_B / t_A)))` ist in
`docs/PHASE1_MATMUL_SPEC.md` 5.3.1 seit dem 19.08.2026 vorregistriert, also vor jeder
A/A-Messung. Die Korrektur bringt den Entwurf mit der bestehenden Vorregistrierung in
Übereinstimmung und wählt keinen Schätzer nach Datenlage; sie ist als Korrektur im
Dokument selbst ausgewiesen.

**Grenzen, die bestehen bleiben.**

- Die vertragskonforme Aggregation ist **nicht** ausgeführt:
  `load_and_aggregate_h0_aa` verlangt exakt sechs `aa_gpu`-Runs und lehnt mit
  `RunnerError: A/A loader requires exactly six aa_gpu runs` ab. Es gibt kein
  formales A/A-Gate-Ergebnis und keinen Bootstrap.
- Die `2,16 %` sind eine eigene Rechnung über fünf Prozesse, kein registriertes Gate.
  Das offizielle Gate verlangt zusätzlich ein hierarchisches 10.000er-Bootstrap-KI
  innerhalb `[0.98, 1.02]`.
- `MDE` ist damit **noch nicht eingefroren**. Stufe 2 (A/B) bleibt gesperrt.
- H0 und H0.1 sind unverändert; keine Reklassifikation, keine Promotion, keine
  Performanceaussage.

### H1 — erste bestätigte Optimierung: Dispatch-Plan (21.08.2026)

Nutzerziel: eine nachweisbare Ersparnis. Ergebnis: **bestätigt.** Kein Download,
keine Installation, kein Modell. `code_sha256` H0 `101cdadf…` und H0.1 `f66e4b5a…`
blieben unverändert; alle neuen Dateien liegen außerhalb beider geschlossener Listen.

**Der Kandidat.** Zwei Ausführungspläne für dieselbe Arbeit:
`serial` dispatcht eine Matmul, wartet auf die GPU, wiederholt das `N`-mal;
`batched` dispatcht alle `N` und wartet genau einmal am Ende. Die Arithmetik ist
identisch, die Ergebnisse sind **bytegleich** — geprüft gegen eine Referenz vor
jeder Zeitmessung, mit Abbruch bei jeder Abweichung.

**Warum die Vorarbeit nötig war.** Ungepaart gemessen erschien `mx.compile` mit
`0,724x`, also `−27,6 %`. Gepaart gemessen ergab derselbe Kandidat `R = 1,0019`
mit `95%-KI [0,9990, 1,0047]` — **kein Effekt**. Der scheinbare Gewinn war
vollständig das Rauschen der ungepaarten Messung. Ohne den A/A-Befund vom selben
Tag wäre dieser Nullbefund als `28 %`-Optimierung ins Projekt gewandert.

**Ausgeschlossene Fehlerquelle.** Bei identischen Operanden zeigte Batching
`0,695x`; mit `16` paarweise verschiedenen Operanden nur noch `0,81x`. Die Differenz
ist Deduplizierung identischer Teilausdrücke. Alle berichteten Zahlen verwenden
deshalb ausschließlich verschiedene Operanden. Gegenprobe: vier verschiedene Matmuls
brauchen `3,64x` der Zeit einer einzelnen, die Arbeit wird also real geleistet.

**Bestätigtes Ergebnis** (`tools/measure_dispatch_plan.py`, `N = 8`, `5` Replikate
à `30` Blöcke, hierarchisches 10.000er-Bootstrap, Seed `0xB00252026`):

- `R = 0,8531`, Effekt **`−14,7 %`**, `95%-KI [0,8263, 0,8777]`
- Replikate `0,861 / 0,853 / 0,838 / 0,857 / 0,853`
- `serial 2,572 ms/Matmul` gegen `batched 2,212 ms/Matmul`, Ersparnis `0,360 ms`
- Correctness `byte_identical`, GPU-Arbeit `5,8 s` gegen Budget `120 s`
- Verdikt `effect_confirmed`, `MDE = 5 %` klar überschritten

**Kurve über `N`** (je `3` Replikate à `25` Blöcke), alle bestätigt:

| `N` | `R` | Effekt | `ms/Matmul` |
| ---: | ---: | ---: | ---: |
| 2 | `0,9212` | `−7,9 %` | `2,362` |
| **4** | **`0,8262`** | **`−17,4 %`** | **`1,939`** |
| 8 | `0,8445` | `−15,6 %` | `1,974` |
| 16 | `0,8806` | `−11,9 %` | `1,966` |

Das Optimum liegt bei `N = 4`; darüber ist der Synchronisations-Overhead amortisiert
und die absolute Zeit bleibt bei rund `1,95 ms/Matmul` stehen.

**Zwei geprüfte Nullbefunde.** Prätransponiertes `B` ist `+3,1 %` langsamer,
`mx.einsum` `+0,6 %` langsamer, ein eigener GPU-Stream `±0 %`. Ein echter
3D-Batch-Matmul über einen gestapelten Tensor bringt gegenüber dem Loop-Batching
nur `−3,9 %` (`N=4`) bzw. `−1,8 %` (`N=8`), beide Konfidenzintervalle enthalten
`1,0`: **kein bestätigter Zusatzeffekt**. MLX bündelt die Schleifenvariante bereits
selbst; die Optimierung ist damit ausgereizt.

**Ehrliche Einordnung der Reichweite.** Der Gewinn ist keine Kernel-Optimierung —
der Matmul-Kernel bleibt unverändert. Er entfernt vermeidbare Synchronisation. Die
`serial`-Baseline ist ein realistischer Anti-Pattern (jedes `mx.eval` in einer
Schleife erzeugt sie), aber erfahrener MLX-Code vermeidet sie ohnehin. Der Befund
gilt für **unabhängige** Operationen; bei datenabhängigen Ketten ist weniger zu
holen. Es ist keine Aussage über Modelle, Transformer-Inferenz oder andere Geräte.

**Neu und verifiziert.** `tools/measure_dispatch_plan.py` mit `--execute`-Gate
(ohne Flag Exit `78` vor jedem MLX-Import), Netzbetriebspflicht, GPU-Budget,
Correctness-Gate vor jeder Zeitmessung und `--self-check` ohne GPU.
`tests/test_dispatch_plan.py` mit `12` Tests sichert die Entscheidungslogik offline
ab, darunter der Fall, dass ein einzelner extremer Block keinen Nullbefund in einen
Fund verwandeln darf. Gesamtsuite `284` Tests und `2.278` Subtests grün,
H0.1-Guard `pass`.

### Cooldown-Effekt isoliert und charakterisiert (21.08.2026)

Nutzerauftrag: den einzigen in H0.1 reproduzierbaren Befund isolieren. Keine
Installation, kein Download, kein Modell. `code_sha256` H0 `101cdadf…` und H0.1
`f66e4b5a…` unverändert; alle neuen Dateien liegen außerhalb beider geschlossener
Listen. GPU-Arbeit `3,2 s` je Lauf gegen Budget `120 s`.

**Design.** Jede Wiederholung ist **intern gepaart**: gemessen wird das erste
Sample gegen den eingeschwungenen Zustand *derselben* Wiederholung, wodurch sich
gemeinsamer Drift herauskürzt. Die Pausenlängen werden in deterministisch
SHA-256-gemischter Reihenfolge besucht, damit ein Drift über den Lauf sich nicht
als Pausenlängeneffekt tarnen kann.

**Dosis-Wirkungs-Beziehung** (`tools/measure_cooldown_effect.py`, `10`
Wiederholungen je Pause, `12` Samples je Burst):

| Pause | erstes Sample | Spanne | Exzess |
| ---: | ---: | :---: | ---: |
| `0,00 s` | `0,94x` | `0,67`–`1,24` | `0,00` |
| `0,05 s` | `1,50x` | `0,63`–`1,97` | `2,04` |
| `0,25 s` | `1,89x` | `1,08`–`3,20` | `2,42` |
| `0,75 s` | `1,91x` | `1,19`–`3,34` | `2,98` |
| `2,00 s` | `3,67x` | `2,06`–`5,04` | `4,09` |
| `5,00 s` | `3,97x` | `3,62`–`6,47` | `4,42` |
| `20,00 s` | `4,12x` | `3,04`–`21,95` | `5,12` |

Der Exzess ist die verlorene Zeit in Sample-Äquivalenten. Die Kurve ist streng
monoton und beginnt bei `0,00` ohne Pause — der Effekt ist damit sauber an die
Pausenlänge gebunden und sättigt bei rund `4x` ab etwa `2 s`.

**Ursache: überwiegend GPU-Taktung, nicht der Allocator.**

- Keep-Alive-Test, gepaart über `10` Paare bei `5 s` Pause: Eine Idle-Pause ergibt
  `R = 4,02`, dieselbe Pause mit periodischer Mini-Matmul nur `R = 2,53`.
  Verhältnis `0,487` mit `95%-KI [0,311, 0,762]`. Beschäftigt gehaltene GPU
  halbiert den Effekt; Taktung ist damit als wesentlicher Anteil belegt.
- Der MLX-Allocator scheidet aus: `get_cache_memory()` bleibt über die Pause
  konstant bei `8,6 MB`. Ein erzwungenes `clear_cache()` erzeugt zwar einen
  eigenen Effekt (`R = 1,71` gegen `1,01` unbehandelt), aber dieser Zustand tritt
  während einer Pause gar nicht ein.

**Keep-Alive lohnt sich netto nicht.** Sieben Dosierungen von `128²` bis `1024²`
und `100 ms` bis `500 ms` Takt wurden gemessen. Die beste Variante senkt das erste
Sample von `10,05 ms` auf `5,11 ms`, kostet aber `14,44 ms` eigene GPU-Zeit; netto
bleibt ein Verlust von `9,50 ms`. Alle sieben Varianten sind netto negativ. Als
Optimierung unbrauchbar, als Ursachennachweis wertvoll.

**Der Effekt erklärt H0.1 nicht.** Post-hoc und rein deskriptiv nachgerechnet: Ohne
die ersten sechs Main-Samples bestünde `trend` weiterhin `1/6`, `changepoint`
`0/6` und `tail` `0/6` — identisch zum realen Ergebnis, bei `changepoint`
teilweise sogar schlechter. Die H0.1-Instabilität stammt von über die gesamte
Session verteilten Ausreißern, nicht vom Cooldown. Es sind zwei unabhängige
Phänomene. Die Study bleibt unverändert `h01_complete_unresolved`.

**Zwei eigene Metrikfehler, beide korrigiert und offengelegt.**

1. Erste Fassung prüfte zweiseitig auf `±5 %` um den Steady-State. Ergebnis:
   `11` kontaminierte Samples nach einer Pause von `0 s` — offensichtlich falsch.
   Ursache: Ein Cooldown macht langsamer, nie schneller; nach unten streuende
   Samples sind Jitter. Korrigiert auf eine einseitige Prüfung.
2. Auch einseitig blieb die Cutoff-Regel instabil (`9/8/6/6/6/3/5`). Grund ist
   grundsätzlich: Dieses Gerät besitzt keinen Steady-State, der ein festes Band
   einhält — genau der H0.1-Befund mit `16,7 %` Ausreißern. Eine Cutoff-Regel
   misst hier etwas, das nicht existiert. Ersetzt durch den kumulierten Exzess,
   der ohne sauberen Cutoff auskommt und eine monotone Kurve liefert.

**Neu und verifiziert.** `tools/measure_cooldown_effect.py` mit `--execute`-Gate
(ohne Flag Exit `78` vor jedem MLX-Import), Netzbetriebspflicht, GPU- und
Wall-Budget, Reproduzierbarkeitsprüfung der Workload und `--self-check` mit `11`
Prüfungen ohne GPU. `tests/test_cooldown_effect.py` mit `15` Offline-Tests,
darunter ein Regressionstest gegen genau den Metrikfehler oben.

### Erster realer Modelltest — Gemma 3 1B (21.08.2026)

Nutzerfreigabe für Download und Installation lag vor, mit der Auflage, alles im
Projektordner zu halten. Umgesetzt: `HF_HOME` zeigt auf
`.friday-data/models`, Pakete gingen ins projektlokale `.venv`.

**Installation ohne Provenienzbruch.** `uv pip install mlx-lm` zog `24` Pakete
(`mlx-lm 0.31.3`, `transformers 5.15.1`, `tokenizers`, `safetensors` u. a.). Vorab
per Dry-Run geprüft und danach verifiziert: **`mlx` blieb bei `0.32.0` und `numpy`
bei `2.5.2`**. Da nur diese beiden in der Environment-Allowlist stehen, blieb
`environment_sha256` bei `74ca2dac…` und alle bisherigen Läufe bleiben
vergleichbar. `code_sha256` H0 `101cdadf…` und H0.1 `f66e4b5a…` ebenfalls unverändert.

**Modell.** `mlx-community/gemma-3-1b-it-4bit`, ID vorab über die HF-API
verifiziert statt geraten. Download `737 MB`, Ladezeit `83,1 s`, aktiver
GPU-Speicher `735,9 MB`. Die Generierung liefert sinnvolle Antworten; die Stufung
`1B` vor `4B` aus dem Entscheid vom 21.08. ist damit eingehalten.

**Baseline.** Folge-Token konstant bei rund `5,0 ms`, entsprechend etwa
`200 Token/s`. Diese Rate ist über alle Pausenlängen stabil.

**Der Cooldown-Effekt überträgt sich auf die Inferenz — auf die Time-to-First-Token.**
Gemessen mit `8` Wiederholungen je Pause in deterministisch randomisierter
Reihenfolge, `32` Generierungen, Laufzeit `350 s`:

| Pause | TTFT Median | Spanne | Folge-Token | Faktor |
| ---: | ---: | :---: | ---: | ---: |
| `0 s` | `205,1 ms` | `193`–`289` | `4,98 ms` | `1,00x` |
| `2 s` | `259,3 ms` | `235`–`294` | `5,00 ms` | `1,26x` |
| `10 s` | `266,3 ms` | `255`–`324` | `5,08 ms` | `1,30x` |
| `30 s` | `280,5 ms` | `233`–`749` | `5,02 ms` | `1,37x` |

Der Effekt betrifft ausschließlich den ersten Token; die Generierungsrate danach
ist unberührt. Das ist praktisch relevant, weil TTFT die spürbare Latenz einer
interaktiven Anwendung ist.

**Korrektur einer eigenen Zwischenzahl.** Ein erster Schnelltest mit nur `4`
Wiederholungen und ohne Randomisierung ergab `503,9 ms` bei `20 s` Pause, also
`+162 %`. Die saubere Messung liefert `+37 %`. Die erste Zahl war eine
Überschätzung aus zu kleiner Stichprobe und wird hiermit verworfen; maßgeblich ist
die Tabelle oben.

**Vorwärmen ist zweifach widerlegt.** Beide Varianten wurden gepaart gemessen:

- Fremde `512²`-Matmul alle `500 ms` über `20 s`: `R = 0,9151`, aber
  `95%-KI [0,5861, 1,4287]` — enthält `1,0`. Kosten `79,7 ms`, netto `−38,3 ms`.
- Forward-Pass durch das Modell selbst alle `2 s`: `R = 1,0416`,
  `95%-KI [0,8864, 1,2241]`. Kosten `507,2 ms`, netto `−518,5 ms`.

Keine der beiden Varianten überschreitet die `5 %`-Schwelle. Das deckt sich mit
dem Matmul-Befund: Vorwärmen kostet mehr, als es bringt. Die naheliegende
Hypothese, bei Inferenz kippe die Rechnung wegen des größeren Gewinns, ist damit
geprüft und **verworfen**.

**Grenze.** Das ist eine Latenzcharakterisierung, kein Performance- oder
Qualitätsnachweis des Modells. Es gibt keine Aussage über größere Modelle, andere
Geräte oder Durchsatz unter Last. H0, H0.1 und die H1-Optimierung bleiben unberührt.

### Stufe 2 — Gemma 3 4B (21.08.2026)

Nutzerauftrag: Stufe 2 der am 21.08. festgelegten Modellstufung ausführen. Modell
`mlx-community/gemma-3-4b-it-4bit`, Download `3,40 GB` in den Projektordner.
`code_sha256` H0 `101cdadf…`, H0.1 `f66e4b5a…` und `environment_sha256`
`74ca2dac…` blieben unverändert.

**Widerlegte eigene Annahme: es gibt keinen Vision-Tower-Offset im Speicher.**
Im Journal war am 21.08. festgehalten, der SigLIP-Vision-Tower belege bei reiner
Text-Inferenz Speicher, ohne zu rechnen, und sei als konstanter Peak-RSS-Offset im
Vertrag festzuschreiben. Das ist **falsch**. Die Messung:

| Bestandteil | auf Disk | Anteil | geladen |
| --- | ---: | ---: | --- |
| `language_model` | `2.560,8 MB` | `75,3 %` | ja |
| `vision_tower` | `833,7 MB` | `24,5 %` | **nein** |
| `multi_modal_projector` | `5,9 MB` | `0,2 %` | **nein** |
| gesamt | `3.400,4 MB` | | `2.560,8 MB` |

`mlx_lm.load` lädt ausschließlich den Sprachteil; `839,6 MB` liegen auf der Platte,
kosten aber keinen Arbeits- oder GPU-Speicher. Der aktive GPU-Speicher entspricht
exakt den geladenen `2.560,8 MB`. Die geplante Bestimmung des Tower-Offsets über
die Peak-RSS-Differenz zwischen Stufe 1 und Stufe 2 ist damit gegenstandslos; der
Anteil wurde stattdessen direkt aus dem safetensors-Index quantifiziert, was ohnehin
das sauberere Verfahren ist. Für Vision-Inferenz wäre `mlx-vlm` nötig, nicht `mlx-lm`.

**Baseline 4B.** Ladezeit `3,5 s` aus dem lokalen Cache, aktiver GPU-Speicher
`2.560,8 MB`, Prozess-Peak-RSS `1.633,6 MB`. Generierung liefert korrekte
Antworten. Folge-Token rund `11`–`12,8 ms`, also etwa `85 Token/s` gegenüber
rund `200 Token/s` bei `1B`.

**Cooldown-Effekt bei 4B bestätigt.** Gemessen mit `10` Paaren im direkten Wechsel
ohne Pause / nach `30 s` Pause, sodass Drift beide Arme gleich trifft:
`TTFT ohne Pause 304,8 ms`, `nach Pause 445,6 ms`, `R = 1,414` also `+41 %`,
`95%-KI [1,209, 1,653]`. Das Intervall liegt vollständig über `1,05`; der Effekt
ist bestätigt.

**Der Effekt skaliert nicht mit der Modellgröße.** `1B` zeigt `1,37x`, `4B` zeigt
`1,414x` bei jeweils `30 s` Pause. Trotz vierfacher Parameterzahl und rund
`3,4`-fachem Speicherbedarf ist die relative Verlangsamung praktisch gleich. Das
spricht dafür, dass der Cooldown eine Eigenschaft des Geräts ist und nicht der
Arbeitslast — konsistent mit dem Matmul-Befund, wo derselbe Effekt ohne jedes
Modell auftrat.

**Zweite Korrektur einer eigenen Zwischenzahl.** Ein erster 4B-Durchlauf mit `6`
Wiederholungen und über alle Pausenlängen randomisierter Reihenfolge ergab `4,16x`
bei `30 s`. Die saubere Wechselmessung ergibt `1,414x`. Ursache der Überschätzung:
Bei durchgehender Randomisierung folgten mehrfach lange Pausen unmittelbar
aufeinander, sodass kumulierte Leerlaufzeit statt der letzten Pause gemessen wurde.
Die `4,16x` sind verworfen; maßgeblich ist `1,414x`.

Damit sind in dieser Sitzung **zwei** eigene Zwischenzahlen durch sorgfältigere
Messung nach unten korrigiert worden (`503,9 ms` bei 1B, `4,16x` bei 4B). Beide
Male war die erste Zahl zu groß. Die Ursache ist in beiden Fällen dieselbe: zu
kleine Stichprobe bei einer rechtsschiefen Verteilung, in der Ausreißer nach oben
häufiger sind als nach unten. Für künftige Messungen gilt daraus: kein Befund aus
weniger als etwa zehn Wiederholungen, und Behandlungsarme im direkten Wechsel statt
frei randomisiert, wenn die Behandlung selbst eine Zeitkomponente hat.

**Grenze.** Latenz- und Speichercharakterisierung, kein Qualitäts- oder
Durchsatznachweis. Keine Aussage über Vision-Inferenz, größere Modelle oder andere
Geräte. H0, H0.1 und die H1-Optimierung bleiben unberührt.

### Self-Optimization-Loop gebaut und bestätigt (21.08.2026)

Nutzerauftrag: den Loop bauen. Damit ist erstmals nicht ein Mensch der Optimierer,
sondern ein geschlossener Mess-Entscheidungs-Kreis. Keine Installation, kein
Download. Provenienz unverändert (H0 `101cdadf…`, H0.1 `f66e4b5a…`,
Environment `74ca2dac…`).

**Aufbau** (`tools/optimization_loop.py`), drei Runden:

1. `explore` — feste Kandidatenmenge `N ∈ {2,4,8,16}`, jeweils gepaart gegen die
   serielle Baseline gemessen.
2. `refine` — der Loop schlägt **selbst** Nachbarn des Überlebenden vor und misst
   sie. Das macht ihn zur Suche statt zur Checkliste.
3. `confirm` — der Anführer wird unabhängig neu gemessen, `3` Replikate mit
   hierarchischem Bootstrap. Ein unrepliziertes Ergebnis wird als
   `no_confirmed_optimization` gemeldet, nicht als Fund.

Jeder Kandidat passiert vor jeder Zeitmessung ein Correctness-Gate: Ein
Ausführungsplan darf Arbeit umsortieren, aber kein einziges Bit ändern. Die
Statistik wird aus `measure_dispatch_plan` importiert statt neu geschrieben, weil
eine zweite Kopie eines Schätzers eine zweite Gelegenheit ist, ihn falsch zu haben.

**Erster Befund: der Loop war nicht reproduzierbar.** Drei Läufe ergaben nur
`1x optimization_confirmed`. In den beiden Fehlläufen gewann in der Exploration
`N=2` mit `R = 0,750` bzw. `0,741`; die unabhängige Bestätigung regressierte auf
`0,87`–`0,96` und fiel durch.

**Ursache: Winner's Curse.** Der Beste aus mehreren verrauschten Kandidaten ist
konstruktionsbedingt zu optimistisch geschätzt. Der Loop verhielt sich korrekt —
er bestätigte nicht —, fand dadurch aber meist nichts. Der Fehler lag in meiner
Rangfolge: Sortierung nach dem Punktschätzer wählt den glücklichsten Ausreißer.

**Korrektur: Rangfolge nach der Konfidenzobergrenze.** Statt „was sah einmal am
besten aus" fragt der Loop nun „was ist zuverlässig gut". Ein breites Intervall
wird damit genau so stark bestraft, wie seine Unsicherheit es rechtfertigt. Die
Schwelle `MDE = 5 %` blieb unverändert; geändert wurde ausschließlich die Auswahl
unter bereits bestehenden Kandidaten, nicht das Bestehenskriterium.

**Ergebnis nach der Korrektur: `3` von `3` Läufen bestätigt.**

| Lauf | gewählt | Effekt | Replikate | KI |
| ---: | :---: | ---: | --- | --- |
| 1 | `N=8` | `−13,60 %` | `0,864 / 0,860 / 0,879` | `[0,8429, 0,8918]` |
| 2 | `N=6` | `−11,13 %` | `0,889 / 0,892 / 0,858` | `[0,8454, 0,9052]` |
| 3 | `N=6` | `−14,11 %` | `0,858 / 0,891 / 0,859` | `[0,8473, 0,9059]` |

Alle drei Läufe wählten in der Exploration `N=8` als Anführer und verfeinerten zu
`{6,7,9,10}`. Der Loop konvergiert damit auf `N=6`–`8`.

**Autonomer Fund.** `N=6` und `N=7` kamen in der manuellen Suche vom selben Tag
nicht vor — dort waren nur `2, 4, 8, 16` geprüft worden. Der Loop hat diese
Kandidaten selbst vorgeschlagen, gemessen und einen davon bestätigt. Das ist der
Nachweis, um den es dem Projekt geht: nicht ein Mensch, der einen Kandidaten
prüft, sondern ein Verfahren, das den Raum selbst absucht und sein eigenes
Ergebnis anzweifelt, bevor es es meldet.

**Verifikation.** `tests/test_optimization_loop.py` mit `15` Offline-Tests,
darunter ein Regressionstest gegen genau den Winner's-Curse-Fehler oben.
`--self-check` mit `9` Prüfungen ohne GPU. `--execute`-Gate wie überall
(ohne Flag Exit `78`). Gesamtsuite `314` Tests und `2.278` Subtests grün,
H0.1-Guard `pass`, GPU-Arbeit rund `5 s` je Lauf gegen Budget `120 s`.

**Grenze.** Der Loop sucht in einem **festen, von Hand definierten** Raum von
Ausführungsplänen. Er generiert keinen Code und schreibt keine Kernel. Das wäre
H2 und ein eigenes Sicherheitsthema. Was gezeigt ist: der geschlossene
Mess-Entscheidungs-Kreis funktioniert, verwirft Nullbefunde zuverlässig und
bestätigt reproduzierbar.

### Projekt für Dritte nutzbar gemacht (21.08.2026)

Nutzerauftrag: fertigstellen, sodass andere es einsetzen können. Keine
Installation, kein Download. Provenienz unverändert (H0 `101cdadf…`,
H0.1 `f66e4b5a…`, Environment `74ca2dac…`).

**Ausgangslage für einen Fremden war schlecht.** Das README beschrieb noch den
Stand vor allen Messungen und forderte auf, „Phase 0/1 zu implementieren" — längst
erledigt. Fünf Werkzeuge lagen ohne gemeinsamen Einstieg in `tools/`. Die
Dokumentation umfasste rund `7.000` Zeilen, davon `2.540` allein im Arbeitsjournal;
niemand liest das, um einen Befund nachzuvollziehen.

**Neu: `tools/friday.py` als einziger Einstieg.**

- `list` zeigt die fünf Werkzeuge mit Zweck.
- `doctor` prüft Python-Version, MLX samt Metal-Antwort, NumPy, optional `mlx-lm`,
  Netzbetrieb und Plattenplatz und benennt, was fehlt.
- `<tool> [args]` reicht an das jeweilige Werkzeug durch.

Ein Detail, das beim Bau auffiel: `guard` und `aa` haben ein parameterloses
`main()`. Ein naives Durchreichen scheiterte mit `TypeError`. Der Dispatcher prüft
jetzt die Signatur und lehnt überflüssige Argumente mit Exit `64` ab, statt sie
stillschweigend zu verwerfen — ein Tippfehler soll sichtbar sein.

**Neu: `docs/ERGEBNISSE.md`.** Eine Seite mit allen belastbaren Befunden, den
sieben geprüften Nullbefunden, den Grenzen und den sechs Messregeln. Jeder Befund
nennt das Kommando, mit dem er reproduziert wird. Das ersetzt für Einsteiger die
Journal-Lektüre.

**README neu geschrieben.** Der Kernbefund steht jetzt im ersten Absatz: Ungepaart
messen ist auf diesem Gerät nahezu wertlos, und `mx.compile` ist das konkrete
Beispiel dafür (`−27,6 %` ungepaart gegen `+0,2 %` gepaart). Dazu Schnellstart,
Werkzeugtabelle, die sechs Messregeln, die Hardwarebudgets und die Grenzen.

**Beobachtung zum Loop, dokumentiert statt versteckt.** Ein vierter Lauf über die
neue CLI wählte `N=16` statt `N=6`/`N=8` — bestätigt mit `−11,08 %`. Das Optimum
ist ein breites Plateau: `N = 4` bis `16` liegen alle zwischen `−11 %` und
`−17 %`, und der Loop landet je nach Rauschen an unterschiedlichen Stellen darauf.
Stabil ist der Effekt, nicht der genaue Punkt. In `docs/ERGEBNISSE.md` ausdrücklich
erklärt, damit es niemand für Instabilität hält.

**Reproduzierbarkeit geprüft.** Keine absoluten Pfade in `tools/`, `friday_h0/`
oder `friday_h01/`. `scripts/bootstrap_apple.sh` arbeitet relativ zum
Repositorywurzelverzeichnis. `requirements-apple-silicon.txt` führt `mlx-lm` jetzt
als auskommentierte optionale Zeile mit dem Hinweis, vor der Installation
`uv pip install --dry-run` zu prüfen — würde dabei `mlx` oder `numpy` hochgezogen,
wären frühere Läufe nicht mehr vergleichbar.

**Verifikation.** `tests/test_friday_cli.py` mit `12` Tests und `34` Subtests
prüft unter anderem, dass jedes registrierte Werkzeug existiert und ein `main`
besitzt, dass alle messenden Werkzeuge ohne `--execute` mit Exit `78` abbrechen,
dass jeder `--self-check` ohne GPU besteht, und dass **jeder relative Link in
README und ERGEBNISSE tatsächlich auflöst**. Gesamtsuite `326` Tests und `2.312`
Subtests grün, H0.1-Guard `pass`.

### Vollständiger Testlauf und Optimierung des Projekts selbst (21.08.2026)

Nutzerauftrag: einmal vollständig testen und optimieren, bis nichts mehr geht;
das Projekt soll selbst schlank, schnell und effizient laufen. Keine Installation
außer `pytest-xdist`, kein Download. Provenienz unverändert (H0 `101cdadf…`,
H0.1 `f66e4b5a…`, Environment `74ca2dac…`).

**Testsuite von `90 s` auf `31 s`, Faktor `2,9`.** Zuerst gemessen statt geraten:
Ein einzelner Test (`test_memory_name_and_reason_are_closed_and_registered_fallbacks_are_accepted`)
brauchte `17,6 s`, also `20 %` der Gesamtzeit, weil er `16`-mal `aggregate_h0_aa`
mit vollem 10.000er-Bootstrap aufruft.

Die naheliegende Optimierung — das Bootstrap beschleunigen — wurde **verworfen**:
`friday_h0/aggregation.py` steht in der geschlossenen Code-Liste, jede Änderung
bricht `code_sha256` und trennt Run22 sowie alle A/A-Läufe von künftigen ab. Eine
Testlaufzeit ist das nicht wert.

Stattdessen `pytest-xdist`. Vorab per Dry-Run geprüft: zieht nur `execnet`, lässt
`mlx` und `numpy` unberührt, `environment_sha256` bleibt. Ergebnis über drei Läufe
stabil (`31,4`/`31,4`/`31,9 s`). Mehr Worker bringen nichts mehr — der `17,6 s`-Test
ist die untere Schranke, das ist Amdahl und keine Konfigurationsfrage. Festgelegt
in `pytest.ini`; sequenziell über `pytest -n 0` (nicht `-p no:xdist`, das entfernt
das Plugin, während `-n auto` in `addopts` stehen bleibt).

**Ein echter Sicherheitsfund.** Die systematische Gate-Prüfung ergab, dass `aa`
**kein** `--execute` besaß. Der Prüfaufruf startete daraufhin real einen
A/A-Lauf. Es wurde nichts aufgezeichnet, weil der Resume-Mechanismus alle sechs
Prozesstupel als vorhanden erkannte und nichts zu tun blieb — das war Glück, kein
Design. Verifiziert: weiterhin exakt `9` `aa_gpu`-Runs mit unveränderten
Zeitstempeln. Gate nachgerüstet; alle vier messenden Werkzeuge sind jetzt gesperrt.

Damit dieselbe Klasse von Lücke nicht wiederkommt, führt `ReleaseGateTest` nun
zwei Gruppen (`MEASURING_TOOLS`, `NON_MEASURING_TOOLS`) und prüft, dass ihre
Vereinigung **exakt** der Werkzeugregistrierung entspricht. Ein neues Werkzeug muss
eingeordnet werden, sonst schlägt die Suite fehl.

**Entdoppelung.** `require_ac_power` existierte in vier Varianten (drei Werkzeuge
plus `friday.py`), das Release-Gate in drei. Beide liegen jetzt in `tools/_bench.py`.
Begründung im Code festgehalten: Genau das Muster, das dupliziert war, ist das,
bei dem eine Kopie fehlte. Der dateiübergreifende Duplikatscan über `tools/` findet
danach nichts mehr.

**Aufgeräumt.** Ungenutzte Importe entfernt (`argparse` und `subprocess` in
`friday.py`, `subprocess` in zwei Messwerkzeugen), toter Boilerplate durch einen
`sys.path`-Eintrag ersetzt.

**Konsistenz hergestellt statt Versprechen abgeschwächt.** Das README sagt, jedes
messende Werkzeug biete `--execute` und `--self-check`. `aa` hatte kein
`--self-check`; statt die Doku zu relativieren, wurde eines ergänzt, das die
Sequenzer-Regeln offline prüft — darunter die Exit-Code-Regel, die schon einmal
falsch war.

**Neue Tests.** `+11` Tests: geteilte Vorbedingungen (Herkunft der Gate-Funktion,
nie werfendes Auslesen der Stromquelle), Sequenzer-Logik (`_process_succeeded` für
Exit `0`/`10`/Fehlercodes, `status=invalid` trotz toleriertem Code, leere Ausgabe),
und die Gruppenabdeckung der Werkzeugregistrierung.

**Verifikation.** `337` Tests und `2.322` Subtests grün in `31,3 s`. H0.1-Guard
`pass`. Alle vier Gates gesperrt, alle vier Self-Checks bestehen. End-to-End nach
dem Refactoring: `loop` meldet `optimization_confirmed` (`batched_5`, `−9,57 %`),
`dispatch` meldet `effect_confirmed` mit `correctness=byte_identical`.
Provenienz beider Phasen unverändert.

### Ausreißer aufgeklärt — der Störprozess ist charakterisiert (21.08.2026)

Rein deskriptive Auswertung vorhandener Daten (H0.1-Sessions und A/A-Läufe). Keine
neue Messung, keine GPU-Zeit, keine Vertragsänderung.

**1. Die Verteilung ist unimodal mit langem rechtem Schwanz.** Kein zweiter Modus,
kein An/Aus-Zustand — die Verlangsamung ist kontinuierlich. Von `480` Main-Samples
liegen `130` zwischen `0,75x` und `1,00x` des Medians, der Schwanz reicht bis `3x`.

**2. Die Ausreißer sind zufällig verteilt, nicht geclustert.** Runs-Test über alle
sechs Sessions: beobachtete Runs `19/15/20/14/17/24` gegen zufällig erwartete
`18,5/15,4/18,5/18,5/20,0/24,1`. Damit sind periodische Störungen und thermische
Cluster ausgeschlossen.

**3. Die Positionen korrelieren nicht zwischen Sessions.** Nur `4` von `80`
Positionen tragen in mindestens drei der sechs Sessions einen Ausreißer. Es gibt
also keine „schlechten Stellen" im Ablauf.

**4. Die Störung ist blockweit — sie trifft beide Arme gleichzeitig.** In den
A/A-Läufen sind baseline und candidate identisch, jede Abweichung ist reine
Störung. Über `150` Blöcke: `22` mit Ausreißer in **beiden** Armen gegen `4,1`
bei Unabhängigkeit erwartete — Faktor `5,4`. Korrelation im selben Block
`r = +0,525`.

**5. Die Störung überdauert Blockgrenzen. Zeitskala rund `340 ms`.**
Autokorrelation über den Blockabstand (je Block rund `68 ms`):

| Lag | Zeit | `r` |
| ---: | ---: | ---: |
| 1 | `68 ms` | `+0,576` |
| 2 | `136 ms` | `+0,474` |
| 3 | `204 ms` | `+0,287` |
| 4 | `272 ms` | `+0,228` |
| 5 | `340 ms` | `+0,124` |
| 6 | `408 ms` | `−0,001` |

**Gesamtbild.** Der Untergrund ist kein Defekt und keine Eigenschaft der Workload,
sondern ein **langsam variierender, gerätweiter Störprozess** mit einer Zeitskala
von einigen hundert Millisekunden — plausibel Betriebssystem-Scheduling und
fremde Systemlast, nicht aus dem Prozess heraus messbar. Er ist nicht eliminierbar.

**Verwertbar ist er trotzdem, und zwar dreifach:**

- Er erklärt vier bisher getrennte Beobachtungen als ein Phänomen: das ungelöste
  H0.1, das zu breite A/A-Bootstrap-Intervall, die nicht funktionierende
  Cutoff-Metrik und die Wertlosigkeit ungepaarter Messung.
- Er belegt, **warum** die gepaarte Messung funktioniert: Beide Arme liegen
  innerhalb derselben Störungsepisode, weshalb sie sich im Quotienten herauskürzt.
  Das war bisher eine empirische Beobachtung, jetzt ist es ein Mechanismus.
- Er liefert eine neue harte Messregel: **Vergleichsarme müssen innerhalb von rund
  `340 ms` gemessen werden.** Größerer Abstand, und sie sehen unterschiedliche
  Störungen; die Paarung verliert dann ihren Vorteil.

Damit ist der größte offene Punkt des Projekts nicht gelöst, aber verstanden — und
die bestehende Methodik ist nachträglich als die richtige Antwort darauf belegt.

### H2 erreicht — ein lokales Modell schlägt Ausführungspläne vor (21.08.2026)

Nutzerauftrag: H2 heute schaffen. Erreicht. Keine neue Installation, kein neuer
Download — das bereits geladene `gemma-3-4b-it-4bit` genügt. Provenienz unverändert.

**Sicherheitsentscheidung, vorab getroffen und begründet: das Modell schlägt
Parameter vor, niemals Code.** Modellgenerierten Code auf der GPU auszuführen ist
ein eigenes Problem mit Sandbox-Anforderungen und wäre ohne ausdrückliche Freigabe
nicht vertretbar. Parametrische Vorschläge zeigen dasselbe Prinzip — ein Modell
liest Messdaten und Gerätefakten und schlägt Ausführungspläne vor — ohne dieses
Risiko. Jeder Vorschlag wird als einfache Ganzzahl geparst und verworfen, wenn er
außerhalb `2..16` liegt, bereits gemessen wurde, kein `int` ist oder die Antwort
nicht als JSON-Array vorliegt.

**Neu: `tools/model_loop.py`, in der CLI als `model-loop`.** Das Modell erhält die
bisher gemessenen Verhältnisse samt Urteil je Kandidat sowie die gemessenen
Gerätefakten (`340 ms` Störungs-Zeitskala, Cooldown-Aufschlag, gepaart gegen
ungepaart) und schlägt drei ungetestete Werte vor. Über mehrere Runden sieht es die
Ergebnisse seiner eigenen Vorschläge und reagiert darauf. Der Messharness stammt
unverändert aus `optimization_loop`; H2 verwendet dieselbe Schwelle, was ein Test
absichert.

**Lauf über drei Runden.**

| Runde | Modellantwort | gemessen |
| ---: | --- | --- |
| 1 | `[3, 10, 16]` | `N=3` `0,907` verworfen · `N=10` `0,864` besteht · `N=16` `0,885` besteht |
| 2 | `[5, 12, 13]` | `N=5` `0,892` · `N=12` `0,862` · `N=13` `0,866`, alle bestehen |
| 3 | `[7, 14, 15]` | `N=7` `0,892` · `N=14` `0,852` · `N=15` `0,872`, alle bestehen |

Insgesamt vorgeschlagen: `[3, 5, 7, 10, 12, 13, 14, 15, 16]` — neun verschiedene
Werte, alle gültig, alle ungetestet, kein einziger unbrauchbarer Vorschlag.
Bestätigung `N=13`: Replikate `0,8847 / 0,8847 / 0,8617`, `95%-KI
[0,8552, 0,8957]`, Verdikt `optimization_confirmed`, Effekt `−11,53 %`.
Wall `21,5 s`, GPU-Arbeit `12,85 s` gegen Budget `120 s`.

**Der Harness blieb streng.** `N=3` wurde verworfen, weil seine
Konfidenzobergrenze `0,954` die Schwelle von `0,95` verfehlte. Das Modell schlägt
vor; es entscheidet nicht.

**Unterschied zum handgebauten Loop.** Der durchsucht Nachbarn des jeweiligen
Siegers und bleibt dadurch lokal. Das Modell verteilte seine Vorschläge über den
gesamten Bereich und deckte in drei Runden neun Werte ab, wo der handgebaute Loop
typischerweise sieben erreicht. Beide finden denselben Effektbereich von rund
`−11 %` bis `−17 %`, was zum bereits dokumentierten breiten Plateau passt.

**Verifikation.** `tests/test_model_loop.py` mit `21` Tests: Parsing (Fences,
bereits gemessene Werte, Bereichsgrenzen, Floats, Booleans als
`int`-Unterklasse, numerische Strings, verschachtelte Strukturen) und die
Vertrauensgrenze (Prosa, `[$(rm -rf /)]`, Backticks, SQL-Fragmente, leere und
abgeschnittene Antworten führen sämtlich zu null ausgeführten Kandidaten). Ein
Test stellt sicher, dass H2 keine mildere Schwelle verwendet als die übrigen
Werkzeuge. Gesamtsuite `358` Tests und `2.337` Subtests grün in `31,9 s`.

Ein Testfehler war lehrreich: Für `[[3], {}, 5]` erwartete ich `[5]`, der Parser
liefert `[]`. Der Extraktor stoppt an der ersten `]`, wodurch ungültiges JSON
entsteht und alles verworfen wird. Das Verhalten ist richtiger als meine Erwartung
— Verschachtelung bedeutet eine defekte Antwort, und Bruchstücke aus defekten
Antworten zu retten ist genau der Weg, auf dem Parser missbraucht werden. Der Test
hält das jetzt so fest.

### H2 vollständig — das Modell schreibt den Ausführungsplan selbst (21.08.2026)

Nutzerfreigabe für die Ausführung modellgenerierten Codes und für ein erhöhtes
GPU-Budget erteilt. Kein zweites Gerät verfügbar, Cross-Device bleibt offen.
Provenienz unverändert (H0 `101cdadf…`, H0.1 `f66e4b5a…`, Environment `74ca2dac…`).

**Zwei unabhängige Schutzschichten, weil jede allein ein Single Point of Failure
wäre.**

`tools/plan_sandbox.py` prüft statisch über eine AST-Allowlist: genau eine
Funktion `plan(mx, a, operands)`, nur erlaubte Knotentypen, nur bekannte Namen,
Attributzugriff ausschließlich als `mx.<Operation>` aus einer Zwölferliste,
keine Importe, keine Dunder, keine String-Literale, keine Lambdas, Größen- und
Komplexitätsgrenze. Erst was das passiert, läuft überhaupt — in einem frischen
Subprozess mit Wall-Timeout, CPU-Zeit-Grenze, bereinigter Umgebung und
MLX-Speicherlimit. Danach folgt das Correctness-Gate: ein Ergebnis je Operand,
jedes bytegleich zur Referenz.

Die Werkzeuge sind `tools/plan_sandbox.py` (Validierung und Isolation) und
`tools/codegen_loop.py` (Schleife), in der CLI als `codegen`.

`tests/test_plan_sandbox.py` greift die Validierung adversarial an: Importe in
drei Varianten, `__class__`/`__reduce__`/`__globals__`, `eval`/`exec`/`compile`/
`open`/`__import__`/`getattr`, String-Literale, Lambdas, falsche Signaturen,
Dekoratoren, überlange und überkomplexe Quellen. `29` Tests, alle abgewehrt.

**Ein Detail, das beim Testen auffiel.** Mein erster Integritätstest schlug bei
`eval` an. Das ist aber `mx.eval`, MLX' Auswertung des lazy Graphen, und hat mit
Pythons `eval` nichts zu tun; letzteres ist ein nackter Name-Aufruf und wird über
`ALLOWED_BUILTINS` blockiert. Der Test hält diese Unterscheidung jetzt explizit
fest, statt sie über einen Substring-Vergleich zu verwischen.

**Erster Lauf: das Modell schrieb viermal die Baseline ab.** Alle vier Pläne waren
zeichengleich mit der im Prompt gezeigten Baseline; die Verhältnisse lagen
erwartungsgemäß bei `1,0149 / 0,9867 / 0,9909 / 1,0252`. Das war unfreiwillig ein
A/A-Test und belegt nebenbei, dass der Harness sauber misst. Zwei Korrekturen:
eine Erkennung für zeichengleiche Baseline-Kopien, die dem Modell als Ablehnung
zurückgemeldet wird, und ein geschärfter Hinweis darauf, dass `mx.eval` eine
**Liste** entgegennimmt und die Baseline gerade deshalb langsam ist, weil sie
achtmal synchronisiert.

**Zweiter Lauf: mein Validator blockierte die richtige Lösung.** Das Modell schrieb

```python
out.append(x)
...
mx.eval(out)
```

also exakt die gesuchte Optimierung — und der Validator lehnte `out.append(x)` als
unerlaubten Attributzugriff ab. Das war mein Fehler, nicht der des Modells: Die
Regel „Attributzugriff nur auf `mx`" verbot genau den Ausdruck, den die Suche
finden sollte. Korrigiert mit einer eng gefassten Erweiterung — nur die
Akkumulatornamen `out`, `result`, `chunk`, nur die Methoden `append` und
`extend`. Fünf neue Tests sichern die Grenze: andere Listenmethoden
(`pop`/`clear`/`sort`/`count`/`__init__`), dieselbe Methode auf `a`, `operands`
oder `mx`, und Dunder auf einer erlaubten Liste bleiben sämtlich abgelehnt.

**Dritter Lauf: bestätigt.** Fünf Pläne geschrieben, fünf gemessen, drei über der
Schwelle. Gewählt wurde nach Konfidenzobergrenze, bestätigt über drei unabhängige
Replikate `0,8742 / 0,8970 / 0,8838`, Ergebnis `R = 0,8838`,
`95%-KI [0,8676, 0,8975]`, Effekt **`−11,62 %`**, Verdikt
`optimization_confirmed`. Wall `20,8 s`.

Der vom Modell geschriebene Gewinnerplan:

```python
def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        out.append(x)
    mx.eval(out)
    mx.synchronize()
    return out
```

**Der Harness blieb streng.** `plan_1` (`R = 0,9820`) und `plan_2` (`R = 0,9892`)
wurden verworfen, weil ihre Konfidenzobergrenzen die Schwelle verfehlten. Beide
hatten `mx.synchronize` aus der Schleife gezogen, aber `mx.eval` darin belassen —
ein halber Schritt, der messbar zu wenig bringt.

**Damit ist der Projekttitel eingelöst.** Bis hierher stammte der Suchraum von mir:
`optimization_loop` variiert eine Batchgröße, `model_loop` lässt ein Modell aus
diesem Raum wählen. Jetzt formuliert das Modell den Ausführungsplan selbst, und
alles, was zwischen generiertem String und berichtetem Ergebnis liegt — Allowlist,
Prozessisolation, Correctness, gepaarte Messung, eingefrorene Schwelle,
unabhängige Replikation — hält unverändert.

**Verifikation.** Gesamtsuite `387` Tests und `2.377` Subtests grün in `32,5 s`,
H0.1-Guard `pass`, alle sechs Werkzeuge hinter ihrem `--execute`-Gate, Provenienz
beider Phasen unverändert.

**Grenze.** Der Plan bleibt eine Umsortierung derselben festen Rechnung. Das Modell
schreibt keine Kernel, wählt keine Algorithmen und ändert keine Numerik — die
Allowlist lässt das nicht zu, und das ist beabsichtigt. Alle Zahlen stammen weiter
von einem einzigen M1 Max.

### Roofline-Messung — die Inferenz ist speicherbegrenzt, nicht rechenbegrenzt (21.08.2026)

Nutzerfrage: Ob lokale KI-Modelle „näher an der nativen Sprache der Hardware"
laufen könnten, wie einst Assembly gegenüber Hochsprachen. Statt darüber zu
argumentieren, wurde gemessen, wo die Zeit tatsächlich hingeht.

**Aufbau.** Zwei unabhängige Ablesungen, damit keine allein getragen werden muss:

1. *Auslastung.* Autoregressive Generierung liest je Token den vollständigen
   Gewichtssatz. Gewichtsbytes geteilt durch gemessene Zeit ergibt die effektive
   Bandbreite; `2 × Parameter` je Token gegen dieselbe Zeit ergibt die effektive
   Rechenleistung. Beides als Anteil der Herstellerspitzenwerte, die ausdrücklich
   als nicht selbst gemessen ausgewiesen sind.
2. *Prefill gegen Generierung.* Prefill verarbeitet viele Token gegen einen
   Durchgang durch die Gewichte, Generierung ein Token je Durchgang. Bei
   Speicherbegrenzung muss Prefill je Token drastisch schneller sein — um wie
   viel, ist eine Messung und keine Annahme.

**Ergebnis** (`tools/measure_roofline.py`, `5` Wiederholungen je Modell):

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Gewichte | `0,73 GB` | `2,56 GB` |
| Generierung | `5,75 ms/Token` (`174 tok/s`) | `12,51 ms/Token` (`80 tok/s`) |
| Prefill je Token | `7,3x` schneller | `5,4x` schneller |
| **Bandbreite genutzt** | **`31,9 %`** | **`51,2 %`** |
| **Rechenwerke genutzt** | **`2,4 %`** | **`3,9 %`** |
| Urteil | `memory_bound` | `memory_bound` |

Der Abstand beträgt in beiden Fällen rund **Faktor 13**. Die Rechenwerke sind bei
echter Inferenz praktisch untätig; sie warten auf Daten. Der Prefill-Vergleich
bestätigt das unabhängig über einen völlig anderen Weg.

**Antwort auf die Ausgangsfrage.** Die Intuition, dass zwischen Modell und
Hardware Ineffizienz liegt, ist richtig — die Verortung nicht. Die
Übersetzungskette Python → MLX → Metal Shading Language → GPU-ISA ist vor dem
ersten Aufruf durchlaufen; der Kernel ist kompiliert und wird wiederverwendet.
Zur Laufzeit kostet die Sprachschicht praktisch nichts. „Näher an der
Maschinensprache" würde damit ausgerechnet den Anteil optimieren, der mit
`2,4`–`3,9 %` ohnehin fast leerläuft.

Wirksam sind nur zwei Richtungen, und beide betreffen Datenbewegung: **weniger
Bytes** (Quantisierung — bei den eingesetzten 4-bit-Modellen bereits eingelöst,
von den Modellautoren) und **weniger Durchgänge** (Kernel-Fusion, das
FlashAttention-Prinzip: nicht schneller rechnen, sondern Zwischenergebnisse gar
nicht erst durch den Speicher schicken).

**Obergrenze, die daraus folgt.** Bei `51,2 %` Bandbreitenauslastung liegt für
das 4B-Modell selbst eine perfekte Optimierung, die die Gewichte unverändert
lässt, bei höchstens rund `2x`. Alles darüber verlangt kleinere Gewichte, nicht
besseren Code.

**Nebenbefund.** Das größere Modell nutzt die Hardware **besser** (`51,2 %` gegen
`31,9 %`). Je mehr je Token zu lesen ist, desto geringer wiegt der fixe Overhead
je Durchgang. Kleine Modelle sind also nicht nur absolut schneller, sondern auch
ineffizienter im Verhältnis zu dem, was die Maschine könnte.

**Zwei eigene Fehler beim Bauen.** Der Self-Check scheiterte an einem
Float-Grenzfall: `0.10 * 3.0` ergibt `0.30000000000000004`, weshalb
`0.30 >= 0.30 * ...` falsch war. Die exakte Schwelle wird jetzt bewusst nicht
mehr getestet, sondern nur klar darüber und klar darunter. Und weil die
Registrierung des Werkzeugs im selben Kommando hinter dem `&&` des
fehlgeschlagenen Self-Checks stand, lief sie nicht — `roofline` war eine Weile
unbekannt, was die CLI korrekt mit `unknown tool` und Exit `64` meldete.

**Grenze.** Die Spitzenwerte `400 GB/s` und `21 TFLOPS` sind Herstellerangaben,
nicht selbst gemessen; sie begrenzen die Verhältnisse, sind aber keine eigene
Evidenz. Die Rechenleistungsschätzung verwendet die übliche Näherung
`2 × Parameter` je Vorwärtsdurchgang. Alle Zahlen stammen von einem M1 Max.

### Die Layer — Fusion über ein unverändertes Modell, bestätigt (21.08.2026)

Nutzerziel: eine Schicht zwischen Modell und Hardware, die das Modell unverändert
lässt und es dennoch effizienter macht. Erreicht und gemessen.

**Weg dorthin, in drei Schritten.**

1. *Fusion auf Op-Ketten geprüft.* Ein Transformer-FFN
   (`matmul → Aktivierung → matmul`) zeigt unter `mx.compile` nur `−2,9 %` —
   unter der Schwelle, weil die Matmuls dominieren. Eine reine elementweise Kette
   aus fünf Operationen zeigt dagegen `−36,4 %`, `95%-KI [0,560, 0,722]`. Das ist
   der Roofline-Befund in Aktion: Fusion wirkt dort, wo Speicherbewegung dominiert.
2. *Anteil abgeschätzt.* Die elementweise Kette macht nur `2,7 %` der FFN-Zeit
   aus; `36 %` davon wären rund `1 %` insgesamt. Der Effekt lohnt sich nur, wenn
   **alle** elementweisen Ketten eines echten Modells erfasst werden.
3. *Genau das getan.* `mx.compile` auf den unveränderten Forward-Pass des Modells
   gelegt, statt einzelne Ketten von Hand zu suchen.

**Ergebnis** (`tools/measure_fusion_layer.py`, `3` Replikate à `20` Blöcke,
hierarchisches Bootstrap):

| Modell | Regime | eager | Effekt | 95%-KI | Correctness |
| --- | --- | ---: | ---: | --- | --- |
| 1B | prefill | `18,96 ms` | `−8,9 %` | `[0,9022, 0,9173]` | bytegleich |
| 1B | **single_token** | `6,64 ms` | **`−12,4 %`** | `[0,8488, 0,8918]` | bytegleich |
| 4B | prefill | `82,35 ms` | `−5,0 %` | `[0,9446, 0,9525]` | knapp unter Schwelle |
| 4B | **single_token** | `13,52 ms` | **`−15,0 %`** | `[0,8449, 0,8547]` | bytegleich |

Verdikt `layer_confirmed`, Generierung in `2` von `2` Modellen bestätigt.

**Warum ausgerechnet der Einzeltoken-Fall am stärksten profitiert.** Bei
autoregressiver Generierung fällt je Durchgang wenig Matmul-Arbeit an, während
Normalisierungen, Residual-Additionen, Rotary Embeddings und Aktivierungen
unverändert anfallen. Der elementweise Anteil wiegt damit relativ mehr — und
genau dort greift Fusion. Das ist dieselbe Physik, die die Roofline-Messung
gezeigt hat, nur aus der anderen Richtung betrachtet.

Praktisch ist das der wichtigste Fall: Reale Nutzung verbringt ihre Zeit in der
Generierung, nicht im Prefill.

**Was die Layer ist und was nicht.** Sie ist ein Wrapper um den Forward-Pass:
keine Änderung an Gewichten, Architektur, Quantisierung oder Numerik. Die Logits
sind in allen vier Messungen **bytegleich** (`0,0e+00` Abweichung), geprüft vor
jeder Zeitmessung. Sie ersetzt keinen Compiler und schreibt keine Kernel; sie
nutzt die Fusion, die MLX bereits kann, und belegt, dass sie beim Laden eines
Modells über `mlx-lm` nicht automatisch aktiv ist.

**Einordnung in das Gesamtziel.** Die Roofline-Messung hatte die Obergrenze für
jede Optimierung ohne Gewichtsverkleinerung bei rund `2x` verortet. Von diesem
Spielraum holt die Layer bei der Generierung `12`–`15 %`. Der Rest liegt in der
Bandbreite selbst und ist ohne kleinere Gewichte nicht erreichbar.

**Ein eigener Fehler beim Messen.** Der erste 4B-Lauf brach mit
`RuntimeError: Item size 2 for PEP 3118 buffer format string B` ab: NumPy kennt
`bfloat16` nicht, das dieses Modell verwendet. Die Umwandlung läuft jetzt über
`mx.float32`, bevor sie NumPy erreicht.

**Grenze.** Zwei Modelle, ein Gerät, ein Prompt. Gemessen wird der Forward-Pass,
nicht eine vollständige Generierungsschleife mit KV-Cache-Verwaltung; der dort
erreichbare Anteil kann kleiner sein. Die `4B`-Prefill-Messung verfehlt die
Schwelle knapp (`−5,0 %` bei einer `5 %`-Grenze) und wird korrekt als
unbestätigt geführt statt aufgerundet.

### Korrektur: die Fusions-Layer bringt in echter Nutzung nichts (21.08.2026)

Der vorstehende Eintrag meldete die Layer als bestätigt. **Das war voreilig.** Die
Nachprüfung an der echten Generierungsschleife widerlegt den praktischen Nutzen.

**Was gemessen wurde und was daran falsch war.** Die Werte `−12,4 %` (1B) und
`−15,0 %` (4B) stammen aus einem direkten Aufruf des Forward-Pass **ohne**
KV-Cache. Sie sind korrekt gemessen, gepaart, repliziert und bytegleich — nur
erfasst dieser Aufruf einen Pfad, den echte Nutzung nie betritt.

**Nachweis.** Über eine vollständige Generierung (Prefill plus `16` Token) mit
gezählten Modellaufrufen: `18` Aufrufe **mit** Cache, `0` **ohne**. Die
End-to-End-Messung in Tokens je Sekunde ergibt entsprechend `−0,5 %` (1B) und
`−0,1 %` (4B), also nichts.

**Drei Ursachen, alle bestätigt.**

1. `mlx_lm.generate` ruft ausnahmslos `model(tokens, cache=prompt_cache)`. Auch
   der Prefill übergibt einen Cache.
2. `mx.compile` kann den Cache nicht entgegennehmen:
   `ValueError: Function arguments must be trees of arrays or constants … but
   received type mlx_lm.models.cache.RotatingKVCache`. Der Cache ist mutabler
   Zustand, `mx.compile` verlangt funktionale Reinheit.
3. **`mlx-lm` fusioniert bereits selbst.** `gemma3_text.py` Zeile `125` und
   `activations.py` tragen `@partial(mx.compile, shapeless=True)` — genau an den
   Stellen, an denen Fusion ohne Cache-Konflikt möglich ist. Der freie Spielraum
   war also bereits ausgeschöpft, bevor diese Untersuchung begann.

**Ein eigener Messfehler auf dem Weg dorthin.** Der erste End-to-End-Versuch
setzte `model.__call__ = compiled` auf der **Instanz**. Python löst `obj()` aber
über `type(obj).__call__` auf und ignoriert das Instanzattribut bei Dunder-Methoden;
der Patch war wirkungslos und maß zweimal dasselbe. Das erklärte das verdächtig
glatte `R ≈ 0,995`. Korrigiert über ein Wrapper-Objekt mit echtem `__call__` auf
der Klasse — womit sich dann die eigentliche Ursache zeigte.

**Was bleibt.** Ein sauberes Negativergebnis mit belegter Ursache: Die
naheliegende Layer greift ins Leere, weil die fusionierbaren Teile bereits
fusioniert sind und der Rest am mutablen Cache-Zustand scheitert. Für das
Projektziel ist das eine wertvolle Grenzmarkierung — sie schließt einen ganzen
Lösungsweg mit Begründung aus, statt ihn offenzulassen.

`docs/ERGEBNISSE.md` und `README.md` sind entsprechend korrigiert; der frühere
Kasten „wichtigstes Ergebnis" mit den `12`–`15 %` ist entfernt. Das Werkzeug
`tools/measure_fusion_layer.py` bleibt bestehen, ist aber ausdrücklich als
Messung des cache-freien Forward-Pass gekennzeichnet und **nicht** als
Generierungsgewinn zu lesen.

### Evidenzaudit, Root-Provenienz und persistente H1/H2-Historie (21.08.2026)

**Auftrag und Grenze.** Der Nutzer beauftragte die vier Auditfolgen vollständig:
Root-Git/Provenienz und persistente H1/H2-Evidenz samt kleiner Historien-UI,
Konsistenz von Status/Plan/Vorregistrierung, Pin und vollständiger Test von
`pytest-xdist` sowie ein Evidenzentscheid zu Phase 1B, Cross-Device und breiterer
Suche. Ausdrückliche Zusatzregel: keine Subagenten. Internetrecherche nach
formalen Architekturquellen war erlaubt. Dieser Arbeitsgang führte keinen GPU-,
MLX-Mess-, Modell-, Download- oder Installationslauf aus.

**ProjectAtlas zuerst.** Der fokussierte Atlas-Kontext wurde vor der
Repository-Arbeit abgerufen. Nach den Änderungen lief ein Index-Refresh mit
`projectatlas watch --once .`: `679` Textkandidaten, `660` indexiert, `19`
übersprungen; `522` Symbolkandidaten, `32` neu geparst, `490` unverändert,
`0` Timeouts. Runtime `0.4.5-rc1`, Root-Bindung `verified=true`, keine
Mismatches. Die generierte Codex-MCP-Konfiguration ist byteinhaltlich zur lokalen
Konfiguration äquivalent: gepinnte Runtime, absolute DB-/Configpfade und korrektes
Projekt-CWD. Die Ignore-Policy übernimmt `.gitignore`; dadurch bleiben
`.friday-data/`, `.venv/`, `.projectatlas/` und lokale Modellgewichte außerhalb
des Atlas-Index.

**Orientierungsfehler und Lösung.** Der erste Refresh-Versuch adressierte
irrtümlich `ProjectAtlas/target/release/projectatlas`; dort liegt in diesem
Checkout kein Binary. `command -v projectatlas` identifizierte die gepinnte
Runtime unter `/Users/tobiasburandt/.local/bin/projectatlas`. Der CLI kennt
`atlas_session_brief` nur als MCP-Tool, nicht als `session-brief`-Subcommand;
dieser Fehlaufruf änderte nichts. Auch `search --path-only` existiert nicht;
Ignore-Verifikation erfolgte stattdessen über `ignore list`, `config` und die
indexierte Dateisuche. Diese Fehler waren reine Diagnose-/CLI-Fehler ohne
Repository- oder Evidenzmutation.

**Root-Provenienz hergestellt.** Vor dem Audit war das Forschungsroot nicht als
eigenes Git-Repository revisionsgebunden. Ein Root-Baseline-Commit
`4095d26` wurde angelegt. `ProjectAtlas/` ist darin als gepinntes Gitlink
`1f576921…` registriert und bleibt unverändert; Root-Projekt und Upstream-
Repository sind damit getrennt. `.gitignore` schließt lokale Atlas-Laufzeitdaten,
virtuelle Umgebung, Research-DBs, Modelle und Rohoutput aus. Künftige native
Evidenz kann dadurch erstmals auf einen sauberen Root-Commit zeigen.

**Dateninventar.** Read-only SQLite-Abfragen ergaben:

- H0: `28` Runs; Modi `aa_gpu=9`, `eager_baseline=4`, sechs weitere Modi je
  `3`. Alle `28` haben keine Root-Revision. Von den neun A/A-Runs sind sieben
  `completed`, zwei `invalid`; mehrere A/A-Generationen teilen sich dieselbe
  append-only DB.
- H0.1: `3` `legacy_h0_warmup_observation`, `6` vollständige Paced-Sessions und
  `1` Study `h01_complete_unresolved`.
- Historische H1/H2-Rohblöcke wurden nie persistent gespeichert. Rekonstruierbar
  sind nur zehn Zusammenfassungen aus Journal und Ergebnisdokument.

**Formale Korrektur.** Unmittelbar vor dem ersten historischen A/B-Lauf waren das
formale A/A-Gate, der hierarchische Bootstrap und die MDE nicht geschlossen. Der
registrierte H0-Loader verlangt global exakt sechs kompatible A/A-Prozesse; die
DB enthält neun aus mehreren Generationen und mindestens einen relevanten
`warmup_unstable`-Prozess. Die später verwendete `5-%`-Schwelle kann deshalb
nicht rückwirkend vorregistriert werden. Dispatch-, Loop-, Modell- und Codegen-
Zahlen bleiben technisch plausible, gepaarte und teils replizierte Beobachtungen,
sind aber kein formaler H1/H2-Nachweis. Status, README, Ergebnisse,
Implementierungsplan, H1-Entwurf und Phase-1-Spezifikation tragen diese
Herabstufung jetzt prominent; die historischen Journalabschnitte bleiben
append-only unverändert.

**Neue Research-Evidenzdomäne.** `friday_evidence/` implementiert ein
geschlossenes SQLite-v1-Schema für sieben Werkzeuge: `dispatch`, `cooldown`,
`loop`, `model-loop`, `codegen`, `roofline`, `fusion`. Native Berichte binden
Git-Revision, leeren Diff, Hash aller registrierten Code-/SQL-/Spec-Dateien,
Paketumgebung und nicht-sensitive Hardwareidentität vor und nach dem Lauf.
Persistenz erfolgt vor stdout; ein gestarteter Fehler wird sanitisiert als
`measurement_failed` gespeichert. Native Provenienz muss sauber und in ihren
Code-/Spec-/Environment-/Hardwareprojektionen selbstkonsistent sein.

Legacy-Werte verwenden eine getrennte Klasse `legacy_summary`, behaupten
explizit `formal_claim=false` und `raw_measurements_available=false`, binden aber
den SHA-256 des versionierten Importmanifests. Es werden keine Rohzeiten,
historischen Git-Revisionen oder Beobachtungszeitpunkte erfunden. Quellidentitäten
sind idempotent; dieselbe ID mit anderen Bytes wird verworfen.

**SQLite-Härtung nach Primärquellenprüfung.** SQLite dokumentiert `mode=ro` als
echten read-only URI-Modus, während `query_only` allein eine Verbindung nicht
wirklich read-only macht. Deshalb verwendet die UI beides. Zusätzlich werden
`SQLITE_DBCONFIG_DEFENSIVE=ON`, `TRUSTED_SCHEMA=OFF`, DQS DDL/DML aus und
Extension-Loading aus gesetzt und zurückgelesen; Nichtverfügbarkeit ist
fail-closed. `integrity_check(1)` ersetzt `quick_check`, weil letzterer laut
SQLite UNIQUE- und Indexkonsistenz nicht vollständig prüft. DB und Parent müssen
private, nutzereigene reale Pfade sein; die Datei wird atomar `0600` angelegt.
Primärquellen:
`https://www.sqlite.org/uri.html`,
`https://www.sqlite.org/pragma.html`,
`https://www.sqlite.org/c3ref/c_dbconfig_defensive.html` und
`https://docs.python.org/3/library/sqlite3.html`.

**Gemeinsame Hardwarebudgets.** Alle sieben Werkzeuge verwenden denselben
`BudgetGuard`: höchstens `120 s` kumulierte GPU-Arbeit, `6 s` kontinuierlich,
reale Pflichtpause mindestens `4 s`, höchstens `25 %` Duty-Cycle in jedem
gleitenden `60-s`-Fenster, Wall höchstens `20 min` und mindestens `60 s`
Kandidatencooldown. Schlafaufrufe werden gegen die monotone Uhr geprüft; ein
vorzeitig zurückkehrender Sleeper setzt die Last nicht zurück. Setup,
Correctness, Warmups, Kandidatenblöcke und Bestätigungsreplikate werden
mitgerechnet. Erfolgreiche Berichte enthalten ihre Roharrays sowie die
Budgetzusammenfassung.

**Codegen-Sicherheitskorrektur.** Die offizielle MLX-Dokumentation bezeichnet
`mx.set_memory_limit` als *guideline*: Eine Ausnahme ist erst garantiert, wenn
das Limit überschritten ist und kein RAM einschließlich Swap mehr verfügbar ist.
Die frühere Bezeichnung als hartes `8-GiB`-Speicherlimit war daher falsch und ist
hiermit korrigiert. Quelle:
`https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html`.
MLX Unified Memory trennt GPU- und CPU-RAM nicht in unabhängige Budgets.

Der Worker besitzt weiterhin hartes `30-s`-Wall-Timeout und `25-s`-Kernel-CPU-
Limit. Speicher wird stattdessen primär über eine geschlossene Plansprache
begrenzt: nur `matmul`, `eval`, `synchronize`, kleine Ganzzahlliterale, maximal
ein Iterationslevel, an höchstens `16` Operanden gebundene Iteration und maximal
`32` statisch gewichtete Matmuls. Freie Allokationsprimitive, selbstwachsende
Listen, verschachtelte Schleifen, ungebundene Ranges, freie Subskription,
Annotationen und Keyword-/Default-Signaturen werden abgelehnt. Der Worker
revalidiert den Quelltext und führt ihn mit explizit eingeschränkten Builtins
aus. Ein eigener Test deckte zunächst einen `KeyError` auf, weil die Arity-Prüfung
einen nicht erlaubten `mx`-Namen vor der Allowlist-Prüfung nachschlug; eine
explizite Membership-Prüfung schloss den Fehler. Danach wurden auch die
semantischen Speicherpfade adversarial getestet.

**Historien-UI.** `tools/evidence.py`/`friday evidence` bietet `verify`,
`snapshot`, `detail` und einen Server ausschließlich auf `127.0.0.1:8767`.
Schreibbefehle `init` und `import-legacy` benötigen zusätzlich `--apply`.
Der echte HTTP-Grenztest bestätigte GET/HEAD, Security-Header, 405 für mutierende
Methoden, 400 für unbekannte Parameter sowie bytegleiche DB vor/nach der Abfrage.

**Produktiver Import.** `.friday-data/research.sqlite3` enthält nun exakt zehn
verifizierte Legacy-Zeilen, null native Zeilen und null Zeilen mit Rohmessungen.
Größe `73.728 B`, Mode `0600`, SHA-256
`4489e6114229f386a74f2066833846fa58a211789dc25e7ad8ded20939ecd74a`,
Snapshot-Revision
`715f7e723081c4ec0fecd4caac20824da33fa6a493145175f3b852275cadce90`.
Der zweite Import meldete `inserted=0`, `already_present=10`; der Dateihash blieb
identisch. Auch `verify` und `snapshot` im URI-read-only-Modus änderten kein Byte.

**Dependency und Verifikation.** `pytest-xdist==3.8.0` ist jetzt im Apple-
Requirements-Lock gepinnt; es wurde nicht installiert, weil exakt diese Version
bereits in `.venv` vorhanden war. Vollständige Standard-Suite mit dem in
`pytest.ini` registrierten xdist-Pfad: `425 passed`, `2.443 subtests passed`,
Wall `31,84 s`. Der gezielte Evidenz-/CLI-/Sandbox-Scope und `git diff --check`
waren ebenfalls grün. H0.1-Guard: `57/57` Tests, `2.244/2.244` Subtests, Wall
`19,414 s`, `0` Errors/Failures/Skips, keine NumPy-/MLX-Importe und keine
Socketkonstruktion. `xcodebuild -checkFirstLaunchStatus` endete mit Exit `0`.

Ein erster vollständiger Testaufruf erreichte sichtbar `100 %`, aber der
Orchestrierungswrapper gab die laufende Session-ID nicht aus; das Endergebnis war
dadurch nicht abrufbar. Die Ursache lag im Diagnosewrapper, nicht in Pytest. Der
Lauf wurde einmal kontrolliert wiederholt und diesmal über die explizit ausgegebene
Session-ID abgeholt; nur dieses vollständige Ergebnis wird oben berichtet. Ein
separater `git diff --check` fand davor genau ein Markdown-Zeilenende mit
Trailing-Whitespace; es wurde entfernt und der Check danach grün wiederholt.

**Forschungsentscheid.** Phase 1B/Custom MLX-Metal bleibt **NO-GO**: formaler H1-
Unterbau und harte RAM-Isolation fehlen, und die explorative Roofline motiviert
keinen reinen ISA-Pfad. Cross-Device bleibt **NO-CLAIM**, weil nur ein M1 Max
vorliegt. Ein breiterer Live-Suchraum bleibt **NO-GO**, weil er Multiple-Testing-
und Winner's-Curse-Risiken erhöht. **GO** ist ausschließlich weitere Offline-
Protokoll-/Testarbeit. Der kleinste nächste wissenschaftlich zulässige Schritt ist
eine neue prospektive H1-v2-Spezifikation für weiterhin genau eine Tensoroperation
mit neuer Study-ID; jeder spätere GPU-Lauf benötigt erneut ausdrückliche Freigabe.

#### Nachhärtung und finale Regression desselben Audits

Das anschließende Eigenreview fand vier Verträge, die für eine spätere formale
Nutzung noch zu implizit waren:

1. `native` hätte als „formal H1“ missverstanden werden können. Schema v1
   erzwingt nun in Report und Storage `formal_claim=false`; ein formaler Claim
   benötigt bewusst Schema/Protokoll v2.
2. Der Bool `raw_measurements_available=true` war nicht an einen sichtbaren
   Rohcontainer gebunden. Für jedes der sieben Werkzeuge ist jetzt ein
   nichtleeres Top-Level-Rohfeld registriert und vor Persistenz verpflichtend.
3. Ein idempotenter Replay verglich zunächst nur Report- und Provenienz-Hash.
   Gleiche Quell-ID mit verändertem Status, Raw-Flag oder Beobachtungszeitpunkt
   hätte fälschlich als identisch gegolten. Der Vergleich umfasst jetzt alle
   semantischen Metadaten; ein Regressionstest deckt den Konflikt ab.
4. Die Cooldown-Studie speicherte den geplanten, aber nicht den tatsächlich
   verstrichenen Pausenwert. `verified_pause` misst die monotone Realzeit,
   verwirft frühe Rückkehr und persistiert `observed_pause_seconds`. Die
   Wall-Zeit aller Werkzeuge stammt nun einheitlich aus dem Guard und beginnt vor
   den jeweiligen Laufzeitimports/Setups.

Weitere Storage-Prüfungen binden Provenienz-Schemaversion, verbieten eine
Nullrevision bei nativer Evidenz und vergleichen beim Readback auch
`git_dirty` und den geschlossenen Workload-Key. Die produktiven zehn Legacy-
Zeilen bestehen diese strengere Verifikation unverändert.

**Finaler Volltest nach diesen Änderungen:** `429 passed`, `2.443 subtests
passed in 31,86 s`, Exit `0`. Damit ersetzt dieser Wert den weiter oben
dokumentierten Zwischenstand `425/2.443`; jener Lauf bleibt als zeitlich
korrekter Zwischencheck im append-only Journal stehen. Es gab weiterhin keinen
GPU-, Modell-, Download- oder Installationslauf.

**Letzter Diagnosefehler.** Ein `rg`-Suchmuster wurde in der Shell irrtümlich in
doppelte Anführungszeichen gesetzt; darin enthaltene Markdown-Backticks wurden
von `zsh` als Command-Substitution interpretiert. Die resultierenden unbekannten
Kommandos hatten keine Seiteneffekte und enthielten keine sensitiven Werte. Die
Suche wurde mit einem einfach quotierten Muster wiederholt und fand die alten
„bestätigt“-/Speicherlimit-Formulierungen nur noch in den bewusst unveränderten
historischen Journalabschnitten; die späteren Auditnachträge korrigieren sie.

#### Abschlussaudit nach dem Implementierungscommit

Der Implementierungsstand wurde als `3c48a9b` committed. Der Root-Worktree war
danach sauber, und die Clean-Git-Provenienzprüfung band den vollständigen Commit
`3c48a9b978d759985b012f54ae71252641450bf5` mit `git_dirty=false`. Die produktive
Evidenzdatenbank bestand den URI-read-only-Check weiterhin mit exakt zehn Zeilen;
ihr SHA-256 blieb
`4489e6114229f386a74f2066833846fa58a211789dc25e7ad8ded20939ecd74a`.

Der separate ProjectAtlas-Status zeigte zwei unversionierte Gradle-Cachebäume in
den Groovy-/Kotlin-Fixtures. Ihre Zeitstempel liegen am `2026-08-20 13:01` und
damit vor diesem Audit; sie wurden als vorbestehende Nutzerdaten weder gelöscht
noch ignoriert. Der getrackte ProjectAtlas-Baum und der gepinnte Gitlink blieben
unverändert auf `1f576921f2c824976a591d57be53e871dcd19cd8`.

Zwei reine Abschlussdiagnosen wurden zunächst mit falschen CLI-Namen aufgerufen:
Der kompakte Provenienz-Ausdruck fragte `code_hash` statt `code_sha256` ab, und
der Evidenz-Check verwendete `--db` statt des dokumentierten globalen Arguments
`--database`. Beide Fehler traten erst nach erfolgreicher Datenerhebung bzw. vor
dem Datenbankzugriff auf und hatten keine Seiteneffekte. Die korrigierten Aufrufe
lieferten anschließend die oben dokumentierten Ergebnisse.

### Neue Rechenfreigabe und prospektiver Explorationslauf (21.08.2026)

Der Nutzer hat CPU-/GPU-Nutzung und Tests mit bereits vorhandenen Gemma-Modellen
ausdrücklich freigegeben. Die Freigabe umfasst keinen neuen Download und keine
Installation; beides blieb aus. Gemäß kleinstem ersten Versuch wurde vor jedem
Modelllauf genau eine registrierte Matmul-Operation geprüft. Hardware-Preflight:
`MacBookPro18,2`, Apple M1 Max, `32 GiB`, Netzbetrieb. Vor dem Lauf war der
Root-Checkout sauber; die Evidenz-DB enthielt zehn Legacy-Zeilen, null native
Zeilen und hatte weiterhin SHA-256 `4489e611…ecd74a`.

**Vorhandene Modelle.** Der projektlokale Cache enthält die ausführungsrelevanten
Dateien beider festgelegten Stufen:

- `mlx-community/gemma-3-1b-it-4bit`, Revision
  `2d44e83dc9e80843d22fb941d3d699a0b1351aa6`, eine MLX-Gewichtsdatei mit
  `732.577.304 B`;
- `mlx-community/gemma-3-4b-it-4bit`, Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, eine MLX-Gewichtsdatei mit
  `3.400.569.562 B`.

`mlx 0.32.0`, `mlx-lm 0.31.3` und `numpy 2.5.2` waren bereits installiert.
Die Offline-Self-Checks für `dispatch`, `model-loop`, `codegen`, `roofline` und
`fusion` bestanden mit `4`, `13`, `14`, `8` beziehungsweise `7` Prüfungen.

**Kleinster GPU-Lauf.** `dispatch`, Shape `2048²`, FP16, acht Matmuls, drei
Replikate mit je 25 gepaarten Blöcken: byte-identische Ergebnisse, aggregiertes
`R=0,780054`, hierarchisches 95%-Intervall `[0,765530; 0,877456]`, Effekt
`−21,995 %`, `2,803 s` GPU-Arbeit und `11,468 s` Wall. Alle Guard-Limits wurden
eingehalten. Native Evidenz-ID:
`b866022ae4775c550dd21451e3aa0a7d435f0a8fce515459846835407fa92eb6`.
Das gesamte Intervall liegt unter der versiegelten praktischen Schwelle `0,95`;
Schema v1 kennzeichnet den Befund dennoch korrekt mit `formal_claim=false`.

**Offline-Modellvertrag.** `snapshot_download(..., local_files_only=True)`
verwarf beide Caches als formal unvollständig, weil bewusst nicht benötigte
Repository-Dateien (`README.md`, `.gitattributes`) fehlen. Es wurde nichts
nachgeladen. Der erste eigene Resolver behandelte danach den vorhandenen
4B-Index fälschlich als ausführungsautoritativ; dieser beschreibt zwei nicht
vorhandene ursprüngliche Shards mit zusammen `8.600.158.944 B`, während der
installierte nichtverteilte MLX-LM-Loader tatsächlich direkt
`model*.safetensors` globbt und die vorhandene monolithische MLX-Datei lädt. Nach
Prüfung des installierten Loadercodes wurde der Vertrag an genau diese reale
Dateiauswahl gebunden. Der Resolver validiert nun lokalen Ref, Snapshot,
Metadaten, Tokenizer und MLX-Gewichte und persistiert ID, Revision und Umfang;
ein Netzwerk-Fallback ist ausgeschlossen.

**Weitere Diagnosekorrekturen.** Ein Pytest-Aufruf über das Konsolenskript ließ
in xdist-Workern den Projektroot aus `sys.path`; derselbe Scope war über
`.venv/bin/python -m pytest` grün. Eine erste Suche im ignorierten `.venv` fand
ohne `--hidden --no-ignore` erwartungsgemäß keine Loaderquelle; mit diesen Flags
wurde die relevante MLX-LM-Stelle gefunden. Ein großer atomarer Patch passte
nicht auf die tatsächliche `codegen_loop.run(rounds)`-Signatur und wurde ohne
Teiländerung abgelehnt; anschließend wurden kleine, exakt gebundene Patches
verwendet. ProjectAtlas meldete nach den Änderungen einen
`dependency_closure_limit`-Freshnessblocker; der exakt empfohlene
`atlas_watch_once`-Refresh schloss ihn erfolgreich.

**Regression und Evidenz-Readback vor Modellstart.** Die vollständige Offline-
Suite bestand final mit `435 passed`, `2.447 subtests passed in 32,07 s`. Darin
sind sechs neue Resolver-/Routingtests für einen dokumentfreien, aber
ausführungsfähigen Snapshot, Pfadtraversal, stalen Upstream-Index, fehlende
MLX-Gewichte, eine ungültige Revision und den lokalen Ladepfad aller vier
Modellwerkzeuge enthalten. Der read-only DB-Check bestätigt elf Zeilen,
davon eine native mit Rohmessungen; Snapshot-Revision
`6d91752999e8c456efc8e793dbc413079d7bc551f90437e4b8da427a641b4eae`,
Datei-SHA-256
`4d352ca890fbe3d232661edd9f5c06b4951f2aedd59b07ac0e64d66e4bd96b02`.

#### Guard-Abbruch des ersten Offline-Modelllaufs und Roofline-Korrektur

Der erste `roofline --execute`-Aufruf auf dem sauberen Commit `29a2b74` lud den
validierten lokalen 1B-Snapshot, wurde dann aber korrekt mit
`BudgetError: continuous GPU work budget exceeded` abgebrochen. Ursache: zwei
Warmups und fünf Messgenerierungen waren jeweils einzeln gebucht, zwischen ihnen
fehlte jedoch die reale Vier-Sekunden-Pflichtpause; ihre GPU-Zeiten addierten sich
deshalb über die harte `6-s`-Kontinuierlichkeitsgrenze. Es entstand keine gültige
Teilmetrik, und Gemma 3 4B wurde nicht gestartet. Das sanitisiert persistierte
Fehlerereignis trägt ID
`ffe98ffa5473dedd495684edf5badbf28c66d01c7b00e788e60f48847c1a0ac4`.
Der read-only Check bestätigte danach zwölf konsistente Zeilen; Datei-SHA-256
`f492b4afc850f860f049d3c455a554a3da6b248006f45844ec58b4169fd7611d`.

`pace_generation` fordert nun vor jeder Warmup-/Messgenerierung nach der ersten
eine durch den gemeinsamen Guard verifizierte Pause. Negative Laufindizes werden
abgelehnt. Drei neue Offline-Regressionstests prüfen ersten Lauf, alle späteren
Läufe und den Fehlerpfad; der Roofline-Self-Check umfasst nun elf Prüfungen.
Vollständige Suite: `438 passed`, `2.447 subtests passed in 32,77 s`. Erst nach
Commit dieses reproduzierbaren Fixes darf der Modelllauf neu beginnen.

#### Erfolgreicher offline erzwungener Gemma-1B/4B-Lauf

Der identische Roofline-Lauf wurde erst auf dem sauberen Fix-Commit
`faa4f882702dcf7113f4f662697b96454925e41c` wiederholt. Zusätzlich waren
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` und das projektlokale `HF_HOME`
gesetzt; beide Loader erhielten ohnehin ausschließlich den vom neuen Resolver
validierten absoluten Snapshotpfad. Es fand kein Netzwerkzugriff, Download oder
keine Installation statt.

Fünf Messwiederholungen je Modell nach zwei Warmups, 369 Prompt-Token:

| native Roofline v1 | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Revision | `2d44e83d…b1351aa6` | `93724907…a65abf2` |
| geladene Gewichte | `732.498.176 B` | `2.560.756.736 B` |
| Snapshot-Gewichtsdatei | `732.577.304 B` | `3.400.569.562 B` |
| Prefill-Median | `0,3271 s` | `0,8735 s` |
| Prefill-Durchsatz | `1.128,0 Token/s` | `422,4 Token/s` |
| Folge-Token-Median | `5,012 ms` | `10,949 ms` |
| Folge-Durchsatz | `199,5 Token/s` | `91,3 Token/s` |
| effektive Bandbreite | `146,1 GB/s` | `233,9 GB/s` |
| Anteil publiziertes Bandbreitenpeak | `36,53 %` | `58,47 %` |
| geschätzte FP16-Leistung | `0,5846 TFLOPS` | `0,9355 TFLOPS` |
| Anteil publiziertes FP16-Peak | `2,78 %` | `4,45 %` |
| Faktor-3-Urteil | `memory_bound` | `memory_bound` |

Der Guard protokollierte `10,359679 s` GPU-Arbeit, maximal `1,128991 s`
kontinuierlich, `52,1227 s` reale Pflichtpausen und `68,111323 s` Wall. Alle
Grenzen wurden eingehalten. Gültige Evidenz-ID:
`31c20b1e756b8f7a3c7118e2fe0f142added2ab359c02bdadda7f0f58a647c36`.
Die Klassifikation verwendet publizierte Peakwerte und nur ein Gerät; sie ist
deshalb eine native Rohbeobachtung mit `formal_claim=false`, kein formaler H2-
oder Cross-Device-Nachweis.

Die produktive DB enthält danach `13` verifizierte Zeilen: zehn Legacy, drei
native, davon zwei mit Rohmessungen und ein sanitisiertes Fehlerereignis.
Snapshot-Revision
`eb23ae5d6d72b32c2c595a04e85ea9cf3a7e1bd5aac19be26359a41ac7546cb0`,
Dateigröße `106.496 B`, Modus `0600`, SHA-256
`f646ac7df8f6034114b808a0b6a5223bab78e977c9f8470f2894b46ce28e656b`.

Ein abschließender Detailaufruf übergab die Record-ID zunächst positional; die
CLI verlangt `detail --id`. Der Aufruf brach vor der Abfrage ab und hatte keine
Seiteneffekte. Mit `--id` wurden danach alle drei nativen Provenienzbindungen
read-only verifiziert: Dispatch `eabdbd3`, Guard-Abbruch `29a2b74`, erfolgreicher
Roofline-Lauf `faa4f88`, jeweils `git_dirty=false`.

**Entscheid:** Die neue Roofline-Rohmessung reproduziert die Richtung der alten
explorativen Zusammenfassung und stärkt das praktische Argument gegen einen
reinen Custom-ISA-/Custom-Metal-Pfad. Sie ändert den formalen Entscheid nicht:
Phase 1B bleibt **NO-GO**; der nächste wissenschaftliche Schritt ist weiterhin
ein versiegelter H1-v2-Vertrag für eine einzelne Tensoroperation.

**Finale Regression nach Aktualisierung aller Einstiegstexte:** `438 passed`,
`2.447 subtests passed in 32,18 s`, Exit `0`; `git diff --check` war davor grün.
Der frühere `32,77-s`-Lauf bleibt als korrekter Zwischenstand im append-only
Journal erhalten.

**Nachgelagerter Frischprozess-Loaderfehler und Schlusskorrektur.** Der gezielte
Einzeldatei-Test `tests/test_friday_cli.py` lief erstmals in Workern, in denen
`_bench` noch nicht in `sys.modules` lag. `friday.py::_shared()` führt das Modul
über `exec_module` aus, ohne es dort vorab einzutragen; die neu eingeführte
`dataclass` wertete wegen `from __future__ import annotations` ihre Stringtypen
gegen diesen fehlenden Moduleintrag aus und scheiterte mit `AttributeError`.
Die vollständige Suite hatte das durch eine günstigere Importreihenfolge
verdeckt. Der reine Datencontainer ist deshalb jetzt eine explizite Slot-Klasse,
die keinerlei globale Modulregistrierung voraussetzt. Ein neuer Regressionstest
entfernt `_bench` bewusst aus `sys.modules` und prüft den gemeinsamen Loader.

Der erste neue Testlauf scheiterte seinerseits vor der Assertion, weil im
Testmodul `import sys` fehlte; der Import wurde ergänzt. Danach waren der
gezielte CLI-/Resolver-Scope, der vollständige H0.1-Guard (`57` Tests,
`2.244` Subtests, keine MLX-/NumPy-Importe oder Sockets) und die Gesamtsuite
grün. **Aktuell maßgeblicher Volltest:** `439 passed`, `2.447 subtests passed in
31,64 s`, Exit `0`. Die vorherigen `438`-Test-Läufe bleiben korrekte
Zwischenstände; der zusätzliche Test schließt den Importreihenfolgepfad.

### 2026-08-21 — H1-v2-Architektur, prospektive Versiegelung und Offline-Gates

**Freigabe und Entscheidung.** Der Nutzer gab die vorgeschlagene formale
H1-v2-Architektur, weitere Tests und kreative Lösungsfindung frei. v1 bleibt
unverändert explorativ. Für die neue Study
`h1v2-dispatch-n8-20260821-01` wurde deshalb eine getrennte append-only
SQLite-v2-Domäne entworfen: genau ein FP16-`2048²`-Dispatch-Kandidat mit acht
Matmuls, sechs A/A-Sessions, deterministische MDE-Ableitung, separates
Bestätigungssiegel und sechs frische A/B-Sessions mit getrennten
Charakterisierungs-/Validierungsgates. Custom Metal, Cross-Device und freie
Modell-/Codesuche bleiben NO-GO.

**Implementierung.** `friday_h1/` enthält geschlossenen kanonischen JSON-Vertrag,
SHA-256-Schedule und hierarchischen Bootstrap, vollständigen History-Replay,
saubere Git-/Code-/Spec-/Environment-/Hardware-Provenienz, sicheren Store,
kontrollierten MLX-Runner, CLI und read-only Loopback-Dashboard. Das Dashboard
zeigt neben Status und Historie auch Ratio, Intervalle, Effekt und MDE. Die
Vorregistrierung steht in `docs/H1_VORREGISTRIERUNG_V2.md`; der historische
Entwurf bleibt als nicht freigegebene Auditspur erhalten. Ohne `--execute` endet
der Sessionpfad mit Exit `78` vor einem MLX-Import.

**Baseline und Regression.** Der erste Baseline-Aufruf verwendete versehentlich
das Homebrew-System-Python `3.14` ohne Pytest und endete nach `0,03 s` mit
`No module named pytest`; es wurde nichts installiert. Der korrigierte Aufruf
über `.venv/bin/python` bestand in `32,01 s`. Nach H1-v2 bestanden alle 16 neuen
Tests sowie die vollständige Suite mit `455` Tests und `2.447` Subtests in
`33,04 s` (User `131,47 s`, System `2,30 s`, Exit `0`). `git diff --check` und
`compileall` waren grün. Ruff, Pyflakes und Black sind in der vorhandenen
Umgebung nicht installiert; entsprechend der Download-/Installationsregel wurden
sie nicht nachinstalliert.

**Gefundene und gelöste Fehler.** Die ersten neuen Tests fanden drei reine
Fixture-/Grenzprobleme: synthetische Sessions waren an einen festen statt den
jeweiligen Provenienz-Hash gebunden, ein UI-Test war von der Wortstellung
abhängig, und ein doppelter Session-Append wurde vor SQLite als allgemeiner
Protokollfehler statt als `StorageConflict` klassifiziert. Alle drei Ursachen
wurden korrigiert. Der vollständige Replay führte denselben 10.000er Bootstrap
bei einem Append zunächst dreimal aus; die Persistenz validiert nun jede alte
Zeile und replayt die kombinierte alte-plus-neue Historie genau einmal vor dem
atomaren Insert. Öffentliche Lesezugriffe replayen weiterhin immer vollständig.
Der entsprechende Vollhistorientest sank reproduzierbar von `49,60 s` auf
`22,58 s`, ohne den Integritätsvertrag zu lockern.

**Messstatus.** Es gab in diesem Schritt noch keinen H1-v2-GPU-Lauf, keinen
Download und keine Installation. Nächster Zustand ist ein lokaler sauberer
Implementierungscommit; erst dieser darf die Präregistrierung und anschließend
A/A versiegeln.

### 2026-08-21 — Formales H1-v2-Ergebnis und begrenzter Runtime-Prototyp

**Versiegelung und A/A.** Die Präregistrierung wurde auf dem sauberen Commit
`1fbe73c69cedeb69284a264c5e3f45e3e393b822` vor jeder neuen Messung in
`.friday-data/h1-v2.sqlite3` gespeichert. Code-, Spec-, Environment-, Hardware-
und Gesamtprovenienz waren
`4377814e4eafe6171535a5723b90a79f517e5bd64aa5a00ba4f4f52bad4f296c`,
`045b8b38c0f02337acba3a4d05de60b648093733f25f8dc2c9c2c7d7e2ad196f`,
`6ef07ef1a2976e4dfc5a0fb7a65b1535a28372c145f8d488c7cd5d0a33ff6624`,
`ee157aaa01de24f2fcb3057bf6cacbfbc361257d2a192eadc3fd75f33f3133b3`
und `e08732640516712818fd1872411acdcbfdf7fb91849a588ee1101a8007e7d7e3`.
Sechs getrennte A/A-Prozesse (`C0,V0,C1,V1,C2,V2`) liefen am Netzteil mit
mindestens 20 Sekunden Inter-Session-Cooldown. Das aggregierte Verhältnis war
`1,0001085273`, 95%-Intervall `[0,9991934290; 1,0005400282]`; Session-SD
`0,0004606442`. Die rohe MDE war `0,0007522288`, deshalb blieb der
vorregistrierte konservative Floor `0,05` maßgeblich. Alle Kalibrierungsgates
bestanden. Summary-Record:
`e7968f129774e514d74a92865e02c0cde516f600557c4e0a0b79012b270bc396`;
Bestätigungssiegel:
`5b44b847f95b162ada893ba19201b6ed170cc7f503885ab54d6df950b18c8c07`.

**Frische A/B-Bestätigung.** Sechs weitere getrennte Prozesse verwendeten den
einzigen versiegelten Kandidaten. Alle Outputs waren byte-identisch. Gesamt:
`R=0,8797176292`, 95%-Intervall `[0,8770453580; 0,8804029998]`, Effekt
`−12,028237 %`. Charakterisierung: `R=0,8794152406`, Intervall
`[0,8780015537; 0,8805123814]`; Validierung: `R=0,8800439669`, Intervall
`[0,8750561199; 0,8809031162]`. Alle drei Gain-Gates lagen vollständig unter
`0,95`; Regression und Äquivalenz wurden verworfen. Der terminale Record
`f508fc9e2b1f44a1b60084bdbeca581024f1f3599535b3dd662a9305c99a9357`
trägt `h1_gain_confirmed`, `formal_claim=true` und ausschließlich die Aktion
`permit_bounded_runtime_prototype`. Scope bleibt ein Gerät, ein Workload, ein
Ausführungsplan; kein Modell- oder Cross-Device-Claim.

**Ressourcen und unveränderte Evidenz.** A/A verbrauchte kumuliert
`6,105916 s` gebuchte GPU-Arbeit und `8,650735 s` Wall; A/B `5,680047 s` GPU
und `8,168236 s` Wall. Maximale kontinuierliche Last blieb bei `1,041439 s`
bzw. `0,954798 s`, MLX-Peak bei rund `411 MB`, RSS bei rund `506 MB`. Die
kleinsten beobachteten Record-Abstände waren `21,719319 s` und `25,070576 s`.
Die DB enthält `16` replaybare Records, eine formale Aussage, Modus `0600`,
`163.840 B`, SHA-256
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`.
Ein späterer read-only Policy-Preflight änderte diesen Hash nicht.

**Bekannte Restgrenze des formalen Replays.** Der Session-Runner erzwang den
20-Sekunden-Cooldown und die tatsächlichen Record-Abstände belegen ihn. Das
H1-v2-Protokoll replayt den Zeitabstand jedoch nicht als eigene Zeileninvariante.
Diese Lücke wird nicht rückwirkend im versiegelten Studiencode verändert; eine
künftige Study-Version soll den Mindestabstand zusätzlich im Protokoll prüfen.
Sie entwertet die tatsächlich eingehaltene Prozedur nicht, begrenzt aber, was
der DB-Replay allein beweist.

**Runtime-Architektur.** Die neue prospektive Spezifikation
`docs/RUNTIME_PROTOTYPE_SPEC.md` friert vor Live-Messungen eine exakte
H1-/Workload-/Hardwarebindung ein. `friday_runtime/` lädt die 16-Record-Historie
einmal read-only, vergleicht H1-Code, H1-Spec, Environment und Hardware und
autorisiert Batching nur für aus den tatsächlichen Tensoren abgeleitetes
FP16-`2048²`-Matmul mit acht RHS. Jede Unsicherheit wählt seriell. Ein
Batch-Fehler wird im laufenden Aufruf nicht wiederholt und verriegelt einen
prozessweiten Circuit Breaker für alle Folgeaufrufe. Relevante Messungen landen
in einer getrennten privaten, append-only SQLite-Datei mit Hash-Kette; die
read-only UI ist für `127.0.0.1:8769` vorgesehen. Die formale
H1-Architekturdatei wurde nicht nachträglich umgeschrieben, weil ihr exakter Hash
Teil der versiegelten H1-Provenienz ist.

**Offline-Verifikation vor Runtime-Live-Lauf.** 13 neue Tests decken exakte und
fremde Evidenz, Dirty-/Code-/Spec-/Environment-/Hardware-Fallback, unbekannte
Workloads, einzelnen Batch-Eval, seriellen Fallback, keinen impliziten Retry,
Circuit Breaker, private Dateimodi, Symlink-Ablehnung, SQLite-Defensive-Controls,
Update/Delete-Sperren, Hash-Kette, read-only UI, CPU-Messdesign und vorbereitete
Korrektheit ab. Zielscope: `13/13`. Vollsuite: `468` Tests, `2.463` Subtests,
`34,58 s` Pytest-Wall (`34,87 s` außen), User `135,36 s`, System `3,38 s`,
Exit `0`; Peak-RSS außen `78.413.824 B`. `compileall` und `git diff --check`
bestanden. Ruff, Pyflakes und Black bleiben lokal nicht vorhanden und wurden
nicht installiert.

**Diagnosefehler und Lösungen.** Eine erste read-only H1-Projektion nahm
irrtümlich ein Feld `sequence` im von `verified_records()` gelieferten Objekt an
und endete mit `KeyError`; die DB war bereits vollständig verifiziert und
unverändert. Die korrigierte Abfrage verwendete `rowid`/lokale Enumeration und
bestätigte alle 16 Records. Ein großer Dokumentationspatch traf wegen eines
abweichenden bestehenden Zeilenumbruchs seinen Kontext nicht und wurde atomar
ohne Teiländerung verworfen; kleine, exakt verankerte Patches ersetzten ihn. Im
absichtlich schmutzigen Entwicklungs-Worktree verifizierte der reale
Runtime-Preflight die Historie, sperrte Batching aber korrekt mit
`worktree_dirty` (Exit `2`). Erst nach einem neuen sauberen Commit dürfen
CPU-Policy- und GPU-Engineering-Validierung laufen.

Ein minimaler, nicht als Performance-Messung verwendeter MLX-API-Smoke mit einem
`1×1`-FP16-Tensor bestätigte die reale Metadatenform: `shape` ist ein Tupel aus
Python-`int`, `dtype` wird als `mlx.core.float16` dargestellt und vom
Scope-Normalisierer zu `float16` reduziert. Damit ist die Offline-Fixture-Annahme
vor dem großen Tensorlauf gegen die installierte MLX-API geprüft.

Es gab in diesem Implementierungsschritt keinen neuen GPU-Messlauf, keinen
Modelllauf, keinen Download und keine Installation. ProjectAtlas wurde vor der
Arbeit benutzt und nach den neuen Dateien mit `atlas_watch_once` erfolgreich auf
Generation 191 aktualisiert.

### 2026-08-21 — Runtime-Gates bestanden und H2-Minimalschritt freigegeben

**Sauberer Implementierungsstand.** Der evidenzgebundene Runtime-Prototyp wurde
lokal auf `main` als Commit
`0b0a893f58e9c757a0aa7b49565a8b1c1eb2a561` gespeichert (`feat: add
evidence-bound runtime prototype`), ohne Push. Der Root-Worktree war danach
sauber. `xcodebuild -checkFirstLaunchStatus` endete mit Exit `0`; ProjectAtlas
meldete Runtime `0.4.5-rc1`, Major `3`, MCP/SQLite/TOON und die korrekte absolute
projektlokale MCP-Konfiguration. Die vorbestehenden ungetrackten Gradle-Caches im
verschachtelten ProjectAtlas-Repository blieben unberührt.

**Realer Policy-Preflight.** Auf dem sauberen Commit replayte die Runtime alle
16 H1-Records, prüfte terminale Record-/Decision-/Preregistration-Identitäten
sowie identische H1-Code-, Spec-, Environment- und Hardware-Fingerprints und
wechselte von der zuvor beobachteten Dirty-Fallback-Entscheidung auf
`formal_h1_gain_exact_scope`, Strategie `batched`. Die formale H1-Datei blieb
SHA-256
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`.

**CPU-Policy-Gate.** Fünf Warmup-Blöcke und 21 Messblöcke mit alternierendem
A/B-/B/A-Order verglichen je 20.000 direkte Planlesevorgänge mit vollständiger,
gecachter Tensorbeobachtung plus Policy-Auswahl. Baseline-Median `28 ns`,
Policy-Median `11.045 ns`, Policy-MAD `14 ns`, p95 `11.078 ns`, gepaarter
zusätzlicher Median `11.017 ns`. Damit bestanden die vorregistrierten Grenzen
`25.000/50.000/20.000 ns`; Circuit Breaker blieb offen. Record:
`a9c08e2b4d79590e1cfa1d5270c53a80a69b1ff1f39507f003fcd6d8d2be1815`.
Außenmessung: `9,72 s` Wall, `9,41/0,22 s` User/System, Peak-RSS
`33.751.040 B`. Der Aufwand wird nicht weiter auf Kosten der Metadatenprüfung
optimiert: Gegen die später gemessene Kandidatenlaufzeit entspricht er nur rund
`0,063 %`; die Scope-Prüfung ist die wichtigere Eigenschaft.

**MLX/GPU-Engineering-Gate.** Am Netzteil liefen eine ungemessene
Korrektheitsprüfung, zwei Warmup-Paare und zwölf alternierende Messblöcke auf dem
exakten H1-Fixture. Serieller Median `20.359.666,5 ns`, MAD `614.208,5 ns`;
Runtime-Batch-Median `17.643.354 ns`, MAD `372.812 ns`; gepaarter Median
`R=0,8792085596`, Effekt `−12,079144 %`. Referenz- und Kandidatendigest waren
identisch
`3efa90dae9c0025c31365b90a01a518e7df540a23f836446f78aef1b973faf54`,
maximaler absoluter Fehler `0,0`. Das `R≤0,95`-Gate bestand, die Policy blieb
`batched`, Circuit Breaker offen. Record:
`643af8606c83cbcd0a591ba63bebb8745ddf5d4a346971c1d733c8d2b566c2dc`.

Der Guard verbuchte `0,667252 s` GPU-Arbeit, maximal `0,667252 s`
kontinuierlich, `1,059850 s` Wall und keine notwendige Pause; alle Grenzen
blieben eingehalten. MLX aktiv/cache/peak: `142.606.336/67.108.864/209.715.200 B`,
RSS-Peak im Bericht `440.401.920 B`. Außen: `5,09 s` Wall,
`4,16/0,28 s` User/System, Peak-RSS `440.909.824 B`. Das Verhältnis reproduziert
den formalen H1-Effekt praktisch (`0,879718`) und ist dennoch ausdrücklich nur
Engineering-Validierung mit `formal_claim=false`.

**Runtime-Historie und UI.** `.friday-data/runtime.sqlite3` enthält zwei
vollständig replaybare Records; Record 2 bindet Record 1 als Vorgänger. Datei:
Modus `0600`, `45.056 B`, SHA-256
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`;
UI-Snapshot-Revision
`a53e6b31c8266b1881ebebfc4dca8c28e9a4177d7648496863fc2b6d4cd6eb3f`.
Read-only Snapshot und vollständiger History-Replay ließen den Dateihash
unverändert; beide Provenienzen tragen `git_dirty=false` und Commit `0b0a893`.

**Nächster Entscheid.** Beide Runtime-Gates sind bestanden. Freigegeben ist nun
nur der kleinste bereits implementierte H2-Pfad: eine explorative Runde mit dem
lokal vorhandenen Gemma 3 4B, höchstens drei Integer-Vorschläge aus `2..16`,
keine Codegenerierung, Harness/Allowlist als alleinige Ausführungsautorität,
BudgetGuard und Schema-v1-Persistenz mit `formal_claim=false`. Offline-Variablen
und der projektlokale Snapshotresolver schließen Netzwerkfallback aus. Weitere
Runden, freier Suchraum, Custom Metal, Download oder Installation bleiben nicht
freigegeben. Vor diesem Modelllauf werden die Runtime-Ergebnisse auf einem
separaten Dokumentationscommit festgehalten.

Nach dem Runtime-Commit meldete ProjectAtlas einmal
`dependency_closure_limit`; der exakt empfohlene `atlas_watch_once`-Pass stellte
Generation 193 erfolgreich her. Die erste Query priorisierte danach wegen des
mehrdeutigen Wortes „runtime“ eine ProjectAtlas-interne Datei; die anschließend
auf Dokumentation und den exakten Forschungsentscheid begrenzte Suche führte zum
richtigen Artefakt. Es erfolgten weiterhin weder Download noch Installation.

### 2026-08-21 — Eine geschlossene H2-Gemma-Runde

**Freigabegrenze und Preflight.** Nach bestandenem Runtime-Gate wurde genau die
im Forschungsentscheid erlaubte einzelne H2-Runde gestartet. Der Root-Worktree
war auf dem Dokumentationscommit
`99267d3422f5a8573cad0f53e7009a4cf8f52198` sauber. Der Offline-Self-Check des
Modellloops bestand `13/13`: Allowlist `2..16`, höchstens drei Integer,
Duplikat-/Already-tried-Filter sowie Verwerfung von Prosa, Shelltext, Floats,
Boolwerten und Strings. Die Research-DB hatte vorher 13 verifizierte Zeilen,
davon drei native, Modus `0600`, SHA-256
`f646ac7df8f6034114b808a0b6a5223bab78e977c9f8470f2894b46ce28e656b`.

**Lokaler Modellvertrag.** Der Prozess setzte `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` und
`TOKENIZERS_PARALLELISM=false`. Der projektlokale Resolver übergab einen
absoluten validierten Snapshotpfad an MLX-LM; es gab keinen Repository-ID-
Downloader und keinen Netzwerkfallback. Modell:
`mlx-community/gemma-3-4b-it-4bit`, Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, eine vorhandene
`model.safetensors` mit `3.400.569.562 B`. Es wurde nichts heruntergeladen oder
installiert.

**Modelloutput und Exploration.** Die einzige Antwort war der JSON-Inhalt
`[3, 10, 16]`; alle drei Werte lagen in der Allowlist. Der Harness führte nur
die bereits bekannte serielle und gebatchte Matmul-Planfamilie aus, niemals
Modellcode. Je Kandidat wurden 20 gepaarte Blöcke ausgewertet:

| N | Ratio | 95%-Intervall | exploratives 5%-Gate |
| ---: | ---: | ---: | --- |
| 3 | `0,8490185242` | `[0,7975674989; 0,9037886515]` | bestanden |
| 10 | `0,7849208913` | `[0,7416857628; 0,8306763275]` | bestanden |
| 16 | `0,8895659839` | `[0,8814235380; 0,8977836485]` | bestanden |

Der Ranking-Harness wählte `N=10`. Die getrennte Bestätigung bestand aus drei
Replikaten mit Ratios `0,6649`, `0,6716`, `0,7014`; der hierarchische Bootstrap
mit 10.000 Draws ergab `R=0,6715729996`, 95%-Intervall
`[0,6488949358; 0,7311898038]`, explorativer Effekt `−32,84 %`. Verdict:
`optimization_confirmed`. Dieser starke Befund ist dennoch **kein** formaler
H2-/Runtime-Claim: Drei Kandidaten wurden modellgestützt aus Vorwissen gewählt,
das Schema bleibt v1 und der Bericht trägt korrekt `formal_claim=false`.

**Ressourcen.** Der gemeinsame Guard verbuchte `9,908610 s` GPU-Arbeit,
maximal `1,730481 s` kontinuierlich, `180,024674 s` Kandidaten-Cooldown,
`16,022076 s` reale Pflichtpausen und `212,268826 s` Wall. Außen:
`213,11 s` Wall, `5,29/3,13 s` User/System, maximales Resident Set
`3.766.992.896 B`, Peak-Memory-Footprint `4.109.553.360 B`; keine Swaps.
Alle registrierten GPU-, Last-, Duty- und Wall-Grenzen blieben eingehalten.

**Persistenz und Readback.** Native Evidenz-ID:
`5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a`,
Status `optimization_confirmed`, Rohmessungen vorhanden, Provenienz
`git_dirty=false` auf Commit `99267d3`. Die Research-DB enthält danach 14
verifizierte Zeilen, vier native und genau eine native `model-loop`-Zeile;
Modus `0600`, `118.784 B`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision
`c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Der vollständige read-only Replay ließ den Hash unverändert. H1- und Runtime-DB
blieben ebenfalls bytegleich (`141f010b…b101e78a4c4` und
`ad4f0ef7…9676473d82`).

**Entscheid und Stopp.** Es wurde keine zweite Modellrunde gestartet. `N=10`
liegt außerhalb der formal versiegelten N=8-Runtime und fällt dort weiterhin
seriell zurück. Ein produktiver oder formaler N=10-Pfad benötigt eine neue
prospektive Ein-Kandidaten-Studie mit frischen A/A-/A/B-Splits; Gemma darf an
dieser Bestätigung nicht erneut selektieren. Das ist eine neue Architektur- und
Studienentscheidung und wird nicht ohne ausdrückliche Nutzerfreigabe umgesetzt.
Phase 1B/Custom Metal, Cross-Device, weitere Modellrunden und freier Suchraum
bleiben NO-GO/NO-CLAIM.

Ein erster kombinierter Dokumentationspatch adressierte `PROJECT_STATUS.md`
zweimal in getrennten Patchoperationen und wurde deshalb vor jeder Änderung vom
Patchwerkzeug abgelehnt. Die Korrektur verwendete genau eine Operation je Datei;
Messdaten und Evidenz waren davon unberührt.

**HTTP-Abschlussprüfung der Runtime-UI.** Der echte Loopback-Server lief auf
`127.0.0.1:8769`. `GET /api/snapshot?limit=2` antwortete mit `200`,
`application/json`, `1.683 B`, `total=2`, zwei Recent-Records und unveränderter
Revision `a53e6b31c8266b1881ebebfc4dca8c28e9a4177d7648496863fc2b6d4cd6eb3f`.
`HEAD /` antwortete mit `200` und lieferte unter anderem `Cache-Control:
no-store`, restriktive CSP-, Frame-, Origin-, Referrer- und MIME-Sicherheitsheader;
`POST /` wurde mit `405` verworfen. Ein erster unbeschränkter Abruf gab den
Snapshot vollständig an die Agentenausgabe weiter und überschritt dort nur das
Ausgabelimit; die korrigierte Prüfung begrenzte die Ausgabe auf Status und
ausgewählte JSON-Felder. Der Server wurde anschließend per `SIGINT` beendet;
der dabei sichtbare `KeyboardInterrupt` und Exitcode `1` sind die erwartete
Folge dieses manuellen Stopps. Kein Prozess lauschte danach mehr auf Port 8769.
Die Runtime-DB blieb bei Modus `0600`, `45.056 B` und SHA-256
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`.

Bei der anschließenden ProjectAtlas-Befehlsprüfung traf eine zu breite
Textsuche zusätzlich eine sehr große einzeilige Benchmark-JSONL-Datei und
überschritt das Ausgabelimit. Ein erster Pfadversuch auf `.mcp.json` war zudem
falsch; die projektlokale Konfiguration liegt unter
`.projectatlas/projectatlas.mcp.json`. Die Korrektur las nur diese kleine Datei
und führte den dort gebundenen CLI-Fallback `watch --once` aus. Der Refresh
bestand mit 714 Kandidaten, 695 indexierten Textdateien sowie 557
Symbolkandidaten (`1` neu geparst, `556` unverändert); beide Diagnosefehler
änderten weder Quellcode noch Messdaten.

Der Abschlussaudit fand im Einleitungsblock von `PROJECT_STATUS.md` noch die
vor dem H2-Lauf gültigen Research-DB-Zähler, Größe, Hash und Snapshot-Revision.
Der weiter unten bereits korrekte neue Stand und der DB-Readback belegten die
Ursache als übersehene Dokumentationsstelle; der Block wurde auf 14 Zeilen,
davon vier native, und die verifizierten Nachlaufwerte korrigiert.

**Abschlussaudit vor dem Ergebniscommit.** `git diff --check` bestand. Die drei
Evidenzdateien blieben auf Modus `0600`; H1-v2 hatte `163.840 B` und SHA-256
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`,
Runtime `45.056 B` und
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`,
Research `118.784 B` und
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`.
Seit der vollständigen Suite mit 468 Tests und 2.463 Subtests wurde kein
Programmcode verändert; die anschließende echte HTTP-Prüfung deckte den
geänderten Betriebszustand ab. Das verschachtelte `ProjectAtlas/` blieb
unverändert; sichtbar waren ausschließlich die bereits vorhandenen
ungetrackten Gradle-Caches in zwei Sprach-Fixtures. Der abschließende Atlas-
Refresh nach allen Dokumentationskorrekturen bestand erneut.

### 2026-08-22 — Prospektiver N10-Vertrag vor dem ersten Messdatum

**Freigabe und Grenze.** Der Nutzer eröffnete ausdrücklich den vorgeschlagenen
Weg: eine prospektive Ein-Kandidaten-Bestätigung für `N=10`, umfassende lokale
CPU-/GPU-Tests und — nur bei positivem Ergebnis — die anschließende Prüfung
eines begrenzten Runtime-/AVO-lite-Pfads. Vorhandene lokale Modelle dürfen
verwendet werden. Die bindende Installationsgrenze bleibt bestehen: Ohne eine
weitere ausdrückliche Freigabe erfolgen weder Download noch Installation. Freie
Kernel-/Codegenerierung und Custom Metal bleiben außerhalb dieser Studie.

**Atlas-first und Baseline.** Vor Quellarbeit wurde der fokussierte
ProjectAtlas-Session-Brief verwendet. Ausgangspunkt war der saubere Root-Commit
`2862b7f`; das verschachtelte `ProjectAtlas/` blieb unangetastet. Vor allen
N10-Arbeiten existierte `.friday-data/n10-v1.sqlite3` nicht. Die drei
Bestandsdatenbanken waren bytegleich zum dokumentierten Stand:

- H1-v2:
  `141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`;
- Runtime:
  `ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`;
- Research:
  `70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`.

**Implementierung.** Der bewährte formale H1-Unterbau wurde mechanisch in das
neue, getrennte Paket `friday_n10/` übertragen und danach auf eine eigene
Study-ID, Application-ID, SQLite-v1-Datei, Hashdomain, Seeds, UI-Port `8770` und
den exakten Zehn-Matmul-Workload verengt. Dadurch blieb `friday_h1/` vollständig
unverändert. Hinzu kamen `tools/run_n10_v1.py`, vier fokussierte Testmodule und
[`N10_VORREGISTRIERUNG_V1.md`](N10_VORREGISTRIERUNG_V1.md). Der Vertrag friert
den einzigen Kandidaten an Research-Record
`5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a`,
Research-DB-Hash, Snapshot-Revision und lokale Gemma-Revision. Gemma ist aus
Kalibrierung und Bestätigung ausgeschlossen; alte Selektionsdaten gehen in
keine Schätzung ein. Vorgesehen sind sechs getrennte A/A- und sechs getrennte
A/B-Prozesse, je zwei Warmup- und 24 balancierte Messpaare, reale 20-s-Pausen,
hierarchischer Bootstrap mit 10.000 Draws, 5-%-MDE-Floor, Split-Gates,
Byteidentität und terminale Fehler ohne Retry.

**Gefundene und behobene Fehler.** Eine unabhängige Nachrechnung zeigte vor dem
Seal, dass die eingefrorenen Seedwerte nicht zu der zunächst dokumentierten
Formulierung „untere 63 Bit“ passten. Die Konstanten waren korrekt aus den
ersten acht SHA-256-Bytes als Big-Endian-Wert mit gelöschtem höchstwertigem Bit
abgeleitet. Dokument und formale Spezifikation benennen nun exakt Algorithmus,
Domain, Labels und alle Session-Seeds; ein neuer Test rekonstruiert Fixture-,
Operand-, Session- und Bootstrap-Seeds. Es wurden keine Seeds nach Messdaten
geändert — es gab noch keine Messung. Zwei kombinierte Mehrdatei-Patches
scheiterten wegen jeweils eines nicht passenden Dokumentkontexts atomar und
änderten nichts; die Korrekturen wurden in kleine, einzeln passende Patches
aufgeteilt. Beim Differenzreview fiel außerdem eine übernommene UI-Bezeichnung
„SQLite v2“ auf; sie wurde vor dem Testlauf auf den tatsächlichen N10-Schemawert
`v1` korrigiert. Ein erster langer Testaufruf lief zwar erfolgreich weiter,
seine abschließende Ausgabe war nach einem zu knapp gewählten Tool-Yield nicht
mehr gebunden; derselbe fokussierte Lauf wurde deshalb vollständig und mit
erhaltener Session-ID wiederholt. Der erste gestagte `git diff --check` fand
zwei Markdown-Zeilen mit beabsichtigten, aber unerwünschten nachgestellten
Leerzeichen; Blankzeilen ersetzen nun diese Darstellung. Mess- und
Datenbankzustand blieben von allen Korrekturen unberührt.

**Offline-Verifikation.** Der fokussierte Lauf bestand `18/18` Tests in
`40,539 s` (`40,65 s` außen, Peak RSS `41.713.664 B`). Die vollständige Suite
bestand `486/486` Tests in `168,276 s` (`168,50 s` außen, User `157,07 s`,
System `1,39 s`, Peak RSS `76.496.896 B`, keine Swaps). Self-Check,
`compileall` und `git diff --check` bestanden. Der finale Offline-Spec-Hash vor
dem Implementierungscommit lautet
`d0da9729e76a1c3df700b1ef70532a7411c000c40b0bca8ec195b88e6b62ab23`.
`black` ist in der bestehenden virtuellen Umgebung nicht installiert; gemäß
Installationsgrenze wurde es nicht nachinstalliert. Der explizite
Clean-Provenienztest und der Live-Preflight stoppten im schmutzigen Worktree
korrekt mit `ProvenanceError`; die Ausführungssperre ohne `--execute` ist
ebenfalls getestet.

**Pre-Seal-Zustand.** Nach den Tests existiert weiterhin keine
`.friday-data/n10-v1.sqlite3`; H1-, Runtime- und Research-DB behielten die oben
genannten Hashes. Es gab in diesem Schritt keine GPU-, CPU-Performance- oder
Modellmessung und keinen Download. Der ProjectAtlas-Refresh nach den neuen
Dateien bestand mit 731 Kandidaten, 712 indexierten Textdateien, 574
Symbolkandidaten, 20 neu geparsten und 554 unveränderten Dateien ohne Timeout.
Der nächste zulässige Schritt ist ein sauberer lokaler Implementierungscommit,
danach das persistierte Präregistrierungssiegel und erst dann A/A.

### 2026-08-22 — N10-v1 terminal vor Timing; eigenständiger N10-v2-Nachfolger

**Sauberer V1-Seal.** Der vollständig geprüfte N10-v1-Stand wurde lokal auf
`main` als Commit `c3e582c33899049ff96ef6a664fb87eb63c58284`
(`feat: preregister formal N10 study`) gespeichert; es erfolgte kein Push. Der
Root-Worktree war danach sauber, `git_diff_sha256` entsprach dem leeren SHA-256,
MLX war Version `0.32.0`, Gerät `MacBookPro18,2`/`arm64`, Netzbetrieb lag an und
`xcodebuild -checkFirstLaunchStatus` bestand. Die versiegelte Provenienz hatte
`provenance_sha256=04ba25dd70bfbddba2f00d0bee5d371728a1ba96b4611b41fe6f91ac09d03f65`.
Der erste und zunächst einzige V1-Record war die Präregistrierung
`3233c5ee0c1d79facb08b6900abebfd7f5570ca6d852e15674c6a18ead58985b`;
der read-only Snapshot hatte Revision
`6cb4661e1ecbc07c76312e44ab432611b06a2aad384ef705bffe91f46998bbc2`.

**Terminaler C0-Stopp.** Der freigegebene A/A-Stage-Launcher startete genau den
nächsten vorregistrierten Prozess C0. Er endete nach `0,47 s` außen mit
`BenchmarkError`, bevor ein Warmup- oder Timingblock entstehen konnte; der
Stage-Launcher brach sofort ab und startete weder V0 noch einen Retry. Der
append-only Fehlerrecord lautet
`3ce4477adf3ca13d30207f37d98f21e36c316c82e3d102abfacf61c091492e49`,
Status `measurement_failed_terminal`, `formal_claim=false`. Die terminale
N10-v1-DB enthält genau zwei Records, null Timing-Sessions und null formale
Claims; Modus `0600`, `40.960 B`, SHA-256
`e0b5f4af62c128938e1e12e388c16b344a66e18eebf9e0568c7ebe34c5a4f0d5`,
Snapshot-Revision
`bbc75d60b5cfc61a1037c0a104e117a89561ec63e13240ca9b84f1bc98c08976`.
Maximales Prozess-RSS außen war `175.931.392 B`, es gab keine Swaps. MLX wurde
initialisiert, aber der Fehler trat bei der CPU-Fixture-Prüfung vor
`backend.from_host`, GPU-Timing oder GPU-Arbeitsverbuchung auf.

**Ursache.** `friday_n10.runner._load_real_workload` verwendete den bewährten
H0-Generator mit einem neuen, korrekt vorab abgeleiteten Fixture-Seed. Für
Produktionsform `2048²` akzeptiert dieser Generator absichtlich nur Seeds mit
vollständig registrierter A-/B-/Metadaten-/Fixture-Identität. Der V1-Seed
`8754882193294599646` war dort nicht registriert; der Test-Backend-Pfad hatte
mit `2×2` gearbeitet und umging deshalb genau diese Produktionsgrenze. Eine
direkte read-only Vertragsprüfung reproduzierte
`CorrectnessContractError: performance fixture identity is not registered`.
Der Guard verhielt sich korrekt; die Lücke lag im V1-Pre-Seal-Testvertrag.

**Kein Retry, sondern V2.** V1 bleibt unverändert terminal. Der eigenständige
Nachfolger `h2n10-dispatch-confirmation-20260822-02` verwendet Paket
`friday_n10_v2/`, Tool `tools/run_n10_v2.py`, DB
`.friday-data/n10-v2.sqlite3`, Application-ID `N10W` und UI-Port `8771`. Er
übernimmt bewusst die bereits registrierte H0-Produktions-Fixture
`0xF17A2026` samt allen vier festen Digests. Operand-, Session- und
Bootstrap-Seeds sind neu aus der V2-Domain abgeleitet. Spezifikation und
Runner binden V1-Study-ID, Fehlerrecord, DB-Hash, Snapshot-Revision, genau zwei
Records und null Timing-Sessions. Seal, jeder Session-Preflight und jede
abgeleitete Record-Mutation replayen diesen Vorgänger read-only; jede
Abweichung sperrt V2. Die letzte Grenze entstand aus der unabhängigen
Pre-Commit-Prüfung und verhindert insbesondere einen terminalen Claim nach
einer zwischenzeitlichen Vorgängeränderung. Ein stabiler interner
Benchmark-Fehlercode wird künftig begrenzt zusammen mit dem Fehlertyp, aber
ohne Fehlermeldung, persistiert.

**Produktions-Fixture- und Offline-Verifikation.** Der echte CPU-Generator
erzeugte die registrierte `2048²`-Fixture in `0,20 s` außen, Peak RSS
`152.633.344 B`, ohne Swap und reproduzierte exakt:

- A: `33043be0345487a8a41b522df292e5288914b9c6c6c4dc823dbec72b9146bf86`;
- B: `dd40817873b24c2e6117e4e6eeebddccf89775bd4ee4453e7d5456a911670ac2`;
- Metadaten:
  `1e26b28978e01ad0faaf296b48043e63803488cdb59e3aa84e79b9ab48a3bb20`;
- Fixture:
  `4776038d9500bad4374410fe2e4a167a6f834e80f0e4d19336592f4ff455dfa4`.

Nach einer zusätzlichen Härtung akzeptiert der V2-Vorgängerhash nur eine private,
begrenzte reguläre Datei, öffnet sie soweit verfügbar mit `O_NOFOLLOW` und prüft
Dateideskriptor, Inode und Größe vor und nach dem Streaming-Hash. Ein neuer Test
weist Symlinks und zu breite Dateirechte ab. Der nach Dateihash- und
Mutationsgrenzen-Hardening wiederholte fokussierte V2-Lauf bestand `22/22`
Tests und `10/10` Subtests in `40,28 s` (`40,54 s` außen, maximales RSS
`57.950.208 B`, keine Swaps). Die danach wiederholte vollständige Suite bestand
`508/508` Tests und `2.480/2.480` Subtests in `207,82 s` (`208,03 s` außen,
User `196,21 s`, System `1,53 s`, maximales RSS `91.357.184 B`, keine Swaps).
Self-Check, Bytecode-Kompilation und `git diff --check` bestanden; der
V2-Study-Spec-Hash lautet
`66a01028b5c7ba6cd7b05faef1f3100413d793c6b4d7e3982bea671fb9bba6cd`.
Ein absichtlich im schmutzigen Pre-Commit-Stand aufgerufener V2-Seal stoppte
mit `ProvenanceError`, Exit `1`, bevor eine V2-Datei angelegt wurde; der
V1-Hash blieb unverändert.
Der read-only V1-Vorgänger-Replay ließ dessen Dateihash unverändert. H1,
Runtime und Research blieben ebenfalls bytegleich bei `141f010b…e78a4c4`,
`ad4f0ef7…6473d82` und `70cbe45b…a0a6d91`.

**Werkzeugfehler während der Reparatur.** Ein kombinierter Patch erwartete die
SQLite-Application-ID im Migrationsfile dezimal, tatsächlich stand sie dort
hexadezimal; der gesamte Patch scheiterte atomar und wurde anschließend mit dem
exakten Kontext angewendet. Eine zu breite Seed-Suche traf außerdem große
einzeilige JSON-Fixtures und überschritt nur das Ausgabelimit; die Korrektur
beschränkte die Diagnose auf `correctness_contract.py` und kleine Ausschnitte.
Beim Abruf des ersten abschließenden Gesamtlaufs war dessen Prozess-ID nach
einem Kontextwechsel bereits geschlossen; weil damit der Abschlussblock nicht
mehr beweisbar war, wurde derselbe sequenzielle Lauf vollständig wiederholt und
oben protokolliert. Eine anschließende `rg`-Suche verwendete Backticks in einem
doppelt quotierten Shell-Argument, wodurch die Shell zwei harmlose, nicht
vorhandene Kommandos zu starten versuchte; die Suche blieb read-only und wurde
nicht als Verifikation gewertet. Keiner dieser Werkzeugfehler änderte
Datenbank oder Messzustand.
Ein späterer kombinierter Dokumentationspatch suchte einen Journalabschnitt
versehentlich im Forschungsentscheid; `apply_patch` verwarf deshalb den
gesamten Patch atomar. Die Korrektur wurde anschließend mit den exakten
Dateikontexten angewendet, ohne einen Teilstand zu hinterlassen.

**Pre-Seal-Zustand V2.** Der ProjectAtlas-Refresh nach dem neuen Paket bestand
mit 748 Kandidaten, 729 indexierten Textdateien und 591 Symbolkandidaten; 19
Dateien wurden neu geparst, 572 blieben unverändert, kein Timeout. Es existiert
noch keine `.friday-data/n10-v2.sqlite3`, keine V2-GPU-Timingmessung und kein
neuer Modelllauf. Der nächste zulässige Schritt ist der saubere V2-Commit,
danach ein eigener V2-Präregistrierungsrecord und erst dann frische A/A-Daten.
Der abschließende Refresh nach Dokumentation und Dateihash-Hardening blieb bei
748 Kandidaten, 729 indexierten Textdateien und 591 Symbolkandidaten; 7 Dateien
wurden neu geparst, 584 blieben unverändert, kein Timeout.

### 2026-08-22 — N10-v2 formal abgeschlossen; begrenzter Runtime-Pfad geöffnet

**Clean Commit und Seal.** Der vollständig geprüfte V2-Stand wurde lokal auf
`main` als Commit `959df09b9d197edbd0a0984eda25092997b4ab23`
(`feat: preregister corrected N10 v2 study`) gespeichert; es erfolgte kein
Push. Root-Worktree und Root-Diff waren danach leer, Xcode-First-Launch bestand,
das Gerät lief am Netzteil und MLX war `0.32.0` auf
`MacBookPro18,2`/Apple M1 Max/`arm64`. Die saubere Provenienz lautet
`17d0dd505e349a4bbb7ffde3c291a3a44226d0fce79c235ce2ce890289e0c9ef`.
Der V2-Seal schrieb genau den Präregistrierungsrecord
`343bbbd14f3551a8c56a4fb16103db7490402a3c2d1361a636bb9c901f556f94`;
die damalige Snapshot-Revision war
`b852b3b347be3b918defaca5c8ace5cd84175ee4abf90f9ee8a3f201f5eddbad`.

**A/A-Kalibrierung.** C0, V0, C1, V1, C2 und V2 liefen in sechs getrennten
Prozessen mit fünf verifizierten 20-Sekunden-Cooldowns. Der Stage-Lauf benötigte
`114,26 s` außen; alle sechs Ausgaben waren byteidentisch, alle Budgets
bestanden. Die einzelnen B/A-Verhältnisse lagen zwischen `0,998708` und
`1,000060`. Aggregiert ergaben sich `R=0,999586`, 95%-KI
`[0,998764; 1,000443]`, rohe MDE `0,0857 %`; der konservative Floor blieb
`5 %`. Gebuchte GPU-Arbeit: `7,556211 s`, Session-Wall zusammen `10,567263 s`,
maximales Session-RSS `590.168.064 B`, MLX-Peak `511.705.088 B`. Die
Kalibrierungszusammenfassung ist Record `3cd3e93b…e2b28b`. Nach mehr als
38 Sekunden Abstand schrieb der separate Confirmation-Seal Record
`d6402bb9…5404487` mit
`confirmation_seal_sha256=7ad8e46102013f238663486eb14b125c405a64ab46798449b02dac97881a7813`.

**A/B-Bestätigung und Entscheid.** Die sechs Confirmation-Sessions liefen
ebenfalls getrennt und ohne Retry oder Fehlerrecord; Stage-Wall einschließlich
Cooldowns und Bootstrap `139,80 s`. Alle Resultate waren byteidentisch bei
`max_abs_error=0`, alle Budgets bestanden. Die Ratios lagen zwischen
`0,870074` und `0,875557`; gebuchte GPU-Arbeit `6,980186 s`, Session-Wall
zusammen `9,912920 s`, maximales Session-RSS `590.725.120 B`, MLX-Peak
`511.705.088 B`, keine Swaps. Der terminale Entscheid lautet:

- gesamt `R=0,874912`, 95%-KI `[0,871768; 0,875614]`, Effekt `-12,509 %`;
- Charakterisierung `R=0,875216`, 95%-KI `[0,869739; 0,876217]`;
- Validierung `R=0,874608`, 95%-KI `[0,871695; 0,875607]`;
- alle Gain-Splits `true`, alle Ausgaben byteidentisch, Status
  `n10_gain_confirmed`;
- Record `47283e73eb6eefa01dc0f2e1760a2a2d350ca51019b8ddfa3a297d4b695e1249`
  trägt als einziger `formal_claim=true` und erlaubt nur
  `permit_bounded_n10_runtime_prototype`.

**Terminaler Store und UI.** `.friday-data/n10-v2.sqlite3` enthält genau 16
vollständig replaybare Records, Modus `0600`, `180.224 B`, SHA-256
`54e9c57ca6b76fa671b94f748b7ee471575b7dd7445bad00ae3cab38f691fc4f`,
Snapshot-Revision
`9c9a94a8f799f2eb29b9e03c4e1b6e681aa945199753158cf8fc8c317b06090d`.
Ein kompletter read-only Replay ließ den Hash bytegleich. Die echte UI auf
`127.0.0.1:8771` lieferte für Root-GET und Snapshot-GET/HEAD `200` und wies
POST mit `405` ab; danach war der DB-Hash weiterhin unverändert. Weil jeder
Request alle Records und drei 10.000-Draw-Bootstraps validiert, benötigte ein
sequenzieller Snapshot `3,42–3,44 s`. Zwei parallele Requests überschritten
deshalb zunächst das zu knappe 5-Sekunden-Clientlimit. `Ctrl-C` schloss den
Server und Port zuverlässig, erzeugte aber einen sichtbaren
`KeyboardInterrupt`/Exit `1`; beides ist für eine Nachfolge-UI dokumentiert und
wird am versiegelten Studiencode nicht nachträglich geändert.

**Integrität und Fehlerprotokoll.** V1, H1, Runtime und Research blieben
bytegleich bei `e0b5f4af…a3f0d5`, `141f010b…e78a4c4`,
`ad4f0ef7…6473d82` und `70cbe45b…a0a6d91`; der Root-Worktree blieb während
der formalen Ausführung sauber. Zwei erste Curl-Aufrufe erreichten die UI nicht,
weil das ungequotete `?` im zsh-Argument als Glob behandelt wurde; die korrekt
quotierten Wiederholungen lieferten die oben genannten Ergebnisse. Die danach
parallel angesetzten 5-Sekunden-Requests liefen in das Clienttimeout, während
der Server den vollständigen Replay korrekt weiterbearbeitete; die sequenzielle
Wiederholung mit 20-Sekunden-Limit bestand. Es gab keine Installation, keinen
Download, keine Modellaktion, keine Custom-Metal-Ausführung und keine Änderung
der bestehenden N8-Runtime. Der ProjectAtlas-Refresh des Ergebnisstands
bestand mit 748 Kandidaten, 729 indexierten Textdateien und 591
Symbolkandidaten; 5 Dateien wurden neu geparst, 586 blieben unverändert, kein
Timeout.

**Nächster zulässiger Schritt.** Der positive formale Claim öffnet genau einen
getrennten, allowlist-basierten N10-Runtime-/AVO-lite-Prototyp mit seriellem
Fallback, Circuit Breaker, eigener Historie und reproduzierbaren CPU-/GPU-Gates.
Freie Codegenerierung, Custom Metal, weitere Modellrunden, Cross-Device-Claims
und ein breiterer Suchraum bleiben geschlossen.

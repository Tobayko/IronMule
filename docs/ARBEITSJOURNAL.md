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
eines begrenzten Runtime-/Runtime-lite-Pfads. Vorhandene lokale Modelle dürfen
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
getrennten, allowlist-basierten N10-Runtime-/Runtime-lite-Prototyp mit seriellem
Fallback, Circuit Breaker, eigener Historie und reproduzierbaren CPU-/GPU-Gates.
Freie Codegenerierung, Custom Metal, weitere Modellrunden, Cross-Device-Claims
und ein breiterer Suchraum bleiben geschlossen.

### 2026-08-22 — N10-Runtime-/Runtime-lite-Prototyp vor Live-Gates

**Atlas-first und Baseline.** Vor der neuen Phase lieferte der fokussierte
ProjectAtlas-Brief `friday_runtime/executor.py`, `tests/test_runtime_policy.py`
und `tests/test_runtime_history.py` als engsten bestehenden Referenzscope. Die
unveränderte N8-Runtime-Baseline bestand `13` Tests und `9` Subtests in
`0,11 s` (`0,34 s` außen, maximales RSS `54.984.704 B`, keine Swaps).
`.friday-data/runtime.sqlite3` blieb bei
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`,
der formale N10-v2-Store bei `54e9c57c…fc4f`; eine
`.friday-data/runtime-n10.sqlite3` existierte nicht.

**Gebundene Spec wiederhergestellt.** Der Ergebnisnachtrag hatte nach dem
formalen Lauf vorübergehend die an N10-v2-Provenienz gebundene Datei
`docs/N10_VORREGISTRIERUNG_V2.md` erweitert. Die Pre-Runtime-Prüfung erkannte,
dass dies eine korrekte Runtime absichtlich mit `n10_spec_mismatch` sperren
würde. Der Nachtrag wurde aus genau dieser Datei entfernt; die Ergebnisse
bleiben in Projektstatus, Forschungsentscheid und diesem Journal erhalten. Die
Datei ist nun wieder bytegleich zum Seal bei
`5b283ce8629b5b802dabc68dccd1396157d739f2796b76d02ce4fdf97563087b`;
Code-, Spec-, Umgebungs- und Hardwareprojektion stimmen wieder mit der formalen
Provenienz überein. Im noch schmutzigen Arbeitsstand bleibt Live-Ausführung
korrekt gesperrt.

**Architektur und Implementierung.** Der neue Vertrag steht in
`docs/N10_RUNTIME_PROTOTYPE_SPEC.md`. Die bestehende N8-Runtime wurde nicht
geändert. Das getrennte Paket `friday_runtime_n10/` samt Tool
`tools/run_n10_runtime.py` verwendet Runtime-ID
`n10-runtime-dispatch-20260822-01`, Application-ID `FRN1`, DB
`.friday-data/runtime-n10.sqlite3` und UI-Port `8772`. Runtime-lite ist eine
geschlossene Zustandsmaschine: reale Tensoren beobachten, exakten formalen
Store plus V1-Vorgänger und aktuelle versiegelte Identität verifizieren, genau
einen Plan auswählen, CPU/GPU/Korrektheit validieren und bei Unsicherheit
seriell zurückfallen. Es gibt kein Modell, keine freie Aktion und keine
Codegenerierung.

Der Policy-Load bindet den exakten N10-v2-Dateihash, die Snapshot-Revision,
16 Recordtypen, den einzigen formalen Decision-Record, Decision-/Prereg-/
Provenienz-Hashes, Workload, Gates und die vier aktuellen formalen
Fingerprint-Projektionen. Er replayt zusätzlich N10-v1 read-only. Die Projektion
wird einmalig gecacht; jeder Bericht zeigt DB-Hash und Snapshot sichtbar. Der
Hot Path autorisiert nur FP16-`2048²` mit genau zehn RHS-Tensoren. Batchfehler
werden im aktuellen Aufruf nicht wiederholt und verriegeln den Circuit Breaker.
Cold Load ist mit `10 s`, Policy-Median mit `25 µs`, p95 mit `50 µs`,
zusätzlicher Median mit `20 µs` und GPU-B/A mit `0,95` begrenzt. Die eigene
History ist privat, append-only und hashverkettet; die read-only UI behandelt
`Ctrl-C` ohne Traceback.

**Offline-Verifikation.** Die fokussierte Suite bestand nach der finalen
Nachweisprojektion `17/17` Tests und `9/9` Subtests in `3,54 s` (`3,72 s`
außen, maximales RSS `58.376.192 B`, keine Swaps). Sie prüft unter anderem
den echten 16-Record-Store, eine byteverschiedene Ersatzdatei, alle
Identitätsfehler, exakten N=10-Scope, Circuit Breaker ohne impliziten Retry,
Cold-Load-Gate, gepaarte Messungen, private Hashkette, read-only UI und
graceful `KeyboardInterrupt`. Eine Zwischen-Vollsuite bestand `525/525` Tests
und `2.489/2.489` Subtests in `210,99 s` (`211,20 s` außen, User `199,59 s`,
System `1,53 s`, maximales RSS `90.652.672 B`, keine Swaps). Der reale
Policy-Load replayte im absichtlich schmutzigen Pre-Commit-Stand alle 16
Records in `3,57 s` außen und fiel exakt auf `worktree_dirty` zurück. Beide
Live-Kommandos endeten ohne `--execute` mit Exit `78`; es entstand keine neue
Runtime-DB und keine GPU-Arbeit.

Nach der zusätzlichen sichtbaren DB-/Snapshot-Projektion in jedem
Policy-Evidenzobjekt wurde die Vollsuite abschließend wiederholt: `525/525`
Tests und `2.489/2.489` Subtests in `211,66 s` (`211,90 s` außen, User
`200,22 s`, System `1,62 s`, maximales RSS `90.832.896 B`, keine Swaps).

**Gefundene Werkzeugfehler.** Der mechanische Kopieraufruf enthielt zunächst
die nicht vorhandene Quelle `friday_runtime/canonical.py`; `cp` meldete Exit
`1`, kopierte aber alle tatsächlich vorhandenen expliziten Quellen. Die
Dateiliste wurde danach vollständig kontrolliert. Ein mehrdeutiger Patch fügte
`N10_DECISION_RECORD_ID` zunächst zusätzlich hinter `__all__` statt nur in die
Importliste ein; die unmittelbare Quellprüfung fand und entfernte die Zeile vor
dem ersten Compile-/Testlauf. Ein kombinierter Status-/Plan-Patch traf einen
nicht exakt vorhandenen Kontext und wurde atomar verworfen; die Korrektur
erfolgte dateigenau. Keiner dieser Fehler veränderte Messdaten oder bestehende
Runtime-Dateien.
Ein späterer kombinierter Abschlusszahlenpatch verfehlte wegen eines
Zeilenumbruchs den Journal-Kontext und wurde ebenfalls atomar verworfen; die
zwei dateigenauen Korrekturen wurden danach erfolgreich angewendet.

**ProjectAtlas und Pre-Live-Zustand.** Der Refresh nach den neuen Dateien
bestand mit 764 Kandidaten, 745 indexierten Textdateien und 607
Symbolkandidaten; 17 Dateien wurden neu geparst, 590 blieben unverändert, kein
Timeout. Der Abschlussrefresh nach Dokumentation blieb bei denselben
Kandidatenzahlen; 4 Dateien wurden neu geparst, 603 blieben unverändert, kein
Timeout. Live bleibt bis zu einem sauberen lokalen Commit geschlossen. Danach
ist genau ein Policy-/CPU-Lauf zulässig; nur bei bestandenem Gate folgt genau
ein MLX/GPU-Lauf.

### 2026-08-22 — N10-Runtime-/Runtime-lite-Live-Gates abgeschlossen

**Sauberer Commit und Provenienz.** Der vollständig geprüfte Runtime-Stand
wurde lokal auf `main` als Commit
`5eaad38ec0f5da4b01bd9d64237d3736f548ff14`
(`feat: add evidence-bound N10 runtime`) gespeichert; es erfolgte kein Push.
Die bestehende `friday_runtime/` blieb bytegleich. Root-Worktree und Root-Diff
waren vor beiden Live-Läufen leer, Xcode-First-Launch bestand, das Gerät lief
am Netzteil und MLX war `0.32.0` auf `MacBookPro18,2`/Apple M1 Max/`arm64`.
Die saubere Runtime-Provenienz lautet
`02784bd7108767008c9951724421cc3f841390d463a8c6b153b059c5c497e22c`;
Code- und Spec-Fingerprint sind
`c3edf267519af02585ae7634c1ad809a183ac05561bbbec9a3ba5c1387c2dbcc`
und `d4e855fc1f3d878fb2a9cf0209b4da69dabeb8a9b4e12594138a59637d933502`.
Der erste saubere read-only Policy-Load replayte den exakten 16-Record-Store
und autorisierte ausschließlich den registrierten N10-Plan; außen dauerte er
`3,59 s`.

**Einmaliges CPU-/Policy-Gate.** Genau der vorregistrierte Befehl
`benchmark-policy --run-id n10-policy-overhead-20260822-01 --execute` lief
einmal und bestand mit Exit `0`. Cold Load: `3.482.664.083 ns`; Baseline-Median:
`28 ns`; Policy-Median: `12.372 ns`; Policy-MAD: `39 ns`; Policy-p95:
`12.448 ns`; zusätzlicher Median: `12.343 ns`. Damit bestanden die Grenzen
`10 s`, `25 µs`, `50 µs` und `20 µs`. Der äußere Prozess benötigte `10,15 s`
Wall, `10,01 s` User, `0,12 s` System, maximales RSS `33.800.192 B` und keine
Swaps. Der append-only Record lautet
`f140083d80ad1d557e03e8370ee5b2eb0ce2923fe3752006c2d8466d92689306`.

**Einmaliges MLX/GPU-Gate.** Weil das CPU-Gate bestand, lief danach genau
`validate-gpu --run-id n10-runtime-validation-20260822-01 --execute` einmal.
Alle zwölf balancierten Blöcke bestanden. Baseline-Median:
`20.797.458,5 ns`; Kandidaten-Median: `18.220.750 ns`; Baseline-MAD:
`27.604 ns`; Kandidaten-MAD: `19.354,5 ns`; Verhältnis `0,8757526334`, Effekt
`−12,4247367 %`. Beide Arme waren byteidentisch, `max_abs_error=0`, mit demselben
Digest
`2443f000817522f2ce376d1f311b05540a17b6f14b4e23caa52f5e12e3d29462`.
GPU-Arbeit und maximale kontinuierliche GPU-Zeit lagen jeweils bei `0,667235 s`,
Wall des Messvertrags bei `1,095774 s`, Prozess-CPU bei `0,148294 s`; alle
Budgets bestanden. MLX meldete `176.160.768 B` aktiv, `83.886.080 B` Cache und
`260.046.848 B` Peak. Der äußere Prozess endete mit Exit `0` nach `4,87 s`
Wall, `4,02 s` User, `0,22 s` System, maximalem RSS `508.313.600 B` und ohne
Swaps. Record und Kettenspitze lauten
`d6143fcada07018968b9f17aeeb0137a76dee5baeaeef76da12e514d6c4c979f`;
der Circuit Breaker blieb offen.

**Persistenz und echte UI.** `.friday-data/runtime-n10.sqlite3` enthält genau
die zwei Status `policy_overhead_passed` und `runtime_validation_passed`; der
zweite Record verweist auf den ersten und der vollständige Hashketten-Replay
bestand. Datei: Modus `0600`, `53.248 B`, SHA-256
`81286ffa2af11a814ffe4e11cdd67ce7fa5804ff42f4efd094cf161dbae22cd5`,
Snapshot-Revision
`a7b9352b913e62b9faf1e59cec2f5531435121d716e08cf2e7f8f24075f6327e`.
Ein vollständiger read-only Replay ließ den Dateihash bytegleich. Die echte UI
auf `127.0.0.1:8772` lieferte für `/` GET `200` (`1.698 B`, `0,003143 s`),
für `/api/snapshot?limit=2` HEAD/GET `200` und beide Records, wies POST mit
`405` ab und beendete sich per `Ctrl-C` mit Exit `0` ohne Traceback. Danach war
der Port geschlossen; der DB-Hash blieb unverändert.

Ein ergänzender manueller read-only SQL-Zählcheck fragte zunächst irrtümlich
die nicht vorhandene Tabelle `runtime_records` statt `records` ab und endete
mit `no such table`, ohne eine Datei zu verändern. Die anschließende
Schemaabfrage bestätigte `metadata` und `records`; die korrigierte Abfrage fand
genau zwei Zeilen, je einen bestandenen CPU- und GPU-Status, mit korrektem
Vorgängerzeiger. Der DB-Hash blieb auch danach unverändert.

**Integrität und Entscheid.** H1-v2, N8-Runtime, Research, N10-v1 und N10-v2
blieben bytegleich bei
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`,
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`,
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
`e0b5f4af62c128938e1e12e388c16b344a66e18eebf9e0568c7ebe34c5a4f0d5`
und `54e9c57ca6b76fa671b94f748b7ee471575b7dd7445bad00ae3cab38f691fc4f`.
Es gab keine Installation, keinen Download, keine Modellaktion, keine freie
Codegenerierung und kein Custom Metal. Ergebnis ist ein Engineering-GO nur für
exakt FP16-`2048²`, zehn Matmuls und den versiegelten Batch-Dispatch-Plan. Die
Messung erweitert weder den formalen N10-Claim noch den Scope; produktive
Integration, andere Tensoren, Cross-Device, weitere Modellrunden und ein
breiterer Suchraum bleiben ohne neuen Architekturentscheid geschlossen.

**Abschlussprüfung der Dokumentation und ProjectAtlas.** `git diff --check`
bestand. Der ergänzende Runtime-Replay bestätigte erneut zwei Records, die
vollständige Hashkette und Snapshot-Revision
`a7b9352b913e62b9faf1e59cec2f5531435121d716e08cf2e7f8f24075f6327e`;
alle sechs oben genannten Evidenzdateien waren vor und nach dem Replay
bytegleich. `xcodebuild -checkFirstLaunchStatus` bestand. Ein erster
ProjectAtlas-Runtime-Info-Aufruf übergab fälschlich den nicht erlaubten
Positionsparameter `.` und endete read-only mit einem CLI-Nutzungsfehler. Die
korrekte Wiederholung meldete ProjectAtlas `0.4.5-rc1` mit CLI-/MCP-/SQLite-/
Symbolindex-Fähigkeiten. `projectatlas watch --once .` bestand anschließend
mit 764 Textkandidaten, 745 indexierten Textdateien und 607 Symbolkandidaten;
5 Dateien wurden neu geparst, 602 blieben unverändert und kein Parser lief in
ein Timeout.

**Post-Commit-Replay.** Nach dem Ergebniscommit war der Root-Worktree leer.
Der echte read-only Policy-Aufruf autorisierte am neuen sauberen HEAD weiterhin
exakt den 16-Record-N10-v2-Claim mit Decision-Record
`47283e73eb6eefa01dc0f2e1760a2a2d350ca51019b8ddfa3a297d4b695e1249`;
alle sechs Evidenzdateien behielten ihre oben dokumentierten SHA-256-Werte und
`xcodebuild -checkFirstLaunchStatus` bestand erneut. Der abschließende
ProjectAtlas-Refresh blieb bei 764/745 Textdateien und 607 Symbolkandidaten;
eine Datei wurde neu geparst, 606 blieben unverändert und kein Timeout trat
auf. Der Worktree war auch danach leer.

### 2026-08-22 — Nächste Runde: Shadow-Router und sicherer Kernelpfad geöffnet

**Explizite Freigabe und Grenze.** Der Nutzer erlaubt notwendige lokale
Softwareinstallationen und die weitere Arbeit bis hin zu Kernelversuchen unter
hohen Sicherheitsanforderungen. Neue Modelle sind ausdrücklich ausgeschlossen.
Vorhandene Modelle werden in dieser Runde ebenfalls nicht benötigt. Die
Reihenfolge bleibt: zuerst ein rein beobachtender N8/N10-Shadow-Router; erst
nach dessen grünen Gates ein einzelner statischer Custom-Metal-Kandidat in
einem getrennten, kontrollierten Worker.

**Atlas-first.** Vor Projektquellen wurde der versionierte ProjectAtlas-Skill
vollständig gelesen und danach Runtime `0.4.5-rc1` sowie ein fokussierter
`atlas_session_brief` über MCP aufgerufen. Der Brief rankte zunächst drei
Dateien des eingebundenen ProjectAtlas-Repositories; die vorgeschriebene erste
Summary bestätigte, dass diese Treffer ausschließlich ProjectAtlas selbst
betreffen und unangetastet bleiben. Eine anschließende begrenzte Atlas-Suche
führte zu `friday_runtime/`, `friday_runtime_n10/`, ihren Tests und den
bestehenden Architekturtexten. Ein erster Skill-Leseaufruf ließ im expandierten
Alias versehentlich eine Paketebene aus und endete read-only mit `No such file`;
der korrigierte Pfad wurde danach vollständig gelesen.

**Unveränderte Baseline.** Ausgangspunkt ist der saubere lokale `main`-Commit
`25bf6513c279900dfde31cefd6015cff58d4c11a`. H1-v2, N8-Runtime, N10-v1,
N10-v2 und N10-Runtime blieben bei den dokumentierten SHA-256-Werten
`141f010b…e78a4c4`, `ad4f0ef7…6473d82`, `e0b5f4af…a3f0d5`,
`54e9c57c…91fc4f` und `81286ffa…22cd5`; Xcode-First-Launch bestand.
Die acht bestehenden N8/N10-Runtime-Testmodule bestanden zusammen `30/30`
Tests in `3,458 s` (`3,73 s` außen, `3,55 s` User, `0,05 s` System,
maximales RSS `45.498.368 B`, keine Swaps). Beide formalen Policy-Loader
autorisierten read-only; N8 benötigte `3,58 s`, N10 `3,63 s` außen.

MLX `0.32.0` stellt lokal bereits `mx.fast.metal_kernel` und Metal bereit;
eine Installation ist daher derzeit nicht notwendig. Der vorhandene lokale
Gemma-3-4B-Konfigurationssnapshot bestätigt `hidden_size=2560`; es wurde nur
dieses JSON-Feld gelesen, kein Modell geladen. Die aktuelle offizielle MLX-
Dokumentation bestätigt JIT-kompilierte Custom-Metal-Kernel, einmalige
Kernelkonstruktion, standardmäßig sicheren Math-Modus und die mögliche
Contiguous-Kopie. Apples Dokumentation bestätigt die Thread-/Threadgroup- und
SIMD-Synchronisationsgrenzen. Diese Quellen bestimmen die späteren
Kernelkontrollen.

**Diagnosefehler.** Ein ergänzender N8-Policy-Aufruf verwendete zunächst den
nicht vorhandenen Launcher `tools/run_runtime.py`; Python endete vor jeder
Projektaktion mit Exit `2`. Die Dateisuche löste den tatsächlichen read-only
Launcher `tools/run_runtime_prototype.py` auf, dessen Wiederholung bestand.

**Vorregistrierung.** Der neue Vertrag
[`AVO_SHADOW_ROUTER_SPEC.md`](AVO_SHADOW_ROUTER_SPEC.md) friert vor jedem neuen
Router-Code und Timingwert die reine Shadow-Semantik, beide exakten
Evidenzbindungen, CPU-Gates, echte Tensor-Negativfälle, eigene private
Hashketten-DB und read-only UI ein. Weder N8 noch N10 noch ProjectAtlas werden
für den Router verändert.

**Implementierung und frühe Fehlerclosure.** Der neue Namespace
`friday_avo_router/` komponiert die unveränderten N8-/N10-Controller, besitzt
absichtlich keine `execute`-Methode, erzwingt immer `serial_shadow_only` und
speichert ausschließlich die drei vorregistrierten Record-Arten. Der erste
Compile-Lauf fand ein nicht-ASCII-Mittelpunktzeichen in einem Python-Bytestring;
es wurde durch die semantisch identische HTML-Entity `&middot;` ersetzt. Der erste
History-Testlauf übergab `strict_json_loads` irrtümlich `str` statt `bytes` und
erzeugte zwei Fehler. ProjectAtlas verlangte wegen der neuen Dateien einen
Refresh; danach bestätigte ein exakter Slice den Bytes-Vertrag und die
Korrektur verwendet nun UTF-8-Bytes plus feste Größenobergrenze. Die fokussierte
Suite bestand anschließend zunächst `11/11`, nach CLI- und Härtungstests
`16/16` Tests. `ruff` ist in der bestehenden Umgebung nicht vorhanden; da
Compile- und Tests keine Installation erfordern, wurde nichts installiert.

**Vor-Commit-Ausführungssperren.** Beide Messkommandos endeten ohne `--execute`
mit Exit `78` und legten keine DB an. Mit `--execute` sperrte der schmutzige
Worktree die Policy-Messung bereits an der Provenienz; die Shadow-Validierung
ohne vorherigen Policy-Record sperrte ebenfalls vor einer DB-Anlage. Der
read-only Policy-Status meldete erwartungsgemäß für beide Policies
`worktree_dirty`, der Router war nicht bereit. `git diff --check` bestand und
`.friday-data/avo-router.sqlite3` blieb abwesend.

**Hochsicherheits-Diff-Review.** Wegen der Nutzeranforderung „unter hoher
Sicherheit“ wurde zusätzlich der lokale `codex-security:security-diff-scan`
verwendet. Sein Capability-Preflight war bereit; fehlende delegierte Worker
sind aufgrund der bindenden Einzelagentenregel erwartet. Der advisory
Trusted-Access-Status war `not_granted` und blockiert die lokale Analyse nicht.
Eine zunächst angenommene globale Referenzposition für
`threat-model-guidance.md` sowie später für `final-report.md` war falsch und
endete jeweils read-only mit `No such file`; `rg --files` löste die
versionierten skill-lokalen beziehungsweise Plugin-Root-Pfade auf, die danach
vollständig gelesen wurden. ProjectAtlas meldete während dieser Analyse eine
zu große inkrementelle Closure (`10001` Pfade); der vorgeschriebene vollständige
MCP-Watcherlauf endete nach rund 14 Sekunden erfolgreich und indexierte wieder
779 Dateien. ProjectAtlas selbst blieb unverändert.

Der Security-Inventarhelfer erfasste die zehn neuen Produktionsdateien. Ein
erster `resolve_security_md --list`-Aufruf kombinierte unzulässig `--scope` und
endete mit Exit `2`; die dokumentierte Form ohne `--scope` bestätigte, dass nur
`ProjectAtlas/SECURITY.md` existiert und nicht für Friday-Root-Code gilt. Der
ältere Rank-Input-Helfer lieferte bei ausschließlich untracked neuen Dateien
null Zeilen, während der maßgebliche Local-Patch-Inventarhelfer alle zehn
Produktionsdateien korrekt erfasste. Tests und die neue Spezifikation wurden
deshalb zusätzlich manuell vollständig gelesen und in der Coverage erfasst.

Eine erste Snapshot-Hash-Pipeline verwendete in `zsh` versehentlich den
reservierten Arraynamen `path`; dadurch war der Suchpfad nur in dieser
Subshell-Schleife überschrieben, `stat`/`shasum` wurden nicht gefunden und der
resultierende Digest wurde ausdrücklich verworfen. Die Wiederholung mit
`relative_file` und absoluten Binärpfaden bestand. Ein späterer read-only
Hashlistenaufruf verwies einmal auf eine nicht gesetzte
`SECURITY_INVENTORY`-Variable und endete ohne Dateizugriff; der explizite
Scanpfad korrigierte ihn.

**Review-Fixes vor jeder Messung.** Die vollständige Quellprüfung fand keinen
aktiven optimierten Ausführungspfad, aber vier zu schließende Vertrags- und
Integritätslücken. Erstens meldeten falsche Form oder falscher Datentyp trotz
serieller Policy noch `route=n8/n10`; sie melden nun auch auf Routerebene
`route=serial`. Zweitens akzeptierte der CLI die erste Messung unter einer
abweichenden Run-ID; beide Befehle prüfen nun vor DB-Anlage exakt die
vorregistrierte ID. Drittens sind alle Provenienzdateien jetzt größenbegrenzt,
per `O_NOFOLLOW` geöffnet und gegen stabile File-Deskriptor- sowie Pfadidentität
geprüft; Git-Revision und Status werden doppelt gelesen und eine saubere
Freigabe vergleicht Code-/Spec-Hashes zusätzlich mit den HEAD-Blobs. Viertens
prüft die History alle inneren Provenienz-Digests und bindet
`created_at_unix_ns` in jeden Recordhash. Restriktivere CSP-, Frame-, Referrer-
und Cross-Origin-Header härten zusätzlich die Loopback-UI. Neue Regressionstests
decken Serialroute, fremde Run-ID, Zeitstempeltampering und inkonsistente innere
Provenienz ab. Compileall sowie die fokussierte Suite bestanden danach `19/19`
Tests in `0,135 s` (`0,23 s` außen, `0,13 s` User, `0,08 s` System,
maximales RSS `35.241.984 B`, keine Swaps).

**Versiegelter Security-Abschluss und Vollregression.** Der terminale
Security-Diff-Scan band den Working-Tree-Snapshot
`362ae329e5eab1e4cd640bc60e0bc3e95002e4fc4fc8040e543ffc3a26faca9b`,
prüfte zehn deterministisch inventarisierte Produktionsdateien plus vier
vollständig gelesene Tests, Spezifikation und Journal-Hunk, normalisierte null
Kandidaten und wurde mit kompletter Coverage sowie null reportablen Findings
erfolgreich finalisiert und erneut validiert. Der generierte Report hat SHA-256
`0c5a558d908d45b9e6561a5caf90e8bc5d929856ba4e441ecda44ccb282d983b`.
Die vollständige Friday-Suite bestand danach `544/544` Tests in `210,574 s`
(`210,84 s` außen, `199,23 s` User, `1,47 s` System,
maximales RSS `76.496.896 B`, keine Swaps).

Xcode-First-Launch blieb grün. ProjectAtlas meldete Runtime `0.4.5-rc1`,
Schema v18/current, gültigen Publikationsfingerprint und die projektlokale
MCP-Datei `.projectatlas/projectatlas.mcp.json`; der abschließende Refresh
indexierte `760/779` Textkandidaten und parste zehn geänderte Symbolquellen ohne
Timeout. Ein Hashcheck verwendete für die N10-Runtime zunächst versehentlich
`.friday-data/n10-runtime.sqlite3`; nur dieser nicht vorhandene Pfad erzeugte
`No such file`. Das deterministische `.friday-data`-Inventar löste die richtige
Datei `.friday-data/runtime-n10.sqlite3` auf. Danach bestätigten alle fünf
Bestandsdateien unverändert ihre dokumentierten SHA-256-Werte, einschließlich
`81286ffa…22cd5` für N10-Runtime.

**Sauberer Router-Commit und Readiness.** Die Implementierung wurde lokal auf
`main` als Commit `70bc451f764d36e75de0a1c9ac61849717e577e8`
versiegelt; kein Push erfolgte. Auf genau diesem sauberen Commit autorisierte
der read-only Routerstatus N8 mit Record `f508fc9e…9a9357` und N10 mit Record
`47283e73…5e1249`; Cold Load des Statusprozesses betrug außen `7,13 s`, die neue
Router-DB war weiterhin abwesend.

**Einmaliger CPU-Lauf.** Der exakt vorregistrierte Lauf
`avo-router-policy-20260822-01` wurde genau einmal ausgeführt und bestand alle
sechs Gates. Über fünf Warmup- und 21 balancierte Messblöcke mit je 10.000
Entscheidungen pro Arm ergaben sich direkter Median `12,138946 µs`, MAD
`0,042467 µs`, Router-Median `13,719000 µs`, MAD `0,044638 µs`, p95
`13,815279 µs` und gepaarter zusätzlicher Median `1,585208 µs`. Cold Load war
`7,176239584 s`; beide Evidenzpfade und jede Entscheidung stimmten überein.
Prozess-Wall war `13,896576666 s`, CPU `13,772746 s`, Peak-RSS `34.471.936 B`,
Netzteilbetrieb. Der terminale Record ist
`a1a1c1a08eb22c41e442becfe7d6a6a2feb67c2322596eaf1d9fc0a595b253fd`.

**Einmalige echte Shadow-Validierung.** Erst nach dem bestandenen CPU-Record
wurde `avo-router-shadow-20260822-01` genau einmal ausgeführt. Exakt acht und
zehn reale MLX-FP16-`2048²`-RHS-Referenzen empfahlen N8/N10. Operandenzahl neun,
falsche Form und FP32 meldeten korrekt `route=serial`; die direkte Policy und
der Router stimmten in allen Fällen überein, der erzwungene Plan blieb immer
`serial_shadow_only` und `no_matmul_executed=true`. Alle fünf Gates bestanden.
MLX meldete `33.554.432 B` aktiv, `33.554.436 B` Peak und `8 B` Cache;
Prozess-Wall `7,232409250 s`, CPU `6,970062 s`, Peak-RSS `51.560.448 B`,
Netzteilbetrieb. Terminaler Record:
`19e36e7b32209d62afa5eae54973e2dc326a1bd0efaa0d8b8a73737463384c6c`.

**Persistenz- und UI-Abschluss.** Die Router-Historie enthält genau diese zwei
Records, ist vollständig replaybar und endet bei Snapshot-Revision
`b1c0832c0957e5a2d0e88bda1409f8d4b04be036a5edd32a20ea8c2d57b2c758`.
Datei: Modus `0600`, Größe `36.864 B`, SHA-256
`128c090de37a79606f35c564d19035f0bcffedcea4b4018fa618cffedc58c6f8`.
Die echte UI auf `127.0.0.1:8773` lieferte für HTML und API GET `200`, für HEAD
`200`, wies POST mit `405` ab und sendete CSP, `X-Frame-Options: DENY`,
`Cross-Origin-Resource-Policy: same-origin`, `Referrer-Policy: no-referrer` und
`Cache-Control: no-store`. `Ctrl-C` beendete sie mit Exit `0` ohne Traceback;
der DB-Hash blieb exakt bytegleich. Der Worktree war danach sauber. Es lief
kein Modell, keine Matmul und kein Custom-Metal-Kernel; nichts wurde installiert
oder heruntergeladen.

**Promotion.** Der Shadow-Router ist damit `shadow_router_validated`. Das ist
kein produktives Routing-GO. Gemäß Nutzerfreigabe und Vorregistrierung darf als
nächster, getrennt zu dokumentierender Schritt genau ein statischer
Residual-Add-plus-RMSNorm-Metal-Kandidat in einem kontrollierten Worker
vorregistriert und geprüft werden. Adaptive Suche, neue Modelle, produktive
Aktivierung und Cross-Device-Claims bleiben ausgeschlossen.

### 2026-08-22 — Phase 1B: statischer Residual-Add+RMSNorm-Kandidat vorregistriert

**Atlas-first und Freigabegrenze.** Vor Phase 1B wurde erneut ein fokussierter
ProjectAtlas-Session-Brief abgerufen und die erste empfohlene Datei
`tests/test_worker.py` über Atlas zusammengefasst. Die relevanten
Architekturslices bestätigten die bereits dokumentierte Reihenfolge: genau eine
reale Tensoroperation, starke MLX-Baselines, statischer Templatekandidat und
opferbarer Worker. Der Nutzer hat den nächsten isolierten Kernelversuch und die
Nutzung vorhandener CPU/GPU ausdrücklich freigegeben; neue Modelle bleiben
ausgeschlossen. Es war keine Installation und kein Download erforderlich.

**Read-only API-Prüfung und korrigierter Fehlversuch.** Ein erster Docstring-
Check verwendete versehentlich `/opt/homebrew/bin/python3` statt des lexikalischen
Projektlaunchers und endete vor jeder MLX-Aktion mit
`ModuleNotFoundError: No module named 'mlx'`. Die Wiederholung mit
`.venv/bin/python` bestätigte MLX `0.32.0`, `mx.fast.rms_norm`, `mx.compile`,
`mx.fast.metal_kernel`, die Memory-/Cache-/Peak-APIs sowie
`mx.device_info()`. Das Zielgerät meldete Apple M1 Max, Architektur
`applegpu_g13s`, `34.359.738.368 B` Speicher und einen empfohlenen maximalen
Working Set von `26.800.603.136 B`. Dieser Schritt konstruierte oder kompilierte
keinen Kernel und führte keine Timingoperation aus.

**Formale und lokale Quellenprüfung.** Die offizielle MLX-Dokumentation
bestätigte Body-only-Custom-Kernel, generierte Signaturen aus Input-/Outputdtype,
Metal-Attributen, `dispatchThreads`-Grid/Threadgroup und Safe-Math als Default.
Die spezialisierte `mx.fast.rms_norm`-API wurde als zwingende starke Baseline
gebunden. Für die lokal betroffene MLX-Version 0.32.0 ist öffentlich ein Fehler
dokumentiert, bei dem verschiedene Quellen mit gleichem Kernelnamen innerhalb
einer Eval-Batch stale Code ausführen können; der offizielle Workaround ist ein
Quellhashsuffix. Daher enthält der Friday-Kernelname zwingend die ersten zwölf
Hexzeichen seines vollständigen SHA-256 und der Worker darf den Kernel nur
einmal konstruieren.

**Eingefrorener Vertrag vor Compilation.** Die neue Vorregistrierung
`docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md` bindet exakt `(1024,2560)` FP16,
`epsilon=1e-6`, eine 256er-Threadgroup, Safe-Math, FP32-Reduktion, FP16-
Materialisierung des Residuals, sechs volle Correctnessfixtures, unabhängige
FP64-Hard-Caps, vier starke MLX-Baselines, drei frische Prozesse je
Charakterisierung/A/A/A/B, Warmup, 31 gepaarte Bestätigungsblöcke,
hierarchisches 10.000er-Bootstrap, 5-%-Mindestgewinn, Speichergrenzen und harte
Abbruchregeln. Qualification und Benchmark besitzen verschiedene once-only-
Run-IDs; ein negatives Gate ist terminal und darf nicht durch Retry oder
Schwellenänderung umgedeutet werden.

Die einzige statische Quelle liegt als reine Konstante in
`friday_phase1b/kernel_source.py`. Ihr SHA-256 lautet
`33b626c16c79819d6995d6bb78745eb1fd81face648b59f505a924d3125da6f6`,
der gebundene Name
`friday_rrms_f16_r1024_h2560_33b626c16c79`. Ein read-only Python-Check
bestätigte Quelle, Geometrie, Hash und Namen. Bis zu diesem Journalpunkt wurde
der Kernel weder mit MLX konstruiert noch kompiliert oder auf der GPU ausgeführt.

**Darwin-Limit-Preflight vor finaler Versiegelung.** Zwei kurzlebige Prozesse
versuchten ohne Kernelkonstruktion, `RLIMIT_AS=24 GiB` beziehungsweise
`RLIMIT_DATA=4 GiB` zu setzen. Darwin/Python meldete in beiden Fällen sowohl für
Hard- als auch Soft-Limit `ValueError: current limit exceeds maximum limit`;
dies entspricht der bereits in der Phase-1A-Architektur dokumentierten
Plattformgrenze. `RLIMIT_CPU=90 s`, `RLIMIT_CORE=0`, `RLIMIT_FSIZE=16 MiB` und
`RLIMIT_NOFILE=64` ließen sich dagegen in einem separaten Prozess setzen. Die
Spezifikation wurde deshalb noch vor jeder Compilation wahrheitsgemäß
versiegelt: kein behauptetes Adressraumgate, stattdessen feste kleine Shapes,
512-MiB-MLX-Guideline, 2-GiB-Parent-RSS-Watchdog sowie CPU-, Datei-, FD- und
Wall-Clock-Prozessgruppenlimits. Eine harte Unified-Memory- oder native
GPU-Sandboxgarantie bleibt ausdrücklich NO-GO.

**Geschlossener Offline-Unterbau.** Danach wurden ausschließlich statischer
Code und Fakes implementiert: vier feste MLX-Baselineadapter, die einzige
quellhashgebundene Metal-Quelle, deterministische PCG64-Fixtures mit FP64-
Oracle, balancierte Blockreihenfolgen, hierarchische Statistik, ein fester
Worker-Entrypoint, Prozessgruppen-Watchdog mit begrenzten Streams und
RSS-Polling, strikt validiertes kanonisches JSON, unveränderliche
Git-/Environment-/Hardwareprovenienz, append-only SQLite mit Hashkette und eine
read-only Loopback-UI. CLI und HTTP akzeptieren keinen Datenbank-, Source-,
Tensor-, Command- oder Compilerpfad. Qualification und Benchmark sind
unterschiedliche once-only-IDs; auch die Storage-Schicht verweigert dieselbe ID
unter einem anderen Recordtyp.

Der erste Syntaxwrapper setzte nach erfolgreichem `compileall` versehentlich die
in zsh read-only reservierte Variable `status` und endete dort mit
`read-only variable: status`. Die Wiederholung mit `task_exit` bestätigte Exit
`0`; Metal war an keinem dieser Checks beteiligt.

**Früh gefundener Layout-Contractfehler.** Ein realer, nicht getimter
MLX-Array-API-Check zeigte, dass MLX 0.32.0 kein öffentliches
`array.strides`-Attribut anbietet; der erste Prüflauf endete deshalb mit
`AttributeError`, bevor die nachfolgende Host-Fixture erzeugt wurde. Der Worker
behauptet nun keine nicht belegbare MLX-Stride-Introspektion: Er akzeptiert nur
intern erzeugte C-kontiguierliche NumPy-Fixtures, prüft deren Flags vor der
Konvertierung und setzt beim geschlossenen Kerneladapter zusätzlich
`ensure_row_contiguous=True`. Die Vorregistrierung wurde vor jeder
Kernelcompilation entsprechend präzisiert. Eine Wiederholung bestätigte
Shape `(2,3)`, FP16 und das fehlende öffentliche Stridefeld sowie für die volle
Cancellation-Fixture `(1024,2560)` vier C-kontiguierliche Arrays, bitgenau
positiven Null-Oracle und den reproduzierbaren Digest
`92f76cb32c546a26d1c7a67ee6fa1717b617237bdf28f803083d1302381cc440`
in `84.021.459 ns` für zwei Erzeugungen.

**Offline-Verifikation vor Security-Review.** Compileall sowie `17/17`
fokussierte Tests für Quellhash/Geometrie, geschlossene Kernelargumente,
Statistik, vollständige Host-Fixture, Supervisorprotokoll, once-only CLI,
SQLite-Integrität/Tampererkennung und read-only Dashboard bestanden in
`0,479 s` (`0,61 s` außen, `0,54 s` User, `0,05 s` System,
maximales RSS `153.452.544 B`, Peak Memory Footprint `137.953.832 B`, keine
Swaps). Der reale CLI-Status meldete null Records; `qualify` ohne `--execute`
endete mit Exit `78`, und die Phase-1B-DB blieb vor und nach beiden Aufrufen
abwesend. Bis hier wurde kein Custom-Metal-Kernel konstruiert, kompiliert oder
ausgeführt und keine Performanceprobe gestartet.

**Security-Härtung vor dem ersten GPU-Lauf.** Die Review des geschlossenen
Unterbaus fand zwei reale Kandidaten, die vor dem finalen Snapshot behoben
wurden. Erstens erbte die Git-Provenienz noch Umgebungsvariablen und lokale
Git-Konfiguration; damit hätten insbesondere `GIT_DIR` oder ein konfigurierter
FSMonitor die Identitätsprüfung beeinflussen beziehungsweise einen Prozess
starten können. Git läuft nun mit festem Minimalenvironment, ohne System- oder
Globalconfig, Hooks, FSMonitor, Untracked-Cache, externen Diff, Pager und
Prompting. Zweitens prüfte die Loopback-UI den HTTP-Host noch nicht und war
damit prinzipiell für DNS-Rebinding erreichbar. Sie akzeptiert nun ausschließlich
`127.0.0.1:<port>` oder `localhost:<port>`. Ergänzende Tests prüfen beide
Grenzen.

Zusätzlich bindet der Supervisor jeden registrierten Worker an die vollständige
kanonische Provenienz. Derselbe saubere Git-/Dateisnapshot wird unmittelbar vor
dem Spawn und nach dem Prozessende erneut geprüft; Drift invalidiert das
Ergebnis. Ein leerer temporärer `PYTHONPYCACHEPREFIX` verhindert, dass ignorierte
Workspace-Bytecodes als Importquelle dienen. Nach diesen Änderungen bestanden
Compileall und `22/22` fokussierte Tests in `0,486 s`. Ein echter, weiterhin
nicht ausführender Provenienzcheck bestätigte MLX `0.32.0`, Source-SHA
`33b626c…da6f6` und Code-Snapshot-SHA
`6bcab3f6c1711e3ca98445f8381bcf603f9ba68fd5b433f73cf70745708fd0d3`;
der Worktree war dabei erwartungsgemäß noch dirty.

**Versiegelter Phase-1B-Security-Diff.** Der vollständige terminale Scan prüfte
alle 16 deterministisch inventarisierten Produktionsdateien über acht
Sicherheitsflächen. Snapshot:
`codex-security-snapshot/v1:sha256:2e20f9d59bce00e8eec4ee25ef2f0751787cacb5af3ee8aaa6e3f36c2983ec9c`.
Coverage ist vollständig, Deferred Work `0`, reportable Findings `0`; beide
zuvor erkannten Kandidaten sind im versiegelten Snapshot bereits geschlossen.
Der generierte Report liegt unter
`/private/tmp/codex-security-scans/Project_Friday/4db9066_worktree_20260822T094428Z/report.md`
und hat SHA-256
`2abe7b3dfea33b80ce199ebc9d2b636299f8b05fbff918084835a1aee68e57e4`.

Der erste Finalizeraufruf verwendete den auf macOS symbolischen Pfad `/tmp`
und verweigerte korrekt mit
`scan directory: expected a canonical non-symlink directory`. Ursache war
`/tmp -> /private/tmp`, nicht ein Scan- oder Quellfehler. Die Fortsetzung prüfte
die vorhandenen unversiegelten Artefakte bytegenau und finalisierte einmalig
über den kanonischen `/private/tmp/...`-Pfad mit Exit `0`. ProjectAtlas wurde
vor der Fortsetzung inkrementell aktualisiert: 27 geänderte Pfade, 784/803
Textkandidaten und 25 geänderte Symbolquellen ohne Timeout. Bis zum versiegelten
Security-Abschluss wurde weiterhin kein Custom-Metal-Kernel konstruiert,
kompiliert oder ausgeführt.

**Vollregression vor Commit.** Auf dem security-geprüften Working-Tree-Snapshot
bestand die vollständige Friday-Suite `566/566` Tests in `212,552 s`
(`212,85 s` außen, `200,95 s` User, `1,53 s` System,
maximales RSS `186.433.536 B`, Peak Memory Footprint `161.055.344 B`, keine
Swaps). Sämtliche ausführbaren Studienpfade blieben ohne `--execute` gesperrt;
insbesondere wurde noch kein Phase-1B-Live-Record erzeugt.

**Readiness vor Versiegelung.** `xcodebuild -checkFirstLaunchStatus` endete mit
Exit `0`; Phase-1B-Status meldete null Records und die feste DB war abwesend.
Source-Hash und Kernelname blieben unverändert. Der erste Runtime-CLI-Check
nahm irrtümlich `.projectatlas/bin/projectatlas` an und meldete
`no such file or directory`; dieser Projektpfad existiert nicht. Der bereits
von Atlas gemeldete kanonische Launcher
`/Users/tobiasburandt/.local/bin/projectatlas` bestätigte anschließend Runtime
`0.4.5-rc1`, Major `3`, MCP/SQLite/TOON sowie die erwarteten Fähigkeiten; die
projektlokale MCP-JSON-Datei ließ sich fehlerfrei parsen. `git diff --check`
blieb grün.

**Finaler Commit-Snapshot.** Das erste vollständige Staging deckte in sieben
neu angelegten Dateien jeweils eine überflüssige Leerzeile am Dateiende auf.
Sie wurde entfernt; `git diff --cached --check` und danach erneut `22/22`
fokussierte Tests (`0,483 s`) bestanden. Weil davon fünf Produktionsdateien
betroffen waren, wurde der vorherige Security-Abschluss trotz ausschließlich
formaler Byteänderungen nicht als Commit-Freigabe wiederverwendet. Ein zweiter
kanonischer Security-Diff band exakt den finalen Produktionssnapshot
`codex-security-snapshot/v1:sha256:232c04fe0b4cfbb896120961d91f779b582e04c71b54cbad6bedcaca4c88fa26`,
verglich die fünf Byteänderungen mit der vollständigen Vorprüfung und wurde mit
kompletter Coverage, acht geprüften Flächen, null Deferred Work und null
Findings finalisiert. Finaler Report:
`/private/tmp/codex-security-scans/Project_Friday/4db9066_worktree_20260822T095956Z/report.md`,
SHA-256 `e38a8091d8152810e2de2f628f0b165a32d4d1ce439ab7a65a878ab1033eba50`.
Die erste versiegelte Fassung bleibt als Auditspur erhalten, ist aber nicht die
Freigabegrundlage des Commits.

**Sauberer Phase-1B-Commit und Provenienz.** Der freigegebene Stand wurde lokal
auf `main` als Commit `ea8f95980ac6da513c374aa658b4d2d4cc4a9d20`
versiegelt; kein Push erfolgte. Der unmittelbar danach leere Rootstatus band
Code-SHA `3d7b3707a2df57c49bc98231a13d778643739d5f9a4b4b71fbf01a3a4b8b70db`,
Spec-SHA `2638a9fbcd1db3b48a7f38af2d3d3bbd1f203774fbc7c633b18907dc6edfaf41`
und Gesamtprovenienz
`ed17935733b2a96cafa8bec8f67fd928cf8e3f6d6917af9633ecfb61cc762a91`.
MLX blieb `0.32.0`, Gerät `MacBookPro18,2` / Apple M1 Max. Qualification und
Benchmark liefen ohne getrackte Änderung unter exakt derselben Provenienz.

**Einmalige Custom-Metal-Qualifikation.** Der erste reale Compile-/GPU-Lauf
`phase1b-qualify-20260822-01` bestand alle Gates. Alle sechs Fixtures sowie
Kandidat, vier Baselines und sämtliche Pairchecks waren korrekt. Der Kandidat
hatte über alle Fixtures maximal `0,001953125` absoluten Fehler,
`rel_q99=0` und maximal normalisiertes L2
`6,662275251620369e-06`. Erste Compilation plus Eval dauerte
`193.043.625 ns`. MLX meldete `26.224.648 B` aktiv, `62.934.024 B` Cache und
`89.153.552 B` Peak; Worker/Parent-RSS erreichten maximal `298.926.080 B`.
Worker-Wall war `1,653064500 s`, Controller-Wall `2,314143750 s`, der äußere
Prozess `2,71 s`; stderr war leer und Netzbetrieb bestätigt. Terminaler Record:
`1ff03f1b7b2377c9c2c69f2f3550960b3523249d882d2d5381f2c712de456d6e`.

**Einmaliger kontrollierter Benchmark und gültiger Negativbefund.** Drei
Charakterisierungsprozesse ließen `fast_rms_norm` und
`compiled_fast_rms_norm` innerhalb der vorregistrierten `0,5 %`-Tiezone; die
festgelegte Präzedenz wählte `fast_rms_norm`. Das anschließende A/A-Gate
bestand vollständig: Verhältnis `1,0034448058`, 95-%-KI
`[0,9977671648; 1,0092399450]`, Sessionratios
`[1,0063780365; 1,0034448058; 1,0011522733]`.

Alle drei A/B-Prozesse bestanden Correctness, Pair-, Speicher- und
Einzelsession-Gates. Der hierarchische Kandidat/Baseline-Quotient war
`0,9812981009`, 95-%-KI `[0,9721235180; 0,9859002169]`, mit Sessionratios
`[0,9812981009; 0,9731865910; 0,9825570960]`. Damit ist der Kandidat zwar
reproduzierbar rund `1,870 %` schneller und das Intervall schließt `1` aus,
er verfehlt aber klar die vorregistrierte Mindestverbesserung von `5 %`.
Flattened Mediane waren `326.383,34 ns` für die Baseline und `319.930,82 ns`
für den Kandidaten. Beide Arme meldeten in jeder Speichersession denselben
MLX-Peak von `15.733.760 B`; maximales Worker-RSS war `237.617.152 B`.

Der Controller endete nach `20,074599083 s`; der äußere Prozess nach `20,46 s`
mit dem vertraglich erwarteten Exit `2`. Der terminale Status ist
`candidate_inconclusive`, Aktion `baseline_fallback`, `formal_claim=false`.
Kein Retry, keine Schwellenänderung und keine Runtimeaktivierung sind zulässig.
Benchmarkrecord:
`f051b1f8ca08d3e7adf68f06cf095c3d961a243f100717a6a7b87dcc965595a6`.

**Persistenz- und UI-Abschluss.** `.friday-data/phase1b-rmsnorm.sqlite3`
enthält genau beide einmaligen Records unter identischer Provenienz, Modus
`0600`, Größe `86.016 B`, SHA-256
`4ba0cbd679083683b2504dbf174691402aa851967b88befadeb4035145558452`
und Snapshot-Revision
`45a65df53f27ad8c79cfb6583be3566a1b4bb41f160e68e530bfeec5a1ab031b`.
Die echte read-only UI auf Port `8774` lieferte GET/HEAD `200`, POST `405`,
wies einen DNS-Rebinding-Host mit `421` ab und sendete CSP,
`X-Frame-Options: DENY`, `Cross-Origin-Resource-Policy: same-origin`,
`Referrer-Policy: no-referrer` sowie `Cache-Control: no-store`. Der API-Snapshot
verifizierte Hashkette, beide Statuswerte und dieselbe Revision; der DB-Hash
blieb bytegleich. `Ctrl-C` beendete die UI mit Exit `0`. Es wurde kein Modell
geladen, nichts heruntergeladen oder installiert.

### 2026-08-24 — Zyklus 11: KV-Reallokationen und fehlende ITL-Quantile

**Ziel.** Die bereits vorregistrierte Beobachtungsstudie
`kv-cache-realloc-20260824-01` unverändert und genau einmal ausführen, Cachewachstum
im Decode lokalisieren und p50/p95/p99 der Inter-Token-Latenz ergänzen.

**Provenienz und Ausführung.** Präregistrierungs-SHA
`ce00e013f98f3c4b22a11cf0dc8d50f3206dbd2ea2151207ad6ffdac112af486`,
Script-SHA `25f638fd414104c5902a88e8e5a49f5dd3175cbe8985f565977c301a95b7ee71`.
Einmaliger Aufruf von
`.venv/bin/python experiments/kv_realloc/measure_kv_realloc.py --execute`, Exit `0`,
Netzbetrieb, `BudgetGuard`, Duty `0,15`. Kein Hardware-Retry.

**Messung.** Acht Wiederholungen nach einem vorregistriert verworfenen Warmup,
`765` Prompt-Token, `48` Decodeschritte. Cacheformänderungen lagen konsistent an
Schritt `1` (`29` rotierende Layer, Überschuss `31,5853` ms) und Schritt `4`
(`5` globale Layer, `0,2968` ms). Median ohne Ereignis `14,2671` ms; ITL p50
`14,2670`, p95 `15,1385`, p99 `46,7879`, min/max `13,8230/49,4430` ms.
Alle Token waren identisch. Guard: `21,086457` s GPU-Arbeit, `116,119931` s
Pflichtpausen, maximal `1,270024` s kontinuierlich, `139,595394` s Wall.

**Rechnung und Entscheidung.** Ereignissumme `31,8821` ms beziehungsweise
`4,4263 %` des Decodes; mittlerer Gruppenüberschuss `15,9411` ms. Damit hielten H1,
H2 und H3 nach der vorab festgelegten Gruppenentscheidung; Status
`candidate_recommended_for_preregistration`, weiterhin `formal_claim=false`.
Die zuvor gerechneten Kopierkosten (`0,7616` und `0,1317` ms, zusammen erwartete
`0,13 %`) waren keine Grenzkostenmessung und wurden widerlegt. Der große Wert an
Schritt `1` bleibt mit sonstigen Erstschrittkosten konfundiert; `4,4263 %` ist kein
behaupteter Optimierungsgewinn. Ein kausaler A/B-Pfad wurde als freigabepflichtige
Architekturänderung dokumentiert und nicht implementiert. Ergebniscommit:
`1dff9a5ec95f0a28709ad261a7cfb2d5d8163c89`.

### 2026-08-24 — Zyklus 12: formale Prefill-Head-Skip-Studie

**Entscheidung.** Unter den noch empfohlenen Kandidaten wurde genau der LM-Head-Skip
gewählt, weil er als einziger den gemessenen Hauptengpass Prefill traf. Der
persistente Prozess und die Readback-Kandidaten blieben unangetastet. Vor Messung
wurden Präregistrierung, Harness, acht fokussierte Tests, A/A-/A/B-Plan,
Correctness-Gates, MDE-Regel, Abbruchregeln, eigene append-only Evidenz und read-only
UI auf Commit `9466bb9f9f01813bcbd86b6d16837e90ad2523da` versiegelt.

**Fehler vor der Versiegelung.** Ein Offline-Selbsttest importierte
`canonical_json` zunächst aus dem falschen Modul `friday_h1.canonical`. Ursache war
eine falsche Annahme über die bestehende API; die tatsächliche Funktion liegt in
`friday_evidence.canonical`. Der Import wurde vor Commit und vor jeder Hardware-/DB-
Evidenz korrigiert, fokussierte und vollständige Tests bestanden danach. Diese
Lösung ist bei Folgearbeiten zu berücksichtigen: bestehende APIs immer aus dem
tatsächlichen Modul importieren, keine Namensähnlichkeit als Beleg verwenden.

**Versiegelter Vertrag.** Studie `head-skip-prefill-v1-20260824`, Kandidat
`prefill-head-skip-20260824-02`, lokaler Gemma-Snapshot-Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`; Dokument-SHA
`8f7a9a854639824d337aa9ff3ef97ae2255c804291577c5021af2e93abbbeec6`,
Script-SHA `b39bd6be0768173d293647d45cc7f0d3b1c469fd234375c8f0d46ce3c227dc14`.
Workload: `897` Prompt-Token, Chunk `256`, Batch `1`, greedy ohne Prompt-Logprobs,
`32` Correctness-Token; sechs A/A- plus sechs A/B-Prozesse, je vier Messpaare nach
zwei Warmup-Paaren. Duty-Policy `0,15`, Pacing `0,14`. Kein Hardwareprozess wurde
wiederholt.

**A/A-Messung und MDE.** Sessionquotienten
`[0,998498; 1,004692; 0,994007; 1,005769; 1,004463; 1,001198]`, aggregiert
`1,002829`, 95-%-KI `[0,994931; 1,005964]`, Session-SD `0,004526`; alle Gates
bestanden. Die rohe, aus diesen Messblöcken gerechnete MDE war `0,7391 %`; gemäß
Vorregistrierung wurde sie unverändert auf den konservativen Boden `5 %` gesetzt und
im Confirmation-Seal eingefroren.

**A/B-Messung.** C/V-Sessionquotienten:
`C0=0,845257`, `V0=0,846173`, `C1=0,843401`, `V1=0,847653`,
`C2=0,846596`, `V2=0,852478`. Alle zwölf Sessiongates der Gesamtstudie meldeten
identische greedy Token-IDs; gemeinsamer Token-SHA
`666dcfb103d263a12b29ed9a1c1ec496c6922f96c3a6e7cec083eab47fb5127c`.
Kein Ausreißer wurde verworfen. Beide Arme meldeten denselben MLX-Peak
`3.213.903.666` Byte; RSS `3.768.795.136` bis `3.769.696.256` Byte.

**Vorregistrierte Rechnung und terminale Entscheidung.** Charakterisierung
`R=0,845257`, KI `[0,840544; 0,848452]`; Validierung `R=0,847653`, KI
`[0,842683; 0,854941]`; gesamt `R=0,846385`, KI
`[0,843147; 0,851284]`. Alle oberen Grenzen lagen unter `0,95`; gerechneter Effekt
`−15,3615 %`. Status `head_skip_gain_confirmed`, Aktion
`permit_bounded_architecture_review`, genau ein Record `formal_claim=true`.
Der Claim ist eng auf ein Gerät, einen Modell-Snapshot, einen Prompt, einen
Prefill-Plan und greedy ohne Prompt-Logprobs begrenzt. Keine Produktaktivierung.

**Ressourcenrechnung.** Aus den zwölf Sessionrecords summiert: `332,277940` s
GPU-Arbeit, `3.077,978881` s Pflichtpausen und `3.430,234516` s Session-Wall.
Die Laufzeit wurde in jedem Messblock vor `charge()` gestoppt; Guard-Ruhezeiten sind
nicht in den Prefill-Endpunkten enthalten.

**Persistenz und UI.** `.friday-data/head-skip-v1.sqlite3`: `16` Records, Modus
`0600`, `77.824` Byte, SHA-256
`15ee462bbad5a8f757373f093fdf2ccfb8bdd0048c03447c1cb635acd38ec8d9`,
Kettenkopf `8a568e61f0e087794b1997f273e580c72e7f5abaa1eb8bad7954b303dd38a2d4`.
Replay bestand und ließ die Datei unverändert. Ein Diagnoseversuch per `HEAD`
antwortete `501`, weil diese UI nur `GET` implementiert; GET lieferte anschließend
`200` und die echte Historie. Das ist die dauerhafte Lösung für diesen Server.
Manuelles `Ctrl-C` stoppte ihn mit sichtbarem `KeyboardInterrupt`/Exit `1`; nach dem
terminalen Entscheid blieb der versiegelte Code bewusst unverändert.

**Grenzen.** Integration ist wegen Architektur- und API-Auswirkungen in
`PERMISSION_REQUIRED.md` eingetragen. Multi-Turn-Fortsetzung und mehrere parallele
Requests fehlen weiterhin als Baselines. Das registrierte Maximum von zwölf Zyklen
ist erreicht. Es gab weder Installation noch Download, Systemänderung, Push,
Mutation einer versiegelten Spezifikation/Evidence-DB oder automatische Aktivierung.

**Abschlussverifikation.** ProjectAtlas-Refresh: `925/944` Textkandidaten indexiert,
`7` geänderte Symbolquellen geparst, `700` unverändert, kein Timeout; Runtime
`0.4.5-rc1`. Die projektlokale MCP-JSON-Konfiguration war parsebar.
`xcodebuild -checkFirstLaunchStatus` endete mit Exit `0`. Das erste reine
Versionsdiagnostik-Kommando nahm irrtümlich `mlx.__version__` an und endete mit
`AttributeError`; die belastbare Paketversion wird über
`importlib.metadata.version("mlx")` gelesen. Ein anschließender Gerätecheck nutzte
zunächst das als veraltet markierte `mx.metal.device_info`; der finale Check über
`mx.device_info()` lief warnungsfrei und bestätigte MLX `0.32.0`, mlx-lm `0.31.3`,
`Device(gpu, 0)` und Apple M1 Max. Beide Diagnosefehler erzeugten keine GPU-Arbeit
und keine Evidenzdatei.

Der read-only Studien-Replay bestand mit `16` Records, genau einem formalen Claim
und Kettenkopf `8a568e61…a2d4`; die DB-SHA war vor und nach dem Replay identisch.
Präregistrierungs- und Script-SHA blieben exakt versiegelt. Die vollständige Suite
`.venv/bin/python -m pytest -q` erreichte `100 %` und Exit `0` in äußerer Wall-Zeit
`41,86` s. Beide Ergebnis-JSON-Dateien, die MCP-JSON-Datei und `git diff --check`
bestanden die abschließende Syntax-/Whitespace-Prüfung.

Der erste read-only JSON-Quervergleich adressierte `limits` irrtümlich als
Top-Level-Feld und endete mit `KeyError`; in der tatsächlichen Matrix liegt es unter
`gates.limits`. Der korrigierte Check verwendete die vorhandene Struktur, verglich
Status, Zykluszahl, Quotient, Decision-SHA und Recordzahlen zwischen Matrix und
Ergebnisdatei und bestand. Keine Datei oder Evidenz wurde durch den Fehlversuch
verändert.

### 2026-08-24 — Head-Skip-Runtime-Prototyp vor Live-Gates

**Freigabe und Grenze.** Auf die einfache Zwischenbilanz, dass der Head-Skip im
engen Versuch rund `15,4 %` Prefill-Zeit sparte, aber noch nicht in das Programm
eingebaut war, erteilte der Nutzer ausdrücklich die angeforderte
Architekturfreigabe. Sie gilt nur für den kleinen, rückrollbaren Prototyp und die
einmalige vorregistrierte Qualifikation. Installation, Download, Systemänderung,
breitere Produktaktivierung und Erweiterung des formalen Einzelworkload-Claims
bleiben ausgeschlossen. Es wurden keine Subagenten verwendet.

**Atlas-first und unveränderte Baseline.** Vor Quellarbeit wurden der vollständig
gelesene ProjectAtlas-Skill, Runtime `0.4.5-rc1`, ein fokussierter MCP-Session-Brief
und anschließend nur begrenzte Atlas-Suchen/Summaries/Slices verwendet. Der Brief
rankte zunächst irrelevante Dateien des eingebundenen ProjectAtlas-Repositories;
die erste Summary bestätigte deren Upstream-Zugehörigkeit, und die Suche wurde auf
die vorhandenen Friday-Runtimes, Tests und Architekturtexte verengt. Die relevante
Bestandsbaseline aus vier bestehenden Runtime-Testmodulen bestand mit Exit `0` in
`4,44 s` außen, `11,08 s` User und `0,58 s` System; maximales RSS
`50.380.800 B`, Peak-Footprint `34.308.912 B`, keine Swaps.

**Vor Messung eingefrorener Vertrag.** `docs/HEAD_SKIP_RUNTIME_SPEC.md` hat SHA-256
`4f0f1a9e4d1cd419f0b7686bd5ca9db2866489c04654b93000cca8fffa45e5a9`, die
Mini-Vorregistrierung `experiments/head_skip_runtime/PREREGISTRATION.md` SHA-256
`38da7b1a180c2107ca4c6a754365c4da18bb7336f9eefe36a7c840c4c77c8306`.
Qualification-ID `head-skip-runtime-qualification-20260824-01`, durchgehend
`formal_claim=false`, kein Zyklus 13. Vorab festgelegt wurden H1 Tokenidentität und
Pfadmarker, H2 exakte Auswahl/Fallback, H3 Zeitquotient höchstens `0,95`, H4
gecachter Policy-Median höchstens `25.000 ns`, p95 höchstens `50.000 ns`, Zusatz
höchstens `20.000 ns` und Load unter `5 s`, H5 höchstens `128 MiB` zusätzlicher
MLX-Peak. CPU: fünf Warmup-, 21 Messblöcke, je 20.000 Aufrufe pro Arm. GPU: ein
Prozess mit Correctness-Paar, einem B/A-Warmup-Paar und vier Messpaaren A/B, B/A,
A/B, B/A. Duty `0,15`, Messende vor Guard-Charge, keine Ausreißer, kein
Hardware-Retry.

**Implementierung.** Das getrennte Paket `friday_head_skip_runtime/` enthält eine
byte- und hashgebundene Prüfung der unveränderten formalen 16-Record-Evidenz, eine
exakte Scopeentscheidung, gecachte Policy, sichtbaren Baseline-Rückfall, Circuit
Breaker ohne Same-Call-Retry, den aus der formalen Studie abgeleiteten MLX-Body/Head-
Pfad, die kontrollierte Qualifikation, private SQLite-v1-Historie und eine nur an
`127.0.0.1:8775` gebundene read-only UI. Der Adapter löst ausschließlich den lokal
gebundenen Gemma-Snapshot auf. Unklare Typen, ungültige Backend-Metadaten oder
unplausible Ausgaben können den schnellen Pfad nicht autorisieren. Die vollständige
Anfrage wird an den Referenzpfad weitergereicht; der enge Qualifikationsadapter
lehnt nicht unterstützte Semantik sichtbar ab, statt sie still zu ignorieren.

**Offline-Verifikation.** `20` neue fokussierte Tests für Evidenzreplay,
Scopeabweichungen, Typverwechslungen, Circuit Breaker, Ausgabevertrag, Messreihen,
Budget, private Hashkette, read-only UI und Ausführungssperren bestanden mit Exit
`0`. Bytecode-Kompilation und `git diff --check` bestanden. Ein absichtlich im
schmutzigen Pre-Commit-Stand ausgeführter read-only Policy-Load replayte alle `16`
formalen Records in `0,94 s` außen und fiel korrekt mit `worktree_dirty` und Exit
`2` auf die Baseline zurück; maximales RSS `37.961.728 B`, Peak-Footprint
`26.411.464 B`, keine Swaps. Es entstand keine
`.friday-data/head-skip-runtime.sqlite3` und keine GPU-Arbeit.

**Fehler, Ursachen und Korrekturen vor Live.** Eine Atlas-Abfrage mit
Inhaltsklassifizierung scheiterte an der Fehlklassifikation von
`tests/0001_initial.sql`; die unklassifizierte, weiterhin begrenzte Wiederholung
bestand. Zwei read-only Diagnoseannahmen über die formale Storage-API waren falsch:
Sie besitzt kein `close()`, sondern einen Context Manager, und
`verified_records()` liefert Payloads mit `kind`, nicht Wrapper mit
`record_kind`; beide Aufrufe wurden entsprechend korrigiert, ohne DB-Mutation. Der
erste neue Testlauf fand `DEFAULT_LIMIT=100` bei einem Historienmaximum von `64`;
der Default wurde auf `64` korrigiert und alle Tests wiederholt. Die vertiefte
Codeprüfung fand außerdem, dass Python-Werte `0/1` als Bool-Werte hätten
durchrutschen und dass der erste Entwurf abweichende Requestsemantik nicht an den
Referenzpfad weiterreichte; strikte Typprüfung, vollständige Requestweitergabe und
Ausgabevalidierung schließen beides. `ruff` ist in der bestehenden Umgebung nicht
installiert; gemäß Installationsgrenze wurde es nicht nachinstalliert. Ein zunächst
angenommener Security-Referenzpfad unterhalb des Skill-Verzeichnisses existierte
nicht; `rg --files` löste die tatsächliche Datei unter dem Plugin-Root auf. Mehrere
zu große kombinierte Skill-Leseausgaben wurden nur in der Anzeige gekürzt; alle
Pflichtdateien wurden danach einzeln bis zum Ende gelesen. Ein kombinierter
Dokumentationspatch nahm den Journal-Schlusshunk fälschlich auch für
`PROJECT_STATUS.md` an und scheiterte atomar; die dateigenauen Patches bestanden.
Ein erster Log-Patch traf wegen eines abweichenden Zeilenumbruchs nicht und wurde
mit dem tatsächlichen Kontext wiederholt. Diese Fehler erzeugten keine Mess- oder
Hardwareevidenz.

**Sicherheitsstand und nächster Schritt.** Der lokale Security-Preflight war
`ready`; die nicht verfügbare Worker-Delegation wird durch vollständige
Elternprüfung jeder Datei kompensiert. Der zusätzliche Codex-Sicherheitszugang ist
nicht freigeschaltet; die Prüfung läuft deshalb lokal weiter. Vor dem sauberen
Implementierungscommit folgen vollständige Diff-Sicherheitsprüfung und Friday-
Vollsuite. Danach ist genau der vorregistrierte CPU-Lauf zulässig; nur bei seinem
Bestehen folgt genau ein MLX/GPU-Qualifikationsprozess. Bis dahin ist noch keine
neue Runtime-Geschwindigkeitsverbesserung gemessen.

**Zusätzliche Schutzprüfung vor Live.** Die vollständige Elternprüfung fand drei
relevante Schutzlücken im ersten Integrationsentwurf. Erstens hätte bereits die
formale Studie den schnellen normalen Aufruf autorisiert, obwohl die neue CPU- und
GPU-Qualifikation noch fehlte. Zweitens erlaubten frei wählbare Lauf-IDs und
Datenbankpfade einen neuen Hardwareversuch. Drittens behandelte das Ressourcengate
einen nicht lesbaren Swap-Wert wie einen Erfolg. Der normale Aufruf verlangt nun
die exakte dreiteilige Historie aus bestandenem CPU-Gate, vor Hardwarearbeit
gespeicherter Startmarke und bestandenem GPU-Gate; Lauf-IDs und Datenbankpfad sind
eingefroren, und unbekannter Swap-Verbrauch scheitert geschlossen. Gegenproben
decken fehlende, zusätzliche und veränderte Records, falsche Pfade und unbekannten
Swap ab. Kein Hardwarelauf fand während dieser Korrektur statt.

Die vier bestehenden Runtime-Testmodule bestanden nach dem Einbau erneut mit Exit
`0` in `0,90 s`, maximal `56.508.416 B` RSS, Peak-Footprint `34.538.312 B` und ohne
Swaps. Die vollständige Projekttestsammlung bestand mit Exit `0` in `38,50 s`,
maximal `192.921.600 B` RSS, Peak-Footprint `46.007.136 B` und ohne Swaps. Der erste
Vollsuite-Aufruf endete genau an der Werkzeug-Ausgabegrenze; der Prozess war danach
beendet, aber sein Rückgabecode nicht mehr verfügbar. Da dies ausschließlich
Softwaretests und kein Hardwarelauf waren, wurde die Suite einmal mit gespeicherter
Sitzungskennung wiederholt und lieferte den dokumentierten Exit `0`.

Der gehostete Sicherheitslauf konnte den absichtlich fremd verschmutzten
`ProjectAtlas`-Unterbaum nicht als sauberen Snapshot übernehmen. Dieser fremde
Unterbaum blieb unverändert; die Prüfung wurde nach dem vorgesehenen lokalen
Diff-Verfahren fortgesetzt. Ein erster lokaler Rangierlauf sah neue, noch nicht im
Index erfasste Dateien nicht; das explizite Staging ausschließlich der eigenen
Dateien stellte den vollständigen Quellumfang her. Eine Diagnose suchte außerdem
zunächst eine nicht vorhandene `results/DECISION.json` der formalen Studie; die
tatsächliche terminale Evidenz wurde stattdessen read-only über ihre versiegelte
SQLite-Historie und den vorhandenen Budgetcode geprüft. Diese Diagnosefehler
änderten weder Quellcode noch Evidenz.

**Abgeschlossene lokale Sicherheitsprüfung.** Der endgültige Quell-Diff wurde als
deterministische 12-Dateien-Liste erfasst und jede Datei vollständig durch den
Elternagenten geprüft. Die versiegelten kanonischen Artefakte liegen unter
`/private/tmp/codex-security-scans/Project_Friday/c7db74f_20260824T084223Z`;
Snapshot-Digest
`codex-security-snapshot/v1:sha256:b8c9cf2187a21b5413cf1e4a57b0c230f7cfb0fc0ec0c185a7553288f40c9d95`,
Abdeckung `complete`, keine Deferred Rows und `0` berichtspflichtige Findings. Der
lokale read-only UI-Zugriff wurde unter dem dokumentierten Single-User-
Loopback-Threat-Model als nicht berichtspflichtig verworfen. Beim Erzeugen der
Deep-Review-Liste wurde zunächst fälschlich `--input` statt der vom vorhandenen
Tool verlangten Option `--rank-input` verwendet; der Aufruf brach vor dem Schreiben
der Zielliste ab, die korrigierte Option erzeugte danach exakt dieselben `12`
Pfade wie das Rangierinventar. Der Finalizer wurde nach allen Vorprüfungen genau
einmal ausgeführt und bestand mit Exit `0`.

**Abschließende Umgebungsprüfung vor Commit.** ProjectAtlas wurde nach den
Änderungen einmal kontrolliert aktualisiert; Runtime `0.4.5-rc1`, projektlokale
MCP-JSON gültig. `xcodebuild -checkFirstLaunchStatus` bestand mit Exit `0`.
Der erste reine Import-/Versionscheck nahm fälschlich `mlx.__version__` an und
endete vor jeder Tensor- oder GPU-Arbeit mit `AttributeError`. Die vorhandenen
Paketversionen wurden danach über `importlib.metadata` gelesen: MLX `0.32.0`,
`mlx-lm 0.31.3`, Maschine `arm64`; `mlx.core` ließ sich importieren. Es wurde kein
zusätzlicher Hardwaretest ausgeführt.

**Vor-CPU-Abbruch ohne Messung.** Nach dem sauberen Commit bestätigte ein read-only
Normalaufruf erwartungsgemäß, dass ohne neue Runtime-Historie nur die Baseline
erlaubt ist (Exit `2`, `runtime_evidence_unavailable_or_invalid`); die Datenbank
blieb abwesend. Unmittelbar vor dem ersten CPU-Lauf zeigte der erneute Abgleich mit
dem eingefrorenen Architekturvertrag, dass dessen Netz- und BudgetGuard-Regel für
die gesamte Live-Qualifikation gilt, während der CPU-Harness sie nur beim späteren
MLX-Lauf erzwang. Es war noch keine CPU- oder GPU-Messung gestartet. Der CPU-Harness
prüft nun ebenfalls Netzbetrieb, läuft unter demselben Guard mit Duty-Grenze `0,15`
und speichert dessen Zusammenfassung; mangels GPU-Arbeit bleibt seine GPU-Zeit `0`.
Hypothesen, Workload, Schwellen und Entscheidungstabelle wurden nicht geändert.

Der abschließende Zwei-Dateien-Sicherheitsdelta für diese Korrektur ist unter
`/private/tmp/codex-security-scans/Project_Friday/5b17bfb_20260824T085130Z`
versiegelt. Beide Quellzeilen wurden vollständig gelesen, die Abdeckung ist
`complete`, es gibt keine offene Arbeit und `0` berichtspflichtige Findings. Die
fokussierten `20` Tests und die vollständige Projektsuite bestanden nach der
Korrektur; die Vollsuite lief `37,23 s`, maximal `189.988.864 B` RSS,
Peak-Footprint `46.285.616 B`, keine Swaps.

### 2026-08-24 — Head-Skip-Runtime-Qualifikation, terminales Ergebnis

**Ausführungsreihenfolge.** Nach dem Korrekturcommit `a151c93` wurde zuerst genau
der vorregistrierte CPU-Lauf
`head-skip-policy-overhead-20260824-01` ausgeführt. Er bestand, bevor der einzige
GPU-Lauf `head-skip-runtime-validation-20260824-01` gestartet wurde. Der
GPU-Prozess endete mit Exit `0` und wurde nicht wiederholt. Spezifikation und
Vorregistrierung blieben mit SHA-256
`4f0f1a9e4d1cd419f0b7686bd5ca9db2866489c04654b93000cca8fffa45e5a9` und
`38da7b1a180c2107ca4c6a754365c4da18bb7336f9eefe36a7c840c4c77c8306`
bytegleich.

**CPU-Messung.** Direkter Aufruf Median `25,1396 ns`, Policy-Aufruf Median
`864,7 ns`, p95 `872,47295 ns`, inkrementeller Median `839,47295 ns`. Netzbetrieb,
Duty-Grenze `0,15` und BudgetGuard waren aktiv; GPU-Arbeit `0`. Das CPU-Gate
bestand. Record-ID:
`cf17bb250cae590a5bf7c21987734599e9d85200b0b5a23a070b92a940f2171a`.

**GPU-Messung und Korrektheit.** Die vier vorregistrierten gepaarten Verhältnisse
waren `0,8478035692961657`, `0,8433268055644042`,
`0,8438689796402571` und `0,8490529640322201`; Median
`0,8458362744682114`. Referenzmedian `1.806.461.854,5 ns`, Kandidatenmedian
`1.528.206.979,0 ns`, daraus berechneter Effekt `-15,416372553178858 %`.
Alle `32` greedy Token waren in Korrektheits- und Messpaaren exakt identisch;
Token-SHA
`666dcfb103d263a12b29ed9a1c1ec496c6922f96c3a6e7cec083eab47fb5127c`.
Beide Pfade führten vier Blöcke aus; der Referenzpfad rief den LM-Head viermal,
der Kandidat einmal auf.

**Ressourcen und Entscheidung.** Peak-Delta `-97.855.968 B`, Swap-Delta `0`,
maximales RSS `3.764.551.680 B`, Prozess-CPU `8.971.375.000 ns`. Der Guard
protokollierte `25,346709 s` GPU-Arbeit, höchstens `2,283342 s` am Stück,
`192,383843 s` Pausen und Duty-Faktor `0,15`. H1 bis H5 bestanden; die
vorregistrierte Entscheidung ist `engineering_go_exact_scope`. Sie ist kein
formaler Claim (`formal_claim=false`) und gilt ausschließlich für den exakten
qualifizierten Request. Alle anderen Requests wählen weiterhin die Baseline.

**Terminale Evidenz und UI.** Die private Datei
`.friday-data/head-skip-runtime.sqlite3` hat Modus `0600`, enthält genau die drei
erwarteten verketteten Records und SHA-256
`6dcf6e4cb942b842dca6e9b0b071df8e7c6cb81ba28fdc5e0fdb05c414d20567`.
Der read-only UI-Test auf `127.0.0.1:8775` lieferte für
`/api/snapshot?limit=3` HTTP `200`, bestätigte die Hashkette und ließ den
Datenbankhash unverändert. Der normale Policy-Aufruf autorisierte danach genau den
qualifizierten Scope mit Grund `runtime_qualification_passed_exact_scope`.

**Fehler bei der Dokumentation.** Ein kombinierter Patch für Status, Nachtlog und
Journal fand den erwarteten Kontext im Nachtlog nicht und wurde atomar ohne
Änderung verworfen. Die Dokumente wurden danach einzeln gegen ihren tatsächlichen
Endstand ergänzt. Dieser reine Patchfehler berührte weder Messprozess noch
Evidenzdatei.

**Gemessen, gerechnet, offen.** Gemessen wurden die gepaarten Laufzeiten,
Ressourcen, Pfadausführung und Tokenidentität. Der Prozentwert wurde aus den
vorregistrierten Laufzeitpaaren berechnet. Nicht gemessen und deshalb offen sind
Multi-Turn-Fortsetzung, mehrere parallele Requests und jede Übertragung auf andere
Prompts oder Einstellungen.

**Abschlussprüfung vor Ergebniscommit.** ProjectAtlas führte nach den
Dokumentänderungen einen einzelnen Aktualisierungslauf aus: `10` geänderte Dateien
wurden geparst, `713` blieben unverändert, kein Timeout. Die vollständige
Projekttestsammlung bestand erneut mit Exit `0`; äußere Laufzeit `51,39 s`, maximal
`193.331.200 B` RSS, Peak-Footprint `46.088.984 B`, keine Swaps. `compileall`, beide
JSON-Parserprüfungen und `git diff --check` bestanden. Ein zusätzlicher read-only
Abgleich bestätigte exakte Übereinstimmung von Runtime-Datenbank,
`results.json` und Experimentmatrix für Laufzeiten, Verhältnis, berechneten Effekt,
Tokenidentität und Record-ID. Der Datenbankhash und die beiden eingefrorenen
Spezifikationshashes blieben unverändert.

### 2026-08-24 — Zyklus 13: persistenter Modellprozess

**Ziel und Abgrenzung.** Nach dem bestätigten Head-Skip wurde genau ein neuer
Kandidat geprüft: Der lokale 4B-Modellprozess bleibt zwischen Anfragen geladen.
Head-Skip, Präfixwiederverwendung, Readback-Änderungen, ein anderes Modell und jede
Produktaktivierung waren ausgeschlossen. Es wurden keine Subagenten verwendet.

**Atlas-first und lokaler Modellbestand.** Vor Quellarbeit wurden der vollständig
gelesene ProjectAtlas-Skill, ein fokussierter MCP-Session-Brief und begrenzte
Atlas-Slices verwendet. Der lokale Cache enthält bereits einen vollständigen
Gemma-3-1B-4bit-Snapshot auf Revision
`2d44e83dc9e80843d22fb941d3d699a0b1351aa6` (`736 MiB` auf Datenträger); daher
wurde trotz der erteilten Erlaubnis nichts heruntergeladen oder installiert. Die
Architekturtexte begrenzen ein lernendes System auf Vorschläge; reale Messgates
bleiben Richter.

**Vorab festgeschriebener Vertrag.** Kandidat und Studie
`persistent-process-20260824-03`, Zyklus `13`, durchgehend
`formal_claim=false`. Die Vorregistrierung wurde vor jeder Hardwaredatei geschrieben
und blieb mit SHA-256
`a9fa83438b7ab30fb85e8cae76a90627b908c469159141286658ac0cc7f6ad9f`
bytegleich. Fester Scope: lokaler 4B-Snapshot Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, vier feste Prompts mit je `897`
Token, `32` greedy Ausgabetoken, Prefill-Chunk `256`, frischer KV-Cache je Anfrage,
zwei A/A-Paare, danach drei Charakterisierungs- und höchstens drei
Validierungspaare. Harness und Vertrag wurden vor Hardware auf Commit
`f9546171aea470385431c64c6318d38ffbe3aeea` gespeichert.

**Fehler vor Hardware und dauerhafte Lösungen.** Die Elternprüfung des ersten
Harnessentwurfs fand, dass ein vollständig korrektes Charakterisierungsergebnis
ohne ausreichenden Zeitgewinn wegen der absichtlich nicht gestarteten Validierung
fälschlich als `correctness_failed` eingestuft worden wäre. Die Pfadprüfung wertet
nun alle tatsächlich abgeschlossenen Phasen aus; ein Tokenmismatch bleibt terminal.
Teilmessungen werden schon beim Start jeder Phase im Ergebniszustand angelegt, und
die Ressourcenrechnung verträgt unvollständige Phasen. Ein früh beendeter warmer
Worker gilt nicht mehr still als sauber gestoppt. Worker-Zeit und Anfragezähler,
Eigentümer und Modus des privaten Startverzeichnisses werden geprüft. Die
Kindumgebung entfernt Python-Pfadinjektion und erzwingt Offlinebetrieb. Die UI baut
Tabellen nur über Textknoten und lehnt fremde Host-Header ab. Neun neue
Fehlerpfadtests brachten die fokussierte Suite von `11` auf `20`; alle bestanden.

Der erste vorbereitende Security-Finalizer-Aufruf adressierte das Werkzeug
irrtümlich unter `skills/security-diff-scan/scripts/` statt unter dem tatsächlichen
Plugin-`scripts/`-Verzeichnis und endete mit Exit `2`, bevor ein Finalizer geladen
wurde. Diese unversiegelte Akte wurde gemäß Einmalregel nicht erneut finalisiert.
Nach den Korrekturen wurde eine neue vollständige Drei-Dateien-Prüfung unter
`/private/tmp/codex-security-scans/Project_Friday/4e202ae_20260824T-security-final.8YP4Rn`
erzeugt und mit dem tatsächlich per `rg --files` aufgelösten Werkzeug genau einmal
versiegelt: Abdeckung `complete`, keine offene Zeile, `0` berichtspflichtige
Befunde. Der zusätzliche Security-Zugang war nicht freigeschaltet; die Prüfung
blieb lokal. Die vollständige Elternprüfung ersetzte die wegen Nutzeranweisung
nicht zulässige Delegation.

**Verifikation vor Hardware.** Worker-Selbsttest `7/7`, Harness-Selbsttest `9/9`,
`compileall`, `20` fokussierte Tests und die vollständige Projektsuite bestanden.
Der erste Vollsuite-Aufruf erreichte die Ausgabegrenze, und der bereits beendete
Prozess ließ keinen Rückgabecode mehr abrufen. Da keinerlei Modell- oder
Hardwarearbeit betroffen war, wurde nur diese Softwaresuite mit gespeicherter
Sitzungskennung wiederholt; sie erreichte `100 %` und Exit `0`. ProjectAtlas wurde
einmal aktualisiert (`5` geänderte Quellen geparst, `723` unverändert, kein
Timeout), Runtime `0.4.5-rc1` und projektlokale Codex-MCP-Konfiguration wurden
bestätigt. `xcodebuild -checkFirstLaunchStatus` bestand. Präregistrierung und
Start-/Ergebnisdateien waren vor dem Lauf unverändert beziehungsweise abwesend.

**Einmalige Messung.** Der Hardwarelauf
`persistent-process-validation-20260824-01` lief am Netzteil genau einmal und
endete mit Exit `0`; kein Retry. A/A-Verhältnisse `0,9224547264` und
`1,0004816149`, Median `0,9614681706`. Charakterisierung:
`[0,3461416900; 0,3431151327; 0,3496472967]`, Median `0,3461416900`.
Validierung: `[0,3442225251; 0,3477940942; 0,3485124631]`, Median
`0,3477940942`. Median aller sechs Paare `0,3469678921`, MAD
`0,0021119878`; kein Wert wurde verworfen. Der Median der sechs gemessenen kalten
TTFT-Werte betrug `5148,7740625` ms, der warmen `1785,1103125` ms.

**Korrektheit, Ressourcen und Entscheidung.** Alle sechs Paare erzeugten exakt
dieselben `32` greedy Token. Kalte PIDs waren jeweils neu; je Phase hatte der warme
Arm genau eine PID und genau einen Modellload. Warmes Peak-RSS
`3.763.077.120 B`, RSS-Wachstum `0 B`, Swap vor/nach
`19.502.071.808 B`, also Delta `0 B`. Budget: `41,586354 s` Modellarbeit,
maximal `3,226913 s` zusammenhängend, `368,707521 s` Pflichtpausen,
`120,015539 s` Kandidatenabkühlung, `576,933889 s` Wall, Duty `0,15`.
Alle Gates bestanden; Entscheidung
`engineering_gain_confirmed_exact_scope`. Der Effekt `−65,30321079 %` ist aus
dem vorregistrierten Median der Paarverhältnisse gerechnet, nicht direkt gemessen.

**Evidenz und UI.** Ergebnis-SHA
`3925d83139cb6278c2b0aa103716e36a33f550f852bd30758976090fa0f7024`;
private Startmarke SHA
`b142b91027c7d261c4753187c82b6ade6ef6aa1d7048c99499e2b8896b4f5536`,
Modus `0600` in Verzeichnis `0700`. Die read-only UI lieferte Entscheidung,
`formal_claim=false`, exakte Ausgabe und acht Verlaufszeilen; ein fremder
Host-Header erhielt HTTP `421`. Der UI-Prozess wurde danach kontrolliert mit
`Ctrl-C` und Exit `0` beendet.

**Gemessen, gerechnet, offen.** Gemessen wurden TTFT, Pfad, Token, RSS, Swap und
Budgetwerte. Das Verhältnis und `−65,3032 %` wurden aus den unveränderten Paaren
gerechnet. Noch nicht implementiert ist der persistente normale Dienstpfad.
Multi-Turn-Fortsetzung und parallele Requests bleiben ungemessen. Das gewünschte
selbstlernende Optimization Memory mit kleinem lokalem 1B-Planner benötigt vor der
Architekturänderung die enge Freigabe aus `PERMISSION_REQUIRED.md`: nur ein
Listenvorschlag je Zyklus, keine Codeausführung, keine Schwellenänderung, kein
Urteil über Korrektheit und keine selbständige Aktivierung.

**Abschlussprüfung nach Ergebnisdokumentation.** Der erneute ProjectAtlas-Refresh
indexierte `948/967` Textkandidaten, parste `7` geänderte Quellen, ließ `721`
unverändert und hatte keinen Timeout. `compileall`, die `20` fokussierten Tests und
die vollständige Projektsuite erreichten erneut Exit `0` und `100 %`.
`xcodebuild -checkFirstLaunchStatus` bestand. Der warnungsfreie Gerätecheck
bestätigte Python `3.12.13`, `arm64`, MLX `0.32.0`, mlx-lm `0.31.3`,
`Device(gpu, 0)` und Apple M1 Max. Beide JSON-Dateien waren parsebar; Ergebnis-,
Startmarken- und Präregistrierungs-SHA blieben unverändert. Der read-only
Matrixabgleich bestätigte Entscheidung, Verhältnis, Effekt, RSS, Swap und Zyklus
`13` exakt gegen die Ergebnisdatei.

## 2026-08-24 — Zyklus 14, 4B-Planertest: Softwarefehler vor Hardware

Beim ersten rein lokalen Testlauf überschrieb die Hilfsfunktion `run` in der neuen
Testklasse unbeabsichtigt die gleichnamige Laufmethode von `unittest.TestCase`.
Dadurch wurden sieben Tests nicht ausgeführt und meldeten einen unerwarteten
Parameter `result`; es wurde weder MLX noch das Modell gestartet. Die Ursache war
allein die Namenskollision. Die Hilfsfunktion heißt nun eindeutig `sample_run`.
Diese Lösung wird vor jedem Hardwarelauf durch die fokussierte und vollständige
Testsuite geprüft. Die versiegelte Vorregistrierung wurde dabei nicht geändert.

Der erste reine Geräte-Anzeigetest fragte danach irrtümlich `mlx.__version__` ab;
dieses Feld stellt das installierte Paket nicht bereit. Der Befehl endete nach den
bereits erfolgreichen Netzteil- und Xcode-Prüfungen mit `AttributeError`, ohne
Modellarbeit. Die dauerhafte Lösung verwendet
`importlib.metadata.version("mlx")`. Der korrigierte Check bestätigte Apple M1
Max, `32 GiB`, `arm64`, MLX `0.32.0`, mlx-lm `0.31.3` und die MLX-GPU. Genau diese
Geräte- und Versionswerte werden nun bereits vor der einmaligen Startmarke geprüft.

`git diff --cached --check` meldete vor dem Vor-Hardware-Commit genau drei
`trailing whitespace`-Zeilen in der versiegelten Vorregistrierung. Dabei handelt
es sich um die schon vor der Versiegelung gesetzten doppelten Leerzeichen für
Markdown-Zeilenumbrüche in den drei Kopfzeilen, nicht um einen Codefehler. Die
Vorregistrierung wird deshalb bytegleich belassen; `diff --check` wird für alle
übrigen neuen Dateien separat ohne Befund ausgeführt und die drei bekannten
Meldungen werden zusätzlich exakt abgeglichen.

### Einmalige 4B-Messung und terminaler Negativentscheid

Der Vor-Hardware-Stand wurde auf Commit
`8067dc6c1fb175f0df539394b2e4dad5894b14b8` gespeichert. Die Vorprüfung bestätigte
sauberen Arbeitsstand, Netzbetrieb, Apple M1 Max, `32 GiB`, `arm64`, MLX `0.32.0`,
mlx-lm `0.31.3`, MLX-GPU sowie den ausschließlich lokal aufgelösten
Gemma-3-4B-Snapshot Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`. Danach wurde
`planner-4b-validation-20260824-01` genau einmal ausgeführt; Exit `1` war der
vorregistrierte negative Entscheid und kein Grund für einen Retry.

Alle drei frischen Prozesse erzeugten exakt dieselben `23` greedy Token, denselben
Text und `stop`; die PIDs waren verschieden und jeder Prozess meldete genau einen
Modellload. Der Rohtext enthielt dreimal die erwartete ID
`persistent_service_qualification`, aber jeweils innerhalb eines Markdown-
Codeblocks. Damit bestanden Tokenidentität, Ressourcen und Budget, während der
strikte Antwortvertrag `0/3` und dadurch auch das erwartete Prioritätsgate nicht
bestand. Terminale Entscheidung: `planner_contract_failed`, durchgehend
`formal_claim=false`. Der Parser wurde nicht gelockert und der Lauf nicht
wiederholt.

Gemessen wurden Rechenzeiten `[1,050531834; 1,041002500; 1,047669750]` s,
Prozesszeiten `[4,913172500; 4,856396167; 4,858676292]` s, maximales RSS
`3.764.961.280 B`, MLX-Peak `3.021.085.374 B` und Swap-Delta `0 B`. Das Budget
meldete `3,139204 s` Modellarbeit, `1,050532 s` längsten Abschnitt,
`48,094525 s` Pflichtpausen, `62,730743 s` Wall und Duty `0,15`. Keine Messung
wurde verworfen. Es wurde kein Geschwindigkeitsgewinn gerechnet, weil dieser
Zyklus keinen Leistungsarm verglich.

Ergebnis-SHA
`64a72331d1a415ae1dac191fecdf9c69cd43f5c11566c2df5ec091cf50a60975`;
Startmarken-SHA
`6e741162f6d02ec69ee74ad7670b8e1a5046a3bc1430b946d7511b1248a6d573`;
Vorregistrierungs-SHA unverändert
`0fa346db7985cdd4dfa49015b395ee0f9d56a097a06f3828b0c161c45e53e5ec`.
Die reale lokale UI lieferte HTTP `200`, drei read-only Verlaufszeilen und bei
fremdem Host HTTP `421`; der Ergebnis-Hash blieb gleich. Auf ausdrücklichen
Nutzerwunsch wurde kein Security-Check ausgeführt. Es gab keinen Download, keine
Installation, keine automatische Kandidatenausführung und keine Aktivierung.

**Abschlussprüfung.** Nach Dokumentation bestanden die `17` fokussierten Tests,
Worker-Selbsttest `11/11`, Harness-Selbsttest `9/9`, `compileall` und die
vollständige Projektsuite mit `100 %` und Exit `0`. Der read-only Matrixabgleich
bestätigte Zyklus `14`, `planner_contract_failed`, RSS, MLX-Peak und Swap exakt
gegen `results.json`. ProjectAtlas-Refresh meldete `0` neu zu indexierende
Dateien; Runtime `0.4.5-rc1` und die lokale Codex-MCP-Konfiguration waren gültig.
`xcodebuild -checkFirstLaunchStatus`, Geräte-, Versions-, GPU-, Netzteil-, JSON-,
Dateimodus-, Hash- und `git diff --check`-Prüfungen bestanden. Der 4B-Lauf wurde
dabei nicht erneut gestartet.

Mehrteilige Dokumentations-Patches trafen wegen inzwischen verschobener
Kontextzeilen nicht zu und wurden vom Patchwerkzeug atomar ohne Teiländerung
abgelehnt. Die Dateien wurden danach mit kleinen, exakt passenden Patches
aktualisiert; `git diff --check` und der JSON-Abgleich prüfen den Endzustand.

## 2026-08-24 — Unabhängiges Dokumentationsaudit von Zyklus 14

Der Zyklus-14-Abschluss wurde nachträglich nur lesend gegen die gespeicherten
Artefakte geprüft. Die getrennte lokale Git-Provenienz ist konsistent: Der
Vor-Hardware-Stand war
`8067dc6c1fb175f0df539394b2e4dad5894b14b8`; der Ergebnis- und
Dokumentationsabschluss ist der separate Commit
`8923467c57d61d3599c430687b949052e397a95c`. Die Präregistrierung blieb
bytegleich, SHA-256
`0fa346db7985cdd4dfa49015b395ee0f9d56a097a06f3828b0c161c45e53e5ec`; die
Ergebnisdatei blieb bei
`64a72331d1a415ae1dac191fecdf9c69cd43f5c11566c2df5ec091cf50a60975`; die
private Startmarke blieb bei
`6e741162f6d02ec69ee74ad7670b8e1a5046a3bc1430b946d7511b1248a6d573`.

Die drei gespeicherten Läufe belegen verschiedene PIDs, je einen Load, denselben
4B-Snapshot, `3/3` identische Tokenfolgen und identischen Rohtext. Alle Antworten
enthielten die richtige ID `persistent_service_qualification`, aber jeweils einen
unerlaubten Markdown-Codeblock. Der Parser wurde nicht gelockert; die
unveränderte Entscheidung ist `planner_contract_failed`, durchgehend
`formal_claim=false`.

Unabhängig wiederholt wurden nur nicht-hardwarebezogene Prüfungen: 17 fokussierte
Offline-Tests mit Exit `0`, Worker-Selbsttest `11/11`, Harness-Selbsttest `9/9`,
AST-/JSON-Parsing, `git diff --check` mit Exit `0` und
`xcodebuild -checkFirstLaunchStatus` mit Exit `0`. Die lokale read-only UI lieferte
GET `200`, Snapshot-GET `200` und für einen fremden Host `421`; die Ergebnis-SHA
blieb vor und nach dem UI-Test identisch. HEAD wurde nicht als erfolgreich
behauptet und antwortete mit `501`.

Offene Punkte bleiben unverändert: Das alte `results.json` enthält keinen
Gewichts-SHA. Der lokale Gewichts-SHA
`94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3` wurde nur
separat verifiziert und nicht rückwirkend in die Evidenz geschrieben. Die drei
absichtlichen Markdown-Trailing-Spaces in der versiegelten Präregistrierung
werden wegen des eingefrorenen Hashes nicht geändert. Dieser Auditlauf startete
weder Hardware noch Modell erneut und führte keinen Security-Check aus.

## 2026-08-24 — Zyklus 15: Vor-Hardware-Stand der engen Zwei-Modell-Studie

Die Nutzerfreigabe wurde für Zyklus 15 ausdrücklich und eng auf die Studie
`dual-model-evidence-planner-20260824-01` begrenzt. Gegenstand ist ausschließlich
ein fester Planungsfall mit den zwei bereits lokal vorhandenen Modellen; es gibt
keine allgemeine Planner-Freigabe, keine automatische Aktivierung und keine
Ausweitung früherer Studien. Eine Matmul-On/Off-Integration ist nicht vorhanden,
und ein Matmul-On/Off-Vergleich wurde nicht durchgeführt oder erfunden. Der
formale Claim bleibt `formal_claim=false`.

### Atlas-first und Arbeitsgrenze

Vor der Dokumentationsarbeit wurde der versionierte ProjectAtlas-Skill vollständig
gelesen. `atlas_runtime_info` bestätigte ProjectAtlas `0.4.5-rc1` mit MCP,
SQLite, TOON, Symbolindex, Textsuche und Watcher. Der initiale fokussierte
Session-Brief meldete `refresh_required` wegen der überschrittenen
Dependency-Closure-Grenze. Beim anschließenden Refresh trat einmalig ein
SQLite-Lock auf; der Retry war erfolgreich und der Index danach verfügbar.
ProjectAtlas und das eingebundene Upstream-Repository wurden nicht geändert.

Diese Teilaufgabe blieb dokumentarisch: Es wurden weder Hardware noch ein Modell
gestartet, keine Startmarke oder Ergebnisdatei erzeugt, kein Test ausgeführt,
nichts installiert oder heruntergeladen und kein Commit erstellt. Geändert wurden
ausschließlich `PROJECT_STATUS.md`, `IMPLEMENTIERUNGSPLAN.md`,
`PERMISSION_REQUIRED.md` und dieses append-only Journal.

### Eingefrorener Studienvertrag

Die gebundenen lokalen Snapshots sind:

- `1b`: `mlx-community/gemma-3-1b-it-4bit`, Revision
  `2d44e83dc9e80843d22fb941d3d699a0b1351aa6`;
- `4b`: `mlx-community/gemma-3-4b-it-4bit`, Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`.

Der feste Zeitplan umfasst sechs Paare und zwölf frische serielle Prozesse. Die
Paare `1–3` laufen `1b → 4b`, die Paare `4–6` `4b → 1b`; jedes Modell wird genau
sechsmal geladen. Der einzige akzeptierte Planerwert ist
`persistent_service_qualification`. Die Studie misst weder allgemeine
Modellqualität noch Lernen, Training, Code, Gewichtsänderung oder Aktivierung.

Der Vor-Hardware-Code ist durch folgende vollständige SHA-256-Werte gebunden:

- Präregistrierung:
  `77d46d63a46065f863e3aa425d74fb2ed6dc756c54a674c8767d58c4c24f59f1`;
- Worker:
  `b1db90d306d5de5c6ff466d046c5c617c5dd42cdaee3f6f7b4bcd5bf2a024bc0`;
- Harness:
  `eea8f0e435456711e2c857a92dcfeb398f0ab574f9cdedb9dbe3a78e6919cd75`;
- read-only UI:
  `c4f132b24fdb95cd97e62763141b1e57620010ab3532081169f14813d5915735`.

Der letzte gemeldete fokussierte Offline-Stand bestand aus `43` Tests und `31`
Subtests mit Exit `0`. Das Ergebnis steht unter dem Vorbehalt der abschließenden
unabhängigen Checks und wurde in dieser Dokumentationsaufgabe nicht erneut
ausgeführt. Der unabhängige Zyklus-14-Dokumentationsaudit ist auf Commit
`ee12bb5` verankert.

### Vor-Hardware-Evidenz und Fehlerkorrekturen

Es existieren noch keine Hardwareevidenz, keine private Startmarke und keine
`results.json`; folglich gibt es keine Messwerte, keine Studienentscheidung und
keinen Hardware- oder Modellclaim. Die vor Hardware gefundenen Fehler und ihre
dauerhaften Lösungen sind:

- Doppelte Paare/Run-Positionen werden mit expliziten Paar- und Schedule-IDs sowie
  Duplicate-Rejection verhindert.
- Unvollständige Erfolgszustände werden nicht aggregiert; partielle Rohereignisse
  bleiben erhalten und der Pfad entscheidet fail-safe.
- Per-Run-Content-Hashing wurde aus den gemessenen Zeitintervallen entfernt. Der
  Parent erstellt das Ausführungsmanifest einmal vor und einmal nach dem Plan;
  diese Hashes liegen außerhalb der Child-/GPU-Messung.
- Snapshot-Revision, Snapshot-/Gewichts-Hashes und der gerenderte Prompt werden
  vom Parent an jeden Worker gebunden und vor/nach dem Modellload geprüft.
- Partial-/Fehlerpfade bleiben terminal und können keinen erfolgreichen Claim
  erzeugen.
- Der Parent erzwingt eine stdout-Obergrenze von `1.000.000` Byte.
- Die read-only UI validiert die geschlossene Model-/Kandidaten-Whitelist und
  nimmt keine Schreiboperationen an.
- Ressourcen- und Budgetprüfungen haben eine feste Reihenfolge vor erfolgreicher
  Aggregation; Ressourcen-/Budgetabbruch hat Vorrang vor Korrektheits- und
  Vertragsauswertung.

Der Zyklus-15-Stand ist damit ein reproduzierbar gebundener Vor-Hardware-Vertrag,
nicht dessen Ergebnis. Ein späterer Studienlauf darf nur innerhalb dieser engen
Freigabe und ohne Matmul-On/Off-Nebenclaim erfolgen.

### 2026-08-24 — Zyklus 15: finaler Artefakt- und Härtungsnachtrag vor Hardware

Nach den letzten Offline-Korrekturen bleiben Präregistrierung und Worker bytegleich:

- Präregistrierung:
  `77d46d63a46065f863e3aa425d74fb2ed6dc756c54a674c8767d58c4c24f59f1`;
- Worker:
  `b1db90d306d5de5c6ff466d046c5c617c5dd42cdaee3f6f7b4bcd5bf2a024bc0`.

Die finalen SHA-256-Werte der geänderten Artefakte sind:

- Harness:
  `5ce1686e0782825e765371301e7099f26e4e135cbf04dc5f74ef537f5cfde131`;
- read-only UI:
  `5db9bf832c17470c0899ee0fd4062b42d524904e1ee3224894e87a7bed049607`.

Die letzten geschlossenen Fehlerpfade ergänzen den bestehenden Vertrag, ohne die
Freigabe zu erweitern:

- Ein vom Parent bereits validiertes Worker-Event wird in der partiellen Evidenz
  behalten, wenn erst die nachfolgende Ressourcenprüfung terminal abbricht. Der
  Abbruch bleibt fail-closed; das Event wird weder verworfen noch als Erfolg
  umgedeutet.
- Die read-only UI prüft zusätzlich die feste Studien-Run-ID und eine geschlossene
  Decision-Allowlist.
- Ein minimaler Fehlerreport ohne `metrics` ist ein kontrollierter Fehlerzustand
  und kann nicht als erfolgreicher vollständiger Report erscheinen.

Die dokumentierte fokussierte Testzahl bleibt bis zur unabhängigen Meldung des
Test-Luna unverändert bei `43` Tests plus `31` Subtests, Exit `0`; in dieser
Dokumentationsaufgabe wurde kein Test ausgeführt. Es wurden weder Hardware noch
ein Modell gestartet, keine Startmarke oder `results.json` erzeugt und kein Commit
erstellt. `formal_claim=false` und die Abgrenzung ohne Matmul-On/Off-Integration
bleiben unverändert.

Der anschließende `git diff --check` für die vier Dokumentationsdateien endete
ohne Befund. Die finale Testzahl bleibt davon unberührt und wartet weiter auf die
unabhängige Test-Luna-Meldung.

**Finale unabhängige Test-Luna-Meldung.** Der fokussierte Endstand besteht aus
`46` Tests und `42` Subtests mit Exit `0`; `py_compile` endete ebenfalls mit
Exit `0`. Dieser Wert ersetzt als aktueller Verifikationsstand die vorstehend
append-only erhaltene vorläufige Meldung `43` Tests plus `31` Subtests. Die
Dokumentationsaufgabe selbst führte die Tests nicht erneut aus. Hardware,
Modellprozesse, Startmarke, `results.json`, `formal_claim=false` und Commitstatus
blieben unverändert.

### 2026-08-24 — Zyklus 15: vollständiger finaler Preflight vor Hardware

Die unabhängige Vor-Hardware-Prüfung wurde vollständig gemeldet und in diesem
Dokumentationslauf nicht erneut ausgeführt. Alle Prüfungen blieben offline oder
read-only:

- Worker-Selbsttest: `17/17`, Exit `0`.
- Harness-Selbsttest: `25/25`, Exit `0`.
- Defaultaufruf ohne Ausführungsfreigabe: Exit `78`; danach waren weder
  `.friday-data/dual-model-planner/attempt.json` noch
  `experiments/dual_model_planner/results.json` vorhanden.
- `py_compile` und `compileall`: jeweils Exit `0`.
- Fokussierte Suite: `46` Tests und `42` Subtests, Exit `0`, Wall `3,36 s`,
  Peak-RSS `60.801.024 B`.
- Vollständige `pytest`-Suite: Exit `0`, Wall `45,43 s`, Peak-RSS
  `200.523.776 B`.
- `git diff --check`: Exit `0`.
- AST-Parsing aller relevanten Python-Dateien: Exit `0`.
- `xcodebuild -checkFirstLaunchStatus`: Exit `0`.
- ProjectAtlas-Runtimeprüfung und projektlokale Konfigurationsprüfung: jeweils
  Exit `0`; Runtime `0.4.5-rc1`.

Die read-only Umgebungsintrospektion bestätigte MLX `0.32.0`, mlx-lm `0.31.3`
und das Defaultgerät `Device(gpu, 0)`. Diese Anzeige führte keine GPU-Rechnung
aus. Der lokale Resolver bestätigte ohne Modellload:

- `mlx-community/gemma-3-1b-it-4bit`, Revision
  `2d44e83dc9e80843d22fb941d3d699a0b1351aa6`, Gewichtsumfang
  `732.577.304 B`;
- `mlx-community/gemma-3-4b-it-4bit`, Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, Gewichtsumfang
  `3.400.569.562 B`.

Die Vorregistrierung blieb bytegleich bei SHA-256
`77d46d63a46065f863e3aa425d74fb2ed6dc756c54a674c8767d58c4c24f59f1`.
Nach dem gesamten Preflight waren Resultat und private Startmarke weiterhin
abwesend. Ignorierte `__pycache__`-Verzeichnisse sind vorhanden; sie sind kein
Studienartefakt und werden nicht Bestandteil des Commits. Es gab keine
Hardwarearbeit, keine GPU-Rechnung, keinen Modellload und keinen Commit.

### 2026-08-24 — Zyklus 15: semantikneutraler Formatfix vor Hardware

Ein staged Diff-Check endete nach dem Preflight mit Exit `2`, weil die
Präregistrierung genau drei Trailing-Spaces enthielt. Noch vor jeder Hardware-
oder Modellausführung wurden ausschließlich diese drei Formatzeichen entfernt.
Inhalt, Modelle, Revisionen, Prompt, Schedule, Grenzwerte, Gates,
Entscheidungstabelle und `formal_claim=false` blieben unverändert; es gab keine
Vertragsänderung.

Die dadurch aktuellen vollständigen SHA-256-Werte sind:

- Präregistrierung:
  `246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`;
- Harness:
  `59691f50a1f33d4930b36ccce24ec701af74ebd0f9f095912a75e15a28978470`.

Worker
`b1db90d306d5de5c6ff466d046c5c617c5dd42cdaee3f6f7b4bcd5bf2a024bc0`
und read-only UI
`5db9bf832c17470c0899ee0fd4062b42d524904e1ee3224894e87a7bed049607`
blieben bytegleich. Die älteren Hashangaben bleiben als append-only Zwischenstand
erhalten und sind durch diesen Nachtrag ausdrücklich ersetzt. Es wurden keine
Tests, keine Hardware, keine GPU-Rechnung und kein Modell gestartet; Startmarke
und `results.json` blieben abwesend, und es wurde kein Commit erstellt.

## 2026-08-24 — Zyklus 15: reales Zwei-Modell-Ergebnis

Die zuvor dokumentierte Vor-Hardware-Freigabe wurde genau einmal ausgeführt. Die
Studie `dual-model-evidence-planner-20260824-01` blieb auf den zwei bereits
lokalen Snapshots begrenzt und lief mit sechs balancierten Paaren in zwölf frischen
seriellen Python-Prozessen. Paare `1–3` liefen `1b → 4b`, Paare `4–6` `4b → 1b`;
jedes Modell wurde genau sechsmal geladen, niemals gleichzeitig. Es gab keine
Wiederholung, keinen Download, keine Installation und keinen Push.

Der feste Vertrag erlaubte ausschließlich
`{"candidate_id":"persistent_service_qualification"}`. Die Entscheidung ist
`no_planner_qualified`, `formal_claim=false`. Beide Modelle waren innerhalb des
jeweiligen Modells in `6/6` Läufen deterministisch, aber Vertrag, strikter Parser
und erkannte `candidate_id` waren jeweils `0/6`. Die direkte dekodierte
Textgleichheit zwischen den Modellen lag bei `0/6`.

Die 1B-Antwort enthielt Markdown, den falschen Schlüssel
`persistent_service_id` und `<end_of_turn>`-Trailer. Die 4B-Antwort enthielt die
richtige ID, aber einen unerlaubten Markdown-Codeblock. Diese Angaben beschreiben
nur den Maschinenvertrag; sie sind keine qualitative Modellbewertung.

### Gemessene Hardwarewerte

| Messwert | 1B | 4B |
| --- | ---: | ---: |
| Ausgabe / Abschlussgrund | `32 Token / length` | `23 Token / stop` |
| TTFT Median / MAD | `0,295451312 / 0,0005528535 s` | `0,796846125 / 0,0088023125 s` |
| Modellarbeit Median / MAD | `0,4608839165 / 0,0005743330 s` | `1,0487644165 / 0,0092854165 s` |
| Prozess-Walltime Median / MAD | `4,2468557705 / 0,0059329165 s` | `4,883630417 / 0,0182606455 s` |
| Peak-RSS | `1.937.965.056 B` | `3.765.420.032 B` |
| MLX-Peak | `1.012.548.526 B` | `3.021.085.374 B` |
| Swap-Delta | `0 B` | `0 B` |

Alle Ressourcen-, Snapshot-, Pairing- und Budgetgates bestanden. Gemessen wurden
`9,205052 s` Gesamt-Modellarbeit, maximal `1,151402 s` zusammenhängend und
`178,475444 s` Walltime bei Duty-Faktor `0,15`; es gab keine Abbrüche.

### Berechnung und Evidenz

Die Paarquotienten und Bootstrap-95-%-KIs sind aus den Rohdaten berechnet, nicht
zusätzliche Messungen (`10.000` Resamples): TTFT `0,373014193`
`[0,365603946; 0,377539933]`, Modellarbeit `0,439069434`
`[0,434598134; 0,444460794]`, Prozess-Walltime `0,872042394`
`[0,864987297; 0,939562889]`, Tokenrate `3,168801108`
`[3,130352029; 3,201472197]`. Daraus folgen berechnet ungefähr `12,8 %`
kürzere 1B-Walltime und `48,5 %` geringerer 1B-Peak-RSS. Wegen der beiden
fehlenden Funktionsgates entsteht keine Präferenz.

Die Rohdatei
`experiments/dual_model_planner/results.json` hat SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`; die
private Startmarke hat SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`; die
Präregistrierung hat SHA-256
`246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`.
Die UI blieb read-only: GET/HEAD `200`, schreibende Methoden `405`, fremde Hosts
`421`, alle Evidenzhashes vor und nachher identisch. Zyklus 15 hat JSON-Rohdaten,
keine eigene SQLite-Evidence-DB.

Die Freigabe für diesen einzelnen Lauf ist verbraucht. Multi-Turn-Fortsetzung und
mehrere parallele Requests bleiben offen. Allgemeine Modellqualität, allgemeine
Planner-Fähigkeit, selbstlernende Runtime und Produktaktivierung sind nicht
belegt. Ein vollständiger Gemma-Matmul-A/B-Pfad mit „mit/ohne Matmul“-Schalter
existiert nicht; er wurde nicht gemessen und bleibt ein separater künftig
vorregistrierungspflichtiger Kandidat.

## 2026-08-24 — Zyklus 15: Postflight-Korrektur und Verifikationsnachtrag

Die früheren `46`-Angaben im bereits geschriebenen Zyklus-15-Preflight bleiben
als historische Zwischenangabe unverändert. Der tatsächliche aktuelle
Postflight ersetzt diese Angabe ausdrücklich durch `47` fokussierte Tests bei
`42` Subtests, Exit `0`. Die vollständige Suite sammelte `744` Tests und endete
mit Exit `0`.

Im Postflight endeten `compileall`, die strikte JSON-Prüfung von
`results.json`, `verification.json` und `EXPERIMENT_MATRIX.json`, `json.tool`,
AST-Prüfung, `git diff --check` und `xcodebuild -checkFirstLaunchStatus` jeweils
mit Exit `0`. ProjectAtlas meldete zunächst `refresh_required`; genau ein
inkrementeller Refresh war danach erfolgreich. Runtime `0.4.5-rc1` und die
projektlokale MCP-Konfiguration waren gültig. MLX `0.32.0`, mlx-lm `0.31.3` und
`Device(gpu, 0)` wurden nur read-only geprüft; es gab keine Modell- oder
GPU-Arbeit.

Die Evidenz blieb nach dem Postflight bytegleich: Ergebnis-SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`,
Verifikations-SHA-256
`24696c679de567519e8f2b3b034f0833de8122569072b71feeae794c05bbf4e6`,
Marker-SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327` und alle
DB-Hashes unverändert. Die Verifikation meldete leere Abweichungen, die
Entscheidung `no_planner_qualified` und `formal_claim=false`.

ProjectAtlas hatte keine getrackten Änderungen. Bestehende untracked Fixture-
`.gradle`-Verzeichnisse wurden nicht angefasst und gehören nicht zu diesem
Nachtrag.

## 2026-08-24 — Zyklus 16: Vor-Hardware-Status des runtime-only Matmul-A/B-Tests

Der Nutzer erteilte am 24.08.2026 genau eine neue Freigabe für die Studie
`matmul-compile-ab-20260824-01` mit dem Kandidaten
`fixed_cache_compiled_decode_v1`. Die Studie ist auf die Laufzeitumgebung
begrenzt: Modell, Gewichte und Quantisierung bleiben unverändert.

Die mathematische Matmul bleibt in allen Armen aktiv. Verglichen werden
`standard_eager`, `fixed_eager` und `fixed_compiled`; „Matmul-A/B“ bedeutet hier
nicht, dass Matmul ausgeschaltet wird. Exakte greedy Token- und Textidentität
ist ein Pflicht-Gate. Die alten Device-Model-Compile-Messungen sind wegen
falscher Token ab Position 2 ungültig und werden nicht als Baseline verwendet.

Die Präregistrierung ist im Arbeitsbaum vorhanden, aber noch nicht versiegelt
(kein lokaler Seal-Commit) und noch nicht gemessen. Es gibt deshalb keine neuen
Hardwarewerte, keine Ergebnisdatei und keinen Performanceclaim.
`formal_claim=false`. Ein negatives Ergebnis ist gültig; automatische
Kandidatenausführung oder Produktaktivierung ist nicht freigegeben.

## 2026-08-24 — Zyklus 16 versiegelt, noch vor Hardware

Die finale Präregistrierung der lokalen runtime-only Studie
`matmul-compile-ab-20260824-01` wurde technisch geprüft und im lokalen Seal-Commit
eingefroren. Status: `sealed_pending_hardware`; SHA-256
`b2487240f926fa95d9b9933c28c57bc616886efb82ee0503a524d4f24f1da6bf`.
Es gibt noch keine Hardwaremessung, keine `results.json` und keine private
Startmarke. `formal_claim=false`. Modellgewichte, Quantisierung und mathematische
Matmul bleiben unverändert; die drei Arme sind `standard_eager`, `fixed_eager`
und `fixed_compiled`.

Vor-Hardware-Review: Lazy MLX-Materialisierung wurde bis `mx.eval`/Synchronisierung
in die Kandidatenfehlerklassifikation einbezogen; der Parent-Timeout folgt nun der
verbleibenden harten Gesamt-Walltime; beobachtete Armzeit, akzeptierte Buchung und
theoretische Duty-Pause werden getrennt belegt. Damit werden Budgetablehnungen
nicht als erfolgreiche Buchung dargestellt. Arm-Längen, Abschlussgrund und
Fehler-Teilereignisse werden zusätzlich streng geprüft.

Verifikation: fokussierte Tests 34 passed/Exit 0, vollständige Suite/Exit 0,
compileall/Exit 0, Worker-Selfcheck 21/Exit 0, Harness-Selfcheck 18/Exit 0,
UI-Selfcheck/Exit 0, Standardaufruf/Exit 78 ohne Marker oder Ergebnisse,
`git diff --check`/Exit 0 und `xcodebuild -checkFirstLaunchStatus`/Exit 0.
ProjectAtlas 0.4.5-rc1 mit MCP, M1 Max/32 GiB/AC, `Device(gpu,0)`, MLX 0.32.0,
mlx-lm 0.31.3, Snapshot-SHA `e6edcd46...eda` und Gewicht-SHA
`94d3d701...74af3` wurden verifiziert. Keine Ergebniswerte wurden erzeugt.

## 2026-08-24 — Zyklus 16: finaler Review-Nachtrag vor Hardware

Nach dem vorherigen Eintrag wurden die letzten evidenzrelevanten Restursachen
behoben. Die Lazy-Konvertierung von Standard- zu Fixed-Cache wird jetzt über
`slice_update`, sämtliche `mx.eval`-Aufrufe und Synchronisierung hinweg korrekt
als `candidate_not_runnable` oder Ressourcenfehler klassifiziert. Worker,
Outputreader, Join und Abbruch teilen eine monotone `worker_deadline`; 15 Sekunden
der Gesamtfrist bleiben als Finalisierungsreserve frei.

Die Guard-Buchung wird vor und nach `record_gpu` erfasst. Dadurch bleiben auch
Budgetablehnungen als echte Teil-Evidenz erhalten, ohne eine nicht akzeptierte
Charge als Erfolg auszugeben. Für die Rolling-Duty-Regel sind nach jedem
akzeptierten Arm mindestens 13 Blöcke à 4 Sekunden vorregistriert. Das Budgetgate
meldet einen terminalen `resource_or_budget_failed`-Status nicht mehr als
bestanden, auch wenn die Rohzusammenfassung formal gültig wirkt.

Finale Offline-Verifikation: fokussierte Tests `39 passed, 55 subtests`, Exit 0;
P0/P1-Befunde `0`; weiterhin keine `results.json`, keine Startmarke und keine
Hardwaremessung. `formal_claim=false`. Der aktuelle Zyklus-16-Status bleibt
`sealed_pending_hardware`; finaler Präregistrierungs-SHA-256:
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.

## 2026-08-24 — Zyklus 16: reales Ergebnis und Abschluss

Die einmalige lokale Studie `matmul-compile-ab-20260824-01` wurde im
Seal-Commit `83ee3ea03f9fb303b8226ab8ad3189f07daec727` ausgeführt und mit
`runtime_compile_wins_exact_scope` entschieden; `formal_claim=false`. Evidence-
Commit `cc6d2ea012a0cd6a858acc9a66d4754e95c421b7`, Result-Hash
`fbcc2fc65ac5d255ed11039a74c34e9a02d942cec17b25a6ed863058e0073b57`,
Verification-Hash `09b1b53841a59bad3c4b1b9a0ef62fb659668b472358c10fa9188cad158f0038`,
Marker-Hash `8adf6f9c2453524bd1e05f4973ee85f84a323e9461a3f9b996ec2d0f7fed3c2f`,
Präregistrierungs-Hash
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.

Sechs frische Prozesse mit drei Armen ergaben 18 abgeschlossene Arm-Ausführungen
(3 × 6). Token und Text waren in allen 18 Arm-Ausführungen exakt gleich.
Gemessene Decode-Medianen/TTFT waren
Standard `0,399939187 s`/`0,638376521 s`, Fixed-Eager
`0,3999597295 s`/`0,638425813 s` und Fixed-Compiled
`0,371848789 s`/`0,6385446665 s`. Die gemessenen gepaarten Ratios waren
`0,9295921887` gegen Standard mit KI `[0,9128789083; 0,9348209684]` und
`0,9296309524` gegen Fixed-Eager mit KI `[0,9256302629; 0,9327708433]`.
Peak-RSS `3.771.564.032 B`, MLX-Peak `3.476.049.782 B`, Swap-Delta `0 B`.

Berechnet, nicht separat gemessen, wurden warme Gesamtprojektion
`0,9829777045`, kalte One-off-Projektion `1,0154895491` und Break-even median
rund 36,47 Decode-Schritte; der Lauf selbst maß 31 Schritte. Matmul blieb in
allen Armen aktiv; Modell, Gewichte und Quantisierung blieben unverändert.
Die Studie belegt keinen allgemeinen Qualitäts-, Selbstlern- oder
Produktivclaim und aktiviert nichts automatisch. Die Freigabe ist genau einmal
verbraucht; es gibt keinen zweiten Lauf.

Der Lifecycle-Selfcheck-Bug wurde erst nach der Messung erkannt und getrennt
behoben. Die read-only UI lieferte GET/HEAD `200`, Schreibmethoden `405`, fremden
Host `421`; die geprüften Hashes blieben unverändert.

## 2026-08-24 — Zyklus 16: Post-Hardware-Verifikation und Lifecycle-Nachtrag

Die Nachprüfung blieb read-only; ein neuer Hardware- oder Modelllauf wurde nicht
ausgeführt. Die vollständige Pytest-Suite bestand mit `787 Tests in 71 Dateien`,
Exit 0. Der fokussierte Test `test_matmul_compile_ab` bestand mit `43 passed,
60 subtests`, Exit 0. Compileall Exit 0, Worker-Selfcheck 21/0,
Harness-Selfcheck 18/0, Dashboard-Selfcheck 0, `xcodebuild` Exit 0, `jq` Exit 0
und `git diff --check` Exit 0. Der Default-Aufruf endete erwartungsgemäß mit
Exit 78 ohne Mutation. Der Harness-`--show`-Aufruf lief einmal mit Exit 0,
stderr blieb leer und er lieferte genau eine gültige JSON-Zeile.

Die reale lokale UI lieferte GET/HEAD `200`, Schreibmethoden `405`, fremden Host
`421` und `no-store`. Sie enthielt keinen unbereinigten Modelltext. Cycle-16-
und Cycle-15-Evidence sowie 12 SQLite-Datenbanken waren vor und nach der UI-
und Auditprüfung identisch; die private Startmarke blieb auf `0600`. Die
geprüften Ergebnis-, Evidence- und Datenbank-Hashes blieben unverändert.

ProjectAtlas wurde genau einmal inkrementell mit `watch_once` aktualisiert:
ein Zyklus, 967 indexierte Textkandidaten, 11 geparste und 732 unveränderte
Symbole. Runtime `0.4.5-rc1` und die projektlokale MCP-Konfiguration waren
gültig. Getrackte ProjectAtlas-Dateien blieben unangetastet; das vorbestehende
verschachtelte `.gradle`-Untracked wurde ebenfalls nicht verändert.

Der Lifecycle-Bug entstand, weil der Selfcheck Evidence fälschlich immer als
fehlend erwartete. Die Korrektur prüft fehlende und vorhandene Evidence
read-only, akzeptiert keine Symlinks, verlangt eine reguläre Markerdatei mit
Modus `0600` und vergleicht Hashes und Modi vor und nach dem Test. Der aktuelle
Arbeitsbaum-Harness ist damit nicht bytegleich zum versiegelten Code; die
Evidence bewahrt aber die Code-Fingerprints des Seal-Stands. `formal_claim=false`,
keine automatische Aktivierung und keine neue Freigabe.

## 2026-08-24 — Zyklus 17: Pre-Hardware-Draft reserviert

Die Nutzerantwort „Dann machen wir das mal“ reserviert genau einen neuen Lauf,
ohne die Freigabe bereits zu verbrauchen. Studie
`fixed-compiled-batched-readback-20260824-01`, Kandidat
`fixed_compiled_batched_readback_n8_v1`, Status `draft_pending_preflight`.
Geplant sind sechs gepaarte frische Prozesse und zwölf Arm-Ausführungen. Die
einzige Variable ist Readback `1` versus `8` auf identischem Fixed-Compiled-4B;
Modell, Gewichte, Quantisierung und Matmul bleiben unverändert. EOS-Tail wird
vollständig getaktet und getrimmt, exakte logische Token-/Textidentität ist
terminales Gate. Noch kein Marker, Resultat oder Modelllauf. Cycle 7 `12,98 %`
bleibt explorativ. `formal_claim=false`; keine Aktivierung, kein Dienst, kein
Multi-Turn- und kein Qualitätsclaim.

## 2026-08-25 — Zyklus 17 sealed_pre_hardware

Offline-Preflight abgeschlossen: `measured=false`, `formal_claim=false`,
`authorization=reserved_not_consumed`; kein Modell-/MLX-/GPU-/Hardwarelauf,
kein Marker und kein Resultat. Readback 1 versus 8 bleibt die einzige Variable
auf identischem Fixed-Compiled-4B-Pfad; sechs frische Paare und zwölf Arme sind
geplant. Prereg `74f63c36ddd141c4b4666d9f15d7b17d3ac9294e2d63cb29f6d9e35a80db21b1`,
Worker `fecf712b44e6d1a8c46565dda59569fa11cdc762fc49917307874435e4a2efde`,
Harness `9c0689be97a1ee5022f7c4b4623af9bd4a9906411291d9c4295b4c16184c7ff0`,
Dashboard `ccbaf05368f21acfa2c627a33f3ee9c5629d335d45f76abb5a74d1399fbeaaee`.
Selfchecks und Offline-Tests (30/30, 817/817) bestanden; kein Produktivdienst
und keine Aktivierung.

## 2026-08-25 — Zyklus 17 Ergebnis

Der einmalige Hardwarelauf ist abgeschlossen: `measured=true`, Freigabe
`consumed_exactly_once`, Entscheidung `no_clear_speedup_baseline_retained`,
`formal_claim=false`. Readback 8 war in allen sechs Paaren schneller, aber der
Ratio-Median `0,9581074518` verfehlte die feste 5-%-Schwelle; der 4,1893-%-Effekt
ist berechnet. Baseline retained. 6 Paare/12 Arme, exakte logische und sichtbare
Identität, kein Qualitäts-, Modell-, Gewichts-, Quantisierungs- oder Matmulclaim.
Independent Evidence-Audit: `evidence_valid=true`; Resultat, Verification und
Marker blieben hash-stabil. Der negative Befund ist gültig.

## 2026-08-25 — Dokumentations-Konsistenzkorrektur

Aktive Zusammenfassungen wurden nach dem Ergebnislauf von Draft-/Preflight-
Status auf `measured=true`, `consumed_exactly_once` und
`no_clear_speedup_baseline_retained` korrigiert. Frühere Draft-/Seal-Stände
bleiben ausdrücklich historisch markiert. Die Matrix-Grenzen wurden auf
Zyklus 17/16 abgeschlossene Zyklen aktualisiert; die Readback-Zeile trägt den
berechneten 4,1893-%-Effekt und die verfehlte feste 5-%-Schwelle.

## 2026-08-25 — Matrix-Zählerkorrektur

Die Matrix weist nun korrekt `max_cycles=17` und `cycles_completed=17` aus;
dies ersetzt den unmittelbar vorherigen Zwischenstand `17/16`.

## 2026-08-25 — Zyklen 18–21: Fused-Greedy-Compile-Chronologie

Zyklus 18 war fail-closed und ohne Modelllauf (`load_count=0`,
`formal_claim=false`): Resultat `ea644a912c9bb20a9fc992d7e24bfecfbb70285f2788ee83a15aeb4937503035`,
Seal `f1383587d585620f75e3c1e9bd40a71cbd0e8af9`, Evidence
`dc2cdced58b629e6a39cb8ed870d847d8ee16c13`. Parent und Worker hatten
verschiedene Environment-Fingerprints; die Provenienzprüfung schlug vor Paar 0
fehl.

Zyklus 19 war ebenfalls ohne Modelllauf: Resultat
`4e02221975f6f1710e96dc70f69b4df6f48a1d93df859c6274ed83460dee0320`, Seal
`7278bda3281161cebdcf395fd4aa50df5de5124e`, Evidence
`59bbe9d698d978dcbd621fe89fb17bf98b286b8a`. Der Worker wertete sein eigenes,
absichtlich erzeugtes Resultat als unerlaubten Git-Dirty-State.

Zyklus 20 war ebenfalls ohne Modelllauf: Resultat
`72e7e0692136766bcd5cea4147f3c106ad64de8ddadba855767d8908ae53200d`, Seal
`6c0e18b17b2febf184fda0fc09552b5613c49dc0`, Evidence
`78f983c71636637b7995eb90500fe689cbe53fee`. Das Parent-Snapshot-Manifest
enthielt `dev`, das Worker-Manifest nicht; Snapshot-Binding scheiterte vor dem
Laden. Diese drei Ergebnisse enthalten keine Performancewerte.

Die Korrekturen sind identische Environment-Fingerprints, eine fail-closed
Allowlist nur für das eigene Resultat und ein identischer Snapshot-Statvertrag
mit `dev`. Kein Fehler wurde als Performancebefund umgedeutet; Matmul blieb
aktiv.

Zyklus 21 wurde mit Seal `ad4c92f32e608a8a0870b37e23a4dba0da1f666c` und Evidence
`4f89e51c3933aa9c9d42563393589da3c2e4a875` abgeschlossen. Prereg-SHA
`a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf`, Resultat-
SHA `55bad770baad66cbebb804288845e9cf2785c0969c77355731ab8a23b3a43a2e`,
Marker-SHA `1c1dc10670c153c4c7430f3320671c08a3d56114e0fc5ee6af988c750ceb14e4`.
Sechs frische serielle 4B-Prozesse mit zwölf Armen liefen vollständig durch;
Token und Text waren exakt gleich. Decode-Medianen: external `0,266399792 s`,
fused `0,266088688 s`. Ratio-Median `1,000510010`, Bootstrap-95-%-KI
`[0,981178182; 1,004700679]`, seed `20260825`, 10.000 Resamples, keine
Ausreißerentfernung. Entscheidung: `fused_greedy_compile_inconclusive`, kein
klarer Gewinn; Exit `1` ist für diesen Nicht-Gewinn korrekt.

Ressourcen- und Budget-Gates bestanden; RSS maximal `3.769.974.784 B`,
MLX-Peak `3.524.169.562 B`, Swap-Delta `0 B`. Rohzeiten sind gemessen;
Mediane, Perzentile, Ratio, KI und Prozentänderungen sind berechnet. Modell,
Gewichte, Quantisierung und Matmul blieben unverändert. Kein Qualitäts-,
Selbstlern-, Produkt- oder Aktivierungsclaim.

## 2026-08-27 — Qwen3.8-27B-Kompatibilitätsprobe mit IronMule

Der ausdrücklich freigegebene Checkpoint
`mlx-community/Qwen3.8-27B-4bit` wurde revisionsgebunden auf Commit
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff` in den lokalen Hugging-Face-Cache
geladen. Der vollständige Snapshot enthält 15 Dateien und drei Safetensors-
Shards mit zusammen `16.081.490.933 B` logischer Dateigröße. Die Konfiguration
weist `model_type=qwen3_5`, 64 Textlayer, `full_attention_interval=4` sowie
Affine-Quantisierung mit 4 Bit und Gruppengröße 64 aus. Nach Abschluss waren
auf dem Datenvolume rund 54 GiB frei.

Der vorgeschaltete IronMule-Gate-Lauf verwendete den strikten Plan, zwei
Requests und 48 maximale Tokens. Er endete mit Exit 1 vor einer verwertbaren
Performance- oder Korrektheitsmessung. Ursache:
`AttributeError: 'ArraysCache' object has no attribute 'keys'` in
`ironmule/runtime.py::_fixed_state_from_standard`. MLX-LM erzeugt für die
linearen Qwen3.5-Gated-Delta-Layer `ArraysCache(size=2)` und nur für die
Full-Attention-Layer `KVCache`; IronMule setzt aktuell für jeden Layer
`layer.keys` und `layer.values` voraus. Die vorgesehenen Stufen mit drei und
sechs Requests sowie die drei vollständigen Wiederholungen wurden deshalb
fail-closed nicht gestartet. Es gibt keine Performanceaussage und keinen
formalen Claim.

Keine Software wurde installiert oder aktualisiert, keine IronMule-
Architektur geändert und kein Workaround aktiviert. Ein erneuter Benchmark ist
erst sinnvoll, nachdem ein separat freigegebener, korrektheitsgeprüfter Vertrag
für den hybriden Qwen3.5-Cache existiert. Der Checkpoint bleibt lokal erhalten;
ein erneuter Download ist nicht erforderlich.

## 2026-08-27 — X2 Abschluss: Qwen3.5-Hybrid-Cache-Kompatibilität

Die Ursache des ursprünglichen Qwen-Fehlers war die unbedingte Annahme
`layer.keys`/`layer.values` in `_fixed_state_from_standard`; lineare
Gated-Delta-Layer liefern jedoch `ArraysCache(size=2)`. Der Adapter klassifiziert
bekannte Cachetypen fail-closed, trägt KV-Layer als `keys`/`values` und rekurrente
Layer als `arrays`, rekonstruiert `ArraysCache` mit nicht-aliasierenden Listen und
behält den Gemma-All-KV-Pfad unverändert. `lengths`/`left_padding` sowie hybride
Speculation werden ausdrücklich abgelehnt.

X2 verwendete Revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, MLX
`0.32.0`, mlx-lm `0.31.3` und Apple M1 Max mit 32 GB. Der korrigierte One-shot-
Referenzloop, Shape-Gate über zwei Dekodierschritte und Hybrid-Hash-Gate waren
bestanden; alle 64 Layer zeigten `AAAK` sechzehnfach. Service-Gates mit 2 und 3
Requests bei maximal 8 Tokens sowie 6 Requests bei maximal 48 Tokens waren exakt,
mit `fallbacks=0` und
`correctness_errors=0`; der kleine kompilierte Gate-Lauf war ebenfalls exakt.
Gemma-Vorher/Nachher-Token blieben für beide getesteten Requests exakt gleich.

Der erste `generate_step`-Harness wurde verworfen: Er splittete den Prompt vor
dem letzten Token und war daher keine One-shot-Prefill-Referenz. Das ist ein
Testdesignfehler ohne Produktbefund. X2 macht keine Performanceaussage und
generalisiert nicht; der kompilierte Gate-Peak von 30,76 GB ist ausschließlich
eine Speicherwarnung gegenüber 17,71 GB im vollständigen Baseline-Lauf.

## 2026-08-27 — B28-01 Preflight

- **Ziel:** Einen revisionsgebundenen B28-Baseline-Control-Lauf mit dem lokalen
  Qwen3.8-27B-Snapshot starten.
- **Beobachtung:** Der einzige Start endete nach `9.33 s` mit Exit-Code `2`,
  Classification `INCONCLUSIVE`, `0/6` Child-Prozesse.
- **Ursache:** Beim direkten Aufruf `python research/b28_baseline.py` war der
  Repository-Root nicht in `sys.path`; `ironmule` ist im verwendeten venv nicht
  installiert. Daher trat `ModuleNotFoundError: No module named 'ironmule'` im
  Parent-Preflight auf.
- **Auswirkung:** Kein Modell wurde geladen; es gibt keine Token-, Timing-,
  Correctness-, Speicher-, Swap- oder Performance-Messdaten. Der vollständige
  lokale Model-Fingerprint wurde lediglich als Preflight-Provenienz erfasst.
- **Lösung:** B28a registriert als reine Harness-Extension. Der Root wird vor
  Parent-Imports in `sys.path` eingefügt; die Child-Umgebung behält ihr
  explizites `PYTHONPATH`. Alle B28-Metriken, Arme und Gates bleiben unverändert.
- **Verifikation:** CPU-Harness-Tests und `py_compile` bleiben der nächste
  Prüfpunkt; ein neuer Hardwarelauf ist eine separate, explizit autorisierte
  Aktion und wird nicht automatisch ausgelöst.

## 2026-08-27 — B28a Lifecycle-Fehler

- **Beobachtung:** Nach der sichtbaren Meldung `child 1/6 failed` startete der
  Parent entgegen der gewünschten fail-closed Semantik einen weiteren Child-
  Aufruf. Der Parent wurde daraufhin manuell mit Exit `130` unterbrochen.
- **Auswirkung:** Kein finaler B28a-Resultat-JSON und kein persistierter Partial-
  Stand; `0` verwertbare Kinder. Die genaue Child-Ursache ist nicht aus dem
  sichtbaren Session-Output rekonstruierbar. Es existieren keine Modell-, Token-,
  Timing-, Correctness-, Speicher-, Swap- oder Performanceclaims.
- **Lösung:** B28b als reine Lifecycle-Extension preregistriert: automatischer
  Stop nach dem ersten Child-Fehler, atomarer versteckter Partial-Sidecar nach
  Preflight und jedem Child, sowie finaler `INCONCLUSIVE`-Publish bei Fehlern
  oder Interrupts. B28-Metriken und Gates bleiben unverändert.

## 2026-08-27 — B28b Lifecycle-Absicherung

- **Entscheidung:** B28b friert die B28-Metriken und Gates erneut unverändert ein
  und ändert ausschließlich die Evidence-Lifecycle-Kontrollen.
- **Mechanismus:** Nach erfolgreichem Preflight und nach jedem Child wird ein
  symlink-sicherer, atomarer versteckter Partial-Sidecar unter `research/raw`
  geschrieben. Ein erster Child-Fehler stoppt sofort; Fehler oder Interrupts
  publizieren, soweit möglich, ein finales `INCONCLUSIVE`-JSON. Der Sidecar wird
  erst nach erfolgreichem Final-Publish entfernt.
- **Verifikation:** Die CPU-Tests simulieren Child-Crash, Interrupt,
  Sidecar-Erhalt/Entfernung, exklusive Ausgabe und Symlink-Ablehnung. Vor B28b
  wurde kein Modell-/GPU-Lauf gestartet.

## 2026-08-27 — B28c Resource-State-Extension

- **Grundlage:** B28b wurde read-only aus dem Raw-JSON geprüft: `INCONCLUSIVE`,
  `1/6` Child, kontrollierter `SafetyAbort` direkt nach `after_model_load`.
- **Ressourcenbefund:** Swap-Baseline `25.876.108.410 B`, danach
  `28.757.133.885 B`, Delta `+2.881.025.475 B`. Es gibt keine Token-, Arm-,
  Correctness- oder Performancewerte.
- **Entscheidung:** B28c als reine Resource-State-Extension registriert. Ein
  neuer Lauf ist nur nach gesundem, read-only bestätigtem Memory-/Resource-
  Preflight zulässig. Swap bleibt strikt positiv-delta-gated (`> 0 B`), und alle
  B28/B28b-Metriken, Schwellen und Gates bleiben unverändert.
- **Verifikation:** Keine Systemzustandsänderung, kein Modell-/GPU-Lauf und
  keine automatische Wiederholung durchgeführt.

## 2026-08-27 — B28b/B28c Safety-Ergebnisse

- **B28b:** Nach `after_model_load` kontrolliert wegen positiver Swap-Differenz
  abgebrochen: `25.876.108.410 B` Baseline zu `28.757.133.885 B`, Delta
  `+2.881.025.475 B`; `1/6` Child, null Tokens/Arme, `INCONCLUSIVE`.
- **B28c:** Derselbe Ressourcen-Gate reproduziert: `26.856.327.741 B`
  Baseline zu `29.211.356.037 B`, Delta `+2.355.028.296 B`; `1/6` Child, null
  Tokens/Arme, `INCONCLUSIVE`.
- **Einordnung:** Beide Raw JSONs sind die Evidenz. Es gibt keinen Performance-
  oder Qualitätsclaim und keinen dritten automatischen Lauf. Ein weiterer Versuch
  ist erst nach extern geändertem Systemzustand zulässig, etwa Reboot oder
  ausdrücklich autorisiertem Beenden speicherintensiver Anwendungen. Das strikte
  positive Swap-Delta-Gate bleibt bestehen.

## 2026-08-27 — B28d Registration

- **Preflight-State:** Nach dem extern geänderten Systemzustand (Reboot) wurde
  read-only `swap=0` und `free=92%` berichtet.
- **Entscheidung:** B28d ist als reine Composite-State-Extension vor dem nächsten
  Modelrun eingefroren. Die aktuellen Claude-/User-Änderungen bleiben erhalten;
  B28/B28b/B28c-Metriken, Arme, Schwellen und Safety-Gates bleiben unverändert.
- **Korrekturen:** Ausschließlich die Referenzlänge (maximal 48 physische Tokens
  inklusive Prefill) und die Metrikdokumentation: Executor-/Decode-`wall_ns`
  ohne Prefill ist primär; äußere `Runtime.serve`-Zeit und End-to-End-Rate sind
  sekundär.
- **Verifikation:** Kein Python-/pytest-/Modell-/GPU-Lauf in dieser Änderung;
  nur Quellpatches und Diff-Prüfung.

## 2026-08-27 — R6/R7-IronMule-Abschluss und Verifikation

Die nicht-architektonischen R6/R7-Kernpunkte wurden im IronMule-Worktree
umgesetzt und geprüft: Paketimport verändert keine Hugging-Face-Umgebungsvariablen;
`load_engine` verwendet eine explizite Offline-Policy ohne Environment- oder
importierte-Modul-Mutation. Lokale Modellverzeichnisse werden direkt geladen,
Offline-Hub-IDs über `snapshot_download(local_files_only=true)` aufgelöst. Der
Cache-Resolver wurde anschließend um explizite Runtime-Allow-Patterns erweitert
(`*.safetensors`, `*.json`, `*.py`, `tokenizer.model`, `*.tiktoken`,
`tiktoken.model`, `*.txt`, `*.jsonl`, `*.jinja`), weil fehlende README- und
Gitattribute-Dateien einen ansonsten lauffähigen Gemma-Cache fälschlich
blockierten.

Profile werden schema- und standardmäßig kompatibilitätsgeprüft; `revalidate`
fordert schema-valide Rohprofile ausdrücklich für die Drift-Canary an und
verwendet die tatsächlich neue Prompt-Tokenlänge. Unsupported-Optimierungs-
kandidaten erhalten typisierte Dispositionen und brechen den Suchlauf nicht ab;
`FusionUnsupported` wird unabhängig vom Fehlertext erkannt, während generische
Fehler laut bleiben. Die Paketversion stammt aus einer einzigen Quelle, die
Dependency-Matrix ist auf `mlx>=0.32,<0.33` und `mlx-lm>=0.31.3,<0.32` begrenzt.
Die langsame GPU-Core-Erkennung wird pro Prozess sicher gecacht.

Ein isolierter CLI-Importfehler im B28-Preflight wurde reproduziert:
`python research/b28_baseline.py` startete ohne Repository-Root in `sys.path`,
und IronMule war im verwendeten venv nicht installiert; Ursache war
`ModuleNotFoundError: No module named 'ironmule'`. Die Lösung in B28a fügt den
Root vor Parent-Imports in `sys.path` ein und hält das explizite `PYTHONPATH`
für die Child-Umgebung.

Finale Verifikation und Messwerte:

- fokussierte Suite: `95 passed, 11 deselected`, `9.75 s` real, Peak-RSS
  `345.243.648 B`, `0` Swap;
- Gemma-Integration: `10` Tests, `26.58 s`, Peak-RSS `3.348.430.848 B`,
  `0` Swap;
- Doctor, finales Wheel und CLI-Metadaten: grün;
- Benchmark-Smoke: Tokenidentität bestanden, berechnet `-18.80 %` nur auf
  `2` Requests × `3` Tokens, KI `[1.0643, 1.3116]`; ausdrücklich kein Claim.
  Das JSON war temporär und hatte SHA-256 `36ba...`;
- systemweiter Swap: vor Integration `13295.19M`, nach Integration
  `13151.19M`, nach Smoke `12959.19M`; daraus folgt kein positives Delta.

Verbleibende Blocker: Architekturthema R2, R3 (Stock-/Fresh-Process-Baseline),
R6 (Modellrevision und Quantisierung), sowie eine separate IronMule-History-UI.
Diese Punkte wurden weder implizit freigegeben noch umgesetzt. Es erfolgten
keine Downloads und keine Installationen. Änderungen an `ProjectAtlas/` wurden
nicht vorgenommen.

## 2026-08-27 — Erratum zum R8-Doctor-Importfehler

Die Formulierung im vorherigen Abschlussabschnitt vermischte zwei getrennte
Fehler. Der aktuelle R8-Fehler trat durch die Reihenfolge der vollständigen
Suite auf: `_load_optional('mlx_lm')` importierte im Parent, fing einen
Transformers-/Torch-Importfehler ab und hinterließ teilweise initialisierte
Module. Die erfolgreiche Lösung sind isolierte Subprozesse für Dependency- und
Metal-Probes, sodass der Parent keine partiell initialisierten Importzustände
übernimmt.

Der B28-PYTHONPATH-Fehler ist separate Vorarbeit: Beim direkten Aufruf
`python research/b28_baseline.py` fehlte der Repository-Root in `sys.path`; die
damalige Lösung war die Root-Einfügung vor Parent-Imports und das explizite
`PYTHONPATH` für Child-Prozesse. Diese Ursache gehört nicht zum R8-Doctor-
Fehler. Die korrigierte Statusfassung nennt außerdem die vollständige
Nicht-Integrations-Suite (`95 passed, 11 deselected`), den vollständigen
Benchmark-Smoke-SHA `36ba45933b3de344116812e34bb451d19124b0ab35db3d3ee659b768dacc6209`
und dass Wheel/Metadatenprüfung sowie CLI-Zipimport-Smoke in einer temporären
Kopie ohne Installation erfolgten.

## 2026-08-27 — Finale Nicht-Integrations-Grenztests

Die zuvor dokumentierten `95 passed, 11 deselected` waren ein Zwischenstand.
Nach zwei zusätzlichen Grenztests ist der finale Nicht-Integrationsstand
`97 passed, 11 deselected`, `7.71 s` real, Peak-RSS `357826560 B` und `0`
Swaps. Neu abgesichert sind der gruppierte Pfad bei EOS bereits im Prefill
sowie ein leerer Runtime-Result-Text bei first EOS. Die Integrationswerte und
alle übrigen Mess- und Architekturgrenzen bleiben unverändert.

## 2026-08-27 — Backlog für Friday Learning Controller v0.1

- Im Project-Friday-Root existierte kein eigenes Backlog; das gefundene
  `docs/BACKLOG.md` gehört zum separaten IronMule-Worktree. Deshalb wurde
  `BACKLOG.md` im Root als lebende Liste offener Project-Friday-Arbeiten angelegt.
- Der neue Eintrag L1 beschreibt den kleinsten vertretbaren Weg zu einer
  begrenzt autonomen, lernenden Optimierungsruntime: zunächst Daten-/Architektur-
  Audit, kanonisches Optimization Memory v2, leakage-sicherer Korpus,
  Random/Grid/BO-Baselines, Regression/GBDT, Unsicherheits-/OOD-Gate und ein
  ausschließlich read-only arbeitender Shadow Controller.
- Dokumentiert wurden die benötigten Environment-, Workload-, Modell-,
  Kandidaten-, Compile-, Correctness-, Performance-, Ressourcen-, Systemzustands-,
  Ergebnis- und Provenienzdaten sowie Corpus-Coverage, Komponenten, Sicherheits-
  grenzen, Phasengates und Kill-/Pivotkriterien.
- Bewusste Entscheidung: v0.1 verwendet einen geschlossenen typisierten
  Aktionsraum; kein RL, kein freier Sourcecode, kein LLM als innerer Tuner oder
  Promotionrichter und keine automatische Aktivierung. Modellvorhersagen dürfen
  nur Messungen priorisieren und niemals Correctness oder Promotion ersetzen.
- Freigabegrenze: Diese Dokumentationsänderung erlaubt keine Implementierung,
  keinen Hardwarelauf, keinen Download, keine Installation und keine
  Architektur- oder Produktaktivierung. Der Startscope und jede spätere autonome
  Stufe benötigen eine ausdrückliche Freigabe.
- Verifikation: `git diff --check` und
  `xcodebuild -checkFirstLaunchStatus` endeten mit Exit `0`; die ProjectAtlas-
  Runtime `0.4.5-rc1` und die generierte projektlokale Codex-MCP-Konfiguration
  waren gültig. Zwei MCP-Refreshversuche trafen auf `sqlite error: database is
  locked`; `atlas_watch_status` zeigte keinen aktiven Watcher, eine eindeutige
  externe Lockursache war daher nicht belegbar. Der versionierte CLI-Fallback
  `projectatlas watch --once` aktualisierte den Index anschließend erfolgreich.
  ProjectAtlas erkannte `BACKLOG.md` danach als 292-zeilige Dokumentation mit
  aktuellem strukturellem Summary. Wegen der reinen Dokumentationsänderung wurde
  kein Modell-, GPU- oder Benchmarklauf gestartet.

## 2026-08-27 — Finale CLI-Loader-Regressions

Der vorherige Stand mit `97 passed, 11 deselected` war ein Zwischenstand. Nach
den CLI-Loader-Regressionen ist der finale vollständige Nicht-Integrationsstand
`103 passed, 11 deselected`, Pytest `5.94 s`, `6.63 s` real, Peak-RSS
`358645760 B` und `0` swaps. Sichere reale No-Model-Smokes für `tune --show`,
`status` und `revalidate` endeten jeweils mit Exit 0.

Ursache des zusätzlichen Testfehlers war der Namenskonflikt zwischen der aus
`ironmule` re-exportierten `tune`-Funktion und dem Submodul `ironmule.tune`.
Die Regressionen verwenden nun explizit
`importlib.import_module('ironmule.tune')`; die übrige Historie bleibt
unverändert.

## 2026-08-27 — Finale `ironmule models`-CLI-Verifikation

Der Stand mit `103 passed, 11 deselected` war ein Zwischenstand. Nach den
`ironmule models`-CLI-Regressionen ist der finale Nicht-Integrationsstand
`108 passed, 11 deselected`, Pytest `7.62 s`, `/usr/bin/time` `8.46 s` real,
Peak-RSS `346996736 B` und `0` swaps. Der read-only Models-Smoke fand Gemma
unter Commit `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, Größe `3439894985 B`,
mit leeren Warnings; es erfolgten keine Resolution und keine Downloads.

## 2026-08-27 — Python-native SIGABRT nach Neustart

Nach dem Neustart traten gegen ca. 21:06 drei native Python-Abstürze auf.
Die Apple-Diagnoseberichte und der vorliegende Screenshot weisen auf Homebrew
Python 3.12.13 (ARM64), `SIGABRT`/„Abort trap 6“ und eine Beteiligung von
`libmlx.dylib`/`AGXMetal` hin; als verantwortlicher Prozess ist Codex
ausgewiesen. Das ist kein abfangbarer Python-Exception-Fehler: Der native
C++/Metal-Pfad beendet den Prozess direkt, wenn aus einer Pytest-/Projekt-
Importumgebung im Sandbox-/Headless-Kontext kein nutzbares Metal-Gerät zur
Verfügung steht.

Operative Regel: Python- oder Pytest-Läufe, die `ironmule`/MLX importieren
können, dürfen nicht in der Sandbox ausgeführt werden, sondern benötigen
`require_escalated` und ein reales Metal-Gerät. Reine-stdlib-Harnesses sind
nur dann eine Ausnahme, wenn ihre vollständige Importabschlussmenge nachweislich
MLX-frei ist; andernfalls ebenfalls erhöht ausführen. Der betroffene
Verifikationslauf wurde beendet, aktuell laufen keine Python-Prozesse. Diese
Beobachtung enthält keine Performanceaussage.

## 2026-08-27 — B28-Integrations-Harness: Request-Cap-Mapping korrigiert

Bei der lokalen B28-Integrationsprüfung scheiterte der Harness vor dem ersten
`Runtime.serve`-Aufruf mit einem `NameError`: Die Request-Comprehension iterierte
mit `i`, griff für das Tokenlimit aber auf `index` zu. Die Ursache war damit ein
reiner Harness-Fehler; es gab keinen Modellbefund und keinen nativen Crash.
Das Mapping verwendet jetzt konsistent `index`, validiert genau eine Cap pro
Prompt und schreibt die RID ebenfalls aus diesem Index.

Der beobachtete Swap-Delta blieb bei `0 B`. Es wurde wegen der bekannten
Python-/MLX-SIGABRT-Grenze kein Modell- oder Metal-Lauf gestartet; die Korrektur
ist source-level geprüft und `git diff --check` bleibt der einzige ausgeführte
Code-Gate.

## 2026-08-27 — B28 True-Batch-Korrektheitsbefund und B29-Start

Der experimentelle B28-Pfad `qwen_native_true_batch_v1` erreichte die
Korrektheitsgrenze bei Breiten 2, 3 und 4: sichtbare Tokens und Stop-Gründe waren
exakt, es gab keine Fallbacks und das Swap-Delta betrug `0 B`; der finale hybride
`kv_hash` wich jedoch vom sequenziellen Zustand ab. Deshalb wird B28 verworfen und
nicht geroutet. Es liegt keine gültige Tokenrate-/Performance-Messung vor. Die
gemeldeten Hashwerte selbst waren im Fehlerbericht nicht aufgezeichnet; sie werden
nicht rekonstruiert.

B29 (`qwen_native_b1_v1`) ist als neuer, noch ungemessener Forschungsversuch
registriert: native Qwen-Caches bleiben pro Session erhalten, während der aktuelle
Grouped-Batch-1-Scheduler unverändert bleibt. Gemma bleibt auf dem bestehenden
All-KV-Pfad; verzögerte Ankünfte bleiben außerhalb des B29-Simultanbereichs und
fallbacken. Es wurde kein Modell-, Python- oder Metal-Lauf gestartet.

## 2026-08-27 — B29a Pilot preregistriert

B29a ist als ein einzelner, lokaler `PILOT_ONLY`-Lauf preregistriert. Der Lauf
verwendet einen frischen Prozess, dieselbe lokale Qwen-Architektur und den
vererbten B28-Arm-/Safety-Harness. Der verpflichtende B29-Korrektheitsgate mit
sechs Fragen, 48 physischen Tokens, Breiten 2/3/4 und 16-token Continuation muss
vor jeder Warmup- oder Zeitmessung bestehen. Ausgabe: `B29a_pilot_20260827.json`.
Eine positive Pilotklassifikation ist keine Routing-, CI- oder allgemeine
Performancefreigabe; automatische Wiederholung ist ausgeschlossen. Es wurde kein
Modell-, Python- oder Metal-Lauf gestartet.

## 2026-08-27 — B29b Operator-Pfadfehler und B29c Exact Binding

Der einzelne B29b-Versuch endete bereits im Preflight mit Exit 2, weil der
Operator den nicht existierenden Pfad `Qwen3.8-27B-4bit-4bit` angab. Es gab keine
Modell-, Metal-, Output- oder Performance-Evidence und keinen nativen Crash;
der Befund steht in `B29b_execution_failure.json`.

B29c bindet den Pilot nun fail-closed an den kanonischen lokalen Snapshot
`.../models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
und den vollständigen Modelldigest
`4a95d45652dbd0399e4a8a6ba86e83278799ebc249776867a9a646ab3d5379b5`. Jeder
andere Pfad oder Digest wird vor Modellarbeit abgelehnt. Gates und Schwellen
bleiben unverändert; es wurde kein Modell-, Python- oder Metal-Lauf gestartet.

## 2026-08-27 — B29a direkter Script-Importfehler und B29b

Der direkte B29a-Pilotstart scheiterte einmal vor Modell- und Metal-Arbeit mit
`ModuleNotFoundError: research` an Importzeile 16. Der Prozess endete mit Exit 1;
es gab keinen nativen Crash, keine Ausgabe und keine Performance-Messung. Der
Befund ist in `B29a_execution_failure.json` festgehalten.

B29b ist als reine Pfadkorrektur preregistriert: Der Repository-Root wird aus
`__file__` abgeleitet und vor dem `research.b28_baseline`-Import in `sys.path`
eingefügt. Alle B29a Gates, Schwellen, Workloads und Pilotgrenzen bleiben
unverändert; Ausgabe ist `B29b_pilot_20260827.json`. Es wurde kein Modell-,
Python- oder Metal-Lauf gestartet.

## 2026-08-27 — B29c unter dem 10%-Ziel; B30 registriert

B29c (`qwen_native_b1_v1`) bestand bei Breiten 2/3/4 die Korrektheits-, State-
und 16-token-Continuation-Gates ohne Fallback und mit Swap-Delta `0 B`. Die
Medianrate betrug `16.0722` gegenüber Interactive `15.6740` (`1.02541x`) und
Throughput `16.0687` (`1.000219x`); damit wurde das `1.10`-Ziel verfehlt und
kein Routing aktiviert. Dieser Befund ist in X4/der B29c-Evidence festgehalten.

B30 (`qwen_wide_grouped_b1_v1`) ist als neuer Forschungsversuch registriert:
Qwen-Sessions dürfen in unabhängigen Fixed-State-Batch-1-Gruppen bis Breite 6
laufen; Native-Cache-Merge, Compile und Gemma-Änderungen bleiben ausgeschlossen.
Der verpflichtende Gate prüft Breiten 4/5/6 vor Timing. Es wurde kein Modell-,
Python- oder Metal-Lauf gestartet.

## 2026-08-27 — B30-Korrektheitsgate bestanden, Ressourcenstopp vor B30a

Der reale B30-Integrationsgate bestand für Breiten 4/5/6 mit exakten Tokens,
Stop-Gründen, Counts und State-Hashes, null Fallbacks, Exit 0 und `152.16 s`
Laufzeit. Es wurde kein Token-ID-Artefakt erzeugt; daraus folgt keine Produkt-
oder Performanceaussage. Die Ressourcenmessung zeigte Swap von `505.75 MiB` auf
`2676.69 MiB` (`+2170.94 MiB`) und freie-Memory-Telemetrie von `82%` auf `87%`;
kein Crash trat auf.

B30a wird deshalb bis zu einem Reboot/Resource-Reset nicht gestartet. Die
Preregistration sowie alle Gates und Schwellen bleiben unverändert. Die Fakten
liegen in `B30_correctness_gate_20260827.json`.

## 2026-08-27 — B30a Pilot nach Warmup wegen Swap gestoppt

Der B30a-Mandatory-Gate bestand für Breiten 4/5/6 exakt und ohne Fallback. Im
ersten Warmup lagen die Executor-Raten bei Interactive `15.8483`, Throughput
`16.4353` und Candidate `16.4240` Tokens/s. Danach stoppte der Safety-Gate bei
Swap-Delta `314111427 B` und MLX-Peak `23882126950 B`. Es liefen keine Messrepeats;
die Klassifikation bleibt `INCONCLUSIVE` ohne Performanceclaim.

B30a wird nicht wiederholt. B30b ist als Resource-Reset-Extension registriert
und darf erst nach extern verifiziertem Reboot/Reset mit Swap `0 B` starten;
Gates und Schwellen bleiben unverändert. Die vorhandene JSON wurde mit `jq`
validiert.

## 2026-08-27 — B24 GPU-Capture-Smoke

Das installierte MLX `0.32` stellt Start-/Stop-Capture und Memory-Counter bereit;
öffentliche Counter-/Profilnamen wurden dabei nicht festgestellt. Der erste
Tiny-Smoke scheiterte, weil keine Capture-Layer eingefügt war. Der Retry mit
`MTL_CAPTURE_ENABLED=1` gelang für eine Tiny-64-Matmul und erzeugte den Trace
`/private/tmp/ironmule_b24_enabled_smoke.gputrace`. Es gab keine Timing-/
Performanceaussage und keinen Crash; der Trace bleibt außerhalb des Repositories.

B24 bleibt offen für genau einen Modelldecode-Trace mit Xcode-Analyse. Maßgebliche
Dokumentation: Apples [GPU-Counter-Statistik](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
und [Metal Developer Tools](https://developer.apple.com/metal/tools/); MLX
`get_active_memory`/`get_peak_memory` werden nur als Memory-Diagnostik behandelt.

## 2026-08-28 — B35 Gemma-Core-Profil: exploratives Portabilitätsscreening

**Ziel und Atlas-first.** Ziel war ausschließlich die Prüfung des bereits für 4B
validierten, nicht modellmutierenden Profils
`Knobs(compiled_fixed_cache=True, head_skip_prefill=True)` gegen
`BASELINE=Knobs()` auf lokalen Gemma-3-1B-, 4B- und 12B-Snapshots. Vor der
Dokumentation wurde ProjectAtlas `0.4.5-rc1` mit fokussiertem Session-Brief für
den Worktree `.worktrees/ironmule-qwen-hybrid-cache` verwendet; der Index war
verfügbar. Es gab keine Downloads oder Installationen. Die Arbeitsliste und
Präregistrierungen waren vor dem ersten Modelllauf per `apply_patch` angelegt:
[`B35_preregistration.md`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_preregistration.md)
und die spätere reine Kontaminationskorrektur
[`B35a_preregistration.md`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35a_preregistration.md).

**Sicherheits- und Laufregel.** Die Modellläufe wurden ausschließlich außerhalb
der Sandbox mit `require_escalated` auf dem vorhandenen Projektinterpreter
`/Users/tobiasburandt/Project_Friday/.venv/bin/python` ausgeführt. Der temporäre
Worker `/private/tmp/b35_worker.py` erzwingt Offlinepfade, einen
`signal.alarm`-Hardtimeout von 840 s innerhalb des Prozesses und wurde zusätzlich
mit einem 900-s-Perl-Timeout gestartet. Pro OS-Prozess wurde das Modell genau
einmal geladen; beide Arme liefen auf derselben Engine/demselben Modell, mit
explizitem Verwerfen des Compile-Caches beim Knobwechsel. Es liefen niemals zwei
Modellprozesse gleichzeitig. Nach jedem Prozess wurden Prozessliste,
Crashreport-Verzeichnis, Swap und Roh-Gates geprüft; am Ende liefen keine
Modell-/Pythonprozesse und es gab keinen neuen Python-Crashreport.

**Frühere Ansätze und Fehler.**

- **Approach A, 1B-Smoke:** Die tatsächlich belegte Agent-A-Messung lief offline
  mit Gemma 3 1B in einem Prozess: `load_count=1`, Load `1.8827675 s`,
  `8` greedy Tokens in `0.419004667 s` (`195.854816 tok/s`), MLX-Peak
  `0.749093638 GB`, Token-SHA
  `c9701f6b75bb08f6116f2c577b7f19e87cdadab7516adacc7c1087f1a0902184`.
  Rohdatei: `/private/tmp/gemma3-1b-smoke-20260828.json`. Dies war ein
  Smoke-Lauf; daraus folgt keine Forward-, Logit- oder Prefill-18,96-ms-
  Aussage.
- **Approach B:** Ein einmaliger IronMule-Benchmarkversuch scheiterte bei der
  Shell-Auswertung an der in zsh read-only reservierten Variable `status`
  (`read-only variable: status`). Es gab keine JSON-Datei und keinen
  Benchmarkoutput; es wurde nicht wiederholt und Gemma 4B nicht gestartet.
- **Approach C / B25:** Der erste Lauf scheiterte mit `NameError` an `_leaves`,
  danach mit falscher `active`-/`peak`-Key-Aggregation. Im finalen Versuch waren
  1B und der bereits vor dem Stop gestartete 4B-Lauf tokenidentisch und
  deterministisch; Active Memory war nach dem ersten Schritt stabil und Peak
  konstant. MLX stellt jedoch keinen Allocation-Event-Counter bereit, daher
  bleibt B25 `INCONCLUSIVE` ohne Speedaussage; 12B wurde nicht gestartet.

**Python-/MLX-Absturzgrenze.** Bei der Diagnose wurden genau diese drei nativen
  Diagnoseberichte festgestellt: `~/Library/Logs/DiagnosticReports/`
  [`Python-2026-08-28-074908.ips`](../../Library/Logs/DiagnosticReports/Python-2026-08-28-074908.ips),
  [`Python-2026-08-28-074922.ips`](../../Library/Logs/DiagnosticReports/Python-2026-08-28-074922.ips)
  und [`Python-2026-08-28-074932.ips`](../../Library/Logs/DiagnosticReports/Python-2026-08-28-074932.ips).
  Sie zeigen Python 3.12.13 ARM64, `SIGABRT`/„Abort trap 6“ mit
  `libmlx.dylib`/`AGXMetal` im Codex-Prozess. Die belastbare Abgrenzung ist:
  sandboxierte MLX-/Metal-Probes abortierten, während derselbe minimale Smoke
  außerhalb der Sandbox lief. Daraus folgt als Evidenz/Inferenz eine
  Sandbox-/Metal-Initialisierungsgrenze; eine bestimmte interne native Ursache
  ist damit nicht sicher bewiesen. Die Lösung war, MLX-/IronMule-Python nur mit
  realem Gerät außerhalb der Sandbox über `require_escalated` auszuführen; die
  reine Worker-Syntaxprüfung blieb getrennt.

**Approach D / Kontaminationskorrektur.** Der erste 1B-AB-Versuch lief zeitweise
  parallel zu breiten `find`-Suchen. Seine Rohdatei
  [`B35_gemma1b_AB_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma1b_AB_20260828.json)
  ist mit `valid_for_metrics: false` markiert und liefert keinen Performancewert.
  B35a fror als einzige Korrektur eine such-, CPU- und I/O-freie Umgebung vor und
  während jedes Laufes ein; Arme, Prompt, Schwellen und Wiederholungen blieben
  unverändert.

**B35-Ergebnis.** Der feste Chat-Prompt hatte 322 Tokens und den
  Prompt-Token-Digest
  `80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad`.
  Jede Modellgröße absolvierte zwei saubere frische Prozesse (AB und BA), je
  zwei Warmups und fünf Rohwiederholungen je Arm. Alle sechs Prozesse bestanden
  Tokenidentität, Determinismus, Peak-Memory- und Swap-Gates. Die Aggregate sind
  die Mediane der beiden Prozess-level-Medianverhältnisse `core/baseline`:

  | Modell | total | prefill | decode | Peak-Ratio | Swap-Delta |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Gemma 1B | `0,8495684` | `0,7562452` | `0,8971580` | `0,8794038` | `0 B` |
  | Gemma 4B | `0,8702504` | `0,8356965` | `0,9444767` | `0,9518492` | `0 B` |
  | Gemma 12B | `0,9840062` | `0,9638611` | `1,0149662` | `0,9853990` | `0 B` |

  Die 12B-Orderwirkung ist wesentlich: AB total `0,9040384`, BA total
  `1,0639741`; Prefill AB/BA `0,8853069`/`1,0424153`, Decode AB/BA
  `0,9410`/`1,0889`. Die 1B-Ausgabetokens haben Digest
  `11ac58e1ae29408d9762daee4df4749281ce24459f218e78b137dd31ae5ce0f7`,
  4B und 12B teilen Digest
  `d9818d21a6a6bef76c4091ef56ba158dfbc553a0f6c90e3d06543034be2a100f`.
  Umgebung war Python `3.12.13`, MLX `0.32.0`, mlx-lm `0.31.3`, NumPy `2.5.2`
  auf Apple M1 Max, 32 GB Unified Memory, Darwin/AC-Power; die lokalen
  Snapshot- und Konfigurationsdigests stehen in den sechs Rohdateien.

  Rohreferenzen: [`B35_gemma1b_AB_clean_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma1b_AB_clean_20260828.json),
  [`B35_gemma1b_BA_clean_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma1b_BA_clean_20260828.json),
  [`B35_gemma4b_AB_clean_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma4b_AB_clean_20260828.json),
  [`B35_gemma4b_BA_clean_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma4b_BA_clean_20260828.json),
  [`B35_gemma12b_AB_clean_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma12b_AB_clean_20260828.json),
  [`B35_gemma12b_BA_clean_20260828.json`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_gemma12b_BA_clean_20260828.json).

**Entscheidung.** Wortlaut des Ledger-Entscheids: „exploratory candidate
qualifies under gate; 12B result order-sensitive/inconclusive for robust
performance“. 1B/4B sind in diesem Screening stabil; 12B wird trotz des
aggregierten `<=0,995`-Werts wegen des starken AB/BA-Effekts nicht aktiviert.
Keine Shipping-, Routing-, Architektur- oder allgemeine Cross-Model-Aussage.
Der nächste offene Schritt ist B36 mit arm-isolierten frischen Prozessen; jetzt
wurde kein weiterer Lauf gestartet.

**Historien-UI.** Die read-only ProjectAtlas-/Dateisuche fand keine eindeutige
append-only JSON- oder SQLite-freie lokale UI-History-Quelle. Die vorhandene
Runtime-Historie ist `.friday-data/runtime.sqlite3` mit bestehender Loopback-UI;
eine B35-Aufnahme dort würde das SQLite-/Architekturformat ändern. Deshalb wurde
keine UI-Datei geändert. B35 bleibt über Ledger, Journal und die sechs Raw-JSONs
reproduzierbar dokumentiert.

## 2026-08-28 — B35 Review-Limitierungen

Der unabhängige Review bestätigte die B35-Ratios, stellte aber vier Grenzen
fest: Das Raw-Swap-Gate beginnt erst nach dem Modell-Load und deckt die Load-
Phase nicht ab; `hard_gates.no_crash` ist im Worker konstant `true`, während
die externen Post-Run-Prozess-/Crashreport-Prüfungen keine neuen Python-
Crashreports fanden, aber nicht im Raw-Gate kodiert sind; beide Arme teilten
Engine und Modell je Prozess, wodurch Allocator-, Compile- und Thermalzustand
mit AB/BA gekoppelt bleiben; und pro Raw-Datei wird nur die erste Repeat-
Tokenliste gespeichert, ohne Stopgründe oder alle Repeat-Tokenlisten.

Die zulässige Aussage bleibt daher explorativ: 1B/4B qualifizieren den
Kandidaten unter dem aufgezeichneten Screen, 12B ist wegen des starken
Order-Effekts für robuste Performance inkonklusiv. Keine Aktivierung oder
allgemeine Performanceaussage. Der Review liegt in
[`B35_review.md`](../.worktrees/ironmule-qwen-hybrid-cache/research/raw/B35_review.md);
B36 verlangt getrennte frische Prozesse pro Arm, Swap ab Prozessstart vor Load,
externe Crashreport-Snapshots und vollständige Repeat-Token-/Stop-Aufzeichnung.

## 2026-08-28 — B37 Phase-/Roofline-Diagnose

Im IronMule-Worktree wurde eine pure, CPU-only testbare
phase_roofline_diagnostic-Berechnung ergänzt. Sie trennt Prefill- und
Decode-Dauer, zählt decode_steps ohne den ersten Prefill-Token und verwendet
nur explizite per-Run-Eingaben für effektive Bandbreite, aktive Gewichtsbytes,
KV-Lese-/Schreibtraffic und Extra-Traffic. Es gibt keine globale
Effizienzkonstante und keine Compute-/Bandwidth-bound-Klassifikation.

Fail-closed: fehlende Phase oder Semantik ergibt inconclusive, ungültige,
negative, boolesche, nicht-finite oder null-denominator Eingaben ergeben
invalid, und ein Zero-Step/EOS-first-Fall ist not_applicable. Effizienzwerte
größer als eins werden nicht gekappt, sondern als
efficiency_above_one_input_consistency gewarnt. Der Helper verändert keine
Runtime-, Profil-, Correctness-, Swap- oder Crash-Gates und erzeugt keinen
Performanceanspruch.

CPU-Test: /Users/tobiasburandt/Project_Friday/.venv/bin/python -m pytest
tests/test_benchmark.py im Worktree, außerhalb der Sandbox, 25 passed.
Keine Modelle, MLX-/Metal-/ANE-Läufe, Downloads oder Installationen.

## 2026-08-28 — B37a Review-Härtung

B37 wurde gegen numerische Randfälle und Bandbreitenprovenienz gehärtet:
extreme Integer-Konvertierung, subnormale Dauerwerte, Traffic-Summen sowie
Idealrate-/Effizienz-Berechnung werden overflow-/underflow-sicher und
JSON-fähig fail-closed behandelt; decode_steps über 2^63-1 werden abgewiesen.
Die Provenienz ist strukturiert und verpflichtend: nur measured_effective darf
Idealrate und Effizienz erzeugen, nominal_peak bleibt ohne Effizienzanspruch
inconclusive. Zero-Step setzt Decode und Roofline auf not_applicable, der
Gesamtstatus bleibt inconclusive; der Helper kennt keinen EOS-Grund.

Der finale CPU-only Lauf war:
/Users/tobiasburandt/Project_Friday/.venv/bin/python -m pytest
tests/test_benchmark.py — 37 passed, exit 0. Keine Modell-, MLX-, Metal-,
ANE-, Download- oder Installationsläufe.

## 2026-08-28 — B36a Präregistrierungs-Klarstellung

Die versiegelte B36-Präregistrierung blieb unverändert. B36a ergänzt als
separates, hashgebundenes Clarification-Artifact ausschließlich die Korrektur
des Hash-/Prefault-Scope: Jeder Manifest-Eintrag wird vor und nach dem
Modell-Load vollständig SHA-256-gehasht; alle übrigen B36-Gates und Konstanten
bleiben unverändert. B36a wird durch Parent und Child neben B36 geprüft.

Zusätzlich wurden der reale memory_pressure-Output, strikte Child-Identity,
vollständige Repeat-Token-/Stop-/Decode-Step-Semantik und Partial-Evidence
explizit abgesichert. Der finale CPU-only Lauf:
/Users/tobiasburandt/Project_Friday/.venv/bin/python -m pytest
tests/test_b36_core_profile.py — 36 passed, exit 0. Kein 12B-, Modell-,
MLX-, Metal- oder ANE-Lauf.

## 2026-08-28 — B36/B36a Ergebnis und unabhängiger Audit

B36 lief nach der vollständigen Harness-Sicherheitsprüfung mit 16 seriellen
Paaren (8 AB/8 BA) und 32 frischen Child-Prozessen. Je Child wurden ein Load,
zwei Warmups und fünf Mess-Repeats mit vollständigen Token-/Stop-/Timing-
Aufzeichnungen durchgeführt. Foundation low_power=0 und thermalState=0,
AC/powermode=2, Memory-Pressure 74/75/66 Prozent, Swap-Gate 0 B,
Crashreport-/Residual-Prozess-Gates und finale H2-Prüfung waren grün.
Der volle Manifest-Hash wurde vor und nach dem Load ausgeführt; wired-/cache-
Limit-Mutationen blieben unverändert aus.

Raw: research/raw/B36_gemma12b_results_20260828.json.
B36 SHA-256: 7bf3997b19dc55d3b75be977c0da8d42d6ab554232ce2bf40617429c478897a4;
B36a SHA-256:
ee5b3e9b250d75eb69ed6e38f9661f656da743098bef318966dc055099c9e492.
Das vollständige Token-/Stop-/Decode-Step-Digest aller 32 Children ist
0c04b2910e0b8e5adc2c66108a79f4cbf233bf7fc8465f0a4d30418b6533019e.

Unabhängig neu berechnet wurden Gesamt-Ratio 0.927147428180255
(95%-CI [0.9197363534291831; 0.9303748490885659]), Prefill
0.9183106745417602 ([0.9081866453423364; 0.9218801379522791]) und Decode
0.9540158794083631 ([0.9419388082376179; 0.9577180135679649]).
AB lag bei 0.9250279042521969, BA bei 0.9294847335372622; Interaktion
0.0044568292850653. Das ergibt rechnerisch 7.29% Gesamt-, 8.17% Prefill-
und 4.60% Decode-Reduktion; ein einzelnes Decode-Paar über eins bleibt
Diagnostik. Candidate-RSS war mit 4.526 GB höher als Baseline mit 3.693 GB,
trotz niedrigerem MLX-Peak von 7.831 GB gegenüber 7.947 GB; daraus wird kein
verdeckter Speicherclaim abgeleitet.

Der Raw-Status ist QUALIFIED, aber activation_allowed=false. Keine
Profilaktivierung, kein Routing und keine allgemeine Performanceaussage.
Die unabhängige Prüfung liegt in research/raw/B36_review.md; die Code-
Digest-Prüfung korrigierte einen scheinbaren Mismatch, der nur durch Entfernen
des Git-Commit-Newlines in der ersten Audit-Rechnung entstanden war. Der
exakte 61-Dateien-Fingerprint und Commit stimmen mit dem Raw überein. Eine
UI-/SQLite-Aufnahme wurde wegen des bestehenden Historienformats nicht
vorgenommen; UI unverändert. B38 bleibt als architekturfreigabepflichtiger
Canary offen.

## 2026-08-28 — B39/B39a/B39b Safety-only Chronologie

**B39 Direct-Script-Fehler.** Der erste B39-Pilot-Aufruf über den direkten
Scriptpfad endete vor Parent-Initialisierung mit Returncode `1` und
`ModuleNotFoundError: No module named 'research'` in
`research/b39_combined_levers.py:22`. Kein Modell und kein Child liefen, es
gab weder JSON noch Partial, Crashreports blieben `30 -> 30`, kein Residual-
Modellprozess blieb zurück. Die nicht-metrische Rohakte ist
`research/raw/B39_pilot_import_failure_20260828.json`.

**B39a Modul-Pilot.** Der korrigierte Modulaufruf versuchte ausschließlich Arm
A. Das Child lud das Modell, stoppte bei `after_model_load` mit Returncode `3`
und `RuntimeError: B36 checkpoint gate failed: after_model_load`; Warmups und
Mess-Repeats starteten nicht. Das Partial bewahrte Parent-Swap von
`1,704,921,661 B` vor dem Child und `8,568,438,784 B` danach, also Delta
`6,863,517,123 B` beziehungsweise rund `6.39 GiB`. Das ist eine starke
Ressourcen-/Swap-Fehlerinferenz, aber der genaue Subgate-Typ ist wegen des
verlorenen Child-Eventstreams nicht beobachtbar. Crash-Delta war `0`, es gab
keinen Residual-Modellprozess. Der Parent folgte anschließend mit
`StatisticsError: no median for empty data`; kein Final-JSON entstand, das
Partial blieb erhalten. Kein Retry, kein B39-Main-Lauf und kein
Performanceclaim. Rohakte: `research/raw/B39a_pilot_failure_20260828.json`.

**B39b.** B39b ist ausschließlich Evidence-/Safety-Korrektur, SHA-256
`403eb1b098d49bff891a52ac16b974857b4fad3e0ed2984f554436acf0e9e7cb`, ohne
Hardwareautorisierung oder Lauf. Vorgesehen sind Event-Erhaltung in Partial-
und Failure-Raw, strukturierte `INCONCLUSIVE`-Ausgabe bei leerem Summary unter
Beibehaltung des Partials sowie ein absoluter Pre-Spawn-Swap-Gate von
`268,435,456 B` (`256 MiB`) zusätzlich zum unveränderten Prozessstart-bis-Ende-
Delta-Gate. B39-Statistik, Arme, Workload, Schwellen und No-Activation bleiben
unverändert. Vollständige Safety-Dokumentation: `research/raw/B39_review.md`.

**Separater xdist-Vorfall.** Ein versehentlicher Testaufruf ohne `-n0` nutzte
den in `pytest.ini` registrierten `xdist -n auto`-Pfad und erzeugte `23`
Python-`SIGABRT`-Reports zwischen `11:38:38` und `11:38:49`, Parent-PID
`80772`, über MLX/`libmlx`. Repräsentative Dateien sind
`Python-2026-08-28-113838.ips` und `Python-2026-08-28-113849.ips` unter
`~/Library/Logs/DiagnosticReports/`. Die Lösung ist die serielle Ausführung
aller B39-Tests mit `-n0`; der spätere serielle CPU-Lauf bestand `46` Tests,
Crashreport-Zähler `30 -> 30`, `git diff --check` grün. Der xdist-Vorfall ist
keine B39-Evidence. UI, Profilaktivierung, Routing und allgemeine
Performanceclaims bleiben unverändert.

## 2026-08-28 — B39b Benchmark-Preflight blockiert

Der angeforderte B39b-Preflight maß `vm.swapusage` mit total `8192.00M`, used
`7143.12M` und free `1048.88M`; `memory_pressure` meldete `75%` freien
Speicher, und es lief kein Gemma-Prozess. Das absolute B39b-Pre-Spawn-Swap-
Gate von `<=256 MiB` schlug fail-closed fehl. Daher wurden kein Modell, Child
oder Benchmark gestartet und keine Optimierung geändert.

Die serielle CPU-Harness-Nachprüfung bestand mit `46` Tests in `7.42 s`, Exit
`0`; Crashreport-Zähler User/System blieben vor und nachher bei `64/61`
(Delta `0`), `git diff --check` war grün und die eingefrorenen Hashes blieben
unverändert. Ein weiterer B39b-Lauf ist erst nach Reboot und sauberem,
verifiziertem Systemzustand zulässig. Der hohe Swap ist ausschließlich ein
Safety-Blocker, kein Runtime-Speedbefund.

## 2026-08-28 — B39b Pilot diagnostisch, INCONCLUSIVE

Nach einem sauberen Preflight (System-Swap `0 B`, `93%` freier Speicher, kein
Gemma-Prozess) liefen vier frische serielle Children in A/B/D/C. Alle vier
Returncodes waren `0`; Correctness-, Environment-, Workload-, Crash- und
Canonical-Gates bestanden. Je Arm wurden zwei Warmups und ein Mess-Repeat
ausgeführt. Alle sechs Requests produzierten `48` physische, logische und
sichtbare Tokens mit Stop-Grund `length`; der Canonical-Output-Digest war
identisch. Swap war `0 B`, relevante Crashreports und Residualprozesse fehlten.

Der Pilot ist trotzdem `INCONCLUSIVE`, weil das Peak-Gate allein an RSS C/A
`3.6523564` scheiterte (D/B `1.0001511`). MLX-Peak-Ratios: C/A `1.0064033`,
D/B `1.0257863`; MLX-Peaks A/B/D/C:
`7,796,516,616`/`7,801,367,483`/`8,002,535,534`/`7,846,439,900 B`.
RSS-Peaks A/B/D/C:
`2,166,931,456`/`7,916,470,272`/`7,917,666,304`/`7,914,405,888 B`.

Die einmaligen diagnostischen Outer-Wall-/Rate-Werte, Ratios und Interaktion
`D*A/(B*C)=0.987856765` stehen in
`research/raw/B39b_pilot_gemma12b_combined_20260828.json`; sie sind kein
Performanceclaim. Die RSS-Form A `2.17 -> 1.26 GB` gegenüber B/D/C nahe
`7.9 GB` bei identischem MLX-Active-Memory nahe `7.188 GB` macht eine
Reihenfolge-/Page-Residency-Konfundierung plausibel; keine Arm-Attribution.
Kein Main-Lauf, kein Retry, keine Aktivierung oder UI-Änderung. B39c mit zwei
neuen Crossover-Blöcken steht nach Clean-State aus; diese Pilotdaten werden
nicht wiederverwendet/gepoolt.

## 2026-08-28 — B39c Memory-Order-Diagnostic

Nach sauberem Preflight (System-Swap `0 B`, kein Residual-Modellprozess)
liefen die zwei neuen seriellen Blöcke `ABDC` und `CDBA` mit acht Children,
alle Returncode `0`. Correctness-, Identity-, Workload-, Crash-, Post-State-,
absolute Memory- und Swap-Gates bestanden; Swap war `0 B`, finales H2 war grün,
keine relevanten Crashreports oder Residualprozesse.

MLX C/A Peak-Ratios: `1.0064022925` und `1.0064018108`; MLX D/B:
`1.0257847094` und `1.0257859921`. RSS C/A: `0.9999502092` und `1.0007563638`;
RSS D/B: `0.9997158295` und `0.9998923418`. Positions-Peaks:
`A@0/C@3 = 1.0000497933`, `C@0/A@3 = 1.0007563638`.

Classification/Top-Status: `INCONCLUSIVE`, weil weder RSS-Orderflip noch
Core-RSS-Reproduktion eintrat. Der historische B39b-RSS-Wert C/A `3.6524`
reproduzierte sich nicht; Block 0 lag bei `0.9999502`, alle RSS-Peaks bei
ungefähr `7.897–7.914 GB`. Keine Arm-Attribution. B39c bleibt
`valid_for_performance=false` und `activation_allowed=false`, ohne
Timing-Summary, Main-Lauf, Retry, Routing oder Aktivierung. B39d mit zwei
neuen positionsbalancierten Crossover-Blöcken steht nach Clean-State aus;
B39c-Daten werden nicht wiederverwendet oder gepoolt.

## 2026-08-28 — B39d Performance Main abgeschlossen

Nach ProjectAtlas-Orientierung und dem abgeschlossenen B39c-RSS-Diagnostic lief
der autorisierte B39d-Hauptlauf exakt seriell mit den acht Orders
`ABDC/BCAD/CDBA/DACB/DACB/CDBA/BCAD/ABDC`, 32 frischen OS-Children, einem
Model-Load je Child, zwei Warmups und fünf Repeats auf sechs X1-strict Requests
mit `max_tokens=48`. Rohdaten:
`research/raw/B39d_gemma12b_combined_20260828.json`.

Der Returncode war `0`; 8/8 Blöcke und 32/32 Children sind vollständig,
`QUALIFIED`, `valid_for_performance=true`, `activation_allowed=false`. Alle
Correctness-/Canonical-/Workload-/Environment-/Resource-/Crash-/H2-Gates
bestanden. Alle 192 Requests lieferten 48 physische, logische und sichtbare
Tokens, jeweils `stop_reason=length`, mit einem Canonical-Digest. Swap-Deltas
waren überall `0 B`; es gab keine neuen relevanten Crashreports oder
Residualprozesse. Maximale Peaks: MLX `8,002,539,246 B`, RSS `7,916,519,424 B`.
RSS war `PASS` mit global C/A `1.000449911553665` und D/B
`1.0002091397755728`; finales H2 war `ok=true`.

Absolute Endpunkt-Mediane (Wall ns; physisch/sichtbar tok/s), jeweils mit
97.5%-CI:

| arm | wall | rate |
| --- | ---: | ---: |
| A | `11,238,261,187.5 [11,160,058,417; 11,407,090,125]` | `25.6268096092 [25.2474554723; 25.8063165298]` |
| B | `9,804,256,146 [9,746,705,041; 9,953,182,750]` | `29.3751295028 [28.9354679035; 29.5484472741]` |
| C | `10,647,817,688 [10,494,913,166; 10,722,052,334]` | `27.0483488952 [26.8605292185; 27.4418659254]` |
| D | `9,206,717,688 [9,178,958,958; 9,380,620,959]` | `31.2815138465 [30.7015922782; 31.3761071727]` |

Ratios (Wall; Rate), jeweils Median und 97.5%-CI:

| ratio | wall | rate |
| --- | ---: | ---: |
| B/A | `0.8758819112 [0.8513996079; 0.8899192300]` | `1.1417105861 [1.1236974843; 1.1745365992]` |
| C/A | `0.9430849603 [0.9376590283; 0.9482680892]` | `1.0603530881 [1.0545540985; 1.0664857585]` |
| D/A | `0.8194867050 [0.8067160263; 0.8394204565]` | `1.2202787058 [1.1912981061; 1.2395935713]` |
| D/B | `0.9383079941 [0.9222134455; 0.9588925258]` | `1.0657544693 [1.0428697410; 1.0843476690]` |
| D/C | `0.8694078240 [0.8560822753; 0.8852142827]` | `1.1502701579 [1.1296699788; 1.1681120248]` |

Die Headline trennt die Skalen: D/A und D/B bedeuten Wall-Reduktionen von
`18.05%` und `6.17%`, aber Rate-Gewinne von `22.03%` und `6.58%`. Interaktion
`D*A/(B*C)` Median `1.0027137194`, 97.5%-CI
`[0.9619774991; 1.0185403335]`. Epoch-/Order-Drift wurde nicht material;
`order:D/B` und `epoch:contrasts:D/B` bleiben wegen kleiner n als uncertain
markiert und ändern die Hauptklassifikation nicht.

Identität: Model-Digest
`e08dd84591588722a11c43d9ff7ee4b3f50d01f15371c8a4429c4f9857d37fb6`, Code-
Digest `3adaa1bf467b0efd9fa7c06b3da628de5bbadcd3d8d1e3250c462c3c9ff49ce4`,
B39d-Präreg-SHA `f6fcfccc14afb0535cd0d360d0b956cb6e2bb86873e6e5cfdc827784a7d0bd49`;
Python/MLX/mlx-lm `3.12.13/0.32.0/0.31.3`, Apple M1 Max, 32 GiB,
macOS `26.5.2`.

X1s historische `+15.42%` ist Rate-Ratio `1.1542` beziehungsweise äquivalente
Wall-Ratio `0.86640097`. Die Raw-Flags `.8458` nutzten semantisch falsch
`1-.1542` und waren nur nicht-gating Deskriptoren. B39d übertrifft X1 korrekt
auf Rate- und äquivalenter Wall-Skala. Keine Aktivierung, kein Routing und keine
Generalisierung. B39 ist abgeschlossen; als nächster offener Test bleibt B40,
Gemma 12B `max_width` 2/3/4, ohne Wiederverwendung oder Pooling dieser Daten.

## 2026-08-28 — B40 Width-Sweep abgeschlossen

B40 lief nach ProjectAtlas- und Safety-Review mit sechs mirrored Orders
`W2/W3/W4`, `W3/W4/W2`, `W4/W2/W3`, `W4/W2/W3`, `W3/W4/W2`, `W2/W3/W4`.
Alle 18 Children waren frische serielle Prozesse mit einem Model-Load, zwei
Warmups und fünf Repeats auf dem unveränderten Gemma-12B-X1-Workload.
Rohdaten: `research/raw/B40_gemma12b_width_sweep_20260828.json`; das Partial
wurde wegen der inconclusive Klassifikation behalten.

Alle 18 Children beendeten mit Returncode `0`, vollständiger Evidence,
Canonical-/Workload-/Environment-Identität und `no_crash=true`. Alle Outputs
hatten 48 Tokens und Stop-Grund `length`; Swap-Delta war `0 B`, MLX-Peak maximal
`8,002,539,246 B`, RSS-Peak maximal `7,921,287,168 B`, final H2 war grün.
RSS bestand post-run mit global W2/W4 `0.9994507845985368` und W3/W4
`0.9995676763827266`. Keine relevanten neuen Crashreports oder Residualprozesse.

Realized Widths: W2 mean/max `2/2`, W3 `3/3`, W4 `3.971830985915493/4`.
Kandidat/W4, sechs Blockmediane und 10.000 Bootstrap-Resamples:

| Vergleich | Wall median [97.5%-CI] | physisch/sichtbar Rate median [97.5%-CI] |
| --- | ---: | ---: |
| W2/W4 | `1.1033961300051331 [1.0849631490673945; 1.1335189058508615]` | `0.9062977565937032 [0.8823856803711523; 0.9218292054497783]` |
| W3/W4 | `1.040445749841422 [1.022514206934345; 1.0723726405010425]` | `0.9611730621034691 [0.9325739636675945; 0.9779881471327478]` |

W2 und W3 waren in allen sechs Blöcken deskriptiv langsamer als W4 (Wall >1,
Rate <1), beide damit robuste Misses. Materialer Epoch-Drift verhinderte aber
eine Auswahl: W3 `0->5 = 1.0313798935311982`, W4 `1->4 = 0.9721113375197978`
und W4 `2->3 = 1.0226246692862697`. Position-Residuals blieben nahe 1.

Formaler Status: `INCONCLUSIVE`, keine Auswahl (`selected_width=null`),
`valid_for_performance=false`, `activation_allowed=false`. W4 bleibt die
unveränderte operative Baseline; kein Timing wird cherry-gepickt, kein Retry
ausgeführt und keine B40-Aktivierung vorgenommen. B40 ist abgeschlossen. Der
bereits vorhandene architektonische Pfad B3 benötigt eine eigene Freigabe; es
wurde kein neuer Wunsch- oder Aktivierungseintrag ergänzt.

## 2026-08-28 — Pre-push Sandbox-Import-Vorfall

Ein erster versehentlicher Sandbox-Collection-Versuch endete mit Exit `134` und
erzeugte während des MLX-Imports den Crashreport
`Python-2026-08-28-174347.ips` (`SIGABRT`). Der Raw-Report-Zähler fiel netto von
`60` auf `59`, weil gleichzeitig eine Systembereinigung lief; ein reiner
Zählervergleich hätte den neuen Report maskiert. Der korrekte serielle
Non-Integration-Lauf außerhalb der Sandbox bestand mit `284 passed`,
`14 deselected` in `20.58 s` und erzeugte keinen weiteren Report. Der
Xcode-Check endete mit Returncode `0`, `git diff --check` war grün.

Verstärkte Regel: Jeder pytest-Lauf mit MLX-Import muss außerhalb der Sandbox
und strikt mit `-n0` laufen. Parallele xdist-/Sandbox-MLX-Imports sind kein
zulässiger Verifikationspfad. Der Vorfall ändert keine Modell-, Produkt- oder
Aktivierungsentscheidung.

## 2026-08-28 — IronMule-README und GitHub-Auffindbarkeit

Ziel war eine einfach verständliche, suchfreundliche öffentliche Erklärung von
IronMule. Die Arbeit erfolgte im separaten Worktree
`.worktrees/ironmule-qwen-hybrid-cache` auf `codex/qwen-hybrid-cache`; vorhandene
nicht zugehörige Änderungen blieben unangetastet. ProjectAtlas `0.4.5-rc1` wurde
vor dem ersten Repository-Lesen per fokussiertem Session-Brief verwendet. Der
zunächst veraltete Index wurde mit zwei jeweils typisiert angeforderten
`atlas_watch_once`-Läufen aktualisiert.

`README.md` erklärt nun zuerst in einfacher Sprache Problem, Zielgruppe,
Interactive-/Throughput-Entscheidung, Source-Installation und lokalen
Benchmark. Messwerte folgen erst danach, behalten den engen M1-Max-Gültigkeits-
bereich und verlinken E15/E16, Skalierung sowie E10/E12 direkt. Die nicht
existierende PyPI-Veröffentlichung wurde über den offiziellen PyPI-Endpunkt
als HTTP 404 geprüft; der falsche Hauptweg `pip install ironmule` wurde daher
durch einen Checkout-/Editable-Install ersetzt. `pyproject.toml` enthält eine
klarere Paketbeschreibung und fokussierte Suchbegriffe. `CHANGELOG.md` hält den
Abschluss fest; der erledigte Eintrag `DOC1` wurde gemäß Backlog-Regel aus
`docs/BACKLOG.md` entfernt.

Die GitHub-About-Beschreibung wurde live auf
`Run and benchmark local LLMs on Apple Silicon with MLX. Compare latency and
throughput, reuse prompt prefixes, and verify every speed change.` gesetzt und
zurückgelesen. Die 20 Topics wurden auf die Produkt- und Suchabsicht fokussiert;
neu sind `ai-inference`, `large-language-models`, `llm` und `local-ai`, entfernt
wurden `fair-code`, `m1`, `metal` und das zu breite `python`. Das vorhandene
`docs/assets/ironmule-social-preview.jpg` konnte nicht als GitHub Social Preview
gesetzt werden: GraphQL bestätigte `usesCustomOpenGraphImage=false`, GitHub
dokumentiert dafür nur den Settings-Upload, und es war keine angemeldete
Browser-Sitzung verfügbar. Das Asset und der Repository-Inhalt wurden dabei
nicht verändert.

Verifikation ohne Download oder Installation:

- `xcodebuild -checkFirstLaunchStatus`: Exit `0`.
- `ironmule doctor`: Apple M1 Max/arm64, Python `3.12.13`, MLX `0.32.0`,
  MLX-LM `0.31.3`, NumPy `2.5.2` und Metal jeweils `[OK]`.
- Direkter Import-Smoke: `metal_available True`, IronMule-Import erfolgreich.
- `tests/test_cli.py tests/test_benchmark.py tests/test_ironmule_runtime.py`:
  80 Tests seriell mit `-n 0` bestanden. Ein vorheriger, ebenfalls grüner
  xdist-Lauf wird wegen der inzwischen im Journal gefundenen seriellen
  MLX-Verifikationsregel nicht als Gate verwendet; es trat kein Fehler auf.
- README-Linkprüfung: 14 eindeutige lokale Ziele vorhanden; `git diff --check`
  grün.

Es wurden kein Runtimecode und keine Messdaten geändert. Performance-, Speicher-,
Laufzeit-, Genauigkeits- und Qualitäts-Baseline/Nachmessung sind für diese reine
Dokumentations- und Metadatenänderung deshalb nicht anwendbar; es wird kein neuer
Performanceclaim abgeleitet. Es gab keinen Download, keine Installation, keinen
Commit und keinen Push.

### Parallelitätskorrektur und Follow-up-Push

Die Aussage „kein Commit und kein Push“ beschreibt nur den Stand beim Schreiben
des vorigen Absatzes und ist durch eine unmittelbar danach erkannte parallele
Änderung überholt: Ein anderer laufender Prozess hatte den Dokumentationsstand
als `897b789` (`docs: clarify project positioning and quick start`) committet,
auf `origin/codex/qwen-hybrid-cache` gepusht und den gemeinsam genutzten Checkout
anschließend auf `codex/b3-u2-decode` gestellt. Der Default-Branch `main` blieb
bei `f3478e0`. Diese fremde Branch-Umschaltung und die neuen B3-U2-Dateien wurden
nicht verändert.

Zwei danach lokal präzisierte README-Sätze — Benchmark-Correctness statt eines
allgemeinen Runtime-Gates sowie die noch offene Revision-/Quantisierungsbindung
von Profilen — wurden über einen temporären Git-Index als isolierter Folgecommit
`2c616d6` (`docs: tighten public runtime claims`) erzeugt. Vor dem Push wurden
Remote-HEAD `897b789`, Parent `897b789` und der auf `README.md` begrenzte Pfad
geprüft. Der Fast-Forward auf `origin/codex/qwen-hybrid-cache` war erfolgreich;
der aktuelle lokale B3-U2-Branch/Index wurde nicht bewegt.

Der erste Pushversuch änderte den Remote nicht: In zsh wurde die Ref-Syntax mit
einer ungekapselten Variablen (`$commit:refs/...`) falsch expandiert. Ein erster
Retry verwendete zudem einen um ein Zeichen falsch übertragenen Hash und endete
vor jedem Push. Dauerhafte Lösung: Variablen vor einer Ref-Spec immer als
`${commit}:refs/...` klammern und den tatsächlichen Dangling-Commit vor dem
erneuten Versuch per Objektprüfung ermitteln. Der abschließende Push verwendete
den verifizierten vollständigen Hash
`2c616d6ffff8ef6267c63601948309a430630fd5`.

Remote-Readback nach dem Push: Arbeitskopie und Commit-README hatten denselben
SHA-256 `a07c284f47123385033c30d03d85a32eec6a9be22b163bb275c10aed25bbd188`.
Der Branch ist Teil des bereits offenen PR
`#1 research: publish privacy-redacted Apple Silicon benchmark evidence` gegen
`main`. GitHub Actions meldete sowohl für `push` als auch `pull_request` den
macOS-Job `SUCCESS`. Ein Merge in `main` erfolgte nicht, weil der PR zusätzlich
umfangreiche Runtime- und Research-Änderungen enthält und kein separater
Mergeauftrag vorlag.

### 2026-08-28 — B3-U2 Pilot und Review-Limitierung

Der B3-U2-Pilot lief nach den finalen CPU-Gates mit vier ausgeglichenen AB/BA-
Paaren und acht frischen seriellen Children. Der Raw-Datensatz
`research/raw/B3-U2_pilot_20260828.json` hat SHA-256
`7b234b6c3464de4fcf847512243ce83f24d9dfecb9d2747d3ce3a3999e10400a`; die
versiegelte Präregistrierung hat SHA-256
`6edd466a599dec5460f6f5e7e5d89126af70c3774aec7a6c49d19a52d127260b`.

Alle acht Children liefen mit Returncode `0`; die 336 Request-Läufe teilen sich
in 96 Warmup- und 240 Messanfragen. Alle hatten 48 Tokens je Anfrage und 42
vergleichbare State-Hashes je Paar; sie bestanden
Canonical-/Correctness-, Cache-, Swap-, Fallback-, Crash-, Residual- und H2-
Gates. Der Cache hatte nur die zwei registrierten Keys, genau zwei Prime-Misses,
keine weiteren Misses und null Evictions. Maximaler MLX-Peak war
`8,007,886,876 B`, maximaler RSS-Peak `8,314,028,032 B`; es gab keinen
relevanten Python/MLX-Crashreport. Eine unabhängige Review fand jedoch, dass der
Parent keine echten per-Child Pre-/Post-Systemzustände speicherte und
`post_evidence_complete` synthetisch setzte. Daher: `PILOT_SAFE` gemäß Raw,
aber `INCONCLUSIVE_FOR_CONFIRMATION`; keine Bestätigung, kein Retry, kein
Performanceclaim und keine Aktivierung. B3 bleibt aktiv; B3a ist als separat zu
autorisierende prospective Korrektur mit echten Pre-/Post-Feldern und strikten
RID/Event/Digest-Gates im Backlog ergänzt. UI unverändert wegen des bestehenden
SQLite-/Architektur-Blockers. Ein unabhängiger SFA-`.diag`-Change im
Crashreport-Verzeichnis war nicht Python/MLX-relevant.

## 2026-08-28 — Interaktives H0-Signal-Board im ASCII-/Matrix-Stil

Ziel war eine kleine, schnell erfassbare Oberfläche für die bereits vorhandene
H0-Evidenz. Der bestehende freigegebene Architekturvertrag blieb unverändert:
SQLite-v1 wird ausschließlich read-only/query-only geöffnet, der Server bindet
nur an `127.0.0.1`, alle Assets sind lokal und es kamen weder Abhängigkeiten noch
Schreib-, Datei- oder frei parametrisierbare SQL-Pfade hinzu. Der erledigte
Eintrag `U1` wurde nach Abschluss wieder aus `BACKLOG.md` entfernt.

Die echte Quelle enthielt beim Umbau `28` Runs: `15 completed`, `10 invalid` und
`3 worker_exit`; `9` Runs tragen Rohsamples. Das neue Signal Board zeigt fünf
filterabhängige KPI-Karten, Status-Timeline, Ergebnis- und Mode-Balken,
Gate-/Sample-Matrix, eine filter- und durchsuchbare Historie sowie einen
read-only Detail-Drilldown. Der Drilldown gruppiert die maximal `200` bereits
begrenzt ausgelieferten Rohsamples nach Sample-Familie und zeichnet Baseline und
Kandidat getrennt. Snapshot und Graphen werden standardmäßig alle fünf Sekunden
neu gelesen; Live-Polling lässt sich pausieren und manuell aktualisieren.

Vorher-Baseline: `16/16` fokussierte Tests, `9.833 s` unittest beziehungsweise
`10.01 s` real, maximale RSS `39,845,888 B`. HTML/CSS/JS belegten
`1,161/1,520/3,903 B`. Der unveränderte `68,341-B`-Snapshot über `28` Runs hatte
nach fünf Warmups in `30` Wiederholungen Median `10.518542 ms` und
Populationsstreuung `0.238928 ms`.

Nachher: `16/16` Tests, `9.856 s` unittest beziehungsweise `9.97 s` real,
maximale RSS `39,190,528 B`. HTML/CSS/JS belegen nun
`5,680/11,908/20,580 B` und bleiben jeweils deutlich unter dem festen
`98,304-B`-Assetlimit. Drei getrennte Snapshot-Sessions mit jeweils fünf Warmups
und `30` Wiederholungen ergaben Mediane `11.501708/11.475229/11.422146 ms`;
Median der Session-Mediane `11.475229 ms`, Populationsstreuungen
`0.182851–0.260471 ms`. Der beobachtete Vorher-/Nachherunterschied von rund
`+9.1 %` wird nicht als UI- oder Serverregression behauptet: Backend,
Snapshotgröße und SQL-Pfad blieben bytegleich unverändert, Messreihenfolge und
Systemzustand waren nicht randomisiert. Der Wert bleibt als konservativer,
nicht qualifizierter Befund erhalten.

Der Realbrowser-Check zeigte `5` KPI-Karten, `25` Punkte und Tabellenzeilen im
Defaultfenster, keinen Konsolenfehler, kein Error-Overlay und keinen horizontalen
Seitenoverflow. Status `invalid` reduzierte Tabelle und Timeline auf `10`, Mode
`aa_gpu` auf `9` und Suche `confirmation` auf `3` Runs. Der jüngste Drilldown lud
`141` Rohsamples, bot vier Familien und zeichnete für `pair_performance`
`60` Samples als zwei Serien. Pause hielt den beobachteten Snapshot-Zeitstempel
über mehr als fünf Sekunden stabil. Bei `390×844` waren alle Controls sichtbar,
der Hauptbereich `366 px` breit, die Tabelle intern scrollbar und die Seite ohne
horizontalen Overflow.

Verifikation: Python-Compileall, JavaScript-Syntaxcheck, fokussierte Tests,
gezieltes `git diff --check`, `xcodebuild -checkFirstLaunchStatus`, ProjectAtlas
Runtime `0.4.5-rc1` und projektlokale MCP-Konfiguration bestanden. Die bestehende
`.venv` meldete read-only Python `3.12.13`, MLX `0.32.0` und
`Device(gpu, 0)`; es lief keine GPU-Rechnung. Asset-/Test-SHA-256 sind
`17dd47e3b799864c7d885e4ded60d6b073d1d3b59bd96040462efe88fbff0f65` und
`2037ba6993f2e6a78e7ef7fc7d40c2add767691798d33ba4eda30a6296612993`.

Fehler und dauerhafte Lösungen:

- Die vorgesehene `agent-browser`-CLI war nicht vorhanden. Es wurde nichts
  installiert; der vorhandene In-App-Browser übernahm denselben visuellen und
  interaktiven Check.
- Dieser Browser unterstützt für den lokalen Tab kein `networkidle`; der
  verlässliche Pfad ist `domcontentloaded` plus ein begrenztes Render-Wait.
- Label-Locators fanden die verschachtelten Selects nicht. Nach geprüftem DOM
  wurden die vorhandenen eindeutigen IDs verwendet; das ist kein UI-Defekt.
- Globales `python3` hatte erwartungsgemäß kein MLX. Die dokumentierte `.venv`
  war der richtige Interpreter. Das Top-Level-Paket stellt kein `__version__`
  bereit; `importlib.metadata.version('mlx')` ist der belastbare Versionspfad.

Es gab keinen Download, keine Installation, keine Messdatenänderung, keinen
Commit und keinen neuen Performance-, Hardware-, Genauigkeits- oder
Produktclaim. `ProjectAtlas/` blieb unverändert.

### 2026-08-30 — L1 Gemma-Optimizer: read-only Architektur-Audit

Der Audit wurde ProjectAtlas-first im Root gestartet; der aktuelle Backlog-Kontext
wurde vor der Dokumentation gelesen. Eine initiale Aktivität von Claude mit einem
Gemma-12B-Prozess wurde erkannt und die eigene Arbeit wartete deshalb zunächst auf
freie Hardware. AC- und ruhige Last-/Speicher-/Swap-Proben wurden nur beobachtet;
es lief kein Modell- oder Benchmarklauf. ProjectAtlas wurde genau zweimal
inkrementell aktualisiert. Drei getrennte Luna-Audits wurden zusammengeführt.

Das read-only Inventar umfasst `12` SQLite-Dateien, `94` JSON-Dateien unter
`.friday-data`, `85` JSON-Dateien unter `experiments` und `2` vorhandene Profile.
Lokal auflösbar sind nur Gemma `1B` und `4B`; `1B/4B/12B` besitzen verwertbare,
aber getrennt zu haltende Rohdaten, während `27B` nur durch eine unqualifizierte
Zusammenfassung belegt ist. Es gab keine Codeänderung, keinen Modellload, keinen
Benchmark, keinen Download und keine Installation. Der Vorschlag
`docs/L1_GEMMA_OPTIMIZER_ARCHITEKTURVORSCHLAG.md` bleibt **NICHT FREIGEGEBEN**;
Implementierung und reale Hardwarearbeit warten auf eine separate Nutzerfreigabe.

### 2026-08-28 — H0-Dashboard: temporärer `SOURCE ERROR`

Der sichtbare `SOURCE ERROR` kam nicht aus SQLite oder der Snapshot-Projektion:
auf `127.0.0.1:8765` lief kein Listener mehr, daher schlug
`/api/snapshot` mit `connection refused` fehl. Der unveränderte read-only
Loopback-Server wurde erneut mit der gebundenen `.friday-data/h0.sqlite3`
gestartet. Der Endpunkt antwortet danach mit HTTP `200`,
`data_state=available` und `28` Runs. Ein frischer Browsercheck meldet
`errorVisible=false`, `LIVE · UPDATED`, `25` sichtbare Tabellenzeilen und keine
Warnungen oder Konsolenfehler. Kein Code, keine Evidenzdaten und keine
Architekturgrenze wurden geändert; der Browser versucht bei einem temporär
nicht erreichbaren Server weiterhin automatisch alle fünf Sekunden erneut zu
verbinden.

### 2026-08-30 — IronMule-Handover, Q2 und E15-Forkbefund

Der Handover zu Commit `b700377e83b2eba39c5d66976d01332f8ab57bc6` wurde geprüft.
Q2 ist ein realer, einmaliger M1-Max-/Gemma-4B-Engineering-Smoke für Allowlist,
Screening, gepaarte Bestätigung und Profilwrite. Er ist nicht formal versiegelt,
nicht produktiv und kein globaler oder Cross-Device-Claim. In der übergebenen
Evidence fehlen die vollständigen Roh-PIDs, Exitcodes, Kandidaten-Outputs und das
Zweitstart-Log; das gespeicherte Profil trägt den Screening-Wert statt der
Bestätigung. Der Fix `0de69b6` gilt deshalb nur für künftige Läufe.

Der erste E15-Aufruf scheiterte vor dem Modellstart wegen fehlendem `PYTHONPATH`.
Der korrigierte einmalige Lauf dauerte `1664.407 s` und lief mit vier eindeutigen
Prozessen; die vier MLX-Peaks blieben flach bei ungefähr `7.07 GB`. Gegenüber dem
Vorher-Lauf mit `1571.59 s` wird daraus kein Speedclaim: Forking belegt hier
Memory-/Messintegrität, nicht automatisch höhere Geschwindigkeit. B7/R12 bleiben
getrennte, scope- und Provenienzgebundene Befunde; B7s nicht vorab verankerte
Präregistrierung bleibt Protokollschuld.

E15-after-fork SHA-256:
`d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`.
Der aktualisierte content-addressed Archive-Handover weist `371` Einträge und
`156` eindeutige Dateien aus. Es gab keine Aktivierung, Installation, keinen
Download und keinen PR-Merge. Der aktuelle Claude-Worktree bleibt unangetastet.

### 2026-08-30 — Abschluss der heutigen Verifikation

Der b7-Source-Stand blieb auf Commit `b700377e83b2eba39c5d66976d01332f8ab57bc6`
unverändert. Für die E15-/R12-Dokumentation wurden ausschließlich `HANDOVER.md`,
`research/LEDGER.md` und `docs/BACKLOG.md` geändert; der offene R11-Eintrag wurde
entfernt und als Tier-0-/Ledger-Befund dokumentiert.

`Q2_profiles.json` wurde aus dem temporären Scratchpad mit SHA-256
`0a1104b248b4aaf532ee8ef7d9c9c0c06196dde0c5111450ee9386358d15509b` content-
addressed archiviert. E15-after-fork ist ebenfalls unter SHA-256
`d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c` archiviert;
der Archive-Handover weist `371` Einträge und `156` eindeutige Dateien aus.

`python -m ironmule.bench` endete mit Exit 0; die Inert-Warnung wurde als
ausgeübter Branch erkannt und der finale Self-Check war erfolgreich. Der erste
ungefilterte Aufruf `pytest -n0` war ein Operatorfehler: Er sammelte `263` Tests
einschließlich `13` lokaler Model-Integrationen, erreichte `82 %` und wurde nach
dem 30-Minuten-Limit per SIGTERM mit Exit 143 beendet; daraus folgt kein Pass-
oder Fail-Ergebnis. Das dauerhafte gültige Gate ist
`pytest -n0 -m 'not integration'`: `250 passed`, `13 deselected` in `6.22 s`.
ProjectAtlas wurde für beide Roots erfolgreich aktualisiert. Kein weiterer
Modelltest, Download, Installation, Aktivierung oder PR-Merge erfolgte.

### 2026-08-30 — Offline-Implementierung freigegeben

Der fortbestehende Nutzerauftrag gibt die Offline-Implementierung der unabhängigen
`friday_optimizer/`-Control-Plane frei. Die Freigabe umfasst nur lokale Code-,
Datenmodell-, Replay- und Fehlerfallarbeit. Echte vorhandene Daten und
End-to-End-Tests haben Vorrang; synthetische Daten bleiben auf Rand- und
Fehlerfälle beschränkt. Reale Modellläufe bleiben separat gate-basiert: manuell,
AC-only, fremdlastfrei, sparsam und höchstens `30` Minuten. Downloads,
Installationen und automatische Produktaktivierung vor dem Promotionsgate bleiben
gesperrt. Der Claude-Worktree `.worktrees/ironmule-b7` bleibt unangetastet.

### 2026-08-30 — Verifizierter Offline-Optimizer-Stand

Die unabhängige `friday_optimizer/`-Control-Plane ist offline materialisiert und
die Module für Memory, Corpus, Dataset, Bridge, Fingerprint, Candidates, Evaluator,
Readiness/Lease, Session, Profile, History, Orchestrator, IronMule-Adapter,
Dashboard und CLI sind dokumentiert. `optimizer-v2.sqlite3` enthält `401` Records
bei bestätigter Chain/Integrity; der Dataset-Snapshot enthält `392` Records und
bleibt `smoke_only/no_learning_claim` (`train=2`, `val=0`, `holdout=0`).
Deterministische Baselines und Shadow-Auswertung sind implementiert und getestet;
Learned Ranking, GBDT und BO bleiben wegen fehlender Validation/Holdout-Daten offen.
Reale Adapterausführung, Promotion, Hardwareläufe, Downloads und Installationen
bleiben gesperrt. Es gab in diesem Dokumentationsschritt keinen Modell- oder
Hardwarelauf; `.worktrees/ironmule-b7` blieb unangetastet.

## 2026-08-30 — Q2 Seal und Readiness-Blockade

Der vollständige statische Q2-Fingerprint ist an Source-C0
`8b63b7b406bad7b380918ff5c2970fab4b36d5af` gebunden; die Preregistration wurde
mit `a7520b7` versiegelt. Der erste live Readiness-Preflight wurde in zwei
read-only Versuchen wegen `foreign_load`, hoher Last/CPU und instabilem Speicher
abgewiesen. Beide Versuche starteten kein Modell (`model_started=false`),
verbrauchten keine Session und erzeugten keine Performance-Evidence. Die
redigierte Evidence liegt in `READINESS_BLOCKED_20260830.json`; ihr SHA-256 ist
`ae150c04d91629d0abd6a75688e739ce1949e02ecc72380b09b7a5f12c7056ea`.
Die fokussierte RealSession-/Optimizer-Suite bestand mit `100 passed`. Während
der Navigation meldete ProjectAtlas einmal `dependency_closure_limit`; ein
anschließender `watch_once`-Refresh stellte den Index wieder bereit. Keine
Downloads, Installationen, Aktivierungen oder Modell-/GPU-Läufe.

### 2026-08-30 — Readiness-History-Reconciliation

Der read-only Audit identifizierte Seq 402 als kanonischen
`readiness_blocked`-Event und Seq 403 als semantisches Übergabe-Duplikat mit
abweichender Event-ID. Beide binden dieselbe Evidence
`ae150c04d91629d0abd6a75688e739ce1949e02ecc72380b09b7a5f12c7056ea`, denselben
Session-/Fingerprint-/Dataset-/Code-Kontext sowie `model_started=false` und
`session_consumed=false`. Lösung: Seq 402 niemals löschen, Seq 403 ignorieren
und die append-only Korrektur in
`experiments/optimizer_shadow_q2/READINESS_HISTORY_RECONCILIATION_20260830.json`
unter SHA-256
`1f41b229860eacbc54a4ecd4cfa732fec796d519ed02598c31b4b12bca871596`
referenzieren. Die Hashkette blieb intakt; kein Runtime-Impact.

---

# Archiv — vollständiger PROJECT_STATUS.md-Stand bis 2026-08-30

Verschoben am 2026-09-01 (Backlog M1, Punkt 1). Inhalt unverändert übernommen;
der neue `PROJECT_STATUS.md` ist eine Kurzfassung mit Verweis hierher.

# Projektstatus

**Stand:** 25. August 2026, Zyklus 17 nach realer Hardwaremessung
**Zielgerät:** Apple M1 Max, 32 GB Unified Memory, 10-Core CPU, 32-Core GPU

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

## Zyklus 15 — Zwei-Modell-Studie: Vor-Hardware-Vertrag und Ergebnis

Die aktuelle Nutzerfreigabe ist exakt auf die Studie
`dual-model-evidence-planner-20260824-01` begrenzt. Sie erlaubt ausschließlich die
vorregistrierte Prüfung der zwei bereits lokal vorhandenen Modelle in einem festen
Planungsfall und hebt keine früheren NO-GO-Grenzen auf. Es gibt **keine Matmul-On/Off-
Integration**; ein solcher Vergleich wurde weder durchgeführt noch erfunden.

Die Vorregistrierung bindet:

| Schlüssel | Modell | Snapshot-Revision |
| --- | --- | --- |
| `1b` | `mlx-community/gemma-3-1b-it-4bit` | `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` |
| `4b` | `mlx-community/gemma-3-4b-it-4bit` | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |

Zyklus `15` nutzt sechs feste Paare und zwölf frische, serielle Prozesse: Paare
`1–3` laufen `1b → 4b`, Paare `4–6` `4b → 1b`; jedes Modell läuft genau sechsmal.
Der einzige akzeptierte Planerfall ist die JSON-Auswahl
`persistent_service_qualification`. Die Studie bewertet keine allgemeine
Modellqualität, kein Lernen, keinen Code und keine Aktivierung. Der Claim bleibt
`formal_claim=false`.

Der eingefrorene Vor-Hardware-Code ist über die folgenden SHA-256-Werte gebunden:

- Präregistrierung: `246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`
- Worker: `b1db90d306d5de5c6ff466d046c5c617c5dd42cdaee3f6f7b4bcd5bf2a024bc0`
- Harness: `59691f50a1f33d4930b36ccce24ec701af74ebd0f9f095912a75e15a28978470`
- read-only UI: `5db9bf832c17470c0899ee0fd4062b42d524904e1ee3224894e87a7bed049607`

Der staged Diff-Check meldete vor Hardware wegen drei Trailing-Spaces in der
Präregistrierung Exit `2`. Ausschließlich diese Formatzeichen wurden entfernt.
Studienvertrag, Modelle, Zeitplan, Gates, Entscheidungstabelle und
`formal_claim=false` blieben semantisch unverändert; es gab keine
Vertragsänderung und keine Hardwareausführung.

Der finale fokussierte Offline-Stand umfasst `47` Tests und `42` Subtests mit
Exit `0`; auch `py_compile` endete mit Exit `0`. Diese unabhängige Test-Luna-
Verifikation wurde in diesem reinen Dokumentationsschritt nicht erneut ausgeführt.
Der Zyklus-14-Dokumentationsaudit ist auf Commit `ee12bb5` verankert.

### Zyklus 15 — einziger realer Hardwarelauf

Der eingefrorene Vertrag wurde nach dem Preflight genau einmal ausgeführt:
`dual-model-evidence-planner-validation-20260824-01`. Der Lauf bestand aus sechs
balancierten Paaren und zwölf frischen, seriellen Python-Prozessen; Paare `1–3`
liefen `1b → 4b`, Paare `4–6` `4b → 1b`. Kein Prozess lud beide Modelle, jeder
Prozess lud genau einmal. Es gab keine Wiederholung, keinen Download, keine
Installation und keinen Push. Der Lauf endete mit Exit `1`, weil die unveränderte
Entscheidungstabelle `no_planner_qualified` zurückgab; das ist ein gültiges
funktionales Negativergebnis, kein Messabbruch. `formal_claim=false`.

| Messung | Gemma 3 1B 4-bit | Gemma 3 4B 4-bit |
| --- | ---: | ---: |
| strikter Vertrag / Parser / `candidate_id` | `0/6` / `0/6` / `0/6` | `0/6` / `0/6` / `0/6` |
| Determinismus innerhalb des Modells | `6/6` | `6/6` |
| Ausgabe / Abschlussgrund | `32 Token / length` | `23 Token / stop` |
| TTFT Median / MAD | `0,295451312 / 0,0005528535 s` | `0,796846125 / 0,0088023125 s` |
| Modellarbeit Median / MAD | `0,4608839165 / 0,0005743330 s` | `1,0487644165 / 0,0092854165 s` |
| Prozess-Walltime Median / MAD | `4,2468557705 / 0,0059329165 s` | `4,883630417 / 0,0182606455 s` |
| Peak-RSS | `1.937.965.056 B` | `3.765.420.032 B` |
| MLX-Peak | `1.012.548.526 B` | `3.021.085.374 B` |
| Swap-Delta | `0 B` | `0 B` |

Die 1B-Antwort scheiterte wegen Markdown, des falschen Schlüssels
`persistent_service_id` statt `candidate_id` und mehrerer
`<end_of_turn>`-Trailer. Die 4B-Antwort enthielt die richtige ID, aber ebenfalls
einen unerlaubten Markdown-Codeblock. Das sind ausschließlich Vertragsbefunde;
keine allgemeine Qualitätsbewertung. Die dekodierten Texte der beiden Modelle
waren direkt in `0/6` Paaren bytegleich.

Alle Ressourcen-, Budget-, Snapshot-, Pairing- und Fresh-Process-Gates bestanden.
Gemessene Guarddaten: `9,205052 s` Gesamt-Modellarbeit, maximal `1,151402 s`
zusammenhängend, `178,475444 s` Walltime, Duty-Faktor `0,15`, keine Abbrüche,
kein Swap-Wachstum. Die gepaarten Verhältnisse und Bootstrap-KIs sind berechnet,
nicht zusätzliche Messungen: TTFT `0,373014193`
`[0,365603946; 0,377539933]`, Modellarbeit `0,439069434`
`[0,434598134; 0,444460794]`, Walltime `0,872042394`
`[0,864987297; 0,939562889]`, Tokenrate `3,168801108`
`[3,130352029; 3,201472197]`. Daraus folgen berechnet ungefähr `12,8 %`
kürzere 1B-Walltime und `48,5 %` weniger 1B-Peak-RSS. Weil beide Modelle das
Funktions-Gate verfehlten, entsteht daraus keine Präferenz und keine Aktivierung.

Evidenz: [`experiments/dual_model_planner/results.json`](experiments/dual_model_planner/results.json)
SHA-256 `7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`;
private Startmarke `.friday-data/dual-model-planner/attempt.json`, SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`;
Präregistrierung SHA-256
`246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`.
Die Markerdatei liegt in einem Verzeichnis mit Modus `0700` und selbst mit
Modus `0600`. Die lokale read-only UI bestätigte GET/HEAD `200`, schreibende
Methoden `405` und fremde Host-Header `421`; Ergebnis- und Markerhash blieben
unverändert. Zyklus 15 verwendet JSON-Rohdaten, keine eigene Zyklus-15-SQLite-DB.

Es gibt weiterhin keinen echten Gemma-Pfad mit einem Matmul-Optimierungsschalter
„mit/ohne“ und keinen vollständigen Matmul-A/B-Vergleich. Das wurde nicht gemessen
und wird nicht behauptet; es bleibt ein separater, künftig vorregistrierungs-
pflichtiger Kandidat. Ebenso sind allgemeine Modellqualität, allgemeine
Planner-Fähigkeit, selbstlernende Runtime und Produktaktivierung nicht belegt.
Multi-Turn-Fortsetzung und mehrere parallele Requests bleiben ungemessen. Die
Freigabe für den einzelnen Zyklus-15-Lauf ist verbraucht; jeder weitere
Hardwarelauf benötigt eine neue ausdrückliche Freigabe und einen neuen Zyklus.

### Vollständiger Zyklus-15-Preflight

Der finale, rein lokale Preflight wurde ohne Hardware- oder Modellausführung
geschlossen:

- Worker-Selbsttest `17/17`, Exit `0`; Harness-Selbsttest `25/25`, Exit `0`;
- Defaultaufruf ohne Ausführungsfreigabe: Exit `78`; weder private Startmarke noch
  `results.json` wurden erzeugt;
- `py_compile` und `compileall`: jeweils Exit `0`;
- fokussierte Suite: `47` Tests plus `42` Subtests, Exit `0`, Wall `3,36 s`,
  Peak-RSS `60.801.024 B`;
- vollständige `pytest`-Suite: Exit `0`, Wall `45,43 s`, Peak-RSS
  `200.523.776 B`;
- `git diff --check`, AST-Parsing und `xcodebuild -checkFirstLaunchStatus`:
  jeweils Exit `0`;
- ProjectAtlas-Runtime- und Konfigurationsprüfung: jeweils Exit `0`, Runtime
  `0.4.5-rc1`;
- read-only Versions-/Geräteprüfung: MLX `0.32.0`, mlx-lm `0.31.3`, Defaultgerät
  `Device(gpu, 0)`; dies war nur Introspektion und keine GPU-Rechnung;
- read-only Resolverprüfung: 1B-Revision
  `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` mit `732.577.304 B`
  Gewichten sowie 4B-Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2` mit `3.400.569.562 B`
  Gewichten; kein Snapshot wurde geladen;
- die Präregistrierung liegt nach dem Formatfix bei SHA-256
  `246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`;
- ignorierte `__pycache__`-Verzeichnisse sind vorhanden, gehören aber weder zum
  Studienartefakt noch zu einem späteren Commit.

Nach allen Prüfungen blieben
`experiments/dual_model_planner/results.json` und
`.friday-data/dual-model-planner/attempt.json` abwesend. Es gab keine
Hardwarearbeit, keine GPU-Rechnung, keinen Modellload und keinen Commit.

Zum Zeitpunkt des dokumentierten Preflights blieben die Ergebnisse absichtlich
leer: Es existierten noch keine Hardwareevidenz, keine private Startmarke und keine
`results.json`. Der nachfolgende Hardwareabschnitt dokumentiert den einzigen
zulässigen Lauf.

### Zyklus-15-Postflight

Nach dem Hardwarelauf bestand die fokussierte Suite mit `47` Tests und `42`
Subtests bei Exit `0`; die vollständige Suite sammelte `744` Tests und endete
ebenfalls mit Exit `0`. `compileall`, die strikte JSON-Prüfung von Ergebnis-,
Verifikations- und Matrixdatei, `json.tool`, AST-Prüfung,
`git diff --check` und `xcodebuild -checkFirstLaunchStatus` endeten jeweils mit
Exit `0`. ProjectAtlas meldete zunächst `refresh_required`; genau ein
inkrementeller Refresh war anschließend erfolgreich. Runtime `0.4.5-rc1` und die
projektlokale MCP-Konfiguration waren gültig. MLX `0.32.0`, mlx-lm `0.31.3` und
`Device(gpu, 0)` wurden nur read-only geprüft; es gab keine Modell- oder
GPU-Arbeit.

Die Evidenz blieb unverändert: Ergebnis-SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`,
Verifikations-SHA-256
`24696c679de567519e8f2b3b034f0833de8122569072b71feeae794c05bbf4e6`,
Marker-SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`;
alle DB-Hashes blieben unverändert. Die Verifikation meldet leere
Abweichungen, Entscheidung `no_planner_qualified` und `formal_claim=false`.
ProjectAtlas hatte keine getrackten Änderungen; bestehende untracked Fixture-
`.gradle`-Verzeichnisse wurden nicht angefasst.

Vorregistrierte Fehler und erfolgreiche Lösungen, die vor Folgeschritten gelten:

- doppelte Paare/Run-Positionen werden durch explizite Paar- und Schedule-IDs sowie
  Duplicate-Rejection ausgeschlossen;
- unvollständige Läufe können nicht mehr als Erfolg aggregiert werden; partielle
  Rohereignisse bleiben erhalten und der Entscheid bleibt fail-safe;
- per-Run-Content-Hashing wurde aus den gemessenen Zeitfenstern entfernt und als
  einmaliger Parent-Pre-/Post-Manifestcheck außerhalb der Messung umgesetzt;
- Snapshot-Revision, Snapshot-/Gewichts-Hashes und der gerenderte Prompt werden
  vom Parent an den Worker gebunden und vor/nach dem Load geprüft;
- Partial-/Fehlerpfade speichern den Teilstand, brechen terminal ab und erzeugen
  keinen erfolgreichen Claim;
- ein bereits validiertes Worker-Event bleibt auch dann in der partiellen Evidenz
  erhalten, wenn erst die anschließende Ressourcenprüfung terminal abbricht;
- eine feste stdout-Obergrenze von `1.000.000` Byte wird vom Parent überwacht;
- die UI akzeptiert nur die feste Run-ID sowie die geschlossenen Model-,
  Kandidaten- und Decision-Allowlisten und bleibt read-only;
- ein minimaler Fehlerreport ohne `metrics` wird als kontrollierte Fehlerform
  verarbeitet und nicht als erfolgreicher Studienreport aufgewertet;
- Ressourcen- und Budgetprüfungen laufen in fester Reihenfolge vor einer
  erfolgreichen Aggregation; ein Ressourcenabbruch gewinnt vor Korrektheits- und
  Vertragsauswertung.

ProjectAtlas meldete beim initialen Start zunächst `refresh_required`; beim
anschließenden Refresh trat einmalig ein SQLite-Lock auf. Der Retry war erfolgreich.
ProjectAtlas selbst und das eingebundene Upstream-Repository wurden nicht geändert.

Die produktive Research-DB enthält `10` verifizierte `legacy_summary`-Zeilen und
`4` native Ereignisse: drei gültige Berichte mit Rohmessungen sowie einen
sanitisierten Guard-Abbruch. Datei: `118.784 B`, Modus `0600`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision `c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Ein zweiter Import war idempotent (`0` neu, `10` bereits vorhanden) und ließ den
damaligen Dateihash unverändert. Der vollständige Offline-Testlauf nach
Implementierung des Runtime-Prototyps bestand mit `468` Tests und `2.463`
Subtests in `34,58 s` (Wall des umgebenden Prozesses `34,87 s`, User `135,36 s`,
System `3,38 s`, Exit `0`). Der letzte H1-v2-Implementierungsstand davor lag bei
`455` Tests und `2.447` Subtests in `33,04 s`; die H1-v2-Baseline bei `439`
Tests in `32,01 s`.

Die eigenständige Head-Skip-Evidenz liegt read-only in
`.friday-data/head-skip-v1.sqlite3`: `16` hashverkettete Records, genau ein
terminaler Record mit `formal_claim=true`, Modus `0600`, Größe `77.824 B`, SHA-256
`15ee462bbad5a8f757373f093fdf2ccfb8bdd0048c03447c1cb635acd38ec8d9` und
Kettenkopf `8a568e61f0e087794b1997f273e580c72e7f5abaa1eb8bad7954b303dd38a2d4`.
Die UI lieferte ihren realen Verlauf per GET aus; der read-only Replay bestand und
ließ die DB unverändert. Es wurde keine Runtime aktiviert.

Die Zyklus-13-Evidenz liegt in
`experiments/persistent_process/results.json`, SHA-256
`3925d83139cb6278c2b0aa103716e36a33f550f852bd30758976090fa0f7024`.
Die einmalige Startmarke liegt privat unter
`.friday-data/persistent-process/attempt.json`; der Hardwareprozess lief genau
einmal. Die lokale read-only UI zeigte acht Verlaufszeilen und denselben Entscheid.
Es wurde kein persistenter normaler Dienst aktiviert.

Die Zyklus-14-Evidenz liegt in `experiments/planner_4b/results.json`, SHA-256
`64a72331d1a415ae1dac191fecdf9c69cd43f5c11566c2df5ec091cf50a60975`.
Der Hardwarelauf lief am Netzteil genau einmal von Vor-Commit `8067dc6`; die
private Startmarke hat SHA-256
`6e741162f6d02ec69ee74ad7670b8e1a5046a3bc1430b946d7511b1248a6d573`.
Peak-RSS `3.764.961.280` Byte, MLX-Peak `3.021.085.374` Byte und
Swap-Delta `0` hielten alle Gerätegrenzen. Die lokale read-only UI zeigte drei
Verlaufszeilen und denselben negativen Entscheid. Das Ergebnis wurde nicht
wiederholt oder nachträglich gelockert.

### Unabhängiges Audit von Zyklus 14

Das Audit bestätigte die getrennte lokale Provenienz: Vor dem Hardwarelauf lag
`8067dc6c1fb175f0df539394b2e4dad5894b14b8` vor; der unveränderte Ergebnis- und
Dokumentationsabschluss liegt in `8923467c57d61d3599c430687b949052e397a95c`.
Die Präregistrierung blieb bei SHA-256
`0fa346db7985cdd4dfa49015b395ee0f9d56a097a06f3828b0c161c45e53e5ec`, die
Ergebnisdatei bei `64a72331d1a415ae1dac191fecdf9c69cd43f5c11566c2df5ec091cf50a60975`
und die private Startmarke bei
`6e741162f6d02ec69ee74ad7670b8e1a5046a3bc1430b946d7511b1248a6d573`.
Die drei PIDs waren verschieden, jeder Prozess meldete einen Modellload, und
Tokenfolge sowie Rohtext waren `3/3` identisch. Der Rohtext enthielt die richtige
ID, aber einen unerlaubten Markdown-Codeblock; die unveränderte Entscheidung ist
`planner_contract_failed`, `formal_claim=false`.

Unabhängig erneut geprüft wurden die 17 Offline-Tests (Exit `0`), der Worker-
Selbsttest `11/11`, der Harness-Selbsttest `9/9`, AST-/JSON-Parsing,
`git diff --check` (Exit `0`) und `xcodebuild -checkFirstLaunchStatus` (Exit `0`).
Die read-only UI antwortete auf GET mit `200`, auf den Snapshot-GET mit `200`
und auf einen fremden Host mit `421`; die Ergebnis-SHA blieb dabei unverändert.
HEAD wurde nicht als erfolgreich dokumentiert und antwortete im Audit mit `501`.

Offen und absichtlich unverändert bleibt, dass `results.json` keinen
Gewichts-SHA speichert. Der lokal separat verifizierte SHA von
`model.safetensors` ist `94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3`,
ist aber keine nachträgliche Ergänzung der alten Evidenz. Die drei absichtlichen
Markdown-Trailing-Spaces in der versiegelten Präregistrierung werden wegen des
eingefrorenen Hashes nicht entfernt. Es gab keinen erneuten Hardwarelauf und
keinen Security-Check.

Abschlussprüfung am 24.08.2026: ProjectAtlas-Refresh ohne Timeout, Runtime
`0.4.5-rc1`, gültige projektlokale MCP-JSON-Konfiguration,
`xcodebuild -checkFirstLaunchStatus` Exit `0`, MLX-GPU-Gerät Apple M1 Max und
vollständige `.venv/bin/python -m pytest -q`-Suite bei `100 %` mit Exit `0`.

Das Evidenzaudit korrigiert die frühere Statussprache: Der dokumentierte formale
A/A-Loader verlangt global genau sechs kompatible Prozesse, die append-only H0-DB
enthält aber neun `aa_gpu`-Runs aus mehreren Generationen. Mindestens ein relevanter
Prozess war `warmup_unstable`; zudem fehlte allen historischen Runs eine Root-Git-
Revision. Unmittelbar vor dem ersten A/B-Lauf waren hierarchischer Bootstrap,
formales A/A-Gate und MDE noch nicht geschlossen. Spätere gepaarte, replizierte
H1/H2-Zahlen bleiben technisch wertvoll, können aber nicht rückwirkend
vorregistriert werden.

Aktueller Entscheid: Der **begrenzte N8-Runtime-Prototyp** hat seine Gates
bestanden; die genau eine H2-Gemma-Runde ist abgeschlossen. N10-v1 bleibt ein
gültiger terminaler Engineering-Negativlauf ohne Timingdaten und wird nicht
wiederholt. Der separate korrigierte N10-v2-Vertrag band diesen V1-Endzustand,
lief ohne Gemma oder adaptive Auswahl vollständig durch und bestätigte den
festen N10-Dispatch-Plan formal. Der davon getrennte, allowlist-basierte
N10-Runtime-/Runtime-lite-Prototyp hat nun auch sein Cold-Load-/CPU-Gate und sein
gepaartes MLX/GPU-Gate bestanden. Der danach getrennt implementierte N8/N10-
Shadow-Router hat ebenfalls alle vorregistrierten CPU-, reale Tensor-,
Persistenz-, UI- und Sicherheitsgates bestanden. Das ist weiterhin keine
produktive Integration und kein allgemeiner Agenten-, Modell- oder
Hardwareclaim. Die bestehenden N8-/N10-Runtimes blieben unverändert.
Der ausdrücklich freigegebene einzelne statische Custom-Metal-Kandidat wurde
inzwischen isoliert geprüft. Er bestand Correctness, Speicher und A/A, erreichte
gegen `fast_rms_norm` aber nur `1,870 %` statt der vorregistrierten `5 %` und
wurde deshalb nicht promoviert. Produktive Phase 1B, adaptive Kernelsuche und
breiterer Live-Suchraum bleiben **NO-GO**, Cross-Device bleibt **NO-CLAIM** und
weitere Modellrunden bleiben **NO-GO**.
Details: [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md),
Persistenzvertrag: [`docs/H1H2_EVIDENZ_ARCHITEKTUR.md`](docs/H1H2_EVIDENZ_ARCHITEKTUR.md).
Der initiale Auditlauf installierte nichts, lud nichts herunter und führte keinen
GPU- oder Modelllauf aus. Nach späterer ausdrücklicher Rechenfreigabe wurden die
unten dokumentierten lokalen Läufe ausgeführt; auch dabei gab es weder Download
noch Installation.

N10-v1 enthält genau zwei replaybare Records: Präregistrierung
`3233c5ee…8985b` und terminalen Fehler `3ce4477a…92e49`; keine Timing-Session und
kein formaler Claim. Datei: Modus `0600`, SHA-256
`e0b5f4af62c128938e1e12e388c16b344a66e18eebf9e0568c7ebe34c5a4f0d5`,
Snapshot-Revision
`bbc75d60b5cfc61a1037c0a104e117a89561ec63e13240ca9b84f1bc98c08976`.
Der V2-Produktions-Fixture-Test reproduzierte alle vier registrierten Digests.
`22` fokussierte Tests und `10` Subtests bestanden in `40,28 s`; die nach dem
abschließenden Dateihash- und Mutationsgrenzen-Hardening wiederholte
vollständige Suite bestand mit `508` Tests und `2.480` Subtests in `207,82 s`.
V2-Spec-Fingerprint:
`66a01028b5c7ba6cd7b05faef1f3100413d793c6b4d7e3982bea671fb9bba6cd`.
Auf dem sauberen Commit `959df09b9d197edbd0a0984eda25092997b4ab23` band der
Seal die Provenienz `17d0dd505e349a4bbb7ffde3c291a3a44226d0fce79c235ce2ce890289e0c9ef`.
Die sechs A/A-Sessions ergaben `R=0,999586`, 95%-KI
`[0,998764; 1,000443]`; rohe MDE `0,0857 %`, konservativ eingefroren auf `5 %`.
Die sechs A/B-Sessions waren byteidentisch und ergaben insgesamt
`R=0,874912`, 95%-KI `[0,871768; 0,875614]`, entsprechend `12,509 %` weniger
Zeit. Charakterisierung (`R=0,875216`) und Validierung (`R=0,874608`)
bestanden das Gain-Gate getrennt. Der terminale Record `47283e73…e1249`
trägt als einziger `formal_claim=true` und erlaubt
`permit_bounded_n10_runtime_prototype`.

`.friday-data/n10-v2.sqlite3` enthält 16 vollständig replaybare Records, Modus
`0600`, Größe `180.224 B`, SHA-256
`54e9c57ca6b76fa671b94f748b7ee471575b7dd7445bad00ae3cab38f691fc4f`,
Snapshot-Revision
`9c9a94a8f799f2eb29b9e03c4e1b6e681aa945199753158cf8fc8c317b06090d`.
Die UI auf Port 8771 lieferte GET/HEAD `200` und wies POST mit `405` ab; der
vollständige Replay kostet derzeit `3,42–3,44 s` je Snapshot. Ein manueller
`Ctrl-C`-Stop beendet den Server, erzeugt aber noch einen sichtbaren
`KeyboardInterrupt`/Exit `1`. Während N10-v2 lief kein Modell und es gab weder
Download noch Installation.

Der getrennte N10-Runtime-/Runtime-lite-Pfad verwendet Runtime-ID
`n10-runtime-dispatch-20260822-01`, Application-ID `FRN1`, DB
`.friday-data/runtime-n10.sqlite3` und UI-Port `8772`. Der Controller prüft den
exakten formalen DB-Hash und Snapshot, replayt N10-v1 als Vorgänger, vergleicht
die versiegelten N10-Code-/Spec-/Umgebungs-/Hardwarefingerprints, beobachtet
reale Tensoren und cached danach nur die unveränderliche Policy. Unsicherheit
fällt seriell zurück; ein Batchfehler wird nicht wiederholt und verriegelt den
Circuit Breaker. `17` fokussierte Tests und `9` Subtests bestanden in `3,54 s`;
die abschließende vollständige Regression bestand mit `525` Tests und `2.489`
Subtests in `211,66 s`. Der Implementierungsstand wurde lokal auf `main` als
Commit `5eaad38ec0f5da4b01bd9d64237d3736f548ff14` versiegelt; es erfolgte kein
Push. Die saubere Runtime-Provenienz lautet
`02784bd7108767008c9951724421cc3f841390d463a8c6b153b059c5c497e22c`.

Der einmalige CPU-Lauf bestand alle Gates: Cold Load `3,482664083 s`,
Policy-Median `12,372 µs`, p95 `12,448 µs` und zusätzlicher Median
`12,343 µs`. Der danach zulässige einmalige MLX/GPU-Lauf bestand ebenfalls:
zwölf balancierte Blöcke, Baseline-Median `20,797459 ms`, Kandidaten-Median
`18,220750 ms`, `R=0,875753`, Effekt `−12,425 %`, byteidentisch und
`max_abs_error=0`. Die zwei Engineering-Records
`f140083d…89306` und `d6143fca…c979f` sind append-only hashverkettet und
erweitern den formalen Studienclaim nicht.

Die finale Runtime-Datei enthält genau zwei Records, Modus `0600`, Größe
`53.248 B`, SHA-256
`81286ffa2af11a814ffe4e11cdd67ce7fa5804ff42f4efd094cf161dbae22cd5`,
Snapshot-Revision
`a7b9352b913e62b9faf1e59cec2f5531435121d716e08cf2e7f8f24075f6327e`.
Der read-only Replay ließ den Dateihash unverändert. Die echte UI lieferte auf
Port `8772` GET/HEAD `200`, wies POST mit `405` ab und beendete sich per
`Ctrl-C` mit Exit `0` ohne Traceback. Es lief kein Modell; weder Download noch
Installation fanden statt.

## Evidenzgebundener N8/N10-Shadow-Router

### Benennung: warum `avo` in Bezeichnern stehen bleibt

Der Konzeptname wurde am 23.08.2026 durchgängig auf **Runtime-lite** und
**Shadow-Router** geändert. Drei Klassen von Vorkommen blieben bewusst unverändert:

- **Bezeichner in versiegelten Records**: `avo-shadow-router-20260822-01`,
  `avo-router-policy-20260822-01`, `avo-router-shadow-20260822-01`. Sie stehen so in
  `metadata.router_id` und `records.entity_key` der append-only Datei und beschreiben,
  was tatsächlich gespeichert ist.
- **Pfade in `provenance_json.code_files`**: die zehn Dateien unter
  `friday_avo_router/` sowie `tools/run_avo_router.py`. Der Record bindet Pfad *und*
  Inhalt an `code_sha256`; ein Umbenennen ließe die Provenienzprüfung ins Leere laufen
  und der Router fiele dauerhaft seriell zurück.
- **Gebundene Spezifikationen**: `docs/AVO_SHADOW_ROUTER_SPEC.md` steht in
  `spec_files` mit `spec_sha256`, `docs/N10_RUNTIME_PROTOTYPE_SPEC.md` ebenso für die
  N10-Runtime. Auch eine reine Textänderung darin bricht den Hash.

Eine vollständige Umbenennung ist damit kein Suchen-und-Ersetzen, sondern eine neue
Vorregistrierung mit neuer ID, neuer Datei und wiederholten Gates. Sie würde die
bestehenden terminalen Records entwerten und ist **nicht** erfolgt.

Der neue Router verwendet ID `avo-shadow-router-20260822-01`, Application-ID
`FRR1`, DB `.friday-data/avo-router.sqlite3` und UI-Port `8773`. Er lädt die
unveränderten N8-/N10-Policies, autorisiert eine Empfehlung nur, wenn beide
exakten Evidenzpfade autorisieren, und leitet den Scope ausschließlich aus
realen Tensor-Metadaten ab. Der Router besitzt keine `execute`-Methode. Auch bei
einer N8-/N10-Empfehlung bleibt der tatsächlich erzwungene Plan
`serial_shadow_only`; falsche Form, falscher Datentyp und nicht registrierte
Operandenzahl melden auch auf Routerebene `route=serial`.

Die Implementierung wurde lokal auf `main` als Commit
`70bc451f764d36e75de0a1c9ac61849717e577e8` versiegelt; es erfolgte kein Push.
`19` fokussierte Tests bestanden in `0,135 s`. Die vollständige Regression
bestand mit `544` Tests in `210,574 s` (`210,84 s` außen, `199,23 s` User,
`1,47 s` System, maximales RSS `76.496.896 B`, keine Swaps). Der zusätzlich
versiegelte Security-Diff-Scan prüfte zehn Produktionsdateien sowie Tests und
Dokumentation als manuelle Add-backs mit kompletter Coverage und null
reportablen Findings. Sein lesbarer Report hat SHA-256
`0c5a558d908d45b9e6561a5caf90e8bc5d929856ba4e441ecda44ccb282d983b`.

Der einmalige CPU-Lauf bestand alle sechs Gates: Cold Load `7,176239584 s`,
direkter Policy-Median `12,138946 µs` (MAD `0,042467 µs`), Router-Median
`13,719000 µs` (MAD `0,044638 µs`), Router-p95 `13,815279 µs` und gepaarter
zusätzlicher Median `1,585208 µs`. Alle `21` balancierten Blöcke verwendeten
je `10.000` Entscheidungen pro Arm; beide direkten Policies und der Router
stimmten überein. Der terminale Record lautet
`a1a1c1a08eb22c41e442becfe7d6a6a2feb67c2322596eaf1d9fc0a595b253fd`.

Die genau einmal zulässige MLX-Shadow-Validierung bestand danach alle fünf
Gates. Exakt acht beziehungsweise zehn FP16-`2048²`-RHS-Tensorreferenzen
empfahlen N8/N10, während Operandenzahl neun, falsche Form und FP32 seriell
blieben. Direkte Policy und Router stimmten in allen Fällen überein;
`no_matmul_executed=true`. MLX meldete `33.554.432 B` aktiven Speicher,
`33.554.436 B` Peak und `8 B` Cache; maximales Prozess-RSS war `51.560.448 B`.
Der Record lautet
`19e36e7b32209d62afa5eae54973e2dc326a1bd0efaa0d8b8a73737463384c6c`.

Die finale Router-Datei enthält genau diese zwei vollständig replaybaren,
hashverketteten Engineering-Records, Modus `0600`, Größe `36.864 B`, SHA-256
`128c090de37a79606f35c564d19035f0bcffedcea4b4018fa618cffedc58c6f8`
und Snapshot-Revision
`b1c0832c0957e5a2d0e88bda1409f8d4b04be036a5edd32a20ea8c2d57b2c758`.
Die echte UI lieferte GET/HEAD `200`, POST `405`, die vorregistrierten
Security-Header und beendete sich per `Ctrl-C` mit Exit `0`. Ihr Replay ließ
den DB-Hash bytegleich. Es lief kein Modell, keine Matmul und kein Custom-
Metal-Kernel; es gab weder Download noch Installation.

## Phase 1B: statischer Residual-Add-plus-RMSNorm-Kandidat

Der einzelne Kandidat wurde vor jeder Compilation in
[`docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md`](docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md)
gebunden, durch zwei vollständige Security-Diff-Snapshots geprüft und lokal als
Commit `ea8f95980ac6da513c374aa658b4d2d4cc4a9d20` versiegelt. Maßgeblicher
Security-Snapshot:
`codex-security-snapshot/v1:sha256:232c04fe0b4cfbb896120961d91f779b582e04c71b54cbad6bedcaca4c88fa26`;
komplette Coverage, acht Flächen, null Findings. Die vollständige Regression
bestand `566/566` Tests in `212,552 s`. Es erfolgte kein Push.

Die einmalige Qualification bestand sechs vollständige Correctnessfixtures und
alle Vergleiche gegen vier MLX-Baselines. Der maximale Kandidatenfehler war
`0,001953125`, die erste Compilation plus Eval dauerte `193,044 ms`, MLX-Peak
war `89.153.552 B`. Record: `1ff03f1b…56d6e`.

Die drei A/A-Prozesse bestanden mit `R=1,003445`, 95%-KI
`[0,997767; 1,009240]`. Die feste Tie-Regel wählte `fast_rms_norm` aus den zwei
innerhalb `0,5 %` liegenden spezialisierten Baselines. Drei frische A/B-Prozesse
ergaben `R=0,981298`, 95%-KI `[0,972124; 0,985900]`, also rund `1,870 %`
Gewinn. Correctness, jede Einzelsession und Speicher bestanden, aber das
vorregistrierte 5-%-Gate scheiterte. Terminaler Record: `f051b1f8…595a6`;
Status `candidate_inconclusive`, Aktion `baseline_fallback`, kein formaler Claim
und keine Aktivierung.

Die Phase-1B-Historie enthält genau diese zwei Records, Modus `0600`, Größe
`86.016 B`, SHA-256
`4ba0cbd679083683b2504dbf174691402aa851967b88befadeb4035145558452`,
Snapshot-Revision
`45a65df53f27ad8c79cfb6583be3566a1b4bb41f160e68e530bfeec5a1ab031b`.
Die read-only UI auf Port `8774` lieferte GET/HEAD `200`, POST `405`, wies einen
fremden Host mit `421` ab und ließ den DB-Hash unverändert. Es lief kein Modell;
nichts wurde heruntergeladen oder installiert.

## Formales H1-v2-Ergebnis und begrenzte N8-Runtime

Die formale Studie lief vollständig auf dem sauberen Commit
`1fbe73c69cedeb69284a264c5e3f45e3e393b822`. Die Präregistrierung bindet Code,
Spezifikation, Python/MLX-Umgebung und Apple-M1-Max-Hardware; alle zwölf Sessions
liefen am Netzteil in getrennten Prozessen mit realem Inter-Session-Cooldown.

Die sechs A/A-Sessions ergaben ein aggregiertes Verhältnis `1,000109` mit
95%-Intervall `[0,999193; 1,000540]`. Die rohe kalibrierte MDE war rund
`0,0752 %`; prospektiv blieb deshalb der konservative Floor von `5 %` maßgeblich.
Alle vier Kalibrierungsgates bestanden. Die sechs anschließenden A/B-Sessions
waren byte-identisch und ergaben insgesamt `R=0,879718`, 95%-Intervall
`[0,877045; 0,880403]`, Effekt `−12,028 %`. Charakterisierung
(`R=0,879415`) und Validierung (`R=0,880044`) bestanden das Gain-Gate getrennt.
Der terminale Record
`f508fc9e2b1f44a1b60084bdbeca581024f1f3599535b3dd662a9305c99a9357`
trägt als einziger `formal_claim=true` und erlaubt nur
`permit_bounded_runtime_prototype`.

Die formale Datei `.friday-data/h1-v2.sqlite3` enthält `16` vollständig
replaybare Records, Modus `0600`, Größe `163.840 B`, SHA-256
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`.
Ein erneuter read-only Runtime-Preflight ließ diesen Hash unverändert.

Der getrennte Prototyp ist in `friday_runtime/` und
[`docs/RUNTIME_PROTOTYPE_SPEC.md`](docs/RUNTIME_PROTOTYPE_SPEC.md) definiert.
Er autorisiert Batching nur bei exakt derselben terminalen H1-Entscheidung,
unverändertem H1-Code/Spec-Fingerprint, derselben Umgebung/Hardware, sauberem
Worktree und aus tatsächlichen Tensoren abgeleitetem Workload. Alle anderen
Fälle wählen seriell. Ein Batch-Fehler wird nicht im selben Aufruf wiederholt,
sondern verriegelt alle Folgeaufrufe seriell. Im absichtlich schmutzigen
Entwicklungsstand verifizierte der reale Preflight alle `16` H1-Records und fiel
korrekt mit `worktree_dirty` auf seriell zurück. Eine Live-Messung erfolgte vor
dem sauberen Runtime-Commit bewusst noch nicht.

Auf dem anschließend sauberen Commit
`0b0a893f58e9c757a0aa7b49565a8b1c1eb2a561` autorisierte derselbe Preflight den
exakten Scope. Das CPU-Gate (5 Warmups, 21 balancierte Blöcke, je 20.000 Aufrufe)
ergab Policy-Median `11.045 ns`, p95 `11.078 ns` und gepaarten zusätzlichen
Median `11.017 ns`; Record
`a9c08e2b4d79590e1cfa1d5270c53a80a69b1ff1f39507f003fcd6d8d2be1815`.
Alle Grenzen von `25/50/20 µs` bestanden.

Die anschließende MLX/GPU-Validierung (2 Warmup-Paare, 12 balancierte Blöcke)
ergab seriell `20,360 ms`, Runtime-Batch `17,643 ms`, gepaartes
`R=0,879209` und Effekt `−12,079 %`. Die acht Outputs waren byte-identisch,
maximaler absoluter Fehler `0,0`; Circuit Breaker blieb offen. GPU-Arbeit
`0,667252 s`, Wall im Guard `1,059850 s`, maximale kontinuierliche Last
`0,667252 s`, MLX-Peak `209.715.200 B`, RSS-Peak `440.401.920 B`. Record:
`643af8606c83cbcd0a591ba63bebb8745ddf5d4a346971c1d733c8d2b566c2dc`.
Der Policy-Median entspricht rund `0,063 %` der Kandidatenlaufzeit.

`.friday-data/runtime.sqlite3` enthält damit zwei vollständig replaybare,
hashverkettete Records, Modus `0600`, `45.056 B`, SHA-256
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`;
Snapshot-Revision
`a53e6b31c8266b1881ebebfc4dca8c28e9a4177d7648496863fc2b6d4cd6eb3f`.
Read-only UI-Snapshot und H1-Readback änderten keine Datei.

## Geschlossener H2-Gemma-Minimallauf

Auf dem sauberen Dokumentationscommit
`99267d3422f5a8573cad0f53e7009a4cf8f52198` lief genau eine Runde des bereits
implementierten `model-loop`. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` und
der projektlokale Resolver schlossen einen Netzwerkfallback aus; geladen wurde
ausschließlich der vorhandene Snapshot
`mlx-community/gemma-3-4b-it-4bit` Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, eine MLX-Gewichtsdatei mit
`3.400.569.562 B`. Es gab keinen Download und keine Installation.

Die Modellantwort war ausschließlich `[3, 10, 16]`; Parser und Allowlist ließen
genau diese drei Integer-Kandidaten zu. Explorative 20-Block-Messungen:

| Batchgröße | B/A-Ratio | 95%-Intervall |
| ---: | ---: | ---: |
| `3` | `0,849019` | `[0,797567; 0,903789]` |
| `10` | `0,784921` | `[0,741686; 0,830676]` |
| `16` | `0,889566` | `[0,881424; 0,897784]` |

Der Harness wählte `N=10` und bestätigte es separat mit drei Replikaten
`0,6649/0,6716/0,7014`: hierarchisch `R=0,671573`, 95%-Intervall
`[0,648895; 0,731190]`, explorativer Effekt `−32,84 %`. Korrektheitsgates und
5%-Schwelle bestanden. Der Bericht bleibt explizit `formal_claim=false`, weil
das Modell drei Kandidaten aus vorhandener Evidenz auswählte und diese Studie
nicht prospektiv als formale N=10-Bestätigung registriert war. Die produktive
Runtime bleibt deshalb auf `N=8`; `N=10` fällt dort seriell zurück.

Der Guard verbuchte `9,908610 s` GPU-Arbeit, maximal `1,730481 s`
kontinuierlich, `180,024674 s` Kandidaten-Cooldown, `16,022076 s` Pflichtpausen
und `212,268826 s` Wall. Evidenz-ID:
`5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a`.
Die Research-DB enthält nun `14` verifizierte Zeilen, davon `4` native und eine
native `model-loop`-Zeile mit Rohdaten; Modus `0600`, `118.784 B`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision
`c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Der read-only Replay ließ den Hash unverändert. Es wurde keine zweite Runde
gestartet.

## Neue native v1-Exploration nach Rechenfreigabe

Alle folgenden Läufe waren an einen sauberen Root-Commit gebunden, liefen am
Netzteil unter dem gemeinsamen Guard und wurden vor stdout append-only
persistiert. Schema v1 erzwingt weiterhin `formal_claim=false`.

**Einzeloperation vor dem Modelltest:** FP16-Matmul `2048²`, acht Operationen,
drei Replikate mit je 25 gepaarten Blöcken. Batched Dispatch war byte-identisch
und erreichte `R=0,780054`, hierarchisches 95%-Intervall
`[0,765530; 0,877456]`, Effekt `−21,995 %`. GPU-Arbeit `2,803 s`, Wall
`11,468 s`; Evidenz-ID `b866022a…a92eb6`. Das überschreitet explorativ die
fixierte 5%-Schwelle, ist aber noch kein formaler H1-v2-Nachweis.

**Lokaler Gemma-Roofline-Lauf:** Die Werkzeuge lösen ausschließlich validierte
Snapshots im Projektcache auf; Repository-ID, Revision und Gewichtsumfang stehen
im Bericht. Fünf Messwiederholungen je Modell, 369 Prompt-Token:

| native v1 | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Snapshot | `2d44e83d…` | `93724907…` |
| Folge-Token | `5,012 ms` / `199,5 Token/s` | `10,949 ms` / `91,3 Token/s` |
| Prefill | `0,3271 s` / `1.128,0 Token/s` | `0,8735 s` / `422,4 Token/s` |
| geschätzte Bandbreitennutzung | `36,53 %` | `58,47 %` |
| geschätzte FP16-Rechennutzung | `2,78 %` | `4,45 %` |
| exploratives Roofline-Urteil | speicherbegrenzt | speicherbegrenzt |

Gesamtbudget: `10,360 s` GPU-Arbeit, maximal `1,129 s` kontinuierlich,
`52,123 s` verifizierte Pausen und `68,111 s` Wall. Evidenz-ID
`31c20b1e…647c36`, Commit `faa4f88`. Ein erster Versuch auf Commit `29a2b74`
wurde korrekt vor 4B abgebrochen, weil Warmups/Wiederholungen ohne Zwischenpause
die `6-s`-Kontinuierlichkeitsgrenze überschritten; Fehler-ID
`ffe98ffa…1a0ac4`. `pace_generation` und drei Regressionstests schließen diese
Lücke.

Die neue Rohmessung reproduziert die Richtung der historischen Roofline-
Zusammenfassung, wertet sie aber nicht formal auf. **Auf diesem damaligen Stand**
blieb Phase 1B/Custom Metal **NO-GO**. Das zuvor fehlende Schema/Protokoll v2 wurde anschließend unter
`friday_h1/` implementiert, offline verifiziert und auf dem sauberen Commit
`1fbe73c` formal ausgeführt; das Ergebnis steht im vorherigen Abschnitt.

## Historisches Arbeitsprotokoll

Die folgenden Abschnitte erhalten die damaligen Messungen und Entscheidungen. Wo
sie „bestätigt“ sagen, ist das die **historische explorative Klassifikation**, nicht
der aktuelle formale Evidenzgrad.

## Historischer H0-Implementierungsstand

- H0 ist eine einzelne FP16-`2048²`-Matmul-Workload, kein Modelltest und kein
  Self-Optimization- oder Hardware-Generalisation-Nachweis.
- Der Offline-Unterbau umfasst SQLite v1 unter `.friday-data/h0.sqlite3`, festen Worker
  Option A und ein read-only Dashboard auf `127.0.0.1`. Die lokale Datenbank enthält
  `22` Runs einschließlich Run22; Run22 ist eine abgeschlossene H0-eager-baseline
  reference. Das ist kein Modell-, A/A- oder Self-Optimization-Nachweis.
- `run_mlx` und der statische Schalter `mlx-run --execute` sind implementiert. Ohne
  `--execute` endet der Befehl vor Runner-/Worker-/Benchmark-/MLX-Import weiterhin mit
  `EXIT_MLX_LOCKED=78`/`state=not_released`.
- W1v3 wurde in Benchmark, Worker, Runner und Aggregation umgesetzt und offline geprüft:
  äußere Warmup-Blöcke von mindestens `50 ms`, maximal `4096` Evals mit
  `repetition_window_unreachable` bei Nichterreichen, Gate `round(block_ns/evals)`,
  `8..16` Blöcke, ±`5 %` für die letzten fünf Gate-Werte, geschlossene bounded
  Block-Summaries sowie Warmup-Fehlerdiagnose Schema v2 mit v1-Readback. Das tote
  `_Timed.output`-Feld ist entfernt.
- Run22 lief genau einmal. Der Common-Wrapper ist `completed/measurement_complete/
  baseline_fallback` mit `error=null`; gemäß Worker-/Runner-Vertrag sind die verschachtelte
  `benchmark_classification=baseline_reference`, `benchmark_action=not_run` und
  `aggregation_required=false` die erfolgreiche eager-baseline-reference. Die anfängliche
  Operator-Deutung von `baseline_fallback` als Fehler wurde anhand dieses Vertrags korrigiert;
  es liegt kein Produktfehler vor.
- H0-Baseline läuft. Weder `aa_gpu` noch Optimierung oder Self-Optimization sind bewiesen.

## Historischer H0.1-Stand — separate Stationaritätsforschung

- H0.1 ist strikt von H0 getrennt. Es besitzt einen vorregistrierten Paced-Trajectory-
  Vertrag mit exakt sechs vorgesehenen Sessions `C0,V0,C1,V1,C2,V2`, einen
  stdlib-only Analyse-/Study-Core, eine eigene append-only SQLite-v1-Datenbank und ein
  read-only Dashboard. Am 21.08.2026 wurden `6` Paced-Sessions und `1` Paced-Study
  auf dem Zielgerät ausgeführt; der H0.1-Stationaritätsentscheid liegt damit vor und
  lautet `h01_complete_unresolved`.
- Vier historische H0-Generationen wurden vollständig über den öffentlichen H0-
  Bundle-Verifier inventarisiert und durch exakte provenance-/strukturgebundene Adapter
  geschlossen behandelt. A (`runtime_unavailable`) ist eine erkannte Exklusion. B, C
  und D wurden am 20.08.2026 in genau einem atomaren Produktions-Execute als rein
  deskriptive `legacy_h0_warmup_observation`-Bundles importiert. Es gab keine
  H0-Reklassifikation und keine Leistungs- oder Stabilitätsaussage.
- `.friday-data/h01.sqlite3`: `3` verifizierte Legacy-Bundles, Größe `53,248 B`,
  Mode `0600`, SHA-256
  `fd2c6e56d5f108d6670745a338930d6050c38b03eac8cc050170a466818d9d57`.
  Execute-Report-SHA-256:
  `4e73ab2d7b0aa0bf0cb7e559550de254ddadfa41c0f31ee86b92b9203bef788f`;
  H0 blieb bytegleich bei SHA-256
  `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`.
- Das read-only H0.1-Dashboard zeigt Total `3`, Kind
  `legacy_h0_warmup_observation=3`, Status `legacy_observation=3` und Revision
  `d9bc6e5ab430b68e16c9b9dfa62463896c9ad9d64ef003a4a862460378b2af3f`.
  Der finale socketfreie Snapshot lief in `0.04000666690990329 s` bei Peak-RSS
  `29,261,824 B`. Die reale read-only HTML-/API-Grenze wurde danach auf
  `http://127.0.0.1:8766/` mit Exit `0` geprüft; der Server läuft in Session `40690`.
- Finale H0.1-Verifikation: `57/57` Tests und `2,244/2,244` Subtests,
  `0` Failures/Errors/Skips, Wall `25.714773458894342 s`, Self-User/System
  `22.572449/0.502031 s`, Peak-RSS `43,368,448 B`; NumPy-/MLX-Importe und
  Socketkonstruktionen jeweils `0`.
- Forschungsgrenze: Der Legacy-Import ist **GO** als Evidenzmigration. Der
  Sechs-Session-Paced-Vertrag wurde am 21.08.2026 auf dem Zielgerät ausgeführt und
  vollständig replayt; siehe den folgenden Abschnitt. Das war kein Modelltest; es
  wurde kein Modell geladen oder installiert.

## Störprozess aufgeklärt (21.08.2026)

- Der Untergrund, der ungepaarte Messung wertlos macht, ist charakterisiert:
  **unimodal mit langem rechtem Schwanz**, **zufällig verteilt** (Runs-Test über
  sechs Sessions: beobachtet ≈ erwartet), **blockweit** (`22` von `150` Blöcken
  treffen beide Arme, erwartet wären `4,1`) und mit einer **Zeitskala von rund
  `340 ms`** (Autokorrelation `+0,576` bei `68 ms`, `0,000` bei `408 ms`).
- Ein langsam variierender, gerätweiter Prozess — plausibel OS-Scheduling und
  fremde Last. Nicht eliminierbar, aber erklärend: Beide Arme eines Blocks liegen
  in derselben Störungsepisode, weshalb sich die Störung im Quotienten herauskürzt.
- Erklärt vier bisher getrennte Beobachtungen als ein Phänomen: das ungelöste
  H0.1, das zu breite A/A-Bootstrap-Intervall, die nicht funktionierende
  Cutoff-Metrik und die Wertlosigkeit ungepaarter Messung.
- **Neue harte Messregel:** Vergleichsarme innerhalb von rund `340 ms` messen.

## Roofline — die Inferenz ist speicherbegrenzt (21.08.2026)

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Bandbreite genutzt | `31,9 %` | `51,2 %` |
| Rechenwerke genutzt | `2,4 %` | `3,9 %` |
| Prefill je Token schneller | `7,3x` | `5,4x` |

- **Faktor `13`** zwischen beiden Auslastungen, in beiden Modellen `memory_bound`.
  Zwei unabhängige Wege (Auslastungsrechnung und Prefill-Vergleich) sagen dasselbe.
- **Konsequenz:** Code „näher an der Maschinensprache" optimiert den Anteil, der
  mit `2,4`–`3,9 %` ohnehin leerläuft. Wirksam sind nur weniger Bytes
  (Quantisierung — bei 4-bit-Modellen bereits eingelöst) und weniger Durchgänge
  (Kernel-Fusion).
- **Obergrenze:** Bei `51,2 %` Bandbreitenauslastung bringt selbst eine perfekte
  Optimierung ohne Gewichtsverkleinerung höchstens rund `2x`.
- Spitzenwerte `400 GB/s` und `21 TFLOPS` sind Herstellerangaben, nicht gemessen.

## Fusions-Layer — geprüft und verworfen (21.08.2026)

- Ein `mx.compile`-Wrapper über den Forward-Pass zeigte `−12,4 %` (1B) und
  `−15,0 %` (4B) bei bytegleichen Logits. **Praktisch wertlos:** die
  End-to-End-Messung an der echten Generierungsschleife ergibt `−0,5 %` und
  `−0,1 %`.
- Ursache belegt: Die Generierung übergibt bei jedem Aufruf einen KV-Cache
  (`18` Aufrufe mit Cache, `0` ohne), `mx.compile` kann `RotatingKVCache` nicht
  entgegennehmen, und **`mlx-lm` fusioniert bereits selbst**
  (`@partial(mx.compile, shapeless=True)` in `gemma3_text.py` und
  `activations.py`).
- Eigener Messfehler auf dem Weg: `model.__call__` auf der **Instanz** gesetzt —
  Python löst `obj()` über `type(obj).__call__` auf und ignoriert das
  Instanzattribut, der Patch war wirkungslos.
- Wert des Ergebnisses: Ein ganzer Lösungsweg ist **mit Begründung**
  ausgeschlossen. Eine wirksame Layer müsste unterhalb ansetzen — Cache-Layout,
  Speicherverwaltung oder eigene fusionierte Kernel.
- Werkzeuge `tools/measure_roofline.py` und `tools/measure_fusion_layer.py`;
  letzteres misst ausdrücklich den cache-freien Forward-Pass, **nicht** einen
  Generierungsgewinn.

## Historische explorative H2-Codegen-Beobachtung (21.08.2026)

- Nutzerfreigabe für Ausführung modellgenerierten Codes und erhöhtes GPU-Budget
  erteilt. Kein zweites Gerät, Cross-Device bleibt offen.
- **Bestätigt: `R = 0,8838`, `−11,62 %`, `95%-KI [0,8676, 0,8975]`**, drei
  Replikate. Fünf Pläne geschrieben, fünf gemessen, drei über der Schwelle.
- Drei Schutzschichten: heute semantisch begrenzte AST-Plansprache (ein
  Iterationslevel, höchstens `32` statisch gewichtete Matmuls, keine freien
  Allokationsprimitive), Prozessisolation mit Timeout/CPU-Grenze/bereinigter
  Umgebung sowie ein Correctness-Gate. Die MLX-Speichereinstellung ist nur eine
  Richtlinie, kein hartes OS-Limit.
- Zwei Anläufe scheiterten an eigenen Fehlern: Der Prompt zeigte die Baseline zu
  prominent (Modell kopierte sie viermal), und der Validator blockierte
  `out.append(x)` — genau die gesuchte Optimierung. Beides korrigiert und
  getestet.
- Werkzeuge `tools/plan_sandbox.py` und `tools/codegen_loop.py`, in der CLI als
  `codegen`. Gesamtsuite `387` Tests / `2.377` Subtests grün.
- Grenze: Der Plan bleibt eine Umsortierung derselben festen Rechnung. Keine
  Kernel, keine Algorithmenwahl, keine Numerikänderung — die Allowlist lässt das
  nicht zu.

## Vollständiger Testlauf und Selbstoptimierung (21.08.2026)

- **Testsuite `90 s` → `31 s`** (Faktor `2,9`) über `pytest-xdist`, in `pytest.ini`
  festgelegt. Untere Schranke ist ein `17,6 s`-Test mit `16` vollen Bootstraps;
  mehr Worker bringen nichts (Amdahl). Das Bootstrap zu beschleunigen wurde
  **verworfen**, weil `friday_h0/aggregation.py` in der geschlossenen Code-Liste
  steht und eine Änderung die Provenienz aller H0-Läufe brechen würde.
- **Sicherheitsfund: `aa` besaß kein `--execute`-Gate.** Der Prüfaufruf startete
  real einen A/A-Lauf; nur der Resume-Mechanismus verhinderte eine Aufzeichnung.
  Gate nachgerüstet, alle vier messenden Werkzeuge gesperrt. `ReleaseGateTest`
  prüft jetzt, dass jedes registrierte Werkzeug einer Gruppe zugeordnet ist —
  ein neues Werkzeug ohne Einordnung lässt die Suite fehlschlagen.
- **Entdoppelung:** `require_ac_power` (vier Kopien) und das Release-Gate (drei
  Kopien) liegen jetzt in `tools/_bench.py`. Dateiübergreifender Duplikatscan
  findet nichts mehr.
- `aa` hat jetzt auch `--self-check`; das README-Versprechen gilt damit für alle
  vier Werkzeuge.
- **`337` Tests / `2.322` Subtests grün in `31,3 s`**, Guard `pass`, Provenienz
  beider Phasen unverändert, End-to-End nach dem Refactoring bestätigt.

## Einsatzreife für Dritte (21.08.2026)

- **Einziger Einstieg `tools/friday.py`:** `list`, `doctor` und Durchreichen an die
  fünf Werkzeuge. `doctor` prüft Python, MLX/Metal, NumPy, `mlx-lm`, Netzbetrieb
  und Plattenplatz.
- **`docs/ERGEBNISSE.md`** fasst alle Befunde, sieben Nullbefunde, Grenzen und
  sechs Messregeln auf einer Seite zusammen; jeder Befund nennt sein
  Reproduktionskommando. Ersetzt für Einsteiger die `2.540` Journalzeilen.
- **README neu:** Kernbefund im ersten Absatz (ungepaart messen ist hier nahezu
  wertlos, Beispiel `mx.compile`), Schnellstart, Werkzeuge, Messregeln, Budgets,
  Grenzen.
- Keine absoluten Pfade im Code; `requirements-apple-silicon.txt` führt `mlx-lm`
  als optionale Zeile mit Dry-Run-Hinweis.
- `tests/test_friday_cli.py` (`12` Tests, `34` Subtests) prüft Werkzeugregistrierung,
  Release-Gates, Self-Checks und dass jeder Dokumentationslink auflöst.
- Gesamtsuite `326` Tests / `2.312` Subtests grün, H0.1-Guard `pass`.

## Historische explorative Self-Optimization-Loop-Beobachtung (21.08.2026)

- **`3` von `3` Läufen: `optimization_confirmed`.** Gewählt `N=8` (`−13,60 %`),
  `N=6` (`−11,13 %`), `N=6` (`−14,11 %`). Der Loop konvergiert auf `N=6`–`8`.
- **Autonomer Fund:** `N=6` und `N=7` kamen in der manuellen Suche nicht vor
  (dort nur `2,4,8,16`). Der Loop schlug sie selbst vor, maß sie und bestätigte
  einen davon unabhängig.
- Drei Runden: `explore` über eine feste Kandidatenmenge, `refine` mit selbst
  vorgeschlagenen Nachbarn, `confirm` mit `3` Replikaten und hierarchischem
  Bootstrap. Correctness-Gate vor jeder Zeitmessung: bytegleich oder verworfen.
- **Erst nicht reproduzierbar (`1/3`), Ursache Winner's Curse.** Rangfolge nach
  dem Punktschätzer wählte den glücklichsten Ausreißer (`0,750`, `0,741`), der bei
  unabhängiger Nachmessung auf `0,87`–`0,96` regressierte. Korrigiert auf
  Rangfolge nach Konfidenzobergrenze; die Schwelle `MDE = 5 %` blieb unverändert.
- Werkzeug `tools/optimization_loop.py` (`--execute`-Gate, Netzbetrieb, GPU- und
  Wall-Budget, `--self-check`), Tests `tests/test_optimization_loop.py` (`15`).
- Grenze: fester, von Hand definierter Suchraum. Kein Codegenerieren, keine
  Kernel — das bleibt H2 mit eigener Sicherheitsfreigabe.

## Historische explorative H2-Vorstufe — Modelltests Gemma 3 (21.08.2026)

- Nutzerfreigabe für Download und Installation erteilt, Auflage Projektordner
  eingehalten: `HF_HOME` auf `.friday-data/models`, Pakete im lokalen `.venv`.
  Belegt sind `3,9 GB`; `16 GB` bleiben frei.
- **Provenienz ungebrochen.** `uv pip install mlx-lm` zog `24` Pakete, ließ aber
  `mlx 0.32.0` und `numpy 2.5.2` unverändert. `environment_sha256` bleibt
  `74ca2dac…`, `code_sha256` H0 `101cdadf…` und H0.1 `f66e4b5a…` ebenfalls.
  Alle früheren Läufe bleiben vergleichbar.
- **Stufe 1, `gemma-3-1b-it-4bit`:** `737 MB`, TTFT `205 ms` ohne Pause,
  Folge-Token `5,0 ms` (rund `200 Token/s`).
- **Stufe 2, `gemma-3-4b-it-4bit`:** `3,40 GB` auf Disk, davon werden nur
  `2.560,8 MB` geladen. Folge-Token `11`–`12,8 ms` (rund `85 Token/s`),
  TTFT `304,8 ms` ohne Pause.
- **Widerlegte Annahme: es gibt keinen Vision-Tower-Offset im Speicher.** Der
  SigLIP-Tower (`833,7 MB`) und der Projektor (`5,9 MB`) liegen im Repo, werden von
  `mlx_lm.load` aber **nicht geladen**. Die früher geplante Bestimmung über die
  Peak-RSS-Differenz beider Stufen ist damit gegenstandslos; der Anteil wurde direkt
  aus dem safetensors-Index quantifiziert.
- **Cooldown-Effekt bei 4B bestätigt:** `R = 1,414` (`+41 %`),
  `95%-KI [1,209, 1,653]`, `10` Paare im direkten Wechsel. Betrifft nur die TTFT.
- **Der Effekt skaliert nicht mit der Modellgröße:** `1B` `1,37x`, `4B` `1,414x`.
  Trotz vierfacher Parameterzahl praktisch gleich — er ist eine Eigenschaft des
  Geräts, nicht der Arbeitslast, konsistent mit dem Matmul-Befund ohne jedes Modell.
- Zwei eigene Zwischenzahlen wurden durch sorgfältigere Messung nach unten
  korrigiert (`503,9 ms` bei 1B, `4,16x` bei 4B). Abgeleitete Regel: kein Befund
  aus weniger als zehn Wiederholungen, und Behandlungsarme im direkten Wechsel statt
  frei randomisiert, wenn die Behandlung eine Zeitkomponente hat.

## Cooldown-Effekt — isoliert und erklärt (21.08.2026)

- **Dosis-Wirkungs-Beziehung nachgewiesen.** Das erste Sample nach einer Pause ist
  verlangsamt, monoton mit der Pausenlänge: `0,94x` bei `0 s`, `1,89x` bei
  `0,25 s`, `3,67x` bei `2 s`, `4,12x` bei `20 s`. Sättigung bei rund `4x` ab
  etwa `2 s`. Ohne Pause ist der Exzess exakt `0,00`.
- **Ursache überwiegend GPU-Taktung.** Eine Idle-Pause von `5 s` ergibt `R = 4,02`;
  dieselbe Pause mit periodischer Mini-Matmul nur `R = 2,53` (Verhältnis `0,487`,
  `95%-KI [0,311, 0,762]`). Der MLX-Allocator scheidet aus: der Cache bleibt über
  die Pause konstant bei `8,6 MB`.
- **Keep-Alive ist keine brauchbare Optimierung.** Sieben Dosierungen gemessen; die
  beste senkt das erste Sample von `10,05 ms` auf `5,11 ms`, kostet aber `14,44 ms`
  eigene GPU-Zeit. Netto `−9,50 ms`. Alle Varianten netto negativ.
- **Der Effekt erklärt H0.1 nicht.** Post-hoc deskriptiv: ohne die ersten sechs
  Main-Samples bestünde `trend` `1/6`, `changepoint` `0/6`, `tail` `0/6` — identisch
  zum realen Ergebnis. Die H0.1-Instabilität stammt von über die Session verteilten
  Ausreißern. Zwei unabhängige Phänomene. Study unverändert
  `h01_complete_unresolved`.
- Praktische Konsequenz für künftige Messungen: Nach einer Pause gehen bis zu
  `5,12` Sample-Äquivalente verloren. Wer nach einer Pause misst, ohne den Anlauf
  zu verwerfen, verzerrt ein 80-Sample-Mittel um bis zu `5,8 %` — mehr als die
  H1-Nachweisschwelle von `5 %`.
- Werkzeuge: `tools/measure_cooldown_effect.py` (`--execute`-Gate, Netzbetrieb,
  GPU-/Wall-Budget, `--self-check`) und `tests/test_cooldown_effect.py` (`15` Tests).

## Historische explorative H1-Beobachtung — Dispatch-Plan (21.08.2026)

- **Ergebnis: `−14,7 %` bei `N = 8`, Optimum `−17,4 %` bei `N = 4`.** Gepaart
  gemessen, über fünf Replikate repliziert, `95%-KI [0,8263, 0,8777]`, Correctness
  `byte_identical`. Verdikt `effect_confirmed` gegen die vorab eingefrorene
  Schwelle `MDE = 5 %`.
- Der Kandidat ist eine **Ausführungsplanänderung**, keine Kernel-Optimierung:
  `N` Matmuls mit einer einzigen Synchronisation statt `N` einzelner
  Synchronisationen. Identische Arithmetik, bytegleiche Ergebnisse.
- `serial 2,572 ms/Matmul` gegen `batched 2,212 ms/Matmul`; Ersparnis `0,360 ms`
  je Matmul. GPU-Arbeit `5,8 s` gegen Budget `120 s`.
- **Ohne die A/A-Vorarbeit wäre das Ergebnis falsch gewesen.** Ungepaart erschien
  `mx.compile` mit `−27,6 %`; gepaart ergibt derselbe Kandidat `R = 1,0019`,
  `KI [0,9990, 1,0047]` — kein Effekt. Der scheinbare Gewinn war reines Rauschen.
- Ausgeschlossene Fehlerquellen: Deduplizierung identischer Teilausdrücke (alle
  Messungen mit paarweise verschiedenen Operanden), Ergebnis-Caching, veränderte
  Arithmetik.
- Geprüfte Nullbefunde: prätransponiertes `B` `+3,1 %`, `mx.einsum` `+0,6 %`,
  eigener GPU-Stream `±0 %`, echter 3D-Batch-Matmul `−3,9 %`/`−1,8 %` mit
  `1,0` im Konfidenzintervall. Die Optimierung ist damit ausgereizt.
- Reichweite: gilt für **unabhängige** Operationen. Keine Aussage über Modelle,
  Transformer-Inferenz oder andere Geräte. H0 und H0.1 bleiben unverändert.
- Werkzeuge: `tools/measure_dispatch_plan.py` (`--execute`-Gate, Netzbetriebspflicht,
  GPU-Budget, Correctness-Gate, `--self-check`) und `tests/test_dispatch_plan.py`
  (`12` Tests, offline).

## H0.1 — ausgeführte Sechs-Session-Paced-Study (21.08.2026)

- Nutzerfreigabe: ausdrückliche Freigabe für den GPU-Lauf. Keine Installation, kein
  Download, kein Modell. H0.1 misst die feste `2048²`-FP16-Matmul, nicht ein Modell.
- Zuvor fehlte der Ausführungspfad vollständig: `build_trace` nahm bereits
  aufgezeichnete `durations_ns` entgegen, aber niemand erzeugte sie. Neu sind
  `friday_h01/provenance.py`, `friday_h01/runner.py`, `friday_h01/cli.py` und
  `tests/test_h01_runner.py`. Der GPU-Pfad liegt hinter demselben
  `--execute`-Release-Gate wie H0 (`state=not_released`, Exit `78`).
- Preflight: `preflight_ok`, Parent Run22, `fixture_seed=4051312678`, Netzbetrieb
  `ac_power`, `thermal_state=api_unavailable`, Proben `24.60/4.80/3.86 ms`.
  Eingefrorener `code_sha256=f66e4b5a2444643fb375a098398bbd3829d717a7b956e62f46a6a54617986e94`.
- Lauf: sechs getrennte Prozesse `C0,V0,C1,V1,C2,V2`, alle `h01_session_complete`,
  Wall je Session `66.00`–`66.37 s`.
- Gate-Bilanz: `changepoint` 6/6 fail, `tail` 6/6 fail, `trend` 5/6 fail,
  `pacing` 5/6 fail, `ess` 1/6 fail, `acf` 0/6 fail; Summe `23`.
- Study: `h01-study-1812a894…c39ca`, `status=h01_complete_unresolved`,
  `conclusion=replicated_stationarity_not_supported`, `session_count=6`,
  `failed_gate_count=23`, `action=no_h0_conclusion`, `h0_reclassification=false`,
  `promotion_applicable=false`.
- **Ergebnis: replizierte Stationarität ist nicht unterstützt.** Das ist ein gültiges
  negatives Ergebnis, kein fehlgeschlagener Lauf. Der Envelope wird deutlich und nicht
  knapp verfehlt. Dominierend ist die Tail-Ratio `2,53`–`3,13` bei Grenze `1,20`;
  `acf` besteht überall und das Trendvorzeichen wechselt zwischen Sessions, die Daten
  sind also von sporadischen Ausreißern geprägt und nicht von gerichtetem Drift.
- Beobachtung am gleichen Messpunkt, ohne Ursachenbehauptung: Run22 misst rund
  `2,07 ms` je Matmul innerhalb eines dichten 32er-Batches, H0.1 misst dieselbe
  Operation einzeln mit `50 ms`/`750 ms` Pacing bei einem Median um `6,97 ms`. Die
  Ursache ist nicht gemessen. Festhaltbar ist nur, dass die H0-Baseline einen dicht
  gepackten Batch-Zustand charakterisiert und nicht die isoliert gepacte Einzeloperation.
- Verifikation: alle sechs Sessions und die Study unabhängig neu berechnet und
  bytegleich; Gesamtsuite `266` Tests / `2.265` Subtests grün; Dashboard-Snapshot
  Total `10` (`paced_session=6`, `paced_study=1`, `legacy_h0_warmup_observation=3`),
  Revision `a2d1b2469e21f01de04e03b747ac897bb602059d5ec5ceeb098dcba5b03b4e1b`.
- Betriebsrisiko: Eine aufgezeichnete Session ist nicht wiederholbar. Die `run_id` ist
  deterministisch aus Provenienz abgeleitet, die Messdaten sind es nicht; der
  append-only Store antwortet mit `StorageConflict`. Ein Abbruch mitten in der Study
  macht die bereits aufgezeichneten Sessions dieser Provenienz unbrauchbar. Ein
  Wiederholungslauf ist eine neue Study, kein Patch.
- Grenze: H0 bleibt unverändert. Keine Reklassifikation, keine Promotion, keine
  Performanceaussage, kein Nachweis von Self-Optimization oder Generalisation.

## Run22 — abgeschlossener eager-baseline-Reference-Lauf (20.08.2026)

- Nutzerfreigabe und Umfang: begrenzte W1v3-/Output-Fix-Umsetzung plus genau ein
  `eager_baseline`-Canary; kein `aa_gpu`, keine Installation und kein Retry. Der einzige
  Live-Befehl lief mit dem freigegebenen `eager_baseline`-Pfad. Run-ID:
  `h0-eager_baseline-characterization-0-14d435dcc2170feec70d8baaa712860e59a6148ca3f211aad98eff1c9d7cf0ff`.
- Ergebnis: äußerer `real=3.79 s`, Exit `10`, DB danach `22` Runs. Der Common-Wrapper ist
  `completed/measurement_complete/baseline_fallback`, `error=null`. Der verschachtelte
  Vertrag meldet `benchmark_classification=baseline_reference`,
  `benchmark_action=not_run`, `aggregation_required=false`; damit ist der eager-baseline
  Reference-Lauf erfolgreich abgeschlossen. Die anfängliche Operator-Deutung von
  `baseline_fallback` als Fail wurde anhand des Worker-/Runner-Vertrags korrigiert und ist
  kein Produktfehler.
- Warmup und Baseline: `8` stabile Warmups; Gate-Werte
  `[2566556,2179783,2188775,2143891,2155069,2174895,2195533,2192185]`, Median der
  letzten fünf `2174895 ns`; `30` Messblöcke mit jeweils `32` Reps; Calibration
  `68155792 ns`; Baseline-Median `2138574.859375 ns`, MAD `17041.671875 ns`, IQR
  `35343.0859375 ns`, Minimum `2105915.34375 ns`, Maximum `2210087.25 ns`.
- Correctness: Gate bestanden; `9/9` Cases und `86/86` Metrics. `abs_max=
  0.0310508173`, `normalized_l2=0.0002074681`, `abs_q99=0.0110023008`,
  `rel_q99_abs_oracle_ge_1=0.0004333980`.
- Speicher: active `16777216 B`, peak `25165824 B`, cache `8422698 B`, RSS-Peak
  `369655808 B`. Der Memory-Gate ist `not_evaluable_missing_required_metric`,
  `hard_limit=false`. Retention-Nachprobe nach dem Fix: `67108864 B` erzeugt,
  `0` Payloads/`0 B` retained; `_Timed` enthält nur `duration_ns`, `evaluation_ns`
  und `synchronize_ns`.
- Freeze und Artefakte: Code
  `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`, Spec
  `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; kein Git-Root.
  Manifest `73058165244fe505035182f0044dc5ab8bd16ef523ebfc44b44d5b6f616e239e`, Result
  `bda3d23d56e49c2d26bf7c3e73d52b61c3ea022c3fb61ab0719bfedef58a6d09`, Evidence
  `edaf6cae5a98185f183fd368189a8be3a56c194540e4f64300903cff42d1a6a0`, Projection
  `a51aa4b3cadf00dc5338eee199206b2b8f876c4fd3aeaaa2d5261364254ed790`, Bundle
  `a566c912032efab919dddf5ca7f67b986f29464a655abf15617733aeb6947c49`.
- Read-only Dashboard-Snapshot: `snapshot_id=325afcc9a45311ba716f64a51e7395cd7f2cf1c872c9a3f349c6daf9361398de`,
  `source_revision=7cdad7edcb6099894d588bb9927de322bd4f7ce02d256673768647db54131c73`,
  `run_count=22`, `completed`; der Dashboard-Socket war frei.
- Nach Abschluss wurde der read-only Dashboard-Server erneut auf
  `http://127.0.0.1:8765/` gestartet: `state=serving`, Session `4414`. Die integrierte
  Browseransicht traf vor dem Serverstart zunächst `connection refused` und blockierte
  danach den lokalen Reload per Browser-URL-Policy. Es gab keine Umgehung und keine
  Datenänderung; der socketfreie Snapshot bleibt die verifizierte UI-Evidenz, der direkte
  lokale Link ist verfügbar.
- Umgebung und Verifikation: voller einmaliger Pytest-Lauf Exit `0`, Wall `66.837 s`,
  `228 passed`, `2211` Subtests in `66.24 s`. Ein engerer Unittest-Lauf wurde nach
  `30.018 s` mit Exit `124` absichtlich gestoppt, nachdem `103` Marker grün und keine
  Fehler/Fehlschläge sichtbar waren; er wird nicht als Fehler verschwiegen. CLI-Lock:
  Exit `78`; Usage-Fehler: Exit `64`; `xcodebuild -checkFirstLaunchStatus`: Exit `0`.
  ProjectAtlas `0.4.5-rc1`; Python `3.12.13`, NumPy `2.5.2`, MLX `0.32.0`, macOS
  `26.5.2 arm64`. Der sandboxed Import meldete kein Metal-Gerät; es lief dabei keine
  MLX-Operation.
- Vorher/Nachher: vorher `206 passed + 47 subtests`, DB `21`, Retention `64` Payloads /
  `67108864 B` lebend, Code `aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c`,
  Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; nachher die
  Freeze- und Testwerte dieses Run22-Abschnitts.

Dieser eine Baseline-Lauf ist keine vergleichende Performanceaussage. H0-Baseline ist
damit ausführbar und referenziert; A/A, Optimierung und Self-Optimization sind nicht
bewiesen. Weitere Ausführung benötigt eine neue ausdrückliche Freigabe.

## Historischer Live-Pfad-/Preflight-Stand vor Run22 — 20.08.2026

Die folgenden Abschnitte dokumentieren frühere Preflight-, Run20- und Run21-Stände.
Sie sind historisch und nicht der aktuelle Projektstatus.

- Live-Pfad `45/45 OK`: Wall `4.022908 s`, User/System `3.149974/0.200018 s`, Peak-RSS
  `42,139,648 B`; keine Self-/Child-Aufteilung belegt. Nach dem Fix auf die reale
  `get_cache_memory`-API: `16/16 OK`, Wall `0.086906 s`, User/System
  `0.140900/0.054489 s`, Peak-RSS `49,938,432 B`, ebenfalls ohne belegte Aufteilung.
- Aktuelle Nicht-Live-Suite: `133/133`, Wall `23.720160 s`, User/System
  `22.722187/0.559409 s`, Self-/Child-Peak-RSS `71,368,704/23,642,112 B`;
  unabhängiger Replay `133/133`, Wall `23.588426 s`, User/System
  `22.769535/0.504137 s`, Self-/Child-Peak-RSS `60,342,272/23,707,648 B`. Ein echter
  Importguard belegte, dass dabei keine MLX-Matmul-/GPU-Workload lief.
- Socketfreies Dashboard: `4/4` plus `3` Setup-Subtests, Wall `0.001793 s`, User/System
  `0.001437/0.000137 s`, Self-/Child-Peak-RSS `31,457,280/0 B`, null Socketaufrufe.
  Historisch bestand eine autorisierte HTTP-Prüfung `13/13`; spätere Sandbox-Bindefehler
  und der nicht wiederholte `16`-er Scope sind weder Produktfehler noch finaler Grünnachweis.
- Der Sandbox-Preflight hatte kein Metal. Der danach autorisierte Zielgeräte-Smoke
  bestätigte MLX `0.32.0` und eine 1-Element-Operation in Tool-Wall `1.741108708 s`;
  das war keine Matmul und kein H0-Ergebnis. Eine spätere reine API-Prüfung bestätigte
  `get_cache_memory`, ohne die API oder eine GPU-Workload aufzurufen.
- Canary: äußerer Wall `0.166578416 s`; Child User/System `0.106607/0.040468 s`, Child-
  Peak-RSS `28,442,624 B`, gespeicherter Worker-RSS `23,150,592 B`. Äußere Self-User/
  System/RSS wurden nicht separat gemessen. Ergebnis
  `invalid/runtime_unavailable/baseline_fallback`, Fehler
  `NumPy import unavailable: ModuleNotFoundError`: `0` Rohsamples, `0` Correctness-
  Zeilen, `3` Supervisor-Scalars und `1` Projection-Artifact; Performance, Ratio/KI,
  Warmup/Repetitions und MLX active/peak/cache fehlen. Kein `aa_gpu`, keine Promotion.
- Canary-Hashes: Code `246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b`,
  Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`, Manifest
  `11ac87fb704169e58ac506eda5d0549a91ad19e8ff52b43c5bb7f28e61d982c1`, Result
  `cb97e223fd26c87aa1f1e3a87e56b4c61c76c5b69e7d0420721392727e31aa02`, Evidence
  `406c42b4a99f72703b9623fd8ba5e5c0e68c46495f5a7bd0db1cef1674e0499d`, Projection
  `d9071855d3b1dc6318aa8c832c66c368314ef9ce4ff790911dd4a96939fdaf24`, Bundle
  `1de0c11763c38462420bb74277d8018b2db1517f9eb17e234938b27681a8c41b`.
- Ursache: `Path(sys.executable).resolve()` kollabiert den lexikalischen Launcher
  `/Users/tobiasburandt/Project_Friday/.venv/bin/python` zum Basisinterpreter
  `/opt/homebrew/Cellar/python@3.12/3.12.13_2/Frameworks/Python.framework/Versions/3.12/bin/python3.12`;
  dadurch fehlt in der bereinigten Worker-Umgebung die venv-Paketsuche.
- Minimaler Vorschlag: den fest erwarteten absoluten, aber lexikalischen venv-Launcher an
  `Popen` übergeben und Launcher, Parent und Ziel vor/nach Spawn eng über Owner, Modus,
  Typ, Device und Inode prüfen. Restgrenze: pfadbasiertes `Popen` ist nicht fd-gebunden;
  vollständige TOCTOU-Schließung erfordert Helper/`fexecve` und eine neue Architektur.
  Status: **AWAITING USER APPROVAL**.
- DB: `16` Runs (`15` unveränderte Offline-Controls + Canary). Zwei stabile read-only
  Dashboard-Snapshots: `snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
  `source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
  `run_count=16`, `returned_count=16`, `truncated=false`, `query_only=1`. Die UI zeigt
  die Historie ohne Schreibzugriff; nur der Beobachtungszeitpunkt variiert.

## Historischer Offline-Pre-Live-Nachweis — anderer Scope

- Hauptsuite ohne Dashboard: `177` bestanden, `3` Windows-Skips und `12` Subtests;
  Wall `26.034290 s`, Total User/System `23.373336/1.227233 s`, Self-Peak-RSS
  `15,499,264 B`, Child-Peak-RSS `74,186,752 B`.
- Socketfreie Dashboard-Prüfung: `4/4` plus `3` Setup-Subbranches; Wall `0.002041 s`,
  Peak-RSS `31,260,672 B`. Eine frühere autorisierte HTTP-Prüfung bestand `13/13`;
  der spätere `16`-er HTTP-Scope wurde nach der letzten Härtung wegen Sandbox-/Usage-
  Limit nicht final wiederholt. Es wird kein grüner finaler `16`-er HTTP-Lauf behauptet.
- Der CLI-Lock wurde bestätigt: `mlx-run` endet mit Exit `78`, ohne Runner oder Worker zu
  importieren.
- Dritte Offline-Control-Generation:

  | Lauf | Wall | Exit | Ergebnis |
  |---|---:|---:|---|
  | slow | `0.191745 s` | `10` | `regression` |
  | known | `0.156192 s` | `0` | nur synthetisch |
  | wrong | `0.157021 s` | `10` | `correctness` |
  | missing | `0.157268 s` | `10` | `missing` |
  | exit70 | `0.145334 s` | `10` | `worker_exit` |
  | replay | `0.159542 s` | — | `idempotent` |

  Sequenz: `0.967681 s`, Self-Peak-RSS `16,334,848 B`, Child-Peak-RSS
  `28,819,456 B`; Provenance `5745e93f…39d57`, Replay-Bundle
  `6ae4a453…b7335`.
- Finale DB-Evidenz: `15` Runs (`3 × 5`), jeder mit genau einem verifizierten
  `common_result`; die älteren `10` Runs blieben unverändert. Snapshot:
  `source_revision=3b70324f…ab658d`, `id=512934c9…b5b52`, `run_count=15`, nicht
  abgeschnitten. DB-Größe `229,376 B`, Datei `0600`, Verzeichnis `0700`,
  `query_only=1`. Das sind Offline-Controls, keine H0-Hardwarewerte.

Der `177`-er Scope ist historisch und anders enumeriert; er ist mit dem aktuellen
`133`-er Scope nicht als Regression oder Zuwachs vergleichbar.

## Verifiziert

- macOS 26.5.2 (Build 25F84)
- Xcode 26.6 (Build 17F113) unter `/Applications/Xcode.app`
- `xcode-select` zeigt `/Applications/Xcode.app/Contents/Developer`
- `xcodebuild -checkFirstLaunchStatus` erfolgreich
- Python 3.12.13 und uv 0.11.19
- MLX 0.32.0 im bestehenden `.venv`; die aktuelle Luna-Read-only-Introspection bestätigte
  `mx.matmul`, `mx.eval`, `mx.synchronize`, `mx.compile` sowie
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`
- ProjectAtlas-Runtime 0.4.5-rc1 unter `/Users/tobiasburandt/.local/bin/projectatlas`
- ProjectAtlas-Codex-Plugin 0.4.5-rc1 installiert; offizieller Marketplace auf `v0.4.5-rc1`
- Codex-MCP-Server `projectatlas` aktiviert und auf die Project-Friday-Datenbank versioniert
- Historischer Setup-Snapshot (nicht aktuell): `projectatlas init` und anschließender
  `watch --once` indizierten 543 Dateien und 257 Ordner; der damalige lokale Index meldete
  281 offene Purpose-Hinweise.
- `scripts/verify_environment.sh` erfolgreich: Xcode, ProjectAtlas, MLX Metal, PyTorch MPS und
  alle drei MCP-JSON-Dateien geprüft

## Heutige Read-only-Audits

- Erster Sandbox-Lauf: Exit 1 wegen `RuntimeError: No Metal device available`; Ursache ist der
  fehlende GPU-Zugriff innerhalb der Sandbox.
- Genehmigter Lauf außerhalb der Sandbox: Exit 0; Tool-Walltime 1.741108708 s.
- Der aktuelle Luna-Read-only-Introspektionslauf im bestehenden `.venv` bestätigte MLX 0.32.0
  und die APIs `mx.matmul`, `mx.eval`, `mx.synchronize`, `mx.compile` sowie
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`. Die Sandbox hatte kein Metal; daher fand in diesem Lauf
  kein GPU-Lauf statt.
- Es wurden keine lokalen KI-, Modell- oder Software-Installationen und keine Downloads ausgeführt.

## Projektintegration

- Repository: `/Users/tobiasburandt/Project_Friday/ProjectAtlas`
- Projektlokale ProjectAtlas-Daten: `/Users/tobiasburandt/Project_Friday/.projectatlas/`
- generierte MCP-Dateien: `projectatlas.mcp.json`, `projectatlas.claude.mcp.json`,
  `projectatlas.opencode.json`
- vollständiges Konzept kopiert nach `docs/TECHNISCHES_KONZEPT.md`
- Phase-1A/H0-Vorregistrierung ergänzt: `docs/PHASE1_MATMUL_SPEC.md` (Matmul-
  Messsystem-Preflight;
  kanonische FP16-`2048²`-Performance-Workload, separate Correctness-Matrix, Correctness-/
  Memory-/Safety-Gates, Prozess-/Bootstrap-Regeln und Fallback).

## Atlas- und Indexstände

- `543 Dateien / 257 Ordner / 281 Purpose-Hinweise` ist ein historischer,
  definitionsgebundener Stand aus dem früheren `init`-/`watch --once`-Audit; die ursprüngliche
  Verifizierungs-Bullet oben bezeichnet ebenfalls nur diesen damaligen Snapshot. Er wird nicht
  mit späteren Atlas-Zählungen zusammengeführt.
- Der aktuelle Post-Edit-Atlas-Snapshot meldet Generation `22`, `549 Dateien` und
  `257 Ordner`.
- Die aktuelle Atlas-Overview meldet `280` fehlende Purpose-Angaben; das ist eine separate
  Overview-/Coverage-Metrik.
- Der aktuelle Session-Brief meldet `805` Blocker. Dieser Wert ist eine separate
  Session-/Health-Metrik und nicht identisch mit Datei-, Ordner- oder Overview-Zahlen.

## Offen nach Run22

- optional: ProjectAtlas-Purpose-Queue gezielt für die wenigen Projektdateien kuratieren; die große
  Upstream-Codebasis muss nicht manuell mit erfundenen Zwecken versehen werden;
- Der Offline-Unterbau für SQLite v1, das read-only `127.0.0.1`-Dashboard und Worker
  Option A einschließlich Pre-Live-Adapter ist offline implementiert und final geprüft.
  Die feste DB enthält `22` Runs; Run22 ist ein einzelner Baseline-Reference-Lauf, kein
  Hardware-Optimization-Loop.
- Der separate H0.1-Unterbau und die historische Evidenzmigration sind implementiert
  und verifiziert. Nächster wissenschaftlicher Schritt ist nicht ein weiterer H0-Retry,
  sondern die einmalige Durchführung des vorregistrierten Sechs-Session-Paced-Protokolls
  und dessen geschlossener Study-Replay. Bis dahin bleibt H0.1 `unresolved`.
- Die kritische Neubewertung reklassifiziert Phase 1A zu H0: Sie darf nur Messsystem-,
  Correctness-, Kontrollarm- und Fallbackverhalten belegen, keine Self-Optimization oder
  Hardware-Generalisation. Ein Forschungspivot auf H1 deterministic template-constrained
  tuning ist freigegeben, aber erst nach H0-Go/No-Go und A/A-3+3-Aggregation
  wissenschaftlich zu planen.
- `docs/PHASE1A_ARCHITEKTURFREIGABE.md` dokumentiert SQLite v1, read-only Loopback-
  Historien-Dashboard und Worker Option A als `approved/implemented-offline`. Nach der
  späteren allgemeinen Nutzerfreigabe wurde ausschließlich der dokumentierte H0.1-
  Legacy-Import ausgeführt; `aa_gpu`, Custom Metal, Modelle und weitere Optimierung
  wurden nicht ausgeführt.

## Phase-1A-Readiness

- Vorregistrierte Operation: `Y = mx.matmul(A, B)`, FP16 C-contiguous `2048²`, exakt
  17.179869184 GFLOP und 25,165,824 Bytes A+B+Y-Nutzdaten.
- Correctness-only-Matrix separat vorregistriert: sichtbare Seeds `0xC0DE0001` bis
  `0xC0DE0005` sowie Holdout-Seeds `0xC0DE1001` und `0xC0DE1002`; sie ist nicht Teil der
  Performanceaggregation. Zero-RHS muss exakt null sein; die `64²`-Sign-Invariante wird
  innerhalb des eingefrorenen Fehler-Envelopes geprüft.
- Timing-Contract: pro Output `mx.eval(out)` und vor Zeitfensterende `mx.synchronize()`;
  `time.perf_counter_ns`, `mx.eval` und `mx.synchronize` werden im Manifest benannt.
- MLX 0.32.0 und die aufgeführten Matmul-, Eval-, Compile- und Memory-APIs wurden in
  einem früheren read-only Introspektionslauf im bestehenden `.venv` bestätigt; die
  Sandbox hatte kein Metal, daher gab es keinen GPU-Lauf.
- `mx.compile` ist als sichere Framework-Vergleichsvariante eingeordnet; A/A ist der echte
  H0-Nullpfad. Es ist kein Custom-MLX-Metal-Kandidat.
- Keine Tests, Downloads, Installationen, GPU-Läufe oder Modelltests in diesem
  Dokumentationsschritt; die früheren fokussierten Offline-Testmetriken sind im
  append-only Arbeitsjournal vermerkt. Es wurden keine Modelle ausgewählt oder
  festgeschrieben.

## Offen: isolierter Ein-Kernel-Versuch und vollständiger Phase-1-DoD

Der vollständige DoD aus `IMPLEMENTIERUNGSPLAN.md` ist mit Phase 1A/H0 nicht erfüllt. Offen
bleiben eine separate Phase 1B mit begrenztem Custom-MLX-Metal-Kandidaten sowie dessen
Ausführung in einem isolierten Worker-Prozess mit Timeout, Ressourcenlimits, Correctness-
Test und Rollback. Die separate Sicherheits-/Architekturfreigabe und das vorgeschaltete
Shadow-Router-Gate liegen inzwischen vor; Vorregistrierung, Worker-Implementierung und
Messung des genau einen statischen Kandidaten sind noch offen. Eine breitere Phase 1B ist
nicht freigegeben. Die H1-Workload-/Shape-Familienaufteilung und eine cluster-level
Powerplanung bleiben ebenfalls offen.

## Grenzen dieses Status

Die Smoke-Tests beweisen nur Erreichbarkeit und einfache Korrektheit. Sie beweisen keine stabile
Performanceverbesserung, keine optimale Kernelkonfiguration, keine Neural-Engine-Kontrolle und keine
Übertragbarkeit auf iOS, Android, NVIDIA oder Rechenzentren.

## Historischer finaler Vertragsabschluss und Run21-Canary — 20.08.2026

Dieser Abschnitt ist der historische Run21-Stand; der vorstehende Run22-Abschnitt ist
maßgeblich.

Der finale Contract-Stand ist dokumentiert und offline geprüft: Core `175/0`, Dashboard
`4/4`, `0` offline MLX-Imports. Die zugehörige Provenienz ist
`575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`; Code-Hash
`aae3245e…` (im übergebenen Evidenzsatz nur als Präfix vorhanden), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac` und Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 wurde genau einmal ausgeführt und fail-closed beendet: Exit `10`, Wall `1.14 s`,
User `0.98 s`, System `0.16 s`, Peak-RSS `369,573,888 B`. Der gespeicherte Befund lautet
exakt `invalid/invalid/baseline_fallback` mit `warmup_unstable` nach `16` Warmups. Die
Agent-Statistik zeigt für `all` Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR
`582,260.25 ns`; für `last5` Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`,
Minimum `2,067,916 ns`, Maximum `2,677,583 ns`, Stabilität `false`. Persistiert wurden
`0` Rohsamples, `0` Correctness-Zeilen, `3` Scalars und `1` Artifact. Es gab kein `aa_gpu`
und daraus folgt weder eine Performance- noch eine Correctness-Aussage.

Die DB-Evidenz vor Run20 trägt den übergebenen Hash `c9a521…`; die Run21-DB den Hash
`420b7c…`. Bundle `027908…`, Result `ac4a82…`, Payload `cd409d…` und Evidence
`837841…` sind im Evidenzsatz nur verkürzt übergeben; die Ellipsen werden nicht durch
erfundene Vollhashes ersetzt. Die lokale UI liest die SQLite-Historie automatisch
read-only; die statische Prüfung von `friday_h0/dashboard.py` bestätigt Run-Auflistung und
Statusübernahme einschließlich `invalid`. In diesem Nachweis wurden weder Server noch
Socket gestartet.

Wissenschaftliche Entscheidung: Der eingefrorene Vertrag `8 → maximal 16` Warmups mit
Stabilität der letzten fünf Werte innerhalb `±5 %` entspricht dem Code. Es liegt kein
Implementierungsdefekt vor. Die Ursache der instabilen Messung bleibt als OS-/Thermik-/MLX-
Unsicherheit offen; Schwelle und Daten wurden nicht nachträglich geändert und Run21 wurde
nicht wiederholt. Der `python`-Aliasfehler und der Dashboard-`self.path`-Fehler sind separat
als Harnessfehler klassifiziert, nicht als Projektfehler. Konvergenzregel: Harnessbefunde
werden nur nach reproduzierbarer Wiederholung und unabhängigem Readback bewertet; sie ändern
keine wissenschaftliche Schwelle und ersetzen keinen Canary.

## Begrenzter Head-Skip-Runtime-Prototyp qualifiziert — 24.08.2026

Der Nutzer hat den zuvor in `PERMISSION_REQUIRED.md` blockierten kleinen
Runtime-Prototyp ausdrücklich freigegeben. Der Vertrag
`docs/HEAD_SKIP_RUNTIME_SPEC.md` und die Mini-Vorregistrierung
`experiments/head_skip_runtime/PREREGISTRATION.md` wurden vor jeder neuen
Runtime-Messung bytegenau eingefroren. Die Qualifikation ist Engineering-Evidenz,
kein Zyklus 13 und bleibt `formal_claim=false`.

Das getrennte Paket `friday_head_skip_runtime/` ist offline implementiert. Es
replayt die unveränderte formale 16-Record-Evidenz, prüft Modell, Prompt,
Tokenizergebnis und alle erlaubten Requestwerte und wählt den schnellen Pfad nur
im exakt bestätigten Einzelfall. Jede Unsicherheit wählt den Referenzpfad. Ein
Fehler des schnellen Pfads wird nicht im selben Aufruf wiederholt und verriegelt
den schnellen Pfad für den Prozess. Der MLX-Adapter lädt ausschließlich den lokal
gebundenen Snapshot. Eine eigene private, append-only SQLite-Historie und eine
read-only Oberfläche auf `127.0.0.1:8775` sind enthalten.

Vor dem neuen Code bestand die unveränderte relevante Baseline mit Exit `0` in
`4,44 s` außen, `11,08 s` User, `0,58 s` System, maximal `50.380.800 B` RSS,
Peak-Footprint `34.308.912 B` und ohne Swaps. Nach der Implementierung bestanden
die `20` neuen fokussierten Offline-Tests, die Bytecode-Kompilation und
`git diff --check`. Der absichtlich im schmutzigen Vor-Commit-Stand aufgerufene
Policy-Check replayte alle `16` formalen Records und fiel erwartungsgemäß auf
`worktree_dirty` zurück; es entstand keine Runtime-Datenbank und keine GPU-Arbeit.

Vor dem Live-Lauf wurde die Aktivierung zusätzlich an genau einen bestandenen
CPU-Record, eine unveränderliche Startmarke und genau einen bestandenen GPU-Record
gebunden. Abweichende Lauf-IDs oder Datenbankpfade können keinen zweiten Versuch
öffnen; unbekannter Swap-Verbrauch schließt das Ressourcengate. Die vollständige
Projekttestsammlung bestand mit Exit `0` in `38,50 s`, maximal `192.921.600 B` RSS,
Peak-Footprint `46.007.136 B` und ohne Swaps. Diese Softwaretest-Laufzeit ist kein
Modell-Geschwindigkeitswert.

Die vollständige lokale Quellprüfung ist unter
`/private/tmp/codex-security-scans/Project_Friday/c7db74f_20260824T084223Z`
versiegelt: alle `12` Quellzeilen wurden vollständig geprüft, Abdeckung
`complete`, keine offene Arbeit und `0` berichtspflichtige Sicherheitsbefunde.
Der nachgeschaltete Zwei-Dateien-Delta nach Ergänzung des CPU-BudgetGuards ist unter
`/private/tmp/codex-security-scans/Project_Friday/5b17bfb_20260824T085130Z`
ebenfalls mit vollständiger Abdeckung und `0` Befunden versiegelt.

Das CPU-Gate bestand mit Policy-Median `864,7 ns`, p95 `872,473 ns`, Zusatz
`839,473 ns` und Evidenzload `824,460209 ms`. Danach wurde genau ein, nicht
wiederholter MLX-Prozess gestartet. Baseline und Kandidat erzeugten im
Korrektheitspaar und in allen vier Messpaaren dieselben `32` Token. Der gemessene
Prefill-Median sank von `1806,4618545 ms` auf `1528,206979 ms`; das vorab
festgelegte gepaarte Verhältnis beträgt `0,8458362745`, entsprechend gerechnet
`−15,4164 %`. Peak-Delta `−97.855.968 B`, Swap-Delta `0`, Netzbetrieb und Duty
`0,15` hielten. Alle H1–H5-Gates bestanden; Entscheidung
`engineering_go_exact_scope`, weiterhin `formal_claim=false`.

Die Runtime-Historie enthält genau CPU-Record, Startmarke und GPU-Record, Modus
`0600`, SHA-256
`6dcf6e4cb942b842dca6e9b0b071df8e7c6cb81ba28fdc5e0fdb05c414d20567`,
Kettenkopf `db4c98e892136930cc515be94417a294c960cfaa9a790a6ea05629d0b796b8f3`.
Der normale Policy-Aufruf autorisiert den schnellen Pfad jetzt nur im exakten
registrierten Fall; die reale read-only Loopback-Abfrage lieferte HTTP `200` und
die verifizierte Dreierhistorie. Ungeprüfte Prompts, Modelle, Logprob-Anfragen,
Multi-Turn- und Parallel-Workloads bleiben auf der Baseline.

## Zyklus 16 — Vor-Hardware-Status: runtime-only Matmul-Umgebungs-A/B

Am 24.08.2026 wurde genau eine neue lokale Studie freigegeben:
`matmul-compile-ab-20260824-01` mit dem Kandidaten
`fixed_cache_compiled_decode_v1`. Die Freigabe betrifft ausschließlich die
Laufzeitumgebung. Modellgewichte, Modellarchitektur und Quantisierung werden
nicht geändert.

Die mathematische Matmul-Operation bleibt in allen drei Armen aktiv. Verglichen
werden nur `standard_eager`, `fixed_eager` und `fixed_compiled` — also der
Standardpfad, ein fester KV-Cache ohne Compile und ein fester KV-Cache mit
Compile. Es gibt keinen echten Matmul-Aus-Schalter und keinen Vergleich mit
veränderten Gewichten.

Die Präregistrierung ist im lokalen Seal-Commit eingefroren und noch nicht
gemessen. Exakte greedy Token- und
Textidentität ist ein Pflicht-Gate; ein Unterschied ist ein terminaler
Korrektheitsfehler. Die alten Device-Model-Compile-Messungen sind wegen falscher
Token ab Position 2 ungültig und werden nicht als Baseline oder Gewinn verwendet.
`formal_claim=false`. Es gibt noch keine Zyklus-16-Ergebnisdatei oder
Hardwareevidenz. Ein negatives Ergebnis ist gültig; bis zum Seal und den
vorgesehenen Messungen wird nichts automatisch ausgeführt oder aktiviert.

## Zyklus 16 — final geprüfter Seal-Stand vor Hardware

Der lokale Commit, der diesen final geprüften Stand enthält, ist der Seal-Commit.
Status danach: `sealed_pending_hardware`. Die Präregistrierung der Studie
`matmul-compile-ab-20260824-01` ist unverändert eingefroren; SHA-256
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Es gibt weiterhin keine Messung, keine `results.json` und keine private
Startmarke. `formal_claim=false`.

Vor-Hardware-Fixes aus dem Review: (1) lazy MLX-Fehler im kalten Compiled- und
Fixed-Eager-Pfad werden bis zur Materialisierung als `candidate_not_runnable`
erfasst, außer Ressourcen-/OOM-Fehlern; (2) der Parent begrenzt jeden Worker
auf die verbleibende harte Gesamt-Walltime und bricht sicher ab; (3) beobachtete
Armzeit und akzeptierte/gebuchte GPU-Zeit sind getrennt, einschließlich der
theoretischen Duty-Pause, sodass Budgetablehnungen als Teil-Evidenz erhalten
bleiben. Zusätzlich werden Arm-Längen, Finish-Grund und Messstatistiken streng
geprüft; minimale Worker-Fehler bleiben als bereinigtes Terminalereignis erhalten.

Verifikation: fokussiert `34 passed` (Exit 0), vollständige Pytest-Suite (Exit 0),
Compileall (Exit 0), Worker-Selfcheck 21 (Exit 0), Harness-Selfcheck 18 (Exit 0),
UI-Selfcheck (Exit 0), Standardaufruf ohne Hardware (Exit 78; keine Artefakte),
`git diff --check` (Exit 0), `xcodebuild -checkFirstLaunchStatus` (Exit 0).
ProjectAtlas 0.4.5-rc1 und MCP sind verfügbar. Gerät: Apple M1 Max, 32 GiB,
Netzbetrieb, `Device(gpu,0)`; MLX 0.32.0 und mlx-lm 0.31.3. Snapshot-SHA
`e6edcd46...eda`, Gewicht-SHA `94d3d701...74af3`.

## Zyklus 16 — reales Ergebnis und unabhängige Einordnung

Die Studie `matmul-compile-ab-20260824-01` wurde im Seal-Commit
`83ee3ea03f9fb303b8226ab8ad3189f07daec727` ausgeführt. Evidence-Commit
`cc6d2ea012a0cd6a858acc9a66d4754e95c421b7`; Ergebnis-Hash
`fbcc2fc65ac5d255ed11039a74c34e9a02d942cec17b25a6ed863058e0073b57`,
Verifikations-Hash
`09b1b53841a59bad3c4b1b9a0ef62fb659668b472358c10fa9188cad158f0038`, Marker-Hash
`8adf6f9c2453524bd1e05f4973ee85f84a323e9461a3f9b996ec2d0f7fed3c2f`,
Präregistrierungs-Hash
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Die Nutzerfreigabe ist damit genau einmal verbraucht; es gibt keinen zweiten Lauf.

Gemessen wurden sechs frische Prozesse mit drei Armen, also 18 Arm-Ausführungen
(3 × 6). Token und Text waren in allen 18 Arm-Ausführungen exakt gleich. Die
Decode-Gesamtzeit-Mediane und
Tokenraten waren:

| Arm | Decode-Median | Tokenrate | TTFT-Median |
|---|---:|---:|---:|
| `standard_eager` | 0,399939187 s | 77,5131895/s | 0,638376521 s |
| `fixed_eager` | 0,3999597295 s | 77,5078153/s | 0,638425813 s |
| `fixed_compiled` | 0,371848789 s | 83,3672240/s | 0,6385446665 s |

Der gemessene gepaarte Ratio `fixed_compiled/standard_eager` beträgt
`0,9295921887`, Bootstrap-95-%-KI `[0,9128789083; 0,9348209684]` — rechnerisch
7,0408 % schnellere Decode-Phase. Gegen `fixed_eager` beträgt der Ratio
`0,9296309524`, KI `[0,9256302629; 0,9327708433]`. Peak-RSS war
`3.771.564.032 B`, MLX-Peak `3.476.049.782 B`, Swap-Delta `0 B`.

Berechnet, nicht separat gemessen: warme End-to-End-Projektion `0,9829777045`
(rund 1,7022 % schneller insgesamt), kalte One-off-Projektion `1,0154895491`
(rund 1,549 % langsamer) und Break-even median rund 36,47 Decode-Schritte;
der Lauf umfasste 31 Schritte. Matmul wurde nie abgeschaltet. Modell, Gewichte
und Quantisierung blieben unverändert; geändert wurden nur feste Cacheform und
MLX-Compile-Laufzeitorganisation. Das ist kein allgemeiner Qualitäts-,
Selbstlern- oder Produktivclaim; automatische Aktivierung bleibt verboten.

Der Lifecycle-Selfcheck-Bug wurde erst nach der Messung entdeckt und getrennt
behoben; er wird nicht rückwirkend als Messwert umgedeutet. Die read-only UI
lieferte GET/HEAD `200`, Schreibmethoden `405`, fremden Host `421`; die Hashes
blieben unverändert. `formal_claim=false`.

## Post-Hardware-Verifikation — Zyklus 16 und historische Evidenz

Die abgeschlossene Studie wurde nach dem Lauf unabhängig verifiziert; ein neuer
Hardware- oder Modelllauf fand nicht statt. Die vollständige Suite meldete `787
Tests in 71 Dateien`, Exit 0. Der fokussierte Lauf
`test_matmul_compile_ab` meldete `43 passed, 60 subtests`, Exit 0. Compileall,
Worker-Selfcheck 21, Harness-Selfcheck 18, Dashboard-Selfcheck 0, Xcode-Check,
`jq` und `git diff --check` endeten jeweils mit Exit 0. Der Standardaufruf blieb
bei erwarteter Exit 78 und veränderte keine Artefakte. Der Harness-`--show`-
Aufruf lief einmal mit Exit 0, stderr blieb leer und er lieferte genau eine
gültige JSON-Zeile.

Die reale read-only UI-Verifikation ergab GET/HEAD `200`, alle Schreibmethoden
`405`, einen fremden Host `421` und `no-store`. Unbereinigter Modelltext wurde
nicht ausgeliefert. Die Cycle-16- und Cycle-15-Evidence sowie alle 12 SQLite-
Datenbanken waren vor und nach der Prüfung identisch; die private Startmarke
blieb auf Modus `0600`. Die Ergebnis- und Datenbank-Hashes blieben unverändert.

ProjectAtlas wurde genau einmal inkrementell mit `watch_once` aktualisiert:
`cycles=1`, `indexed=967` Textkandidaten, `parsed=11`, `unchanged=732` Symbole;
Runtime `0.4.5-rc1` und die projektlokale MCP-Konfiguration waren gültig.
Getrackte ProjectAtlas-Dateien blieben unangetastet; das bereits vorhandene
verschachtelte `.gradle`-Untracked blieb ebenfalls unangetastet.

Der nachträglich gefundene Lifecycle-Selfcheck-Fehler lag in der Annahme, die
Evidence müsse fehlen. Die Korrektur prüft nun read-only sowohl fehlende als auch
vorhandene Evidence, verbietet Symlinks, verlangt Modus `0600` für die Marke und
vergleicht Hashes sowie Modi vor und nach der Prüfung. Der Arbeitsbaum-Harness
unterscheidet sich dadurch vom versiegelten Code; die Evidence bewahrt jedoch die
Code-Fingerprints des Seal-Stands. `formal_claim=false`; die Freigabe ist
verbraucht und es wurde nichts automatisch aktiviert.

## Zyklus 17 — historischer Pre-Hardware-Draft

Studie `fixed-compiled-batched-readback-20260824-01`, Kandidat
`fixed_compiled_batched_readback_n8_v1`, Status `draft_pending_preflight`.
Die Nutzerantwort „Dann machen wir das mal“ reserviert genau einen neuen Lauf;
die Freigabe ist reserviert, aber noch nicht verbraucht. Noch kein Marker,
Ergebnis oder Modelllauf.

Einzige Variable ist Readback `1` versus `8` auf identischem Fixed-Compiled-4B-
Pfad; Modell, Gewichte, Quantisierung und Matmul bleiben unverändert. Geplant
sind sechs gepaarte frische Prozesse (zwölf Arm-Ausführungen). EOS-Tail wird
vollständig zeitlich erfasst und getrimmt; exakte logische Token und sichtbarer
Text sind terminales Gate. `formal_claim=false`; keine Aktivierung, kein Dienst,
kein Multi-Turn- und kein Qualitätsclaim. Cycle 7 `12,98 %` bleibt explorativ.

## Zyklus 17 — historischer sealed_pre_hardware-Stand (25.08.2026)

Offline-Preflight abgeschlossen: `measured=false`, `formal_claim=false`,
`authorization=reserved_not_consumed`; kein Modell-/MLX-/GPU-/Hardwarelauf,
kein Marker, kein Resultat, kein Dienst und keine Autoaktivierung. Readback 1
versus 8 bleibt die einzige Variable auf identischem Fixed-Compiled-4B-Pfad;
sechs gepaarte frische Prozesse und zwölf Arm-Ausführungen sind geplant.
Prereg `74f63c36ddd141c4b4666d9f15d7b17d3ac9294e2d63cb29f6d9e35a80db21b1`,
Worker `fecf712b44e6d1a8c46565dda59569fa11cdc762fc49917307874435e4a2efde`,
Harness `9c0689be97a1ee5022f7c4b4623af9bd4a9906411291d9c4295b4c16184c7ff0`,
Protocol `a58b6298a22e676b9213cc0e4b8fc22ecdc7e0adb25eb07a58f663d268164c30`, Dashboard `ccbaf05368f21acfa2c627a33f3ee9c5629d335d45f76abb5a74d1399fbeaaee`.
Selfchecks, 30/30, 817/817 (~195 s), Compileall, 89 JSON, Xcode und Diff
bestanden; ProjectAtlas Runtime `0.4.5-rc1` gut, Session-Brief
`refresh_required` wegen `dependency_closure_limit`. Swap 19359465472 Bytes,
Preflight-Delta 0; Snapshot/Gewicht/Generation-Config- und Hardware-Gates
bestanden. Die Nutzerfreigabe bleibt reserviert und ungenutzt.

## Zyklus 17 — Ergebnis

`measured=true`, Freigabe `consumed_exactly_once`, Entscheidung
`no_clear_speedup_baseline_retained`, `formal_claim=false`. Alle 6 Paare/12
Arme waren korrekt; Readback 8 war in jedem Paar schneller, aber Ratio-Median
`0,9581074518`, KI `[0,9534714914;0,9598849359]` verfehlte die feste 5-%-
Schwelle. Der 4,1893-%-Effekt ist berechnet; Baseline retained. Keine Modell-,
Gewichts-, Quantisierungs- oder Matmuländerung, keine Qualitäts- oder
Aktivierungsaussage. Evidence-Audit `evidence_valid=true`.
Result-SHA `d2eb29fe31dcf47fe294d2b0bec2d724fe6cc6e6f6d88d2035783613338fbddd`,
Verification-SHA `be91320e44507573601b81b90016ff0b9c8bee9458465cebff5ae438e6bf9214`,
Marker-SHA `18801fdfca677eb1b0d6a82ac95954990ac05476876447ca1170a42e754710dc`.

## Zyklen 18–21 — Fused-Greedy-Compile: Fehlerkette und gültiger Abschluss

Zyklen 18 bis 20 waren terminale Vorläufe ohne Performance-Evidence. Alle
blieben `formal_claim=false` und luden kein Modell (`load_count=0`):

| Zyklus | Resultat-SHA256 | Seal-Commit | Evidence-Commit | Ursache |
|---|---|---|---|---|
| 18 | `ea644a912c9bb20a9fc992d7e24bfecfbb70285f2788ee83a15aeb4937503035` | `f1383587d585620f75e3c1e9bd40a71cbd0e8af9` | `dc2cdced58b629e6a39cb8ed870d847d8ee16c13` | Parent/Worker-Environment-Fingerprint verschieden; Provenienzprüfung vor Paar 0. |
| 19 | `4e02221975f6f1710e96dc70f69b4df6f48a1d93df859c6274ed83460dee0320` | `7278bda3281161cebdcf395fd4aa50df5de5124e` | `59bbe9d698d978dcbd621fe89fb17bf98b286b8a` | Worker sah sein eigenes absichtlich erzeugtes Resultat als unerlaubte Git-Änderung. |
| 20 | `72e7e0692136766bcd5cea4147f3c106ad64de8ddadba855767d8908ae53200d` | `6c0e18b17b2febf184fda0fc09552b5613c49dc0` | `78f983c71636637b7995eb90500fe689cbe53fee` | Parent-Manifest enthielt `dev`, Worker-Manifest nicht; Snapshot-Binding vor dem Laden. |

Die Fehler wurden fail-closed erhalten. Korrekturen waren identische
Environment-Fingerprints, eine Allowlist nur für das eigene Resultat und ein
gemeinsamer Snapshot-Statvertrag inklusive `dev`. Es gab keinen Matmul-
On/Off-Nebenclaim.

Zyklus 21 wurde mit Seal `ad4c92f32e608a8a0870b37e23a4dba0da1f666c` und
Evidence `4f89e51c3933aa9c9d42563393589da3c2e4a875` abgeschlossen. Prereg-SHA
`a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf`, Resultat-
SHA `55bad770baad66cbebb804288845e9cf2785c0969c77355731ab8a23b3a43a2e`,
Marker-SHA `1c1dc10670c153c4c7430f3320671c08a3d56114e0fc5ee6af988c750ceb14e4`.

Gemessen wurden sechs frische serielle 4B-Prozesse und zwölf Arme. Token und
Texte waren exakt gleich. Decode-Median: external `0,266399792 s`, fused
`0,266088688 s`. Der gepaarte Ratio-Median ist `1,000510010`, Bootstrap-95-%-
KI `[0,981178182; 1,004700679]`, seed `20260825`, 10.000 Resamples, ohne
Ausreißerentfernung. Die Entscheidung lautet
`fused_greedy_compile_inconclusive`, also kein klarer Gewinn; der erwartete
`--execute`-Exit für diesen Nicht-Gewinn ist `1`.

Ressourcen- und Budget-Gates bestanden; RSS maximal `3.769.974.784 B`,
MLX-Peak `3.524.169.562 B`, Swap-Delta überall `0 B`. Rohzeiten und
Identitäts-Hashes sind gemessen; Mediane, Perzentile, Prozentänderungen und
Konfidenzintervalle sind berechnet. Modell, Gewichte, Quantisierung und
Matmul blieben unverändert. Kein Qualitäts-, Selbstlern-, Produkt- oder
Aktivierungsclaim.

## 2026-08-27 — R6/R7- und Abschlussstatus

Die nicht-architektonischen R6/R7-Fixes im IronMule-Worktree sind verifiziert:
importseitige HF-Policy-Mutationen wurden entfernt; `load_engine` lädt lokale
Pfade direkt und löst Offline-Hub-IDs mit `snapshot_download(local_files_only=true)`
und Runtime-Allow-Patterns auf. Profile prüfen Schema und aktuelle Bedingungen;
`revalidate` misst die aktuelle Prompt-Tokenlänge und erhält Driftprofile nur
explizit roh. Unsupported-Kandidaten werden typisiert übersprungen, Paketversion
und Dependency-Obergrenzen sind zentral bzw. festgelegt, und GPU-Core-Erkennung
wird pro Prozess gecacht. Der aktuelle R8-Doctor-Fehler trat bei der Reihenfolge
der vollständigen Suite auf: `_load_optional('mlx_lm')` importierte im Parent,
fing einen Transformers-/Torch-Importfehler ab und hinterließ teilweise
initialisierte Module. Isolierte Subprozesse für Dependency- und Metal-Probes
beheben diesen Fehler. Der B28-PYTHONPATH-Fehler ist davon getrennte Vorarbeit:
Damals fehlte beim direkten CLI-Aufruf der Repository-Root in `sys.path`.

Finale Evidenz: vollständige Nicht-Integrations-Suite `108 passed, 11 deselected`,
Pytest `7.62 s`, `/usr/bin/time` `8.46 s` real,
Peak-RSS `346996736 B`, `0` swaps; Gemma-Integration `10` Tests in `26.58 s`,
Peak-RSS `3348430848 B`, `0` Swap; Doctor grün. Das Wheel wurde gebaut, seine
Metadaten geprüft und die CLI per Zipimport in einer temporären Kopie gesmoked;
es wurde nicht installiert. Finales CLI-Metadata: grün.
Sichere reale No-Model-CLI-Smokes für `tune --show`, `status` und `revalidate`
beendeten sich jeweils mit Exit 0.
Der read-only `ironmule models`-Smoke fand Gemma unter Commit
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, Größe `3439894985 B`, mit leeren
Warnings; es erfolgten keine Resolution und keine Downloads.
Der Benchmark-Smoke war tokenidentisch und zeigte `-18.80 %` ausschließlich auf
`2` Requests × `3` Tokens, KI `[1.0643,1.3116]`; das ist ausdrücklich kein
Claim. Das JSON war temporär, SHA-256
`36ba45933b3de344116812e34bb451d19124b0ab35db3d3ee659b768dacc6209`. Swap systemweit:
`13295.19M` vor Integration, `13151.19M` danach, `12959.19M` nach Smoke — kein
positives Delta.

**Offen / nicht freigegeben:** Architekturblocker R2, R3
(Stock-/Fresh-Process-Baseline), R6 (Modellrevision und Quantisierung) sowie eine
separate IronMule-History-UI. Es wurden keine Downloads oder Installationen
ausgeführt; `ProjectAtlas/` blieb unverändert.

## Interaktives H0-Signal-Board — 28.08.2026

Das freigegebene lokale H0-Dashboard ist nun ein interaktives, responsives
ASCII-/Matrix-Signal-Board. Es liest weiterhin ausschließlich die gebundene
SQLite-v1-Datenbank query-only auf `127.0.0.1`, verwendet keine externen Assets
oder neuen Abhängigkeiten und besitzt keinen Schreibpfad. Filter für Status,
Mode, Fenster und freie Suche passen fünf KPI-Karten, vier Visualisierungen und
die Historientabelle sofort an. Alle fünf Sekunden wird ein begrenzter Snapshot
neu eingelesen; Live lässt sich pausieren. Run-Drilldown und Sample-Familien
zeigen vorhandene Rohsamples als getrennte Baseline-/Kandidatenserien.

Aktuelle Quelle: `28` Runs (`15 completed`, `10 invalid`, `3 worker_exit`), davon
`9` mit Rohsamples. Browser-E2E: Default `25` Runs/Punkte, Filterergebnisse
`invalid=10`, `aa_gpu=9`, `confirmation=3`; ein echter Detailaufruf lud `141`
Samples und zeichnete `60` `pair_performance`-Werte in zwei Serien. Desktop und
`390×844` hatten keinen horizontalen Seitenoverflow; Konsole und Overlay blieben
fehlerfrei.

Verifikation: Dashboard `16/16`, Compileall, JS-Syntax, Diff, Xcode,
ProjectAtlas `0.4.5-rc1` plus MCP-Konfiguration und read-only MLX-Introspektion
`0.32.0`, `Device(gpu, 0)` bestanden. Assets: HTML/CSS/JS
`5,680/11,908/20,580 B` bei `98,304 B` Limit je Asset. Drei Nachmesssessions mit
je `30` Snapshot-Wiederholungen ergaben Session-Mediane
`11.501708/11.475229/11.422146 ms`; wegen nicht randomisierter Messreihenfolge
und unverändertem Backend wird daraus kein Performanceclaim abgeleitet. Details,
Vorherwerte und Fehlerlösungen stehen im Arbeitsjournal.

## 2026-08-30 — L1 Gemma-Optimizer-Architekturvorschlag

Der read-only Audit für den gewünschten Apple-Silicon-/Gemma-Optimizer ist
abgeschlossen. Der Vorschlag
[`docs/L1_GEMMA_OPTIMIZER_ARCHITEKTURVORSCHLAG.md`](docs/L1_GEMMA_OPTIMIZER_ARCHITEKTURVORSCHLAG.md)
steht ausdrücklich auf **NICHT FREIGEGEBEN**. Er definiert eine unabhängige
`friday_optimizer/`-Control-Plane mit AC-only-Readiness, Mehrfachproben gegen
Claude-/Fremdlast, atomarer Lease, manueller 5–30-Minuten-Gesamtfrist, fester
Candidate-Allowlist, token-/textidentischer Korrektheit, TTFT und Decode-only
Tokens/s, automatischer Aktivierung mit Nutzer-Override sowie sofortigem
Baseline-Rollback. Der aktuell von Claude verwendete
`.worktrees/ironmule-b7` bleibt außerhalb des Änderungsbereichs.

Ist-Evidenz: lokale Snapshots nur Gemma 1B/4B; verwertbare Rohdaten für 1B/4B/12B
bleiben getrennt; 27B ist nur unqualifizierte Summary ohne lokalen Snapshot; eine
Cross-Device-Evidence existiert nicht. Es gab in diesem Audit keine
Codeänderung, keinen Modell- oder Benchmarklauf, keinen Download und keine
Installation. Implementierung, erster realer Hardwarelauf und jede Aktivierung
bleiben bis zu einer separaten ausdrücklichen Nutzerfreigabe blockiert.

## 2026-08-30 — Q2/E15-Handover und Architekturkorrektur

Der Handover zu IronMule-Commit `b700377e83b2eba39c5d66976d01332f8ab57bc6`
wurde geprüft. Q2 ist ein einmaliger M1-Max-/Gemma-4B-Engineering-Smoke für
Allowlist, Screening, Paired Confirmation und Profilwrite, kein formal versiegelter,
globaler oder produktiver Claim. Roh-PIDs, Exitcodes, Kandidaten-Outputs und das
Zweitstart-Log fehlen; das gespeicherte Profil trägt den Screening-Wert. Der Fix
`0de69b6` gilt nur für künftige Läufe.

Der erste E15-Aufruf scheiterte vor dem Modellstart wegen fehlendem `PYTHONPATH`.
Der korrigierte einmalige Lauf dauerte `1664.407 s`, nutzte vier eindeutige PIDs
und hielt die Peaks bei ungefähr `7.07 GB` flach; gegenüber `1571.59 s` vorher
ist dies kein Speedclaim. Forking gilt als Memory-/Messintegritätsmechanismus.
E15-after-fork SHA-256:
`d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`.

Der Architekturvorschlag ist auf `friday_optimizer` als Control Plane mit strikt
commit-/fingerprintgebundenem `IronMuleTuneAdapter` angepasst; dadurch wird keine
zweite GemmaWorker-/Tuner-Implementierung geplant. `BACKLOG.md` bleibt offen und
die Freigabe **NICHT FREIGEGEBEN**. Der aktualisierte Handover weist `371`
Archiveinträge und `156` eindeutige Dateien aus. Keine Aktivierung, Installation,
kein Download und kein PR-Merge.

## 2026-08-30 — Abschluss der heutigen Verifikation

Der b7-Source-Stand blieb auf Commit `b700377e83b2eba39c5d66976d01332f8ab57bc6`
unverändert; für E15/R12 wurden nur `HANDOVER.md`, `research/LEDGER.md` und
`docs/BACKLOG.md` dokumentarisch geändert. `Q2_profiles.json` ist mit SHA-256
`0a1104b248b4aaf532ee8ef7d9c9c0c06196dde0c5111450ee9386358d15509b` und
E15-after-fork mit SHA-256 `d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`
content-addressed archiviert; Manifest: `371`/`156`.

Das gültige Test-Gate ist `pytest -n0 -m 'not integration'`: `250 passed`,
`13 deselected` in `6.22 s`. Der vorherige unfiltered Lauf war ein
Operatorfehler und nach 30 Minuten SIGTERM/Exit 143 ohne Pass-/Fail-Ergebnis.
`python -m ironmule.bench` endete mit Exit 0 und bestandenem Self-Check.
ProjectAtlas wurde für beide Roots aktualisiert. Kein weiterer Modelltest, Download,
Installation, Aktivierung oder PR-Merge.

## 2026-08-30 — Freigabe der Offline-Implementierung

Die Offline-Implementierung der unabhängigen `friday_optimizer/`-Control-Plane ist
durch den fortbestehenden Nutzerauftrag freigegeben. Der Status bleibt
**HARDWARE/PROMOTION GATE-BASIERT**: reale Modellläufe nur manuell, AC-only,
fremdlastfrei, sparsam und maximal `30` Minuten; Downloads und Installationen
bleiben gesperrt. Echte Daten und End-to-End-Tests gehen synthetischen Daten vor,
die nur Rand- und Fehlerfälle abdecken. Automatische Produktaktivierung bleibt bis
zum Promotionsgate blockiert. Der Claude-Worktree `.worktrees/ironmule-b7` wurde
nicht verändert.

## 2026-08-30 — Verifizierter Offline-Optimizer-Stand

Die unabhängige `friday_optimizer/`-Control-Plane ist offline implementiert und
getestet: Memory, Corpus, Dataset, Bridge, Fingerprint, Candidates, Evaluator,
Readiness/Lease, Session, Profile, History, Orchestrator, IronMule-Adapter,
Dashboard und CLI. Das Memory enthält `401` Records bei bestätigter
Chain/Integrity; der Dataset-Snapshot enthält `392` Records und ist
`smoke_only/no_learning_claim` (`train=2`, `val=0`, `holdout=0`). Deterministische
Baselines und Shadow-Auswertung sind umgesetzt. Learned Ranking, GBDT und BO,
reale Adapterausführung und Promotion bleiben offen beziehungsweise gesperrt.
Hardware-/Modellläufe bleiben manuell, AC-only, fremdlastfrei, sparsam und maximal
`30` Minuten; Downloads und Installationen bleiben gesperrt. Der Claude-Worktree
`.worktrees/ironmule-b7` wurde nicht verändert.

## 2026-08-30 — Q2 sealed readiness gate

Der gebundene Friday-Source-Stand ist C0 `8b63b7b406bad7b380918ff5c2970fab4b36d5af`;
die versiegelte Q2-Präregistrierung wurde mit Commit `a7520b7` dokumentiert.
Der statische Fingerprint `11242a3a1343fc2b56653a89f30a0f7204b3f5fa5b61b1d1ee171c37a065abe5`
liegt vor. Der erste Readiness-Versuch blieb wegen `foreign_load`, hoher Last,
hoher CPU und instabilem Speicher blockiert; `model_started=false` und
`session_consumed=false`. Es gab keinen Modell-, GPU- oder Benchmarklauf.
Die fokussierte RealSession-/Optimizer-Suite bestand mit `100 passed`.
ProjectAtlas meldete zwischenzeitlich `dependency_closure_limit`; der danach
ausgeführte `watch_once`-Refresh war erfolgreich. Readiness-Evidence:
`READINESS_BLOCKED_20260830.json`.

## 2026-08-30 — Readiness-History-Reconciliation

Die Records Seq 402 und 403 binden dieselbe Readiness-Evidence
`ae150c04d91629d0abd6a75688e739ce1949e02ecc72380b09b7a5f12c7056ea`. Seq 402
bleibt als kanonischer `readiness_blocked`-Event bestehen; Seq 403 ist wegen
abweichender Event-ID ein semantisches Übergabe-Duplikat und wird ignoriert,
nicht gelöscht. Die Hashkette blieb intakt (`403` Records vor der Korrektur,
`chain_breaks=0`, SQLite-Integritätsprüfung `ok`). Die Korrekturdatei
`READINESS_HISTORY_RECONCILIATION_20260830.json` trägt SHA-256
`1f41b229860eacbc54a4ecd4cfa732fec796d519ed02598c31b4b12bca871596`.
Runtime-Impact, Modellstart und Sessionverbrauch bleiben `false`.

## 2026-09-01 — Repo-Hygiene M1, Punkte 1–4

Reine Dokumentations- und Strukturarbeit; kein Hardware-, Modell- oder
Evidenzlauf, keine versiegelte Datei berührt, alle SQLite-Datenbanken
unverändert.

1. `PROJECT_STATUS.md` auf eine Kurzfassung reduziert (Audit-Tabelle,
   Kurzstand Zyklen 16–21/Optimizer, geltende Entscheide, Verweise). Der
   vollständige frühere Inhalt steht byteidentisch oben in diesem Journal
   unter „Archiv — vollständiger PROJECT_STATUS.md-Stand bis 2026-08-30".
2. `EXPERIMENT_BACKLOG.md` → `docs/KANDIDATENLISTE.md`, als Studienakte
   deklariert; `BACKLOG.md` ist das einzige Backlog.
3. Regel „Gemeinsamer Code für neue Studien" in `AGENTS.md`: neue Studien
   nutzen `friday_evidence` statt Paketkopien; versiegelte Pakete bleiben
   eingefroren. Befund: vier divergierte `statistics.py`-Kopien
   (`friday_h0/h1/n10/n10_v2`).
4. Leeres `src/` gelöscht; `HANDOFF_PROMPT.md`, `CODEX_START.md`,
   `PERMISSION_REQUIRED.md`, `NEXT_PREREGISTRATION_CANDIDATE.md`,
   `IMPLEMENTIERUNGSPLAN.md`, `OVERNIGHT_RESEARCH_LOG.md` → `docs/`.
   Vorab-Grep: keine Provenienz-/DB-/Code-Bindung an diese Pfade; Links in
   `AGENTS.md` und `docs/ANWEISUNGEN_UND_DOKUMENTE.md` angepasst. Historische
   bzw. hashgebundene Dokumente (Journal, Specs, `PERFORMANCE_BASELINE.md`)
   wurden nicht editiert; dortige Alt-Referenzen nennen weiterhin die alten
   Root-Pfade.

M1 Punkt 5 (`.friday-data/models/`, 3,9 GB) bleibt offen im Backlog.

## 2026-09-01 — R0 und R1: Entscheidungslogging und Replay-Environment

Reine Offline-Implementierung nach dem RL-Fahrplan in
[`docs/FABLE_ERFOLGSPFAD.md`](FABLE_ERFOLGSPFAD.md). Kein Modellstart, kein
Hardwarelauf, keine versiegelte Datei berührt, keine bestehende SQLite-Datei
verändert. Kein Lern-, Hardware- oder Performanceclaim.

**R0 — `friday_optimizer/decisions.py`.** Jede Tuner-Entscheidung wird als
unveränderlicher Feature-Record und jedes Ergebnis als zugehöriger
Label-Record in Optimization Memory v2 geschrieben:

- `DecisionEvent` (Schema `friday.optimizer.decision.v1`): Kontextvektor aus
  dem exakten Fingerprint, vollständige Kandidatenmenge, gewählte Aktion,
  Auswahlregel, **Propensity**, Policy-/Registry-Hash, Hints und — bei
  stochastischer Auswahl — der Seed.
- `OutcomeEvent` (Schema `friday.optimizer.decision-outcome.v1`): Reward mit
  Zensierungsstatus. `observed` trägt als einziger Status eine Zahl; ein
  Timeout, ein Fehler oder ein gerissenes Gate wird als eigener terminaler
  Status gespeichert und **nie verworfen**.
- `SelectionPolicy` mit den Regeln `deterministic_order`, `epsilon_greedy`
  und `user_forced`. Die deterministische Regel wählt den besten gehinteten
  zulässigen Kandidaten, sonst `baseline` — bei leerem Korpus ist der
  unveränderte Referenzpfad die einzig ehrliche Voreinstellung. Die
  Propensity ist die exakte Wahrscheinlichkeit der tatsächlich gezogenen
  Aktion, keine Näherung.

Bewusste Schemaentscheidung: **kein neuer SQL-`kind`.** Die Memory-Schema-SQL
wird byteexakt geprüft (`_validate_schema`); ein neuer `kind`-Wert hätte jede
bestehende Datenbank ungültig gemacht. Die Records nutzen `RecordKind.SYSTEM`
mit versioniertem Payload-Schema, Phase `feature` beziehungsweise `label`. Die
Hashkette bleibt gültig; `verify_chain()` ist Teil der Tests.

Kill-Kriterium aus dem Backlog eingehalten: Das Logging liegt außerhalb jedes
Messfensters. `OptimizerOrchestrator.select()` läuft vor dem Sessionstart,
`record_outcome()` danach; kein Aufruf liegt im gemessenen Pfad, also kann
das Logging kein Timing verschieben.

**R1 — `friday_optimizer/replay.py`.** Deterministische, read-only
Replay-Umgebung über den geloggten Entscheidungen plus Off-Policy-Evaluation:

- `ReplayEnv` mit striktem Action-Masking auf der versiegelten Allowlist. Eine
  maskierte Aktion ist ein Fehler, keine stille Nulllösung. Wählt die
  bewertete Policy eine andere als die geloggte Aktion, ist der Reward
  `None` — das Kontrafaktische wurde nie gemessen und wird nicht imputiert.
- Zensierte Läufe behalten einen konfigurierten, nie positiven Reward
  (Default `0,0` = „kein Gewinn"); ein positiver Wert wird konstruktiv
  abgelehnt.
- Reward-Konvention: gespeichert wird das Verhältnis `Kandidat / Baseline`
  (kleiner ist schneller), `default_reward` liefert `1 − ratio`, also den
  relativen Gewinn, damit für alle Schätzer „größer ist besser" gilt.
- Schätzer: `ips`, `snips` (selbstnormalisiert), `doubly_robust` und
  `replayer` (Rejection Sampling), dazu Kish-`effective_sample_size` und ein
  Bootstrap-Intervall.
- Ehrlichkeitsgate: unterhalb `DEFAULT_MIN_SAMPLES = 30` effektiver Stichproben
  meldet jeder Schätzer `insufficient_data` und `conclusive=false`; ohne
  Überlappung `no_overlap`, ohne Label `no_labels`. Ein Wert darf nur gelesen
  werden, wenn `conclusive` wahr ist.

**Bedienoberfläche.** Neue CLI-Kommandos `decide`, `outcome` (beide
schreibend, daher `--execute`-pflichtig) und `replay` (read-only). Neue
Orchestrator-Methoden `select`, `record_outcome`, `replay`, `evaluate_policy`.
Neues read-only Dashboard-Panel „Decision log (RL-ready)" unter
`/api/decisions` mit Policy, Regel, Aktion, Propensity, Zensierung, Reward und
dem Schätzstatus; die Antwort trägt dauerhaft `learning_claim=false` und
`no_activation=true`.

**Verifikation.** Vollsuite `1459 passed, 2630 subtests passed`. Neue Tests:
`tests/test_optimizer_decisions.py` (10), `tests/test_optimizer_replay.py`
(13), vier CLI-Tests, zwei Dashboard-Tests. End-to-End über die CLI geprüft:
`decide` → `outcome` → `replay` schreibt zwei Records, die Kette verifiziert,
und `replay` meldet bei einem Datenpunkt korrekt `insufficient_data` mit
Exitcode `UNAVAILABLE`.

**Gefundener Fehler und Ursache.** `test_subprocess_stage_is_hard_deadline_bounded`
schlug fehl mit `AssertionError: 'error' != 'timeout'`, Grund `exit:-9`.
Ursache: `SubprocessStageRunner` führt eine **Kopie** der allowlisteten Binary
aus. macOS killt eine kopierte, Apple-signierte Systembinary mit SIGKILL —
`cp /usr/bin/python3 … && ./py3 -c …` liefert reproduzierbar `rc=137`. Der
Produktionspfad ist nicht betroffen, weil eine reale Session den
venv-Interpreter benutzt, dessen Kopie ausführbar bleibt. Behoben in der
Fixture: `STAGE_EXECUTABLE = os.path.realpath(sys.executable)`. Merksatz für
Folgearbeiten: **eine Apple-signierte Systembinary darf nie Stage-Executable
sein.**

**Grenzen unverändert.** Downloads und Installationen bleiben gesperrt, reale
Hardwareläufe bleiben manuell, AC-only, fremdlastfrei, maximal 30 Minuten und
einzeln freigegeben. RL bleibt NO-GO bis R2 mit Off-Policy-Evaluation auf
vorregistriertem Holdout belegt ist; R0 und R1 begründen selbst keinen
Lernclaim, sie halten die Option offen.

## 2026-09-01 — F1: Vorregistrierung, Projektion und ein korrigierter Erwartungswert

Reine Offline-Analyse und Implementierung. Kein Modellstart, kein Hardwarelauf,
keine versiegelte Datei und keine SQLite-Datei verändert. Datenquelle sind
ausschließlich bereits versiegelte Ergebnisse auf Platte.

**Neuer Baustein `friday_optimizer/integration.py`.** Die bisherige Bewertung
kannte nur zwei getrennte Metriken (TTFT und `decode_tps`). F1 verlangt eine
End-to-End-Zahl, also gibt es jetzt die Anfragezusammensetzung
`request_seconds = ttft + tokens / decode_tps`, gepaarte Anfrage-Ratios,
Bootstrap-Intervall und die vorregistrierte Entscheidungsregel
`evaluate_integration` mit den Zuständen `qualified`, `below_threshold`,
`inconclusive` und `rejected`. Statistik und Paarungsprüfung kommen aus
`evaluator.py` — nichts davon ist neu implementiert; die vier divergierten
`statistics.py`-Kopien sind genau die Geschichte, die hier nicht wiederholt wird.

**Projektion aus versiegelter Evidenz** (`experiments/f1_integration/project_f1.py`,
Quelle `experiments/persistent_process/results.json`, sechs Paare, Prompt
`897` Token, `32` generierte Token):

- warme Baseline: TTFT `1,7851 s`, Decode `0,4367 s` (`70,99` Token/s), gesamt `2,2218 s`
- kalte Baseline: gesamt `5,5969 s`
- Prefill-Anteil der warmen Anfrage: `79,84 %`

| Kandidat | Ratio | End-to-End |
| --- | --- | --- |
| nur Head-Skip | `0,877356` | `12,26 %` |
| nur `fixed_compiled` | `0,985805` | `1,42 %` |
| beide (Arm `warm`) | `0,863161` | `13,68 %` |
| beide + persistenter Prozess (Arm `cold`) | `0,299489` | `70,05 %` |

**Korrigierter Erwartungswert — Phasen multiplizieren nicht.** Der bisher im
Erfolgspfad genannte Korridor „20–70 %" ist am oberen Ende richtig und am
unteren Ende zu optimistisch. Die naive Rechnung `0,846385 × 0,9295921887 =
0,786793` (`21,32 %`) unterstellt, dass sich ein Prefill- und ein Decodegewinn
multiplizieren. Sie wirken auf verschiedene Teile derselben Anfrage; die
zusammengesetzte Wirkung ist das zeitgewichtete Mittel. Korrekt sind
`13,68 %` im warmen Arm.

Zweiter Befund: `fixed_compiled` trägt bei `32` Token nur `1,42 %` end-to-end
bei — unter der überall verwendeten MDE von `5 %`. Sein Einzelbeitrag ist in
einer End-to-End-Studie prinzipiell nicht bestätigungsfähig. Es bleibt im
Kandidatenprofil, weil es mitläuft, aber die Studie darf dafür keinen
Einzelnachweis beanspruchen.

Dritter Befund, Sensitivität: der zusammengesetzte warme Gewinn *sinkt* mit der
Antwortlänge — `14,87 %` bei `8` Token, `13,68 %` bei `32`, `11,18 %` bei
`128`, `8,69 %` bei `512`. Die Antwortlänge gehört deshalb in den
Studienvertrag, nicht in die Auswertung.

**Vorregistrierung** liegt als `docs/F1_INTEGRATION_VORREGISTRIERUNG.md` vor:
zwei Arme (`cold`, `warm`), Schwellen `50 %` und `10 %`, MDE `5 %` aus A/A
dieser Studie, Tokenidentität als terminales Gate, Kill-/Pivotkriterien. Noch
ohne Umgebungs-Hashes — die kommen beim Versiegeln.

**Blocker für die Ausführung, verifiziert.** `optimizer_identity()` liefert
`optimizer_checkout_dirty`. Eine reale Session kann erst geplant werden, wenn
der Arbeitsbaum committet ist; `code_manifest_sha256` bindet jede
`friday_optimizer/*.py`. Das ist kein Stilwunsch, sondern eine
Ausführungsvorbedingung. Nebenwirkung derselben Bindung: jede Änderung an
`friday_optimizer/` — auch die heutigen R0/R1/F1-Module — macht eine früher
versiegelte Vorregistrierung stale; sie muss gegen den neuen sauberen Stand neu
versiegelt werden.

**Verifikation.** Vollsuite `1472 passed, 2630 subtests passed`. Neu:
`tests/test_optimizer_integration.py` mit 13 Tests, darunter der explizite
Nachweis, dass die naive Produktannahme um mehr als fünf Prozentpunkte danebenliegt.

### Nachtrag — Amdahl-Decken je Phase, und was die Decode-Hebel kosten durften

Aus derselben warmen Baseline (`ttft=1,7851 s`, `70,99` Token/s) folgt eine
Obergrenze je Hebelklasse. „Decke" heißt: der Hebel wird perfekt, die andere
Phase bleibt unverändert.

| generierte Token | Prefill-Anteil | Decke Decode-only | Decke Prefill-only |
| --- | --- | --- | --- |
| `8` | `94,06 %` | `5,94 %` | `94,06 %` |
| `32` | `79,84 %` | `20,16 %` | `79,84 %` |
| `128` | `49,75 %` | `50,25 %` | `49,75 %` |
| `512` | `19,84 %` | `80,16 %` | `19,84 %` |

Für die registrierte Antwortlänge `32` gilt: **jeder** Decode-Hebel konkurriert
um höchstens `20,16 %`. Die tatsächlich gemessenen Decode-Kandidaten
end-to-end umgerechnet:

| Kandidat | Decodegewinn | end-to-end |
| --- | --- | --- |
| Fused Greedy (Zyklus 21) | `−0,05 %` | `−0,01 %` |
| gebündelter Readback (Kandidat 18) | `4,19 %` | `0,84 %` |
| `fixed_compiled` (Zyklus 16) | `7,04 %` | `1,42 %` |
| Custom Metal Kernel (Phase 1B) | `1,87 %` | `0,38 %` |

Der eine Prefill-Hebel, der gemessen wurde — Head-Skip, Kandidat 19 — liefert
`12,26 %` end-to-end und ist damit größer als alle vier Decode-Kandidaten
zusammen. Die Kandidatenliste hielt seit Zyklus 9 fest, dass der Engpass
Prefill ist (Notiz zu Kandidat 14); die Arbeit ging danach überwiegend in
Decode-Kandidaten. Diese Tabelle beziffert, warum die alle unter der 5-%-Schwelle
landen mussten: sie *konnten* dort nicht landen.

Einschränkung, die mitgehört werden muss: das gilt für die registrierte
Workload — Prompt `897` Token, `32` generierte Token. Ab etwa `128` generierten
Token kippt das Verhältnis, und für lange Generierung sind Decode-Hebel die
richtige Klasse. Die Aussage ist workloadbedingt, nicht allgemein.

## 2026-09-01 — Identitäts-Forensik: alle Prefill-Fehlschläge zeigen auf dieselbe Position

Reine Offline-Auswertung bereits versiegelter Dateien
(`experiments/identity_forensics/divergence_positions.py`). Kein Modellstart,
kein Hardwarelauf, keine Datei verändert.

Anlass: die Amdahl-Rechnung desselben Tages zeigt, dass Prefill `79,84 %` der
registrierten Anfrage ist und damit die einzige Hebelklasse mit echtem
Spielraum. Genau diese Klasse ist im Projekt geschlossen — Kandidat 1
(Präfix-/KV-Wiederverwendung) und Kandidat 2 (Blockgrößen-Policy) stehen auf
`candidate_correctness_failed`. Die Frage war, woran genau sie gescheitert sind.

**Befund.** Über vier Quelldateien, zwei unabhängige Mechanismen, vier
Promptlängen und fünf Blockgrößen gibt es `49` Identitätsbeobachtungen: `38`
identisch, `11` abweichend. Die Divergenzposition ist:

| Position der ersten Abweichung | Fälle |
| --- | --- |
| `10` | `10` |
| `20` | `1` |

Alle elf Abweichungen im Detail:

| Quelle | Konfiguration | erste Abweichung |
| --- | --- | --- |
| chunk_identity | Prompt `677`, Chunk `128`, 6 Blöcke | `10` |
| chunk_identity | Prompt `677`, Chunk `512`, 2 Blöcke | `10` |
| chunk_identity | Prompt `1997`, Chunk `64`, 32 Blöcke | `10` |
| chunk_identity | Prompt `1997`, Chunk `512`, 4 Blöcke | `10` |
| chunk_confirmation | Prompt `1513`, Chunk `256`, 6 Blöcke | `10` |
| chunk_confirmation | Prompt `1513`, Chunk `256`, 6 Blöcke (Wiederholung) | `10` |
| prefix_reuse | Präfix `666` von `677` | `10` |
| prefix_reuse | Präfix `1326` von `1337` | `10` |
| prefix_reuse | Präfix `4406` von `4417` | `20` |
| prefill_chunking | `512+Rest`, 2 Blöcke | `10` |
| prefill_chunking | `128er`, 6 Blöcke | `10` |

**Warum das gegen einen strukturellen Fehler spricht.** Ein falscher
KV-Cache, ein falsches Attention-Fenster oder eine falsch zusammengesetzte
Blockstruktur macht sich am **ersten** generierten Token bemerkbar, weil der
Zustand ab dann falsch ist, und die Position streut über Konfigurationen.
Beobachtet wird das Gegenteil: dieselbe späte Position, über zwei Mechanismen,
die nichts miteinander zu tun haben, und über Promptlängen von `677` bis
`1997`. Der einzige Ausreißer `20` ist der längste und inhaltlich andere
Prompt (`4417`) — also eine andere Promptfamilie mit einer anderen sensiblen
Position.

**Hypothese, ausdrücklich noch nicht belegt.** An genau dieser Position liegen
die beiden besten Kandidatentoken so dicht beieinander, dass jede Änderung der
Fließkomma-Akkumulationsreihenfolge das `argmax` kippt. Chunking und
Präfixwiederverwendung ändern beide die Summationsreihenfolge im Prefill. Der
Mechanismus wäre dann korrekt und die Workload degeneriert — nicht umgekehrt.

**Was das nicht heißt.** Ein gekipptes `argmax` ist trotzdem ein Bruch der
Tokenidentität, und das Gate hat richtig ausgelöst. Es wird hier **kein**
Vorschlag gemacht, das Identitätsgate aufzuweichen, zu tolerieren oder
schwellwertbasiert umzudeuten; Schwellwerte bleiben unantastbar. Die
Konsequenz ist eine andere: bei bestätigter Hypothese ist die richtige Antwort,
für eine Prefill-Studie eine Promptfamilie ohne degenerierte Position zu
registrieren — nicht das Gate zu ändern.

**Nebenbefund, der F1 entlastet.** Die für F1 registrierte Workload — Prompt
`897`, Chunk `256`, vier Blöcke — steht in `chunk_confirmation` auf
`identical=true`. F1 liegt also auf einem identitätssauberen Punkt.

**Nächster Schritt und Kill-Kriterium** stehen als Backlog-Eintrag P2. Ein
einziger gegateter Kurzlauf entscheidet: Top-2-Logit-Abstand je generierter
Position für Prompt `677`. Ist der Abstand an Position `10` an der
Auflösungsgrenze, ist die Hypothese bestätigt und die Prefill-Klasse wieder
offen. Ist er groß, ist die Hypothese tot und die Mechanismen sind
tatsächlich defekt.

## 2026-09-01 — R2-Korpus: was eine Freigabe wirklich hergibt

Reine Offline-Planung (`friday_optimizer/campaign.py`,
`experiments/r2_campaign/plan_r2_corpus.py`). Kein Modellstart, kein
Hardwarelauf.

**Verworfene Idee, zuerst.** Der Plan, F1s gegatete Sessions gleichzeitig als
R2-Korpus zu nutzen, trägt nicht. F1s Kandidat ist vorregistriert; jede dort
geloggte Entscheidung hätte Propensity `1,0` und damit keinerlei Überlappung.
Ein Korpus ohne Überlappung ist für Off-Policy-Evaluation wertlos. Die Idee
wird nicht weiterverfolgt.

**Was stattdessen trägt.** Eine vorregistrierte Explorationskampagne
versiegelt nicht die *Aktion*, sondern die *Regel*: Policy, Epsilon,
Seed-Basis, Punktzahl und Budget stehen vor der ersten Messung fest. Die
gezogene Folge ist aus dem Seal reproduzierbar und nachträglich nicht
umsortierbar — sie bleibt also vorregistriert und erzeugt trotzdem
Propensity-Überlappung. Umgesetzt als `CampaignPlan` mit deterministischer
Seed-Ableitung je Punkt und CLI-Kommando `campaign`.

**Budgetrealität aus versiegelter Evidenz.** Der Wall-Clock wird nicht von der
Rechenzeit bestimmt, sondern von der vorgeschriebenen Pause:

| Studie | Läufe | GPU-Arbeit | Pause | Wall | Wall je Punkt |
| --- | --- | --- | --- | --- | --- |
| matmul-compile-ab | `6` | `34,02 s` | `937,71 s` | `1000,41 s` | `166,7 s` |
| batched-readback | `6` | `20,34 s` | `625,16 s` | `672,95 s` | `112,2 s` |

Die Pause übersteigt die Rechenzeit um Faktor `28` bis `31`. Daraus folgt die
Kapazität einer Freigabe: **`10` Messpunkte je 30-Minuten-Block**, nicht
mehr. Hebel 4 aus dem Erfolgspfad („mehrere Messpunkte je Freigabe") ist damit
beziffert — er ist real, aber durch die Abkühlung gedeckelt, nicht durch die
Uhr.

**Wie viele Blöcke bis R2 auswertbar ist.** Für eine deterministische
Zielpolicy kollabieren die Gewichte auf eine Aktion, und die erwartete
effektive Stichprobe ist `Punkte × p_logging(Zielaktion)`. Bei fünf zulässigen
Kandidaten:

| Epsilon | `p(Hint)` | Ziel = gehintete Aktion | Ziel = selten gezogene Aktion |
| --- | --- | --- | --- |
| `0,2` | `0,840` | `36` Punkte, `4` Blöcke | `750` Punkte, `75` Blöcke |
| `0,3` | `0,760` | `40` Punkte, `4` Blöcke | `501` Punkte, `51` Blöcke |
| `0,5` | `0,600` | `50` Punkte, `5` Blöcke | `300` Punkte, `30` Blöcke |
| `0,8` | `0,360` | `84` Punkte, `9` Blöcke | `188` Punkte, `19` Blöcke |

Das ist die zentrale Planungszahl des Eintrags: **R2 ist mit rund fünf
freigegebenen Blöcken erreichbar**, solange die Frage lautet „schlägt die
gehintete Aktion die Baseline?". Die Frage „ist eine selten gezogene Aktion
gut?" kostet das Sechsfache und ist mit diesem Budget praktisch nicht zu
beantworten. Höheres Epsilon verschiebt genau zwischen diesen beiden Fragen.

**Querprüfung.** Die analytische Vorhersage wurde gegen die echten Schätzer
geprüft: für `50` Punkte bei Epsilon `0,5` sagt die Formel `30,00` effektive
Stichproben voraus, der gezogene Korpus liefert `29,00` — eine Ziehung
Streuung. Der Status bleibt korrekt `insufficient_data`, weil `29 < 30`; das
Gate rundet nicht zu seinen Gunsten.

**Verifikation.** Vollsuite `1485 passed, 2630 subtests passed`. Neu:
`tests/test_optimizer_campaign.py` (9) und vier CLI-Tests.

## 2026-09-02 — P2 ist gebaut und wartet nur noch auf eine Freigabe

Offline-Implementierung. Kein Modellstart, kein Hardwarelauf; der Worker
verweigert ohne `--execute` den Start und wurde ausschließlich mit
`--self-check` ausgeführt.

Der Messlauf zur Tie-Hypothese liegt vollständig vor:

- `experiments/identity_forensics/PREREGISTRATION.md` — Frage, Messung,
  vorregistrierte Klassifikation, Konsequenzen je Ausgang.
- `experiments/identity_forensics/gap_analysis.py` — die Entscheidungsregel,
  festgelegt **vor** der Messung: `structural` bei Abweichung an Position `0`
  oder `1`; `tie` nur bei Abstand ≤ `1e-2` **und** Median ≥ `20 ×` Abstand;
  bei genau einer verfehlten Bedingung `inconclusive`. Zwei unabhängige
  Formen, damit weder Skala noch Ausreißer allein entscheiden.
- `experiments/identity_forensics/measure_logit_gap.py` — ein Prozess,
  Referenzlauf plus die zwei Chunkings `128` und `512`, die bei `677` Token an
  Position `10` abwichen. AC-Pflicht, `BudgetGuard`, Pausenlogik,
  Offline-Snapshot, `release_gate`.

Der Worker bricht ab, wenn der Prompt nicht exakt `677` Token ergibt: eine
andere Promptfamilie hat eine andere sensible Position und würde eine Frage
beantworten, die niemand gestellt hat.

**Wiederholte Grenze.** Diese Arbeit fasst das Tokenidentitätsgate nicht an.
Ein gekipptes `argmax` bleibt ein Identitätsbruch. Die Schwellen in
`gap_analysis.py` klassifizieren eine Hypothese über eine Messung, nicht eine
Modellausgabe. Bei bestätigter Hypothese lautet die Konsequenz, eine
Promptfamilie ohne degenerierte Position zu registrieren.

**Verifikation.** Vollsuite `1497 passed, 2630 subtests passed`. Neu:
`tests/test_identity_forensics.py` mit 12 Tests nach der im Repository
etablierten Bauart — kein MLX-Import, kein Modell, kein Gerät; die
Entscheidungsregel direkt geprüft, der Worker per Quelltextinspektion auf
AC-Gate, Budgetguard, Offline-Resolver und fehlende Netzwerkpfade, dazu die
beiden Gate-Ausgänge (`--self-check` gibt `0`, fehlendes `--execute` gibt
`78`).

## 2026-09-02 — M1 Punkt 5: die 3,9 GB bleiben, mit Begründung

Reine Dateisystemprüfung, nichts verändert, nichts gelöscht.

**Befund.** `.friday-data/models/hub` enthält `gemma-3-1b-it-4bit` (`736 MB`)
und `gemma-3-4b-it-4bit` (`3,2 GB`); der globale HF-Cache
`~/.cache/huggingface/hub` (`26 GB`) enthält dieselben beiden plus `12b` und
ein `27B`. Die Revisionen stimmen überein
(`2d44e83dc9e80843d22fb941d3d699a0b1351aa6` und
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`). Die Blobs haben **getrennte
Inodes** — es ist also echte doppelte Belegung, keine Hardlink-Illusion und
kein `du`-Artefakt. Der größte Blob wurde zur Sicherheit beidseitig gehasht:
`94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3`,
`3 400 569 562` Byte, identisch. Der Blobname *ist* der SHA-256, also gilt das
für jeden LFS-Blob ohne weitere Prüfung.

**Entscheid: unangetastet lassen.** `resolve_local_model_snapshot` bindet per
Default genau `.friday-data/models/hub`; jede versiegelte Messung ist über
diesen Pfad aufgelöst worden. Der globale Cache ist dagegen von jedem anderen
Werkzeug auf der Maschine beschreibbar. Genau diese Isolation macht den
Snapshot-Claim überhaupt prüfbar. Ein Verweis statt der Kopie würde
Provenienz gegen `3,9 GB` Plattenplatz tauschen; Hardlinks würden die
Isolation still wieder aufheben, weil beide Pfade dann dieselben Inodes
teilten. Das im Backlog hinterlegte Kill-Kriterium („wenn Snapshots
evidenzgebunden nur dort liegen, bleibt alles unangetastet") greift damit.
`.friday-data` ist ohnehin gitignoriert, die Kopie belastet nur die Platte,
nicht das Repository.

`.friday-data/models/xet` sind `1,0 MB` HF-Xet-Logs und Staging — zu klein,
um eine Maßnahme zu rechtfertigen.

M1 Punkt 5 ist damit beantwortet und aus dem Backlog entfernt; offen bleibt
nur noch Punkt 6.

## 2026-09-02 — Roofline je Phase: wie weit ist das Gerät wirklich ausgereizt?

Reine Offline-Analyse (`experiments/roofline/phase_roofline.py`). Kein
Modellstart, kein Hardwarelauf. Quellen: der safetensors-Header des
gebundenen Snapshots, dessen `config.json` und die versiegelte
persistente-Prozess-Evidenz.

**Eingangsgrößen, exakt statt geschätzt.** Der Header nennt
`3,400 GB` Gewichte (`2,276 GB` U32-gepackt plus `1,125 GB` BF16), also
`4,55 G` quantisierte Parameter. Die `config.json` nennt `34` Layer und
Hidden `2560`. Die warme Baseline stammt unverändert aus sechs Paaren:
Prompt `897` Token, TTFT `1,7851 s`, `32` Token bei `70,99` Token/s.

**Auslastung.**

| Phase | gemessen | Gerätespitze | Auslastung | Luft |
| --- | --- | --- | --- | --- |
| Decode | `241,4 GB/s` | `400 GB/s` | `60,3 %` | `1,66×` |
| Prefill | `4,73 TFLOP/s` | `10,4 TFLOP/s` | `45,5 %` | `2,20×` |

Prefill-Arbeit ist `8,165 TFLOP` Matmul plus `0,280 TFLOP` Attention; die
Attention macht `3,32 %` aus, ist also mitgezählt und ändert das Bild nicht.
Sie wegzulassen hätte die Auslastung geschönt, nicht das Gerät.

**Was jede Phase end-to-end überhaupt wert sein kann** (32 generierte Token):

| Szenario | Ratio | Gewinn |
| --- | --- | --- |
| Decode auf Roofline (`100 %`) | `0,920052` | `7,99 %` |
| Decode realistisch (`85 %`) | `0,941523` | `5,85 %` |
| Prefill auf Roofline (`100 %`) | `0,564782` | `43,52 %` |
| Prefill realistisch (`85 %`) | `0,628871` | `37,11 %` |
| beide realistisch (`85 %`) | `0,570394` | `42,96 %` |

**Der zentrale Schluss: die Decode-Klasse ist rechnerisch erschöpft.** Selbst
bei perfekter Bandbreitenausnutzung ist die gesamte Decode-Klasse `7,99 %`
end-to-end wert, realistisch `5,85 %`. Davon sind mit `fixed_compiled` bereits
`1,42 %` gehoben. Es bleiben rund `4,4` Prozentpunkte **für alle künftigen
Decode-Kandidaten zusammen**. Ein Decode-Kandidat kann sein eigenes
Decode-Gate weiterhin bestehen — `fixed_compiled` tat das mit `7,04 %` auf der
Decodemetrik — aber er kann die End-to-End-Schwelle der F1-Studie (`10 %` im
warmen Arm) grundsätzlich nicht mehr erreichen.

Das erklärt die Projektgeschichte rückwirkend. Fused Greedy (`−0,01 %`
end-to-end), gebündelter Readback (`0,84 %`) und der Custom Metal Kernel
(`0,38 %`) waren nicht schlecht ausgeführt; sie zielten auf eine Phase, in der
kaum etwas zu holen war.

**Prefill dagegen ist offen.** Realistisch sind `37,11 %` end-to-end
verfügbar, gehoben sind mit Head-Skip `12,26 %`. Es liegen rund `25`
Prozentpunkte auf dem Tisch. Und `45,5 %` Rechenauslastung im Prefill ist ein
*niedriger* Wert für eine compute-gebundene Phase — das deutet auf echte
Ineffizienz (Dequantisierungsaufwand, fehlende Fusion, Tiling), nicht auf eine
physikalische Grenze. Das ist eine zweite Prefill-Mechanik neben der
Blockstruktur aus Kandidat 5 und wandert als solche in Backlog P1.

**Grenzen dieser Rechnung, ausdrücklich.** `400 GB/s` und `10,4 TFLOP/s` sind
Datenblattwerte für den M1 Max mit 32-Kern-GPU, auf dieser Maschine nicht
nachgemessen. Die Decode-Rechnung unterstellt, dass je Token alle Gewichte
gelesen werden. Beide Annahmen sind grob, aber in dieselbe Richtung robust:
die Kernaussage — Decode fast erschöpft, Prefill weit offen — hängt an einem
Faktor `7,1×` zwischen den Phasen und an einem Prefill-Anteil von `79,84 %`,
nicht an der zweiten Nachkommastelle. `formal_claim=false`; das ist eine
Planungsrechnung, keine Messung.

## 2026-09-02 — Der Kreuzungspunkt: das Projekt optimiert eine 32-Token-Antwort

Fortsetzung der Roofline-Rechnung desselben Tages, gleiche Quellen, gleiche
Grenzen. Offline, kein Lauf.

**Frage an die eigene Schlussfolgerung.** Die Aussage „Decode ist erschöpft,
Prefill ist offen" hing an einer Zahl, die ich nicht geprüft hatte: `32`
generierte Token. Diese Länge stammt aus der persistenten Prozessstudie und
war dort eine Budgetentscheidung, keine Aussage darüber, wie eine Anfrage
aussieht.

**Kreuzungspunkt.** Bei welcher Antwortlänge wechselt der führende Hebel? Mit
den realistischen Decken (`85 %` Auslastung je Phase):

| generierte Token | Decode-Decke | Prefill-Decke | führend |
| --- | --- | --- | --- |
| `16` | `3,25 %` | `41,27 %` | Prefill |
| `32` | `5,85 %` | `37,11 %` | Prefill |
| `64` | `9,73 %` | `30,89 %` | Prefill |
| `128` | `14,58 %` | `23,13 %` | Prefill |
| `256` | `19,40 %` | `15,39 %` | **Decode** |
| `512` | `23,25 %` | `9,22 %` | **Decode** |
| `1024` | `25,81 %` | `5,12 %` | **Decode** |

Der Wechsel liegt bei **`203` generierten Token**; dort sind beide Klassen
`17,86 %` wert, der Prefill-Anteil ist auf `38,43 %` gefallen.

**Was das über das Projekt sagt.** Jede Optimierungsstudie dieses Projekts
liegt unterhalb des Kreuzungspunkts: Head-Skip `32`, Chunk-Identity `16`,
persistenter Prozess `32`, der versiegelte Optimizer-Workload-Vertrag
(`optimizer_shadow_q2/WORKLOAD_CONTRACT.json`) `32`. Das Projekt hat also
durchgehend in dem Bereich optimiert, in dem Prefill führt — und das ist auch
der Bereich, in dem die gestrige Aussage „Decode ist zu" gilt.

Gleichzeitig hat dasselbe Projekt in *anderen* Experimenten längst mit
realistischeren Längen gearbeitet: `segmented_decode` mit `240` Token,
`self_consistency` mit `224` bis `640`, `prompt_lookup` mit `96`,
`divergence` mit `160`. Diese liegen überwiegend jenseits des
Kreuzungspunkts. Die Optimierungsseite und die Verhaltensseite des Projekts
messen also verschiedene Regime.

**Korrektur meiner gestrigen Aussage.** „Die Decode-Klasse ist erschöpft" gilt
**nur unterhalb von rund `200` generierten Token**. Für eine typische
Chatantwort ist sie nicht erschöpft, sondern die führende Klasse. Die
Priorisierung in Backlog P1 wird entsprechend eingegrenzt.

**Kosten eines zweiten Arms**, gerechnet mit der Pausenlogik der bestehenden
Worker (`(1−0,15)/0,15` Pause je Sekunde GPU-Arbeit) und dem gemessenen
Punktpreis von `167 s`:

| Antwortlänge | zusätzliche GPU-Zeit | Wall je Punkt | Punkte je Block |
| --- | --- | --- | --- |
| `32` | — | `167,0 s` | `10` |
| `128` | `1,35 s` | `176,0 s` | `10` |
| `256` | `3,16 s` | `188,0 s` | `9` |
| `512` | `6,76 s` | `212,1 s` | `8` |

Ein Arm bei `256` Token kostet also `9` statt `10` Punkte je Freigabe. Das ist
billig für die Frage, ob das Projekt das richtige Regime optimiert.

**Was ich nicht getan habe.** Ich habe F1 nicht stillschweigend erweitert. F1
erntet bestätigte Gewinne in genau dem Regime, in dem sie bestätigt wurden —
das ist methodisch richtig, und ein zusätzlicher Arm bei `256` Token würde die
Bindung an die versiegelte Evidenz lockern. Die Regimefrage steht deshalb als
eigener Backlog-Eintrag W1 und ist eine Entscheidung des Nutzers, keine
stillschweigende Ausweitung.

## 2026-09-02 — M1 letzter Punkt: Systembinaries werden vor dem Staging abgewiesen

Offline-Änderung an `friday_optimizer/session.py`. Kein Modellstart, kein
Hardwarelauf.

**Kill-Kriterium des Backlogeintrags geprüft — es greift nicht.** Der Eintrag
durfte entfallen, „wenn jede reale Session ohnehin nur den venv-Interpreter
staged". Das stimmt fast: `_purelib_binding` in `ironmule_adapter.py` verlangt,
dass der gebundene Interpreter der laufende ist. Aber der erste Zweig lautet
`if sys.prefix == sys.base_prefix: return None, None` — außerhalb eines venv
wird ohne Fehler durchgelassen. Ein Lauf mit `/usr/bin/python3` als laufendem
und gebundenem Interpreter ist damit erreichbar.

**Warum das teuer ist.** Der Runner führt eine *Kopie* der allowlisteten Binary
aus; macOS killt die Kopie einer Apple-signierten Systembinary mit SIGKILL.
Der Fehler zeigt sich erst nach dem Prozessstart als `exit:-9` — also nachdem
ein freigegebener Messblock angebrochen ist. Ein Block fasst nach der
Budgetrechnung vom selben Tag genau `10` Messpunkte; ihn an einer
undurchsichtigen `-9` zu verlieren, ist der teuerste denkbare Weg, diesen
Fehler zu lernen.

**Guard.** `SubprocessStageRunner._validate_spec` weist ein Executable
zurück, dessen realer Pfad unter `/usr/bin/`, `/usr/sbin/`, `/usr/libexec/`,
`/bin/`, `/sbin/` oder `/System/` liegt, mit der Meldung „stage executable is a
system binary and cannot be staged". Kein Unterprozess, kein `codesign`-Aufruf,
keine neue Abhängigkeit — nur ein Präfixvergleich, der genau die Fehlerklasse
trifft. Ein legitimer Lauf staged immer den Projektinterpreter und wird davon
nicht berührt; ein Test prüft genau das mit.

Damit ist Backlog M1 vollständig abgeschlossen und aus der Datei entfernt.

**Verifikation.** Vollsuite `1499 passed, 2630 subtests passed`. Zwei neue
Tests: die Abweisung der Systembinary vor jedem Prozessstart und der Nachweis,
dass der Projektinterpreter weiterhin durchläuft.

**Nebenwirkung, bewusst jetzt.** Die Änderung an `friday_optimizer/session.py`
verändert `code_manifest_sha256`. Weil noch keine der offenen Studien (F1, P2,
W1) versiegelt ist, ist genau jetzt der richtige Zeitpunkt für solche
Änderungen; nach dem Versiegeln wäre jede eine Neuversiegelung.

## 2026-09-02 — W1 gebaut, und ein Befund über F1s eigene Schwelle

Offline-Implementierung. Kein Modellstart, kein Hardwarelauf; der Worker
verweigert ohne `--execute` und wurde nur mit `--self-check` ausgeführt.

**Warum W1 keine A/B-Studie braucht.** Die Regimefrage hängt nur an drei
Größen — TTFT, Antwortlänge und Decode-Rate. Es genügt, die Baseline bei zwei
Antwortlängen im selben Prozess zu vermessen; Kandidaten müssen dafür nicht
geschaltet werden. Damit kostet W1 keinen ganzen Freigabeblock, sondern einen
Bruchteil davon.

**Was gemessen wird.** Ein Prozess, ein Prompt von rund `900` Token, nach
einem Warmlauf `32` Token als Kontrolle und `256` Token. Aufgezeichnet werden
TTFT, Decode-Dauer, Gesamtrate und die Rate im ersten und letzten Viertel der
Schritte — der Verlauf, nicht nur der Mittelwert, weil der KV-Cache über die
Antwort wächst.

**Warum die Frage scharf ist.** Bei `256` Token liegen die beiden bestätigten
Kandidaten rechnerisch `0,38` Prozentpunkte auseinander:

| Ratenabfall | `head_skip` | `fixed_compiled` | führend |
| --- | --- | --- | --- |
| `0 %` (`70,99` tok/s) | `5,09 %` | `4,71 %` | `head_skip` |
| `10 %` (`63,89` tok/s) | `4,73 %` | `4,87 %` | **`fixed_compiled`** |
| `20 %` (`56,79` tok/s) | `4,36 %` | `5,04 %` | **`fixed_compiled`** |

Die Kandidaten-Kreuzung liegt bei `276` Token, die Decken-Kreuzung bei `203`.
Ein Ratenabfall von zehn Prozent verschiebt sie unter `256`. Genau deshalb
wird gemessen statt angenommen; die Toleranz `±10 %` steht vor der Messung
fest.

**Befund über F1, aus derselben Rechnung.** F1s kombinierter warmer Arm ist
längenabhängig:

| generierte Token | kombinierter Gewinn |
| --- | --- |
| `32` (registriert) | `13,68 %` |
| `128` | `11,18 %` |
| `256` | `9,80 %` |
| `512` | `8,69 %` |

Bei `256` Token unterschreitet derselbe Kandidat F1s eigene vorregistrierte
Schwelle von `10 %`; die Studie meldete dann korrekt `below_threshold`. F1
bleibt richtig — es erntet bestätigte Gewinne in dem Regime, in dem sie
bestätigt wurden — aber die Zahl `13,68 %` darf nicht als allgemeines Ergebnis
gelesen werden. Der Geltungsbereich steht jetzt ausdrücklich in F1s
Vorregistrierung, und F1 wird dafür nicht erweitert.

**Verifikation.** Vollsuite `1510 passed, 2630 subtests passed`. Neu:
`tests/test_w1_regime.py` mit 11 Tests derselben Bauart wie bei P2 — kein
MLX-Import, kein Modell, Entscheidungsregel direkt geprüft, Worker per
Quelltextinspektion, beide Gate-Ausgänge.

## 2026-09-02 — F1s Ausführungspfad geprüft: er trägt, aber nicht für die registrierte Workload

Offline-Prüfung, kein Lauf. `session-plan` wurde read-only gegen die
versiegelten Q2-Artefakte ausgeführt.

**Der Pfad funktioniert.** Der Planer läuft durch, erzeugt ein vollständiges
Plandokument und schließt korrekt fail-closed mit benannten Gründen:

```
blocked_reasons: fingerprint_mismatch, optimizer_head_mismatch,
                 optimizer_source_identity_mismatch,
                 optimizer_code_manifest_mismatch
```

Alle vier sind Provenienzbindungen gegen die inzwischen veraltete
Q2-Vorregistrierung (`code_manifest_sha256 e2579250…` gegen aktuell
`fc9e46c6…`), kein struktureller Defekt. Zwei Bedienfallen fürs Protokoll: der
Interpreterpfad muss aufgelöst übergeben werden, sonst `symlink_path_refused`,
und das Feld `optimizer_head` im Plan ist der von der Vorregistrierung
geforderte Wert, nicht der aktuelle — ich hatte es zuerst falsch gelesen.

**Der eigentliche Befund: zwei Promptlängen.** Der gegatete IronMule-Pfad
fährt `ironmule.tune.DEFAULT_PROMPT` mit **`322`** Prompt-Token
(`context_bucket: prompt_tokens_322`, `max_tokens: 32`). F1s Evidenz stammt
aber aus zwei Quellen mit unterschiedlicher Länge:

| Gewinn | gemessen bei | Quelle |
| --- | --- | --- |
| persistenter Prozess `−65,30 %` | Prompt `897` | `experiments/persistent_process` |
| Prefill-Head-Skip `−15,4 %` | Prompt `897` | formale Head-Skip-Studie |
| `fixed_compiled` `−7,04 %` | Prompt `322` | `experiments/matmul_compile_ab` |

Meine Projektion vom selben Tag hat durchgehend das `897`-Profil benutzt. Das
ist für die Decode-Zahl vertretbar — eine Decode-Ratio hängt kaum an der
Promptlänge —, aber es heißt: F1s Kopfzahl gilt für eine Workload, die der
vorhandene gegatete Harness nicht fährt.

**Wie stark es sich auswirkt:**

| Prompt | Antwort | Prefill-Anteil | `head_skip` | `fixed_compiled` | kombiniert |
| --- | --- | --- | --- | --- | --- |
| `897` | `32` | `79,84 %` | `12,26 %` | `1,42 %` | `13,68 %` |
| `322` | `32` | `58,70 %` | `9,02 %` | `2,91 %` | **`11,93 %`** |
| `897` | `256` | `33,11 %` | `5,09 %` | `4,71 %` | `9,80 %` |
| `322` | `256` | `15,09 %` | `2,32 %` | `5,98 %` | `8,30 %` |

(TTFT bei `322` linear aus der gemessenen `897`-Zeit skaliert, weil Prefill
compute-gebunden ist; das ist eine Schätzung, keine Messung.)

Nur eine der vier Zellen hat komfortablen Abstand zu F1s vorregistrierter
`10 %`-Schwelle. Auf der Workload, die der Harness tatsächlich fährt
(`322`/`32`), bleiben `1,93` Prozentpunkte Abstand statt `3,68`.

**Entscheidung, die beim Nutzer liegt.** Drei Wege:

1. **F1 auf `322`/`32` registrieren** — die Workload, die der gegatete Harness
   fährt. Nutzt die vorhandene, auditierte Infrastruktur; `fixed_compiled`s
   Evidenz kommt ohnehin von dort. Erwartung sinkt auf `11,93 %`.
2. **Eigenen F1-Worker bauen** wie bei P2 und W1, mit dem `897`-Prompt. Passt
   zur Evidenz von Head-Skip und persistentem Prozess, dupliziert aber
   Messinfrastruktur, die es schon gibt.
3. Schwelle senken — **ausgeschlossen**. Eine Schwelle an das anzupassen, was
   herauskommen soll, ist genau der Fehler, den die Vorregistrierung verhindert.

Empfehlung: Weg 1. Eine Studie, die ihr Harness nicht fahren kann, ist keine
Studie. Die Prefill-Ratio des Head-Skip ist ein Phasenverhältnis und sollte
über die Promptlänge übertragen, ist aber nur bei `897` bestätigt — F1 misst
den zusammengesetzten Pfad ohnehin direkt und muss das nicht annehmen.

Kein Code geändert; die Entscheidung wird nicht stillschweigend getroffen.

## 2026-09-02 — R2-Pipeline trockengelaufen: sie funktioniert, kostet aber 40 Blöcke

Offline-Systemtest der Kette R0 → Kampagne → R1, bevor sie die teuerste
Ressource des Projekts verbraucht. Kein Modell, keine Hardware; die Belohnungen
sind synthetisch und begründen keinerlei Performanceaussage. Geprüft wird die
Maschinerie, nicht das Modell.

**Aufbau.** In `experiments/r2_campaign/recover_ground_truth.py` wird eine
bekannte Wahrheit eingepflanzt — je Aktion ein Verhältnis, Rauschen
`sd=0,010`, `5 %` zensierte Läufe — eine Kampagne gezogen, alles in eine echte
Optimization Memory geschrieben (Hashkette wird verifiziert) und anschließend
mit den echten Schätzern zurückgewonnen.

**Ergebnis bei `400` Punkten:** alle fünf Zielaktionen erreichen `ok`,
Medianfehler `0,21` Prozentpunkte, schlechtester Fehler `0,52`, die Rangfolge
wird exakt reproduziert. Die Kette trägt.

**Ergebnis nach Korpusgröße — und die Korrektur meiner eigenen Planzahl:**

| Punkte | Blöcke | belastbare Schätzungen | Rangfolge |
| --- | --- | --- | --- |
| `50` | `5` | `0/5`, ESS `4`–`26` | falsch |
| `150` | `15` | `1/5` | zufällig richtig |
| `400` | `40` | `5/5` | richtig |

Die am 2026-09-02 genannten „rund fünf Blöcke" beantworten genau **eine**
Frage: „schlägt die gehintete Aktion die Baseline?" — und auch das nur bei
Epsilon `0,5`. Dieser Trockenlauf benutzte Epsilon `0,6`, dort erreicht selbst
die gehintete Aktion bei `50` Punkten nur ESS `26` und bleibt unter der
Untergrenze. Eine **vollständige Rangfolge über alle fünf Aktionen kostet rund
`400` Punkte, also `40` Freigabeblöcke** — etwa zwanzig Stunden gegatete
Messzeit. Das ist die realistische Eintrittskarte für R2. Die Zahl gehört
korrigiert ins Backlog, damit niemand mit `5` plant.

**Fund im Code, durch den Trockenlauf ausgelöst.** Beim Versuch, mehrere
Zielaktionen zu bewerten, zeigte sich: `replay` bewertete jede Ziel-Policy
unter den *Hints des Loggers*. Damit konzentrierte sich jede deterministische
Ziel-Policy zwangsläufig auf dieselbe Aktion, und nur eine einzige Frage war
überhaupt stellbar. Das ist keine Kleinigkeit — es hätte R2 auf eine
Ein-Aktions-Auswertung verengt.

Behoben durch `target_hints` in `ips`, `snips`, `doubly_robust`, `replayer`
und `evaluate`. Der Default bleibt die Kontext-Lesart (geloggte Hints, gilt für
jede Policy); ein expliziter Wert bewertet eine Policy, die anders gehintet
hätte. Der Korpus wird dabei nie verändert, nur die bepreiste Policy — ein
Test prüft genau das. Die Formulierung in `campaign.py`, Replay bewerte
*immer* unter den geloggten Hints, war damit falsch und ist angeglichen.

**Nebenbeobachtung.** Bei `150` Punkten war die Rangfolge bereits korrekt,
obwohl vier von fünf Schätzungen unter der Untergrenze lagen. Die Untergrenze
ist für Ordnungsentscheidungen konservativer als für Größenaussagen. Das
rechtfertigt keine Absenkung; es wäre allenfalls ein Grund, ein eigenes
vorregistriertes Ordnungsgate zu entwerfen — nicht ungefragt.

**Verifikation.** Vollsuite `1513 passed, 2630 subtests passed`; drei neue
Tests für `target_hints`.

## 2026-09-02 — F1 ist ausreichend bestimmt, aber nur solange das Rauschen klein bleibt

Offline-Simulation gegen die echte Entscheidungsfunktion
(`experiments/f1_integration/power_f1.py`, `400` Versuche je Zelle). Kein
Modell, keine Hardware; die Stichproben sind synthetisch und begründen keine
Performanceaussage. Geprüft wird, ob F1 überhaupt entdecken kann, was es zu
entdecken behauptet.

**Anlass.** Auf der Workload, die der gegatete Harness fährt, erwartet F1
`11,93 %` gegen eine Schwelle von `10 %` — `1,93` Punkte Marge. Ob sechs Paare
(`MIN_PAIRS`) dafür reichen, hatte niemand gerechnet.

**Rauschen aus versiegelter Evidenz.** Head-Skip-Kalibrierung
`session_ratio_sd = 0,004526` (`0,45 %`), sechs Prozesspaare relative Streuung
`0,734 %`. Nebenbefund: dieselbe Kalibrierung rechnete ein `raw_mde` von
`0,0074` aus, das vom vorregistrierten Boden auf `0,05` angehoben wurde — das
Gate ist rund siebenmal konservativer als das Messrauschen verlangt.

**Trefferquote bei wahrem Gewinn `11,93 %`:**

| Paare | `0,5 %` | `1,0 %` | `2,0 %` | `3,0 %` | `5,0 %` |
| --- | --- | --- | --- | --- | --- |
| `6` | `100,0 %` | `99,2 %` | `65,2 %` | `39,2 %` | `22,2 %` |
| `12` | `100,0 %` | `100,0 %` | `86,8 %` | `61,3 %` | `30,2 %` |
| `20` | `100,0 %` | `100,0 %` | `97,5 %` | `76,2 %` | `36,2 %` |
| `30` | `100,0 %` | `100,0 %` | `99,8 %` | `88,5 %` | `49,0 %` |

**Falschqualifikation bei wahrem Gewinn `8 %`**, also unter der Schwelle:
zwischen `0,0 %` und `0,8 %` in jeder Zelle. Die Regel irrt in die sichere
Richtung — sie qualifiziert nicht, was nicht qualifiziert gehört.

**Schluss.** Bei dem Rauschen, das dieses Projekt tatsächlich misst, ist F1
mit sechs Paaren gut bestimmt. Das Risiko liegt nicht in der Paarzahl, sondern
darin, dass die zusammengesetzte End-to-End-Messung verrauschter sein könnte
als die Einzelphasenmessungen es waren. Genau das messen F1s A/A-Sessions
ohnehin.

**Konsequenz, vorregistriert.** Die Paarzahl je Arm folgt jetzt verbindlich
aus dem in den A/A-Sessions gemessenen Rauschen: ≤ `1 %` → sechs Paare,
≤ `2 %` → zwanzig, ≤ `3 %` → dreißig (grenzwertig, wird vermerkt), > `3 %` →
die Studie ist auf dieser Schwelle unterbestimmt und bricht vor dem A/B
terminal ab. Angepasst wird die Stichprobe, nicht das Kriterium; die `10 %`
bleiben unberührt. Regel steht in F1s Vorregistrierung, Abschnitt 4b.

**Verifikation.** Vollsuite `1516 passed, 2630 subtests passed`; drei neue
Regressionstests halten die drei Kernaussagen fest — sechs Paare genügen bei
`0,8 %`, sie genügen bei `3 %` nicht, und eine Wahrheit unter der Schwelle
wird in keinem von vierzig Läufen qualifiziert.

## 2026-09-02 — Die Lücke zwischen Messung und Urteil geschlossen

Offline-Ergänzung. Kein Modellstart, kein Hardwarelauf.

**Befund.** Der Planpfad war geprüft, die Analyse war geprüft — aber nicht die
Verbindung dazwischen. Der gegatete Sessionpfad gibt die gepaarten Messungen
bereits in genau der `MetricSample`-Drahtform aus (`_metric_dict` im
Stage-Worker erzeugt sie, `baseline_samples`/`candidate_samples` tragen sie).
F1s Entscheidungsfunktion `evaluate_integration` nimmt genau diese Objekte.
Nur: **nichts verband beides.** Nach einem F1-Lauf hätte Evidenz vorgelegen
und kein Urteil, und die Glue-Arbeit wäre unter Zeitdruck nach einem
verbrauchten Messblock entstanden.

**Geschlossen mit `integrate`.** Neues read-only CLI-Kommando; es liest eine
oder mehrere Ergebnisdateien, benutzt denselben Sample-Leser, den
`RealSession` ohnehin anwendet (`_unwrap_stage_payload` und `_metric_samples`),
und ruft `evaluate_integration`. Nichts am Ergebnisformat wird neu
interpretiert, und `integration.py` bleibt reine Analyse — der Leser sitzt im
CLI, damit das Analysemodul nicht die schwere `real_session`-Kette importieren
muss.

**Nebenbefund aus den Tests.** Werden zwei Ergebnisdateien mit denselben
Paar-IDs zusammengeführt, meldet die Auswertung `rejected` beziehungsweise
`inconclusive` statt dieselbe Evidenz doppelt zu zählen. Der
Duplikat-Guard des Evaluators greift also auch über Dateigrenzen — das war
nicht selbstverständlich und ist jetzt festgehalten.

**Verifikation.** Vollsuite grün; vier neue CLI-Tests: Urteil aus einem
Ergebnis, `below_threshold` ohne es zum Fehler zu erklären, Duplikaterkennung
über zwei Dateien, und Abweisung eines Ergebnisses ohne gepaarte Samples.
Der Auswertepfad steht als Abschnitt 4c in F1s Vorregistrierung.

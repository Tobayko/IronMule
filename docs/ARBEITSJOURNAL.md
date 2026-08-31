# Arbeitsjournal

## 2026-08-31 — Q3a vorregistrierter Pfadinteraktions-Pilot

- **Entscheidung:** Q3a vergleicht den finalen Q2-Incumbent (`compiled_fixed_cache=True`, `head_skip_prefill=True`, `readback_every=2`) mit demselben Arm plus `fused_argmax=True`; es gibt keine Promotion oder Aktivierung.
- **Umsetzung:** `docs/BACKLOG.md`, `research/raw/Q3a_preregistration.md`, der SHA-256-Begleiter, der stdlib-only Dry-Run-/Worker-Harness und die Unit-Tests wurden ergänzt. Der Worker verlangt nun eine einmalige Parent-Pipe/Nonce-Capability und die exakt gepinnte lokale 4B-Identity, bevor IronMule/MLX importiert werden.
- **Messung:** Keine Hardware-/MLX-Ausführung. Der Dry-Run bindet 6 Kinder, 7 Wiederholungen, 2 Warmups, 35-Sekunden-Kindtimeouts, 240-Sekunden-Workerbudget und 300-Sekunden-Gesamtdeadline. Loadavg-, Prozess-, AC-, Thermal-, Swap- und Git-Gates sind vorab und vor jedem Kind vorgesehen.
- **Verifikation:** Q3a-Unit-Tests, Python-Kompilierung, Dry-Run-No-Write, Hash- und `git diff --check` werden nach der Review-Härtung erneut ausgeführt.
- **Offene Risiken:** RSS bleibt eine konservative Child-High-Water-Mark statt einer per Arm gemessenen Größe; GPU-Auslastung wird nicht behauptet, Prozessinventar und Loadavg sind nur ein Proxy. Die 35-Sekunden-Kindgrenze ist ein Pilotlimit, kein Performanceversprechen.

## 2026-08-31 — Q3a-Safety-Nachhärtung im uncommitteten Worktree

- **Änderung:** Der Q3a-Worker startet die einzige neue Prozessgruppe; `ab.run`-Kinder erben sie. Direkte Child-Timeouts werden bereinigt, und bereits abgeschlossene Child-Records werden über `ABRunError` zurückgegeben. Ausgabe und JSON-Konstanten sind begrenzt/strict.
- **Änderung:** Q3a bindet vor dem Worker-Import das vollständige Python-Ausführungssurface (`ironmule/*.py` plus Q3a) und den exakt gepinnten 4B-Manifest-Hash; Prozess-, Thermal-, Load-, Swap-, AC- und Git-Kommandos bleiben absolute, fail-closed Gates.
- **Änderung:** Resultatschema validiert exakte Arme, Referenz-Tokens, deterministische Per-Repeat-Felder und eindeutige Progress-Marker; unerwartete Ausführungsfehler werden als `FAILED` mit `BASE`-Fallback persistiert.
- **Verifikation:** `py_compile`, Q3a-Dry-Run/No-Write, Preregistration-SHA, `git diff --check` und injizierte Preflight-/Schema-/Marker-Checks bestanden. `pytest` war in den vorhandenen Python-3.12/3.14-Umgebungen nicht installiert; keine Installation und keine Hardware-/MLX-Ausführung vorgenommen.
- **Nachprüfung:** Bei Timeout wird auch nach beendetem Gruppenführer nochmals `SIGKILL` an die Prozessgruppe versucht, damit Nachkommen nicht still weiterlaufen; Git-Statuszeilen und fehlende System-/Identity-Evidence verweigern jetzt strikt.

## 2026-08-31 — Q3a-P1/P2-Abschlusskorrektur

- **Änderung:** Die Worker-Prozessgruppe bleibt die einzige Gruppe; `ab.run`-Kinder erhalten keine eigene Session und werden direkt mit `terminate → wait → kill → wait` bereinigt. Der äußere Q3a-Timeout beendet die Worker-Gruppe nur bis zum erfolgreichen `wait`.
- **Änderung:** Befehle verwenden höchstens 1 s, der gemeinsame monotone Pilot reserviert 10 s für Postflight, und Postflight prüft erneut Loadavg sowie Prozessinventar. `ps` validiert PID/RSS/%CPU strikt und blockiert aktive native Ollama/llama.cpp/Claude- sowie weitere bekannte Modellaktivität.
- **Änderung:** Thermal ist nur mit beiden separaten nominalen `pmset`-Zeilen gültig; Resultate benötigen sechs exakt an Raw gebundene Progress-Marker, exakte Preflight-Identity und erlauben negative Swap-Änderungen innerhalb des Delta-Limits. Fehler-Resultate bewahren begrenzte Partial-Children plus Marker.
- **Verifikation:** Preregistration wurde inhaltlich aktualisiert und SHA neu berechnet; Hardware-/MLX-Ausführung bleibt ausgeschlossen.

## 2026-08-31 — Q3a Safety Review: Prozess-/Deadline-/Gate-Fixes

- **Entscheidung:** Der Worker ist die einzige neue Prozessgruppe; direkte A/B-Kinder erben die Worker-PGID. Direkte Timeout-Bereinigung eskaliert nur bei Bedarf (`terminate → wait → kill → wait`), der äußere Worker-Kill beendet nach erfolgreichem Wait sofort.
- **Änderung:** OS-Kommandos sind auf 1 s begrenzt und deadline-gebunden; 10 s bleiben für Postflight reserviert. Drei Loadavg-Samples liegen 1 s auseinander (Sleeper injizierbar), und Postflight wiederholt Load-/Prozess-Gates.
- **Änderung:** `ps` prüft strikt PID/RSS/%CPU und blockiert aktive native Ollama/llama.cpp/Claude-Prozesse unabhängig von Python; Thermal akzeptiert nur beide separaten nominalen Warnzeilen. Erfolg verlangt sechs 1:1 an Raw gebundene Marker und vollständige Preflight-Identity. Swap-Deltas dürfen negativ sein, solange sie unter 256 MiB bleiben.
- **Änderung:** Unvollständiges Child-JSON wird vor Aggregation als indexierter `ABRunError` abgelehnt. Normale Worker-Fehler bewahren begrenzte Partial-Children und Marker; der noch gepufferte Streaming-Cap ist als P2-Backlog dokumentiert.
- **Verifikation:** `py_compile`, Dry-Run, Gate-/Parser-/Deadline-/Dirty-Gate-Smokes, Prereg-SHA und `git diff --check` bestanden. `pytest`/`uv run --offline pytest` waren nicht ausführbar (`pytest` fehlt; uv-Cache nicht zugreifbar); keine Installation, kein MLX und keine Hardware-Ausführung.
- **Letzte P1-Korrektur:** Outer-Timeout prüft die gesamte PGID per `killpg(pgid, 0)` und erzwingt SIGKILL nur bei verbliebener Gruppe; Thermal akzeptiert nur die beiden exakten No-Warning-Zeilen plus sichere CPU-/Nullwerte; native `llama-server`/`llama-cli`/`mlx_lm`/ähnliche Prozesse blockieren unabhängig von CPU/RSS, während inaktives Claude dem Load-Gate überlassen bleibt.
- **Robustness/Data-Consistency:** `_finite` und Schema-Validation lehnen riesige JSON-Integer ohne Overflow ab. `per_arm`-Summaries, Child-Medians, Paare, Median-Verhältnis und deterministische Bootstrap-CIs werden vollständig aus Raw-Daten rekonstruiert; gefälschte Timing-/Ratio-/Summary-Werte failen geschlossen. Worker-`communicate()`-Timeout und -OSError nutzen denselben bounded Gruppen-Cleanup mit Reap.
- **Flag-Consistency:** `token_identity`, `token_count_identity`, `stop_reason_identity` und `deterministic` werden aus sämtlichen Raw-Armen und Per-Repeat-Feldern rekonstruiert und exakt gegen die gemeldeten Top-Level-Flags geprüft; forged candidate token/count/stop/physical data failen vor Interpretation.

## 2026-08-31 — Q3a-Abschluss-Reconciliation und Readiness

- **Verifikation:** Targeted Suite `127 passed`; breite Suite `336 passed, 1 skipped`; separate MLX-Testmodell-Integration `12 passed`; Gesamtstand `348 passed, 1 skipped`. `ab`-Self-Check und `tune`-Self-Check beendeten sich mit Exit 0; der vorhandene `runpy`-Warnhinweis beim Tune-Self-Check bleibt rein diagnostisch. `py_compile`, `git diff --check`, Dry-Run und Preregistration-SHA `eb9cefd97d37af938689e0bcca66d8418628ed76097157da28f940e2a5ecf2ec` bestanden.
- **Abgrenzung:** Es wurde kein Gemma-/Real-Performance-Benchmark ausgeführt und kein 27B-Modell verwendet. Die MLX-Integrationseinheitstests liefen separat und sind nicht als Performance-Evidence zu interpretieren.
- **Readiness:** AC und Thermal waren nominal; der freie Systemspeicher lag bei 50 %. Der Swap-Stand betrug 1101.62 MiB und überschritt damit das 256-MiB-Gate; auch der Load war während der Beobachtung nicht sauber. Der Q3a-Modelstart bleibt daher korrekt blockiert und wurde nicht ausgeführt.
- **Reviewstatus:** Die finalen Reviews zeigen keine offenen P0/P1-Findings. Das bekannte P2-Risiko des gepufferten Worker-Output-Caps bleibt als Backlog-Eintrag offen. Vorherige append-only Einträge sowie Preregistration und SHA wurden nicht verändert.

## 2026-08-31 — Q3a-Abschluss-Reconciliation und Readiness (final)

- **Verifikation:** Targeted Suite `127 passed`; breite Suite `336 passed, 1 skipped`; separate MLX-Testmodell-Integration `12 passed`; Gesamtstand `348 passed, 1 skipped`. `ab`-Self-Check und `tune`-Self-Check beendeten sich mit Exit 0; der vorhandene `runpy`-Warnhinweis beim Tune-Self-Check bleibt rein diagnostisch. `py_compile`, `git diff --check`, Dry-Run und Preregistration-SHA `eb9cefd97d37af938689e0bcca66d8418628ed76097157da28f940e2a5ecf2ec` bestanden.
- **Abgrenzung:** Kein Gemma-/Real-Performance-Benchmark und kein 27B-Modell; die separaten MLX-Integrationseinheitstests liefen, sind aber keine Performance-Evidence.
- **Readiness:** AC und Thermal waren nominal, freier Systemspeicher 50 %. Der Swap-Stand lag bei 1101.62 MiB und damit über dem 256-MiB-Gate; auch der Load war während der Beobachtung nicht sauber. Der Q3a-Modelstart bleibt korrekt blockiert und wurde nicht ausgeführt.
- **Reviewstatus:** Keine offenen P0/P1-Findings. Das bekannte P2-Risiko des gepufferten Worker-Output-Caps bleibt im Backlog. Vorherige append-only Einträge sowie Preregistration und SHA wurden nicht verändert; Raw-Evidence ist unter `research/raw/Q3a_preflight_refusal_20260831.json` erhalten.

## 2026-08-31 — Q3a-Preflight-Portability-Fix

- **Ausgangslage:** Der Stand von Commit `771d133` blieb korrekt `FAILED` mit `BASE`-Fallback, ohne Modellstart und mit `partial_children=0`. Die ursprüngliche Gate-Lesung war bei Thermal/Situation teilweise `unknown`; der beobachtete Load-Maximalwert lag bei `8.696`. Die vorhandene Raw-Evidence unter `research/raw/Q3a_preflight_refusal_20260831.json` und der ignorierte Datenpfad wurden unverändert erhalten.
- **Änderung:** Low-Power wird nun ausschließlich über die absolute öffentliche Foundation-Abfrage `/usr/bin/osascript -l JavaScript -e 'ObjC.import("Foundation"); JSON.stringify($.NSProcessInfo.processInfo.isLowPowerModeEnabled)'` mit exakt `true`/`false`-Auswertung gelesen; der ungültige `pmset -g lowpowermode`-Pfad ist entfernt. Thermal normalisiert optional exakt das reale `Note: `-Präfix, verlangt weiterhin beide No-Warning-Zeilen und erlaubt nur bekannte CPU/GPU-Null-/No-status-Zeilen. Swap entfernt abschließende Newlines und parst reale Werte wie `used = 1553.81M`; das manuelle Ist-Ergebnis betrug `1553.81 MiB`.
- **Sicherheitsgrenzen:** Nur die absolute `ps`-Inventarabfrage darf bis 512 KiB lesen; alle anderen Kommandos bleiben bei 64 KiB. Größere Inventare failen geschlossen, und argv wird nie persistiert. Sicherheits-Schwellen, Preregistration und Preregistration-SHA sowie die 4B-/No-27B-Bindung blieben unverändert.
- **Verifikation:** Die modellfreie Q3a-Suite bestand mit `26 passed`; `py_compile` und `git diff --check` bestanden. Ein direkter stdlib-`system_environment()`-Diagnoselauf startete kein Modell und importierte kein MLX; auf der isolierten Umgebung ergab er `AC`, Low-Power `false`, Thermal/Swap wegen OS-Berechtigungsfehlern `unknown`/`None`. Kein 27B-Modell und kein Hardware-/MLX-Performance-Test wurden ausgeführt.

## 2026-08-31 — Q3a korrigierter zweiter Preflight

- **Ergebnis:** Der korrigierte Lauf auf Commit `0ec9237` blieb korrekt `FAILED` mit `BASE`-Fallback, `promotion_allowed=false`, ohne Modellstart und mit `partial_children=0`. Das Raw-Ergebnis `research/raw/Q3a_preflight_refusal2_20260831.json` blieb als ignorierte Evidence erhalten; die beiden ignorierten Q3a-Refusal-Raw-Dateien sind unverändert vorhanden.
- **Gates:** Grün waren AC, Git-Bindung, installierter Speicher, Low-Power-off, exakte lokale 4B-Identity (Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, Manifest `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`), Preregistration-SHA, Zeitbudget und Thermal. Rot waren Loadavg mit `max=21.723` und `spread=0.699`, Swap mit `1641021440` Bytes (über dem 256-MiB-Limit) sowie aktive Claude-Prozessaktivität. Ein 27B-Modell wurde nicht verwendet.
- **Replay-/Resume-Grenze:** Offline-Replay ist derzeit nur für `BASE` zulässig; die Datenbasis ist für eine belastbare adaptive/RL-Aussage unzureichend, daher ist RL nicht anwendbar. Ein späterer Q3a-Start ist ausschließlich bei AC, Low-Power-off, nominalem Thermal, Load `<=4` und Spread `<=1`, Swap `<=256 MiB`, ohne Modell- oder aktive-Claude-Prozesse sowie mit eindeutigem neuem Outputpfad zulässig.
- **Verifikation:** Targeted `127` Tests, breite Suite `336 passed, 1 skipped`, separate Integration `12 passed`, modellfreie Q3a-Suite `26 passed`; `ab`-/`tune`-Selfchecks waren erfolgreich. Der bekannte gepufferte Worker-Output-Cap bleibt ein P2-Backlogpunkt. SQuAD-Daten bleiben untracked/lokal und die Lizenzfrage offen; PR #2 bleibt owner-only.

## 2026-08-31 — Q3a-Versuch 3: Preflight verweigert den Modellstart

- **Ausgangslage und Scope:** Der ausdrücklich autorisierte Lauf wurde im Worktree auf HEAD `28b2ef4` ausgeführt; der geprüfte Q3a-Code stammt aus `0ec9237`. Verwendet wurde ausschließlich der exakt vorregistrierte lokale `mlx-community/gemma-3-4b-it-4bit`-Arm, Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, Manifest-SHA `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`. Kein Download, kein 27B-Modell und keine Codeänderung.
- **Ergebnis:** Der Harness schrieb `research/raw/Q3a_attempt3_20260831.json` (ignoriert, SHA-256 `019b1aafbc7342a782476d62e2921e8141b185c50490df4b72d5c0f0fbb9e8a2`) und endete vor jedem Modellstart mit `FAILED`, `BASE`, `promotion_allowed=false` und `partial_children=0`. Es wurden keine Modellkinder gestartet und keine Prozesse beendet.
- **Gates:** Grün waren AC, Git-Bindung, installierter Speicher bekannt, Low-Power-off, exakte 4B-Identität, Preregistration, Zeitbudget und Thermal. Rot waren Swap `1774777794 B` (`1692.56 MiB`, Limit `256 MiB`), Loadavg max `6.1792` (Limit `4`) und das Prozess-Gate `no_competing_model_process=false`.
- **Redigierte OS-Evidence:** Obwohl der Nutzer Claude als beendet meldete, fanden die Gates zwei echte `Claude`-Executables: PID `55295` mit CPU `4.1%` und RSS `292688 KiB` sowie PID `55345` mit CPU `2.8%` und RSS `395344 KiB`. Argumente wurden nicht gespeichert; kein Prozess wurde beendet. Selbst nach dem Stoppen von Claude würden Swap und Loadavg den Q3a-Start weiterhin blockieren.
- **Messgrenze:** Analyse blieb `PATH_INTERACTION_ONLY`; es gibt keine Modellmessung, kein Ergebnisverhältnis und keine Performance-Evidence. Sicherste Fortsetzung ist nach gesicherter Nutzerarbeit und Neustart bzw. sauberem Zustand; kein automatischer Neustart. Das untracked SQuAD-File blieb unberührt.

## 2026-08-31 — Q3b Residual-Swap-Safety-Canary preregistration and implementation

- **Entscheidung:** Wegen des ausdrücklich gewünschten Verzichts auf einen Neustart
  wurde ein separater Q3b-Safety-Canary eröffnet. Er nutzt ausschließlich die lokal
  gepinnte Gemma-4B-Revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, nie 27B,
  und darf weder Promotion noch Performance-/RL-Aussagen erzeugen.
- **Mechanismus:** Baseline und `fused_argmax`-Kandidat laufen in zwei separaten
  frischen Single-Arm-Stage-Workern über den bestehenden `ironmule.ab.run`-Pfad;
  der Worker lädt die unveränderten Q3a-Helfer erst nach der Capability-Prüfung.
  Pro Stage werden Warmup 1, drei Roh-Repeats und `max_tokens=32` verwendet. Ein
  begrenzter 0,25-s-Sampler hält die maximale Swap-Nutzung während der gesamten
  Stage fest; eine vollständige Post-Stage-Snapshot-Gate entscheidet vor Stage 2.
- **Sicherheitsgrenzen:** Start-Swap bekannt und `<=4 GiB`, Swap-Anstieg vom
  Startmaximum `<=128 MiB`, freier Speicher `>=35%` am Start und `>=20%` je Stage,
  MLX/RSS `<=60%` des installierten Speichers, Load `max<=8`/Spread `<=2`, AC,
  Low-Power-off, Thermal nominal, Child 35 s, Worker 120 s, Gesamt 180 s mit
  Cleanup-Reserve. Nur die exakt verifizierte Claude-Desktop-Executable wird
  ignoriert; generisches/unklares Claude bleibt blockiert.
- **Verifikation:** Keine Hardware-/MLX-Ausführung, kein Download und kein Commit
  in diesem Arbeitsschritt. Preregistration und SHA wurden vor Tests angelegt und
  nach der Review-Härtung aktualisiert. Q3a-Dateien und Q3a-SHA blieben unverändert.
- **Offene Risiken:** Die reale Memory-Pressure-Ausgabe und Swap-Sampler-Granularität
  bleiben plattformabhängige Beobachtungen; unbekannte Werte führen fail-closed zu
  `FAILED`/`BASE`. Ein bestandener Canary ist ausschließlich `SAFETY_CANARY_PASS`
  mit `performance_valid=false` und `promotion_allowed=false`.

## 2026-08-31 — Q3b P1-Review-Korrekturen

- **Änderung:** Die Claude-Desktop-Ausnahme prüft nun das sicher geparste, exakte
  `argv[0]`-Token. Der kanonische Pfad ist erlaubt; `ClaudeX`, generische Claude-
  CLI/Server-Pfade und fehlerhaftes Quoting bleiben harte Blocker.
- **Änderung:** Der Stage-Worker nimmt synchron einen Start- und End-Swap-Sample
  sowie periodische 0,25-s-Samples auf. Werte, monotone Zeitstempel und Worker-
  Start-Offsets werden in gleich langen, auf 512 Einträge begrenzten Arrays
  gespeichert. Jeder Command-/Read-/Parse-/Thread-/Zeitfehler wird in
  `sampler_errors` festgehalten und beendet die Stage; erfolgreiche Resultate
  verlangen `sampler_errors=[]`, mindestens zwei Samples und maximal 1,75 s
  Zeitabstand.
- **Änderung:** Das Stage-Gate verlangt zusätzlich den bekannten frischen Swap-
  Endpunkt. Die Cross-Stage-Identity vergleicht neben Tokens, Counts und Stops
  nun Kapazitäten, Decode-Schritte, Prompt-Tokens und Determinismus.
- **Preregistration:** Q3b-Preregistration und SHA wurden vor jeder Hardware-
  oder Modellmessung aktualisiert; SHA `35854a6c13dcbf93ab3ad19b2e4dd90620dd11583831e29a2c85c573b285a7c2`.
- **Verifikation:** Q3b-Tests `20 passed`, `py_compile`, `git diff --check` und
  Dry-Run-No-Write bestanden. Keine Hardware-/MLX-Ausführung, kein Download,
  kein Modellstart und kein Commit.

## 2026-08-31 — Q3b P1 Live-Swap-Abbruch behoben

- **Befund:** Der periodische Q3b-Sampler setzte bei Swap-Fehlern oder einem
  Überschreiten des 128-MiB-Highwaters bisher nur sein Stop-Event; eine aktive
  `ironmule.ab`-Kindprozessgruppe konnte dadurch weiterlaufen. Der synchrone
  Worker-Start und `before_child` prüften den Highwater ebenfalls nicht streng
  gegen die Parent-Referenz.
- **Änderung:** Der Worker vergleicht den synchronen Startwert vor jedem
  IronMule-Import mit dem Parent-Initialwert. `before_child` lehnt Samplerfehler
  und überschrittenes Highwater ab. Während eines aktiven Kindes wird Safety-
  Evidence einmalig und begrenzt (`reason`, Samples, monotone Zeiten/Offsets,
  Fehler, ohne argv) erfasst, als flushbarer `@SAFETY`-Marker ausgegeben und
  unmittelbar mit `os.killpg(os.getpgrp(), SIGTERM)` beendet; Kill-/Markerfehler
  bleiben fail-loud.
- **Parent-/Cleanup-Fix:** `_start_stage` erkennt `@SAFETY` auch ohne finalen
  `@@`-Marker, bewahrt Safety- und Partial-Evidence, bereinigt/reapt die
  Workergruppe auf Safety-, Nonzero-, Malformed- und No-Marker-Pfaden und
  kennzeichnet den Nachweis `group_gone`. Der finale synchrone Read nach einem
  beendeten Kind löst keinen Live-Kill mehr aus.
- **Verifikation:** Q3b-Suite `23 passed`; `py_compile`, `git diff --check`,
  Preregistration-SHA und Dry-Run geprüft. Keine Hardware-/MLX-Ausführung,
  kein Download, kein Modellstart und kein Commit. Q3a blieb unverändert.

## 2026-08-31 — Q3b finaler P1-Swap-Read und TERM-Fallback

- **Befund:** Der finale Swap-Read lief zwar nach dem Child-Reap, konnte aber
  bei Read-/Samplerfehlern oder einem späten Highwater-Verstoß noch in einen
  normalen Worker-Fehlerpfad ohne `@SAFETY` fallen. Außerdem behandelte der
  Safety-Kill einen fehlgeschlagenen TERM nicht mit einer unmittelbaren
  KILL-Eskalation.
- **Änderung:** Der finale Sample wird jetzt strikt gegen Samplerfehler und
  `max(samples) - initial_swap > 128 MiB` geprüft. Jeder Verstoß erzeugt ein
  einmaliges begrenztes `@SAFETY`-Event und verhindert den Erfolgsmarker, auch
  wenn ein injizierter Kill-Helfer zurückkehrt. `_capture_live_safety` versucht
  bei TERM-Fehler unmittelbar SIGKILL auf derselben PGID; erst wenn beide
  Signale scheitern, werden Kill-Fehler ausgegeben und der Pfad fail-loud
  beendet.
- **Verifikation:** Q3b-Suite `27 passed`; `py_compile`, `git diff --check`,
  Preregistration-SHA und Dry-Run geprüft. Keine Hardware-/MLX-Ausführung,
  kein Download, kein Modellstart und kein Commit. Q3a blieb unverändert.

## 2026-08-31 — Q3b Claude-Prozess-Gate auf Bundle-Vertrauen korrigiert

- **Befund:** Die bisherige Ausnahme vertraute auf ein einzelnes `argv[0]`-Token
  und konnte weder Helper-/Crashpad-Prozesse noch den vollständigen signierten
  App-Bundle-Zustand belegen. Ein einzelnes `ps`-Inventar war zudem nicht gegen
  die separate `comm`-Darstellung abgeglichen.
- **Änderung:** Q3b fragt nun zwei begrenzte absolute `ps`-Inventare ab und
  verlangt eine strikte PID-Karte. Claude wird nur ignoriert, wenn der
  whitespace-erhaltende `comm`-Pfad lexikalisch innerhalb des exakten
  `/Applications/Claude.app/Contents/`-Baums liegt und der komplette Bundle-
  Vertrauenshelfer mit `/usr/bin/codesign --verify --deep --strict` sowie
  `-dv --verbose=4` exakt Identifier `com.anthropic.claudefordesktop`, Team
  `Q6L2SF6YDW` und die erste Authority `Developer ID Application: Anthropic PBC
  (Q6L2SF6YDW)` bestätigt. Unbekannt, fehlerhaft, außerhalb oder untrusted
  bleibt blockiert; Modellmuster werden vor der Claude-Ausnahme weiter blockiert.
- **Verifikation:** Q3b-Tests decken Desktop-, Helper- und Crashpad-Pfade,
  generische/außerhalb liegende Claude-Pfade, Boundary-Spoofing, Trust-
  Metadaten-/Verify-Fehler, PID-Mismatch und malformed `comm` ab. Es wurde kein
  Hardware-/MLX-Test, kein Modellstart, kein Download und kein Commit ausgeführt;
  Q3a-Code und Q3a-SHA blieben unverändert.

## 2026-08-31 — Q3b Canary-Versuch 2: Preflight verweigert den Modellstart

- **Ergebnis:** Der ausdrücklich gestartete Q3b-Lauf schrieb
  `research/raw/Q3b_canary2_20260831.json` (ignorierte Raw-Evidence,
  SHA-256 `90e040f090111bb990377b4e8bfecfd8ec5a3753288955353f97cffdac1d5a2c`)
  und endete mit `FAILED`, `BASE`, `promotion_allowed=false`, ohne Modellkind
  und ohne Stage-Start.
- **Gates:** Alle übrigen Preflight-Gates waren grün: AC, Low-Power-off,
  Thermal nominal, exakte lokale 4B-Identity, Git-Bindung, Preregistration,
  installierter Speicher und Load. Der Startspeicher lag bei `66%`, Swap bei
  `1707668930 B` und der Loadavg-Maximalwert bei `1.7099609375`. Rot blieb nur
  `no_competing_model_process`, weil die zwei `ps`-Inventare wegen eines
  zwischen den Snapshots verschwundenen/neu erschienenen PID-Eintrags nicht
  als identische Gesamtmenge behandelt werden konnten.
- **Korrektur:** Die Q3b-Preregistration wurde vor weiterer Messung um die
  PID-Race-Regel ergänzt und neu gehasht. Das Gate wertet nun relevante
  `args`-Records strikt aus: fehlendes `comm` ist nur bei per injizierbarem
  `kill(pid, 0)`-Probe nachgewiesenem Prozessende tolerierbar; alive,
  permission-denied und unknown failen geschlossen. Extra `comm`-Records sowie
  irrelevante fehlende Records werden ignoriert; Modell-/Inference-Tokens
  blockieren weiterhin direkt. Wiederholte Pre-Child-/Post-Stage-Gates bilden
  die verbleibende nicht-atomare Snapshot-Grenze.
- **Deskriptive Ausgabe:** Für einen vollständigen Safety-PASS ist zusätzlich
  eine klar als `descriptive_only=true` markierte Timing-Zusammenfassung
  vorregistriert: Mediane aus den exakten Raw-Repeats, Token-/Decode-Zähler,
  endliche Durchsatzformeln und Kandidat/Baseline-Ratios mit
  richtungsabhängiger Prozentformel (`100*(1-ratio)` für Zeiten,
  `100*(ratio-1)` für Durchsatz). Sie trägt `performance_valid=false`,
  `order_confounded=true`, `statistical_confidence=none`, enthält weder CI,
  Winner noch Promotion und wird bei FAILED/unvollständigen Stages nicht
  berechnet; sie ist kein Gate.
- **Messgrenze:** Kein Modell wurde gestartet, keine Hardware-/MLX-Leistung
  gemessen und keine Performance-, Optimierungs- oder RL-Aussage abgeleitet.
  Q3a blieb unverändert.

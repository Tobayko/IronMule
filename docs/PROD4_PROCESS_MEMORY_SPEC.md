# PROD4-P — 12B-Ladung mit tatsächlichem Prozess-Footprint

Vorregistrierung 2026-09-07, vor dem ersten Modelllauf dieses Diagnosepfads.

Frage: Was belastet den eigenen Worker während des bekannten 12B-Abbruchs?
RSS allein genügt nicht: zuletzt +468.587.643 B System-Swap bei 792.756.224 B
Worker-RSS, noch vor `ready`. Der neue, explizite `--process-memory`-Pfad ergänzt
den bestehenden Load-only-Test, ohne Loader, Gewichte oder Grenzen zu verändern.

- Öffentliche Darwin-API `proc_pid_rusage`, exakt Flavor 4 und SDK-verifizierte
  Struktur `rusage_info_v4` (296 Bytes). Nicht CURRENT/V6 mit einem V4-Puffer.
- Zusätzliche tatsächliche Werte: resident, physical footprint, lifetime/interval
  peak footprint, wired bytes, opaker Prozessstartwert, Messzeit/-dauer.
  Keine Prozessnamen, UUIDs, Energieannahmen oder erfundenen verfügbaren Bytes.
- Nach vollständiger Modell-/Umgebungsbindung und stabiler Readiness zunächst
  fünf Sekunden Kontrollphase ohne Modellworker, circa 250-ms-Samples des
  eigenen Controllers plus System-Swap. Dann genau eine 12B-Modellladung mit
  derselben Zusatzmessung für die tatsächlich besessene Kind-PID.
- Unverändert: Netzbetrieb, Low Power aus, Thermik-/Last-/Readiness-Gates,
  maximal 256 MiB zusätzlich beobachteter Swap und 60 % RAM für RSS/MLX-Peak,
  Timeout, Pflichtpausen, BudgetGuard und bestätigtes Prozessende.
- Messwerte werden vor der Beurteilung im vorhandenen EventJournal gespeichert.
  Fehlende Zusatztelemetrie stoppt die Diagnose; ein gleichzeitig bereits
  beobachteter Speicherfehler bleibt primär und wird nicht überschrieben.
- Keine Inferenz in diesem Diagnosepfad, kein Speedup-/Speicherersparnisbeweis,
  keine neue Qualifikation aus einem einzelnen Ladewert und kein versteckter
  Retry. Der bekannte Abbruch wird nicht mit erhöhten Limits wiederholt.

Interpretation: Footprint und RSS sind verschiedene Größen und dürfen nicht
addiert oder als garantiert ineinander enthalten behandelt werden. Steigt der
Kind-Footprint zeitgleich mit Swap nach stabiler Kontrollphase, unterstützt dies
Ladedruck als Erklärung, beweist aber noch keine einzelne Loader-Unteroperation.
Bleibt der Kind-Footprint flach, muss Fremdlast/anderes Systemverhalten weiter
geprüft werden. Die Kontrollphase ist keine Garantie für spätere Fremdlastfreiheit.

Kill: unklare ABI, fehlende/ungültige Beobachtungen, unbestätigtes Cleanup,
veränderte Quellen/Modelle/Umgebung, verletzte Gates oder eine Aussage, die aus
RSS/Footprint bereits universelle Modell-Admission oder Bandbreite ableitet.

## Nach bestandenem Load-only-Test: echte 12B-Referenzgenerierung

Der Diagnoseversuch `ffab58946c0a47fd9db22524bb9ff263` hat die Ladung und
Readiness mit unveränderter Identität, Null-Swapdelta und normalem Exit bestanden.
Vor dem jetzt folgenden Generierungslauf wird festgelegt: derselbe 12B-Snapshot,
ein Warmup plus drei greedy Acht-Token-Anfragen mit dem unveränderten Apples-
Prompt, exakter Token-/Text-/Finish-/Zählervergleich mit der tatsächlich
gemessenen Stock-Referenz aus PROD1 Versuch 3. Der vorhandene Referenz-Harness
wird nur auf diese zweite fest gebundene Modellrevision erweitert.

Pro Anfrage maximal sechs Sekunden, anschließend mindestens vier Sekunden Pause;
die bestehende Last-/Swap-/RSS-Prüfung und maximal 60 % RAM MLX-Peak bleiben aktiv.
Maximal 120 Sekunden für diesen kurzen Gesamtversuch. Beobachtete Ausgabedaten
werden nur gehasht gespeichert, auch bei einem verletzten Gate. Kein
Geschwindigkeitsvergleich, keine Aktivierung und kein Claim, dass die geänderte
Systemlage den früheren Swapfehler grundsätzlich behoben hätte.

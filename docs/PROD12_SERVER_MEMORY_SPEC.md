# PROD12 — getrennte Server-/Worker-Speicherbeobachtung

## Frage und Grenze

PROD12 beantwortet ausschließlich PROD10 Punkt 3: Wie unterscheiden sich die
beobachteten Speicherwerte des echten HTTP-Serverprozesses von denen seines
Modellworkers? Es ist weder ein Speedup-Versuch noch eine Wiederholung der
einstündigen PROD10-S-Prüfung. RSS und Physical Footprint bleiben
prozessbezogene Beobachtungen, keine Aussage über freien Gesamtspeicher.

## Vorregistriertes Protokoll

- Modell: `mlx-community/gemma-3-12b-it-4bit`, Revision
  `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.
- Ein frischer, mit `python -I` gestarteter Serverprozess besitzt `ProductStore`,
  `ProductService`, HTTP-Server und Modell-Lease. Nur dieser Prozess startet den
  bestehenden `MLXWorkerClient`; der Modellworker ist sein Kind. Der gewählte
  Python-Interpreter muss die zuvor qualifizierte Wheel-Distribution enthalten;
  der Harness fügt den Repository-Root ausdrücklich nicht zu `sys.path` hinzu.
- Beide Prozesse werden neutral im Abstand von einer Sekunde beobachtet. Die
  Modellbeobachtungen werden einzeln zum Controller gestreamt und dort
  gejournalt; das Serverkind hält keine wachsende Telemetriehistorie. Diese
  Kadenz ist Ressourcenbeobachtung und ausdrücklich kein Performancebenchmark.
- Der Controller lädt weder MLX noch das Modell. Er sendet reale Loopback-HTTP-
  Requests und beobachtet Server-PID und Modellworker-PID getrennt.
- Plan: `long_8`, `short_32`, `long_32`, je vier Aufrufe (erster Warmup), danach
  genau ein Burst mit vier Clients. Insgesamt 16 endliche Anfragen.
- Ausgabe-Gate: Prompt-/Completion-Zahl, Finish-Grund und Text-Hash müssen je
  Fall den Stock-Zeilen aus
  `research/raw/PROD10_12B_open_20260907_attempt2.json` entsprechen. Der Bericht
  bindet zusätzlich deren exakten Output- und Token-Hash, obwohl HTTP selbst
  keine Token-IDs exponiert. Prompts und Ausgabetext werden nie persistiert.
- Vor und nach dem Requestplan werden Quellmanifest, installierte Pakete,
  Providerdistribution, Modellsnapshot und Runtime-Identität innerhalb der vom
  Serverkind gehaltenen Modell-Lease gebunden.

## Ressourcen- und Fehlerregeln

Generation, Laufzeit, RSS, Swap, Duty Cycle und Pausen besitzen keine
künstlichen Abbruchgrenzen. Sampling steuert die Last nicht. Protokollframes,
Loopback-Antworten und Cleanup-Wartezeiten bleiben endlich, weil sie
Kontrollflächen und keine Hardwarelimits sind. Es gibt keinen automatischen
Retry. Fehler behalten Teil-Samples, Prozessdaten und Abschlussstatus; eigene
Kinder werden beendet und tatsächlich gewartet. Fremde Prozesse bleiben
unangetastet.

## Gates und Kill-Kriterien

Bestanden ist der Lauf nur mit 16 referenzidentischen HTTP-Antworten,
Telemetrie beider verschiedenen PIDs, stabiler Provenienz und Exitcode 0 für
Server und Modellworker. Vermischte PIDs, fehlende Telemetrie, geänderte
Referenz, unvollständiger Cleanup oder eine abweichende Ausgabe beenden den
Lauf negativ. Native Ausführung erfolgt ausschließlich mit `--execute`; reine
Tests starten weder MLX noch ein Modell.

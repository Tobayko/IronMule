# PROD10 — lokale Modell- und Serverprüfung ohne künstliche Hardwaregates

## Ergebnis der Integrationsmatrix

**Gemma 3 1B, 4B und 12B haben jeweils 35 vollständige echte Anfragen bestanden.**
Zusätzlich wurde je Modell ein echter Streaming-Disconnect ausgelöst und die
korrekte Folgeantwort auf demselben geladenen Worker nachgewiesen. Kein
versteckter Neustart. Alle sechs Stock-/Produktworker endeten normal mit Code 0.

Die drei Fälle sind 1.077 Prompttokens mit Ausgabelimit 8/32 und 16 Prompttokens
mit Limit 32. Je Fall: ein Warmup und drei aufgezeichnete Wiederholungen in
einem unabhängigen Stock-MLX-LM-Prozess, danach ebenso im Produktworker.
Zusätzlich je Fall JSON/SSE, ein Vierfach-Clientburst und eine Recovery-Antwort.
Der Backendpfad serialisiert Anfragen; vier gleichzeitige Clients sind kein
Nachweis von vierfach paralleler GPU-Inferenz oder dynamischem Batching.

| Modell | Vollständige Anfragen | Warmes long8, Produkt-Median | Prozess-Footprint-Peak | Stock-MLX-Peak |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 1B 4bit | 35/35 | 0,512371 s | 2.083.882.784 B | 1.612.622.432 B |
| Gemma 3 4B 4bit | 35/35 | 2,130594 s | 4.059.057.224 B | 3.455.666.348 B |
| Gemma 3 12B 4bit | 35/35 | 6,441981 s | 9.108.642.008 B | 8.284.434.536 B |

Die Zeiten sind deskriptive Mediane aus drei aufgezeichneten Wiederholungen,
keine AB/BA-gepaarten Speedup-Claims. Modelle unterscheiden sich in Qualität
und Größe; diese Tabelle erklärt kein Modell zum generell besten. Footprint
und MLX-Peak sind unterschiedliche, überlappende Größen und werden nicht addiert.
Alle langen Prompts hatten tatsächlich 1.077 Tokens. Kein Telemetriefehler.
Ein unsteter oder langfristiger Speicheranstieg ist damit noch nicht ausgeschlossen.

## Neuer Nutzerentscheid und Gültigkeit

Die explizite Nutzerfreigabe hebt für neue lokale Prüfungen die künstlichen
6-s-/Arbeitszeit-/Duty-/Pausen-/Readiness-/RSS-/Swap-Abbruchgates auf. Sie sind im
neuen Runner nicht durch andere versteckte Hardwarelimits ersetzt. Ressourcen
werden weiterhin beobachtet. macOS-/Thermalschutz, Datenschutz, Ausgabekorrektheit,
Benutzerabbruch und kontrolliertes Cleanup bleiben aktiv. Keine Kaggle-Nutzung.

Der öffentliche Produktzustand unterstützt ausdrücklich `request_timeout_s=null`
über `ProductStore.set_request_timeout(None)`; der Service reicht die unbegrenzte
Anfrage an den Backendpfad weiter. Bestehende Konfigurationen behalten ihren
120-s-Standard als explizite Request-Policy. Startup kann ebenfalls explizit
ohne Deadline gestartet werden. Frühere versiegelte Studien bleiben unverändert.

Alle erfolgreichen Läufe binden denselben installierten Code
`47c416fb3ec3ae8f875a02e95abde01f554a0c5711eee5a0576f77b42edf7b7b`
und dieselbe Bibliotheksumgebung
`6e32542c2cd4d2950ec828640d196c7e91be2b4790439041770c93cb7f60ee95`:
Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3, NumPy 2.5.2, Transformers 5.15.1.
Modell-/Code-/Umgebungs-/Hardwareidentität, Provider-, Metadaten- und Quellhashes
sind vor/nach jedem erfolgreichen Lauf gleich. Gerät: M1 Max, 32 GiB.

## Tatsächlicher Fehler und Korrektur

Der erste neue 12B-Lauf erreichte bereits 34 korrekte vollständige Anfragen,
scheiterte aber nach dem echten Disconnect: `cancelled_requests=1`,
`ready=false`, Folgeanfrage HTTP 503. Ursache im Service: Das Schließen eines
noch nicht vollständig gelesenen Backend-Streams machte dessen Protokollzustand
unbrauchbar und verwarf den Worker.

Die Korrektur signalisiert Cancel und liest denselben Iterator bis zu seinem
geprüften terminalen Frame leer; verworfene Antwortteile werden nicht gespeichert.
Nur danach bleibt der Worker verwendbar. Fehler beim Leeren beenden ihn weiter;
es gibt keinen automatischen Neustart/Retry. Native Nachprüfung auf allen drei
Modellen: nach Disconnect `ready=true`, genau ein Cancel, keine aktiven/queued
Anfragen, exakte Folgeantwort, derselbe Produkt-PID.

Der erste Fehlerbericht bleibt erhalten und wird nicht mit den erfolgreichen
Läufen gepoolt. Seine separate Nachlauf-Identitätsprüfung bestätigt unveränderte
Identität. Ein unabhängiges Read-only-Review bestätigt alle vier Berichte gegen
das kettenverifizierte Journal.

Der erste Codezustand lässt sich aus Commit `0498e22` mit
[`PROD10_DEADLINE_BASELINE.patch`](../research/PROD10_DEADLINE_BASELINE.patch)
rekonstruieren (`git apply --unidiff-zero`). Der Patch enthält nur die vier
eigenen Baseline-Änderungen und keine Daten/Änderungen der parallelen DATA1-Arbeit.

## Diagnose und offene Arbeit

Beim ersten 12B-Lauf liegt die Stock-Phasengrenze am ersten Token für warmes
long8 bei 6,343–6,381 s, rund 96,1 % der Anfragezeit. Das weist auf den Weg
bis zur ersten Ausgabe als Schwerpunkt hin. Die Phasen stammen aus dem
unveränderten MLX-LM-Callback nach Cache-/First-token-Evaluation: **keine reinen
GPU-Zeitstempel**, keine isoliert bewiesenen Compilerkosten. Der alte 6-s-Test
konnte diese vollständig korrekten längeren Antworten nicht abwarten.

Das separat vorregistrierte [einstündige Serverprofil ist inzwischen bestanden](PROD10S_RESULTS_2026-09-08.md).
Offen bleiben eine weitergehende GPU-/Engpassdiagnose sowie autonome
Optimierung/RL mit echten Reward-/Validierungsdaten. Keine Performanceaktivierung und kein allgemeiner
Produktions-, Multi-Mac- oder Lernclaim aus dieser Integrationsmatrix.

## Reproduzierbare Quellen

- [Vorregistrierung und Ablauf](PROD10_OPEN_VALIDATION_SPEC.md)
- [12B vor Korrektur](../research/raw/PROD10_12B_open_20260907_attempt1.json)
- [Separater Identitätsaudit](../research/raw/PROD10_12B_open_20260907_attempt1_identity_audit.json)
- [12B nach Korrektur](../research/raw/PROD10_12B_open_20260907_attempt2.json)
- [1B bestanden](../research/raw/PROD10_1B_open_20260907_attempt1.json)
- [4B bestanden](../research/raw/PROD10_4B_open_20260907_attempt1.json)

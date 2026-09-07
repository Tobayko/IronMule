# PROD8 — echter 12B-Langkontextversuch, 2026-09-07

**Terminal fehlgeschlagen am Zeitgate.** Der Stock-MLX-LM-Referenzprozess wurde
auf dem lokalen M1 Max/Metal geladen und bestätigte den unveränderten Prompt mit
1.077 Tokens. Der erste reale Generierungsaufruf lieferte innerhalb des festen
6-s-Host-Timeouts keine vollständige Antwort. Er wurde bei beobachteten
6,002551458 s abgebrochen; keine Wiederholung und keine Schwellenänderung.

Das ist weder ein erfolgreicher Langkontexttest noch ein gemessener Speedup.
Die Produkt-, JSON- und SSE-Stufen wurden **nicht erreicht**. Der erfolgreiche
kurze [90-Anfragen-12B-Lauf](PROD6_12B_RESULTS_2026-09-07.md) bleibt ein separater
Nachweis mit anderem Prompt und wird nicht mit diesem Fehlversuch vermischt.

## Tatsächlich beobachtet

- Lauf `a04826bffa5b4fe9b626553603167706`, Fehlcode `stock_request_timeout`.
- Gefrorener lokaler 12B-4bit-Snapshot
  `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`, unveränderte installierte Runtime.
- Stock-Worker meldete `Device(gpu, 0)` und 1.077 Prompttokens.
- Ein tatsächlicher Generierungsaufruf begonnen, null vollständige Antworten.
- Elf Readiness-Beobachtungen, 79 Prozess-/Swapbeobachtungen.
- Maximal beobachtetes Swapdelta 0 B, RSS maximal 3.840.688.128 B.
- MLX-Ladepeak 7.188.274.696 B; **kein** vollständiger Inferenzpeak verfügbar.
- Eigener Worker PID 46891 beendet/geerntet, Returncode −15 (kontrollierter
  SIGTERM-Abbruch, kein normaler erfolgreicher Exit). Kein Produktworker gestartet.
- Quellmanifest vor/nach gleich; die für einen erfolgreichen Lauf geplante
  vollständige Abschlussidentität wurde nach dem Fehler nicht erreicht.

Das BudgetGuard-Summenfeld enthält hier 0 s, weil `record_gpu` den Block über
6 s vor der Aufnahme in die akzeptierte Summe ablehnt. **Dies bedeutet nicht,
dass keine GPU-Arbeit stattfand.** Der fehlgeschlagene Aufruf samt
6,002551458 s konservativer Host-Arbeitszeit steht separat in
`request_resource_events` und im terminalen Journal. Reine GPU-Zeit wurde nicht
gemessen; Tokenizer-, Transport-, Steuerungs- und Warteanteile werden damit
nicht voneinander getrennt.

## Konsequenz

Die eingefrorene PROD8-Ausführung bleibt geschlossen. Ein möglicher nächster
Weg wäre ein eigener, vorab geprüfter phasenweiser Budgetentwurf, der
synchronisierte Prefill-/Decode-Blöcke und echte Pausen nachweist. Das ist
**nur eine offene Hypothese** (PROD9), keine nachträgliche Rechtfertigung dieses
Laufs und keine Freigabe für anderes Chunking, gelockerte Limits oder einen Retry.

[Vorregistrierung](PROD8_LONG_CONTEXT_SPEC.md),
[vollständiger Rohbericht](../research/raw/PROD8_12B_long_context_20260907_attempt1.json).
Der Harness bleibt opt-in; acht reine Kontrolltests und statische Checks sind
Softwarebelege, nicht Ersatz für die nicht erreichten Modell-/HTTP-Stufen.

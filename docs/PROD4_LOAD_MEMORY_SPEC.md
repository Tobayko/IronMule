# PROD4 — beobachtete Ladegrenzen, keine erfundene RAM-Reserve

Vorregistrierung vom 2026-09-07, vor dem neuen Hardwarelauf.

## Mechanismus und Geltungsbereich

Der installierte PROD3-12B-Versuch wurde erst nach Laden und anschließender
Readiness-Wartephase wegen +2.457.463.685 B Swap verworfen. Der erste PROD4-Schritt
überwacht deshalb schon den eigenen startenden Worker und prüft erneut vor der
ersten Generierung. Das ist eine frühere Abbruchmöglichkeit, **keine Garantie**,
dass macOS zwischen zwei Stichproben niemals auslagert, und noch keine belastbare
Vorabprognose des verfügbaren Modell-Headrooms.

Die Elternseite fragt nur den RSS ihres eigenen Kindes und den systemweiten
Swapverbrauch ab. Zielintervall während Startup: 250 ms; tatsächliche Anfangs-/
Endzeiten werden gespeichert. Das ist keine Echtzeitgarantie. Bounded OS-Probes,
Worker-Timeout und reaping bleiben erforderlich. Nach dem Laden kommen tatsächliche
Darwin-`ru_maxrss`- und MLX-Active-/Peak-/Cache-Bytes aus dem Worker hinzu. Der
geladene MLX-LM-0.31.3-Referenzpfad verwendet `lazy=False` und evaluiert seine
Modellparameter vor `ready`. Kein Lazy-Load-Trick verschiebt den Bedarf unbemerkt
in die erste Anfrage.

## Eingefrorene Gates

- Unverändert: Swapdelta höchstens 256 MiB gegenüber dem aufgezeichneten
  Vorlade-Baselinewert, MLX-Peak höchstens 60 % des installierten RAM.
- Zusätzlich konservativ: beobachteter Prozess-RSS und dessen gemeldeter Peak
  jeweils höchstens 60 % RAM. RSS und MLX können überlappen; sie werden **nicht
  addiert**. Dies behauptet weder GPU-Speicherisolation noch einen Footprint-Ersatz.
- Fehlende, unplausible oder übergroße Telemetrie ist kein zulässiger Wert.
  Ein verletzender Messwert wird vor dem Abbruch unverändert protokolliert.
- Cancel/Pause/Timeout während Startup erhalten ihren Grund; der tatsächlich
  gestartete PID und das tatsächliche Prozessende bleiben nachprüfbar.
- Weder Energieeinstellungen noch fremde Prozesse oder deren Speicher werden
  verändert. Kein automatischer Wiederholungsversuch nach einem Fehler.
- Bestehende BudgetGuard-, Netzbetrieb-, Low-Power-, Thermal- und Lastgates bleiben
  aktiv. Die Poller laufen nicht innerhalb der gemessenen Inferenzintervalle.

## Warum keine einfache Available-Bytes-Formel?

Der installierte Apple-SDK-Header `os/proc.h` markiert
`os_proc_available_memory()` als `API_UNAVAILABLE(macos)`; außerdem beschreibt er
ein prozessbezogenes Dirty-Memory-Limit, nicht freien System-RAM.
`mach/vm_statistics.h` sagt ausdrücklich, dass speculative pages bereits in
`free_count` enthalten sind. Inaktive, purgeable, komprimierte und dateigestützte
Seiten werden daher nicht blind zu einer angeblich sicheren Reserve addiert.

Apple beschreibt Memory Pressure als Kombination mehrerer Zustände, nicht als
allokierbare Bytezahl. Der bisherige Prozentwert bleibt ein Readiness-Indiz und
kein Beweis, dass ein bestimmtes Modell passt.
[Apple: Memory usage](https://support.apple.com/en-euro/guide/activity-monitor/actmntr1004/mac).
MLX meldet den Peak seit Programmstart beziehungsweise letztem Reset; dieser
Wert wird für die Ladephase vor irgendeinem Inferenz-Reset gelesen.
[MLX: get_peak_memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html).

## Native Prüfung und Kill-Kriterien

1. Kontrolltests mit echten Pipes/Kindprozessen: Callback-Fehler, Deadline,
   KeyboardInterrupt, fehlerhafte Metadaten, Cleanup. Kleine Grenzwert-Fixtures
   sind ausschließlich Protokolltests, keine Hardware- oder Modellaussage.
2. Neues Wheel in unabhängiger Umgebung installieren; Umgebung vor/nachher
   binden. Projektumgebung und alte versiegelte Studien unverändert lassen.
3. Ein initialer Load-only-Durchgang für jeden vorhandenen Gemma-Snapshot
   (1B, 4B, 12B, aufsteigend), jeweils frischer Worker, echte stabile Readiness,
   mindestens 60 s zwischen geschlossenen Workern. Laden, 4 s Pause und erneute
   stabile Readiness bleiben überwacht. Alle Messwerte, Fehler und Cleanup
   speichern. Ein Fehler beendet den jeweiligen Modellversuch ohne Retry.
   Dieser Durchgang prüft Telemetrie/Schutzfunktion, **nicht Geschwindigkeit**;
   aus einem Ladezeitwert entsteht keine Leistungsbehauptung.
4. Erst bei gültigem Load-only-Ergebnis wird ein gesondert registrierter
   Generierungstest zugelassen. Ein bestandener Load-only-Test qualifiziert
   keine Ausgabequalität, Kontextgröße oder Performance und aktiviert nichts.

Kill: unerkannte bekannte Telemetrieverletzung, verschluckter Fehler, verwaister
Worker, erhöhte Schwelle, erfundene Headroom-Formel oder verschobene Allocation
als angebliche Verbesserung. Speicherbedarf und Modell-/Last-Admission bleiben
offen, bis reproduzierbare reale Profile eine vorsichtige Entscheidung tragen.

## Nachtrag nach PROD4-1B-Versuch 2026-09-07

Ein Load-only-Bericht gilt nur dann als bestanden, wenn der tatsächlich gestartete
Worker nach dem kontrollierten Shutdown mit Returncode `0` beendet wurde. Ein anderer
Returncode wird als `forced_abort` (wenn ein zuvor aufgezeichneter MemoryGuard-Grund
den Abbruch auslöste) oder als `crashed` klassifiziert; der ursprüngliche Guard-Grund
bleibt der primäre Fehler. Der archivierte Rohbericht des ersten 1B-Versuchs bleibt
byteidentisch und wird durch diese nachträgliche Korrektur nicht umgeschrieben.

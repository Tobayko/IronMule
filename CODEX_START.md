# Codex-Startbriefing

## Auftrag

Baue in diesem Verzeichnis einen kleinen, reproduzierbaren Demonstrator für einen hardwarebewussten
Measure–Optimize–Validate-Loop. Ziel ist nicht, ein vollständiges LLM oder einen neuen Compiler zu
entwickeln. Der erste Nachweis soll zeigen oder widerlegen, dass ein Agent aus realen Messdaten eine
Kandidatenkonfiguration auswählt, numerisch validiert, statistisch korrekt mit der Baseline vergleicht
und ein Ergebnis wiederverwendbar speichert.

## Aktueller Übergabestand — 19.08.2026

`JA — Ich gebe den Forschungspivot H0 → H1 → H2 und die Implementierung von Phase 1A/H0 mit SQLite v1, read-only Loopback-Dashboard und festem Worker Option A frei. Keine Downloads, Installationen, Custom-Metal-Kernels oder Modellgewichte.`

H0 ist offline implementiert als einzelne FP16-`2048²`-Matmul mit SQLite v1, festem
Worker Option A und read-only Dashboard auf `127.0.0.1`. Das ist kein Modelltest und
belegt noch keine reale MLX-/GPU-Performance, Correctness, Memory- oder Safety-Gate;
`mlx-run` bleibt `EXIT_MLX_LOCKED=78`/`not_released`.

## Reihenfolge für den nächsten kontrollierten Lauf

1. `AGENTS.md`, dieses Dokument, `IMPLEMENTIERUNGSPLAN.md` und
   `docs/ANWEISUNGEN_UND_DOKUMENTE.md` lesen.
2. **Zuerst immer ProjectAtlas** für eine fokussierte Übersicht des Project-Friday-Roots verwenden;
   die verschachtelte Upstream-Codebasis nicht breit öffnen. Falls MCP nicht geladen ist, den
   versionierten CLI-Fallback verwenden und den Grund im Status vermerken.
3. finalen Gesamttest und danach die Offline-Control-Historie/UI abschließen.
4. Einen separat angekündigten MLX-H0-Go/No-Go-Lauf für dieselbe einzelne Operation
   durchführen; bis dahin keine produktiven Rohdaten behaupten.
5. A/A mit 3 Charakterisierungs- und 3 Bestätigungsprozessen aggregieren.
6. Erst nach A/A-Pilot und vor jeder Kandidatensichtung die H1-Powerplanung einfrieren.
7. H2-Modelle und ein begrenzter Custom-MLX-Metal-Kandidat bleiben spätere, separat
   freizugebende Schritte.

## Architekturentscheidung

Der LLM-Agent ist zunächst ein äußerer Planer und Erklärer. Die eigentliche Suche soll deterministisch
und reproduzierbar durch Parameter-Suche, Bayesian Optimization oder ein kleines Cost Model erfolgen.
Keine LLM-Ausgabe darf ungeprüft kompiliert oder ausgeführt werden. Control Plane und Execution Plane
bleiben getrennt.

## Definition of Done für den ersten Demonstrator

- gleiche Eingaben liefern reproduzierbare Correctness-Ergebnisse;
- Baseline und Kandidat werden mit Warmup und mehreren Wiederholungen gemessen;
- der Vergleich verwendet mindestens Median und Streuung sowie eine Mindestverbesserung;
- ein schlechter Kandidat wird verworfen und ein guter Kandidat mit Hardware-/Workload-Schlüssel
  gespeichert;
- ein Lauf kann vollständig lokal ohne Cloud-LLM wiederholt werden;
- ein negativer Befund wird nicht als Fehler umgedeutet.

## Nicht tun

- keine GPU-Assembly, keine Apple-ISA-Reverse-Engineering-Arbeit;
- keine direkte Neural-Engine-Programmierung;
- keine Änderung an ProjectAtlas ohne separaten Auftrag;
- keine Performancebehauptung ohne gespeicherte Messdaten;
- keine globale Installation oder Löschung außerhalb des Projekts ohne Rückfrage.

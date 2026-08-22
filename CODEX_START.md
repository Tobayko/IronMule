# Codex-Startbriefing

## Auftrag

Baue in diesem Verzeichnis einen kleinen, reproduzierbaren Demonstrator für einen hardwarebewussten
Measure–Optimize–Validate-Loop. Ziel ist nicht, ein vollständiges LLM oder einen neuen Compiler zu
entwickeln. Der erste Nachweis soll zeigen oder widerlegen, dass ein Agent aus realen Messdaten eine
Kandidatenkonfiguration auswählt, numerisch validiert, statistisch korrekt mit der Baseline vergleicht
und ein Ergebnis wiederverwendbar speichert.

## Aktueller Übergabestand — 22.08.2026

H1-v2 und N10-v2 haben den festen N8-/N10-Batch-Dispatch jeweils formal im
exakten FP16-`2048²`-Scope bestätigt. Beide begrenzten Runtimes sowie der
darüberliegende N8/N10-Shadow-Router haben ihre vorregistrierten CPU-, MLX-,
Correctness-, Persistenz- und UI-Gates bestanden. Der Router ist auf Commit
`70bc451` gebunden, besitzt keine `execute`-Methode und erzwingt immer
`serial_shadow_only`; seine DB enthält genau zwei terminale Engineering-Records.

Der Nutzer hat die nächste Runde ausdrücklich freigegeben: notwendige Software
und ein sicher isolierter Kernelversuch sind erlaubt, neue Modelle bleiben
verboten. Der aktuelle Stand benötigt keine Installation. Das bestandene
Shadow-Gate erlaubt ausschließlich die getrennte Vorregistrierung und Prüfung
eines statischen Custom-Metal-Kandidaten; es ist kein produktives Routing-GO.

## Reihenfolge für den nächsten kontrollierten Lauf

1. `AGENTS.md`, dieses Dokument, `IMPLEMENTIERUNGSPLAN.md` und
   `docs/ANWEISUNGEN_UND_DOKUMENTE.md` lesen.
2. **Zuerst immer ProjectAtlas** für eine fokussierte Übersicht des Project-Friday-Roots verwenden;
   die verschachtelte Upstream-Codebasis nicht breit öffnen. Falls MCP nicht geladen ist, den
   versionierten CLI-Fallback verwenden und den Grund im Status vermerken.
3. Die finale Shadow-Router-Historie read-only replayen; keine der beiden
   terminalen Messungen wiederholen.
4. Vor jeder Kernelkompilierung eine neue Spezifikation für genau einen
   statischen Fusionskandidaten `residual_add + RMSNorm` einfrieren: feste
   Gemma-nahe Shape/Precision, starke MLX-Baseline, Toleranzen, Warmup,
   Wiederholungen, A/A- und A/B-Gates, Speicherbudget und Abbruchregel.
5. Den Kandidaten ausschließlich in einem kontrollierten Worker mit Timeout,
   CPU-/Adressraumgrenzen, fester Quellkonstante, Correctness-Oracle und harter
   Prozessgruppenbeendigung implementieren. Kein freier Quelltext-Input.
6. Offline- und Negativtests sowie einen sauberen Implementierungscommit
   abschließen. Erst danach den vorregistrierten A/A-/A/B-Lauf genau einmal
   ausführen; ein negatives Gate ist terminal und gültig.
7. Rohwerte, Median, Streuung, Correctness, Speicher, Runtime, DB-Hash und UI-
   Historie dokumentieren. Eine Promotion darf nur den exakt getesteten Scope
   betreffen.

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
- keine neuen Modelle oder Modellgewichte;
- keine adaptive Kernel- oder Codesuche, keine GPU-ISA und kein eigener Compiler;
- keine globale Installation oder Löschung außerhalb des Projekts ohne Rückfrage.

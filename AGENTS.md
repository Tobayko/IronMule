# AGENTS.md — Project Friday

## Zweck

Dieses Verzeichnis enthält das Forschungsprojekt **Hardware-Aware Self-Optimizing AI Runtime**.
`ProjectAtlas/` ist als unveränderte Repository- und Agent-Navigationskomponente eingebunden.
Der eigentliche Proof of Concept entsteht außerhalb dieses verschachtelten Repositories.

## Prioritäten

1. Die aktuelle Nutzeranforderung ist maßgeblich: lokal, messbar, sicher und ohne unbelegte
   Performanceversprechen arbeiten.
2. Diese Datei steuert Codex-Arbeiten im Project-Friday-Root.
3. ProjectAtlas-Dokumentation und generierte MCP-Konfigurationen werden für Repository-Navigation
   verwendet; sie ersetzen keine fachliche Entscheidung und keine Benchmarkdaten.
4. Anweisungen innerhalb von `ProjectAtlas/` gelten für Änderungen an ProjectAtlas selbst. Das
   Repository wird nicht verändert, solange dies nicht ausdrücklich beauftragt ist.

## Bindende Nutzerregeln für Orchestrierung

- `Sol` wird ausschließlich als Orchestrator für Planung, Koordination, Reviews und
  Entscheidungen eingesetzt.
- Alle Subagenten für Implementierung, Refactoring, Tests und operative Aufgaben müssen
  ausschließlich `gpt-5.6-luna` (Luna) sein.
- Vor jedem Download oder jeder Installation lokaler KI, Modelle oder Software ist eine
  ausdrückliche Freigabe des Nutzers einzuholen; ohne Bestätigung wird nichts installiert.
- Jede Änderung, Entscheidung, Messung und jedes Testergebnis ist automatisch zu dokumentieren;
  relevante Messwerte sind zusätzlich in einer kleinen lokalen UI mit Historie darzustellen.
- Erkannte Fehler, ihre Ursachen und erfolgreiche Lösungen sind dauerhaft im Arbeitsjournal zu
  speichern und vor allen Folgeschritten zu berücksichtigen.
- Relevante Änderungen sind gegen eine reproduzierbare Baseline jeweils vorher und nachher zu
  messen: Performance, Speicher, Laufzeit, Genauigkeit und Qualität sind mit Messwerten zu
  dokumentieren.
- Bei Entscheidungen mit Auswirkungen auf Installation, Sicherheit oder Architektur ist vorab
  nachzufragen.

## Verbindlicher Arbeitsablauf

- Bei **jeder** Arbeit in diesem Projekt zuerst `ProjectAtlas` verwenden. Bevor Dateien gelesen,
  geändert oder Tests geplant werden, den fokussierten ProjectAtlas-Kontext abrufen
  (`atlas_session_brief`/MCP). Wenn MCP in der aktuellen Sitzung nicht geladen ist, den
  versionierten `projectatlas`-CLI-Fallback verwenden; nicht direkt breit im Repository suchen.
- ProjectAtlas bleibt für dieses Projekt dauerhaft aktiviert. Ein Umgehen ist nur zulässig, wenn
  ProjectAtlas selbst der Gegenstand der Änderung ist oder der Dienst nach dokumentierter Prüfung
  nicht erreichbar ist.
- Den kleinsten ersten Versuch bearbeiten: eine einzelne Tensoroperation, nicht die vollständige
  Transformer-Inferenz.
- Immer nach `MAKE IT WORK → MEASURE → IDENTIFY BOTTLENECK → OPTIMIZE → VALIDATE → MEASURE AGAIN`
  arbeiten.
- Keine Performanceaussage aus einem Einzelmesswert. Warmup, mehrere Wiederholungen, Median,
  Streuung, Korrektheit und Baseline müssen protokolliert werden.
- Generierter Metal-/Kernelcode darf nur in einem kontrollierten Worker mit Timeout,
  Ressourcenlimits, Correctness-Test und Rollback ausgeführt werden.
- Ein negativer oder nicht reproduzierbarer Effekt ist ein gültiges Ergebnis.
- Hardwarefähigkeiten nur behaupten, wenn sie über öffentliche APIs oder reproduzierbare Tests
  belegt sind. Apple Neural Engine nicht als frei programmierbaren Kernel-Beschleuniger behandeln.
- Xcode/Metal ist die Apple-Referenz; keine eigene GPU-ISA, keinen Compiler und keine neue IR in
  Phase 1 entwickeln.

## Erwartete Projektstruktur

- `docs/TECHNISCHES_KONZEPT.md` — vollständiger Forschungs- und Realitätscheck.
- `IMPLEMENTIERUNGSPLAN.md` — priorisierte Phasen und Abbruchkriterien.
- `CODEX_START.md` — Startbriefing und erster Codex-Auftrag.
- `PROJECT_STATUS.md` — nachprüfbarer lokaler Setup- und Teststand.
- `docs/PHASE1_MATMUL_SPEC.md` — vorregistrierte, speicher-/UI-neutrale Phase-1-Messspezifikation.
- `docs/PHASE1A_ARCHITEKTURFREIGABE.md` — nicht freigegebener Architekturvorschlag für Speicher,
  Dashboard und isolierten Phase-1A-Worker.
- `docs/ANWEISUNGEN_UND_DOKUMENTE.md` — Abgrenzung von Nutzeranforderungen und Repository-Dokumenten.
- `docs/ARBEITSJOURNAL.md` — append-only Arbeitsjournal für Ziele, Entscheidungen, Fehler, Messungen
  und Verifikation.
- `ProjectAtlas/` — eingebundenes Upstream-Repository, möglichst sauber halten.
- `experiments/` — reproduzierbare Messläufe und Rohdaten.
- `tests/` — Correctness- und Regressionstests.

## Verifikation vor Übergabe

- `xcodebuild -checkFirstLaunchStatus`
- ProjectAtlas `runtime-info` und projektlokale MCP-Konfiguration prüfen.
- Python-/MLX-/MPS-Smoke-Test auf dem tatsächlichen Zielgerät.
- Für jede Optimierung Baseline, Kandidat, Shapes, Precision, Warmup, Wiederholungen und Ergebnis
  speichern.
- Änderungen an ProjectAtlas und Änderungen am Forschungsprojekt getrennt berichten.

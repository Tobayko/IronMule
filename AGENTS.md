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
- Downloads und Installationen sind seit dem Nutzerentscheid vom 2026-09-02 ohne Einzelrückfrage
  zulässig; die frühere Bestätigungspflicht entfällt. Bedingungen im Abschnitt „Hardwarefreigabe".
- Jede Änderung, Entscheidung, Messung und jedes Testergebnis ist automatisch zu dokumentieren;
  relevante Messwerte sind zusätzlich in einer kleinen lokalen UI mit Historie darzustellen.
- Erkannte Fehler, ihre Ursachen und erfolgreiche Lösungen sind dauerhaft im Arbeitsjournal zu
  speichern und vor allen Folgeschritten zu berücksichtigen.
- Relevante Änderungen sind gegen eine reproduzierbare Baseline jeweils vorher und nachher zu
  messen: Performance, Speicher, Laufzeit, Genauigkeit und Qualität sind mit Messwerten zu
  dokumentieren.
- Bei Entscheidungen mit Auswirkungen auf Installation, Sicherheit oder Architektur ist vorab
  nachzufragen.

## Hardwarefreigabe (Nutzerentscheid 2026-09-02, dauerhaft gültig)

Diese Regel ersetzt die frühere Einzelfreigabe je Messlauf. Sie gilt bis der
Nutzer sie ausdrücklich widerruft, auch in künftigen Sitzungen.

- **Tests laufen auf echter Hardware. Simulationen, Mocks und Fakes sind für
  Hardwarepfade unzulässig.** Ein Test, der GPU-, MLX- oder Modellverhalten
  behauptet, muss es auf dem Zielgerät ausgeführt haben. Synthetische Daten
  bleiben ausschließlich für Rand- und Fehlerfälle zulässig und begründen
  weiterhin keine Performance-, Hardware- oder Modellaussage.
- **GPU, CPU und die übrige verfügbare Hardware werden genutzt, nicht
  umgangen.** Ein Pfad, der mangels Freigabe auf eine serielle oder
  CPU-Ersatzvariante ausweicht, ist kein gültiges Messergebnis mehr, sondern
  ein offener Punkt.
- **Reale Messläufe brauchen keine Einzelbestätigung mehr.** Sie werden
  gestartet, sobald sie fachlich an der Reihe sind. Die frühere Regel
  „jeder Lauf einzeln freigegeben, maximal 30 Minuten, manuell gestartet"
  entfällt als Freigabehürde.
- **Downloads und Installationen sind freigegeben.** Pakete, Werkzeuge und
  Modelle dürfen ohne Einzelrückfrage geladen und installiert werden, wenn sie
  die Arbeit voranbringen.

### Was bei einer Installation zwingend mitläuft

Das ist keine Erlaubnisfrage, sondern eine Folge der Evidenzbindung: jede
versiegelte Studie bindet ihre Umgebung über `environment_sha256`
(`friday_head_skip_runtime/policy.py:255-269`). Eine Installation ändert diesen
Hash und kann bestehende Bindungen ungültig machen — ein Runtime-Pfad, der
gestern autorisiert war, fällt danach still in die Baseline.

- Vor der Installation den aktuellen `environment_sha256` festhalten.
- Nach der Installation prüfen, welche Pakete ihre Autorisierung verlieren, und
  das Ergebnis im Arbeitsjournal vermerken.
- Betroffene Pfade werden neu qualifiziert oder als offener Punkt geführt — sie
  gelten nicht stillschweigend weiter.
- Modelle bevorzugt weiter aus dem validierten projektlokalen Cache; ein neu
  geladenes Modell ist ein neuer Snapshot und erbt keine Evidenz.

### Was dadurch ausdrücklich **nicht** entfällt

Die folgenden Grenzen sind **keine** Freigabehürden, sondern Bedingungen dafür,
dass eine Zahl überhaupt etwas bedeutet. Der zentrale Befund des Projekts ist,
dass der Störuntergrund die meisten realen Effekte übersteigt
(`docs/ERGEBNISSE.md`: ungepaart `20,5 %` Variationskoeffizient gegen gepaart
`1,32 %`). Wer sie fallen lässt, misst nicht schneller, sondern erzeugt
schneller bedeutungslose Zahlen.

- `BudgetGuard`-Pausen, Duty-Cycle und die Kontinuitätsgrenze bleiben aktiv.
- Netzbetrieb, kein Low-Power, Fremdlastfreiheit und die Speicher-/Swap-Grenzen
  bleiben Vorbedingung jeder Messung.
- Gepaarte Messung, Warmup, Wiederholungen, Median und Streuung bleiben Pflicht;
  kein Ergebnis aus einem Einzelmesswert.
- Vorregistrierung, eingefrorene Schwelle und exakte Tokenidentität als
  terminales Gate bleiben unverändert.
- Generierter Metal-/Kernelcode läuft weiterhin nur im kontrollierten Worker
  mit Timeout, Ressourcenlimit, Correctness-Test und Rollback.

Kurz: **die Erlaubnisfrage ist beantwortet, die Messhygiene bleibt.**

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

## Gemeinsamer Code für neue Studien

Neue Studienpakete kopieren keine Infrastruktur mehr. Statistik, Storage,
Provenienz, Guard und read-only UI kommen aus `friday_evidence` (bei Bedarf dort
erweitern). Hintergrund: vier divergierte `statistics.py`-Kopien in
`friday_h0/h1/n10/n10_v2`. Bestehende versiegelte Pakete bleiben wegen
Code-Hash-Bindung byteidentisch eingefroren und werden weder umbenannt noch
dedupliziert (siehe „warum `avo` in Bezeichnern stehen bleibt" im
Arbeitsjournal-Archiv).

## Erwartete Projektstruktur

- `docs/TECHNISCHES_KONZEPT.md` — vollständiger Forschungs- und Realitätscheck.
- `docs/IMPLEMENTIERUNGSPLAN.md` — priorisierte Phasen und Abbruchkriterien.
- `docs/CODEX_START.md` — Startbriefing und erster Codex-Auftrag.
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

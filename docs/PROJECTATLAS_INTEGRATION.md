# ProjectAtlas-Integration

## Rolle im Projekt

ProjectAtlas ist die lokale Repository-Intelligence- und MCP-Schicht für Codex. Es mappt Dateien,
Symbole, Beziehungen, Zwecke, Suchdaten und Token-Telemetrie. Es ist **nicht** der Optimierungsagent,
kein GPU-Compiler und keine Hardwareabstraktion.

## Installierter Stand

- Quelle: `https://github.com/styler-ai/ProjectAtlas`
- Checkout: `ProjectAtlas/`
- Commit beim Klonen: `1f576921f2c824976a591d57be53e871dcd19cd8`
- Runtime: `0.4.5-rc1`
- Runtime-Pfad: `/Users/tobiasburandt/.local/bin/projectatlas`
- Codex-Marketplace: `projectatlas`, Ref `v0.4.5-rc1`

## Verwendung

ProjectAtlas ist in diesem Projekt keine optionale Hilfe, sondern die verbindliche erste
Navigationsschicht für die KI. Jede neue Aufgabe beginnt mit ProjectAtlas-Kontext; direkte breite
Dateisuche ist nur ein dokumentierter Fallback bei fehlender oder defekter ProjectAtlas-Verbindung.

Im Project-Friday-Root:

```bash
projectatlas init
projectatlas --format json runtime-info
projectatlas overview
```

Wenn MCP verfügbar ist, zuerst `atlas_session_brief` mit `compact: true` verwenden und die darin
empfohlenen Selektoren weiterreichen. Ohne MCP ist `projectatlas overview` der sichere CLI-Fallback.
Breite `find`-/Source-Lektüre erst danach.

Die absolute Runtime und die projektlokale Datenbank stehen in `.projectatlas/`. Datenbank, Locks,
MCP-Ausgabedateien und Telemetrie sind lokaler Zustand und gehören nicht in einen öffentlichen Commit.

## Warum ProjectAtlas hier sinnvoll ist

Der Hardware-Agent soll später viele Dateien, Messadapter, Backend-Implementierungen und Benchmark-
Ergebnisse sicher navigieren. ProjectAtlas reduziert unnötige Kontextlektüre und hält diese Navigation
lokal. Die eigentliche Experimentausführung bleibt deterministisch und in der Execution Plane.

## Nicht daraus ableiten

ProjectAtlas liefert keine zusätzlichen Apple-Metal-, iOS-, Android- oder NPU-Fähigkeiten. Es ersetzt
weder MLX/Metal/Compiler noch Profiler und liefert keine Performancegarantie.

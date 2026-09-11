# SSOT — eine Datenbank für alle Messdaten

`.friday-data/ssot.sqlite3` ist der abgeleitete Gesamtindex über jede Messquelle
des Projekts: 19 Studiendatenbanken mit sechs verschiedenen Tabellenformen und
über 400 JSON-/JSONL-Dateien aus `research/`, `experiments/` und `.friday-data/`.

Der Index ist **abgeleitet, nicht autoritativ für den Rohinhalt**. Quellen werden
ausschließlich read-only geöffnet (`?mode=ro`, `PRAGMA query_only`); versiegelte
Studien und Rohdaten bleiben byteidentisch. Der Index wird vollständig neu gebaut
und atomar ersetzt, er kann jederzeit gelöscht und regeneriert werden.

## Bedienung

```bash
python3 tools/ssot.py build                     # Neuaufbau, ~32 s
python3 tools/ssot.py status                    # Quellen, Studien, Agenten, Fehler
python3 tools/ssot.py metrics tokens_per_second # Metriknamen suchen
python3 tools/ssot.py query "SELECT study, agent, value FROM measurement
                             WHERE metric='median_ms' ORDER BY value LIMIT 20"
```

## Schema

| Tabelle | Inhalt |
| --- | --- |
| `sources` | jede eingelesene Datei mit Pfad, Form, Größe, SHA-256, mtime, Agent |
| `runs` | ein Datensatz je Messung: Studie, native ID, Art, Status, Zeit, Payload, Provenienz, Agent |
| `metrics` | jeder numerische Blattwert eines Payloads unter seinem Punktpfad |
| `measurement` | View: `runs` × `metrics` × `sources` für Querschnittsabfragen |

Ein JSON-Export eines hashverketteten Journals (`events` mit `seq`, `kind`,
`payload`, `sha256`, `prev_sha256`) wird in derselben Körnung gelesen wie die
`event_journal`-Tabelle, aus der er stammt: ein Lauf je Ereignis mit `run_id:seq`
als nativer ID, der Kettenhash in `runs.provenance_json`. Der Envelope bleibt
zusätzlich als eigener Lauf erhalten. Dieselben Ereignisse stehen mehrfach im
Index, wenn mehrere Exporte sie enthalten; die Dublette ist über den Kettenhash
auffindbar und wird nicht stillschweigend zusammengeführt.

Payloads bis 1 MiB liegen inline in `runs.payload_json`; größere bleiben über
`sources.path` und `runs.payload_sha256` adressierbar. Numerische Arrays über 32
Elementen werden als `count/min/max/mean/median` zusammengefasst statt indiziert.

## Wer hat gemessen

Jeder Lauf trägt `runs.agent` (`claude`, `codex`, `gemini`, `unattributed`) und
`runs.agent_source`, das die verwendete Evidenz benennt — stärkste zuerst:

| `agent_source` | Evidenz |
| --- | --- |
| `payload_field` | die Messung nennt ihren Agenten selbst |
| `provenance_commit` | die Provenienz referenziert einen Commit |
| `archive_manifest` | das Archivmanifest nennt den erzeugenden Commit |
| `git_history` | jüngster Commit, der die Datei berührt hat |
| `commit_at_time` | Commit, der zum Messzeitpunkt der jüngste war |
| `file_mtime` | nur die Schreibzeit der Quelldatei — schwächste Stufe |
| `no_evidence` | nichts Belastbares; bleibt `unattributed` |

Die Commitzuordnung selbst kommt aus `Co-Authored-By: Claude …` (`commit_trailer`),
aus dem Commit-Scope `feat(gemini): …` (`commit_scope`) oder, wenn beides fehlt,
aus der Voreinstellung dieses Codex-getriebenen Baums (`repo_default_codex`).
Die letzte Stufe ist eine Schlussfolgerung, kein Beleg.

**Grenzen.** `file_mtime` trägt derzeit 753 der Läufe (`optimizer-v2`, ohne eigene
Zeitstempel); dieser Anteil sagt aus, wer zuletzt geschrieben hat, nicht wer
gemessen hat. `h0` und `h01` liegen vor dem ersten Commit des Repositories und
bleiben `unattributed`. Verlässlich wird die Zuordnung erst, wenn Messwerkzeuge
den Agenten selbst in ihre Provenienz schreiben; `payload_field` liest dafür
`agent`, `produced_by`, `measured_by`, `author_agent` und `agent_id`.

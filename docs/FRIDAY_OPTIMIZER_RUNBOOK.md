# Friday Optimizer — Runbook

**Stand:** 2026-08-30
**Status:** Offline-Control-Plane verifiziert; reale Hardware und Promotion bleiben gate-basiert gesperrt.

Dieses Dokument beschreibt den sicheren Offline-Weg. Es installiert nichts, lädt
kein Modell und startet keine GPU-Arbeit. Alle Beispiele sind absichtlich mit
relativen Pfaden gehalten. Im Projekt-Root ausführen:

```text
<PROJECT_ROOT>/
├── friday_optimizer/
├── .friday-data/
└── tests/test_optimizer_*.py
```

## 1. Was vorhanden ist

Die Control Plane besteht aus den Modulen `Memory`, `Corpus`, `Dataset`, `Bridge`,
`Fingerprint`, `Candidates`, `Evaluator`, `Readiness`, `Lease`, `Session`,
`Profile`, `History`, `Orchestrator`, `IronMuleAdapter`, `Dashboard` und `CLI`.
Die Befehle sind `doctor`, `audit`, `import`, `dataset`, `status`, `shadow` und
`dashboard`. Es gibt bewusst keinen `tune`-, `activate`-, Download- oder
Installationsbefehl.

Der aktuelle Offline-Stand wurde nach dem letzten Dashboard-Schema-Fix ohne offene
P0/P1-Sicherheitsbefunde übergeben. Dieser Dokumentationsschritt führt keinen neuen
Security-Scan aus. Die zugehörigen Tests liegen unter `tests/test_optimizer_*.py`;
dieses Runbook führt sie nicht erneut aus.

## 2. Prerequisites

- macOS/Apple Silicon ist für den späteren Modellpfad vorgesehen; die Offline-
  Befehle benötigen jedoch kein Modell und keine GPU.
- Eine vorhandene Projektumgebung verwenden: `.venv/bin/python`.
- Im Projekt-Root starten, damit relative Pfade und `.friday-data` stimmen.
- Keine Abhängigkeit installieren und keine Netzwerkverbindung voraussetzen.
- `.friday-data/optimizer-v2.sqlite3` und die Dataset-Datei nur lesen, außer der
  konkrete Schreibbefehl enthält ausdrücklich `--execute`.

## 3. Sichere Befehle

Alle folgenden Befehle sind Beispiele; sie wurden hier nicht ausgeführt.

### Diagnose ohne Systemprobe

```bash
.venv/bin/python -m friday_optimizer doctor
```

Das prüft Pfade, Schema, Kette, Profile und Memory-Zustand. Es schreibt nichts.
Eine optionale Systemprobe liest nur lokale Zustände und startet keine Modellarbeit:

```bash
.venv/bin/python -m friday_optimizer doctor --probe-system
```

### Read-only Inventar

```bash
.venv/bin/python -m friday_optimizer audit --root .
```

Der Audit entdeckt und normalisiert Quellen read-only. Im verifizierten Stand wurden
`406` Quellen entdeckt, `392` Records normalisiert und `2` Records als eligible
(`Q2`/`B27d`) erkannt. Contract-Hashes werden ausgegeben und gebunden.

### Expliziter Import in Optimization Memory

```bash
.venv/bin/python -m friday_optimizer import \
  --root . \
  --memory .friday-data/optimizer-v2.sqlite3 \
  --execute
```

`import` schreibt nur mit `--execute`. Der Import ist idempotent: dieselbe Quelle
wird bei wiederholtem Import erkannt und nicht als neuer Datensatz dupliziert. Die
alten Evidence-Quellen bleiben unverändert. Vor dem Schreiben werden Bindung,
Kette und Dateistabilität geprüft.

### Dataset-Snapshot erzeugen

Read-only Vorschau ohne Ausgabedatei:

```bash
.venv/bin/python -m friday_optimizer dataset --root .
```

Explizites Erzeugen einer neuen Datei:

```bash
.venv/bin/python -m friday_optimizer dataset \
  --root . \
  --out .friday-data/optimizer-dataset-next.json \
  --execute
```

Die Ausgabedatei muss neu sein; vorhandene Dateien werden nicht überschrieben.
Der aktuelle materialisierte Snapshot heißt `.friday-data/optimizer-dataset-v1.json`.

### Status anzeigen

```bash
.venv/bin/python -m friday_optimizer status \
  --memory .friday-data/optimizer-v2.sqlite3 \
  --profiles .friday-data/optimizer-profiles.json
```

`status` ist read-only. Es zeigt Schema, Kette, Profilstatus, Fingerprint und
Datenzustand, nicht private Rohinhalte.

### Shadow-Entscheidung ohne History-Schreibzugriff

```bash
.venv/bin/python -m friday_optimizer shadow \
  --request <REQUEST_JSON> \
  --memory .friday-data/optimizer-v2.sqlite3
```

`<REQUEST_JSON>` ist ein bereits vorhandener, geprüfter Request. Dieser Befehl
entscheidet nur im Shadow-Modus und aktiviert nichts. Eine History-Aufzeichnung ist
eine gesonderte, explizite Schreibentscheidung:

```bash
.venv/bin/python -m friday_optimizer shadow \
  --request <REQUEST_JSON> \
  --memory .friday-data/optimizer-v2.sqlite3 \
  --write-history --execute
```

Der Request wird vor und nach dem Lesen auf Dateistabilität geprüft. Ein veränderter,
unvollständiger oder unbekannter Request fällt auf einen Datenfehler.

### Lokales Dashboard

```bash
.venv/bin/python -m friday_optimizer dashboard \
  --memory .friday-data/optimizer-v2.sqlite3 \
  --profiles .friday-data/optimizer-profiles.json \
  --dataset .friday-data/optimizer-dataset-v1.json \
  --port 8776
```

Das Dashboard bindet ausschließlich an `127.0.0.1`, ist read-only und läuft bis
`Ctrl-C`. Es bietet keine Schreibroute, keine Aktivierung und keinen Upload. Einen
zufälligen freien Port erhält man mit `--port 0`; der ausgegebene Port ist dann
lokal abzulesen.

## 4. Exitcodes

| Code | Bedeutung | Typische Reaktion |
| ---: | --- | --- |
| `0` | Vorgang erfolgreich | Ergebnis prüfen, nichts automatisch ableiten |
| `1` | lokal nicht verfügbar oder nicht qualifiziert | Baseline beibehalten, Ursache in Ausgabe prüfen |
| `64` | falsche Argumente | Befehl korrigieren; keine Datei wird überschrieben |
| `65` | Daten-, Schema-, Hash-, Pfad- oder Integritätsfehler | stoppen, Artefakt sichern, nicht löschen |
| `70` | interner Fehler | Vorgang stoppen und lokale Diagnose sichern |
| `78` | ausdrückliche Ausführung fehlt | nur bei bewusstem Schreibvorgang `--execute` ergänzen |

`shadow` kann bei einer nicht qualifizierten Empfehlung Code `1` liefern. Das ist
kein Aktivierungsfehler: `no_activation=true` bleibt Teil der Entscheidung.

## 5. Materialisierte Artefakte

| Artefakt | Stand |
| --- | --- |
| `.friday-data/optimizer-v2.sqlite3` | `401` Records, `1,212,416 B`, Modus `0600`, SHA-256 `5f5d286c...ab2aa`, Chain/Integrity `true` |
| `.friday-data/optimizer-dataset-v1.json` | `392` Records, `2,208,967 B`, Modus `0600`, SHA-256 `79ce...f5c8d` |
| Dataset-Qualität | `train=2`, `val=0`, `holdout=0`, `smoke_only/no_learning_claim` |
| Bridge/History | `400` Bridge-Records und `1` History-Record importiert |
| Profile | zwei vorhandene Profile; nur exakt passende Fingerprints dürfen gelesen werden |

Die vollständigen Hashes sind in den lokalen Status-/Memory-Ausgaben und im
Archivmanifest zu prüfen; verkürzte Hashes in diesem Runbook dienen nur der
Orientierung. Ein Hashpräfix ist kein Ersatz für eine vollständige Prüfung.

## 6. Bedeutung von `no_learning_claim`

`no_learning_claim` bedeutet: Der Korpus ist für einen Pipeline-Smoke ausreichend,
aber nicht für ein gelerntes Rankingmodell. Mit `2` Trainingsrecords, `0`
Validation- und `0` Holdout-Records gibt es keinen belastbaren Test für GBDT,
Regression, Bayesian Optimization oder Generalisierung.

Daraus folgt:

- Historische Daten dürfen nur die Reihenfolge erlaubter Offline-Prüfungen
  beeinflussen.
- Kein Learned Model darf automatisch Kandidaten aktivieren.
- Keine Zahl darf als Cross-Device-, Cross-Model- oder 27B-Prognose ausgegeben
  werden.
- L1.2 bleibt offen, bis echte Records getrennte Validation- und Holdout-Splits
  erlauben.
- Synthetische Daten sind nur für Rand- und Fehlerfälle zulässig.

## 7. Recovery, Manipulation und Rollback

Bei Code `65`, einer falschen Kette, einem Hashfehler, einem unstabilen Input oder
einem fehlenden Fingerprint:

1. Vorgang stoppen und die aktuelle Datei nicht überschreiben.
2. Artefakt und Ausgabe unverändert sichern beziehungsweise archivieren.
3. Mit `doctor` und anschließend read-only `status` den Zustand prüfen.
4. Nur nach menschlicher Prüfung einen idempotenten Import oder einen neuen
   Dataset-Ausgabepfad verwenden.

Die Offline-Control-Plane aktiviert kein Profil. Es gibt deshalb derzeit keinen
produktiven Rollback-Befehl; die sichere Betriebsentscheidung ist Baseline bzw.
`no_recommendation`. Eine spätere Promotion muss einen atomaren aktiven Zeiger,
vorherige Baseline, Canary und Rollback-Latch besitzen. Ein fehlerhaftes Profil wird
nicht still repariert oder überschrieben.

Es gibt keinen `--force`-Schalter. Unsicherheit wird nicht übersprungen. Ein realer
Hardwarelauf darf erst in einer später separat freigegebenen Phase stattfinden und
muss manuell, AC-only, fremdlastfrei, sparsam und auf maximal 30 Minuten begrenzt
sein. Fremdlast während eines Laufs beendet ihn sicher und hält die Baseline aktiv.

## 8. Aktuelle Grenzen

- Es gibt noch keinen realen Optimizer-Hardwarelauf aus dieser Control Plane.
- Nur `2` Records sind für Shadow-/Eligibility relevant (`Q2`/`B27d`).
- Gemma 27B besitzt keinen lokalen Snapshot in diesem Offline-Stand.
- Es gibt keine Cross-Device-Evidence und keine automatische Übertragung auf
  andere Macs.
- Learned Ranking, GBDT und BO sind wegen fehlender Validation/Holdout-Daten
  blockiert.
- Der IronMule-Adapter bleibt Shadow-only; der aktuelle Claude-Worktree
  `.worktrees/ironmule-b7` darf nicht verändert werden.
- Rohdaten und alte Evidence werden nicht gelöscht. Besonders `b7`-Rohdaten dürfen
  nur nach einem separaten, überprüften Archivieren und mit ausdrücklicher Freigabe
  bewegt werden.

## 9. Abschlusskriterium für die nächste Phase

L1.1 ist offline abgeschlossen. L1.2 darf erst von `smoke_only/no_learning_claim`
zu einer Lernbewertung wechseln, wenn echte neue Records, getrennte Train-/Val-/
Holdout-Splits, reproduzierbares Training, Unsicherheits-/OOD-Gates und ein
vergleichbarer Random/Grid/Regression/GBDT/BO-Nachweis vorliegen. Vor dem ersten
realen Modell- oder Hardwarelauf sowie vor jedem Download oder jeder Installation
ist eine separate ausdrückliche Nutzerfreigabe erforderlich.

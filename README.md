# Hardware-Aware Self-Optimizing AI Runtime

Kann ein Verfahren reale Hardware- und Laufzeitdaten nutzen, um Ausführungspläne zu
verändern, sicher zu prüfen und reproduzierbar zu bewerten? Dieses Projekt beantwortet
das für eine feste Operation auf Apple Silicon — mit Werkzeugen, die auch Nullbefunde
zuverlässig als solche melden.

Forschungsprototyp für Apple Silicon. Kein Ersatz für CUDA, XLA, TorchInductor oder
einen GPU-Compiler.

> **Evidenz-Audit vom 21. August 2026:** Die früher als „bestätigt“ bezeichneten
> H1/H2-Läufe waren intern gepaart, repliziert und correctness-geprüft, erfüllen
> aber den formalen Projektvertrag nicht: Das A/A-Gate war nicht geschlossen und
> die MDE nicht vor dem ersten A/B-Lauf versiegelt. Zudem blieben nur
> Zusammenfassungen, keine Rohmessungen. Sie werden daher ausschließlich als
> **explorative Legacy-Beobachtungen** geführt. Formale H1-, H2-, Cross-Device-
> oder Phase-1B-Claims: **keine**. Siehe
> [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md).

## Der Kernbefund in einem Absatz

Auf diesem Gerät ist ein **ungepaarter** Performancevergleich nahezu wertlos: Die
Streuung zwischen Läufen liegt bei rund `20 %` und übertrifft damit die meisten
realen Effekte. Konkret erschien `mx.compile` ungepaart mit `−27,6 %` als klarer
Gewinn — gepaart gemessen blieben `+0,2 %` mit Konfidenzintervall
`[0,999, 1,005]`, also **kein Effekt**. Derselbe Datensatz liefert je nach
Auswertung eine Nachweisgrenze von `33 %` oder `2,2 %`, ein Faktor `15`.

Die aktuellen Werkzeuge vergleichen deshalb beide Arme **innerhalb desselben
Blocks**, verlangen eine prospektiv versiegelte Schwelle und speichern neue
Rohmessungen mit Git-/Code-/Spec-/Umgebungsprovenienz. Das wertet die historischen
Läufe nicht rückwirkend auf. Nach ausdrücklicher Rechenfreigabe liegen inzwischen
zwei native v1-Berichte mit Rohmessungen vor; v1 kennzeichnet sie weiterhin
explizit mit `formal_claim=false`.

Alle Befunde kompakt: **[`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md)**

## Zweiter explorativer Befund: die Inferenz wirkt speicherbegrenzt

Ein neuer offline erzwungener Lauf verwendete ausschließlich die bereits im
Projektcache vorhandenen Gemma-Snapshots, je fünf Wiederholungen:

| native v1 | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Folge-Token | `199,5 Token/s` | `91,3 Token/s` |
| Bandbreite genutzt | `36,53 %` | `58,47 %` |
| Rechenwerke genutzt | `2,78 %` | `4,45 %` |

**Faktor rund `13` zwischen Bandbreiten- und Rechenanteil.** Die Beobachtung
deutet darauf, dass die Rechenwerke bei dieser Inferenz auf Daten warten. Code
„näher an der Maschinensprache" optimiert den Anteil, der ohnehin leerläuft.
Wirksam sind nur **weniger Bytes** (Quantisierung, bei 4-bit-Modellen schon
eingelöst) und **weniger Durchgänge** (Kernel-Fusion).

Die vollständigen Rohsamples sind jetzt append-only gespeichert. Wegen nur eines
Geräts, veröffentlichten Peakwerten und Schema v1 bleibt die Klassifikation
explorativ und ist keine allgemeine Hardwaregrenze oder formale H2-Aussage.

Die naheliegende Fusions-Layer über ein unverändertes Modell wurde geprüft und
**verworfen** — `mlx-lm` fusioniert bereits selbst, und der KV-Cache verhindert
den Rest. Details in `docs/ERGEBNISSE.md`.

## Was der Loop selbst findet

`loop` exploriert Ausführungspläne, verfeinert um den Überlebenden und misst den
eigenen Sieger erneut. Die historischen vier Läufe beobachteten explorativ
`−11 %` bis `−14 %`; sie sind heute Legacy-Zusammenfassungen, kein formaler H1-
Nachweis. `codegen` erprobt separat eine stark eingeschränkte modellgeschriebene
Plansprache.


## Schnellstart (OpenAI Server mit Live-Terminal-Cockpit)

Auf jedem Apple Silicon Mac in unter einer Minute startklar:

```bash
# 1. Umgebung vorbereiten (Python 3.12 + uv)
./scripts/bootstrap_apple.sh

# 2. Hardware-Preflight prüfen
./friday doctor

# 3. Friday mit Live-Terminal-Cockpit starten
./friday serve
```

Das Modell wird in das Unified Memory geladen und das **interaktive Live-Cockpit** (10-FPS-Tacho für UMA-Bandbreite, TTFT, Tokens/s und VRAM) startet direkt im selben Terminalfenster.

### Anfragen senden (OpenAI-kompatibel)

Jeder OpenAI-kompatible Client (Cursor, OpenWebUI, Python `openai`, curl) kann sich direkt an Port 8080 verbinden:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hallo Friday!"}],
    "stream": true
  }'
```

Im selben Terminalfenster visualisiert das Cockpit in Echtzeit Speicherbandbreite (GB/s gegen 400 GB/s Bus), TTFT (mit Prefix-Cache-Trefferanzeige), Generierungsrate und Zero-Swap.

### Hardware Auto-Tuner (<15 Sekunden)

Kalibriert und zertifiziert automatisch die optimalen Hardware-Knöpfe für diesen spezifischen Mac auf realer Hardware:

```bash
./friday autotune
```

Status und Knöpfe jederzeit einsehen:
```bash
./friday status
```

**Netzbetrieb ist Pflicht, nicht Komfort:** Auf Akku begrenzt macOS das
GPU-Power-Budget, Läufe sind dann weder vergleichbar noch schonend. Die Werkzeuge
verweigern den Start auf Batterie.

## Werkzeuge

| Kommando | Zweck |
| --- | --- |
| `./friday serve` | Startet OpenAI-kompatiblen Server mit interaktivem Terminal-Live-Cockpit |
| `./friday autotune` | Universeller Hardware-Auto-Tuner: kalibriert und zertifiziert Hardware-Knöpfe in <15s |
| `./friday status` | Hardware-Fakten, zertifizierte Knöpfe, Latenz und Runtime-Zustand auf einen Blick |
| `./friday doctor` | Preflight-Check: prüft Metal-GPU, Python, Netzbetrieb und Speicher |
| `./friday monitor` | Remote-Cockpit-Monitor für separate Terminals oder Remote-Hosts |
| `./friday loop` | Exploriert Ausführungspläne, verfeinert und misst den Sieger erneut (benötigt `--execute`) |
| `./friday dispatch` | Misst einen Plan gegen eine Baseline, gepaart, gegen feste Schwelle (benötigt `--execute`) |
| `./friday aa` | Vorregistrierte A/A-Nullkontrolle (Kalibrierung, keine Optimierung) |
| `./friday roofline` | Misst, ob Inferenz speicher- oder rechenbegrenzt ist |
| `./friday evidence` | Verifiziert/liest die append-only H1/H2-Historie ohne GPU |

Jedes messende Werkzeug hat zwei Sicherungen:

- **`--execute` ist Pflicht.** Ohne das Flag endet der Aufruf mit `not_released`
  und Exit `78`, **bevor** MLX importiert oder die GPU berührt wird.
- **`--self-check`** prüft die Statistik offline, ohne GPU und ohne MLX.

Die sieben H1/H2-Werkzeuge speichern einen Bericht erst dann als native Evidenz,
wenn der Root-Checkout sauber ist und Git-, Code-, Spec-, Paket- und
Hardwareidentität vor und nach dem Lauf übereinstimmen. Architektur und
Historien-UI: [`docs/H1H2_EVIDENZ_ARCHITEKTUR.md`](docs/H1H2_EVIDENZ_ARCHITEKTUR.md).
Auch native Schema-v1-Berichte tragen ausdrücklich `formal_claim=false`; ein
formaler H1-v2-Lauf benötigt einen neuen versiegelten Vertrag.

### H2: das Modell schlägt vor, der Harness entscheidet

`model-loop` gibt einem lokalen Gemma-3-Modell die bisherigen Messungen und die
gemessenen Gerätefakten und lässt es Kandidaten vorschlagen. Über mehrere Runden
sieht es die Ergebnisse seiner eigenen Vorschläge und kann darauf reagieren.

**`model-loop` schlägt Parameter vor, niemals Code.** Modellgenerierter Code ist
ein separates Sicherheitsproblem und wird ausschließlich vom experimentellen
`codegen`-Werkzeug in einer stark begrenzten Plansprache behandelt. In
`model-loop` wird jeder Vorschlag als einfache Ganzzahl geparst und verworfen,
wenn er außerhalb des registrierten Bereichs liegt. Antwortet das Modell mit Prosa,
einem Shell-Kommando oder `900`, wird nichts davon ausgeführt — die Runde ist
verloren, mehr nicht.

## Wie hier gemessen wird

Sechs Regeln, jede aus einem konkreten Fehlschlag entstanden:

1. **Immer gepaart** — beide Arme im selben Block, damit sich der gemeinsame
   Störuntergrund herauskürzt.
2. **Schwelle vor dem Lauf festlegen**, nie danach.
3. **Mindestens zehn Wiederholungen.** Zwei Zwischenzahlen mussten nach unten
   korrigiert werden, beide aus zu kleiner Stichprobe.
4. **Behandlungsarme im direkten Wechsel** statt frei randomisiert, wenn die
   Behandlung eine Zeitkomponente hat.
5. **Nach Konfidenzobergrenze auswählen**, nicht nach Punktschätzer, sobald aus
   mehreren Kandidaten gewählt wird — sonst gewinnt der glücklichste Ausreißer.
6. **Correctness vor Timing.** Ein Ausführungsplan darf Arbeit umsortieren, aber
   kein einziges Bit ändern; sonst wird die Messung verworfen.
7. **Arme innerhalb von ~`340 ms`.** Der Störprozess dieses Geräts hat eine
   gemessene Zeitskala von rund `340 ms`. Liegen die Vergleichsarme weiter
   auseinander, sehen sie unterschiedliche Störungen und die Paarung verliert
   ihren Vorteil.

## Hardwareschonung

Verbindliche Budgets, fail-closed für die Berichterstattung: GPU-Arbeit `≤ 120 s`
je Lauf, ununterbrochene Last `≤ 6 s`, reale Pflichtpause `≥ 4 s`, höchstens
`25 %` Duty-Cycle im gleitenden `60-s`-Fenster, Wall `≤ 20 min` und bei
Kandidatensuche `≥ 60 s` Cooldown. Netzbetrieb ist verpflichtend. Eine
Überschreitung verwirft den Lauf. Zum Vergleich: die
sechs-Session-H0.1-Studie belastete das Gerät mit `5,26 s` GPU-Arbeit über
`6,6 min`, einem Duty-Cycle von `1,33 %`.

Eine Temperaturschwelle ist bewusst **nicht** registriert: `ProcessInfo.thermalState`
hat keine stdlib-Bindung und `powermetrics` benötigt erhöhte Rechte. Eine Schwelle,
die niemand prüfen kann, wäre Schein-Sicherheit. Die Budgets begrenzen stattdessen
die Ursache des Wärmeeintrags.

## Modelltests

Optional, benötigt `mlx-lm` und **vor Installation/Download eine ausdrückliche
Nutzerfreigabe**:

```bash
VIRTUAL_ENV=.venv uv pip install mlx-lm
export HF_HOME="$PWD/.friday-data/models"     # Modelle im Projektordner halten
```

Geprüft mit `mlx-community/gemma-3-1b-it-4bit` und `gemma-3-4b-it-4bit`.

Vor einer solchen Installation lohnt ein `uv pip install --dry-run`: Würde dabei
`mlx` oder `numpy` hochgezogen, ändert sich die Umgebungsidentität und alle
früheren Läufe wären nicht mehr vergleichbar.

## Aufbau

```
friday_h0/    H0: Messsystem für eine feste FP16-2048²-Matmul, SQLite v1, Worker
friday_h01/   H0.1: vorregistrierte Stationaritätsstudie, stdlib-only Analysekern
friday_evidence/ H1/H2: provenancegebundene SQLite-v1-Evidenz und Historien-UI
tools/        Messwerkzeuge, Einstieg über friday.py
tests/        vollständige Suite; läuft ohne GPU und ohne Netz
docs/         Spezifikationen, Ergebnisse, Arbeitsjournal
```

Tests: `.venv/bin/python -m pytest` — zuletzt `439` Tests plus `2.447` Subtests
in `31,64 s` parallel.
Für besser lesbare Fehlerausgaben sequenziell: `pytest -n 0` (rund `90 s`).

## Weiterführend

- **[`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md)** — alle Befunde, Nullbefunde und
  Grenzen kompakt
- [`docs/ARBEITSJOURNAL.md`](docs/ARBEITSJOURNAL.md) — vollständige Herleitung samt
  Fehlversuchen und Korrekturen
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — aktueller Stand je Phase
- [`docs/PHASE1_MATMUL_SPEC.md`](docs/PHASE1_MATMUL_SPEC.md) — H0-Messvertrag
- [`docs/H01_PACED_TRAJECTORY_SPEC.md`](docs/H01_PACED_TRAJECTORY_SPEC.md) —
  vorregistriertes H0.1-Design
- [`docs/TECHNISCHES_KONZEPT.md`](docs/TECHNISCHES_KONZEPT.md) — was auf Apple
  Silicon messbar ist und was nicht
- [`docs/H1H2_EVIDENZ_ARCHITEKTUR.md`](docs/H1H2_EVIDENZ_ARCHITEKTUR.md) —
  Persistenz-, Provenienz-, Budget- und UI-Vertrag
- [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md) —
  aktueller Go/No-Go-Entscheid

## Grenzen

Alle historischen Zahlen stammen von **einem** Gerät (M1 Max, 32 GB) und liegen
für H1/H2 nur als Legacy-Zusammenfassungen vor. `loop` sucht in einem festen, von
Hand definierten Raum; `codegen` darf nur `matmul`, `eval` und `synchronize` in
einer begrenzten Plansprache kombinieren und schreibt keine Kernel. Die
H0.1-Stationaritätsstudie blieb **ungelöst** —
`16,7 %` aller Samples liegen über dem `1,5`-fachen Median, und diese Ausreißer
sind der größte offene Punkt des Projekts.

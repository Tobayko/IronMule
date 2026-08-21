# Hardware-Aware Self-Optimizing AI Runtime

Kann ein Verfahren reale Hardware- und Laufzeitdaten nutzen, um Ausführungspläne zu
verändern, sicher zu prüfen und reproduzierbar zu bewerten? Dieses Projekt beantwortet
das für eine feste Operation auf Apple Silicon — mit Werkzeugen, die auch Nullbefunde
zuverlässig als solche melden.

Forschungsprototyp für Apple Silicon. Kein Ersatz für CUDA, XLA, TorchInductor oder
einen GPU-Compiler.

## Der Kernbefund in einem Absatz

Auf diesem Gerät ist ein **ungepaarter** Performancevergleich nahezu wertlos: Die
Streuung zwischen Läufen liegt bei rund `20 %` und übertrifft damit die meisten
realen Effekte. Konkret erschien `mx.compile` ungepaart mit `−27,6 %` als klarer
Gewinn — gepaart gemessen blieben `+0,2 %` mit Konfidenzintervall
`[0,999, 1,005]`, also **kein Effekt**. Derselbe Datensatz liefert je nach
Auswertung eine Nachweisgrenze von `33 %` oder `2,2 %`, ein Faktor `15`.

Deshalb vergleichen alle Werkzeuge hier beide Arme **innerhalb desselben Blocks**
und verlangen, dass ein Effekt eine **vor** dem Lauf festgelegte Schwelle
überschreitet.

Alle Befunde kompakt: **[`docs/ERGEBNISSE.md`](docs/ERGEBNISSE.md)**

## Der zweite Kernbefund: die Inferenz ist speicherbegrenzt

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Bandbreite genutzt | `31,9 %` | `51,2 %` |
| Rechenwerke genutzt | `2,4 %` | `3,9 %` |

**Faktor `13`.** Die Rechenwerke laufen bei echter Inferenz fast leer — sie warten
auf Daten. Das entscheidet, welche Optimierungen überhaupt helfen können: Code
„näher an der Maschinensprache" optimiert den Anteil, der ohnehin leerläuft.
Wirksam sind nur **weniger Bytes** (Quantisierung, bei 4-bit-Modellen schon
eingelöst) und **weniger Durchgänge** (Kernel-Fusion).

Daraus folgt eine harte Obergrenze: Bei `51,2 %` Bandbreitenauslastung bringt
selbst eine perfekte Optimierung ohne Gewichtsverkleinerung höchstens rund `2x`.

Die naheliegende Fusions-Layer über ein unverändertes Modell wurde geprüft und
**verworfen** — `mlx-lm` fusioniert bereits selbst, und der KV-Cache verhindert
den Rest. Details in `docs/ERGEBNISSE.md`.

## Was der Loop selbst findet

`loop` sucht ohne Zutun: Er exploriert Ausführungspläne, verfeinert um den
Überlebenden und bestätigt seinen eigenen Sieger unabhängig. Vier Läufe, vier
bestätigte Optimierungen im Bereich `−11 %` bis `−14 %`, mit Kandidaten, die in
der manuellen Suche nicht vorkamen. `codegen` geht einen Schritt weiter: Dort
schreibt ein lokales Modell den Plan als Python, sandboxed und geprüft.


## Schnellstart

```bash
./scripts/bootstrap_apple.sh          # Python 3.12 + uv, legt .venv an
.venv/bin/python tools/friday.py doctor
```

`doctor` prüft Python, MLX samt Metal-Zugriff, NumPy, Netzbetrieb und Plattenplatz
und sagt, was fehlt.

```bash
.venv/bin/python tools/friday.py list             # verfügbare Werkzeuge
.venv/bin/python tools/friday.py loop --execute   # Optimierungen selbst suchen
```

**Netzbetrieb ist Pflicht, nicht Komfort:** Auf Akku begrenzt macOS das
GPU-Power-Budget, Läufe sind dann weder vergleichbar noch schonend. Die Werkzeuge
verweigern den Start auf Batterie.

## Werkzeuge

| Kommando | Zweck |
| --- | --- |
| `loop` | Sucht selbst: exploriert Ausführungspläne, verfeinert, bestätigt den eigenen Sieger |
| `dispatch` | Misst einen Plan gegen eine Baseline, gepaart, gegen feste Schwelle |
| `cooldown` | Charakterisiert, wie eine Leerlaufpause die nächste Operation verlangsamt |
| `aa` | Vorregistrierte A/A-Nullkontrolle (Kalibrierung, keine Optimierung) |
| `model-loop` | **H2:** ein lokales Modell schlägt Ausführungspläne vor, der Harness bewertet sie (benötigt `mlx-lm`) |
| `codegen` | **H2 vollständig:** ein Modell *schreibt* Ausführungspläne, sandboxed und bewertet (benötigt `mlx-lm`) |
| `roofline` | Misst, ob Inferenz speicher- oder rechenbegrenzt ist — entscheidet, welche Optimierung überhaupt helfen kann |
| `fusion` | Misst `mx.compile` über den cache-freien Forward-Pass — **kein** Generierungsgewinn, siehe ERGEBNISSE |
| `guard` | Belegt, dass der H0.1-Analysekern stdlib-only bleibt |

Jedes messende Werkzeug hat zwei Sicherungen:

- **`--execute` ist Pflicht.** Ohne das Flag endet der Aufruf mit `not_released`
  und Exit `78`, **bevor** MLX importiert oder die GPU berührt wird.
- **`--self-check`** prüft die Statistik offline, ohne GPU und ohne MLX.

### H2: das Modell schlägt vor, der Harness entscheidet

`model-loop` gibt einem lokalen Gemma-3-Modell die bisherigen Messungen und die
gemessenen Gerätefakten und lässt es Kandidaten vorschlagen. Über mehrere Runden
sieht es die Ergebnisse seiner eigenen Vorschläge und kann darauf reagieren.

**Das Modell schlägt Parameter vor, niemals Code.** Modellgenerierten Code auf der
GPU auszuführen ist ein eigenes Sicherheitsproblem und ausdrücklich nicht Teil
dieses Werkzeugs. Jeder Vorschlag wird als einfache Ganzzahl geparst und verworfen,
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

Verbindliche Budgets, fail-closed: GPU-Arbeit `≤ 120 s` je Lauf, ununterbrochene
Last `≤ 6 s`, Wall `≤ 20`–`30 min`, Netzbetrieb verpflichtend. Jede Überschreitung
bricht ab und **verwirft** den Lauf, statt ihn zu kürzen. Zum Vergleich: die
sechs-Session-H0.1-Studie belastete das Gerät mit `5,26 s` GPU-Arbeit über
`6,6 min`, einem Duty-Cycle von `1,33 %`.

Eine Temperaturschwelle ist bewusst **nicht** registriert: `ProcessInfo.thermalState`
hat keine stdlib-Bindung und `powermetrics` benötigt erhöhte Rechte. Eine Schwelle,
die niemand prüfen kann, wäre Schein-Sicherheit. Die Budgets begrenzen stattdessen
die Ursache des Wärmeeintrags.

## Modelltests

Optional, benötigt `mlx-lm`:

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
tools/        Messwerkzeuge, Einstieg über friday.py
tests/        vollständige Suite; läuft ohne GPU und ohne Netz
docs/         Spezifikationen, Ergebnisse, Arbeitsjournal
```

Tests: `.venv/bin/python -m pytest` — läuft parallel und braucht rund `31 s`.
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

## Grenzen

Alle Zahlen stammen von **einem** Gerät (M1 Max, 32 GB). Der Loop sucht in einem
festen, von Hand definierten Raum von Ausführungsplänen; er generiert keinen Code
und schreibt keine Kernel. Die H0.1-Stationaritätsstudie blieb **ungelöst** —
`16,7 %` aller Samples liegen über dem `1,5`-fachen Median, und diese Ausreißer
sind der größte offene Punkt des Projekts.

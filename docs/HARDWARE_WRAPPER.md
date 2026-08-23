# `friday_hardware` — was gemessen wurde, benutzbar gemacht

**Stand:** 23. August 2026 · **Status:** Werkzeug, kein Forschungsergebnis.
Die zugrundeliegenden Messungen stehen in
[`DECODE_WIDTH_BEFUND_2026-08-23.md`](DECODE_WIDTH_BEFUND_2026-08-23.md),
[`GERAETEMODELL_2026-08-23.md`](GERAETEMODELL_2026-08-23.md) und
[`POLICY_GELTUNGSBEREICH_2026-08-23.md`](POLICY_GELTUNGSBEREICH_2026-08-23.md).

## Zweck

Elf Runden Messung ergeben nur dann etwas, wenn etwas sie liest. `friday_hardware`
trägt, was auf **einem** Gerät für **ein** Modell bei **einer** Quantisierung gemessen
wurde, und leitet daraus Planungsentscheidungen ab. Es misst selbst nichts.

## Benutzung

```python
from pathlib import Path
from friday_hardware import HardwareProfile

p = HardwareProfile.load(Path("profiles/m1max_gemma-3-4b-4bit-g64.json"))
p.require("mlx-community/gemma-3-4b-it-4bit", bits=4, group_size=64)

plan = p.plan(items=8, max_new_tokens=240, continuous_limit_s=6.0)
# Plan(width=32, steps_per_segment=51, segments=5, estimated_seconds=20.87,
#      reason='width 10 costs 90.1 ms and width 32 costs 87.0 ms; the extra
#              positions are free')
```

Profil erzeugen, aus vorhandenen Messberichten:

```
.venv/bin/python tools/build_hardware_profile.py --model 4b \
  --width-report  experiments/decode_width/report.json \
  --device-report experiments/device_model/report.json \
  --prefill-report experiments/decode_width/prefill.json \
  --out profiles/m1max_gemma-3-4b-4bit-g64.json
```

## Drei Verweigerungen, keine Funktionen

Der Entwurf besteht im Wesentlichen aus dem, was das Profil **nicht** tut.

**Es antwortet nicht für eine fremde Konfiguration.** `require()` wirft, wenn Modell,
Bitbreite oder Gruppengröße abweichen. Grund: die Breitenkurve verschob sich zwischen
zwei Modellen auf derselben Maschine und zwischen zwei Gruppengrößen beim selben
Modell. Ein Profil, das trotzdem antwortet, ist genau der Fehler, an dem diese
Messreihe viermal an eigenen Konstanten gescheitert ist — eine Zahl, die bei einer
Größenordnung stimmte und bei einer anderen still nicht mehr.

**Es interpoliert nicht zwischen gemessenen Breiten.** Die Kostenkurve ist eine
Treppenfunktion mit Plateaus und Klippen. Zwischenwerte zu erfinden hieße, Struktur zu
behaupten, die nie beobachtet wurde. Angeboten werden nur gemessene Breiten.

**Es projiziert nicht mit halben Angaben.** `project()` verlangt Bandbreite *und*
Dispatch-Kosten des Zielgeräts. Die Bandbreite eines Telefons mit den Dispatch-Kosten
dieses Laptops zu mischen unterstellt gleich schnelle Scheduler und lässt eine
Projektion wie eine Messung aussehen.

## Was es entscheidet

| Frage | Grundlage |
| :--- | :--- |
| Welche Breite für `n` Elemente? | gemessene Kurve, Regressionen ausgeschlossen, Gratis-Upgrades genutzt |
| Wie viele Schritte je Lastblock? | gemessene Schrittkosten gegen das Continuous-Limit, mit `25 %` Reserve |
| Was kostet ein Token bei Breite 1? | Zweitermmodell `Layer × je_Layer + GB × je_GB` |
| Wo steckt die Zeit? | `cost_shares()` trennt Dispatch von Bandbreite |
| Wie viele Stimmen? | `sample_budget()` meidet den Bereich `5`–`15`, der schlechter maß als eine |

Dieselbe Anfrage führt je nach Profil zu gegenteiligen Entscheidungen, und beide sind
richtig:

| Anfrage | 4B | 1B |
| ---: | :--- | :--- |
| 8 Elemente | Breite **`32`**, `5` Segmente, `20,87` s | Breite **`8`**, `1` Segment, `3,88` s |

Beim 4B ist Breite `8` eine Klippe; das Ausweichen auf `32` liefert viermal so viele
Positionen für weniger Zeit. Beim 1B gibt es diese Klippe nicht.

## Grenzen

Die mitgelieferten Profile gelten für **diesen** M1 Max, Gemma 3 `1B`/`4B`, `4 bit`
Gruppe `64`. Für jedes andere Gerät, Modell oder Quantisierungsformat ist ein eigenes
Profil zu messen; die Werkzeuge dafür liegen in `tools/`.

`sample_budget()` stützt sich auf Genauigkeitsmessungen, nicht auf Hardware. Die
Empfehlung `1` oder `32` und die Meidung von `5`–`15` stammen aus einer einzigen
Aufgabenfamilie (mehrschrittige Arithmetik mit prüfbarer Lösung) und übertragen sich
nicht ungeprüft auf offene Aufgaben.

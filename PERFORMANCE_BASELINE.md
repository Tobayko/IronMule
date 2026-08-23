# Performance-Baseline

Stand: 24. August 2026, Gemma 3 4B 4-bit g64 auf Apple M1 Max, MLX `0.32.0`,
mlx-lm `0.31.3`. Alle Werte gemessen, keiner geschätzt. Alles `formal_claim=false`.

## Gerätemodell

```
ms_je_Token(Breite 1) = Layer × 0,16669 ms  +  Gewichte_GB × 2,79005 ms
```

Gefittet auf zwei Modellen, geprüft an vier zurückgehaltenen Konfigurationen
(Fehler `+0,9` bis `+15,5 %`). Effektive Bandbreite `358,4` GB/s = `89,6 %` der
veröffentlichten Spitze.

| Anteil am 4B-Einzeltoken | ms | Anteil |
| :--- | ---: | ---: |
| Dispatch (`34` Layer) | `5,67` | `48 %` |
| Gewichte lesen | `6,09` | `52 %` |

## TTFT nach Klassen

Klassen werden **nicht** in einem gemeinsamen Schätzer vermischt.

| Klasse | Wert | Bedingung |
| :--- | ---: | :--- |
| Modellladen | `1,47`–`1,76` s | Anteil an `cold_process` |
| `warm_uncached` | `1702,86` ms | `898`-Token-Prompt |
| `warm_prefix_hit` | `131,02` ms | `886` Token wiederverwendet — **korrektheitsungeprüft**, siehe unten |
| `warm_full_cache_hit` | nicht gemessen | – |

## Prefill

| Blockgröße | Positionen/s | ms je Position |
| ---: | ---: | ---: |
| 256 | `505,4` | `1,979` |
| 512 | `530,7` | `1,884` |
| 1024 | `543,5` | `1,840` |
| 2048 | `544,1` | `1,838` |

Sättigt ab `1024`. Chat-Template und Tokenisierung liegen bei `0,044`–`0,649` ms und
sind kein Faktor.

## Decode-Durchsatz

| Modell | Einzelstrom | bester Batch | Faktor | Peak |
| :--- | ---: | ---: | ---: | ---: |
| 4B | `82,4` tok/s | `64` → `493,0` | `5,98x` | `6,55` GB |
| 1B | `225,1` tok/s | `256` → `3093,0` | `13,74x` | `3,60` GB |

Breitenkurve ist eine Treppenfunktion. Regressionen 4B: `6,7,8,9,48`; 1B: `48`.
Breite `48` regressiert in allen sechs geprüften Quantisierungskonfigurationen und ist
der einzige als konstant behandelbare Wert.

## Kontextbasierte Spekulation

Auf echtem Projektinhalt, Median aus Wiederholungen nach Aufwärmlauf:

| Prompt | Akzeptanz | Speedup |
| :--- | ---: | ---: |
| Quelldatei umschreiben | `1,000` | `1,097` |
| Journal-Extraktion | `0,682` | `1,029` |
| Testausgabe | `0,375` | `0,994` |

Tiefe folgt der Trefferlänge: Übereinstimmungen ab `9` Token wurden `48/48` akzeptiert,
kürzere zu `53,6 %` (4B) bzw. `47,2 %` (1B).

## Erkannter Engpass

**Prompt Prefill.** Ein `898`-Token-Prompt kostet `1,70` s bis zum ersten Token,
gegenüber `12,1` ms je Ausgabetoken. Für einen Agenten mit stabilem Präfix dominiert
der Prefill die wahrgenommene Latenz um mehr als zwei Größenordnungen je Anfrage.

## Unsicherheiten und offene Punkte

- **Tokenidentität ist bei geänderter Prefill-Zerteilung nicht gegeben.** Siehe
  `OVERNIGHT_RESEARCH_LOG.md`, Zyklus 1. Das betrifft jede Optimierung, die die
  Blockstruktur verändert — Präfix-Cache und Batching eingeschlossen.
- `warm_full_cache_hit` ungemessen.
- Energie je Token ungemessen (benötigt Freigabe).
- Wired Memory, Memory Compression und Thermal State bisher nicht erfasst.
- Alle Werte gelten für **ein** Gerät, **ein** Modell, **eine** Quantisierung.

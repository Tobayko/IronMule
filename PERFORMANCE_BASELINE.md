# Performance-Baseline

Stand: 24. August 2026, nach Zyklus 12, Gemma 3 4B 4-bit g64 auf Apple M1 Max, MLX `0.32.0`,
mlx-lm `0.31.3`. Werte sind gemessen, sofern sie nicht ausdrücklich als Rechnung
markiert sind. Der einzige neue formale Claim ist die unten abgegrenzte
Prefill-Head-Skip-Studie; alle übrigen Werte dieser Baseline bleiben
`formal_claim=false`.

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

## Formal bestätigter Prefill-Head-Skip

Zyklus 12 prüfte prospektiv genau einen Kandidaten: Beim greedy Prefill ohne
Prompt-Logprobs wird der LM-Head nur auf die tatsächlich gelesene letzte
Promptposition angewendet. Die versiegelte Studie verwendete einen lokalen,
revisionsgebundenen Modell-Snapshot, `897` Prompt-Token, Prefill-Chunk `256`, Batch
`1`, sechs A/A- und sechs A/B-Sessionprozesse sowie vier Messpaare je Session.

| Vorregistrierter Endpunkt | Ergebnis |
| :--- | ---: |
| A/A-Verhältnis | `1,002829` |
| A/A-95-%-KI | `[0,994931; 1,005964]` |
| eingefrorene MDE | `5 %` |
| A/B-Verhältnis gesamt | **`0,846385`** |
| Effekt | **`−15,3615 %`** |
| Charakterisierung, 95-%-KI | `[0,840544; 0,848452]` |
| Validierung, 95-%-KI | `[0,842683; 0,854941]` |
| Gesamt, 95-%-KI | `[0,843147; 0,851284]` |
| Greedy-Tokenidentität | **`12/12` Sessiongates** |

Das Verhältnis, der Effekt und die Intervalle sind die vorregistrierte Auswertung
der unverändert gespeicherten Messblöcke. Aus den gemessenen Sessionmedianen
abgeleitet lagen die Armmediane bei `1995,444239` ms und `1688,116333` ms. Beide
Arme meldeten denselben MLX-Peak von `3.213.903.666` Byte; der beobachtete RSS-Bereich
war `3.768.795.136` bis `3.769.696.256` Byte.

Der terminale Status lautet `head_skip_gain_confirmed`, die Aktion lediglich
`permit_bounded_architecture_review`. `formal_claim=true` gilt ausschließlich für
**ein Gerät, einen Modell-Snapshot, einen Prompt, einen Prefill-Plan und greedy ohne
Prompt-Logprobs**. Es gibt keine automatische Produktaktivierung und keinen
allgemeinen TTFT-, Modell- oder Cross-Device-Claim.

## Decode-Durchsatz

| Modell | Einzelstrom | bester Batch | Faktor | Peak |
| :--- | ---: | ---: | ---: | ---: |
| 4B | `82,4` tok/s | `64` → `493,0` | `5,98x` | `6,55` GB |
| 1B | `225,1` tok/s | `256` → `3093,0` | `13,74x` | `3,60` GB |

Breitenkurve ist eine Treppenfunktion. Regressionen 4B: `6,7,8,9,48`; 1B: `48`.
Breite `48` regressiert in allen sechs geprüften Quantisierungskonfigurationen und ist
der einzige als konstant behandelbare Wert.

## Inter-Token-Latenz und KV-Reallokationen

Zyklus 11: `765` Prompt-Token, `48` Decodeschritte, Batch `1`, acht
Wiederholungen nach einem verworfenen Aufwärmlauf. Es wurden keine Ausreißer
verworfen.

| Endpunkt | ms |
| :--- | ---: |
| ITL p50 | `14,2670` |
| ITL p95 | `15,1385` |
| ITL p99 | `46,7879` |
| Minimum | `13,8230` |
| Maximum | `49,4430` |

Die Cacheformen änderten sich in allen acht Wiederholungen an denselben Stellen:

| Decodeschritt | Cacheklasse | Layer | gemessener Überschuss | vorab gerechnet |
| ---: | :--- | ---: | ---: | ---: |
| `1` | rotierend | `29` | `31,5853` ms | `0,7616` ms |
| `4` | global | `5` | `0,2968` ms | `0,1317` ms |

Die Summe der gemessenen Überschüsse entspricht `4,4263 %` der Decodezeit und
überschreitet die vorregistrierte `1-%`-Schwelle. Die vorhergesagten `0,13 %` waren
eine Bandbreitenrechnung, kein Messwert. Der große Ausschlag an Schritt `1` erklärt
den p99, ist aber zugleich mit sonstigen Kosten des ersten Decodeschritts konfundiert.
Die Messung lokalisiert daher einen Kandidaten; sie beweist noch nicht, dass eine
Cache-Änderung `4,4263 %` gewinnt. `formal_claim=false`.

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
- Multi-Turn-Fortsetzung bisher nicht als Baseline gemessen.
- Mehrere parallele Requests bisher nicht als Baseline gemessen; `concurrent_32` in
  `EXPERIMENT_MATRIX.json` definiert nur den Workload.
- Alle Werte gelten für **ein** Gerät, **ein** Modell, **eine** Quantisierung.

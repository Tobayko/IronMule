# Wie weit trägt die Breiten-Policy?

**Stand:** 23. August 2026 · **Status:** explorativ, `formal_claim=false`.
Drei Prüfungen an den Aussagen aus
[`DECODE_WIDTH_BEFUND_2026-08-23.md`](DECODE_WIDTH_BEFUND_2026-08-23.md). Keine davon
war eine neue Optimierung; jede war ein Versuch, eine bestehende Aussage zu brechen.

## 1. Gemischte Promptlängen — hält

Alle bisherigen Batch-Messungen wiederholten **einen** Prompt. Echte Anfragen sind
verschieden lang, dann muss aufgefüllt werden, und Padding ist Arbeit ohne Ertrag.
Falls das den gemessenen Durchsatz auffrisst, beschreibt die Hauptzahl eine Last, die
niemand fährt.

Batch `16`, gleiche mittlere Länge, Steady-State über die Steigung:

| Arm | Längen | Padding-Verschwendung | ms/Schritt | tok/s | vs. identisch |
| :--- | :--- | ---: | ---: | ---: | ---: |
| identisch | alle `64` | `0 %` | `82,48` | `194,0` | `1,000` |
| **gemischt** | `18`–`110` | **`40,6 %`** | `83,39` | `191,9` | **`0,989`** |

`40,6 %` verschwendete Positionen kosten `1,1 %` Decode-Durchsatz.

Der Grund ist strukturell: Padding trifft den **Prefill**, nicht den Decode-Schritt.
Der Prefill wurde entsprechend teurer (`2,706` gegen `1,789` s, `+51 %`), aber sobald
dekodiert wird, liefert jede Sequenz genau eine Position je Schritt, unabhängig davon,
wie lang ihr Prompt war.

**Einschränkung:** Bei sehr kurzen Ausgaben dominiert der Prefill, und dort schlägt das
Padding voll durch. Die Aussage lautet "Decode ist immun", nicht "Padding ist gratis".

## 2. Jedes Modell braucht seine eigene Policy

| | 4B | 1B |
| :--- | ---: | ---: |
| beste Breite | `64` | `64` |
| ms je Position | `2,3028` | **`0,431`** |
| Gewinn gegen Breite 1 | `6,16x` | **`15,03x`** |
| **Regressionen** | **`6, 7, 8, 9, 48`** | **`48`** |
| vom Generierungsloop realisiert | `4,651x` | **`8,847x`** |

Die Klippe bei `6`–`9` **existiert auf dem 1B nicht**. Ein Router, der die 4B-Policy
auf das 1B anwendet, meidet Breiten, die dort einwandfrei sind, und lässt einen
Gewinn von `15x` auf `6x` schrumpfen.

Das 1B profitiert deutlich stärker vom Batching, weil es bei Breite `1` stärker
overhead-dominiert ist — konsistent mit den `36,5 %` Bandbreitenausnutzung aus dem
früheren Roofline-Lauf.

## 3. Was an der Quantisierung hängt und was nicht

Die Klippe sitzt im quantisierten Matmul, also wurde sie dort geprüft: derselbe
FFN-Kettentest über sechs Konfigurationen.

| Konfiguration | Regressionen | beste Breite |
| :--- | :--- | ---: |
| 4 bit, Gruppe 32 | `48` | `64` |
| **4 bit, Gruppe 64** | **`5, 6, 48`** | `64` |
| 4 bit, Gruppe 128 | `48` | `64` |
| 8 bit, Gruppe 32 | `48` | `64` |
| **8 bit, Gruppe 64** | **`5, 48`** | `64` |
| 8 bit, Gruppe 128 | `5, 48` | `64` |

**Breite `48` bricht in allen sechs Konfigurationen ein**, bei beiden Bitbreiten, allen
drei Gruppengrößen und beiden Modellen. Das ist eine Kernelgrenze, keine Eigenschaft
des Modells oder der Quantisierung — und damit verallgemeinerbar.

**Die schmale Klippe bei `5`–`6` tritt nur bei Gruppengröße `64` auf.** Bei `32`
verschwindet sie vollständig. Die produktiven Modelle dieses Projekts laufen auf
`4 bit` mit Gruppe `64`, also genau in der Konfiguration, in der sie existiert.

## 4. Folgerung für eine produktive Integration

Die Policy ist **teils Werkzeug, teils Momentaufnahme**:

| Aussage | Geltung |
| :--- | :--- |
| Breite `48` meiden | konfigurationsunabhängig, beide Modelle, alle sechs Quantisierungen |
| Breite `64` ist optimal | in allen geprüften Fällen |
| Breiten `5`–`9` meiden | **nur** 4B bei Gruppengröße `64` |
| Gratis-Upgrades auf `32` | modellspezifisch, je Modell zu messen |
| Decode ist padding-tolerant | gilt für lange Ausgaben, nicht für kurze |

Wer die Policy in einen Router einbaut, muss sie also **je Paar aus Modell und
Quantisierung** messen. Als feste Konstante taugt nur die `48`-Grenze. Genau diesen
Fehler — eine Zahl, die bei einer Konfiguration stimmt und bei einer anderen still
nicht mehr — hat diese Messreihe an eigenen Konstanten viermal gemacht; er gehört
nicht in ein produktives System.

## 5. Reproduktion

```
.venv/bin/python experiments/policy_scope/measure_mixed_prompt_lengths.py
.venv/bin/python experiments/policy_scope/measure_quantisation_scope.py <ausgabeverzeichnis>
.venv/bin/python tools/measure_decode_width.py --execute --model 1b --out <pfad>
.venv/bin/python tools/measure_decode_width.py --execute --model 4b --out <pfad>
```

Alle Läufe am Netzteil unter `BudgetGuard`, offline, ohne Download oder Installation.

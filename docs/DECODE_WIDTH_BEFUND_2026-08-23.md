# Decode-Breite: warum Phase 1B den falschen Kernel traf

**Stand:** 23. August 2026 · **Gerät:** Apple M1 Max, 32 GB · **Status:** explorativ,
`formal_claim=false`. Kein Vertrag, keine Vorregistrierung, kein Promotionsanspruch.

Reproduktion:

```
.venv/bin/python tools/measure_speculative.py  --execute --out <pfad>
.venv/bin/python tools/measure_decode_width.py --execute --out experiments/decode_width/report.json
```

Beide Werkzeuge nutzen `release_gate`, `require_ac_power`, den Offline-Snapshot-Resolver
und `BudgetGuard`. Es wurde nichts heruntergeladen und nichts installiert.

## 1. Der Anlass: spekulatives Decoding schlägt fehl

Gemma 3 1B als Draft für Gemma 3 4B, greedy (`temp=0`), drei balancierte Blöcke mit
alternierender Armreihenfolge, 64 Token je Lauf:

| Draft-Länge k | tok/s | Speedup | Akzeptanzrate | Token-identisch |
| ---: | ---: | ---: | ---: | :--- |
| 0 (Baseline) | `92,96` | `1,000` | – | ja |
| 2 | `52,09` | `0,560` | `0,500` | ja |
| 3 | `44,84` | `0,482` | `0,402` | ja |
| 4 | `42,52` | `0,457` | `0,390` | ja |
| 6 | `28,31` | `0,305` | `0,278` | **nein** |

Jede Draft-Länge ist langsamer als gar kein Draft. Der Baseline-Wert `92,96` tok/s
deckt sich mit dem früheren Roofline-Lauf (`91,3`), das Messsystem ist also mit dem
bestehenden konsistent.

Zwei separat gemessene Ursachen:

**(a) Der Draft ist zu teuer.** Direkt gemessen: 4B `10,8481` ms/Token, 1B `4,9994`
ms/Token, Kostenverhältnis **`c = 0,4609`**. Ein Draft, der fast die halbe
Zielmodellzeit kostet, muss nahezu jeden Vorschlag treffen, um sich zu lohnen.
Gemessen wurden `0,28`–`0,50`.

**(b) Der Verify-Pass ist nicht gratis.** Das ist die eigentliche Ursache, und sie
widerspricht der Schlussfolgerung, auf der das ganze Verfahren ruht.

## 2. Der Befund: der Forward-Pass ist keine Konstante in der Breite

Der Roofline-Lauf stufte die Generierung als memory-bound ein. Daraus folgt scheinbar,
dass `k+1` Positionen in einem Pass etwa so viel kosten wie eine — die Gewichte werden
so oder so einmal gelesen. **Diese Folgerung gilt hier nur stückweise.**

Vollmodell, warmer Cache, Median aus 12 Wiederholungen:

| Breite | ms | ms/Position | Durchsatz vs. Breite 1 |
| ---: | ---: | ---: | ---: |
| 1 | `13,049` | `13,049` | `1,00` |
| 2 | `20,504` | `10,252` | `1,27` |
| 4 | `33,074` | `8,269` | `1,58` |
| 8 | `81,400` | `10,175` | `1,28` |
| 12 | `81,879` | `6,823` | `1,91` |
| 16 | `79,998` | `5,000` | `2,61` |
| 24 | `89,367` | `3,724` | `3,50` |
| **32** | **`78,190`** | **`2,443`** | **`5,34`** |
| 48 | `139,446` | `2,905` | `4,49` |
| 64 | `136,203` | `2,128` | `6,13` |

Entscheidend sind die Grenzkosten je zusätzlicher Position:

| Übergang | ms/Position |
| :--- | ---: |
| 1 → 2 | `+7,455` |
| 2 → 4 | `+6,285` |
| **4 → 8** | **`+12,082`** |
| **8 → 12** | **`+0,120`** |
| 12 → 16 | `−0,470` |
| 16 → 24 | `+1,171` |
| 24 → 32 | `−1,397` |
| 32 → 48 | `+3,829` |
| 48 → 64 | `−0,203` |

Die Kosten sind eine **Treppenfunktion**, keine Gerade. Es gibt ein Plateau von Breite
`8` bis `32`: `81,4` → `78,2` ms, also viermal so viele Positionen zu **negativen**
Mehrkosten. Davor liegt eine pathologische Zone: Breite `1`–`8` zahlt `6`–`12` ms je
Position, fast eine volle Passkosten pro zusätzlichem Token.

Negative Grenzkosten bedeuten, dass zwischen den Breiten ein anderer Kernelpfad greift.
Breite `32` ist absolut billiger als Breite `8`.

## 3. Warum das Kapitel 1 vollständig erklärt

Mit den gemessenen Verify-Kosten statt der Gratis-Annahme:

```
Speedup = E[Token](α,k) · T(1) / ( T(k+1) + k · T_draft(1) )
```

Für k=2, α=0,5: `E = 1,75`; `T(3) ≈ 26,54` ms; `2 · 6,69 = 13,37` ms.
Ergibt `1,75 · 13,54 / 39,91` = **`0,594`** vorhergesagt gegen **`0,560`** gemessen.

Das Modell ist damit prädiktiv. Spekulatives Decoding mit kleinem `k` landet
zwangsläufig in der pathologischen Zone `2`–`6` — genau dort, wo eine zusätzliche
Position am teuersten ist. Das Verfahren scheitert nicht an der Idee, sondern an der
Breite, die es benutzt.

## 4. Was das Plateau tatsächlich einbringt

`batch_generate`, greedy, 24 Token je Prompt, Median aus zwei Wiederholungen:

| Batchgröße | tok/s | vs. Batch 1 |
| ---: | ---: | ---: |
| 1 | `52,63` | `1,000` |
| 2 | `72,38` | `1,375` |
| 4 | `89,36` | `1,698` |
| 8 | `80,28` | `1,525` |
| 16 | `122,09` | `2,320` |
| **32** | **`169,25`** | **`3,216`** |

Die Klippe bei Breite `8` erscheint hier wieder als Einbruch bei Batch `8`. Das ist
dieselbe Ursache, einmal im Kernel und einmal im Generierungsloop sichtbar.

**Ehrliche Basislinie:** `batch_generate` bei Größe `1` liefert `52,63` tok/s, während
`stream_generate` auf demselben Modell `92,18` tok/s erreicht. Der Batchpfad trägt also
eigenen Overhead. Gegen den *besten* Einzelstrompfad gemessen sind es
`169,25 / 92,18` = **`1,84x`**, nicht `3,22x`. Der kleinere Wert ist der belastbare.

## 5. Die Lücke ist der eigentliche Befund

| Größe | Wert |
| :--- | ---: |
| Forward-Pass-Obergrenze (Breite 64) | `6,13x` |
| Vom Generierungsloop realisiert | `3,22x` |
| **Nicht realisiert** | **`1,91x`** |

Fast eine Verdopplung liegt zwischen dem, was ein Forward-Pass auf dieser Hardware
kann, und dem, was der Generierungsloop daraus macht. Diese Lücke sitzt in der
Softwareschicht, nicht in der Hardware.

## 6. Konsequenz für Phase 1B

Phase 1B fusionierte Residual-Add plus RMSNorm und erreichte `1,870 %` gegen ein
`5 %`-Gate — ein gültiger Negativbefund. Die Messungen hier ordnen ihn ein: RMSNorm
war nie die Engstelle. Die breitenabhängige Matmul-Pfadwahl ist es. Der Befund
"Custom Metal zahlt sich nicht aus" ist damit **nicht** gestützt; gestützt ist nur
"dieser Kernel zahlte sich nicht aus".

Die naheliegende Anschlussfrage ist die Breitenabhängigkeit selbst: Breite `1`–`8`
läuft weit unter dem, was Breite `32` auf derselben Hardware zeigt. Ob das ein
MLX-Kernelpfad, eine Gruppengrößenwahl oder eine Dispatchgrenze ist, ist hier **nicht**
geklärt und wäre der nächste isolierte Versuch.

## 7. Anschluss an den bestehenden Shadow-Router

`friday_avo_router/` wählt bereits evidenzgebunden zwischen seriellem und gebündeltem
Plan und erzwingt `serial_shadow_only`. Die dort gebaute Struktur — reale
Tensor-Metadaten beobachten, versiegelte Evidenz prüfen, genau einen bekannten Plan
wählen, sonst seriell zurückfallen — ist auf die Decode-Breite unverändert anwendbar.
Der Unterschied ist die Größenordnung des Hebels: bisher `12 %` auf Matmul-Anzahl,
hier `1,84x` gemessen bei `6,13x` Obergrenze.

Das ist **kein** Freigabevorschlag. Eine Übertragung verlangt eigene Vorregistrierung,
eigenes A/A-Gate und eine eigene MDE, genau wie N8, N10 und Phase 1B sie hatten.

## 8. Nebenbefund: Korrektheit bei k=6

Unter greedy Sampling muss spekulatives Decoding tokenidentisch zur Baseline sein; das
Akzeptanzkriterium lässt nichts anderes zu. Für `k ∈ {2,3,4}` war es das auch. Für
`k=6` wich die Ausgabe ab. Ursache nicht geklärt — numerisches Gleichstand-Verhalten
oder ein Implementierungsdefekt in `mlx_lm 0.31.3` sind beide möglich. Der Lauf wurde
nicht wiederholt, und der Befund wird hier nur festgehalten, nicht bewertet.

## 9. Grenzen

Ein Gerät, ein Modell, eine Quantisierung (`4 bit`, Gruppengröße `64`), eine
MLX-Version (`0.32.0`), ein `mlx_lm` (`0.31.3`). Kein Cross-Device-Anspruch, kein
Modellanspruch, keine Aussage über Antwortqualität außer der gemessenen
Tokenidentität. Alle Läufe am Netzteil unter `BudgetGuard`; im kanonischen Lauf
`40,25` s GPU-Arbeit, maximal `4,573` s kontinuierlich, `216,27` s verifizierte
Pausen, `268,87` s Wall.

---

# Nachtrag: die Klippe lokalisiert und in eine Policy gegossen

**Ergänzt:** 23. August 2026, dritte Messrunde.

## 10. Wo die Klippe steckt: nicht im LM-Head

Der Verdacht lag auf dem Vokabular — Gemma 3 trägt `262208` Einträge, der LM-Head
ist `1,343` GFLOP je Position. Die Zerlegung eines Schritts widerlegt das:

| W | full | **body** | head | sampler | Head-Anteil |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `14,40` | **`12,69`** | `1,99` | `0,53` | `13,8 %` |
| 4 | `35,61` | **`30,51`** | `4,34` | `0,53` | `12,2 %` |
| 8 | `82,19` | **`71,52`** | `8,20` | `0,74` | `10,0 %` |
| 16 | `88,17` | **`78,57`** | `8,36` | `0,72` | `9,5 %` |
| 32 | `84,60` | **`78,38`** | `7,81` | `0,69` | `9,2 %` |

Der Head amortisiert vorbildlich: `8,20` ms bei `W=8` gegen `7,81` ms bei `W=32`, also
`0,68` auf `5,5` TFLOPS. Der Sampler ist mit `0,7` ms irrelevant. **Die gesamte Klippe
sitzt im Transformer-Body.**

## 11. Die Schwelle wandert mit der Modellform

Feine Abtastung des Body, beide Modelle:

| W | 4B ms | 4B ms/Pos | 1B ms | 1B ms/Pos |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `13,015` | `13,015` | `6,713` | `6,713` |
| 5 | `36,940` | `7,388` | `12,044` | `2,409` |
| **6** | **`68,771`** | `11,462` | `14,708` | `2,451` |
| 32 | `76,593` | **`2,394`** | `17,023` | **`0,532`** |

Beim 4B kostet der Schritt von `W=5` auf `W=6` — **eine einzige Position** —
`+31,83` ms. Der Schritt von `W=6` auf `W=32`, also `26` Positionen, kostet
`+7,82` ms. Das 1B hat diese Klippe **nicht**; seine Grenzkosten bleiben zwischen
`0,6` und `2,7` ms.

Weil die Klippe mit der Modellform wandert, ist sie ein **Tiling-Effekt**, keine feste
Dispatch-Konstante. Das schließt eine ganze Klasse von Erklärungen aus.

Nebenbefund mit Konsequenz: bei `W=1` kostet 4B nur `1,94x` das 1B, bei `W=32` aber
`4,50x`. Der in Abschnitt 3 des Nachbardokuments beschriebene Vorteil "größeres Modell
ist billig" gilt für **Einzelstrom**, nicht für Batch-Betrieb. Diese Einschränkung war
in der ersten Fassung nicht sichtbar.

## 12. Gemessene Dispatch-Policy

Aus der Kostenkurve abgeleitet, Toleranz `5 %`:

| Größe | Wert |
| :--- | :--- |
| Bestbreite | `64` (`2,282` ms/Position, `6,355x`) |
| Regressionen | `6, 7, 8, 9, 48` |
| Gratis-Upgrades | `9→32`, `10→32`, `12→32`, `14→32`, `16→32`, `24→32`, `48→64` |

**Regression** heißt: bei dieser Breite ist die Kosten pro Position schlechter, als eine
*schmalere* Breite bereits erreicht hat. Breiter zu werden hat aktiv geschadet. Kein
Kostenmodell sagt das vorher.

**Gratis-Upgrade** heißt: die breiteste Breite, deren *absolute* Kosten innerhalb der
Toleranz dieser liegen. Wer `12` braucht, nimmt `32` — gleiche Kosten, `2,7x` der
Positionen.

Beide Aussagen existieren nur, weil die Kurve Plateaus hat. Die triviale Aussage "nimm
die breiteste Breite" wurde bewusst nicht als Policy ausgegeben; sie gilt für jede
monoton fallende Kurve und sagt einem Dispatcher nichts.

## 13. Streuung zwischen Läufen

Der zweite kanonische Lauf wich vom ersten ab: `W=6` `78,74` gegen `68,77` ms,
Batch `32` `156,78` gegen `169,25` tok/s. Die **Struktur** — Klippe, Plateau,
Gratis-Upgrades — war in beiden Läufen identisch; die **Absolutwerte** driften um
rund `7 %`. Jede Schwelle, die enger als das gesetzt würde, wäre Rauschen.
Die `5 %`-Toleranz der Policy liegt bewusst in derselben Größenordnung.

---

# Korrektur: die realisierte Zahl war zu niedrig gemessen

**Ergänzt:** 23. August 2026, vierte Messrunde. **Dieser Abschnitt hebt Zahlen aus
Abschnitt 4, 5 und 12 auf.** Die aufgehobenen Werte bleiben oben stehen, damit die
Korrektur nachvollziehbar ist.

## 14. Der Fehler

Die Abschnitte 4 und 5 berichteten `1,84x` beziehungsweise `3,22x` als realisierte
Batch-Beschleunigung, gegen eine Forward-Pass-Obergrenze von `6,13x`, und schlossen
auf `1,91x` ungenutztes Potenzial in der Softwareschicht.

Diese Messung teilte die Gesamtzeit eines `batch_generate`-Aufrufs durch die Anzahl
erzeugter Token. Damit wird der **Prompt-Prefill jedem einzelnen Schritt angelastet**.
Bei Batch `32` beträgt dieser Prefill `2,80` s; auf `24` Token verteilt erfindet er
rund `117` ms Kosten je Schritt, die es nicht gibt.

Nachgewiesen über einen linearen Fit gegen `max_tokens ∈ {2, 6, 12, 20}`:

| Größe | Wert |
| :--- | ---: |
| Fixkosten (Prefill plus Setup) | `2072` ms |
| echte Kosten je Schritt | `98,3` ms |
| naiv gerechnet (`total / 20`) | `198,7` ms |

Der Prefill selbst ist **nicht** verschwenderisch: `32 × 25 = 800` Positionen bei
`2,28` ms je Position sind `1,8` s. Er gehört nur nicht in die Schrittkosten.

Eine Zwischenhypothese wurde dabei ebenfalls widerlegt. `batch_generate` berechnet je
Schritt volle Log-Probabilities über das gesamte Vokabular
(`generate.py:1352`, `33,6` MB bei Batch `32`) und materialisiert sie je Sequenz
(`generate.py:1367`). Das sah nach der Ursache aus. Direkt gemessen kostet es nichts:
`argmax` `83,46` ms, `+logsumexp` `84,53` ms, `+Materialisierung` `82,73` ms,
`+Host-Sync` `82,22` ms — alle vier innerhalb der Streuung.

## 15. Korrigierte Zahlen

Steady-State über die Steigung zwischen zwei Tokenzahlen, Prefill herausgerechnet:

| Batch | ms/Schritt | Fixkosten | tok/s | naiv | vs. Batch 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12,13` | `0,20` s | `82,44` | `48,78` | `1,000` |
| 2 | `19,36` | `0,25` s | `103,29` | `66,93` | `1,253` |
| 4 | `30,82` | `0,41` s | `129,78` | `83,47` | `1,574` |
| **8** | `77,48` | `0,67` s | **`103,25`** | `75,93` | **`1,252`** |
| 16 | `80,27` | `1,39` s | `199,33` | `115,83` | `2,418` |
| **32** | `83,46` | `2,80` s | **`383,44`** | `159,99` | **`4,651`** |

| Größe | alt (falsch) | neu |
| :--- | ---: | ---: |
| realisiert | `1,84x` / `3,22x` | **`4,651x`** |
| Framework-Overhead je Schritt | `54 %` | **nicht messbar** |
| nicht realisiert | `1,91x` | `1,324x` |

Ein minimaler Decode-Loop — Forward, Sample, Anhängen, ohne Detokenisierung und ohne
Stop-Sequenz-Maschinerie — erreicht `83,5` ms je Schritt und `383` tok/s. Das ist
derselbe Wert, den `batch_generate` im Steady-State liefert. **Der Framework-Pfad
kostet nichts Messbares**; die frühere Aussage, dort lägen `1,9x`, war falsch.
`mx.async_eval` bringt ebenfalls nichts (`84,22` gegen `84,33` ms): der Loop ist
GPU-gebunden, es gibt keine Host-Arbeit zu überlappen.

## 16. Die Policy sagt den Generierungsloop korrekt vorher

Die aus dem Forward-Pass abgeleitete Policy nannte `regression_widths`
`[6, 7, 8, 9, 48]`. Der Generierungsloop, unabhängig gemessen, bricht bei Batch `8`
auf `103,25` tok/s ein und liegt damit **unter** Batch `4` mit `129,78` tok/s.

Eine aus Mikrostruktur abgeleitete Vorhersage, die sich im Makroverhalten bestätigt,
ist der eigentliche Beleg dafür, dass die Policy etwas Reales beschreibt und nicht
eine Eigenheit der Messmethode.

## 17. Was jetzt offen bleibt

`1,324x` zwischen realisiertem Batch-32-Durchsatz und der Forward-Pass-Obergrenze bei
Breite `64`. Der Generierungsarm wurde nur bis Batch `32` gemessen; Breite `64` ist im
Forward-Pass um `13 %` besser je Position. Ob Batch `64` das einlöst, ist **nicht**
gemessen — der Speicherbedarf des KV-Caches wächst linear mit dem Batch, und ein
unbedachter Lauf wäre genau die Art Überlastung, die hier vermieden werden soll.

## 18. Batch 48 und 64: Policy erneut bestätigt, Batch 64 operativ nicht nutzbar

Kurzer Prompt (`14` Token), damit der Prefill nicht den ganzen Lastblock füllt:

| Batch | ms/Schritt | tok/s | Fixkosten | Peak |
| ---: | ---: | ---: | ---: | ---: |
| 32 | `91,61` | `349,3` | `1,83` s | `4,80` GB |
| **48** | `160,22` | **`299,6`** | `2,54` s | `5,13` GB |
| 64 | `164,94` | `388,0` | `3,50` s | `5,13` GB |

Batch `48` liegt **unter** Batch `32`. Die Policy hatte `48` in `regression_widths`
gelistet, aus einer Forward-Pass-Messung ohne jeden Bezug zum Generierungsloop. Das
ist nach Batch `8` die **zweite unabhängige Bestätigung** derselben Vorhersage.

Batch `64` bringt `+11 %` gegenüber Batch `32` und liegt damit nahe an den `+17 %`,
die der Forward-Pass je Position vorhersagt. Ein Lauf mit `max_tokens=16` **überschritt
das `6`-s-Continuous-Limit des Guards** und wurde fail-closed abgebrochen; nur mit
`max_tokens ≤ 8` war er messbar. Der Gewinn ist real, aber unter der
Sicherheitsrichtlinie dieses Projekts nicht abrufbar.

**Operative Empfehlung: Batch `32`.** Batch `48` meiden, Batch `64` nur, wenn das
Continuous-Load-Budget bewusst neu verhandelt wird.

## 19. Endstand der Messreihe

Gegen den **besten Einzelstrompfad** (`stream_generate`, `92,18` tok/s):

| Pfad | tok/s | Faktor |
| :--- | ---: | ---: |
| Einzelstrom | `92,18` | `1,00x` |
| Batch 32, Steady-State | `383,44` | **`4,16x`** |
| Forward-Pass-Obergrenze, Breite 64 | – | `6,16x` |

Der Faktor `4,651x` aus dem Werkzeugbericht verwendet `batch_generate` bei Batch `1`
(`82,44` tok/s) als Nenner. Gegen den tatsächlich schnellsten Einzelstrompfad sind es
`4,16x`; das ist die belastbarere Zahl und die, die berichtet werden sollte.

---

# Nachtrag: der Konflikt aus Abschnitt 11 ist gelöst

**Ergänzt:** 23. August 2026, sechste Messrunde.
Werkzeug: `tools/measure_segmented_decode.py`, Bericht:
`experiments/segmented_decode/report.json`.

## 20. Das Problem

Die effizienteste Breite (`32`, `2,61` ms je Sample-Schritt) erlaubt unter dem
`6`-s-Continuous-Limit des Guards nur `71` Schritte je Lastblock. Realistische
Antworten brauchen `240`–`288`. Der bisher einzige konforme Ausweg war eine schmale
Breite, die `3,71x` teurer je Sample ist. Zwei Läufe der Vorrunde scheiterten daran
fail-closed — korrekt, aber es kostete die Messung.

## 21. Die Beobachtung, die es auflöst

Ein KV-Cache ist Zustand. Eine Pause zwischen zwei Decode-Schritten ändert nichts
daran, was das Modell als Nächstes rechnet. Also lässt sich die Generierung in
Segmente schneiden, jedes ein eigener Lastblock mit Guard-Pause danach, und die
effiziente Breite über die gesamte Länge halten.

Das ist eine Behauptung über Numerik und wurde entsprechend geprüft, **bevor**
irgendeine Zeit gemessen wurde: unter greedy Sampling muss ein segmentierter Lauf
tokenidentisch zu einem unsegmentierten sein. Bei `48` Token, aufgeteilt in `3`
Segmente zu `16`, war er das. Hätte er es nicht, wäre jeder Geschwindigkeitsvergleich
bedeutungslos, weil die Arme nicht dasselbe täten.

Die Segmentlänge wird aus den gemessenen Schrittkosten und dem Limit des Guards
**abgeleitet**, nicht gesetzt: `6` s mal `75 %` Reserve geteilt durch `90` ms sind
`50` Schritte. Eine feste Konstante wäre auf einem anderen Modell still unsicher
geworden. Die Reserve ist kein Zierrat — Schrittkosten schwanken mit Temperatur und
KV-Länge, und ein Segment, das das Limit exakt ausfüllt, scheitert beim ersten
langsameren Lauf.

## 22. Ergebnis

| Arm | Batch | Segmente | Token | Sample-Token/s |
| :--- | ---: | ---: | ---: | ---: |
| segmentiert (wide) | `32` | `3` | `151` | **`376,94`** |
| konform bisher (narrow) | `2` | `1` | `240` | `92,05` |

| Größe | Wert |
| :--- | ---: |
| Vorteil | **`4,095x`** |
| Ausgabe identisch | **ja** |
| maximale kontinuierliche Last | `5,215` s (Limit `6,0`) |
| Peak-Speicher | `4,22` GB |

Vorhergesagt waren `3,71x`; gemessen `4,095x`. Der Unterschied stammt daher, dass
Batch `2` mit `21,5` ms je Schritt etwas schlechter lief als die `19,36` ms, aus denen
die Vorhersage gebildet war.

**Nebenbefund mit Gewicht:** Batch `2` liefert `92,05` Sample-Token/s. Der beste
Einzelstrompfad liefert `92,18`. Die bisher einzige richtlinienkonforme Breite brachte
also **überhaupt keinen** Durchsatzvorteil gegenüber gar keinem Batching. Der
`3,71x`-Aufschlag war nicht der Preis für Konformität — er war der ganze Gewinn.

## 23. Abbruch bei Stop-Token

Der Loop prüft einmal je Segment, ob alle Sequenzen ein Stop-Token erzeugt haben, nicht
einmal je Schritt. Eine Prüfung je Schritt bräuchte eine Host-Synchronisation je
Schritt, und genau die zerstört das Pipelining, das dieser Loop ausnutzen soll. Die
Segmentgrenze zahlt ohnehin eine Synchronisation, dort ist die Prüfung gratis. Der
Preis ist ein Überlauf von höchstens einem Segment.

Im Lauf oben stoppte der wide-Arm nach `151` statt `240` Token. Die Rate wird auf
tatsächlich erzeugte Token normiert, damit ein Arm nicht dafür belohnt wird, dass er
zufällig weniger überlief.

## 24. Grenzen

Ein Modell, ein Prompt, greedy Sampling. Die Korrektheitsprüfung deckt `48` Token und
`3` Segmente ab, nicht beliebige Segmentierungen. Der Loop behandelt keine
unterschiedlich langen Prompts (kein Padding, keine Maske) und ersetzt
`batch_generate` **nicht**; er zeigt, dass die effiziente Breite unter der bestehenden
Sicherheitsrichtlinie erreichbar ist. Eine Integration in den Qualitätspfad ist
**nicht** erfolgt.

---

# Nachtrag: die Breitenkurve jenseits von 32, und wo sie sättigt

**Ergänzt:** 23. August 2026, achte Messrunde.
`tools/measure_segmented_decode.py --execute --sweep`, Bericht:
`experiments/segmented_decode/sweep.json`.

## 25. Warum das vorher nicht messbar war

Abschnitt 18 endete bei Batch `64` mit dem Vermerk, ein Lauf über `max_tokens=16` habe
das Continuous-Limit gerissen. Das war keine Aussage über die Hardware, sondern über
das Messverfahren: ein einzelner `batch_generate`-Aufruf ist nicht unterbrechbar. Der
segmentierte Loop hebt diese Grenze auf, also lässt sich die Kurve jetzt zu Ende
messen.

## 26. Ergebnis

`64` Token je Zeile, greedy, Segmentlänge je Batch aus einer eigenen Messung
abgeleitet:

| Batch | ms/Schritt | ms je Sample-Token | Sample-Token/s | Peak | vs. Batch 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | `73,36` | `9,13` | `109,54` | `3,12` GB | `1,00` |
| 16 | `74,13` | `4,61` | `216,87` | `3,66` GB | `1,98` |
| **32** | `73,81` | `2,30` | **`434,47`** | **`4,64` GB** | `3,97` |
| **48** | `133,33` | `2,78` | **`360,19`** | `5,69` GB | `3,29` |
| **64** | `130,40` | **`2,03`** | **`493,02`** | `6,55` GB | **`4,50`** |
| 96 | `191,82` | `2,08` | `480,12` | `7,13` GB | `4,38` |
| 128 | `248,82` | `2,09` | `478,80` | `7,98` GB | `4,37` |

Drei Dinge stehen darin.

**Das Plateau ist direkt sichtbar.** Batch `8`, `16` und `32` kosten alle rund `74` ms
je Schritt. Viermal so viel Arbeit zum selben Preis — das ist dieselbe Struktur, die
Abschnitt 11 im Forward-Pass fand, hier im Generierungsloop.

**Batch 48 bricht ein**, auf `360,19` gegen `434,47` bei Batch `32`. Die Policy hatte
`48` in `regression_widths`. Nach Batch `8` und dem Lauf aus Abschnitt 18 ist das die
**dritte unabhängige Bestätigung** derselben Vorhersage.

**Die Kurve sättigt bei 64.** Batch `128` liefert `478,80` gegen `493,02` bei Batch
`64` — nichts, bei `22 %` mehr Speicher. Ab `64` läuft das Gerät gegen rund
`2,0` ms je Sample-Token; mehr Breite kauft nichts mehr.

## 27. Endstand

Gegen den besten Einzelstrompfad (`92,18` tok/s):

| Betriebspunkt | Sample-Token/s | Faktor | Peak |
| :--- | ---: | ---: | ---: |
| Einzelstrom | `92,18` | `1,00x` | – |
| Batch 32 | `434,47` | `4,71x` | `4,64` GB |
| **Batch 64** | `493,02` | **`5,35x`** | `6,55` GB |
| Forward-Pass-Obergrenze (Breite 64) | – | `6,16x` | – |

**`87 %` der Obergrenze sind realisiert.** Die Lücke von `1,91x`, die Abschnitt 5
behauptete, existierte nie; sie war ein Messfehler (Abschnitt 14). Was übrig bleibt,
sind `13 %`, und die stecken in Sampling, Cache-Verwaltung und Segmentgrenzen.

**Empfehlung: Batch `32`, wenn Speicher zählt** — `88 %` des Spitzendurchsatzes bei
`71 %` des Speichers. **Batch `64`, wenn nicht.** Batch `48` in keinem Fall, Batch
`96` und `128` ohne Nutzen.

## 28. Ein dritter Kalibrierungsfehler

Die Segmentlänge wird aus einer Sonde von vier Decode-Schritten abgeleitet. Diese Sonde
lief unmittelbar nach dem Prefill und maß damit einmalige Kosten mit — Allokation und
Kernel-Aufbau für diese Form. Bei Batch `96` meldete sie `756` ms je Schritt, während
der tatsächliche Steady-State bei `199` ms lag: Faktor `3,8` zu hoch.

Das war konservativ, also nie unsicher, aber falsch berichtet und es zerteilte die
Generierung unnötig fein (`13` Segmente statt `3`). Mit zwei Aufwärmschritten vor der
Messung liegt die Sonde jetzt richtig: `73,8` ms bei Batch `32`, `248,8` bei Batch
`128`.

Damit sind es drei Kalibrierungsfehler in dieser Messreihe — Duty-Fenster, Prefill in
Positionen, und dieser. Alle drei hatten dieselbe Form: **ein Wert, der bei einer
Größenordnung stimmte und bei einer anderen still nicht mehr.** Alle drei sind jetzt
aus einer Messung abgeleitet statt gesetzt.

---

# Abschluss: Prefill, die bis hierhin nicht gemessene Hälfte

**Ergänzt:** 23. August 2026, elfte und letzte Messrunde.
`experiments/decode_width/measure_prefill.py`, Bericht:
`experiments/decode_width/prefill.json`.

## 29. Warum das fehlte

Alles bisherige misst Decode. Für einen Agenten mit langem Prompt und kurzer Antwort
ist aber der Prefill die Uhr, nicht der Decode — und Prefill ist eine andere Form:
eine lange Zeile statt vieler kurzer. Die für Decode abgeleitete Breiten-Policy sagt
darüber nichts.

## 30. Ergebnis

`2048` Positionen, in Blöcken gefüllt, nur Kernelzeit gezählt:

| Blockgröße | GPU-s | Positionen/s | ms je Position |
| ---: | ---: | ---: | ---: |
| 256 | `4,052` | `505,4` | `1,979` |
| 512 | `3,859` | `530,7` | `1,884` |
| **1024** | `3,768` | **`543,5`** | `1,840` |
| 2048 | `3,764` | `544,1` | `1,838` |

**Prefill und Batch-Decode laufen gegen dieselbe Schranke.** Prefill sättigt bei
`544` Positionen/s, Batch-64-Decode bei `493` — ein Verhältnis von `1,10`. Dieselbe
Rechnung bei derselben effektiven Breite trifft dieselbe Grenze; die Form der Eingabe
ändert daran fast nichts.

Zwei Folgerungen mit Praxiswert:

**Chunked Prefill ist fast gratis.** `512`er-Blöcke kosten `2,5 %` gegenüber
ungeteiltem Prefill. Die guard-konforme Zerlegung aus Abschnitt 21 zahlt also fast
nichts — anders als beim Decode, wo eine zu schmale Breite `3,7x` kostete.

**Der frühere Roofline-Wert lag unterhalb der Sättigung.** Dort wurden `422` tok/s
Prefill bei `369` Prompt-Token gemessen. Ab `1024` Positionen sind es `544`, also
`29 %` mehr. Der alte Wert ist nicht falsch, er beschreibt nur einen Betriebspunkt vor
dem Knie der Kurve.

## 31. Ein vierter Zeitmessfehler

Der erste Lauf berichtete `13,566` ms je Position. Die Messung umfasste die Wall-Clock
der ganzen Schleife und damit die verifizierten Guard-Pausen, die per Konstruktion
rund `5,7x` der Arbeit betragen. Nur Kernelzeit summiert ergibt `1,838` ms.

Damit sind es vier Messfehler in dieser Reihe: Prefill in Schrittkosten (Runde 4),
rollendes Duty-Fenster (Runde 7), Prefill in Positionen (Runde 7), Kalibriersonde ohne
Aufwärmung (Runde 8), und dieser. Alle waren zu meinen Ungunsten oder harmlos, keiner
hat eine Sicherheitsgrenze verletzt — der Guard hat in dieser Reihe fünf Läufe
fail-closed gestoppt, jedes Mal auf einen echten Fehler.

---

# Nachtrag: die 1B-Kurve zu Ende gemessen

**Ergänzt:** 23. August 2026.
`tools/measure_segmented_decode.py --execute --sweep --model 1b`,
Bericht: `experiments/segmented_decode/sweep_1b.json`.

## 32. Das kleinere Modell sättigt viel später

Der Batch-Sweep war bisher nur für das 4B gefahren, das bei `64` sättigte. Das 1B
läuft weiter:

| Batch | ms/Schritt | ms je Sample-Token | Sample-Token/s | Peak | vs. Batch 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | `15,05` | `1,941` | `515,3` | `1,07` GB | `1,00` |
| 16 | `16,05` | `1,005` | `995,5` | `1,32` GB | `1,93` |
| 32 | `16,00` | `0,501` | `1996,2` | `1,89` GB | `3,87` |
| **48** | `25,42` | `0,544` | **`1836,8`** | `2,21` GB | `3,57` |
| 64 | `25,84` | `0,402` | `2485,6` | `2,54` GB | `4,82` |
| 96 | `34,97` | `0,362` | `2764,9` | `2,49` GB | `5,37` |
| 128 | `44,05` | `0,343` | `2914,1` | `2,59` GB | `5,66` |
| 192 | `63,94` | `0,330` | `3029,9` | `2,91` GB | `5,88` |
| **256** | `82,37` | `0,323` | **`3093,0`** | `3,60` GB | **`6,00`** |

Batch `8`, `16` und `32` kosten alle rund `16` ms je Schritt — dasselbe Plateau wie
beim 4B, nur breiter. Und **Batch `48` bricht wieder ein**, auf `1836,8` gegen
`1996,2` bei Batch `32`. Nach Batch `8`, dem Lauf aus Abschnitt 18 und dem
4B-Sweep ist das die **vierte unabhängige Bestätigung** derselben Vorhersage aus der
Policy.

Von `192` auf `256` bringt nur noch `2 %`. Die Kurve ist damit ausgemessen.

## 33. Beide Modelle nebeneinander

Gegen den jeweiligen Steady-State-Einzelstrom:

| Modell | Einzelstrom | bester Batch | Sample-Token/s | Faktor | Peak |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 4B | `82,4` tok/s | `64` | `493,0` | `5,98x` | `6,55` GB |
| **1B** | `225,1` tok/s | `256` | **`3093,0`** | **`13,74x`** | `3,60` GB |

Das kleinere Modell gewinnt mehr als doppelt so viel aus dem Batching und braucht
dabei **weniger** Speicher als das größere. Der Grund steht im Gerätemodell: das 1B
ist bei Breite `1` zu `73 %` dispatch-gebunden, das 4B nur zu `48 %`, und Batching
verteilt genau diesen festen Anteil.

Für einen Dienst, der viele Anfragen gleichzeitig bedient, ist das 1B damit nicht
einfach das schwächere Modell, sondern das mit dem deutlich besseren Durchsatzprofil —
was die gemessene Genauigkeitslücke (`27,1 %` gegen `81,2 %` auf schweren Aufgaben)
allerdings nicht aufhebt.

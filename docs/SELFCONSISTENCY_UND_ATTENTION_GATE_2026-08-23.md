# Genauigkeit kaufen, und ein Gate für den Fused-Attention-Auftrag

**Stand:** 23. August 2026 · **Gerät:** Apple M1 Max, 32 GB · **Status:** explorativ,
`formal_claim=false`. Fortsetzung von
[`DECODE_WIDTH_BEFUND_2026-08-23.md`](DECODE_WIDTH_BEFUND_2026-08-23.md).

## 1. Frage

Das Breiten-Plateau macht `32` Token-Positionen etwa so teuer wie `8`. Batch-Serving
verwandelt das in Durchsatz für viele Nutzer. Diese Messung fragt, ob ein *einzelner*
Nutzer es stattdessen in **Genauigkeit** umsetzen kann: `k` unabhängige Samples je
Frage, Mehrheitsantwort.

Die Aufgaben werden erzeugt, nicht geladen: jede hat eine eindeutige ganzzahlige
Lösung und wird arithmetisch geprüft, nicht von einem Modell. Fester Seed, damit alle
Arme exakt dieselben Aufgaben sehen. Schwierigkeit kalibriert — eine Vier-Faktor-
Variante ließ 1B bei `17 %` mit einem Drittel abgeschnittener Antworten, eine
Ein-Schritt-Variante bei `80 %` ohne Spielraum.

Zwei Metriken, die zweite wird leicht vergessen: **Genauigkeit** (Mehrheitsantwort
gleich Wahrheit) und **Coverage** (überhaupt eine Antwort extrahierbar). Ein Modell,
das mitten im Satz ins Token-Limit läuft, liefert keine Antwort; das als falsche
Antwort zu zählen würde ein Versagen verstecken, das eine völlig andere Ursache hat.

## 2. Ergebnis: Self-Consistency ist kein freier Gewinn

Standardaufgaben, 12 Stück:

| Arm | Genauigkeit | Coverage | s/Aufgabe |
| :--- | ---: | ---: | ---: |
| 1B greedy | `66,7 %` | `0,75` | `1,769` |
| 1B Self-Consistency `k=8` | **`91,7 %`** | **`1,00`** | `4,897` |
| 4B greedy | **`100 %`** | `1,00` | **`1,358`** |

Schwere Aufgaben, 12 Stück, für 4B (Standardsatz war Deckeneffekt):

| Arm | Genauigkeit | Coverage | s/Aufgabe |
| :--- | ---: | ---: | ---: |
| 4B greedy | **`66,7 %`** | `0,917` | **`1,967`** |
| 4B Self-Consistency `k=8` | `50,0 %` | `1,00` | `12,057` |

Beim schwachen Modell hilft das Verfahren stark: `+25` Punkte, und die Coverage steigt
von `0,75` auf `1,00`, weil acht Versuche fast immer eine Antwortzeile produzieren.
Beim starken Modell **schadet** es: `66,7 %` auf `50,0 %` bei sechsfacher Zeit.
`mean_distinct_answers` war `1,667` — es gibt Streuung, aber die Mehrheit trifft nicht
zuverlässig. Sampling bei `temp=0,7` zerstört bei einem verlässlichen Top-1 mehr, als
die Mehrheitswahl repariert.

**Statistische Grenze:** `n=12`. Die Wilson-Intervalle von `8/12` und `6/12` überlappen
deutlich. Der 4B-Befund ist ein Hinweis, kein Beweis; der 1B-Befund (`8/12` gegen
`11/12`, plus die Coverage-Änderung) ist deutlich robuster.

## 3. Der überraschende Teil: größer schlägt öfter

`4B greedy` dominiert `1B + Self-Consistency` auf **jeder** Achse — auch bei der Zeit
(`1,358` gegen `4,897` s je Aufgabe). Der Grund ist hardwarespezifisch: der
4B-Forward-Pass kostet nur **`1,95x`** den 1B-Pass (`13,05` gegen `6,69` ms), nicht
`4x`. Bei Unified Memory ist Modellgröße billig.

Das kehrt die Rechenzentrums-Intuition um. Dort lohnt "kleines Modell, viele Samples",
weil Modellgewichte pro GPU knapp sind. Hier lohnt "größeres Modell, greedy". Wer auf
Apple Silicon Genauigkeit sucht, sollte zuerst das größere Modell nehmen und erst dann
über Test-Time-Compute nachdenken.

## 4. Gate für den Fused-Quantized-KV-Attention-Auftrag

Vor jeder Kernelarbeit wurden zwei Vorfragen geprüft.

**Existiert das schon?** `mx.fast.quantized_scaled_dot_product_attention` gibt es in
MLX `0.32.0` **nicht**. Aber `mlx_lm` implementiert
`quantized_scaled_dot_product_attention` bereits über `mx.quantized_matmul`
(`models/base.py:64`). Volle Dequantisierung des KV-Caches wird also **heute schon
vermieden**. Das Mindestziel des Auftrags ist damit weitgehend erreicht; neu wäre nur
die Fusion mit Online-Softmax, also gesparte Dispatches und ein nicht materialisierter
Score-Tensor.

**Wie groß ist der Hebel?** Attention ist der einzige Teil eines Decode-Schritts, dessen
Kosten mit der KV-Länge wachsen; FFN und Projektionen nicht. Die Steigung der
Schrittzeit gegen die Kontextlänge isoliert Attention daher ohne Profiler:

| Tkv | Schritt | f_attention |
| ---: | ---: | ---: |
| 0 | `13,284` ms | – |
| 1024 | `14,104` ms | `5,8 %` |
| 2048 | `14,449` ms | `8,1 %` |
| 4096 | `14,968` ms | `11,3 %` |
| 8192 | `15,135` ms | `12,2 %` |
| 16384 | `15,982` ms | `16,9 %` |

Amdahl auf das Primärziel `Tkv ≥ 8192` (`f = 0,169` bei 16K):

| S_attention | S_total |
| ---: | ---: |
| `1,2` | `1,029` |
| `2,0` | `1,092` |
| `∞` (Attention gratis) | **`1,203`** |

**Die Obergrenze des gesamten Programms liegt bei `1,20x`.** Zum Vergleich: das
gemessene Breiten-Plateau liegt bei `5,39x` und ist unausgeschöpft.

Ursachen, alle gemessen: `n_heads=8`, `n_kv_heads=4`, `head_dim=256`, und ein
Sliding-Window-Muster mit `sliding_window_pattern=6` — `29` der `34` Layer sehen nie
mehr als `1024` Token. Peak-Memory bei 16K Kontext: `4,64` GB. Das 50-%-Speicherziel
des Auftrags löst für dieses Modell kein vorhandenes Problem.

**Geltungsbereich.** Das falsifiziert den Auftrag nicht allgemein, sondern für dieses
Modell. Ein dichtes 7–9B-Modell mit voller Attention über `32768` Token — wie der
Auftrag es vorsieht — hätte ein erheblich höheres `f_attention`, und dort wären die
Ziele plausibel. Ein solches Modell liegt lokal nicht vor; ein Download wäre
freigabepflichtig und ist nicht erfolgt.

## 5. Reproduktion

```
.venv/bin/python tools/measure_self_consistency.py --execute --model 1b --samples 1 --problems 12
.venv/bin/python tools/measure_self_consistency.py --execute --model 1b --samples 8 --problems 12
.venv/bin/python tools/measure_self_consistency.py --execute --model 4b --samples 1 --problems 12
.venv/bin/python tools/measure_self_consistency.py --execute --model 4b --samples 1 --problems 12 --difficulty hard --chunk 1
.venv/bin/python tools/measure_self_consistency.py --execute --model 4b --samples 8 --problems 6 --offset 0 --difficulty hard --chunk 2
.venv/bin/python tools/measure_self_consistency.py --execute --model 4b --samples 8 --problems 6 --offset 6 --difficulty hard --chunk 2
.venv/bin/python experiments/attention_fraction/measure_f_attention.py
```

Der `--offset`-Schalter shardet einen Aufgabensatz über mehrere Prozesse. Das
GPU-Budget von `120` s je Prozess trägt ein großes Modell mit vielen Samples sonst
nicht, und Sharding hält alle Arme auf identischen Aufgaben, statt den teuren Armen
still den Satz zu kürzen.

## 6. Grenzen

Ein Gerät, zwei Modelle, eine Quantisierung, `n=12` je Arm, eine Aufgabenfamilie
(mehrschrittige Arithmetik mit ganzzahliger Lösung). Kein Anspruch auf andere
Aufgabentypen, andere Temperaturen oder andere Sampling-Verfahren. Alle Läufe am
Netzteil unter `BudgetGuard`; kein Download, keine Installation.

---

# Nachtrag: das Plateau in Genauigkeit umgesetzt, mit ausreichender Teststärke

**Ergänzt:** 23. August 2026, fünfte Messrunde. **Dieser Abschnitt ersetzt die
Zahlen aus Abschnitt 2**, die bei `n=12` gemessen wurden und dafür zu klein waren.

## 7. Was an Abschnitt 2 nicht trug

Zwei Mängel, beide erst beim Nachrechnen sichtbar:

**Teststärke.** Bei `n=12` lauten die Wilson-Intervalle für `8/12` und `6/12`
`[0,391; 0,862]` und `[0,254; 0,746]`. Sie überlappen fast vollständig. Für einen
Unterschied von `17` Punkten bei `80 %` Teststärke wären rund `n=133` je Arm nötig.
Die dort berichtete Richtung war nie signifikant, und das hätte deutlicher dastehen
müssen als es tat.

**Temperatur als Confound.** Der 4B-Arm lief bei `temp=0,7`. Bei einem Modell mit
verlässlichem Top-1 ist das heiß genug, um den Befund allein zu erklären. Nachgemessen
bei `temp=0,3`: `5/12` = `41,7 %`, KI `[0,193; 0,680]` — **schlechter**, nicht besser.
Die Confound-Hypothese ist damit widerlegt; Self-Consistency hilft dem 4B auf dieser
Aufgabenfamilie bei keiner der beiden Temperaturen. Signifikant ist auch das nicht.

## 8. Die Policy auf Test-Time-Compute angewandt

Die Breiten-Policy aus dem Nachbardokument ändert, wie viele Stimmen man sich leisten
kann. Für das 1B-Modell:

| chunk | ms je Schritt | ms je Sample-Schritt |
| ---: | ---: | ---: |
| 8 | `14,73` | `1,84` |
| **32** | `17,02` | **`0,53`** |

`k=32` bei chunk `32` kostet je Sample **`3,5x` weniger** als `k=8` bei chunk `8`.
Vier Mal so viele Stimmen für gut ein Sechstel des Preises je Stimme. Ein
Sample-Budget an der Klippe auszugeben ist die teuerste Art, es auszugeben.

## 9. Ergebnis mit ausreichender Teststärke

Beide Arme: Gemma 3 1B, Standardaufgaben, identischer Seed, `max_tokens=240`.

| Arm | korrekt | Genauigkeit | 95%-KI | Coverage | s/Aufgabe |
| :--- | ---: | ---: | :--- | ---: | ---: |
| greedy | `31/48` | `64,6 %` | `[0,504; 0,766]` | `0,81` | `1,50` |
| **Self-Consistency `k=32`** | **`31/32`** | **`96,9 %`** | **`[0,843; 0,994]`** | **`1,00`** | `5,28` |

Zwei-Stichproben-z-Test: **`z = 3,388`, `p = 0,0007`** (zweiseitig). Die Intervalle
trennen sauber. Effekt **`+32,3` Punkte** bei **`3,51x`** Zeit.

Die Coverage steigt von `0,81` auf `1,00`: bei `32` Versuchen produziert praktisch
immer mindestens einer eine vollständige Antwortzeile, während ein einzelner greedy
Lauf in `19 %` der Fälle ins Token-Limit läuft. Ein Teil des Gewinns ist also
repariertes Abschneiden, nicht besseres Rechnen — beide zählen für den Nutzer, aber
sie haben verschiedene Ursachen und wären getrennt zu behandeln.

**Die relevante Größe ist das Verhältnis:** `32` Stimmen für `3,51x` Zeit statt `32x`.
Das Plateau macht Test-Time-Compute um **`9,1x`** billiger als die naive Rechnung.
Ohne diese Messung wäre `k=32` auf diesem Gerät als unbezahlbar eingestuft worden.

## 10. Was das an Abschnitt 3 **nicht** ändert

`4B` greedy erreichte auf denselben Standardaufgaben `100 %` bei `1,358` s. Das bleibt
besser als `1B` mit `k=32` (`96,9 %` bei `5,28` s) — auf beiden Achsen. Die Empfehlung
"erst das größere Modell, dann Test-Time-Compute" steht.

Der Fall, in dem dieses Ergebnis zählt, ist ein anderer: wenn das größere Modell nicht
in den Speicher passt oder nicht verfügbar ist, holt Self-Consistency das `1B` von
`64,6 %` auf `96,9 %` — und das Plateau macht diesen Weg bezahlbar.

## 11. Konflikt zwischen Sicherheitsrichtlinie und optimaler Breite

Das `6`-s-Continuous-Limit des `BudgetGuard` und die optimale Breite widersprechen sich
für realistische Antwortlängen:

| chunk | ms/Schritt | ms/Sample-Schritt | max. Token in `6` s |
| ---: | ---: | ---: | ---: |
| **32** | `83,46` | **`2,61`** | **`71`** |
| 4 | `30,82` | `7,71` | `194` |
| 2 | `19,36` | `9,68` | `309` |

Beim 4B erlaubt die effizienteste Breite nur `71` Token je Lastblock; die schweren
Aufgaben brauchen `288`. Wer die Länge will, muss auf chunk `2` und zahlt **`3,71x`**.
Genau daran scheiterten in dieser Runde zwei Läufe fail-closed — der Guard arbeitete
korrekt.

Das ist **kein Hardwareproblem**. Drei Auswege, in aufsteigender Eingriffstiefe:

1. kürzere Antworten akzeptieren — kostet Genauigkeit;
2. das Continuous-Limit neu verhandeln — eine Richtlinienentscheidung, keine technische;
3. **die Generierung segmentieren**: `≤64` Schritte, Guard-Pause, mit demselben
   KV-Cache weiter. Das nutzt die optimale Breite **und** hält das Limit ein.

Weg 3 ist mit einem eigenen Decode-Loop machbar — der Cache überlebt die Pause — und
mit `batch_generate` nicht, weil dessen Aufruf die Generierung nicht unterbrechbar
macht. Er ist **nicht implementiert**.

## 12. Kleiner Mangel im Aufgabengenerator

Bei `n=48` sind `47` der Aufgaben verschieden; eine kommt doppelt vor. Das senkt die
effektive Stichprobe geringfügig und ist bei `p=0,0007` unkritisch, wäre aber für
größere Läufe zu beheben.

---

# Nachtrag: Self-Consistency ist keine Gerade, sondern eine U-Kurve

**Ergänzt:** 23. August 2026, siebte Messrunde. **Dieser Abschnitt hebt die
Schlussfolgerung aus Abschnitt 2 und 7 auf** — nicht die Messwerte, sondern ihre
Deutung.

## 13. Was ich falsch geschlossen hatte

Abschnitt 2 berichtete, Self-Consistency **schade** dem starken Modell: `4B` greedy
`66,7 %`, mit `k=8` nur `50,0 %`. Abschnitt 7 prüfte die naheliegende Erklärung
(Temperatur zu hoch) und verwarf sie. Beide Male blieb die Deutung "das Verfahren
hilft starken Modellen nicht".

Diese Deutung war falsch. Zwei getrennte Fehler:

**Die Basislinie war zu schwach gemessen.** `4B` greedy auf den schweren Aufgaben bei
`n=48` liegt bei `39/48` = **`81,2 %`**, nicht bei `66,7 %`. Der `n=12`-Wert war ein
Ausreißer nach unten. Jeder Vergleich gegen ihn überschätzte den Effekt.

**Die Stimmenzahl war der Faktor, nicht die Temperatur.** Mit `k=32` statt `k=8`
erreicht dasselbe Modell auf denselben Aufgaben `11/12` = `91,7 %`.

## 14. Korrigierte Zahlen

| Arm | korrekt | Genauigkeit | 95%-KI | s/Aufgabe |
| :--- | ---: | ---: | :--- | ---: |
| greedy (`n=48`) | `39/48` | **`81,2 %`** | `[0,681; 0,898]` | `1,97` |
| `k=8`, `t=0,7` (`n=12`) | `6/12` | `50,0 %` | `[0,254; 0,746]` | `12,06` |
| `k=32` segmentiert (`n=16`) | `14/16` | `87,5 %` | `[0,640; 0,965]` | `18,40` |

| Vergleich | Effekt | z | p |
| :--- | ---: | ---: | ---: |
| greedy vs. `k=8` | `−31,2` Punkte | `−2,236` | **`0,025`** |
| `k=8` vs. `k=32` | `+41,7` Punkte | `+2,245` | **`0,025`** |
| greedy vs. `k=32` | `+6,2` Punkte | `+0,574` | `0,566` |

Exakter Binomialtest gegen die starke Basislinie:
`P(X ≤ 6 | n=12, p=0,8125) = 0,0142`. Der `k=8`-Einbruch ist echt.

## 15. Die U-Kurve

**Wenige Stimmen sind schlechter als gar keine.** Greedy nimmt den zuverlässigsten
einzelnen Pfad. `k=8` bei `t=0,7` wirft diese Zuverlässigkeit weg und ersetzt sie durch
eine Mehrheit, die über acht Stimmen und einen breiten Antwortraum oft keine ist.
`k=32` stellt sie wieder her, weil die Mehrheit dann trägt.

Das ist die praktische Regel, und sie ist nicht die, die man erwartet:
**entweder viele Stimmen oder gar keine — dazwischen wird es schlechter.** Ein halbes
Sample-Budget ist schlimmer als keines.

**`k=32` schlägt greedy nicht messbar.** Bei `n=12` waren es `+10,4` Punkte
(`p = 0,387`), bei `n=16` nur noch `+6,2` (`p = 0,566`). Ein Vorsprung, der mit
wachsender Stichprobe schrumpft, ist der übliche Verlauf eines Effekts, den es nicht
gibt. Als Nichtbefund berichtet, nicht als offene Chance.

## 16. Warum das ohne die Breiten-Arbeit nicht messbar gewesen wäre

`k=32` bei voller Antwortlänge (`288` Token) kostet auf dem alten, richtlinienkonformen
Pfad (chunk `2`) rund `90` s je Aufgabe. Über den segmentierten Loop sind es `19,73` s
— **`4,6x` weniger**. Der `k=8`-Arm war nicht deshalb gewählt worden, weil `8` eine
gute Zahl ist, sondern weil das Continuous-Limit nichts Größeres zuließ.

Die Kette ist damit geschlossen: Breiten-Plateau gemessen → Policy abgeleitet →
segmentierter Loop gebaut und auf Byteidentität geprüft → `k=32` bezahlbar → der
`k=8`-Befund als Artefakt der Sample-Zahl entlarvt. Ohne die Hardwaremessung wäre die
falsche Deutung stehen geblieben.

## 17. Kostenwahrheit

`k=32` kostet `18,40` s gegen `1,97` s je Aufgabe, also **`9,3x`**. Der belegte Gewinn
gegenüber greedy ist keiner. Der belegte Gewinn gegenüber `k=8` ist groß, aber
`k=8` sollte man ohnehin nicht fahren. Wer heute auf diesem Gerät Genauigkeit will,
nimmt weiterhin greedy auf dem größeren Modell; `k=32` ist ein Kandidat, dessen Nachweis
noch aussteht.

## 18. Zwei Pacing-Fehler, die der Guard gefunden hat

Beide Fehler waren meine, beide wurden fail-closed abgefangen, und beide lohnen die
Notiz, weil sie dieselbe Form haben: **eine Konstante, die bei einer Größe sicher war
und bei einer anderen still nicht mehr.**

**Rollendes Fenster ist nicht gleich Einzel-Event.** Die Pause bemaß sich am gerade
beendeten Lastblock (`4x` Arbeit für `20 %` Auslastung). Der Guard prüft aber eine
gleitende `60`-s-Summe gegen `25 %`. Bei ungleich langen Blöcken ist das nicht
dasselbe: ein kurzer Block verdient wenig Pause, und der lange danach fällt in ein noch
volles Fenster. Gemessener Abbruch: Blöcke von `3,88`, `3,98`, `2,98` und `5,31` s
erfüllten jeder für sich das Ziel, summierten sich im Fenster aber auf `16,0` s gegen
`15` s Grenze. Ziel auf `0,15` gesenkt; die Ereignisfolge steht als Testfall im
Self-Check, damit es nicht zurückwandert.

**Prefill wird in Positionen bezahlt, nicht in Token.** Der Prefill teilte nur die
Sequenzachse in Blöcke von `256`. Bei Batch `32` ist ein Prompt von `98` Token aber
`3136` Positionen in **einem** Block — und riss die `6`-s-Grenze. Die Blocklänge wird
jetzt aus Batch und gemessenen Positionskosten abgeleitet: `50` Token bei Batch `32`,
`803` bei Batch `2`. Dieselbe Disziplin wie bei der Segmentlänge, aus demselben Grund.

Der `BudgetGuard` hat in dieser Messreihe insgesamt fünf Läufe gestoppt, jedes Mal
korrekt und jedes Mal auf einen echten Fehler in meinem Code. Er ist nicht die Bremse
gewesen, als die er sich anfühlte.

---

# Abschluss: die vollständige Qualitätsmatrix

**Ergänzt:** 23. August 2026, neunte Messrunde. Damit ist die Frage "werden die
Antworten genauer" für dieses Gerät und diese Aufgabenfamilie beantwortet.

## 19. Alle vier Zellen

Identischer Seed je Aufgabenfamilie, greedy für `k=1`, `t=0,7` für `k=32`,
`max_tokens=288`, `k=32` über den segmentierten Loop bei chunk `32`.

| Aufgaben | Modell | greedy | `k=32` | Effekt | p |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Standard | 1B | `64,6 %` (n=48) | **`96,9 %`** (n=32) | **`+32,3`** | **`0,0007`** |
| schwer | 1B | `27,1 %` (n=48) | **`65,6 %`** (n=32) | **`+38,5`** | **`0,00063`** |
| schwer | 4B | `81,2 %` (n=48) | `87,5 %` (n=16) | `+6,2` | `0,566` |
| Standard | 4B | `100 %` (n=12) | – | Deckeneffekt | – |

Das Muster ist eindeutig und in beiden 1B-Zellen unter `p < 0,001`:
**Self-Consistency hilft dem schwachen Modell stark und dem starken Modell nicht
messbar.** Test-Time-Compute kauft Genauigkeit dort, wo Kapazität fehlt, und nichts,
wo sie vorhanden ist.

Die Coverage erzählt einen Teil davon getrennt: `0,688` auf `1,000` bei den schweren
1B-Aufgaben. Fast ein Drittel der greedy-Läufe lief ins Token-Limit, ohne je eine
Antwortzeile zu schreiben. Bei `32` Versuchen schafft es praktisch immer einer. Das ist
repariertes Abschneiden, nicht besseres Rechnen, und wäre mit einem höheren Token-Budget
teilweise auch ohne Sampling zu haben.

## 20. Der Vergleich, der die praktische Antwort gibt

Auf denselben schweren Aufgaben:

| Weg | Genauigkeit | s/Aufgabe |
| :--- | ---: | ---: |
| 1B greedy | `27,1 %` | `1,50` |
| 1B mit `32` Stimmen | `65,6 %` | `5,03` |
| **4B greedy** | **`81,2 %`** | **`1,97`** |
| 4B mit `32` Stimmen | `87,5 %` (n. s.) | `18,40` |

**Das größere Modell greedy schlägt das kleinere mit 32-fachem Sample-Budget auf
beiden Achsen** — genauer *und* `2,6x` schneller.

Der Grund ist hardwarespezifisch und wurde separat gemessen: der 4B-Forward-Pass
kostet bei Breite `1` nur `1,95x` den 1B-Pass (`13,05` gegen `6,69` ms). Unified Memory
macht Modellkapazität billig; Sampling bleibt teuer, weil es Zeit kostet, die sich
nicht teilen lässt.

Im Rechenzentrum gilt die umgekehrte Rechnung: dort ist Modellkapazität pro
Beschleuniger knapp und Batching billig, also lohnt "kleines Modell, viele Samples".
Auf einem M1 Max lohnt **"größeres Modell, greedy"**.

Einschränkung, die dabei stehen bleiben muss: bei Breite `32` kostet das 4B `4,50x`
statt `1,95x` (Abschnitt 11 des Nachbardokuments). Die Regel gilt für Einzelanfragen.
Wer viele Anfragen gleichzeitig bedient, rechnet neu.

## 21. Wann das 1B-Ergebnis trotzdem zählt

`+38,5` Punkte sind kein kleiner Effekt, und er ist zweifach unter `p < 0,001`
abgesichert. Er zählt, wenn das größere Modell nicht zur Verfügung steht: zu wenig
Speicher, ein anderes Gerät, oder ein Modell, zu dem es keine größere Variante gibt.
Dann hebt `k=32` das kleine Modell von `27,1 %` auf `65,6 %`, und das Breiten-Plateau
macht diese `32` Stimmen für `3,35x` Zeit statt `32x` verfügbar.

Was **nicht** gilt: `k=8`. Auf dem 4B war das signifikant schlechter als greedy
(Abschnitt 14). Wenige Stimmen sind schlechter als keine.

## 22. Grenzen dieser Matrix

Eine Aufgabenfamilie — mehrschrittige Arithmetik mit eindeutiger ganzzahliger Lösung,
maschinell erzeugt und maschinell geprüft. Das ist genau die Sorte Aufgabe, bei der
Mehrheitswahl gut funktioniert, weil es eine richtige Antwort gibt und Fehler streuen.
Auf offene Aufgaben ohne prüfbare Lösung überträgt sich davon nichts, und für Aufgaben,
bei denen alle Stimmen denselben systematischen Fehler machen, ebenfalls nicht.
Ein Gerät, zwei Modelle, eine Quantisierung, eine Temperatur für alle
Mehrstimmen-Arme.

---

# Nachtrag: den eigenen Hauptbefund auf einen Confound geprüft

**Ergänzt:** 23. August 2026, zehnte Messrunde.

## 23. Der Verdacht

Abschnitt 19 berichtete `+38,5` Punkte für `k=32` auf den schweren 1B-Aufgaben. Im
selben Abschnitt stand aber auch, dass die Coverage des greedy-Arms nur `0,688` betrug:
`15` von `48` Läufen erzeugten nie eine Antwortzeile und zählten als falsch, während
der `k=32`-Arm bei `32` Versuchen praktisch immer eine bekam.

Damit ist offen, ob der Effekt **Mehrheitswahl** ist oder bloß **Wiederholung, bis der
Lauf nicht abgeschnitten wird**. Im zweiten Fall wäre ein größeres Token-Budget der
billigere Weg zum selben Ergebnis, und `k=32` hätte keinen eigenen Beitrag.

## 24. Kontrollarm

Derselbe greedy-Arm, Token-Budget von `288` auf `640` mehr als verdoppelt:

| Arm | gesamt | beantwortet | bedingt richtig | Coverage | s/Aufgabe |
| :--- | ---: | ---: | ---: | ---: | ---: |
| greedy `@288` | `27,1 %` | `33/48` | `39,4 %` | `0,688` | `1,50` |
| greedy `@640` | `29,2 %` | `34/48` | `41,2 %` | `0,708` | `3,21` |
| `k=32` `@288` | `65,6 %` | `32/32` | `65,6 %` | `1,000` | `5,03` |

Das doppelte Budget bringt `+2,1` Punkte bei `p = 0,820` — nichts. Gegen den
großzügigen greedy-Arm bleiben **`+36,5` Punkte, `p = 0,00128`**. Der Confound ist
widerlegt.

## 25. Warum das Budget nicht hilft

Die Coverage bleibt bei `0,708`, **obwohl** mehr als doppelt so viele Token zur
Verfügung stehen. Das Modell läuft also nicht aus dem Budget — es **degeneriert**.
Genau das war im allerersten Probe-Lauf dieser Messreihe zu sehen: Ausgaben, die in
`<end_of_turn><end_of_turn>` kippen und dort bleiben. Greedy-Decoding findet aus dieser
Schleife nicht heraus, gleich wie viel Platz es bekommt. Sampling bei `t=0,7` bricht sie
auf.

Das ist eine bessere Erklärung als die ursprüngliche und sie war nur durch den
Kontrollarm zu bekommen.

## 26. Zerlegung des Effekts

| Anteil | Mechanismus | Punkte |
| :--- | :--- | ---: |
| 1 | Coverage `0,688 → 1,000`, bedingte Quote konstant gehalten | `+12,3` |
| 2 | bedingte Quote `39,4 % → 65,6 %` | `+26,2` |
| | **Summe** | **`+38,5`** |

Rund **ein Drittel** des Gewinns ist Ausbruch aus der Degeneration, **zwei Drittel**
sind echte Mehrheitswahl unter gültigen Antworten. Beides sind reale Mechanismen; keiner
davon ist ein Messartefakt.

Die Kontrolle dazu: das größere Token-Budget verschiebt die bedingte Quote kaum
(`39,4 %` auf `41,2 %`). Wäre der Gewinn Budget gewesen, hätte sich genau diese Zahl
bewegen müssen.

## 27. Die Vorhersage aus der Zerlegung — und ihre Widerlegung

Die Zerlegung legte nahe, der billigere Teil des Gewinns sei getrennt zu haben: wenn
Greedy in einer Schleife hängen bleibt und Sampling sie aufbricht, müsste **ein
einzelner** Lauf bei `t > 0` den `+12,3`-Anteil zum Preis eines Laufs holen.

Gemessen, `n=48`, dieselben Aufgaben:

| Arm | gesamt | Coverage | bedingt richtig |
| :--- | ---: | ---: | ---: |
| greedy `@288` | `27,1 %` | `0,688` | `39,4 %` |
| **ein Sample, `t=0,7`** | **`27,1 %`** | `0,646` | `41,9 %` |
| `k=32`, `t=0,7` | `65,6 %` | `1,000` | `65,6 %` |

**Die Vorhersage ist falsch.** Ein einzelnes Sample bei `t=0,7` ist von Greedy nicht zu
unterscheiden — gleiche Trefferquote, sogar leicht schlechtere Coverage.

## 28. Das bessere Modell: Coverage ist Wiederholungsstatistik

Sampling bricht die Degeneration nicht auf. **Jedes Sample degeneriert unabhängig mit
rund `33 %`**, und `k=32` erreicht Coverage `1,000` schlicht deshalb, weil mindestens
einer von `32` durchkommt. Das Modell passt ohne freien Parameter:

| `k` | erwartete Coverage `1 − 0,333^k` | gemessen |
| ---: | ---: | ---: |
| 1 | `0,6667` | `0,646` / `0,688` |
| 4 | `0,9877` | – |
| 8 | `0,9998` | – |
| 32 | `1,0000` | `1,000` |

Damit haben die beiden Anteile des Gewinns **völlig verschiedene Sample-Bedürfnisse**:

| Anteil | Punkte | gesättigt ab |
| :--- | ---: | ---: |
| Coverage (Wiederholung) | `+12,3` | `k ≈ 4` |
| Mehrheitswahl | `+26,2` | `k ≈ 32` |

Das erklärt die U-Kurve aus Abschnitt 15 nachträglich: bei `k=8` ist die Coverage
längst repariert, aber die Mehrheit trägt noch nicht — und der wahrscheinlichste Pfad,
den Greedy nimmt, ist bereits aufgegeben. Beides zusammen ergibt genau das Tal.

Die Vorhersage war ein Fehlschlag, und ein nützlicher: ohne den Kontrollarm wäre die
falsche Empfehlung "einfach mit Temperatur dekodieren" stehen geblieben.

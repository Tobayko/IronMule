# Ergebnisse

> **Das wichtigste Ergebnis zuerst:** Auf diesem Gerät ist ein *ungepaarter*
> Performancevergleich nahezu wertlos — die Streuung zwischen Läufen übertrifft
> die meisten realen Effekte. Gepaart gemessen liegt die Nachweisgrenze bei
> `2,2 %` statt `33 %`. Alle Werkzeuge hier messen deshalb gepaart und gegen eine
> vorab eingefrorene Schwelle.


Kompakte Übersicht aller belastbaren Befunde. Die vollständige Herleitung samt
Fehlversuchen steht in [`ARBEITSJOURNAL.md`](ARBEITSJOURNAL.md); dieses Dokument
ist der Einstieg.

**Gerät:** Apple M1 Max, 32 GB Unified Memory, 32-Core GPU, macOS.
**Stand:** 21. August 2026. Alle Zahlen sind auf diesem einen Gerät gemessen.

---

## Der wichtigste Befund: ungepaart messen ist hier wertlos

Auf dieser Hardware streuen wiederholte Läufe derselben unveränderten Operation so
stark, dass ein ungepaarter Vergleich die meisten realen Effekte nicht auflösen kann.

| Messart | Variationskoeffizient | Nachweisgrenze bei 3 gegen 3 Läufen |
| --- | ---: | ---: |
| ungepaart, Lauf-Mediane | `20,5 %` | rund `33 %` |
| **gepaart, Blockratios** | **`1,32 %`** | **rund `2,2 %`** |

Faktor `15` Unterschied — aus **denselben** Messdaten. Der Grund: Beide Arme eines
Blocks erleben denselben Störuntergrund, der sich im Quotienten herauskürzt.
Ungepaart bleibt er vollständig stehen.

**Was das praktisch heißt:** `mx.compile` erschien ungepaart mit `−27,6 %` als
klarer Gewinn. Gepaart gemessen: `R = 1,0019`, Konfidenzintervall
`[0,9990, 1,0047]` — **kein Effekt**. Der gesamte scheinbare Gewinn war Rauschen.
Wer auf diesem Gerät MLX-Performance ungepaart misst, misst mit hoher
Wahrscheinlichkeit Zufall.

Alle Werkzeuge in diesem Projekt vergleichen deshalb beide Arme innerhalb desselben
Blocks und verlangen, dass ein Effekt eine **vor** dem Lauf festgelegte Schwelle
überschreitet.

---

## Bestätigte Optimierung: Dispatch-Batching

`N` Matmuls mit **einer** Synchronisation statt `N` einzelnen. Identische
Arithmetik, bytegleiches Ergebnis, nur der Ausführungsplan unterscheidet sich.

| `N` | Effekt | ms je Matmul |
| ---: | ---: | ---: |
| 2 | `−7,9 %` | `2,362` |
| **4** | **`−17,4 %`** | **`1,939`** |
| 8 | `−15,6 %` | `1,974` |
| 16 | `−11,9 %` | `1,966` |

Hauptlauf bei `N=8`: `R = 0,8531`, `95%-KI [0,8263, 0,8777]`, fünf Replikate,
Correctness bytegleich. Das Optimum liegt bei `N ≈ 4`–`8`; darüber ist der
Synchronisations-Overhead amortisiert.

**Reichweite, ehrlich:** Das ist keine Kernel-Optimierung — der Matmul-Kernel
bleibt unverändert. Es entfernt vermeidbare Synchronisation und gilt für
**unabhängige** Operationen. Die serielle Baseline ist ein realer Anti-Pattern
(jedes `mx.eval` in einer Schleife erzeugt sie), den erfahrener MLX-Code ohnehin
vermeidet.

Reproduzieren: `python tools/friday.py dispatch --execute --n 8`

---

## Der Loop findet Optimierungen selbst

Ein geschlossener Mess-Entscheidungs-Kreis: explorieren, um den Überlebenden herum
verfeinern, den eigenen Sieger unabhängig bestätigen. Vier Läufe:

| Lauf | gewählt | Effekt | Verdikt |
| ---: | :---: | ---: | --- |
| 1 | `N=8` | `−13,60 %` | bestätigt |
| 2 | `N=6` | `−11,13 %` | bestätigt |
| 3 | `N=6` | `−14,11 %` | bestätigt |
| 4 | `N=16` | `−11,08 %` | bestätigt |

`N=6` und `N=7` kamen in der manuellen Suche nicht vor; der Loop hat sie selbst
vorgeschlagen und einen davon bestätigt.

**Dass die gewählte Batchgröße zwischen Läufen wechselt, ist kein Mangel.** Das
Optimum ist ein breites Plateau: `N = 4` bis `16` liegen alle im Bereich
`−11 %` bis `−17 %`, und der Loop landet je nach Rauschen an unterschiedlichen
Stellen darauf. Stabil ist der *Effekt*, nicht der genaue Punkt. Was in allen vier
Läufen gleich blieb: ein bestätigter Gewinn in dieser Größenordnung.

**Eine Fallgrube, die dokumentiert bleiben soll:** Zunächst bestätigte der Loop nur
in `1` von `3` Läufen. Ursache war der *Winner's Curse* — der Beste aus mehreren
verrauschten Kandidaten ist konstruktionsbedingt zu optimistisch. Rangfolge nach
dem Punktschätzer wählte den glücklichsten Ausreißer (`0,750`, `0,741`), der bei
Nachmessung auf `0,87`–`0,96` regressierte. Die Lösung war, nach der
**Konfidenzobergrenze** zu ranken statt nach dem Punktschätzer: nicht „was sah
einmal am besten aus", sondern „was ist zuverlässig gut". Danach `4` von `4`.

Reproduzieren: `python tools/friday.py loop --execute`

---

## H2: ein lokales Modell schlägt die Pläne vor

`gemma-3-4b-it-4bit` erhält die bisherigen Messungen und die gemessenen
Gerätefakten und schlägt ungetestete Kandidaten vor. Über drei Runden sieht es die
Ergebnisse seiner eigenen Vorschläge.

| Runde | Antwort | Ergebnis |
| ---: | --- | --- |
| 1 | `[3, 10, 16]` | `N=3` verworfen, `N=10` und `N=16` bestehen |
| 2 | `[5, 12, 13]` | alle drei bestehen |
| 3 | `[7, 14, 15]` | alle drei bestehen |

Neun Werte vorgeschlagen, alle gültig und ungetestet. Bestätigt `N=13` mit
`−11,53 %`, `95%-KI [0,8552, 0,8957]`.

**Das Modell schlägt Parameter vor, niemals Code.** Modellgenerierten Code auf der
GPU auszuführen ist ein eigenes Sicherheitsproblem und nicht Teil dieses Werkzeugs.
Prosa, Shell-Fragmente, Floats, Booleans und Werte außerhalb `2..16` führen
sämtlich zu null ausgeführten Kandidaten — abgesichert durch `21` Tests.

**Der Harness bleibt streng:** `N=3` wurde verworfen, weil seine
Konfidenzobergrenze `0,954` die Schwelle `0,95` verfehlte. Das Modell schlägt vor,
es entscheidet nicht.

Reproduzieren: `python tools/friday.py model-loop --execute`

---

## H2 vollständig: das Modell schreibt den Plan selbst

`codegen` lässt `gemma-3-4b-it-4bit` den Ausführungsplan als Python schreiben.
Drei Schutzschichten stehen zwischen generiertem String und berichtetem Ergebnis:

1. **AST-Allowlist** — genau eine Funktion `plan(mx, a, operands)`, nur bekannte
   Namen, Attributzugriff nur als `mx.<Operation>` aus einer Zwölferliste plus
   `append`/`extend` auf Akkumulatorlisten. Keine Importe, Dunder, String-Literale
   oder Lambdas. `29` adversariale Tests.
2. **Prozessisolation** — frischer Subprozess mit Wall-Timeout, CPU-Zeit-Grenze,
   bereinigter Umgebung und MLX-Speicherlimit.
3. **Correctness** — ein Ergebnis je Operand, jedes bytegleich zur Referenz.

Ergebnis: fünf Pläne geschrieben, fünf gemessen, drei über der Schwelle.
Bestätigt mit `R = 0,8838`, **`−11,62 %`**, `95%-KI [0,8676, 0,8975]`,
Replikate `0,8742 / 0,8970 / 0,8838`.

Der vom Modell geschriebene Gewinnerplan:

```python
def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        out.append(x)
    mx.eval(out)
    mx.synchronize()
    return out
```

**Zwei Anläufe scheiterten, beide an mir.** Zuerst schrieb das Modell viermal die
Baseline ab, weil der Prompt sie zu prominent zeigte — unfreiwillig ein A/A-Test,
der bestätigte, dass der Harness sauber misst. Dann blockierte mein Validator
`out.append(x)` und damit genau die gesuchte Optimierung. Beides korrigiert; die
Erweiterung ist eng gefasst und durch fünf Tests begrenzt.

**Der Harness blieb streng:** zwei Pläne, die `mx.synchronize` aus der Schleife
zogen aber `mx.eval` darin beließen, wurden verworfen (`0,982` und `0,989`).

Reproduzieren: `python tools/friday.py codegen --execute`

---

## Cooldown: die erste Operation nach einer Pause ist langsamer

| Pause | erste Operation | Exzess in Sample-Äquivalenten |
| ---: | ---: | ---: |
| `0 s` | `0,94x` | `0,00` |
| `0,25 s` | `1,89x` | `2,42` |
| `2 s` | `3,67x` | `4,09` |
| `20 s` | `4,12x` | `5,12` |

Monoton, Sättigung bei rund `4x` ab etwa `2 s`, und bei null Pause exakt null
Exzess. **Ursache überwiegend GPU-Taktung:** Eine beschäftigt gehaltene GPU
halbiert den Effekt (`4,02` gegen `2,53`, `95%-KI [0,311, 0,762]`). Der
MLX-Allocator scheidet aus — sein Cache bleibt über die Pause konstant.

**Vorwärmen lohnt sich trotzdem nicht.** Sieben Dosierungen gemessen, alle netto
negativ; die beste spart `4,9 ms` und kostet `14,4 ms`. Auch bei echter
Modell-Inferenz widerlegt (zwei Varianten, beide Konfidenzintervalle enthalten
`1,0`).

**Praktische Konsequenz:** Wer nach einer Pause misst, ohne den Anlauf zu
verwerfen, verzerrt ein 80-Sample-Mittel um bis zu `5,8 %` — mehr als die
Nachweisschwelle von `5 %`.

Reproduzieren: `python tools/friday.py cooldown --execute`

---

## Fusion über ein unverändertes Modell — verworfen

Ein `mx.compile`-Wrapper um den Forward-Pass zeigte zunächst starke Werte:

| Modell | Regime | Effekt | Correctness |
| --- | --- | ---: | --- |
| 1B | prefill | `−8,9 %` | bytegleich |
| 1B | Generierung | `−12,4 %` | bytegleich |
| 4B | Generierung | `−15,0 %` | bytegleich |

**Diese Zahlen sind sauber gemessen und praktisch trotzdem wertlos.** Die
Nachprüfung an der echten Generierungsschleife ergab `−0,5 %` und `−0,1 %`, also
nichts. Ursache:

- Die Generierung übergibt bei **jedem** Aufruf einen KV-Cache. Gezählt über eine
  vollständige Generierung: `18` Aufrufe mit Cache, **`0` ohne**. Der cache-freie
  Pfad, den die Messung oben erfasst, wird real nie betreten.
- `mx.compile` kann den Cache nicht annehmen: `RotatingKVCache` ist kein Baum aus
  Arrays, der Aufruf endet mit `ValueError`.
- **`mlx-lm` fusioniert bereits selbst.** `gemma3_text.py` und `activations.py`
  tragen `@partial(mx.compile, shapeless=True)` an den Stellen, wo Fusion ohne
  Cache-Konflikt möglich ist.

Der Befund ist damit ein Negativergebnis mit klarer Ursache: Die naheliegende
Layer greift ins Leere, weil die fusionierbaren Teile schon fusioniert sind und
der Rest am mutablen Cache-Zustand scheitert.

Reproduzieren: `python tools/friday.py fusion --execute` — misst weiterhin den
cache-freien Forward-Pass und ist als solcher zu lesen, **nicht** als
Generierungsgewinn.


---

## Woher die Streuung kommt

Der Untergrund, der ungepaarte Messung wertlos macht, ist charakterisiert:

- **Unimodal mit langem rechtem Schwanz** — kein An/Aus-Zustand, kontinuierliche
  Verlangsamung.
- **Zufällig verteilt** — Runs-Test über sechs Sessions: beobachtet ≈ erwartet.
  Periodische Störungen und thermische Cluster sind damit ausgeschlossen.
- **Blockweit** — in den A/A-Läufen trafen `22` von `150` Blöcken **beide** Arme
  gleichzeitig; bei Unabhängigkeit wären `4,1` erwartet, Faktor `5,4`.
- **Zeitskala rund `340 ms`** — Autokorrelation über den Blockabstand: `+0,576`
  bei `68 ms`, `+0,124` bei `340 ms`, `0,000` bei `408 ms`.

Ein langsam variierender, gerätweiter Störprozess — plausibel
Betriebssystem-Scheduling und fremde Last, aus dem Prozess heraus nicht messbar.
**Nicht eliminierbar, aber erklärend:** Beide Arme eines Blocks liegen innerhalb
derselben Störungsepisode, weshalb sie sich im Quotienten herauskürzt. Damit wird
aus der empirischen Beobachtung „gepaart ist besser" ein Mechanismus.

---

## Wo die Zeit wirklich hingeht: speicherbegrenzt, nicht rechenbegrenzt

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| **Bandbreite genutzt** | **`31,9 %`** | **`51,2 %`** |
| **Rechenwerke genutzt** | **`2,4 %`** | **`3,9 %`** |
| Prefill je Token schneller | `7,3x` | `5,4x` |
| Urteil | `memory_bound` | `memory_bound` |

Rund **Faktor 13** Abstand. Die Rechenwerke sind bei echter Inferenz praktisch
untätig — sie warten auf Daten. Der Prefill-Vergleich bestätigt das über einen
unabhängigen Weg: Dort werden die Gewichte einmal für viele Token gelesen.

**Was daraus folgt.** Die Übersetzungskette Python → MLX → Metal → GPU-ISA ist vor
dem ersten Aufruf durchlaufen; der Kernel ist kompiliert und wird wiederverwendet.
Code „näher an der Maschinensprache" würde den Anteil optimieren, der mit
`2,4`–`3,9 %` ohnehin leerläuft. Wirksam sind nur **weniger Bytes**
(Quantisierung — bei 4-bit-Modellen bereits eingelöst) und **weniger Durchgänge**
(Kernel-Fusion, FlashAttention-Prinzip).

**Harte Obergrenze:** Bei `51,2 %` Bandbreitenauslastung bringt selbst eine
perfekte Optimierung, welche die Gewichte unverändert lässt, höchstens rund `2x`.
Alles darüber verlangt kleinere Gewichte, nicht besseren Code.

**Nebenbefund:** Das größere Modell nutzt die Hardware *besser* (`51,2 %` gegen
`31,9 %`) — je mehr je Token zu lesen ist, desto weniger wiegt der fixe Overhead.

Die Spitzenwerte `400 GB/s` und `21 TFLOPS` sind Herstellerangaben, nicht selbst
gemessen; sie begrenzen die Verhältnisse, sind aber keine eigene Evidenz.

Reproduzieren: `python tools/friday.py roofline --execute`

---

## Modelltests: Gemma 3

| | 1B (4bit) | 4B (4bit) |
| --- | ---: | ---: |
| Speicher geladen | `737 MB` | `2.561 MB` |
| Time to First Token | `205 ms` | `305 ms` |
| Durchsatz | rund `200 tok/s` | rund `85 tok/s` |
| Cooldown nach `30 s` | `1,37x` | `1,414x` |

**Der Cooldown-Effekt skaliert nicht mit der Modellgröße.** Trotz vierfacher
Parameterzahl ist die relative Verlangsamung praktisch gleich, und er tritt auch
ganz ohne Modell bei der reinen Matmul auf. Er ist damit eine Eigenschaft des
Geräts, nicht der Arbeitslast.

**Widerlegte Annahme:** Gemma 3 4B ist multimodal, aber `mlx-lm` lädt den
SigLIP-Vision-Tower **nicht**. `833,7 MB` liegen auf der Platte und kosten null
Arbeitsspeicher. Für Vision-Inferenz wäre `mlx-vlm` nötig.

---

## Was **nicht** gezeigt ist

- Keine Aussage über andere Geräte. Alle Zahlen stammen von einem M1 Max.
- Der Loop sucht in einem **festen, von Hand definierten** Raum von
  Ausführungsplänen. Er generiert keinen Code und schreibt keine Kernel.
- Keine Aussage über Transformer-Durchsatz unter Last, Qualität oder Genauigkeit.
- **H0.1 blieb ungelöst** (`h01_complete_unresolved`, `23` verfehlte Gates): Die
  Stationarität einer einzeln gepacten Operation ist auf diesem Gerät nicht
  belegt. Ursache sind über die Session verteilte Ausreißer — `16,7 %` aller
  Samples liegen über dem `1,5`-fachen Median. Das ist der größte offene Punkt.

## Geprüfte Nullbefunde

Diese Wege sind gemessen und führen nicht weiter — sie müssen nicht erneut
gegangen werden:

| Kandidat | Ergebnis |
| --- | --- |
| `mx.compile` | `+0,2 %`, KI `[0,999, 1,005]` — kein Effekt |
| `B` prätransponiert | `+3,1 %` langsamer |
| `mx.einsum` | `+0,6 %` langsamer |
| eigener GPU-Stream | `±0 %` |
| echter 3D-Batch-Matmul | `−3,9 %`/`−1,8 %`, KI enthält `1,0` |
| GPU-Vorwärmen (Matmul) | sieben Dosierungen, alle netto negativ |
| GPU-Vorwärmen (Inferenz) | zwei Varianten, beide KI enthalten `1,0` |

## Messregeln, die sich bewährt haben

1. **Immer gepaart** — beide Arme im selben Block.
2. **Schwelle vor dem Lauf festlegen**, nie danach.
3. **Mindestens zehn Wiederholungen.** Zwei eigene Zwischenzahlen mussten nach
   unten korrigiert werden, beide aus zu kleiner Stichprobe bei rechtsschiefer
   Verteilung.
4. **Behandlungsarme im direkten Wechsel** statt frei randomisiert, wenn die
   Behandlung selbst eine Zeitkomponente hat.
5. **Nach Konfidenzobergrenze auswählen**, nicht nach Punktschätzer, sobald aus
   mehreren Kandidaten gewählt wird.
6. **Correctness vor Timing.** Ein Ausführungsplan darf umsortieren, aber kein Bit
   ändern.
7. **Arme innerhalb von ~`340 ms`.** Der Störprozess dieses Geräts hat eine
   gemessene Zeitskala von rund `340 ms`. Liegen die Vergleichsarme weiter
   auseinander, sehen sie unterschiedliche Störungen und die Paarung verliert
   ihren Vorteil.

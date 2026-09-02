# Ergebnisse

> **Evidenz-Audit (21.08.2026):** H0/H0.1 besitzen verifizierbare Rohdaten in
> ihren SQLite-Domänen. Für die unten beschriebenen H1/H2-Läufe existieren dagegen
> nur historische Zusammenfassungen: Das formale A/A-Gate war nicht abgeschlossen,
> die MDE nicht vor dem ersten A/B-Lauf versiegelt und die Rohmessungen wurden
> nicht persistent gespeichert. Diese Zahlen sind deshalb **explorative
> Legacy-Beobachtungen**, auch wenn einzelne Läufe intern gepaart, repliziert und
> correctness-geprüft waren. Sie belegen weder formales H1/H2 noch Cross-Device-
> Übertragbarkeit. Die Korrektur ist nicht rückwirkend heilbar; siehe
> [`FORSCHUNGSENTSCHEID_2026-08-21.md`](FORSCHUNGSENTSCHEID_2026-08-21.md).

> **Das wichtigste Ergebnis zuerst:** Auf diesem Gerät ist ein *ungepaarter*
> Performancevergleich nahezu wertlos — die Streuung zwischen Läufen übertrifft
> die meisten realen Effekte. Gepaart gemessen liegt die Nachweisgrenze bei
> `2,2 %` statt `33 %`. Alle Werkzeuge hier messen deshalb gepaart und gegen eine
> vorab eingefrorene Schwelle.


Kompakte Übersicht der Beobachtungen mit ihrem heutigen Evidenzgrad. Die vollständige Herleitung samt
Fehlversuchen steht in [`ARBEITSJOURNAL.md`](ARBEITSJOURNAL.md); dieses Dokument
ist der Einstieg.

**Gerät:** Apple M1 Max, 32 GB Unified Memory, 32-Core GPU, macOS.
**Stand:** 21. August 2026. Alle Zahlen stammen von diesem einen Gerät; H1/H2
historisch liegen nur als explizit herabgestufte Zusammenfassungen vor. Zwei neue
native Schema-v1-Berichte besitzen Rohmessungen, bleiben aber ausdrücklich
`formal_claim=false`.

---

## Neue native v1-Exploration nach ausdrücklicher Rechenfreigabe

Kein Download und keine Installation: Gemma 3 1B/4B wurden ausschließlich aus
dem validierten projektlokalen Cache geladen. Jeder Lauf war an einen sauberen
Commit gebunden, lief am Netzteil unter dem gemeinsamen BudgetGuard und wurde
vor stdout append-only gespeichert.

| Messung | Ergebnis | Evidenz-ID |
| --- | --- | --- |
| Dispatch, 8× FP16-Matmul `2048²` | byte-identisch; `R=0,780054`, 95%-KI `[0,765530; 0,877456]`, `−21,995 %` | `b866022a…a92eb6` |
| Gemma 3 1B Roofline | `5,012 ms/Token`, `199,5 Token/s`, Bandbreite `36,53 %`, Compute `2,78 %` | `31c20b1e…647c36` |
| Gemma 3 4B Roofline | `10,949 ms/Token`, `91,3 Token/s`, Bandbreite `58,47 %`, Compute `4,45 %` | `31c20b1e…647c36` |

Roofline-Budget insgesamt: `10,360 s` GPU-Arbeit, `68,111 s` Wall, maximal
`1,129 s` kontinuierliche GPU-Arbeit und `52,123 s` verifizierte Pausen. Beide
Modelle werden durch die festgelegte Faktor-3-Regel explorativ als
`memory_bound` klassifiziert. Die Modellrevisionen
`2d44e83dc9e80843d22fb941d3d699a0b1351aa6` und
`93724907d4ed1745d2fe50baadf3b0b01a65abf2` sind im Bericht gebunden.

Ein erster Roofline-Aufruf wurde korrekt als `measurement_failed` persistiert,
bevor 4B startete: Zwischen Warmups und Wiederholungen fehlten Guard-Pausen, so
dass die `6-s`-Kontinuierlichkeitsgrenze griff (`ffe98ffa…1a0ac4`). Nach Fix,
drei neuen Pacing-Tests und vollständiger Regression (`439` Tests,
`2.447` Subtests) war der identische Wiederholungslauf erfolgreich.

**Evidenzgrenze:** Die Dispatch-MDE ist in diesem Werkzeug eingefroren und die
Roofline-Rohsamples sind vollständig gespeichert. Schema v1 validiert jedoch
keine formale Study-ID, A/A-Generation oder Familien-/Splitverträge. Deshalb sind
beide Ergebnisse prospektive Exploration, kein formaler H1/H2-, Cross-Device-
oder Phase-1B-Nachweis.

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

## Explorative Legacy-Beobachtung: Dispatch-Batching

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

**Evidenzgrad:** technisch plausibel, aber kein formaler H1-Nachweis; Rohblöcke
fehlen und die verwendete `5-%`-Schwelle war nicht prospektiv versiegelt.

**Reichweite, ehrlich:** Das ist keine Kernel-Optimierung — der Matmul-Kernel
bleibt unverändert. Es entfernt vermeidbare Synchronisation und gilt für
**unabhängige** Operationen. Die serielle Baseline ist ein realer Anti-Pattern
(jedes `mx.eval` in einer Schleife erzeugt sie), den erfahrener MLX-Code ohnehin
vermeidet.

Ein neuer Aufruf wäre eine neue, separat freizugebende Messung:
`python tools/friday.py dispatch --execute --n 8`

---

## Explorative Legacy-Beobachtung: der Loop findet Kandidaten

Ein geschlossener Mess-Entscheidungs-Kreis: explorieren, um den Überlebenden herum
verfeinern, den eigenen Sieger unabhängig bestätigen. Vier Läufe:

| Lauf | gewählt | Effekt | Verdikt |
| ---: | :---: | ---: | --- |
| 1 | `N=8` | `−13,60 %` | bestätigt |
| 2 | `N=6` | `−11,13 %` | bestätigt |
| 3 | `N=6` | `−14,11 %` | bestätigt |
| 4 | `N=16` | `−11,08 %` | bestätigt |

`N=6` und `N=7` kamen in der manuellen Suche nicht vor; der Loop hat sie selbst
vorgeschlagen und einen davon im damaligen explorativen Protokoll erneut gemessen.

**Dass die gewählte Batchgröße zwischen Läufen wechselt, ist kein Mangel.** Das
Optimum ist ein breites Plateau: `N = 4` bis `16` liegen alle im Bereich
`−11 %` bis `−17 %`, und der Loop landet je nach Rauschen an unterschiedlichen
Stellen darauf. Historisch wiederholte sich der Effekt, aber ohne formales Gate
darf daraus kein bestätigter H1-Gewinn abgeleitet werden.

**Eine Fallgrube, die dokumentiert bleiben soll:** Zunächst bestätigte der Loop nur
in `1` von `3` Läufen. Ursache war der *Winner's Curse* — der Beste aus mehreren
verrauschten Kandidaten ist konstruktionsbedingt zu optimistisch. Rangfolge nach
dem Punktschätzer wählte den glücklichsten Ausreißer (`0,750`, `0,741`), der bei
Nachmessung auf `0,87`–`0,96` regressierte. Die Lösung war, nach der
**Konfidenzobergrenze** zu ranken statt nach dem Punktschätzer: nicht „was sah
einmal am besten aus", sondern „was ist zuverlässig gut". Danach `4` von `4`.

Ein neuer Aufruf wäre eine neue, separat freizugebende Messung:
`python tools/friday.py loop --execute`

---

## Explorative H2-Vorstufe: ein lokales Modell schlägt Parameter vor

`gemma-3-4b-it-4bit` erhält die bisherigen Messungen und die gemessenen
Gerätefakten und schlägt ungetestete Kandidaten vor. Über drei Runden sieht es die
Ergebnisse seiner eigenen Vorschläge.

| Runde | Antwort | Ergebnis |
| ---: | --- | --- |
| 1 | `[3, 10, 16]` | `N=3` verworfen, `N=10` und `N=16` bestehen |
| 2 | `[5, 12, 13]` | alle drei bestehen |
| 3 | `[7, 14, 15]` | alle drei bestehen |

Neun Werte vorgeschlagen, alle gültig und ungetestet. Explorativ erneut gemessen wurde `N=13` mit
`−11,53 %`, `95%-KI [0,8552, 0,8957]`.

**`model-loop` schlägt Parameter vor, niemals Code.** Modellgenerierter Code ist
ein separates Sicherheitsproblem und gehört nur zum nachfolgenden `codegen`-Werkzeug.
Prosa, Shell-Fragmente, Floats, Booleans und Werte außerhalb `2..16` führen
sämtlich zu null ausgeführten Kandidaten — abgesichert durch `21` Tests.

**Der Harness bleibt streng:** `N=3` wurde verworfen, weil seine
Konfidenzobergrenze `0,954` die Schwelle `0,95` verfehlte. Das Modell schlägt vor,
es entscheidet nicht.

Ein neuer Aufruf wäre eine neue, separat freizugebende Modell-/GPU-Messung:
`python tools/friday.py model-loop --execute`

---

## Explorative Codegen-Beobachtung: das Modell schreibt den Plan selbst

`codegen` lässt `gemma-3-4b-it-4bit` den Ausführungsplan als Python schreiben.
Drei Schutzschichten stehen zwischen generiertem String und berichtetem Ergebnis:

1. **AST-Allowlist** — genau eine Funktion `plan(mx, a, operands)`, nur die MLX-
   Operationen `matmul`, `eval`, `synchronize`, kleine Ganzzahlliterale und
   operandengebundene Schleifen plus `append`/`extend` auf Akkumulatorlisten.
   Keine Importe, Dunder, String-Literale, freien Allokationsprimitiven oder Lambdas.
2. **Prozessisolation** — frischer Subprozess mit Wall-Timeout, Kernel-CPU-Grenze
   und bereinigter Umgebung. `mx.set_memory_limit` ist nur zusätzliche
   Best-Effort-Abwehr, keine harte RAM-Isolation.
3. **Correctness** — ein Ergebnis je Operand, jedes bytegleich zur Referenz.

Ergebnis: fünf Pläne geschrieben, fünf gemessen, drei über der Schwelle.
Historisch als explorativer Treffer klassifiziert: `R = 0,8838`, **`−11,62 %`**,
`95%-KI [0,8676, 0,8975]`,
Replikate `0,8742 / 0,8970 / 0,8838`.

Auch dieser Befund ist wegen der fehlenden formalen H1-Basis, nicht versiegelten
MDE und fehlenden Rohpersistenz kein formaler H2-Nachweis.

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

Ein neuer Aufruf wäre eine neue, separat freizugebende Modell-/GPU-Messung:
`python tools/friday.py codegen --execute`

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

Ein neuer Aufruf wäre eine neue, separat freizugebende Messung:
`python tools/friday.py cooldown --execute`

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

Ein neuer, separat freizugebender Aufruf `python tools/friday.py fusion --execute`
misst weiterhin den
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

## Explorative Roofline-Einordnung: speicherbegrenzt auf dem beobachteten Lauf

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| **Bandbreite genutzt** | **`31,9 %`** | **`51,2 %`** |
| **Rechenwerke genutzt** | **`2,4 %`** | **`3,9 %`** |
| Prefill je Token schneller | `7,3x` | `5,4x` |
| Urteil | `memory_bound` | `memory_bound` |

Rund **Faktor 13** Abstand. Die Rechenwerke sind bei echter Inferenz praktisch
untätig — sie warten auf Daten. Der historische Prefill-Vergleich stützt das über einen
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

Ein neuer Aufruf wäre eine neue, separat freizugebende Modell-/GPU-Messung:
`python tools/friday.py roofline --execute`

---

## Explorative Modelltests: Gemma 3

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

---

## Prospektives N10-Ergebnis und begrenzte Runtime-lite-Runtime

Die explorative Gemma-Auswahl wurde nicht noch einmal verwendet. Stattdessen
wurde `N=10` als einziger Kandidat in einer neuen prospektiven Studie geprüft.
V1 stoppte korrekt vor jedem Timing an einer nicht registrierten
Fixture-Identität und wurde nicht wiederholt. Der eigenständige V2-Vertrag
verwendete eine registrierte Fixture, frische übrige Seeds und einen eigenen
16-Record-Store.

| Messung | Ergebnis |
| --- | ---: |
| N10 A/A, 6 Sessions | `R=0,999586`, 95%-KI `[0,998764; 1,000443]` |
| eingefrorene MDE | `5 %` |
| N10 A/B, 6 Sessions | **`R=0,874912`, `−12,509 %`**, 95%-KI `[0,871768; 0,875614]` |
| Charakterisierung / Validierung | `R=0,875216` / `R=0,874608` |
| Korrektheit | byteidentisch |

Damit ist der feste Batch-Dispatch-Plan formal nur für einen M1 Max,
FP16-`2048²` und genau zehn Matmuls bestätigt. Der terminale Record ist der
einzige formale Claim; er erlaubt ausschließlich einen begrenzten
Runtime-Prototyp.

Dieser getrennte N10-Runtime-/Runtime-lite-Prototyp bestand anschließend seine
eigenen Engineering-Gates auf sauberem Commit `5eaad38`:

| Runtime-Gate | Ergebnis | Grenze |
| --- | ---: | ---: |
| Cold Load | `3,482664 s` | `≤10 s` |
| Policy-Median | `12,372 µs` | `≤25 µs` |
| Policy-p95 | `12,448 µs` | `≤50 µs` |
| zusätzlicher Policy-Median | `12,343 µs` | `≤20 µs` |
| MLX/GPU, 12 Blöcke | **`R=0,875753`, `−12,425 %`** | `R≤0,95` |
| Runtime-Korrektheit | byteidentisch, `max_abs_error=0` | exakt |

Runtime-lite bedeutet hier bewusst keine allgemeine autonome Architektur: Der
Controller beobachtet reale Tensoren, verifiziert die versiegelte Evidenz,
wählt genau einen bekannten Plan und fällt sonst seriell zurück. Es gibt keine
freie Suche, keine Codegenerierung, keine Modellaktion und kein Custom Metal.
Die zwei Runtime-Messungen sind Engineering-Evidenz und erweitern den formalen
N10-Claim nicht.

## F1 — Integrationsstudie, warmer Arm (2026-09-02)

**Erste End-to-End-Zahl des Projekts.** Persistenter Prozess vorausgesetzt
(warmer Arm), gemessen wurden `head_skip_prefill` und `compiled_fixed_cache`
gemeinsam gegen den unveränderten Pfad.

| Größe | Wert |
| --- | --- |
| Anfrage-Ratio (Median) | `0,8600567` |
| End-to-End-Gewinn | **`13,99 %`** |
| Konfidenzintervall | `[0,853444 ; 0,873056]` |
| vorregistrierte Schwelle | `10 %` (Ratio `0,90`) |
| Paare | `6`, Tokenidentität `6/6` |
| A/A-Rauschen | `0,612 %` |
| Status | **`qualified`** |

Phasenweise: TTFT-Ratio `0,849479` (einzeln bestätigt `0,846385`),
Decode-Ratio `1,094084` (einzeln `1,075741`). Die Knobs stören sich nicht.

Die vorab veröffentlichte Projektion `13,68 %` liegt im Konfidenzintervall und
ist damit bestätigt. Die naive Produktrechnung hätte `21,32 %` versprochen —
Prefill- und Decodegewinn setzen sich als zeitgewichtetes Mittel zusammen,
nicht multiplikativ.

**Geltungsbereich:** ein Gerät, Snapshot `93724907…`, IronMule `03e884cb…`,
`897`-Token-Promptfamilie, `32` generierte Token, Batch `1`, greedy.
`formal_claim=false`, keine Aktivierung, kein Cross-Device- oder
Cross-Model-Claim. Bei `256` generierten Token fällt die Erwartung auf
`9,80 %` und damit unter die Schwelle.

**Offen:** der kalte Arm (mit persistentem Prozess, Erwartung rund `70 %`).

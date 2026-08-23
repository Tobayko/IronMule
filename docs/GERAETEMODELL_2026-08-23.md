# Ein Gerätemodell, das die Latenz in zwei Terme trennt

**Stand:** 23. August 2026 · **Status:** explorativ, `formal_claim=false`.
Werkzeug: `tools/measure_device_model.py`, Bericht: `experiments/device_model/report.json`.

## 1. Die Frage

Wie viel würde dasselbe Modell auf einem anderen Gerät kosten — einem Telefon etwa,
mit rund einer Größenordnung weniger Speicherbandbreite? Die naheliegende Antwort
lautet "mit der Bandbreite skalieren", weil Dekodieren bei Batch `1` jedes Gewicht
einmal je Token liest.

Diese Antwort ist hier nachweislich falsch. Das 4B trägt `3,9x` die Gewichte des 1B,
braucht aber nur `1,99x` so lange. Ein reines Bandbreitenmodell kann das nicht
erzeugen.

## 2. Zwei Terme

Jeder Layer kostet etwas, das mit der Gewichtsgröße nichts zu tun hat — Dispatch,
Start, Synchronisation — und darüber hinaus müssen die Gewichte gelesen werden:

```
ms_je_Token  =  Layer × je_Layer_ms  +  Gewichte_GB / effektive_Bandbreite
```

Gefittet auf den beiden vollen Modellen, geprüft an vier zurückgehaltenen
Konfigurationen mit halbierter und geviertelter Layerzahl:

| Parameter | Wert |
| :--- | ---: |
| je Layer | `0,16669` ms |
| je GB Gewichte | `2,79005` ms |
| **effektive Bandbreite** | **`358,4` GB/s** |
| Anteil der veröffentlichten Spitze (`400` GB/s) | **`89,6 %`** |

| Zurückgehalten | Layer | GB | gemessen | vorhergesagt | Fehler |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 4B halb | 17 | `1,2804` | `6,040` | `6,406` | `+6,0 %` |
| 4B viertel | 8 | `0,8024` | `3,093` | `3,572` | `+15,5 %` |
| 1B halb | 13 | `0,3663` | `3,161` | `3,189` | `+0,9 %` |
| 1B viertel | 6 | `0,2605` | `1,836` | `1,727` | `−5,9 %` |

Der Fehler wächst bei sehr flachen Stacks, wo ein Fixkostenanteil je *Aufruf* — nicht
je Layer — zu wiegen beginnt, den das Modell nicht führt. Für die realen Tiefen
`26` und `34` liegt es innerhalb weniger Prozent.

## 3. Was das an einer früheren Aussage korrigiert

Frühere Abschnitte berichteten "`65 %` Bandbreitenausnutzung, also rund `1,48x`
Kopfraum für einen besseren Kernel". Diese Zahl entstand, indem die gesamte Zeit durch
die Gewichtsbytes geteilt wurde — und rechnete damit den Dispatch-Overhead der
Bandbreite zu.

Getrennt betrachtet läuft das Speichersystem bei **`89,6 %`** der Spitze. Dort ist
fast nichts mehr zu holen. Die Zeit steckt woanders:

| 4B, ein Token | ms | Anteil |
| :--- | ---: | ---: |
| `34` Layer × Dispatch | `5,67` | **`48 %`** |
| Gewichte lesen | `6,09` | `52 %` |

**Fast die Hälfte der Einzelanfrage-Latenz ist Kernel-Start, nicht Speicher.** Das
erklärt auch das Breitenplateau: mit wachsender Breite verteilt sich derselbe feste
Betrag auf mehr Token.

## 4. Projektion auf schmalbandigere Geräte

Beide Parameter sind Geräteeigenschaften und müssen beide gesetzt werden. Die
Bandbreite eines fremden Geräts einzusetzen und die Dispatch-Kosten dieser Maschine
beizubehalten, unterstellt gleich schnelle Scheduler — genau die Annahme, die eine
Projektion wie eine Messung aussehen lässt. Die letzte Zeile zeigt deshalb, was ein
doppelt so langsamer Dispatch ändert.

| Gerät | Modell | ms/Token | Token/s | fix | Bandbreite |
| :--- | :--- | ---: | ---: | ---: | ---: |
| M1 Max (gemessen) | 1B | `5,90` | `169` | **`73 %`** | `27 %` |
| M1 Max (gemessen) | 4B | `11,76` | `85` | `48 %` | `52 %` |
| `100` GB/s | 1B | `9,96` | `100` | `44 %` | `56 %` |
| `100` GB/s | 4B | `27,50` | `36` | `21 %` | `79 %` |
| `50` GB/s | 1B | `15,59` | `64` | `28 %` | **`72 %`** |
| `50` GB/s | 4B | `49,33` | `20` | `11 %` | **`89 %`** |
| `50` GB/s, `2x` Dispatch | 1B | `19,92` | `50` | `44 %` | `56 %` |
| `50` GB/s, `2x` Dispatch | 4B | `55,00` | `18` | `21 %` | `79 %` |

## 5. Der Flaschenhals kippt mit dem Gerät

Auf dem M1 Max ist das 1B zu `73 %` Dispatch-Overhead. Auf einem Gerät mit `50` GB/s
ist **dasselbe Modell** zu `72 %` Bandbreite. Der begrenzende Faktor tauscht die
Seiten, und damit die richtige Optimierung:

| Gerät | Hebel | warum |
| :--- | :--- | :--- |
| M1 Max | **weniger Kernel-Starts** (Layer-Fusion, Graph-Capture) | Speicher läuft schon bei `89,6 %` |
| schmalbandiger SoC | **weniger Bytes** (Quantisierung) | Dispatch ist beim 4B nur noch `11 %` |

Eine Optimierung, die auf dem Laptop gewinnt, kann auf dem Telefon nahezu wirkungslos
sein und umgekehrt. Ein Profil je Gerät ist deshalb keine Sorgfaltspflicht, sondern
Voraussetzung.

## 6. Grenzen

Die beiden Zeilen für den M1 Max sind gemessen. Alles darunter ist **Projektion**:
gerechnet, nicht beobachtet. Sie setzt voraus, dass dieselbe Zweitermform auf anderer
Hardware gilt — plausibel, weil beide Terme physikalische Ursachen haben, aber auf
keinem zweiten Gerät geprüft. Die Bandbreitenwerte `100` und `50` GB/s sind gesetzte
Annahmen, keine Herstellerangaben; das Werkzeug nimmt jeden Wert entgegen.

Das Modell beschreibt **Batch `1`**. Bei größerer Breite verteilt sich der feste Term,
was das Breitenplateau erzeugt und hier nicht abgebildet ist.

## 7. Ein widerlegter Versuch: `mx.compile` auf dem Decode-Schritt

Aus Abschnitt 3 folgt eine naheliegende Vorhersage: wenn `48 %` der Latenz
Kernel-Starts sind, muss Kernel-Fusion den **je-Layer-Term** senken und den
Bandbreiten-Term unberührt lassen. Gemessen sah das zunächst nach einem Treffer aus:

| | je Layer | je GB | 4B gesamt | 1B gesamt |
| :--- | ---: | ---: | ---: | ---: |
| eager | `0,16561` | `2,7695` | `11,677` ms | `5,864` ms |
| `mx.compile` | `0,12615` | `2,9896` | `10,816` ms | `4,962` ms |
| Änderung | **`−23,8 %`** | `+7,9 %` | `−7,4 %` | `−15,4 %` |

Genau der vorhergesagte Term bewegte sich, und das stärker dispatch-gebundene 1B
gewann mehr. Das Modell schien bestätigt.

**Der Korrektheitstest widerlegt es.** Über `48` Token wich die Ausgabe ab
Position `2` ab; der Text war ein anderer. Der gemessene Gewinn ist der Preis einer
Rechnung, die nicht dasselbe tut.

Die Ursache ist strukturell und liegt in `KVCache.update_and_fetch`:

```python
self.keys[..., prev : self.offset, :] = keys      # mutiert bestehenden Speicher
return self.keys[..., : self.offset, :], ...      # Rückgabeform wächst je Token
```

`mx.compile` tracet auf Formen und setzt reine Funktionen voraus. Der Cache ist
gefangener, mutierender Zustand, **dessen Form mit jedem Token um eins wächst**. Ein
korrekt deklarierter Zustand hieße deshalb Neukompilierung je Token — langsamer, nicht
schneller. Ohne Deklaration wird ein veralteter Trace mit falschem Cache-Ausschnitt
wiederverwendet, was genau die beobachtete Abweichung erzeugt.

`mlx_lm` kompiliert den Forward-Pass folgerichtig nicht; die Sampler dagegen schon,
mit ausdrücklich deklariertem RNG-Zustand
(`sample_utils.py`, `@partial(mx.compile, inputs=mx.random.state, ...)`).

**Der Befund aus Abschnitt 3 bleibt bestehen**: `48 %` der Einzelanfrage-Latenz sind
Kernel-Starts, und das ist weiterhin das richtige Ziel. Nur ist der Hebel keine
Codezeile.

Der konkrete Weg wäre ein **KV-Cache fester Form mit Maskierung** statt eines
wachsenden Ausschnitts. Damit blieben die Formen über viele Token konstant,
Kompilierung würde gültig, und der gemessene `−23,8 %`-Dispatch-Effekt wäre echt
abrufbar. Der Preis ist verworfene Attention-Arbeit über dem maskierten Bereich —
bei `f_attention` von `12`–`17 %` vermutlich tragbar, aber **nicht gemessen**.

Prüfskript: `experiments/device_model/check_compiled_decode.py`.

## 8. Wie weit der Dispatch-Hebel von außen erreichbar ist

Abschnitt 7 schloss mit dem Vorschlag, ein KV-Cache fester Form würde Kompilierung
gültig machen und die gemessenen `−23,8 %` Dispatch-Kosten echt abrufbar. Das wurde
geprüft, in drei Stufen, und endet an einer klaren Grenze.

**Stufe 1 — RoPE.** Der Offset wird an `mx.fast.rope` übergeben; als Python-Integer
würde ihn ein Trace einbacken. Geprüft: RoPE akzeptiert auch einen `mx.array` als
Offset. Diese Hürde fällt weg.

**Stufe 2 — feste Formen.** Der Offset wurde eingefroren, sodass jeder Schritt in
denselben Platz schreibt und dieselbe Ausdehnung zurückliest — genau das Formprofil,
das ein Fenster-Cache hätte. **MLX verweigert die Kompilierung** mit
`Attempting to eval an array without a primitive … make sure all the inputs and
outputs are captured`.

Das ist bemerkenswert im Vergleich zu Abschnitt 7: dort **warnte MLX nicht** und
lieferte still falsche Token. Sobald die Formen konstant sind, greift der Schutz.

**Stufe 3 — Zustand deklarieren.** Die Cache-Tensoren wurden als
`inputs`/`outputs` übergeben. Weiterhin verweigert. Der Grund steht in
`KVCache.update_and_fetch`: die Methode **bindet `self.keys` neu**, statt den
bestehenden Puffer zu mutieren. Eine von außen erfasste Liste zeigt danach auf die
alten Arrays, und MLX kann den Zustand nicht verfolgen.

**Ergebnis:** Der Dispatch-Hebel ist **von außen nicht erreichbar**. Er verlangt eine
Cache-Implementierung, deren Puffer in einem verfolgbaren Container liegen und die
in-place schreibt statt neu zu binden — also eine Änderung im Framework- oder
Modell-Layer, kein Konfigurationsschalter.

**Der Befund aus Abschnitt 3 bleibt unberührt.** `48 %` der Einzelanfrage-Latenz sind
Kernel-Starts. Das ist weiterhin das größte Latenzziel auf diesem Gerät, und es ist
jetzt genau beziffert, was es kosten würde, es zu heben: einen neuen Cache, keinen
Einzeiler. Ob die `−23,8 %` dann tatsächlich anfallen, ist **nicht gezeigt** — die
einzige Messung dazu stammt aus einer Konfiguration, die falsche Ergebnisse lieferte.

Prüfskripte: `experiments/device_model/check_compiled_decode.py`,
`experiments/device_model/check_fixed_shape_compile.py`.

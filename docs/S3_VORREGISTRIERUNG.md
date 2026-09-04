# S3 — Vorregistrierung: reproduziert sich der Identitätsbruch, und was zeigt er?

**Studien-ID:** `identity-break-20260902-01`
**Registriert:** 2026-09-02, vor dem ersten Lauf, ohne dass eine Messung auf dem
Dump-fähigen Messkern existiert.

## Anlass

H1.0 hat bei `4B`/`128` Token/Entwurfsbreite `2` die Tokenidentität gebrochen
(`experiments/switch_point/switch_4b_128_w2.json`,
`token_identity_broken:pair_0`). `ironmule/runtime.py:_decode_speculative`
behauptet Identität **per Konstruktion**: ein Entwurfstoken wird nur übernommen,
wenn es dem entspricht, was das Modell selbst für diese Position gewählt hat.
Ein Bruch heißt also, dass die Konstruktion nicht hält, was sie behauptet, oder
dass die Verifikation nicht das vergleicht, was sie zu vergleichen glaubt.

`friday_serve/speculation.py` stützt seine Freigabe ohne Geräteprofil
ausdrücklich auf diese Identitätsbehauptung. Deshalb schlägt dieser Befund jede
Geschwindigkeitsfrage, auch nachdem der `(K,N)`-Dispatcher am selben Tag
gestorben ist.

**Der Lauf hat den Bruch erkannt und die Evidenz weggeworfen.** Die Datei
enthält die Meldung und keine Sequenzen; `friday_calibrate.runner.Sample` trug
nur `token_sha256`, und die Tokenfolge starb in `build_runner.run()`. Das ist
die erste Sache, die diese Studie repariert.

## Was diese Studie beantwortet — und was ausdrücklich nicht

**Stufe A, diese Vorregistrierung:** reproduziert sich der Bruch, und wie sieht
er aus? Beide Tokenfolgen, der erste divergierende Index, die Token an dieser
Position, und bei welchem Paar er auftrat.

**Stufe B, eigene Vorregistrierung, erst nach A:** der Top-2-Abstand der Logits
**an der ersten divergierenden Position**. Nur dieser Wert prüft den
Numerikzweig; das Minimum der Promptfamilie prüft nichts. Stufe B wird erst
geschrieben, wenn A eine Position geliefert hat.

## Zwei Zweige, gleichrangig vorregistriert

1. **Numerik/Formabhängigkeit.** `Engine._body` kompiliert je
   `(capacity, width)`. Der Referenzarm läuft mit `width = 1`, der Kandidat mit
   `width = speculate_k + 1`, hier also `3`. Zwei kompilierte Graphen, andere
   Kernel, andere Reduktionsreihenfolge — an einer knappen Position könnte der
   `argmax` kippen.
   **Gegen diesen Zweig spricht die eigene Evidenz:** P2 misst als *kleinsten*
   Top-2-Abstand dieser Promptfamilie `0,500`. Ein Abstand von `0,5` in den
   Logits ist kein knappes Rennen; Reduktionsunterschiede verschieben Logits um
   Größenordnungen weniger. Wenn `0,500` das Minimum ist, gibt es hier keine
   entarteten Positionen, und dieser Zweig ist der unwahrscheinlichere.
2. **Akzeptanzlogik und Cache-Rücknahme.** Der Aufbau von `accepted`, die
   Rücknahme `state["position"]["offset"] = mx.array(offset - 1, ...)` und die
   Maske über verworfene Slots. Durchgerechnet wirkt die Rücknahme konsistent —
   der nächste Forward überschreibt die verworfenen Positionen —, aber „nichts
   gefunden" ist kein Ausschluss.

### Wie diese Studie beide Zweige berührt — und wo ihre Trennschärfe endet

Der erste Entwurf dieses Dokuments nannte beide Zweige gleichrangig und maß nur
den ersten: Position, Sequenzen und der Top-2-Abstand aus Stufe B prüfen
ausschließlich die Numerik. Aus den Ausgabetoken allein ist Zweig 2 nicht
rekonstruierbar — ob die divergierende Position auf eine **Teilannahme** folgte,
steht in ihnen nicht.

**Was ohne jede Instrumentierung und ohne GPU-Zeit trotzdem geht.**
`ironmule.runtime._lookup_draft` ist eine **reine Funktion** aus bisheriger
Sequenz und `(ngram, k)`. Damit lässt sich die Schleife von
`_decode_speculative` offline **nachspielen**, und zwar vollständig, weil das
Gatter den Entwurf gegen genau die ausgegebenen Token prüft:

```python
accepted = [chosen[0]]
for i in range(1, width):
    if i - 1 < len(draft) and draft[i - 1] == chosen[i - 1]:
        accepted.append(chosen[i])
```

Eine Annahme setzt `draft[i-1] == chosen[i-1]` voraus, und `chosen[i-1]` ist das
zuvor ausgegebene Token. Aus Prompt plus ausgegebener Kandidatenfolge folgen
also je Iteration der Entwurf, die Zahl der angenommenen Token und damit die
Iterationsgrenzen — genau die Annahmebuchführung, für die sonst eine
Instrumentierung in `_decode_speculative` nötig wäre.

**Ein früherer Entwurf dieses Abschnitts hatte hier ein zirkuläres Kriterium.**
Er wollte prüfen, ob das divergierende Token das dort vorgeschlagene
Entwurfstoken ist. Das unterscheidet nichts: `_decode_speculative` gibt **nie**
ein Entwurfstoken aus, sondern immer `chosen[j]`; an jeder gegatterten Position
stimmen beide per Konstruktion überein, im gesunden wie im kranken Fall.

**Der tragfähige Diskriminator ist die Position *innerhalb ihrer Iteration*:**

- **`j = 0`** — das ungegatterte freie Token. Der breite Forward wählt dort
  selbst, ohne dass ein Entwurf im Spiel war. Divergenz hier zeigt auf
  **Zweig 1** und schließt Zweig 2 für diese Position weitgehend aus. **Das ist
  das schärfere der beiden Signale.**
- **`j > 0`** — eine gegatterte Ausgabe, deren Gatter am Schritt davor
  durchgelassen hat. Divergenz hier **belastet** Zweig 2, beweist ihn aber
  nicht: Numerik kann auch eine gegatterte Position treffen. **Indiz, kein
  Beweis** — die beiden Signale sind ausdrücklich nicht gleich stark.

**Zwei Randbedingungen des Nachspielers.**

1. Vertrauenswürdig ist die Rekonstruktion nur **bis einschließlich** der
   divergierenden Position. Danach läuft der Kandidat auf einer abgewichenen
   Sequenz; jede weitere Iteration ist Folge, nicht Ursache. Zahlen dahinter
   sind kein Befund.
2. Der Nachspieler wird gegen einen Lauf geprüft, in dem die Identität
   **gehalten** hat: dort muss an jeder gegatterten Ausgabe
   `draft[j-1] == emittiert[j-1]` gelten. Bricht die Rekonstruktion an einem
   gesunden Lauf, ist der Nachspieler falsch und nicht die Runtime. Die
   gesunden Sequenzen liefert der Warmup derselben Zelle, kostet also keine
   zusätzliche GPU-Zeit.

**Wo die Trennschärfe endet, ausdrücklich.** Der Nachspieler sieht **nicht**,
ob eine Cache-Rücknahme aus einem früheren Schritt Spuren hinterlassen hat, die
sich erst später zeigen. Dafür bleibt die Instrumentierung in
`_decode_speculative` — zweite angefasste Datei im gepinnten Worktree, dieselbe
Hash- und `patch.diff`-Mechanik wie im Amendment — mit eigener Vorregistrierung.
„Gleichrangig" beschreibt den Stand der Hypothesen, nicht die Trennschärfe
jeder einzelnen Messung.

## Die Zelle

- `mlx-community/gemma-3-4b-it-4bit`, versiegelter `897`-Token-Prompt.
- `128` generierte Token, Entwurfsbreite `2` — exakt die Zelle, die gebrochen ist.
- Baselinearm wie in H1.0: `head_skip_prefill`, `compiled_fixed_cache`,
  `readback_every = 8`. Kandidat: derselbe Satz plus `speculate_k = 2`,
  `speculate_ngram = 3`.
- **`10` Paare**, abwechselnd AB/BA, Warmup je Arm. Der Lauf endet beim ersten
  Bruch — der Bruch ist das Ereignis, nicht die Zeit.
- Budget unverändert: `120 s` GPU, `6 s` ununterbrochen, Deckel wird nicht
  angehoben. Die Zelle kostet höchstens `10` Paare à zwei Aufrufen zu rund
  `4,3 s`, also rund `90 s` GPU.

**Zeiten werden nicht ausgewertet.** Diese Studie misst ein Ereignis, keinen
Gewinn; es gibt kein A/A, keine Schwelle und kein `wins`/`loses`.

**Umstand, nicht Frage:** bei `128` Token rundet `Engine._capacity` beide Arme
auf `1088` auf — der Kandidat über `897 + 128 + 2`, die Baseline über
`897 + 128`. Die Kapazität ist also in **beiden** Armen gleich; verschieden ist
allein die kompilierte `width`. Das ist festgehalten, damit Stufe B nicht die
falsche Variable verdächtigt, und es ist kein Teil der hiesigen Frage.

**Was H1.0 zusätzlich weiß und was hier nicht nachgemessen wird:** bei `128`
Token hielt Breite `1` die Identität über `6` Paare. Breite `3` wurde nie
gemessen, weil der Lauf nach Vorregistrierung endete.

## Die Änderung am gemeinsamen Messkern

Der Dump braucht zwei Ergänzungen in `friday_calibrate/runner.py`:

- `Sample` bekommt `token_ids: tuple[int, ...] = ()`, gefüllt in
  `build_runner.run()`. Default leer, alle bestehenden Aufrufer unberührt.
- `paired_arms` bekommt `on_break: Callable[[int, Sample, Sample], None] | None`.
  Kein Rückgabetyp ändert sich; `noise_mde` und `verdict_for` bleiben
  unverändert.

**Reihenfolge, verbindlich:** die Änderung geht **vor** die erste S3-Messung,
mit eigenem Test, und danach ist der Kern wieder eingefroren. Nicht zwischen
zwei Zellen. Grund ist die Erfahrung aus dem Amendment: ein Kern, der sich
zwischen zwei Zellen ändert, macht die entscheidende Zelle unvergleichbar.

## Ausgänge — alle drei vorab

- **Der Bruch reproduziert sich** (mindestens ein Paar von `10`): der Dump
  liefert beide Folgen, den ersten divergierenden Index und die beiden Token an
  dieser Position. Das ist die Evidenz, die H1.0 verloren hat. Stufe B wird
  vorregistriert und misst dort den Top-2-Abstand.
- **Der Bruch reproduziert sich nicht** (`10` Paare ohne Abweichung): das ist
  ein **anderer** Befund, nicht das Ausbleiben eines Befunds. Ein Bruch, der
  beim Wiederholen verschwindet, macht die Identitätsbehauptung nicht besser,
  sondern schlechter: sie wäre dann nicht deterministisch prüfbar. Berichtet
  wird „bei `10` Paaren nicht reproduziert, einmal in `1` Paar aufgetreten";
  eine Aussage über die Rate wird daraus **nicht** abgeleitet. S3 bleibt offen,
  und der nächste Schritt ist mehr Paare, nicht eine Ursachensuche.

  **Der Confounder dieses Ausgangs wird vorab geschlossen, nicht hinterher
  benannt.** S3 läuft mit geändertem Messkern; H1.0 hat den Bruch unter
  `friday_calibrate/runner.py` `de2eb778c4a353867bbee087efa628a05938d510a860bb35b5e10b1f3f259b7f`
  erzeugt, S3 läuft unter dem Hash, den der Bericht führt. Dass die Änderung
  nicht verhaltensändernd ist, wird **belegt**, nicht behauptet: ein Test fährt
  denselben Fall mit und ohne `on_break` und vergleicht Sequenzen,
  `token_sha256` und Verdikt. Ohne diesen Beleg steht der geänderte Kern
  hinterher als Ausrede zur Verfügung.
- **Der Lauf scheitert am Budget oder an einem Fehler**: die Zelle gilt als
  nicht gemessen, mit Fehlertext im Bericht — wie bei `64`/w3 in H1.0.

## Kill-Kriterium

Lässt sich der Bruch weder reproduzieren noch auf eine Ursache zurückführen,
darf Spekulation im Auslieferungspfad **nicht** als tokenidentisch geführt
werden. Dann ist sie ein Kandidat mit Qualitätsgate wie jeder andere — was der
Nutzer am 2026-09-02 ausgeschlossen hat — und fällt damit aus dem
Auslieferungspfad. Der Eintrag `friday_serve/speculation.py` verliert seine
Begründung „braucht kein Promotionsgate, weil identisch per Konstruktion".

## Ergebnis, gemessen am 2026-09-02: `break_reproduced`

**Der Bruch reproduziert sich sofort.** Paar `0` von `10`, wie in H1.0. Er ist
also nicht selten und nicht zufällig, sondern auf dieser Zelle deterministisch
genug, um beim ersten Versuch aufzutreten. `15,2 s` GPU, `99,4 s` Wall.

| | Wert |
| --- | --- |
| erster divergierender Index | `10` |
| Baseline-Token dort | `44505` |
| Kandidaten-Token dort | `3797` |
| Iteration | `9` |
| `j` in der Iteration | **`0`** (freies Token) |
| Entwurf in dieser Iteration | **leer** |

**Der Diskriminator zeigt auf Zweig 1.** Die divergierende Position ist das
ungegatterte freie Token ihrer Iteration, und in dieser Iteration wurde gar kein
Entwurf vorgeschlagen. Kein Gatter war beteiligt; die Auswahl kam aus dem breiten
Forward selbst. **Unter der Annahme, dass die Rekonstruktion der
Iterationsgrenzen korrekt ist** — der Nachspieler ist wohlgeformt geprüft, aber
nicht gegen den Zähler der Engine abgeglichen (siehe Grenze oben) — belastet das
die Akzeptanzlogik nicht.

**Die Ursache ist mit vorhandener Evidenz benennbar, und sie widerlegt meine
eigene Einschätzung von heute Vormittag.** In `experiments/identity_forensics/logit_gap.json`
trägt Position `10` `top1_id = 44505` mit Logit `75,0` und `top2_id = 3797` mit
Logit `74,5` — **exakt die beiden Token, die hier divergieren**, in exakt dieser
Rolle: die Baseline wählt den Top-1, der Kandidat den Top-2.

Der Abstand `0,500` ist **kein großer Abstand**, sondern bei dieser
Logit-Größenordnung **genau ein ULP**. Das muss man nicht über das Zahlenformat
behaupten — es ist aus den Werten derselben Datei **ablesbar**: alle `26`
Logitwerte in `[32, 64)` sind Vielfache von `0,25`, alle `6` in `[64, 128)` sind
Vielfache von `0,5`. Das ist das bf16-Raster (`2^5·2^-7 = 0,25` beziehungsweise
`2^6·2^-7 = 0,5`), abgelesen statt unterstellt. Bei `75,0` gegen `74,5` sind die
beiden Token damit **benachbarte darstellbare Zahlen**; enger geht nicht. Eine
andere Reduktionsreihenfolge — hier ein Forward der Breite `3` statt `1` —
kippt eine Ein-ULP-Entscheidung ohne jede Anomalie. `summary.answer` derselben
Datei sagt dazu bereits `tie_hypothesis_supported`.

**Damit fällt das Argument, mit dem dieser Zweig als unwahrscheinlich geführt
wurde.** Es lautete: „`0,5` in den Logits ist kein knappes Rennen." Bei bf16 und
Logits um `75` ist `0,5` das knappstmögliche Rennen überhaupt. Der Satz stand in
`BACKLOG.md` S3 und in dieser Vorregistrierung und ist hiermit widerlegt — durch
die Messung, nicht durch ein besseres Argument.

**Zwei Einschränkungen, die dazugehören.**

1. `logit_gap.json` wurde mit `677` Prompttoken und `16` generierten Token
   gemessen, diese Zelle mit `897` und `128`. Die Übereinstimmung beider
   Token-IDs an derselben Position ist empirisch stark, aber sie stammt aus
   einer verwandten, nicht identischen Workload. Stufe B misst den Abstand auf
   **dieser** Workload und schließt das.
2. Der Warmup-Vergleich, der dem Nachspieler den gesunden Fall liefern sollte,
   war **nicht gesund**: `warmup_arms_identical = false`. Der Bruch trat also
   schon dort auf. Die Rekonstruktion ist zwar wohlgeformt (`127` Iterationen
   für `128` Token), aber **der Nachspieler hat seine Prüfung gegen einen
   intakten Lauf nie bekommen** — das ist eine offene Lücke, kein erfüllter
   Punkt. Geschlossen wird sie nicht durch den Warmup, sondern durch den
   `acceptance`-Zähler weiter unten, der dieselbe Aussage unabhängig belegt.

**Nebenbefund, der H1.0s Ursachensatz kippt — und er ist nicht rekonstruiert,
sondern vom Zähler der Engine bestätigt.** `127` Iterationen für `128` Token
heißt: kein einziges angenommenes Entwurfstoken. `Engine.generate` gibt
`acceptance` selbst zurück; ein direkter Aufruf braucht kein `Sample` und fasst
den eingefrorenen Kern nicht an (`experiments/identity_break/acceptance.py`).
Gemessen auf derselben Workload, `128` Token:

| `speculate_k` | `acceptance` | Decode-Sekunden |
| --- | --- | --- |
| `1` | **`0,0`** | `2,398` |
| `2` | **`0,0`** | `3,273` |
| `3` | **`0,0`** | `4,054` |

Damit ist die Bedingung geschlossen, unter der die Zweigzuordnung stand: die
Nullannahme ist gemessen, nicht abgeleitet. Der `3`-Gramm-Lookup findet in
diesem Prompt keine Fortsetzung für den generierten Text.

**Die Decode-Zeit skaliert mit der Breite, und das ist die tragende Ursache.**
Bei Nullannahme leistet jede Iteration einen Forward über `k+1` Positionen und
liefert genau **ein** Token. `2,398` → `3,273` → `4,054` Sekunden bei `k = 1, 2, 3`;
die Baseline decodiert dieselben `128` Token in rund `1,9` Sekunden. Der
gemessene Rückstand ist damit **vervielfachte Rechenarbeit ohne Gegenwert**, und
die Readback-Asymmetrie ist der kleinere Rest. Der Ursachensatz in
`PROJECT_STATUS.md` und `ERGEBNISSE.md` wird entsprechend korrigiert.

**Und die Reichweite ändert sich mit.** Der Kill gilt für **diese Workload**, auf
der der Lookup nichts findet — nicht für Spekulation an sich. Das erklärt S1
besser als alles bisher: `journal.txt` ist wiederholungsreich, dort trifft der
Lookup und Spekulation gewinnt `3`–`6 %`; der versiegelte Prompt trifft nie.
Zwei Workloads, zwei Vorzeichen, eine Ursache.

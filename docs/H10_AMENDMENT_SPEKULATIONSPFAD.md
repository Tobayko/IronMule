# Amendment zu H1.0 — misst H1.0 das Verfahren oder eine Implementierung?

**Studien-ID:** `spec-path-amendment-20260902-01` (eigene ID, eigenes
Kill-Kriterium). **Registriert:** 2026-09-02, vor dem Lauf.
**H1.0 bleibt unangetastet:** dessen Regeln, Schwellen und die gemessenen
4B-Ergebnisse stehen wie gemessen.

## Anlass

H1.0 misst auf dem 4B gegen den Auslieferungspfad einen Verlust bei jeder
Breite und jeder Länge, und der Verlust **wächst** mit der Länge (`32` Token
`−12,23 %`, `48` `−15,98 %`, `64` `−19,46 %` bei Breite `1`). S1 misst bei
denselben Längen einen Gewinn.

**Eine frühere Erklärung dieses Widerspruchs war falsch und wird hier
zurückgenommen:** die Wahl des Baselinearms (`combined` statt knobs-off) erklärt
ihn nicht. In `experiments/lookup_order/order.json` ist der Index `0` die
Entwurfstiefe `0`, also ein Arm **ohne** Spekulation mit `speedup = 1,0`. S1
vergleicht Spekulation gegen Nicht-Spekulation innerhalb derselben
Implementierung und misst `1,0488`–`1,0590` bei `96` Token, `all_identical_to_greedy = true`.

## Was die beiden Studien tatsächlich unterscheidet

**Zwei verschiedene Spekulationsimplementierungen im selben Repository**, nicht
zwei Baselines:

| | S1 | H1.0 |
| --- | --- | --- |
| Code | `friday_hardware/speculate.py` (aus `provenance.code_files_sha256`) | `.worktrees/friday-optimizer-ironmule/ironmule/runtime.py` auf `03e884cb` |
| Auswertung je Iteration | `picks = sampler(...).tolist()`, `mx.eval(logits)` — nur die Logits (`speculate.py:223-224`) | `mx.eval(picks, *_leaves(state))` — **der gesamte KV-Cache**, jede Iteration (`runtime.py:485`) |
| Barriere | `mx.synchronize()` **einmal am Ende** (`speculate.py:264-265`) | `mx.synchronize()` **je Iteration** (`runtime.py:486`) |
| KV-Cache | `mlx_lm`-Cache, Rücknahme über `trim_prompt_cache` | kapazitätsfester Cache, Rücknahme über `position.offset` |
| Prompt | `journal.txt`, `749` Token | versiegelt, `897` Token |
| Statistik | `repeats = 3`, Median | `6` Paare, AB/BA, Bootstrap-KI |
| Knöpfe | ohne `head_skip`, ohne `compiled_fixed_cache` | beide an, plus `readback_every = 8` |

Dazu eine Asymmetrie, die H1.0s Zahl mitträgt und benannt gehört: im
Auslieferungspfad bündelt `_decode` den Readback über `readback_every = 8`,
`_decode_speculative` kann das nicht. Der Baselinearm synchronisiert also alle
acht Token, der Spekulationsarm jeden Schritt. Ein Teil des gemessenen Verlusts
ist damit der **Verzicht auf `bundled_readback`**, nicht die Spekulation selbst.

S1 ist methodisch das schwächere Papier — anderer Prompt, drei Wiederholungen,
Median statt Intervall, andere Knöpfe. Schwächer heißt nicht widerlegt, und es
zeigt in die Gegenrichtung.

## Warum H1.0s Kill-Kriterium bis dahin nicht greifen darf

H1.0s Kill schreibt „der (K,N)-Dispatcher wird nicht gebaut" und „H1.1/H1.2
entfallen ersatzlos" nach `PROJECT_STATUS.md`. Mit der obigen Datenlage wäre das
ein Befund über **eine Implementierung**, aufgeschrieben als Befund über **das
Verfahren**. Ein späterer Leser liest „Spekulation verliert auf dieser
Hardware"; das trägt die Evidenz nicht.

## Die Änderung — zwei Zeilen, kein Umbau

In `ironmule/runtime.py:_decode_speculative`:

- `*_leaves(state)` aus dem `mx.eval` streichen,
- `mx.synchronize()` je Iteration entfernen.

`chosen = picks.reshape((-1,)).tolist()` in der Folgezeile erzwingt die
Auswertung, die für die Rückgabe nötig ist, ohnehin. Keine neue Abstraktion,
kein neuer Knopf, kein `readback_every`-Umbau im Spekulationspfad.

Wörtlich, `runtime.py:485-486` wird zu einer Zeile:

```python
-            mx.eval(picks, *_leaves(state))
-            mx.synchronize()
+            mx.eval(picks)
```

**Der Patch ist gepinnt, und der Lauf scheitert ohne ihn.** Protokollieren
allein genügt nicht: ein vergessener Patch würde den unveränderten Pfad messen,
`ironmule_worktree_dirty: false` in eine Datei schreiben, die niemand noch
einmal liest, und ein sauber aussehendes „verliert auch" liefern — also genau
die Evidenz, auf der H1.0s Kill dann ruhen würde. Eine Musterprüfung auf
`mx.synchronize()` oder `*_leaves(state)` leistet das nicht: beide kommen in
derselben Datei fünf weitere Male vor, in `_prefill` und `_decode`. Gepinnt sind
deshalb die Dateihashes:

| Datei | SHA-256 |
| --- | --- |
| `ironmule/runtime.py`, ungepatcht (`03e884cb`) | `9d30965eb7073771f2620fae7cb0cd42d799ca047bb59e4661f82d71b98a9f3b` |
| `ironmule/runtime.py`, gepatcht | `1252f53891800dfa4efecb3cf135523452cf1dedc41296cecb14361302062a9d` |

`experiments/spec_path/measure.py` bricht vor dem ersten GPU-Aufruf ab, wenn der
Hash nicht der gepatchte ist. Das greift in beide Richtungen: ein Lauf ohne
Patch ist unmöglich, und ein versehentlicher Wiederholungslauf **nach** dem
Reset ebenso.

**Ehrlich zum Risiko:** ohne die Auswertung der State-Blätter kann der
Berechnungsgraph über Iterationen wachsen. Bricht der Lauf am Speicher, ist auch
das ein Ergebnis und geht so ins Protokoll. Ein Vorzeichen wird **nicht**
erwartet; die Behauptung ist allein, dass der jetzige Kill diesen Pfad nicht
ausschließen darf.

## Die Messung — eine Zelle, bei `96` Token

- 4B, versiegelter `897`-Token-Prompt, **`96`** generierte Token.
- Breiten `1` und `2` gegen denselben `combined`-Baselinearm wie H1.0.
- Gepaart, abwechselnd AB/BA, Warmup je Arm, Tokenidentität terminal.
- A/A **wiederverwendet** aus `experiments/switch_point/aa_4b_96.json`
  (`0,5227 %`, `6` Paare nach derselben eingefrorenen Regel): gleiches Modell,
  gleiche Länge, gleicher Baselinearm — kein Regimewechsel. Der reparierte
  Kandidat ändert das Rauschen des **Baselinearms** nicht.

**Warum `96` und nicht `64` — entschieden, bevor eine einzige Zahl vom
reparierten Pfad existierte.** H1.0s gemessenes A/A über die vier Längen:

| Länge | `32` | `48` | **`64`** | **`96`** |
| --- | --- | --- | --- | --- |
| A/A | `1,131 %` | `1,040 %` | **`2,217 %`** | **`0,523 %`** |

Drei Gründe, jeder unabhängig vom Ausgang:

1. **Rauschband.** `64` ist der verrauschteste Punkt der Reihe, `96` der
   ruhigste — Faktor vier. Ein besserer Messpunkt löst die Ausreißerfrage, statt
   sie mit einem Zweibänderbericht zu verwalten.
2. **Mechanismus.** Der Defektanteil — die volle Materialisierung des KV-Caches
   je Iteration — skaliert mit der Iterationszahl. Wirkt der Patch, wirkt er bei
   `96` am stärksten. Ein Test gehört dorthin, wo der erwartete Effekt am
   größten ist, nicht dorthin, wo er am kleinsten ist.
3. **Direkter Vergleich.** S1s Hauptdatei `experiments/lookup_order/order.json`
   misst `96` Token (`1,0488`–`1,0590`). Bei `96` stehen beide
   Implementierungen bei derselben Länge nebeneinander.

**Ausgewählt wird ein Messregime nach seiner Rauscheigenschaft, nicht ein
Ergebnis nach seinem Vorzeichen.** Vom reparierten Pfad existiert zum Zeitpunkt
dieser Festlegung keine einzige Messung. Das ist der Unterschied zwischen einem
Amendment und einem nachgeschobenen Freibrief.

Der zuvor hier vorgesehene Zweibänderbericht entfällt damit ersatzlos: bei
`0,523 %` gibt es keine zweite Lesart.
- Budget unverändert: `120 s` GPU, `6 s` ununterbrochen, Deckel wird nicht
  angehoben.

## Provenienz der Abweichung

`friday_calibrate.plan.EXPECTED_IRONMULE_HEAD` bindet den Commit, nicht den
Arbeitsbaum. Die Änderung liegt als unversionierte Arbeitsbaumänderung vor, der
Kopf bleibt `03e884cb` — das wäre eine stille Lücke. Deshalb:

- der Bericht führt `ironmule_worktree_dirty`, den `git status --porcelain` des
  Worktrees und den SHA-256 der gepatchten `runtime.py`,
- der Diff wird als `patch.diff` neben dem Ergebnis abgelegt.

**Reihenfolge, verbindlich und nicht aus dem Kopf:**

1. Patch anwenden.
2. Lauf.
3. Ergebnisbericht schreiben.
4. `patch.diff` schreiben.
5. Beides auf der Platte prüfen (Existenz und Hash gegen den Bericht).
6. **Erst danach** den Arbeitsbaum zurücksetzen.

Ein Abbruch zwischen Lauf und Reset darf nicht genau die Provenienz kosten, für
die dieser Mechanismus da ist. Ein Commit dieser Artefakte ist Sache des
Nutzers; die Dateien liegen dafür vollständig auf der Platte.

## Protokollnotiz zur einen Anomalie der Reihe

`4B`/`64`/Breite `3` scheiterte im **ersten** Anlauf, bevor eine Datei
geschrieben war, und lief im Wiederholungslauf sauber durch
(`max_continuous_gpu_seconds 3,435` gegen die Grenze `6,0`). Der Fehler
wörtlich, damit die Anomalie eine Fußnote bleibt und keine offene Frage:

```
File "experiments/switch_point/measure.py", line 177, in main
    warm(knobs)
File "friday_calibrate/runner.py", line 323, in run
    charge(time.perf_counter() - at)
File "friday_evidence/budget.py", line 84, in record_gpu
    raise BudgetError("continuous GPU work budget exceeded")
friday_evidence.budget.BudgetError: continuous GPU work budget exceeded
```

Der Aufruf war der Warmup, also der erste Aufruf mit dieser Form — dort fällt
das Kompilieren an. Reproduziert hat es sich nicht; ein zweiter Lauf zur Klärung
wird dafür **nicht** ausgegeben.

## Wiederholungsregel — vorab, nicht nach dem ersten Abbruch

Der Traceback oben sagt mehr als „einmalig": `record_gpu` schlug **im Warmup**
zu, beim ersten Aufruf mit dieser Form, und der Wiederholungslauf war sauber.
Der Unterschied zwischen beiden ist der Metal-Shader-Cache, den macOS über
Prozessgrenzen hinweg auf der Platte hält — der erste Lauf einer Form
kompiliert, jeder spätere zieht aus dem Cache. Das ist ein systematischer
Erstlauf-Effekt, und die Vorregistrierung belastet den Warmup ausdrücklich dem
Budget.

Die Amendment-Zelle läuft auf **neuem Code** bei `96` Token, also mit einer Form,
die dieser Cache nie gesehen hat. Dieselbe `BudgetError` ist dort
wahrscheinlicher als bei `64`/w3, nicht unwahrscheinlicher. Deshalb, **vor** dem
Lauf:

- Wiederholt wird **ausschließlich** bei `BudgetError` **im Warmup**, also bevor
  ein einziges Paar gemessen ist.
- **Nie** wiederholt wird ein abgeschlossener Lauf, dessen Zahl nicht gefällt.
  Diese Grenze ist der Zweck der Regel: ohne sie steht am Ende „dritter Versuch,
  diesmal gewonnen", und niemand kann mehr unterscheiden, warum wiederholt wurde.
- Der Bericht führt `attempt` und `attempt_log` mit dem Grund jedes Abbruchs;
  die Versuche überleben den Prozess in
  `experiments/spec_path/amendment_4b_96_attempts.json`.
- **Dreimal** derselbe Warmupabbruch ist kein Kompilierartefakt mehr, sondern
  ein Befund über den gepatchten Pfad. Dann gilt die Zelle als **nicht gemessen**
  (`warmup_budget_error_persistent`), und das steht so im Bericht.

Ausdrücklich **nicht**: den Deckel anheben, den Warmup aus dem Budget nehmen
oder die Vorregistrierung an dieser Stelle lockern. Alle drei wären
Regeländerungen unter Druck.

**Zur Frage, die sich beim Lesen von selbst stellt:** `64`/w3 lief im
Wiederholungslauf mit warmem Shader-Cache, `w1` und `w2` nicht. Das verzerrt die
**gemessenen Paare** nicht. Der Warmup existiert genau dafür, die
Kompilierkosten aus Paar `0` herauszuhalten; dass er im zweiten Versuch billiger
war, ändert an den danach gemessenen Paaren nichts. Die Zahl, die ein warmer
Cache verschiebt, ist die Warmupdauer, und die wird nicht ausgewertet.

## Wie ein Identitätsbruch in dieser Zelle zu lesen ist — vorab

Der Patch streicht `*_leaves(state)` und ändert damit die Auswertungsreihenfolge.
Ob das die Numerik berührt, ist offen, in beide Richtungen. Deshalb vorab, damit
die Zuordnung hinterher keine Auslegungssache ist:

- Ein Identitätsbruch in dieser Zelle ist **kein Geschwindigkeitsbefund**. Die
  Zelle gilt dann als nicht gemessen.
- Er ist auch **kein automatischer Beleg gegen den Patch**. Er ist derselbe
  Korrektheitsbefund wie `4B`/`128`/w2 — mit dem einen Zusatz, der ihn vom
  bisherigen unterscheidet: der gepatchte Pfad zeigt ihn bei `96`, wo der
  ungepatchte ihn nicht gezeigt hat. Dieser Zusatz ist ein Hinweis auf den
  Patch, keine Zuordnung; ihn zu einer zu machen bräuchte den ungepatchten Pfad
  bei `96` mit mehr Paaren, also eine eigene Frage.
- In beiden Fällen greift der Dump aus dem Abschnitt „Beweislast bei
  Identitätsbruch", und der Befund geht in denselben Backlogeintrag.

## Kill-Kriterium dieses Amendments

**Was der Patch nicht entfernt, und warum das die Verzweigung bestimmt.**
`chosen = picks.reshape((-1,)).tolist()` (`runtime.py:487`) bleibt stehen und
erzwingt weiterhin je Iteration eine Synchronisation. Das ist auch nicht
wegzuoptimieren: Spekulation **muss** die Picks je Schritt zurücklesen, sonst
kann sie die Annahme des Entwurfs nicht prüfen. Der Verzicht auf
`bundled_readback` überlebt den Patch. Der Zweizeiler trennt also nicht Defekt
von Struktur — er entfernt den Defektanteil (volle Materialisierung des
KV-Caches plus redundante Barriere) und lässt den strukturellen Anteil stehen.

- **Verliert der reparierte Pfad bei 4B/`64` ebenfalls** (KI vollständig über
  `1 + s`), dann liegt der Rest **nicht** an einer überflüssigen Barriere,
  sondern am strukturell unvermeidbaren Rücklesen je Schritt — gegen einen
  Baselinearm, der über acht Token bündelt. H1.0s Kill greift dann belastbar,
  und zwar mit dieser Ursache, nicht mit „Spekulation ist langsam".
- **Gewinnt er** (KI vollständig unter `1 − s`), war H1.0s Befund die Messung
  eines Defekts. H1.1 und H1.2 leben, und H1.0 wird auf dem reparierten Pfad
  wiederholt, bevor irgendein `K` behauptet wird.
- **Bleibt es unentschieden**, dann heißt das bei `s = 2,2171 %` und `6` Paaren:
  **kein messbarer Vorteil bei 4B/`64` im Auslieferungspfad.** Der Dispatcher
  bleibt ungebaut — das ist ohnehin der Default, ein Nichtbau braucht keinen
  Beweis, nur ein Bau braucht einen. Der Befund wird regimespezifisch
  formuliert, nicht als „Spekulation verliert", und H1.1/H1.2 wandern als **ein**
  Backlogeintrag mit eigenem Kill-Kriterium weiter, statt als offener Zweig im
  Masterplan zu bleiben. Ein Zweig ohne Abbruchbedingung ist ein Wunsch.

Unabhängig vom Ausgang: die Divergenz zweier Spekulationsimplementierungen im
selben Repository geht mit beiden Dateipfaden in `BACKLOG.md`.

## Ergebnis, gemessen am 2026-09-02: `repaired_path_loses`

Beide vorregistrierten Breiten, `6` Paare, Tokenidentität gehalten, `attempt 1`
in beiden Zellen — die Wiederholungsregel wurde nie gebraucht, der erwartete
Warmupabbruch trat nicht ein.

| 4B / `96` Token | ungepatcht (H1.0) | gepatcht (Amendment) | Differenz |
| --- | --- | --- | --- |
| Breite `1` | `−26,46 %` | `−26,16 %` | `0,30` Punkte |
| Breite `2` | `−51,69 %` | `−51,29 %` | `0,40` Punkte |

Intervalle des gepatchten Pfades: `[1,2528; 1,2652]` und `[1,5053; 1,5341]`;
A/A des Regimes `0,52 %`. Beide Läufe unter demselben Messkern
(`friday_calibrate/runner.py` `de2eb778…`) und derselben gepatchten
`ironmule/runtime.py` (`1252f538…`).

**Die Hypothese ist gemessen und falsch.** Das Entfernen der vollen
KV-Materialisierung und der redundanten Barriere bewegt `0,3`–`0,4`
Prozentpunkte bei Intervallbreiten von rund `1,2` bis `2,9` Punkten. Der
Defektanteil ist praktisch gratis; die Divergenz zu S1 erklärt er nicht.

**Damit gilt die erste Verzweigung des Kill-Kriteriums.** Der Rückstand liegt
nicht an einer überflüssigen Barriere, sondern am strukturell unvermeidbaren
Rücklesen je Schritt — gegen einen Baselinearm, der über acht Token bündelt.
Genau diese Asymmetrie fehlt in S1, dessen Nichtspekulationsarm ebenfalls je
Schritt zurückliest; sie ist nach Ausschluss des Defekts die tragende Erklärung
der `30`-Punkte-Lücke.

**Reichweite des Befunds, ausdrücklich.** Der Auslieferungspfad bündelt den
Readback, weil `bundled_readback` dort bleibt (Nutzerentscheidung D4 vom
2026-09-02). Gegen **diesen** Pfad verliert Prompt-Lookup-Spekulation bei jeder
gemessenen Länge und Breite. Wird D4 je revidiert, verliert der Baselinearm
seinen strukturellen Vorteil, und die Frage ist neu zu stellen.

**Nicht berührt:** der Identitätsbruch bei `4B`/`128`/w2 bleibt ein offener
Korrektheitsbefund (`BACKLOG.md` S3). Er verschwindet nicht dadurch, dass der
Dispatcher nicht gebaut wird.

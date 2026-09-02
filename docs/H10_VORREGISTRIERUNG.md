# H1.0 — Vorregistrierung: der Umschaltpunkt der Spekulation

**Studien-ID:** `switch-point-20260902-01`
**Registriert:** 2026-09-02, vor dem ersten Lauf.
**Status bei Registrierung:** keine Messdaten vorhanden.

## Die Frage

Der (K,N)-Dispatcher aus dem Masterplan steht und fällt mit einer Zahl, die es
nicht gibt: **ab welcher Antwortlänge gewinnt Prompt-Lookup-Spekulation gegen
den Auslieferungspfad ohne Spekulation?**

Die Evidenzlage vor dem Lauf:

| Quelle | Regime | Befund |
| --- | --- | --- |
| D5 (`experiments/serve_gain/gain_4b_32_spec.json`) | 4B, `897`/`32` | gegen die Knöpfe-aus-Baseline: `N=1` `+3,86 %`, `N=2` `−6,25 %`, `N=3` `−14,81 %`; `combined` ohne Spekulation liegt bei `+15,61 %`. Spekulation **kostet** hier also gegen `combined`. |
| S1 (`experiments/lookup_order/`) | 4B, `96` Token | Spekulation gewinnt bei jeder Breite `1`–`4` (`1,003`–`1,059`), Breitenwahl im Rauschen. |
| D5 (`experiments/serve_gain/gain_1b_32_spec.json`) | 1B, `897`/`32` | anderes Bild als 4B — der Punkt liegt woanders. |

Zwischen `32` und `96` Token liegt ein Vorzeichenwechsel, der nie gemessen
wurde. H1.0 misst ihn. **Ergebnis ist eine Tabelle gemessener `K` je Modell,
keine Formel und keine Interpolation.**

## Mechanismus, warum es überhaupt einen Umschaltpunkt geben muss

`ironmule/runtime.py:_decode` und `:_decode_speculative` sind zwei verschiedene
Schleifen, nicht dieselbe Schleife mit einem Schalter:

- `_decode` bündelt den Readback (`readback_every = 8`) und synchronisiert
  einmal je acht Token.
- `_decode_speculative` ruft `mx.eval` + `mx.synchronize()` **je Iteration** und
  liest die Auswahl mit `.tolist()` zurück; der Knopf `bundled_readback` ist
  dort wirkungslos.

Spekulation zahlt also je Iteration einen festen Synchronisationsaufwand und
verdient nur über akzeptierte Entwurfstoken zurück. Kurze Antworten haben zu
wenige Iterationen, in denen sich die Lookup-Trefferquote aufbaut. Das ist die
Mechanik, die einen Umschaltpunkt erzwingt, und sie ist der Grund, warum die
Zahl gemessen und nicht aus einem Phasenverhältnis abgeleitet wird.

## Arme

Gemessen wird gegen den **Auslieferungspfad**, nicht gegen die nackte Baseline.
Der Dispatcher schaltet von `combined` auf `combined + Spekulation` um; genau
dieser Schritt ist der zu messende.

- **Baseline (Breite `0`):** `head_skip_prefill = True`, `compiled_fixed_cache = True`,
  `readback_every = 8`.
- **Kandidat (Breite `N`):** Baseline **plus** `speculate_k = N`,
  `speculate_ngram = 3`, für `N ∈ {1, 2, 3}`.

Breite `0` ist damit die Baseline selbst und wird nicht als eigener Arm
gemessen. `bundled_readback` bleibt im Baselinesatz, weil D5 so gemessen hat;
seine Zulässigkeit im Auslieferungspfad hängt an der offenen Nutzerfrage D4 und
wird hier **nicht** mitentschieden.

## Workload

- Versiegelter Prompt, wortgleich aus `friday_calibrate.runner.build_runner`
  (`897` Token unter dem 4B-Tokenizer; auf dem 1B wird die Zahl protokolliert,
  nicht erzwungen).
- Antwortlängen `32`, `48`, `64`, `96`, `128` generierte Token.
- Modelle: `mlx-community/gemma-3-4b-it-4bit`, danach
  `mlx-community/gemma-3-1b-it-4bit`.

`128` ist die obere Grenze und keine Vorliebe: `BudgetPolicy.continuous_gpu_limit_s = 6,0`
deckelt den ununterbrochenen GPU-Block, und D5 hat die Messdecke bei `287`
generierten Token belegt. Die Grenze wird für diese Studie **nicht** angefasst.

## Messverfahren

- Gepaart, abwechselnd `AB`/`BA` je Paar, in einem Prozess, ein geladenes Modell.
- Ein Warmuplauf je Arm vor der Messung, damit die `mx.compile`-Kosten des
  ersten Aufrufs nicht in Paar `0` landen. Der Warmup wird dem Budget belastet,
  aber nicht ausgewertet.
- **Tokenidentität terminal:** `token_sha256` je Paar; die erste Abweichung
  beendet den Lauf und der Arm gilt als gescheitert, nicht als langsam.
- Metrik je Anfrage: `ttft + tokens / decode_tps` (`friday_optimizer.integration.request_seconds`).
- Ratio-Median über die Paare, Bootstrap-Intervall (`2 000` Resamples, Seed `11`)
  aus `friday_optimizer.evaluator`. Nichts Statistisches wird neu implementiert.
- `BudgetGuard` mit den Standardgrenzen, unverändert: `120 s` GPU je Prozess,
  `6 s` ununterbrochen, `4 s` Pflichtpause, `25 %` Duty-Cycle, `20 min` Wall.
  Eine Studie, die nicht in ein Budget passt, läuft in Scheiben — je
  (Modell, Länge, Breite) ein Prozess. Der Deckel wird nicht angehoben.
- Netzbetrieb wird erzwungen (`require_ac_power`), Offline wird erzwungen
  (`enforce_offline`).

## A/A-Rauschen je Regime — eingefroren

Das A/A-Rauschen wird **je (Modell, Antwortlänge)** neu gemessen und **nicht**
aus einem anderen Regime übernommen. Begründung ist gemessen, nicht vermutet:
`0,612 %` (F1), `3,69 %` (D5 4B/32), `2,21 %` (D5 4B/256), `14,25 %` (D5 1B/32)
liegen zu weit auseinander, um übertragbar zu sein.

Ablauf je Regime, zweistufig:

1. A/A mit `6` Paaren (derselbe Arm gegen sich selbst) ergibt die Streuung
   `s = max(|1 − KI_low|, |1 − KI_high|)`.
2. Die Paarzahl der Messarme folgt daraus:
   `pairs = clamp(ceil(6 · (s / 0,03)²), 6, 24)`, aufgerundet auf eine **gerade**
   Zahl. Gerade, weil `_pair_ratios_checked` gleich viele `AB`- wie
   `BA`-Paare verlangt und einen ungeraden Lauf sonst als
   `ab_order_unbalanced` verwirft.

`0,03` ist das eingefrorene **Auflösungsziel**: eine Änderung der Entwurfsbreite
unter `3 %` je Anfrage rechtfertigt keinen Dispatcher im Auslieferungspfad. Der
Faktor `(s/0,03)²` ist die übliche `1/√n`-Skalierung der Intervallbreite. Die
Obergrenze `24` ist eine Kostengrenze: wird sie erreicht, ist das Regime bei
`3 %` Auflösung **nicht** messbar, und genau das wird berichtet — nicht ein
gelockertes Ziel.

## Entscheidungsregel — eingefroren

Je Zelle (Länge `L`, Breite `N`) mit `min_gain = mde = s` des jeweiligen Regimes:

| Verdikt | Bedingung |
| --- | --- |
| `wins` | Bootstrap-KI vollständig unter `1 − s` (`status = qualified`) |
| `loses` | Bootstrap-KI vollständig über `1 + s` (`status = rejected`) |
| `tie` | sonst (`below_threshold` oder `inconclusive`) |
| `identity_break` | Tokenidentität gebrochen — terminal, kein Zeitvergleich |

**`K` je Modell** ist die kleinste gemessene Antwortlänge, bei der mindestens
eine Breite `wins` erreicht. Zusätzlich wird `K` je Breite berichtet.

## Kill-Kriterium

Erreicht keine Breite bei keiner Länge bis `128` Token ein `wins`, dann liegt
der Umschaltpunkt für dieses Modell oberhalb des messbaren Bereichs. Dann gilt:
**der (K,N)-Dispatcher wird nicht gebaut**, `speculate_k` bleibt im
Auslieferungspfad auf `0`, und H1.1/H1.2 entfallen ersatzlos. Das gehört dann so
in `PROJECT_STATUS.md`. Ein Nachrücken der Schwelle, damit der Dispatcher
gerechtfertigt erscheint, ist ausgeschlossen.

Bricht die Tokenidentität in irgendeiner Zelle, ist der Befund ein Defekt in
`_decode_speculative` und keine Geschwindigkeitsaussage; die Studie endet dort
und der Defekt wird als eigener Backlogeintrag geführt.

## Was diese Studie ausdrücklich nicht tut

- Sie aktiviert nichts (`no_activation`), sie promoviert keinen Knopf und sie
  ist keine formale Studienpromotion (`formal_claim = false`).
- Sie leitet keine Zahl aus einem anderen Regime, einem Phasenverhältnis oder
  einer Fremdstudie ab. Fehlt eine Zelle, fehlt sie.
- Sie beantwortet **nicht**, ob eine gestufte Regel „Schritt `1..K` ohne, ab
  `K+1` mit Spekulation" tatsächlich beides schlägt. Der hier gemessene
  Umschaltpunkt ist eine Aussage über **ganze Antwortlängen**. Die gestufte
  Regel aus H1.1 braucht ihre eigene gepaarte Messung gegen `combined`, bevor
  sie behauptet werden darf.


## Nachtrag 2026-09-02 — der 1B-Arm wurde verkleinert gemessen

Registriert war „Modelle: `4B`, danach `1B`" über dieselben fünf Längen und drei
Breiten. Gemessen wurden auf dem `1B` **zwei** Zellen: A/A bei `32` Token und
`32`/Breite `1`.

**Grund, vor dem Lauf festgehalten.** Die 4B-Reihe hat die Frage nach einem
Umschaltpunkt für dieses Modell vollständig beantwortet — kein Vorzeichenwechsel
über fünf Längen und drei Breiten, monoton fallend. Für das `1B` entscheidet
allein, ob der Satz in `PROJECT_STATUS.md` gemessen allgemein gilt oder auf das
4B eingeengt werden muss. `32`/Breite `1` ist dafür der schärfste Test: die
kürzeste Länge und die schmalste Breite, also der Punkt, an dem Spekulation in
der 4B-Reihe am wenigsten schlecht dasteht und eine Vorzeichenumkehr die beste
Chance hat.

**Was dieser Nachtrag nicht tut.** Er ändert keine Schwelle, keine Paarzahlregel
und kein Verdikt. Die verbleibenden 1B-Zellen sind **nicht gemessen** und werden
auch nicht projiziert; was über sie gesagt wird, ist „nicht gemessen".

**Ergebnis.** A/A `11,766 %`, Paarzahl nach der eingefrorenen Regel `92`,
gedeckelt auf `24`. Damit ist das Auflösungsziel `3 %` in diesem Regime
**nicht** erreicht, und das wird berichtet statt gelockert.
`32`/Breite `1`: Ratio-Median `1,1559`, Gewinn `−15,59 %`, KI
`[1,1381; 1,2353]`, `24` Paare, Tokenidentität gehalten, Verdikt `loses` —
das Intervall liegt vollständig **über** `1 + s = 1,1177`, also eine
statistisch bestätigte Verschlechterung selbst bei diesem groben Band.

**Zur Lesbarkeit des Bandes.** Bei `s = 11,766 %` hätte ein `wins` ein Intervall
vollständig unter `0,8823` verlangt, also mehr als rund `12 %` Vorsprung. Ein
Gewinn in der Größenordnung, die S1 auf dem 4B gemessen hat (`3`–`6 %`), wäre in
diesem Regime **nicht** unterscheidbar gewesen. Hier greift das nicht, weil die
Messung eine bestätigte Verschlechterung zeigt und kein `tie`; für künftige
1B-Messungen bleibt die Einschränkung bestehen.

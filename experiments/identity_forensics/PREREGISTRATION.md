# P2 — Vorregistrierung: Tie oder Defekt?

**Study-ID:** `identity-gap-20260902-01`
**Status:** Designvertrag. Der Lauf ist erst nach ausdrücklicher Einzelfreigabe
des Nutzers zulässig; ohne `--execute` verweigert der Worker den Start.

## 1. Frage

Von `11` aufgezeichneten Tokenidentitäts-Abweichungen liegen `10` an
generierter Position `10`, eine bei `20` — über zwei unabhängige Mechanismen,
vier Promptlängen und fünf Blockgrößen (Herleitung:
`divergence_positions.py`, Journal 2026-09-01). Geprüft wird, ob an dieser
Position die beiden besten Kandidatentoken so dicht liegen, dass jede Änderung
der Akkumulationsreihenfolge das `argmax` kippt.

## 2. Was ausdrücklich nicht zur Debatte steht

Das Tokenidentitätsgate. Ein gekipptes `argmax` bleibt ein Identitätsbruch,
unabhängig vom Ausgang dieser Messung. Es wird keine Toleranz, kein
Schwellwert und keine Umdeutung des Gates vorgeschlagen. Die Schwellen unten
klassifizieren eine **Hypothese über eine Messung**, nicht eine Ausgabe.

## 3. Messung

Ein Prozess, ein Modell (`mlx-community/gemma-3-4b-it-4bit`, lokaler
Snapshot, offline). Prompt exakt wie in der Chunk-Identity-Studie
(`677` Token; abweichende Tokenzahl bricht den Lauf ab, weil die sensible
Position dann nicht vergleichbar wäre). `16` greedy Token.

1. Referenzlauf, Prefill in einem Block. Je generierter Position werden
   Top-1- und Top-2-Logit und ihr Abstand aufgezeichnet.
2. Zwei Varianten mit Chunk `128` und `512` — genau die beiden, die in
   `chunk_identity/results.json` bei `677` Token an Position `10` abwichen.
3. Je Variante: erste Abweichung bestimmen, gegen die Referenzabstände
   klassifizieren.

Budget: drei Prefills von `677` Token plus `48` Decodeschritte, deutlich
unter einer 30-Minuten-Freigabe. AC-Pflicht, `BudgetGuard`, Pausenlogik wie in
allen bisherigen Läufen.

## 4. Vorregistrierte Klassifikation (revidiert 2026-09-02)

**Die ursprüngliche Schwelle war falsch und ist zurückgezogen.** Registriert
war „Abstand ≤ `1e-2`". Am 2026-09-02 fand sich
`experiments/divergence_source/results.json` — eine undokumentierte Studie auf
**genau diesem Prompt** (`677` Token, Splits `[677]` gegen `[512, 165]`). Sie
misst:

| Größe | Wert |
| --- | --- |
| erste Schicht mit Unterschied | `0` |
| maximale Logit-Differenz durch Chunking | `1,1875` |
| Top-1/Top-2-Abstand an Position `0` | `1,75` |
| kann der Unterschied die Wahl kippen? | `false` |

Die Störung ist also von der Größenordnung **eins**, nicht von der Größenordnung
Fließkomma-Epsilon. Eine Schwelle bei `1e-2` hätte jede Abweichung als
`structural` eingestuft, unabhängig von der Wahrheit. Sie wird ersatzlos
gestrichen; `TIE_ABSOLUTE` existiert nicht mehr.

Und sie erklärt zugleich, warum Position `0` nie kippt: dort ist der Abstand
`1,75` größer als die Störung `1,1875`.

**Neues Kriterium: zwei gemessene Größen vergleichen.** Der Lauf misst je
Position zusätzlich die maximale absolute Logit-Differenz zwischen Referenz
und Variante. Eine Abweichung gilt als Tie, wenn der lokale Top-2-Abstand
**kleiner ist als die tatsächlich gemessene Störung** — nicht kleiner als eine
gesetzte Zahl.

| Bedingung | Verdikt |
| --- | --- |
| erste Abweichung an Position `0` oder `1` | `structural` |
| Störung nicht gemessen | `inconclusive` |
| Abstand ≤ Störung × `TIE_MARGIN` (`1,0`) | `tie` |
| Abstand > Störung × `TIE_RATIO` (`20,0`) | `structural` |
| dazwischen | `inconclusive` |

`TIE_MARGIN` ist die einzige verbliebene gewählte Zahl und als solche im Code
gekennzeichnet. `TIE_RATIO` bleibt reine Plausibilitätsgrenze.

## 5. Konsequenzen

- `tie_hypothesis_supported`: die Prefill-Hebelklasse (Amdahl-Decke `79,84 %`,
  Backlog P1) ist nicht mechanisch defekt. Die Konsequenz ist, für eine
  Prefill-Studie eine Promptfamilie **ohne** degenerierte Position zu
  registrieren — nicht das Gate zu ändern.
- `tie_hypothesis_rejected`: die Mechanismen sind tatsächlich defekt, P1 wird
  geschlossen, P2 und P1 werden aus dem Backlog gelöscht.
- `inconclusive` oder `no_divergence_reproduced`: gültiger Abschluss ohne
  Folgearbeit; ein nicht reproduzierbarer Effekt ist ein Ergebnis.

`formal_claim=false` in jedem Fall. Kein Performanceclaim entsteht aus diesem
Lauf.

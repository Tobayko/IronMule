# Mini-Vorregistrierung — logsumexp-Normalisierung bei greedy überspringen

**Kandidaten-ID:** `logsumexp-skip-20260824-01`
**Zyklus:** 10 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Beobachtung

`mlx_lm/generate.py`, `generate_step._step` rechnet **unbedingt**:

```python
logprobs = logits - mx.logsumexp(logits, keepdims=True)
```

`make_sampler(temp=0)` ist danach `mx.argmax`. Argmax ist invariant gegenüber dem
Abzug einer Konstante — bei greedy Decoding ohne Logprob-Ausgabe ist die
Normalisierung damit ohne Wirkung auf das Ergebnis.

Sie umfasst eine Reduktion über `262208` Elemente, eine Exponentialfunktion und ein
Ergebnisarray derselben Größe.

Gleiche Form wie Zyklus 8: gerechnet und für das Ergebnis nicht gebraucht.

## Erwartung, vorab benannt

**Ich erwarte einen kleinen Effekt.** Überschlägig sind es rund `1,5` MB Verkehr, bei
`358` GB/s also `0,004` ms gegen `12,1` ms Schrittzeit — unter `0,1 %`. Die
transzendenten Operationen könnten teurer sein als der reine Speicherverkehr, weshalb
die Messung überhaupt lohnt.

Ein Nullergebnis ist hier der wahrscheinliche Ausgang und wird als solches berichtet.

## Hypothesen

**H1 (Korrektheit).** Argmax über den rohen Logits liefert dieselben Token wie Argmax
über den normalisierten.

**H2 (Wirkung).** Die Normalisierung kostet mindestens `2 %` der Schrittzeit.

## Genau geänderte Variable

Ob vor dem Argmax normalisiert wird. Sonst nichts.

## Workload

`128` Decode-Schritte nach `~900`-Token-Prefill, greedy, Batch `1`. Drei
Wiederholungen je Arm, Arme abwechselnd, Median. Aufwärmlauf vorab, verworfen.

## Primärer Endpunkt

Sekunden je Token ohne Normalisierung, relativ zu mit.

## Sekundäre Endpunkte

Tokenidentität; Kosten der Normalisierung isoliert gemessen, also ohne den
Modellaufruf.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im
selben Prozess. Keine nachträgliche Änderung der Schwelle.

## Vorab festgelegte Deutung

| H1 | H2 | Entscheid |
| :--- | :--- | :--- |
| hält | hält | `candidate_recommended_for_preregistration` |
| hält | verfehlt | `candidate_characterized` — korrekt, aber unter der Messschwelle |
| scheitert | egal | `candidate_correctness_failed`, terminal |

Ein Scheitern von H1 wäre bemerkenswert, weil Argmax mathematisch invariant ist. Es
träte nur ein, wenn die Normalisierung selbst die Rangfolge durch Rundung verändert —
was nach Zyklus 9 (`bfloat16`, `8` Mantissenbits) nicht auszuschließen ist.

## Grenze

Zulässig nur, wenn der Aufrufer keine Logprobs braucht. `generate_step` gibt sie
zurück; ein Pfad, der sie überspringt, muss das in seiner Schnittstelle sichtbar
machen.

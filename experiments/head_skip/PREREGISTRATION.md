# Mini-Vorregistrierung — LM-Head beim Prefill überspringen

**Kandidaten-ID:** `prefill-head-skip-20260824-01`
**Zyklus:** 8 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Beobachtung

`gemma3_text.Model.__call__` wendet den LM-Head auf **alle** Positionen an:

```python
out = self.model(inputs, cache, input_embeddings)
out = self.lm_head(out)
```

Beim Prefill wird davon genau **eine** Zeile gelesen — die letzte des letzten Blocks.
Bei einem `256`-Token-Block und `262208` Vokabular sind das
`256 × 2560 × 262208 ≈ 172` GFLOP, von denen `1/256` verwendet wird.

Dies fällt unter Punkt 6 der Kandidatenliste des Auftrags, „Entfernung unnötiger
Kopien", der nach sieben Zyklen ungeprüft war.

## Hypothesen

**H1 (Korrektheit).** Wird der Head beim Prefill nur auf die letzte Position
angewandt, sind die erzeugten Token identisch.

Das ist erwartbar, weil die übersprungenen Logits **nie gelesen** werden — anders als
bei Blockgröße oder Batchbreite ändert sich keine Form einer Rechnung, deren Ergebnis
verwendet wird. Geprüft wird es trotzdem: in dieser Messreihe hat dreimal eine
Änderung, die „nichts ändern konnte", die Ausgabe verändert.

**H2 (Wirkung).** Der Head macht mindestens `10 %` der Prefill-Zeit aus.

## Genau geänderte Variable

Ob der LM-Head auf alle Positionen eines Prefill-Blocks angewandt wird oder nur auf
die letzte des letzten Blocks. Sonst nichts.

## Workload

`~900`-Token-Prompt, Blockgrößen `128`, `256`, `512`. Zusätzlich der Head allein bei
denselben Breiten, um seinen Anteil direkt zu beziffern. `32` Ausgabetoken, greedy.
Zwei Wiederholungen, Median, Aufwärmlauf vorab.

## Primärer Endpunkt

Prefill-Sekunden mit Head auf allen Positionen gegen Head nur auf der letzten.

## Sekundäre Endpunkte

Anteil des Heads am Prefill je Blockgröße; TTFT gesamt; Tokenidentität.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im selben
Prozess. Keine nachträgliche Änderung der Schwelle.

## Vorab festgelegte Deutung

| H1 | H2 | Entscheid |
| :--- | :--- | :--- |
| hält | hält | `candidate_recommended_for_preregistration` |
| hält | verfehlt | `candidate_characterized` — der Head ist nicht der Engpass |
| scheitert | egal | `candidate_correctness_failed`, terminal |

## Einschränkung, vorab benannt

Ein Überspringen ist nur zulässig, solange **niemand** die übersprungenen Logits
braucht. Das gilt für greedy Decoding ohne Logprob-Ausgabe. Sobald Logprobs je
Prompt-Token verlangt werden — etwa für Perplexität oder Bewertung — ist der Kandidat
nicht anwendbar. Diese Grenze gehört in jede spätere Umsetzung.

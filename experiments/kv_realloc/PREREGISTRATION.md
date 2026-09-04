# Mini-Vorregistrierung — KV-Cache-Reallokation im Decode lokalisieren

**Kandidaten-ID:** `kv-cache-realloc-20260824-01`
**Zyklus:** 11 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Auftragsbezug

Schritt 2 verlangt: *„Suche nach KV-Cache-Kopien, Konkatenationen und
Reallokationen. Dokumentiere den Befund mit konkreten Dateien, Funktionen und
Zeilennummern."* Schritt 5, Punkt 6 heißt *„Entfernung unnötiger Kopien"*.
Beides ist bisher unbelegt geblieben. Dieser Zyklus liefert die Zahlen.

## Beobachtung, statisch

`mlx_lm/models/cache.py`. Gemma 3 baut in `gemma3_text.py:247` `make_cache()`
zwei Cachetypen: `KVCache` für Layer mit `i % 6 == 5` (fünf von `34`),
`RotatingKVCache(max_size=1024)` für die übrigen `29`.

Drei Stellen kopieren den **gesamten** bisherigen Cache:

| Datei:Zeile | Funktion | Auslöser |
| :--- | :--- | :--- |
| `cache.py:347` | `KVCache.update_and_fetch` | Offset überschreitet die in `step=256`-Blöcken allokierte Breite |
| `cache.py:484` | `RotatingKVCache._update_in_place` | erster Decodeschritt, solange der Puffer `max_size` noch nicht erreicht hat |
| `cache.py:426` / `cache.py:259` | `_trim` / `_update_concat` | jeder Prefill-Block nach dem ersten |

Alle drei sind `mx.concatenate` über Achse `2`, also ein voller Lese- und
Schreibdurchlauf über Keys und Values aller betroffenen Layer.

## Erwartung, vorab beziffert

Je Position und Tensor: `n_kv_heads(4) × head_dim(256) × 2 B` = `2048` B.
Bei `358,4` GB/s effektiver Bandbreite und dem gewählten Prompt von `765` Token:

| Ereignis | Layer | Verkehr | erwartet |
| :--- | ---: | ---: | ---: |
| `RotatingKVCache`-Wachstum, Decodeschritt `1` | `29` | `273,0` MB | `0,762` ms |
| `KVCache`-Wachstum, Decodeschritt `4` | `5` | `47,2` MB | `0,132` ms |
| Prefill-Block `2` (`mx.concatenate`) | `29` | `121,6` MB | `0,339` ms |
| Prefill-Block `3` (`mx.concatenate`) | `29` | `182,3` MB | `0,507` ms |

Summe im Decode: `0,893` ms auf `48 × 14,3` ms = **`0,13 %`**.

**Ich erwarte, dass H2 verfehlt wird.** Die Kopien sind real und lokalisierbar,
aber zu klein, um den Decode zu tragen.

Ein Unterschied zu Zyklus 10 ist vorab zu benennen: die `logsumexp`-Operation
lag **neben** dem kritischen Pfad und wurde vom Framework überlappt. Eine
Cache-Konkatenation liegt **auf** dem kritischen Pfad — die Attention desselben
Layers liest ihr Ergebnis. Sie ist also nicht wegzuüberlappen. Falls H1 dennoch
scheitert, ist das ein Befund über die Messauflösung, nicht über Überlappung.

## Hypothesen

**H1 (Lokalisierung).** Die Inter-Token-Latenz an den Schritten, an denen eine
Reallokation **beobachtet** wird, liegt mindestens `0,30` ms über dem Median der
Schritte ohne Reallokation. Die Schwelle ist die konservative Hälfte der für das
große Ereignis erwarteten `0,758` ms.

**H2 (Wirkung).** Die Summe aller Reallokationskosten beträgt mindestens `1 %`
der Decodezeit.

**H3 (Korrektheit).** Die erzeugten Token-IDs sind über alle Wiederholungen
identisch.

## Genau geänderte Variable

Keine. Dies ist eine reine Messung am unveränderten Referenzpfad. Es gibt keinen
B-Arm und damit kein A/B — der Auftrag verlangt einen Kandidaten je Zyklus, nicht
zwingend einen Eingriff.

## Unveränderte Variablen

Modell, Revision, Tokenizer, Chat-Template, Prompt, Blockgröße `256`,
`temperature=0`, Ausgabelänge, Cacheklassen, Prozess.

## Workload

Prompt `765` Token, `48` Ausgabetoken, greedy, Batch `1`, Prefill in Blöcken zu
`256`. Acht Wiederholungen in einem Prozess nach einem verworfenen Aufwärmlauf.

Die Promptlänge ist so gewählt, dass **beide** Reallokationsklassen früh und
innerhalb von `48` Schritten fallen: `765` liegt knapp unter der `768`-Grenze des
`KVCache` (Ereignis bei Schritt `4`) und unter der `1024`-Grenze des
`RotatingKVCache` (Ereignis bei Schritt `1`). Das Werkzeug bricht ab, wenn die
tatsächliche Promptlänge nicht in `[757, 767]` liegt.

## Instrument

Je Decodeschritt: Zeitstempel, sowie `keys.shape[2]` **jedes** der `34` Caches.
Eine Formänderung gegenüber dem Vorschritt ist eine beobachtete Reallokation.
Vorhergesagte Positionen werden erst **nach** der Messung mit den beobachteten
verglichen.

Das Ablesen der Form ist eine Python-Eigenschaft und löst keine Synchronisation
aus. Die Zeitmessung je Schritt erfordert dagegen eine Synchronisation, die nach
Zyklus 6 `2,199` ms kostet. Diese Kosten sind über alle Schritte **konstant** und
verschieben deshalb die Differenz zwischen Schritten nicht.

## Primärer Endpunkt

Differenz der Inter-Token-Latenz zwischen Schritten mit und ohne beobachtete
Reallokation, in Millisekunden, über acht Wiederholungen gemittelt.

## Sekundäre Endpunkte

`p50`, `p95`, `p99` der Inter-Token-Latenz — bisher in der Baseline nicht
erfasst, vom Auftrag aber unter den verbindlichen Metriken verlangt.
Tokenidentität. Beobachtete gegen vorhergesagte Reallokationspositionen.
Gemessenes Rauschband je Schrittindex.

## Timing- und Speichergrenzen

`BudgetGuard`-Policy, Duty-Faktor `0,15` (strenger als die geforderten `0,25`).
Erwartete GPU-Arbeit rund `20` s. Netzbetrieb Pflicht.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im
selben Prozess. Keine nachträgliche Schwellenänderung. Kein Verwerfen von
Ausreißern.

## Vorab festgelegte Deutung

| H3 | H1 | H2 | Entscheid |
| :--- | :--- | :--- | :--- |
| hält | hält | hält | `candidate_recommended_for_preregistration` |
| hält | hält | verfehlt | `candidate_characterized` — Mechanismus belegt, Wirkung zu klein |
| hält | scheitert | verfehlt | `candidate_characterized` — unter der Auflösung des Instruments |
| scheitert | egal | egal | `correctness_failed`, terminal |

## Grenze

Die Messung gilt für Prompt `765`, Blockgröße `256`, Batch `1`. Bei längeren
Prompts wächst die je Ereignis kopierte Menge linear, die Zahl der Ereignisse
aber nur logarithmisch in der Prompthöhe. Eine Übertragung auf andere Längen ist
Rechnung, nicht Messung, und wird als solche gekennzeichnet.

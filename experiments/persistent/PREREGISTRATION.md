# Mini-Vorregistrierung — persistenter Prozess, Korrektheit und Wirkung

**Kandidaten-ID:** `persistent-process-20260824-02`
**Zyklus:** 5 · **Status:** vor der Messung geschrieben · `formal_claim=false`
**Vorgänger:** `persistent-process-20260824-01` (Zyklus 3) endete
`candidate_characterized`. Er wird nicht fortgesetzt, sondern durch diesen
Kandidaten mit korrigierter Schwelle und einem Korrektheitsgate ersetzt.

## Korrektur der Schwelle aus Zyklus 3

Die frühere Vorregistrierung verlangte, **Modellladen plus Warm-up** müssten `30 %`
der `cold_process`-TTFT erreichen. Das bildet den Kandidaten falsch ab: ein
persistenter Prozess entfällt auch die **Importe**, die mit `1,881` s den größten
Einzelposten stellen. Die Schwelle wurde damals knapp verfehlt (`28,7 %`), und ich
habe das als Mangel der Vorregistrierung ausgewiesen statt sie umzudeuten.

Diese Vorregistrierung fasst die Schwelle korrekt: **entfernt ein persistenter Prozess
mindestens `50 %` der `cold_process`-TTFT?** Gemessen werden dafür Interpreterstart,
Importe, Snapshot-Auflösung, Modellladen und Warm-up gemeinsam — genau das, was ein
persistenter Prozess einmal statt je Anfrage zahlt.

## Das eigentlich Offene ist die Korrektheit

Die Wirkung ist aus Zyklus 3 bereits absehbar. Ungeprüft ist etwas anderes:

**Liefert ein warmer Prozess dieselben Token wie ein frischer?**

Nach vier Zyklen, in denen dreimal eine formabhängige Numerik die Ausgabe still
veränderte, ist das keine rhetorische Frage. Ein persistenter Prozess trägt
Speicherlage, Allokatorzustand und aufgebaute Kernel über Anfragen hinweg.

## Hypothesen

**H1 (Korrektheit).** Ein warmer Prozess erzeugt für denselben Prompt dieselben Token
wie ein frischer, auch nach zwischenzeitlich bedienten anderen Anfragen.

**H2 (Wirkung).** Ein persistenter Prozess entfernt mindestens `50 %` der
`cold_process`-TTFT.

Beide Gegenhypothesen sind gleichwertig verwertbar. Scheitert H1, endet der Kandidat
`candidate_correctness_failed` — und dann ist **kein** Kandidat dieses Auftrags mehr
übrig, der den Vertrag erfüllen kann.

## Aufbau

**Arm A, kalt.** Frischer Prozess, Modell laden, Prompt `P` beantworten, beenden.

**Arm B, warm.** Ein Prozess: Modell laden, dann in dieser Reihenfolge
`P`, `Q`, `P`, `R`, `P`. Die drei `P`-Antworten werden gegen Arm A geprüft.

Die eingeschobenen `Q` und `R` prüfen Zustandsverschleppung: ein persistenter Dienst
bedient nicht dieselbe Anfrage hintereinander.

Jede Anfrage nutzt einen **frischen** KV-Cache. Präfix-Wiederverwendung ist in
Zyklus 1 gescheitert und hier ausdrücklich nicht enthalten.

## Workload

Drei verschiedene Prompts zu je `~900` Token, `32` Ausgabetoken, greedy, `temp=0`.

## Primärer Endpunkt

Tokenidentität der drei warmen `P`-Antworten gegen die kalte. Binär, keine Toleranz.

## Sekundäre Endpunkte

TTFT je Anfrage in Arm B; Anteil der in Arm A einmalig gezahlten Kosten; RSS-Verlauf
über die fünf Anfragen als Hinweis auf Speicherzuwachs.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im selben
Prozess. Kein Nachziehen weiterer Anfragen nach Sicht der Ergebnisse.

## Vorab festgelegte Deutung

| H1 | H2 | Entscheid |
| :--- | :--- | :--- |
| hält | hält | `candidate_recommended_for_preregistration` |
| hält | verfehlt | `candidate_characterized` — korrekt, aber zu klein |
| scheitert | egal | `candidate_correctness_failed`, terminal |

Wird nach Sicht der Zahlen nicht geändert.

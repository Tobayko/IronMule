# Mini-Vorregistrierung — Wo entsteht die Zerteilungs-Divergenz?

**Kandidaten-ID:** `divergence-source-20260824-01`
**Zyklus:** 9 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Warum

Zyklus 2 zeigte, dass die Prefill-Zerteilung in rund `26 %` der Fälle andere Token
erzeugt, und Zyklus 4, dass diese Abweichungen die **Antwort** ändern. Beide Befunde
stehen, aber **keiner nennt die Ursache**. „Sporadisch" ist eine Beschreibung, keine
Erklärung.

Der Auftrag verlangt für jede Kerneloptimierung den vorherigen Nachweis eines
Kernelengpasses. Diese Diagnose ist genau dieser Nachweis — oder sein Ausschluss.

Sie ist **keine Optimierung** und schlägt keine vor.

## Frage

Bei welcher Schicht und welcher Größe entsteht der erste Unterschied, wenn dieselben
Token einmal als ein Block und einmal zerteilt durch den Prefill laufen?

## Hypothesen

**H1.** Der Unterschied entsteht bereits im KV-Cache, also vor dem LM-Head, und
wächst über die Schichten hinweg an.

**H2.** Er beginnt in einer bestimmten Schicht und nicht in allen gleichzeitig.

Trifft H2 nicht zu — differieren also alle Schichten von Beginn an gleichmäßig —,
liegt es an der Eingangsverarbeitung und nicht an einer einzelnen Operation.

## Aufbau

Derselbe Prompt (`677` Token, aus Zyklus 2 als abweichend bekannt bei `512`+`165`),
zweimal durch den Prefill:

- Arm A: ein Block
- Arm B: Blöcke `512` und `165`

Danach werden die KV-Caches beider Arme **schichtweise** verglichen: maximaler
absoluter Unterschied und relativer Unterschied je Schicht, getrennt für `keys` und
`values`.

Zusätzlich der finale Hidden State und die Logits der letzten Position, weil dort die
Entscheidung fällt.

## Primärer Endpunkt

Index der ersten Schicht mit einem Unterschied über `0` sowie die Größenordnung des
Unterschieds je Schicht.

## Sekundäre Endpunkte

Verlauf über die Schichten (wächst er, bleibt er konstant, klingt er ab); Unterschied
in den Logits der letzten Position; Abstand zwischen den beiden führenden Logits, um
einzuordnen, ob ein kleiner Unterschied überhaupt eine andere Wahl bewirken kann.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im selben
Prozess.

## Vorab festgelegte Deutung

| Befund | Folge |
| :--- | :--- |
| Unterschied beginnt in einer Schicht und wächst | Mechanismus lokalisiert; ein gezielter Kandidat wäre begründbar |
| Unterschied ab Schicht 0 überall | Eingangsverarbeitung; kein einzelner Kernel als Ziel |
| Unterschied unterhalb der Logit-Abstände | Divergenz wäre Rundung, nicht Struktur — widerspräche Zyklus 1, wo der Abstand `0,344` betrug |

Diese Deutung wird nach Sicht der Zahlen nicht geändert. Ein lokalisierter Mechanismus
**begründet** einen Kandidaten, er ist selbst keiner.

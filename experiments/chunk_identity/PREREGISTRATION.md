# Mini-Vorregistrierung — Blockgrößen-Identität beim Prefill

**Kandidaten-ID:** `chunk-identity-20260824-01`
**Zyklus:** 2 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Hypothese

Es existiert mindestens eine Prefill-Blockgröße, unter der ein in mehrere Blöcke
zerteiltes Prefill **tokenidentisch** zu einem Einzelblock bleibt, und zwar über
verschiedene Promptlängen hinweg.

Gegenhypothese, gleichwertig verwertbar: es existiert keine solche Größe, oder sie
hängt von der Gesamtlänge beziehungsweise vom Modell ab.

## Genau geänderte Variable

Die **Blockgröße** des Prefills. Nichts sonst.

## Unveränderte Variablen

Modell und Revision, Quantisierung, Tokenizer, Chat-Template, Prompt-Token,
Sampling (`greedy`, `temp=0`), Ausgabelänge, Stop-Token, Gerät, Prozess.

## Workload

| Größe | Wert |
| :--- | :--- |
| Promptlängen | `~300`, `~680`, `~1200`, `~2000` Token |
| Blockgrößen | `64`, `128`, `256`, `512`, `1024`, Einzelblock |
| Ausgabe | `16` Token, greedy |
| Referenz | derselbe Prompt als **ein** Block, gleicher Prozess |

`16` Ausgabetoken, weil die in Zyklus 1 beobachteten Abweichungen an Position `10`
und `20` auftraten. Eine kürzere Ausgabe könnte eine Abweichung übersehen.

## Primärer Endpunkt

**Tokenidentität** je Zelle `(Promptlänge, Blockgröße)` gegen den Einzelblock.
Binär. Keine Toleranz, keine Ähnlichkeitsmetrik.

## Sekundäre Endpunkte

Position der ersten Abweichung; Zahl der Blöcke; Prefill-Sekunden je Zelle.

## Mindestwirkung

Nicht anwendbar — der Endpunkt ist binär. Eine Blockgröße gilt als **identitätserhaltend**,
wenn sie in **allen** geprüften Promptlängen identisch bleibt. Ein einziger
Fehlschlag genügt zum Ausschluss.

## Aufbau

Ein Prozess, alle Zellen im selben geladenen Modell, damit Ladezustand und
Speicherlage nicht mit der Blockgröße konfundieren. Reihenfolge: Referenz zuerst je
Länge, dann Blockgrößen aufsteigend. Kein Wiederholen einer Zelle.

## Timing- und Ressourcengrenzen

`BudgetGuard` mit der Projektpolicy. Jeder Prefill-Block und jede Decode-Phase wird
**getrennt** verbucht — beides als einen Block zu zählen meldet eine kontinuierliche
Last, die es nie gab.

## Abbruchregeln

- Budgetverletzung → Lauf endet, Teilergebnis wird persistiert;
- ein Fehler wird nicht im selben Prozess wiederholt;
- kein Nachziehen zusätzlicher Zellen nach Sicht der Ergebnisse.

## Auswertung

Matrix `Promptlänge × Blockgröße` mit booleschem Ergebnis. Zusätzlich geprüft, ob die
abweichenden Blockgrößen mit den bereits bekannten Kernel-Regressionen der
Breitenkurve (`6`–`9`, `48`) oder mit Zweierpotenzen zusammenfallen.

## Vorab festgelegte Deutung

| Ergebnis | Deutung | Folge |
| :--- | :--- | :--- |
| eine Größe hält überall | harte Architekturregel | vier blockierte Kandidaten wieder zugänglich |
| keine hält | Tokenidentität über verschiedene Zerteilungen ist auf dieser Plattform nicht herstellbar | Korrektheitsvertrag braucht eine Nutzerentscheidung |
| hängt von der Länge oder dem Modell ab | gehört ins Profil | wie Breite, Sample-Zahl und Nachschlagfenster zuvor |

Keine dieser Deutungen wird nach Sicht der Zahlen geändert.

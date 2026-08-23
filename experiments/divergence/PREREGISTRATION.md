# Mini-Vorregistrierung — Wirkt sich die Zerteilungs-Divergenz auf die Antwort aus?

**Kandidaten-ID:** `divergence-impact-20260824-01`
**Zyklus:** 4 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Warum diese Frage

Zyklus 2 hat gezeigt, dass die Prefill-Zerteilung in rund `26 %` der geprüften
Konfigurationen andere Token erzeugt, und dass keine Blockgröße das zuverlässig
verhindert. Daraus folgte eine Entscheidung, die dem Nutzer vorgelegt wurde: den
Korrektheitsvertrag halten und vier Kandidaten dauerhaft sperren, oder ihn
präzisieren.

Diese Entscheidung wurde bisher **ohne Daten** vorgelegt. Die einzige beobachtete
Abweichung war ein Satzzeichen (`:` gegen `.`). Ob das repräsentativ ist, ist offen.

Dieser Zyklus **lockert den Vertrag nicht** und schlägt das auch nicht vor. Er
beschafft die Zahl, die für die Entscheidung fehlt.

## Hypothese

Die durch Zerteilung entstehenden Abweichungen sind überwiegend **oberflächlich** —
sie ändern Formulierung, nicht die inhaltliche Antwort.

Gegenhypothese, gleichwertig verwertbar: sie ändern die Antwort. Dann ist die
Vertragsfrage entschieden, und zwar gegen jede Lockerung.

## Warum das prüfbar ist

Als Aufgaben dienen die bereits im Projekt vorhandenen, maschinell erzeugten
Rechenaufgaben mit **eindeutiger ganzzahliger Lösung**
(`tools/measure_self_consistency.py`, `hard_problems`). Ihre Antwort ist ohne Modell
prüfbar. Ein langer, für alle Aufgaben identischer Vorspann bringt den Prompt über die
Länge, ab der Zerteilung überhaupt greift.

## Genau geänderte Variable

Die **Blockgröße** des Prefills. Alles andere identisch: Modell, Tokenizer, Template,
Prompt-Token, greedy, Ausgabelänge, Gerät, Prozess.

## Primärer Endpunkt

Bei denjenigen Aufgaben, deren Token **abweichen**: stimmt die extrahierte
`ANSWER:`-Zahl beider Arme überein?

## Sekundäre Endpunkte

Anteil abweichender Aufgaben; Position der ersten Abweichung; Anteil abweichender
Token an der Gesamtausgabe; Fälle, in denen ein Arm eine Antwort liefert und der
andere nicht.

## Workload

| Größe | Wert |
| :--- | :--- |
| Aufgaben | `10`, fester Seed, schwere Familie |
| Vorspann | identisch, bringt den Prompt auf `~900` Token |
| Arme | Einzelblock (Referenz) gegen Blockgröße `512` |
| Ausgabe | `160` Token, greedy |

Blockgröße `512` gewählt, weil sie in Zyklus 2 bei zwei von drei geprüften Längen
abwich — sie erzeugt also zuverlässig Fälle, ohne dass danach gesucht werden müsste.

## Mindestwirkung

Nicht anwendbar; der Endpunkt ist ein Anteil. Vorab festgelegte Deutung:

| Ergebnis bei abweichenden Aufgaben | Deutung |
| :--- | :--- |
| Antwortzahl überall gleich | Abweichungen sind oberflächlich; die Vertragsfrage ist eine Abwägung |
| Antwortzahl weicht in Einzelfällen ab | Vertrag lockern hieße Fehler in Kauf nehmen, deren Rate erst zu bestimmen wäre |
| Antwortzahl weicht häufig ab | Vertrag halten; jede Lockerung ist ausgeschlossen |

Diese Deutung wird nach Sicht der Zahlen nicht geändert.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Keine
Wiederholung im selben Prozess. Kein Nachziehen weiterer Aufgaben nach Sicht der
Ergebnisse.

## Was dieser Zyklus ausdrücklich nicht tut

Er empfiehlt **keine** Lockerung des Korrektheitsvertrags, unabhängig vom Ergebnis.
Eine Antwortgleichheit wäre **kein** Ersatz für Tokenidentität, sondern eine
zusätzliche Beobachtung. Der Auftrag verbietet Qualitätsmetriken als Ersatz für
Tokenidentität, und dieser Zyklus hält sich daran.

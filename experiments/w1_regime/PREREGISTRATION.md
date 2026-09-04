# W1 — Vorregistrierung: hält die Decode-Rate über eine realistische Antwort?

**Study-ID:** `w1-regime-20260902-01`
**Status:** Designvertrag. Der Lauf ist erst nach ausdrücklicher
Einzelfreigabe zulässig; ohne `--execute` verweigert der Worker den Start.

## 1. Frage

Die gesamte Regimerechnung (Journal 2026-09-02) beruht auf einer Annahme, die
für dieses Modell bei dieser Promptlänge nie gemessen wurde: dass der
Decode-Durchsatz ungefähr konstant bleibt, während der KV-Cache wächst. Jede
Optimierungsstudie dieses Projekts erzeugte `32` Token oder weniger.

## 2. Warum die Frage entscheidet

Bei `256` generierten Token liegen die beiden bestätigten Kandidaten
rechnerisch nur noch `0,38` Prozentpunkte auseinander:

| Antwortlänge | `head_skip` | `fixed_compiled` | führend |
| --- | --- | --- | --- |
| `32` | `12,26 %` | `1,42 %` | `head_skip` |
| `128` | `7,64 %` | `3,54 %` | `head_skip` |
| `256` | `5,09 %` | `4,71 %` | `head_skip` |
| `512` | `3,05 %` | `5,64 %` | `fixed_compiled` |

Die Kandidaten-Kreuzung liegt bei `276` Token. Ein Abfall der Decode-Rate um
`10 %` genügt, um die Reihenfolge bereits bei `256` zu kippen:

| Ratenabfall | `head_skip` | `fixed_compiled` | führend |
| --- | --- | --- | --- |
| `0 %` (`70,99` tok/s) | `5,09 %` | `4,71 %` | `head_skip` |
| `10 %` (`63,89` tok/s) | `4,73 %` | `4,87 %` | **`fixed_compiled`** |
| `20 %` (`56,79` tok/s) | `4,36 %` | `5,04 %` | **`fixed_compiled`** |

Die Vorhersage ist also empfindlich genau in dem Bereich, den niemand gemessen
hat. Deshalb wird gemessen statt angenommen.

## 2b. Vorwissen aus vorhandener Evidenz (ergänzt 2026-09-02)

Bei der Suche nach undokumentierten Messungen fanden sich zwei Punkte, die
W1s Grundannahme betreffen — und sie widerlegen sie teilweise:

| Studie | Kontext | Batch-1-Rate |
| --- | --- | --- |
| `decode_width` | `256` | `82,44` tok/s |
| `persistent_process` | `897` | `70,99` tok/s |

Die Decode-Rate ist also **nicht konstant**, sondern fällt mit dem Kontext:
`−0,01786` tok/s je Kontexttoken, `−13,9 %` über `641` Token. Zwei Punkte aus
zwei Studien mit unterschiedlichen Definitionen und Prompts — ein Vorwissen,
keine Messung.

**Folge für die Rechnung.** Der Kreuzungspunkt der Kandidaten verschiebt sich
von `276` auf `267` generierte Token; bei `256` Token schrumpft der Abstand
zwischen `head_skip` (`4,97 %`) und `fixed_compiled` (`4,76 %`) von `0,38` auf
`0,21` Prozentpunkte. Die Frage wird dadurch **schärfer**, nicht stumpfer.

**Folge für das Verdikt.** Ein Rückgang, der lediglich der Kontexterwartung
entspricht, ist keine Anomalie. Der Lauf hält deshalb neben der gemessenen
Änderung auch die aus dem Vorwissen erwartete fest (`−3,22 %` bei `256`
generierten Token). Die Schwellen bleiben unverändert; ergänzt wird nur die
Vergleichsgröße, damit „stabil" nicht mit „kein Rückgang" verwechselt wird.

Dass der Worker die Rate im ersten und letzten Viertel getrennt aufzeichnet,
war ursprünglich Vorsicht. Es ist jetzt der Kern der Messung: die Rate fällt
*während* der Generierung, und nur der Verlauf zeigt, ob sie so fällt wie
erwartet.

## 3. Messung

Ein Prozess, ein Modell (lokaler Snapshot, offline), ein Prompt von rund
`900` Token. Nach einem Warmlauf zwei Anfragen: `32` Token als Kontrolle,
die die bekannte Rate reproduzieren muss, und `256` Token. Je Anfrage werden
TTFT, Decode-Dauer, Gesamtrate sowie die Rate im ersten und im letzten Viertel
der Schritte aufgezeichnet — der Verlauf, nicht nur der Mittelwert.

Budget: drei Prefills und rund `290` Decodeschritte, deutlich unter einer
30-Minuten-Freigabe. AC-Pflicht, `BudgetGuard`, Pausenlogik wie in allen
bisherigen Läufen.

## 4. Vorregistrierte Klassifikation

Festgelegt vor der Messung in `regime_analysis.py`, Toleranz `10 %` gegen die
Kontrolle:

| Bedingung | Verdikt |
| --- | --- |
| Rate innerhalb `±10 %` der Kontrolle | `rate_stable` |
| Rate mehr als `10 %` darunter | `rate_degrades` |
| Rate mehr als `10 %` darüber | `rate_improves` |

Aus der gemessenen Rate wird anschließend deterministisch die Rangfolge der
beiden Kandidaten bei `256` Token berechnet.

## 5. Konsequenzen

- `rate_stable`: das Modell trägt, die Priorisierung wird längenabhängig
  geführt, und die Kreuzungspunkte `203` (Decken) und `276` (Kandidaten)
  gelten als Planungsgrößen.
- `rate_degrades`: das Modell unterschätzt den Decode-Anteil bei langen
  Antworten; die Kreuzungspunkte wandern nach unten und werden aus der
  gemessenen Rate neu berechnet. Die Decode-Klasse ist dann früher führend als
  bisher angenommen.
- `rate_improves`: unerwartet; der Befund wird als eigener Journaleintrag
  festgehalten und die Rechnung neu aufgesetzt.

## 6. Was diese Studie nicht ist

Keine A/B-Messung und kein Performanceclaim: es wird nur die Baseline bei zwei
Antwortlängen im selben Prozess vermessen. `formal_claim=false`. Die Studie
begründet keine Aktivierung und keine Promotion; sie richtet ausschließlich
die Priorisierung im Backlog aus.

## 7. Nebenbefund, der F1 betrifft

Aus derselben Rechnung: F1s kombinierter warmer Arm ist bei `32` Token
`13,68 %` wert, bei `256` Token nur noch `9,80 %` — **unterhalb der für F1
vorregistrierten Schwelle von `10 %`**. F1 bleibt damit richtig, aber
ausdrücklich auf das kurze Antwortregime beschränkt; die Zahl darf nicht als
allgemeines Ergebnis gelesen werden. Der Hinweis steht auch in F1s eigener
Vorregistrierung.

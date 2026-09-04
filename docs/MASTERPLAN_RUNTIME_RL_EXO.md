# Masterplan „Friday Runtime, RL und exo-Cluster" — korrigierte Fassung

**Angelegt:** 2026-09-02. Ersetzt die ursprüngliche Planfassung dort, wo diese
Zahlen behauptet, die nicht auf dieser Hardware gemessen wurden.

## Rahmen — die bindende Regel dieser Fassung

**Jede Zahl in einem Bericht stammt aus einem Lauf auf dieser Maschine mit einem
echten Modell.** Keine Projektion aus Phasenverhältnissen fremder Studien, keine
Hochrechnung, keine aus Fremdzahlen abgeleitete Erwartung. Fehlt eine Zahl, wird
sie gemessen oder sie fehlt — sie wird nicht geschätzt. Rückwirkend gilt: eine
bestehende projizierte Zahl wird als Projektion gekennzeichnet oder durch eine
Messung ersetzt.

Reale Modellläufe brauchen keine Einzelbestätigung (`AGENTS.md`,
Hardwarefreigabe vom 2026-09-02). Die Messhygiene bleibt vollständig: gepaart,
Warmup, Wiederholungen, Median, Streuung, Vorregistrierung, eingefrorene
Schwelle, Tokenidentität als terminales Gate.

**Keine Bearbeitung gehashter Dateien, solange ein Lauf in der Luft ist.**
`tools/_bench.study_provenance` hasht die Codedateien beim **Schreiben** des
Berichts, nicht beim Start des Laufs. Wer eine dieser Dateien während eines
laufenden Laufs ändert, erzeugt damit eine stille Falschangabe in der
Studienakte: der Bericht führt den neuen Hash, gemessen hat der alte Code. Das
gilt für Code **und** für die Vorregistrierung, die mit hineingehasht wird.
Gefunden am 2026-09-02 beim Amendment; die Regel ist allgemein, nicht auf diesen
Fall beschränkt. Sie gehört sachlich in `AGENTS.md` zur Messhygiene — das ist
eine Nutzerentscheidung und steht bis dahin hier.

**A/A-Rauschen wird je Modell und Regime neu gemessen und nie übernommen.**
Gemessene Belege für die Nichtübertragbarkeit, alle vom 2026-09-02:

| Regime | A/A-Streuung | Quelle |
| --- | --- | --- |
| 4B, warmer Arm, F1 | `0,612 %` | `docs/ERGEBNISSE.md` |
| 4B `897`/`32`, Baseline ohne Knöpfe | `3,69 %` | `experiments/serve_gain/gain_4b_32.json` |
| 4B `897`/`256`, Baseline ohne Knöpfe | `2,21 %` | `experiments/serve_gain/aa_4b_256.json` |
| 4B `897`/`32`, **combined-Arm** | `1,13 %` | `experiments/switch_point/aa_4b_32.json` |
| 4B `897`/`48`, combined-Arm | `1,04 %` | `experiments/switch_point/aa_4b_48.json` |
| 4B `897`/`64`, combined-Arm | `2,22 %` | `experiments/switch_point/aa_4b_64.json` |
| 4B `897`/`96`, combined-Arm | `0,52 %` | `experiments/switch_point/aa_4b_96.json` |
| 4B `897`/`128`, combined-Arm | `0,70 %` | `experiments/switch_point/aa_4b_128.json` |
| 1B `897`/`32`, Baseline ohne Knöpfe | `14,25 %` | `experiments/serve_gain/aa_1b_32.json` |
| 1B `897`/`32`, **combined-Arm** | `11,77 %` | `experiments/switch_point/aa_1b_32.json` |

Das Rauschen hängt nicht nur am Modell und an der Antwortlänge, sondern **auch am
gemessenen Arm**. Derselbe Prompt, dieselbe Länge, dasselbe Modell, derselbe Tag:
`3,69 %` mit Knöpfen aus, `1,13 %` mit Knöpfen an; auf dem 1B `14,25 %` gegen
`11,77 %`.

Zweitens ist die Schätzung selbst verrauscht: auf **demselben** Arm springt sie
zwischen benachbarten Längen um das Vierfache (`64`: `2,22 %`, `96`: `0,52 %`),
ohne Trend. Bei sechs Paaren ist das die plausibelste Lesart. Wer eine dieser
Zahlen als Schwelle einfriert, friert eine Schätzung mit großer eigener Streuung
ein — das ist zulässig, wenn es vorher hingeschrieben wird, und unzulässig, wenn
es hinterher nachjustiert wird.

Drittens hat die Paarzahlregel `clamp(ceil(6·(s/0,03)²), 6, 24)` in sechs
Regimen nur ihre **Randwerte** geliefert: fünfmal die Untergrenze `6`, einmal
den Deckel `24` (1B). Nie ein Zwischenwert. Weiterbehandlung als `BACKLOG.md`
P3; die Ersatzgrenze ist vorregistrierungspflichtig und darf **nicht** aus
diesen sechs Punkten kalibriert werden.

## Horizont 1

### H1.0 — gemessen am 2026-09-02: es gibt keinen Umschaltpunkt

Vorregistrierung: [`H10_VORREGISTRIERUNG.md`](H10_VORREGISTRIERUNG.md), Amendment
[`H10_AMENDMENT_SPEKULATIONSPFAD.md`](H10_AMENDMENT_SPEKULATIONSPFAD.md),
Rohdaten `experiments/switch_point/` und `experiments/spec_path/`, Ergebnis in
[`ERGEBNISSE.md`](ERGEBNISSE.md) und `PROJECT_STATUS.md`.

Gegen den Auslieferungspfad verliert Spekulation auf dem 4B bei jeder gemessenen
Länge (`32`–`128`) und jeder Breite (`1`–`3`), monoton wachsend, alle Intervalle
weit außerhalb des jeweiligen A/A-Rauschens. Der Verdacht, das läge an einer
überflüssigen Barriere im IronMule-Spekulationspfad, ist gemessen und widerlegt:
`0,30` und `0,40` Prozentpunkte bei Intervallbreiten von `1,2` bis `2,9`.

**Die Ursache ist gemessen und sie ist workloadbedingt, nicht verfahrensbedingt.**
Auf der versiegelten Auslieferungsworkload nimmt der `3`-Gramm-Lookup **kein
einziges Entwurfstoken** an: `acceptance = 0,0` bei `k = 1, 2, 3`, direkt aus
dem Zähler von `Engine.generate`. Spekulation zahlt dort also vollen Zusatz­aufwand
für null Gegenwert. Decode-Zeiten aus demselben Direktaufruf, `128` Token:
`2,398` / `3,273` / `4,054 s` gegen rund `1,9 s` der Baseline — die Kosten
wachsen **linear in `k`**, rund `0,8 s` je Entwurfsplatz.

Das ist ausdrücklich **keine** Vervielfachung mit der Breite: ein Forward der
Breite `4` kostet das `2,13`-fache, nicht das Vierfache, weil Decode
bandbreitengebunden ist und die Gewichte einmal statt `k`-mal gestreamt werden.
Der Hebel des Verfahrens ist also intakt — die Workload liefert ihm nur nichts.
Wer den Befund als „Spekulation taugt nicht" liest, liest ihn falsch.

Korroboration über zwei unabhängige Wege: `2,398 / 1,9 = 1,262` gegen den
gepaarten `ratio_median 1,2616` bei `96`/w1. `1,9` ist gerundet, die
Übereinstimmung wird deshalb nicht auf die dritte Stelle behauptet.

Die Readback-Asymmetrie — Spekulation muss je Schritt zurücklesen, der
Baselinearm bündelt über acht Token (`bundled_readback`, Nutzerentscheidung D4
vom 2026-09-02) — bleibt richtig, ist aber der **kleinere Rest**. Bis zum
Abend des 2026-09-02 stand sie hier als tragende Ursache; das war unvollständig
und ist durch S3 ersetzt.

**Das 1B ist damit nicht erledigt.** Der erste Entwurf dieses Abschnitts
erklärte den 1B-Teil für gegenstandslos, mit der Begründung, die strukturelle
Ursache gelte dort verschärft, weil dieselbe Zahl Rücklesevorgänge auf eine
kürzere Anfrage trifft. Das war eine **Projektion** und damit genau das, was der
Rahmen dieses Dokuments verbietet — dieselbe Regel, mit der S1 zu Recht nicht
übertragen wurde. Der Satz ist gestrichen; er hatte keinen Beweiswert.

Gemessen wurde stattdessen die schärfste Stelle des 1B — `32` Token, Breite `1`,
der Punkt, an dem Spekulation in der 4B-Reihe am wenigsten schlecht dasteht und
eine Vorzeichenumkehr die beste Chance hat. Ergebnis: `−15,59 %`, KI
`[1,1381; 1,2353]` bei einem A/A von `11,77 %` — bestätigte Verschlechterung.
Der Kill ist damit **gemessen** statt behauptet, auf beiden Modellen. Die
übrigen 1B-Zellen bleiben ungemessen und werden nicht fortgeschrieben.

### H1.1 und H1.2 — entfallen

Beide bauten auf einem `K`, das gemessen nicht existiert — auf dem 4B über die
volle Reihe, auf dem 1B an der schärfsten Stelle. Die gestufte
Spekulation ab Schritt `K+1` (H1.1) und die deterministische `(K,N)`-Tabelle
(H1.2) werden nicht gebaut, `speculate_k` bleibt im Auslieferungspfad `0`.

Was ausdrücklich **nicht** wiederkommt: der Bandit und das Propensity-Logging —
die R1b-Sackgasse ist unabhängig von H1.0 geschlossen.

**Zurückgenommen am 2026-09-02:** hier stand, die Wiederholungsraten-Berechnung
in `friday_serve/speculation.py` sei „unabhängig davon brauchbar und bleibt im
Baum". Das ist widerlegt. Der Schätzer entscheidet über die **Unigrammrate** des
Prompts, während der Lookup **Trigramme** sucht. Auf dem versiegelten Prompt ist
die Unigrammrate hoch (vielfach wiederholte Floskel), die Trigrammrate null — er
sagt „lohnt sich" genau dort voraus, wo gemessen nie etwas angenommen wird. Das
ist kein Kalibrierfehler, sondern die falsche Statistik. Weiterbehandlung als
`BACKLOG.md` S4; dessen Kill lautet **entfernen statt kalibrieren**.

**Offen und unberührt vom Kill:** der Identitätsbruch bei `4B`/`128`/Breite `2`.
Ursache seit 2026-09-02 gemessen (`BACKLOG.md` S3, `docs/S3_VORREGISTRIERUNG.md`):
reproduziert sofort in Paar `0`, erster divergierender Index `10`, ungegattertes
Token (`j = 0`, Entwurf leer) — also der breite Forward selbst. Die beiden Logits
sind `75,0` und `74,5`; bei bf16 ist das **exakt ein ULP**, benachbarte
darstellbare Zahlen. Belegt statt behauptet: in
`experiments/identity_forensics/logit_gap.json` sind alle Werte in `[32,64)`
Vielfache von `0,25` und alle in `[64,128)` Vielfache von `0,5` — das bf16-Raster
ist aus den Zahlen ablesbar. Ein Forward der Breite `3` statt `1` kippt eine
Ein-ULP-Entscheidung ohne jede Anomalie.

Damit ist die Tokenidentität von `_decode_speculative` **nicht** garantiert, wie
die Konstruktion behauptet. `friday_serve/speculation.py` stützt seine Freigabe
ohne Geräteprofil ausdrücklich auf diese Behauptung — das ist der Grund, warum
S3 und S4 nicht mit dem Dispatcher gestorben sind.

### H1.3 — `BudgetGuard` gehört nicht in den Serving-Pfad

Der ursprüngliche Plan wollte `BudgetGuard` in `Server.generate` ziehen, mit
`≥ 4 s` Pflichtpause und `≤ 25 %` Duty-Cycle. Das ist ein Kategorienfehler.
Diese Grenzen sind Messhygiene: sie sorgen dafür, dass eine Zahl etwas bedeutet.
Im Produktivpfad hießen sie: wer zwei Fragen hintereinander stellt, wartet vier
Sekunden vor der zweiten, und die GPU darf ein Viertel der Zeit laufen. Das
macht das Produkt langsamer als die Baseline, also das Gegenteil des Ziels.

**Stattdessen:**

- `BudgetGuard` bleibt unverändert im Mess- und Kalibrierpfad.
- Der Serving-Pfad bekommt **nichts**, solange nicht gemessen ist, dass
  überhaupt gedrosselt wird: Dauerlast über mehrere Minuten, Token/s über die
  Zeit protokolliert. Ohne diesen Beleg wird nichts eingebaut.
- Zeigt die Messung eine Drosselung, bekommt Serving einen eigenen, schwächeren
  Schutz mit dem Zweck Stabilität statt Messgüte: beobachten statt drosseln,
  eingreifen nur bei messbarer Drosselung. Thermik im Normalbetrieb ist die
  Aufgabe von macOS.

## Horizont 2

### H2.0 — KV-Anteil messen, bevor eine FP8-Studie geplant wird

Der Plan behauptet, FP8-KV verdopple die effektive Decode-Bandbreite. Das kann
so nicht stimmen: je Token werden `3,400 GB` Gewichte gelesen, und Gemma 3
deckelt die meisten Schichten über ein Sliding Window. Der reale KV-Anteil wird
**am laufenden Modell bei unseren Kontexten gemessen**, nicht gerechnet. Liegt
er im niedrigen einstelligen Prozentbereich, ist die Studie ihre Kosten nicht
wert, und genau das ist dann der Befund.

### H2.1 — FP8-KV braucht eine Nutzerentscheidung, die nicht vorliegt

Der Nutzer hat am 2026-09-02 ausdrücklich strikte Tokenidentität gewählt und ein
Qualitätsgate ausgeschlossen. FP8-KV kehrt das um. Nicht starten, bevor das
entschieden ist; die Entscheidung wird vorgelegt, sobald H2.0 die Größenordnung
liefert.

### H2.2 / H2.3 — Ringpuffer (Kandidat 21), asynchroner Readback (Kandidat 17)

Beide vorregistrierungsempfohlen, beide können laufen, wenn Messzeit frei ist.
Erwartung klein halten: der verwandte `bundled_readback` lieferte in D5
end-to-end `1,14 %` (`experiments/serve_gain/gain_4b_32.json`).

## Horizont 3 — ehrlich benannt, nicht gestrichen

**Tensor-Parallelismus macht ein Modell, das auf eine Maschine passt, nicht
schneller.** Jede Schicht kostet einen Netzwerk-Hop. Er ermöglicht Modelle, die
auf keine einzelne Maschine passen. Horizont 3 ist damit eine
**Fähigkeitserweiterung, keine Performanceverbesserung**, und wird so
dokumentiert.

Zwei Zahlen des Plans sind offene Fragen, keine Annahmen:

1. **„RDMA over Thunderbolt, `< 1 %` Overhead".** macOS bietet Thunderbolt
   Bridge, kein RDMA im InfiniBand-Sinn; MLX-Distributed läuft über Sockets.
   Der Overhead ist zu messen, sobald Zweit-Hardware da ist, nicht zu behaupten.
2. **„`80` auf `> 250` Token/s bei `5`–`10` parallelen Anfragen".** Korrigiert
   gegen die Messung in `experiments/decode_width/report.json` (4B, Kontext
   `256`, eine Maschine):

   | Batch | Token/s | gegen Batch 1 |
   | --- | --- | --- |
   | `1` | `82,44` | `1,00` |
   | `2` | `103,29` | `1,25` |
   | `4` | `129,78` | `1,57` |
   | `8` | `103,25` | `1,25` — **Regression gegen Batch 4** |
   | `16` | `199,33` | `2,42` |
   | `32` | `383,44` | `4,65` |

   Bei `5`–`10` parallelen Anfragen sind damit rund `103`–`199` Token/s zu
   erwarten, nicht `> 250`. Die Zahl ist außerdem eine Batching-Messung auf
   **einer** Maschine und sagt über verteiltes Serving nichts aus.

### Compliance

Der Mechanismus ist richtig: hashverkettete Provenienz ohne Promptinhalt.

**Nicht schreiben:** „100 % DSGVO-konform". Das ist keine Eigenschaft einer
Runtime, sondern hängt an Rechtsgrundlage, DSFA, Betroffenenrechten und
Löschfristen.

**Belegbar schreiben:** „vollständig lokal, keine Datenübermittlung,
auditierbare Provenienz."

Der Prompthash entfällt oder wird gesalzen: der Hash eines erratbaren Prompts
kann selbst personenbezogen sein.

## Reihenfolge — Stand Abend 2026-09-02

**Erledigt und nicht wieder aufzumachen:** `H1.0` beantwortet (4B vollständig,
1B an der schärfsten Stelle), `H1.1` und `H1.2` entfallen, `S3` gemessen und
ursächlich geklärt, `D4` und `D4b` beim Nutzer entschieden. Die **Decode-Klasse
ist erschöpft** — der letzte offene Decode-Kandidat hat heute sein Urteil
bekommen.

**Als Nächstes, in dieser Reihenfolge:**

1. **`D4b` ausführen** — Nutzerentscheidung vom 2026-09-02: die schwächere
   Serving-Latte gilt **nicht** allgemein. `friday_calibrate.profile.KnobVerdict`
   wird auf die Promotionsschwelle gehoben, `bundled_readback` bleibt als
   **benannte Einzelentscheidung** bestehen.
   ⚠️ **Das ist keine Schwellenänderung, sondern eine Datenänderung.**
   `KnobVerdict.__post_init__` **validiert**: nach dem Anheben wirft derselbe
   Konstruktor bei jedem gespeicherten Profil, dessen Knopf unter der alten
   Latte `verified` war. `bundled_readback` selbst ist so ein Fall
   (`0,9581`, KI `[0,95347; 0,95989]`) — die Ausnahme muss **im Code** kodiert
   sein, sonst ist die Nutzerentscheidung nicht ausführbar.
   Davor: alle vier Knöpfe aus `CALIBRATED_KNOBS` (`head_skip`,
   `fixed_compiled`, `prefill_step_size`, `bundled_readback`) mit Verhältnis,
   Intervall und Verdikt unter **beiden** Latten auflisten. Kippt dabei ein
   Knopf, der nicht `bundled_readback` heißt, ist das eine **neue
   Nutzerfrage** — vorlegen, nicht entscheiden.
2. **`S4`** — der Trefferschätzer in `friday_serve/speculation.py` misst die
   falsche Statistik (Unigramm statt Trigramm). Gate: gemessene `acceptance`
   gegen beide Raten über drei Promptfamilien. Kill: **entfernen statt
   kalibrieren**. Das ist Aufräumen eines widerlegten Codepfads, keine
   Optimierung — es steht hier oben, weil `speculation.py` seine Freigabe ohne
   Geräteprofil auf eine widerlegte Identitätsbehauptung stützt.
3. **`P1` / `D3` — Prefill.** Ab hier liegt die verbleibende Leistung. Die
   Decode-Klasse ist gemessen erschöpft; `D3` (Profilerlauf Prefill) ist der
   einzige verbleibende Weg zu `20 %` auf dem 4B.
4. **`H1.3`** — Driftmessung im erlaubten Duty-Cycle. Dauerlast über mehrere
   Minuten widerspricht der Kontinuitätsgrenze und bleibt **Nutzerentscheidung**;
   ohne Beleg einer Drosselung wird in den Serving-Pfad nichts eingebaut.
5. **`H2.0`** — KV-Anteil messen, bevor eine FP8-Studie geplant wird.
6. `H2.2` / `H2.3`, wenn Messzeit frei ist. Horizont 3 erst mit Zweit-Hardware.

## Für den nächsten Agenten — was zuerst zu lesen ist

1. Der **Rahmen** oben. Er ist bindend, und drei seiner Regeln sind heute
   teuer erkauft worden: keine Projektionen, kein Bearbeiten gehashter Dateien
   während eines Laufs, A/A je Regime neu.
2. `PROJECT_STATUS.md` für den Bestand, `ERGEBNISSE.md` für die Zahlen,
   `BACKLOG.md` für das, was offen ist.
3. Vorregistrierungen vor den Rohdaten: `H10_VORREGISTRIERUNG.md`,
   `H10_AMENDMENT_SPEKULATIONSPFAD.md`, `S3_VORREGISTRIERUNG.md`.

**Vier Sackgassen, die nicht noch einmal gelaufen werden müssen** — jede
gemessen, jede mit Datei:

- Spekulation gegen den Auslieferungspfad, beide Modelle, `speculate_k = 0`
  (`experiments/switch_point/`).
- Der Verdacht, es liege an einer überflüssigen Barriere im Spekulationspfad:
  `0,3`–`0,4` Punkte, widerlegt (`experiments/spec_path/`).
- Der Bandit / Thompson Sampling über Entwurfsbreiten (R1b, geschlossen).
- FP8-KV ohne Nutzerentscheidung (H2.1).

## Was daneben offen bleibt

- Das A/A-Rauschen selbst: sechs Regime, Spanne `0,52 %` bis `14,25 %`, und die
  Schätzung springt auf demselben Arm zwischen benachbarten Längen um das
  Vierfache. Siehe die Tabelle im Rahmen und `BACKLOG.md` P3.
- Die `287`-Token-Messdecke durch `BudgetPolicy.continuous_gpu_limit_s = 6,0`.
- Die Divergenz zweier Spekulationsimplementierungen im selben Baum —
  `friday_hardware/speculate.py` (gewinnt `3`–`6 %` auf `journal.txt`) gegen
  `ironmule/runtime.py` (verliert `26 %` auf dem versiegelten Prompt). Nach S3
  ist die tragende Erklärung die **Trefferquote der Workload**, nicht die
  Implementierung; `BACKLOG.md` S2 trägt das Gate.
- Der Nachspieler der Iterationsgrenzen ist gegen den Engine-Zähler nur für
  `acceptance` geprüft, nicht für die Grenzen selbst — der Warmup, der ihm den
  gesunden Fall liefern sollte, war selbst gebrochen.
- Der Rückbau der zwölf `dashboard.py`, sobald `status` ihre Inhalte trägt (U1).

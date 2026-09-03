# Backlog — Project Friday

Diese Datei enthält nur offene Arbeiten. Jeder Eintrag benennt Mechanismus,
Voraussetzungen, messbare Gates und ein Abbruch- oder Pivotkriterium. Erledigte
Einträge werden entfernt; Ergebnisse und verworfene Wege wandern in
`docs/ARBEITSJOURNAL.md`, `PROJECT_STATUS.md` oder die jeweilige Studienakte.

## C1 — Reste aus dem Codex-Review vom 2026-09-03

Die kritischen Defekte sind behoben (`docs/ARBEITSJOURNAL.md`, Eintrag 2026-09-03).
Offen bleiben vier eng umrissene Punkte:

1. **`friday_serve/speculation.py` entfernen (S4).** Nur noch von
   `experiments/speculation_bandit/replay.py` und `tests/test_speculation.py`
   referenziert, im Serving-Pfad tot (`knobs_for()` kann `speculate_k` nicht
   emittieren). Kill-Kriterium S4 verlangt Entfernung statt Kalibrierung.
   *Gate:* Suite grün nach Löschung von `speculation.py`, `model_speculation.py`,
   `tools/bench_draft_speculation.py` und den zugehörigen Tests/Experimenten.
2. **`h01.sqlite3` ohne Kettentest.** Steht in der `known`-Allowlist von
   `tests/test_sealed_evidence.py`, hat aber keinen `RECORD_CHAINS`-Eintrag und
   keinen dedizierten `verified_records()`-Test — vorbestehend, nicht vom
   Gemini-Branch. *Gate:* eigener Test analog `test_the_device_profile_chain_still_verifies`.
3. **Radix-Cache-Speicherbudget zählt Trie-Tokens, nicht KV-Bytes.**
   `RadixCache._check_eviction` hat einen `ponytail:`-Kommentar mit dem Ceiling.
   *Kill:* nur angehen, wenn Eviction unter realem Speicherdruck nachweislich
   falsch liegt (`tensor.nbytes`-Buchhaltung).
4. **Batcher-Admission ohne `_check_marker`/`guard` nach dem Lauf.**
   `Server.plan_request` gatet die Knopfauswahl vor der Session; der Batcher prüft
   die angewendeten Knöpfe bei `_admit_session` und latcht den Breaker bei einem
   Optimierungspfad-Fehler. Was fehlt: eine `Server`-Methode, durch die die
   Batcher-Admission komplett wie `Server.generate` läuft (post-hoc
   Marker-Verifikation je Session, `guard`-Kontext). *Gate:* der Batcher-Pfad
   nutzt dieselbe `guard`/`_check_marker`-Kette wie der Single-Flight-Pfad.

## F1 — abgeschlossen am 2026-09-02

**Warmer Arm gemessen und bestanden:** `13,99 %` end-to-end, Ratio-Median
`0,8600567`, KI `[0,853444; 0,873056]`, `6` Paare, Tokenidentität `6/6`,
A/A-Rauschen `0,612 %`, Status `qualified` gegen die Schwelle `10 %`. Ergebnis in
`docs/ERGEBNISSE.md`.

**Kalter Arm entfällt.** Er maß den persistenten Modellprozess mit. `friday_serve`
hält den Prozess, also gibt es den kalten Fall im Auslieferungspfad nicht mehr —
die Frage ist gegenstandslos, nicht offen. Keine Hardwarezeit dafür.

## P1 — Prefill-Hebelklasse; die Decode-Klasse ist erschöpft

**Status:** offen, direkt nach F1. Herleitung im Arbeitsjournal unter
„2026-09-01 — F1" (Amdahl) und „2026-09-02 — Roofline je Phase" (Physik);
reproduzierbar mit `experiments/roofline/phase_roofline.py`.

**Befundlage.** Für die registrierte Workload (Prompt `897`, `32` generierte
Token) ist Prefill `79,84 %` der Anfrage. Die Roofline sagt dazu:

| Phase | Auslastung | end-to-end realistisch verfügbar | bereits gehoben |
| --- | --- | --- | --- |
| Decode | `60,3 %` der Bandbreite | `5,85 %` | `1,42 %` (`fixed_compiled`) |
| Prefill | `45,5 %` der Rechenspitze | `37,11 %` | `12,26 %` (Head-Skip) |

**Konsequenz 1 — Decode ist zu, aber nur unterhalb von rund `200` Token.**
Für die registrierte Workload bleiben allen künftigen Decode-Kandidaten
zusammen rund `3,65` Prozentpunkte end-to-end (korrigiert am 2026-09-02: die
beiden gemessenen Decode-Knobs sind gestapelt, nicht alternativ — Ratio
`0,9296 × 0,9581 = 0,8906`, also `2,20 %` end-to-end statt `1,42 %`). Ein Decode-Kandidat kann sein
eigenes Decode-Gate noch bestehen, aber die End-to-End-Schwelle von F1
(`10 %` warm) grundsätzlich nicht mehr erreichen. **Diese Aussage endet bei
`203` generierten Token**: darüber führt die Decode-Klasse (siehe W1). Neue
Decode-Kandidaten werden nicht vorregistriert, solange die Workload kurz
bleibt — für eine lange Antwort gilt das Gegenteil.

**Konsequenz 2 — Prefill hat zwei Mechaniken, nicht eine.**

1. **Blockstruktur** — **am 2026-09-02 durch P2 wieder geöffnet.** Der Lauf
   bestätigte `tie_hypothesis_supported`: Position `10` trägt mit `0,500` den
   kleinsten Top-2-Abstand der Sequenz (Median `4,0`), die Störung durch
   Chunking beträgt dort `2,25`–`2,50`. Kandidat 1 und 2 scheiterten an einer
   Workload mit degenerierter Position, nicht an ihrem Mechanismus.
   Voraussetzung für die nächste Studie: eine Promptfamilie **ohne**
   degenerierte Position registrieren — vorab prüfbar, indem der
   Top-2-Abstandsverlauf gegen die erwartete Störung gehalten wird. Kandidat 5
   (Prefill-Step-Size-Sweep) ist der offene Hebel darin.
2. **Rechenauslastung** — `45,5 %` der Spitze ist für eine compute-gebundene
   Phase niedrig. Verdächtig sind Dequantisierungsaufwand, fehlende Fusion und
   Tiling. Neu aus der Roofline-Rechnung, noch ohne Profilerbeleg; ein
   Profilerbeleg ist nach `AGENTS.md` Voraussetzung, bevor hier
   Kernelarbeit überhaupt vorgeschlagen werden darf.

**Korrektur am 2026-09-02 zu Kandidat 5 (Prefill-Schrittweite).** Er ist auf dem
Auslieferungspfad **kein Hebel**. `ironmule/runtime.py:_prefill` fährt den
ganzen Prompt in einem Forward, solange kein Prefix-Cache gesetzt ist, und F1
setzt keinen — die Baseline liegt bereits am schnellen Ende der gemessenen
Kurve. Dazu misst `experiments/decode_width/measure_prefill.py:39-43` jede
Chunkgröße genau einmal, ohne Wiederholung, in aufsteigender Reihenfolge in
einem Prozess; die beobachtete Monotonie ist genau die Form, die W1 als
Aufwärmdrift nachgewiesen hat. **Am 2026-09-02 terminal geschlossen.** Voraussetzung für eine neue Studie war
eine Promptfamilie **ohne** degenerierte Position, vorab offline prüfbar. Der
Vorfilter `experiments/prefill_step_size/screen.py` läuft über die gesamte
vorhandene Gap-Evidenz — `logit_gap.json` plus beide Replikationen — und meldet
dreimal reproduzierbar `degenerate` mit **acht von sechzehn** Positionen unter
der Schwelle, kleinster Abstand `0,500`. Es gibt genau eine gemessene
Promptfamilie, und sie erfüllt die Voraussetzung nicht. Damit greift das
eingetragene Kill-Kriterium: die Blockstruktur-Klasse bleibt geschlossen, die
kurzen Tasks bleiben bei `13,99 %`, und es wird dafür keine Hardwarezeit
ausgegeben. Herleitung in `docs/AUSBAUPLAN_UMSETZUNG_2026-09-02.md` B4.

**Gate:** wie üblich — vorregistrierte Schwelle über MDE, exakte
Tokenidentität, gepaarte Sessions.

**Kill/Pivot:** findet sich keine längenunabhängig identitätserhaltende
Blockstruktur und zeigt der Profiler keinen adressierbaren Prefill-Engpass,
bleibt die Klasse geschlossen. Ab `203` generierten Token kippt die Rechnung
zugunsten der Decode-Klasse; diese Priorisierung gilt ausdrücklich nur für die
registrierte kurze Antwort.

## W1 — beantwortet am 2026-09-02, Rest steht in P1

**Gemessen.** Zwei gegatete Läufe, je `66 s`. Kontrolle `32` Token
`63,83` tok/s, Langlauf `256` Token `72,36` tok/s. Verdikt `rate_improves`,
gemessene Änderung `+13,36 %` gegen vorhergesagte `−3,58 %`.

**Das Vorwissen war falsch.** Die aus zwei Studienpunkten abgeleitete
Kontextabnahme (`−0,01786` tok/s je Token) ist widerlegt; dominierend ist
Aufwärmen, nicht Kontextwachstum. Innerhalb des langen Laufs steigt die Rate
weiter (`68,34` auf `77,23`), der stationäre Zustand ist bei `256` Token noch
nicht erreicht.

**Priorisierung bleibt.** Kreuzungspunkt `271` statt `267` Token; bei `256`
führt `head_skip` mit `5,02 %` gegen `4,74 %`. Der kombinierte Gewinn `9,76 %`
liegt unter F1s `10 %`-Schwelle — F1 bleibt damit auf das kurze Antwortregime
beschränkt, wie in seiner Vorregistrierung festgehalten.

**Offen bleibt nur**, ob das Zielregime dieses Projekts kurz oder lang ist.
Das ist eine Produktentscheidung, keine Messfrage, und gehört zu P1.
Herleitung im Arbeitsjournal unter „2026-09-02 — W1 gelaufen".

## G1 — Ist `max_load_1m = 0.75` die richtige Grenze? (neu 2026-09-02)

**Status:** offen, Entscheidung des Nutzers. Betrifft **jeden** künftigen
gegateten Pfad über `ReadinessGate`, nicht nur F1.

**Befund.** `ReadinessPolicy.max_load_1m = 0.75` wird in `readiness.py:297`
gegen die rohe Ein-Minuten-Last verglichen; eine Kernzahl kommt in
`readiness.py` nicht vor. Auf dieser 10-Kern-Maschine bedeutet das eine
Auslastung von `7,5 %`. Beste je gemessene Last über zwei bewusste Anläufe:
`1,614` (Q2, 2026-08-30). Im Ruhezustand mit geschlossenen Anwendungen:
`4,0`–`6,0`. Die Grenze wurde nie erreicht und ist auf einem macOS-Desktop mit
angemeldetem Nutzer praktisch nicht erreichbar.

**Was zu klären ist:** ob `0,75` als absolute Last gemeint war oder als Last
je Kern (dann wären `7,5` das Äquivalent), und ob die CPU-Grenze `35 %` — die
in beiden Q2-Anläufen bis zur dritten Probe unterschritten wurde — die
eigentlich tragende Größe ist.

**Ausdrücklich nicht:** die Grenze senken, damit eine Studie läuft. Diese
Entscheidung steht auf ihren Sachgründen oder gar nicht. F1 weicht ihr
stattdessen aus, indem es den Standalone-Pfad nutzt.

**Kill:** ergibt die Prüfung, dass `0,75` absolut gemeint und sachlich richtig
ist, bleibt sie, und jeder `ReadinessGate`-Pfad gilt auf diesem Gerät als
dauerhaft blockiert — dann gehört das so in `PROJECT_STATUS.md`.

## R1b — geschlossen am 2026-09-02: Kill-Kriterium erfüllt

**Die Frage war,** ob ein realer Messpfad existiert, dessen Kandidat sachlich
offen ist — ohne einen solchen erntet `epsilon_greedy`-Logging im Regelbetrieb
nichts, und R2s Kampagne bleibt der einzige Korpusweg.

**Die Entwurfsbreite war der Kandidat, und S1 hat sie geschlossen.** Gemessen
gewinnt Spekulation bei jeder Breite `1`–`4`, und der Abstand zwischen den
besten liegt mit `0,016` im Rauschen. Ein Kandidat, dessen Alternativen sich
nicht messbar unterscheiden, erzeugt Explorationsdaten ohne Informationsgehalt.
Ohne Bandit gibt es zudem keinen stochastischen Entscheidungsstrom.

**Damit greift das eingetragene Kill-Kriterium:** es findet sich kein Pfad mit
sachlich offenem Kandidaten, Epsilon-Logging im Regelbetrieb entfällt, und
**R2s Kampagne (`40` Blöcke, rund zwanzig Stunden gegatete Messzeit) bleibt der
einzige Korpusweg.** Das gehört so in `PROJECT_STATUS.md`.

## S1 — beantwortet am 2026-09-02: der Bandit hat keine Rechtfertigung

**Gemessen.** Sweeps über Breiten `0..4` auf `journal` und `tests` (4B,
`ngram 3`), dazu `experiments/lookup_order/` mit `3` Wiederholungen je Arm in
beiden Messrichtungen, bei `64` und `96` Token. Zwölf Messungen je Breite,
Tokenidentität durchgehend.

**Spekulation gewinnt auf `journal` bei jeder Breite, in beiden Richtungen, bei
beiden Antwortlängen** (`1,003`–`1,059`). Auf `tests` ebenso (`1,021`–`1,056`).
Die beste Breite wechselt zwischen `1` und `3` bei einem Abstand von `0,016` —
die Wahl zwischen den Breiten liegt im Rauschen, das Ob nicht.

**Die Prämisse des Banditen fällt damit.** Sie lautete „der Gewinn liegt im
Abschalten" und stützte sich auf `journal 0,992` und `tests 0,974` aus
`experiments/prompt_lookup/real/results.json`. Diese Verluste reproduzieren
nicht. Es gibt nichts zuverlässig abzuschalten; eine feste Entwurfsbreite im
Bereich `1`–`3` ist das, was die Evidenz trägt. `friday_serve/speculation.py`
bleibt im Baum, wird aber nicht als Standard verdrahtet.

**Nebenbefund, ebenfalls ein Negativergebnis.** Der Verdacht, die
Einzelschussmessung in `tools/measure_prompt_lookup.py:196-201` (jeder Arm
einmal, aufsteigend, Baseline zuerst) verwechsle Aufwärmdrift mit Entwurfsbreite,
war **falsch**: der Ordnungseffekt beträgt `-0,0159` bis `+0,0008`.

**Terminal:** Phase 3 des Ausbauplans entfällt. `friday_serve/speculation.py`
behält eine feste Entwurfsbreite; das Thompson Sampling wird nicht verdrahtet.
Welcher Wert es wird, entscheidet D5 — die Breiten `1`–`3` liegen hier innerhalb
von `0,016`, also wird der Wert dort gemessen und nicht hier geraten.

Herleitung im Arbeitsjournal unter „2026-09-02 — D2 bestanden, und die Prämisse
von Phase 3 hält nicht".

## M1 — beantwortet am 2026-09-02 durch D5, nicht durch Forensik

**Die Frage war,** welcher von zwei widersprüchlichen Messharnessen recht hatte.
Sie wird nicht durch einen Vergleich der beiden beantwortet, sondern dadurch,
dass beide ersetzt sind: `friday_serve` ist seit D2 gegen
`mlx_lm.stream_generate` äquivalenzgeprüft und damit das einzige Werkzeug mit
belegter Referenz.

**D5 misst Spekulation durch `friday_serve` und zeigt, dass die Streitfrage
falsch gestellt war.** Beide alten Harnesse maßen bei fester Antwortlänge und
variierten die Entwurfsbreite. Gemessen entscheidet aber das Verhältnis von
Antwortlänge zu Prompt: auf `897`/`32` verliert Spekulation bei jeder Breite
über `1`, auf `897`/`96` (S1) gewinnt sie bei jeder. Zwei Werkzeuge, die
verschiedene Antwortlängen benutzten, mussten sich widersprechen.

Messzeit ist damit in die Zahl geflossen, die das Projekt braucht, statt in eine
Forensik zweier Werkzeuge, die ohnehin ersetzt sind.

## R2 — Offline-RL auf dem geloggten Korpus

**Status:** offen; Korpus fehlt, Code und Kampagnenplanung stehen. R0/R1 am
2026-09-01 implementiert, Kampagnenplanung am selben Tag — Ergebnisse im
Arbeitsjournal unter „2026-09-01 — R0 und R1" und „2026-09-01 — R2-Korpus".
Gesamtplan R0–R4: [`docs/FABLE_ERFOLGSPFAD.md`](docs/FABLE_ERFOLGSPFAD.md).

**Mechanismus:** konservative Offline-RL-Verfahren ohne Live-Exploration
(CQL/IQL-Klasse) über `friday_optimizer.replay`; Policy-Klasse klein und
erklärbar. Bewertung ausschließlich per Off-Policy-Evaluation gegen
Random/Grid/BO unter identischem Budget auf vorregistriertem Holdout.

**Korpusweg, beziffert.** Eine vorregistrierte Explorationskampagne
(`friday_optimizer/campaign.py`, CLI `campaign`) versiegelt die Regel statt der
Aktion und erzeugt so Überlappung. Aus versiegelter Budgetevidenz: die
vorgeschriebene Pause übersteigt die Rechenzeit um Faktor `28`–`31`, also
passen **`10` Messpunkte in einen 30-Minuten-Block**.

**Am 2026-09-02 end-to-end trockengelaufen** (`recover_ground_truth.py`): eine
bekannte Wahrheit wird in einen echten Korpus eingepflanzt und von den echten
Schätzern zurückgewonnen. Ergebnis, ehrlich nach Korpusgröße:

| Punkte | Blöcke | belastbare Schätzungen | Rangfolge |
| --- | --- | --- | --- |
| `50` | `5` | `0/5` — nichts erreicht die Untergrenze | falsch |
| `150` | `15` | `1/5` (nur die gehintete Aktion) | zufällig richtig |
| `400` | `40` | `5/5`, Medianfehler `0,21` Punkte | richtig |

**Die früher genannten `5` Blöcke beantworten genau eine Frage** — „schlägt
die gehintete Aktion die Baseline?" — und auch das nur bei Epsilon `0,5`; bei
Epsilon `0,6` reicht es nicht einmal dafür. Eine **vollständige Rangfolge über
alle fünf Aktionen kostet rund `400` Punkte, also `40` Blöcke** und damit etwa
zwanzig Stunden gegatete Messzeit. Diese Zahl ist die realistische
Eintrittskarte für R2, nicht die `5`.

Nebenbeobachtung aus demselben Lauf: bei `150` Punkten war die *Rangfolge*
bereits korrekt, obwohl vier von fünf Schätzungen unter der Untergrenze lagen.
Die Untergrenze ist für Ordnungsentscheidungen konservativer als für
Größenaussagen. Das rechtfertigt **keine** Absenkung; es wäre allenfalls ein
Grund, ein eigenes vorregistriertes Ordnungsgate zu entwerfen.

**Amortisierung statt Einmalzahlung:** R1b prüft, ob ein Teil dieser Punkte
im Regelbetrieb anfallen kann statt vollständig in der Kampagne.

**Nicht gangbar:** F1s Sessions als Korpus mitzunutzen. F1s Kandidat ist
vorregistriert, jede Entscheidung hätte Propensity `1,0` und damit keine
Überlappung. Geprüft und verworfen am 2026-09-01.

**Gate:** OPE-Vorteil mit Konfidenzintervall und `conclusive=true`, keine
schlechtere Invalid-Suggestion-Rate als die deterministische Suche.

**Kill/Pivot:** bleibt der OPE-Vorteil über Seeds und Holdouts aus, bleibt es
bei Optimization Memory plus deterministischer Suche plus BO; RL bleibt NO-GO
und wird nicht als Abkürzung wiedereröffnet.

## D1 — Die Hashbindung ist auf dem Versiegelungsgerät selbst gerissen (neu 2026-09-02)

**Status:** offen, Code liegt. Der Kalibrierlauf braucht eine Freigabe.

**Befund, gemessen auf dieser Maschine.** `friday_runtime_n10` weicht in
`code_sha256`, `spec_sha256` und `hardware_sha256` von seinen eingefrorenen
Konstanten ab; `friday_head_skip_runtime.load_policy()` meldet
`formal_code_mismatch`. `hardware_sha256` enthält `platform.mac_ver()`, und die
Maschine läuft inzwischen auf macOS `26.6.2`. **Alle drei Runtime-Pakete stehen
dauerhaft in der Baseline — auf dem Gerät, das sie versiegelt hat.** Die
gemessene Evidenz ist davon unberührt (F1 lief über IronMule, nicht über den
gegateten Runtime-Pfad); die Bauform ist es nicht.

**Mechanismus.** `friday_calibrate` ersetzt die eingefrorenen Konstanten durch
ein auf dem Zielgerät erzeugtes, hashverkettetes Profil: A/A-Rauschen als MDE,
je Knopf ein Urteil mit Ratio und Intervall, Entwurfsbreitenkurve. `friday_serve`
schaltet ausschließlich Knöpfe mit Urteil `verified`. Geschätzt `146 s` GPU.

**Gate:** alle drei mit einer Engine-Entsprechung (`head_skip`,
`fixed_compiled`, `bundled_readback`) kommen `verified` zurück — sie sind hier
einzeln bestätigt. Trifft das nicht zu, ist die Kalibrierung falsch, nicht die
Evidenz. `prefill_step_size` wird `not_applicable` erwartet, siehe P1.

**Kill/Pivot:** verifiziert die Kalibrierung auf einem fremden Gerät keinen
einzigen Knopf, ist die Bauform nicht übertragbar und der Community-Anspruch
fällt. Das gehört dann so in `PROJECT_STATUS.md`.

## D2 — beantwortet am 2026-09-02: `friday_serve` ist derselbe Decoder

**Gemessen.** `friday_serve.Server.generate` gegen `mlx_lm.stream_generate`,
drei Promptfamilien (`897`, `54`, `16` Prompttoken), `24` Token, alle Knöpfe aus.
Verdikt `equivalent`, Tokenfolgen identisch auf allen drei. Budget `5,93 s` GPU.
Rohdaten `experiments/serve_equivalence/equivalence.json`.

**Offen bleibt der zweite Teil:** derselbe Vergleich mit einem Geräteprofil, das
Knöpfe verifiziert hat. Der hängt an `D1`, und `D1` hängt an einem sauberen
Arbeitsbaum — `collect_provenance` verweigert einen Lauf auf ungetracktem Stand,
und das zu Recht: ein Geräteprofil ist dauerhafte Evidenz und muss an einen
Commit gebunden sein.

## D3 — Profilerlauf Prefill, nur Diagnose (neu 2026-09-02)

**Status:** offen, braucht eine Freigabe. Liefert **keinen** Kandidaten.

**Mechanismus.** `45,5 %` der Rechenspitze ist für eine rechengebundene Phase
niedrig; verdächtig sind Dequantisierungsaufwand, fehlende Fusion und Tiling.
`AGENTS.md` verlangt einen Profilerbeleg, bevor hier überhaupt ein Kandidat
vorgeschlagen werden darf. Dieser Lauf liefert genau den Beleg.

**Gate:** der Profiler weist einen benennbaren, adressierbaren Anteil der
Prefill-Zeit aus. Ergebnis geht als neuer Backlog-Eintrag mit Mechanismus und
Kill-Kriterium ein — nicht als Zahl und nicht als Versprechen.

**Kill/Pivot:** zeigt der Profiler keinen adressierbaren Engpass, bleibt die
Prefill-Rechenklasse geschlossen und die verbleibenden Prozentpunkte gelten als
nicht hebbar. Zusammen mit der Korrektur an Kandidat 5 heißt das: das
`20 %`-Ziel ist unter strikter Identität auf dem kurzen Workload nicht
erreichbar, und das gehört so in `PROJECT_STATUS.md`.

## D5 — gemessen am 2026-09-02: `friday_serve` liefert `15,61 %`

**Die Zahl, für die das Projekt da war.** `friday_serve` gegen sich selbst,
Knöpfe aus gegen Knöpfe an, gepaart, Tokenidentität terminal, alle Arme gehalten.

| Regime | A/A | `combined` | 95%-KI |
| --- | --- | --- | --- |
| 4B `897`/`32` | `3,69 %` | **`15,61 %`** | `[0,8417; 0,8566]` |
| 4B `897`/`256` | `2,21 %` | **`14,10 %`** | `[0,8543; 0,8641]` |
| 1B `897`/`32` | `14,25 %` | **`30,40 %`** | `[0,6726; 0,7659]` |

Rohdaten in `experiments/serve_gain/`, Herleitung im Arbeitsjournal unter
„2026-09-02 — D5".

**Offen bleiben drei Punkte, jeder mit eigenem Kill-Kriterium.**

1. **Die 1B-Zerlegung trägt nicht.** Bei `14,25 %` A/A-Rauschen schließt
   `fixed_compiled` die `1,0` ein und `head_skip` streift sie. Die Kombination
   ist belegt, die Einzelknöpfe sind es nicht. *Mechanismus:* Paarzahl aus dem
   gemessenen 1B-Rauschen ableiten statt vom 4B zu übernehmen — eine Konsequenz
   für `friday_calibrate`, nicht nur für diese Studie. *Kill:* bleibt das
   Rauschen auch bei mehr Paaren so hoch, ist das 1B auf diesem Gerät kein
   Messziel und nur die Kombination wird berichtet.
2. **Antwortlängen über `287` Token sind nicht messbar.**
   `BudgetPolicy.continuous_gpu_limit_s = 6,0` begrenzt den ununterbrochenen
   GPU-Block; `512` Token brauchen rund acht Sekunden. *Kill:* bleibt die
   Grenze, gilt jede Aussage dieses Projekts nur bis `287` generierte Token, und
   das gehört so in `PROJECT_STATUS.md`. Die Grenze zu senken, damit eine Studie
   läuft, ist ausdrücklich **nicht** vorgesehen — sie steht auf ihren
   Sachgründen oder gar nicht.
3. **Das 4B-A/A-Rauschen war `3,69 %` gegen F1s `0,612 %` am selben Tag.**
   Sechsfach, unerklärt. Für `combined` (`15,61 %`) irrelevant, für
   `bundled_readback` (`1,14 %`) nicht. *Kill:* lässt sich die Differenz nicht
   auf eine Ursache zurückführen, gelten kleine Effekte auf diesem Gerät als
   nicht auflösbar und `bundled_readback` fällt aus dem Auslieferungspfad.

## S2 — Zwei Spekulationsimplementierungen, und die Barriere erklärt es nicht

**Status:** offen, aber eingeengt. Der erste Verdächtige ist gemessen und fällt.

**Befund.** Dasselbe Repository enthält zwei Prompt-Lookup-Spekulationen mit
gegenläufigem Vorzeichen bei derselben Antwortlänge:

- `friday_hardware/speculate.py` — S1 misst bei `96` Token `1,0488`–`1,0590`
  gegen die eigene Nichtspekulation (`experiments/lookup_order/order.json`,
  Entwurfstiefe `0` ist der Baselinearm).
- `.worktrees/friday-optimizer-ironmule/ironmule/runtime.py:_decode_speculative`
  — H1.0 misst bei `96` Token `−26,46 %` gegen den Auslieferungspfad.

**Der Barriereverdacht ist widerlegt.** Das Amendment
(`docs/H10_AMENDMENT_SPEKULATIONSPFAD.md`, `experiments/spec_path/`) entfernte
`mx.eval(picks, *_leaves(state))` und `mx.synchronize()` je Iteration. Wirkung:
`0,30` Punkte bei Breite `1`, `0,40` bei Breite `2` — nichts gegen
Intervallbreiten von `1,2` bis `2,9`.

**Was danach die tragende Erklärung ist:** der Baselinearm. H1.0 misst gegen
einen Pfad, der über acht Token bündelt (`readback_every = 8`, D4); S1s
Nichtspekulationsarm liest je Schritt zurück wie der Spekulationsarm. Die
Asymmetrie existiert nur in H1.0 — und sie ist gewollt, weil der Dispatcher von
genau diesem Pfad umschalten würde.

**Was damit offen bleibt.** Ob die verbleibende Lücke vollständig aus dieser
Asymmetrie stammt oder ob auch der Cache-Typ beiträgt (`mlx_lm`-Cache mit
`trim_prompt_cache` gegen kapazitätsfesten Cache mit `offset`-Rücknahme).

**Gate:** S1s Implementierung gegen denselben gebündelten Baselinearm messen —
dieselbe Länge, derselbe Prompt, dieselbe Statistik. Gewinnt sie auch dort
nicht, ist die Asymmetrie die ganze Erklärung und eine der beiden
Implementierungen kann gelöscht werden.

**Kill/Pivot:** ist die Frage nur noch akademisch — der Dispatcher ist nicht
gebaut, `speculate_k` bleibt `0` —, bleibt der Eintrag als Warnung stehen und
kostet keine Messzeit. Er wird erst wieder relevant, wenn D4 revidiert wird.

## S3 — beantwortet am 2026-09-02: die Spekulation ist nicht tokenidentisch, und der Grund ist bf16

**Status:** Ursache gemessen, Zweig 1 (Numerik) bestätigt. Der Bruch
reproduziert sofort in Paar `0`; erster divergierender Index `10`, `j = 0`
(ungegattertes Token, Entwurf leer), Logits `75,0` gegen `74,5` — bei bf16
**exakt ein ULP**, benachbarte darstellbare Zahlen. Ein Forward der Breite `3`
statt `1` kippt das ohne jede Anomalie. Die Identitätsbehauptung „per
Konstruktion" hält damit dauerhaft nicht.

**Offen bleibt allein die Konsequenz**, nicht die Ursache: `friday_serve/speculation.py`
stützt seine Freigabe ohne Geräteprofil auf diese Behauptung — siehe S4.

**Nicht geschlossen:** der Nachspieler der Iterationsgrenzen ist gegen den
Engine-Zähler nur für `acceptance` geprüft, nicht für die Grenzen selbst; der
Warmup, der ihm den gesunden Fall liefern sollte, war selbst gebrochen. Die
Zweigzuordnung `j = 0` steht damit unter der Annahme korrekter
Iterationsgrenzen.

**Befund, gemessen.** H1.0, `4B`, versiegelter `897`-Token-Prompt, `128`
generierte Token, Entwurfsbreite `2`: `token_identity_broken:pair_0`
(`experiments/switch_point/switch_4b_128_w2.json`). Bei `32`, `48`, `64` und
`96` hielt die Identität bei allen Breiten, bei `128` hielt sie bei Breite `1`.

**Warum das schwer wiegt.** `ironmule/runtime.py:_decode_speculative` übernimmt
ein Entwurfstoken nur, wenn es dem entspricht, was das Modell selbst für diese
Position gewählt hat. Identität ist dort eine Eigenschaft der Konstruktion, kein
Messergebnis. Der Bruch heißt also: entweder die Konstruktion hält nicht, was
sie behauptet, oder die Verifikation vergleicht nicht das, was sie zu
vergleichen glaubt. `friday_serve/speculation.py` stützt seine Freigabe ohne
Geräteprofil ausdrücklich auf diese Identitätsbehauptung — sie ist damit offen.

**Zwei Zweige, beide offen. Der Eintrag trägt bewusst nicht nur einen.**

1. **Numerik/Formabhängigkeit.** `_body` kompiliert je `(capacity, width)`; der
   Referenzarm läuft mit `width = 1`, der Kandidat mit `width = k+1`. Zwei
   kompilierte Graphen, andere Kernel, andere Reduktionswege — an einer knappen
   Position könnte der `argmax` kippen. **Am 2026-09-02 gemessen und bestätigt**
   (`experiments/identity_break/identity_break.json`): der Bruch reproduziert
   sich sofort (Paar `0` von `10`), sitzt bei Index `10` als **freies Token**
   einer Iteration ohne Entwurf, und die divergierenden Token sind `44505` gegen
   `3797` — exakt Top-1 und Top-2 an Position `10` in
   `experiments/identity_forensics/logit_gap.json`, Logits `75,0` gegen `74,5`.
   **Das frühere Gegenargument war falsch:** `0,500` ist bei dieser Größenordnung
   kein großer Abstand, sondern genau ein ULP. Aus derselben Datei ablesbar —
   alle Logits in `[32, 64)` sind Vielfache von `0,25`, alle in `[64, 128)`
   Vielfache von `0,5`, also das bf16-Raster. Die beiden Token sind benachbarte
   darstellbare Zahlen; ein Forward der Breite `3` statt `1` kippt das ohne
   Anomalie.
2. **Akzeptanzlogik und Cache-Rücknahme.** `accepted`-Aufbau,
   `state["position"]["offset"] = mx.array(offset - 1, ...)`, die Maske über
   verworfene Slots. Durchgerechnet wirkt die Rücknahme konsistent — der nächste
   Forward überschreibt die verworfenen Positionen —, aber „nichts gefunden" ist
   kein Ausschluss.

**Gate der Folgestudie:** an der **ersten divergierenden Position** den
Top-2-Abstand protokollieren, dazu beide Tokenfolgen und den Index. Nur der
erste Wert prüft etwas; das Minimum der Familie prüft nichts. Ist der Abstand
dort groß, fällt Zweig 1 und die Suche geht in die Akzeptanzlogik. Ist er nahe
null, ist Zweig 1 belegt, und die Identitätsbehauptung ist grundsätzlich nur bis
auf Gleichstände haltbar.

**Beweislast, offen und benannt.** Der Lauf hat den Bruch erkannt und die beiden
Sequenzen mit dem Prozess weggeworfen; `switch_4b_128_w2.json` enthält die
Meldung und keine Evidenz. `friday_calibrate.runner.Sample` trägt nur
`token_sha256`, die Folge stirbt bereits in `build_runner.run()`. Der Dump
(Feld für die Token-IDs, `on_break`-Callback in `paired_arms`) war gebaut und
wurde bewusst zurückgebaut, um die Amendment-Zelle nicht mit zwei Fassungen des
Messkerns zu messen. Er gehört zur Vorregistrierung dieser Folgestudie, mit
eigenem Test.

**Kill/Pivot:** lässt sich der Bruch nicht auf eine Ursache zurückführen, darf
Spekulation im Auslieferungspfad **nicht** als tokenidentisch geführt werden.
Dann ist sie ein Kandidat mit Qualitätsgate wie jeder andere — was der Nutzer
am 2026-09-02 ausgeschlossen hat — und fällt damit aus dem Auslieferungspfad.

## P3 — Die Paarzahlregel kennt nur ihre beiden Ränder (neu 2026-09-02)

**Status:** offen, Gate beantwortet. Nicht während einer laufenden Studie
anfassen.

**Befund.** H1.0s eingefrorene Regel `clamp(ceil(6·(s/0,03)²), 6, 24)`,
aufgerundet auf gerade, hat in **sechs** gemessenen Regimen genau zwei Werte
ausgegeben:

| Regime | A/A | Regel rechnet | Regel gibt aus |
| --- | --- | --- | --- |
| 4B `32` | `1,13 %` | `0,9` | `6` (Untergrenze) |
| 4B `48` | `1,04 %` | `0,8` | `6` |
| 4B `64` | `2,22 %` | `3,3` | `6` |
| 4B `96` | `0,52 %` | `0,2` | `6` |
| 4B `128` | `0,70 %` | `0,4` | `6` |
| 1B `32` | `11,77 %` | `92,3` | `24` (Deckel) |

**Nie ein Zwischenwert.** Der Freiheitsgrad, für den die Formel gebaut wurde,
wurde in keinem einzigen Regime benutzt; sie hat ausschließlich zwischen ihren
beiden Rändern geschaltet.

**Zwei Lesarten, und sie sind nicht gleichwertig.** Entweder ist das
Auflösungsziel `3 %` gegenüber dem tatsächlichen Rauschen dieses Geräts falsch
kalibriert — die 4B-Regime liegen alle darunter, das 1B weit darüber. Oder das
Ziel müsste an der kleinsten Effektgröße hängen, die eine Serving-Entscheidung
überhaupt auslösen soll, statt an einer runden Zahl.

**Kill:** Die Regel wird durch eine ehrliche Zweipunktentscheidung ersetzt:
`6` Paare, wenn das gemessene `s` unter einer festzulegenden Grenze liegt, sonst
`24`. Keine Formel, die so tut, als würde sie interpolieren. Die Grenze selbst
ist vorregistrierungspflichtig und wird **nicht** aus den sechs vorhandenen
Punkten angepasst, sondern aus der kleinsten Effektgröße abgeleitet, die eine
Entscheidung auslösen soll.

**Nebenbefund aus demselben Datensatz, der eigenständig zählt:** bei `s = 11,77 %`
verlangt ein `wins` ein Intervall vollständig unter `0,8823`, also mehr als rund
`12 %` Vorsprung. Ein Gewinn der Größenordnung, die S1 auf dem 4B misst
(`3`–`6 %`), wäre auf dem 1B in diesem Regime **nicht** unterscheidbar. Ein
`tie` heißt dort also „kein Vorteil oberhalb von rund `12 %` nachweisbar", nicht
„bringt nichts". Das gehört in jede künftige 1B-Berichterstattung.

## D4b — Gilt die schwächere Serving-Latte allgemein? (neu 2026-09-02)

**Status:** offen, Entscheidung des Nutzers. Blockiert nichts.

**Abgrenzung.** D4 ist entschieden, aber eng: „bundled_readback bleibt drin, D4
so entscheiden" ist eine Entscheidung über **einen** Knopf. Ob die dabei
angewandte Latte — Bootstrap-Intervall vollständig unter `1,0` plus exakte
Tokenidentität, also schwächer als eine Studienpromotion — **allgemein** für
künftige Serving-Knöpfe gilt, ist damit nicht entschieden. Die allgemeine
Fassung senkt die Hürde für alles Kommende; das ist ein zweiter, größerer
Beschluss.

**Was daran hängt.** `friday_calibrate.KnobVerdict` setzt die schwächere Latte
bereits im Code; solange D4b offen ist, ist sie durch genau einen Knopf gedeckt
und nicht durch eine Regel.

**Kill:** wird die allgemeine Latte abgelehnt, muss jeder künftige Serving-Knopf
dieselbe Schwelle tragen wie eine Promotion, und `friday_calibrate.KnobVerdict`
wird darauf gehoben — `bundled_readback` bleibt als benannte Einzelentscheidung
bestehen und wird nicht rückwirkend entfernt.

## S4 — Prompt-Lookup trifft auf der Auslieferungsworkload nie (neu 2026-09-02)

**Status:** offen. Folgt aus S3 und stellt die Frage neu, wofür Prompt-Lookup in
diesem Projekt gedacht ist.

**Befund, vom Zähler der Engine.** `Engine.generate()['acceptance']` auf dem
versiegelten `897`-Token-Prompt, `128` generierte Token:
`0,0` bei `speculate_k = 1`, `2` und `3`
(`experiments/identity_break/acceptance.json`). Kein einziges Entwurfstoken
wurde je angenommen. Der Nachspieler aus S3 kam unabhängig auf dasselbe: `127`
Iterationen für `128` Token.

**Mechanismus.** `_lookup_draft` sucht die letzten `ngram = 3` Token der Sequenz
im Prompt. Der versiegelte Prompt besteht aus einer vierzigfach wiederholten
Anweisungsfloskel; der generierte Text — eine Erklärung zu False Sharing —
enthält deren Trigramme nicht. Die Wiederholung des Prompts **mit sich selbst**
nützt nichts, solange die Antwort ihn nicht zitiert.

**Warum das zählt.** Die Kosten steigen trotzdem: Decode `2,398` / `3,273` /
`4,054` Sekunden bei `k = 1, 2, 3` gegen rund `1,9` der Baseline. Das ist die
tragende Ursache von H1.0s Verlust und der Grund, warum der Befund
**workloadbedingt** ist. `friday_serve/speculation.py` schätzt genau diese
Trefferwahrscheinlichkeit vorab über die Unigrammrate des Prompts — und liegt
für diesen Prompt zu hoch, weil die Unigrammrate hoch und die Trigrammrate null
ist.

**Gate:** die Trigrammrate zwischen Prompt und *erwarteter Antwort* ist der
richtige Prädiktor, nicht die Unigrammrate des Prompts allein. Zu messen: für
drei Promptfamilien die gemessene `acceptance` gegen beide Raten. Sagt die
Trigrammrate die Annahme vorher und die Unigrammrate nicht, ist der Schätzer in
`speculation.py` falsch kalibriert.

**Kill:** trifft der Lookup auch auf einer wiederholungsreichen Workload dieses
Produkts nicht an, ist Prompt-Lookup für dieses Produkt die falsche Technik und
`friday_serve/speculation.py` wird entfernt statt kalibriert.

## H1.3d — Darf für die Drosselungsmessung Dauerlast gefahren werden? (neu 2026-09-02)

**Status:** offen, Entscheidung des Nutzers. Blockiert nichts: ohne Beleg wird
in den Serving-Pfad ohnehin nichts eingebaut.

**Befund.** Der Masterplan verlangt vor jedem Serving-Schutz den Beleg, dass
diese Maschine überhaupt drosselt: Dauerlast über mehrere Minuten, Token/s über
die Zeit protokolliert. Genau das verlangt, `BudgetPolicy.continuous_gpu_limit_s
= 6,0` und den `25 %`-Duty-Cycle zu überschreiten. Diese Grenzen sind
Nutzerregeln (`AGENTS.md`, „Was dadurch ausdrücklich nicht entfällt"), keine
Studienparameter.

**Was ohne Entscheidung läuft.** `experiments/thermal_drift/measure.py` misst
die Ratendrift **innerhalb** des erlaubten Duty-Cycles über mehrere Minuten.
Das beantwortet „driftet die Rate über eine lange Sitzung, so wie dieses Projekt
die GPU fährt" — nicht „drosselt die Maschine unter Volllast".

**Kill:** bleibt die Entscheidung aus, bleibt die Vollastfrage unbeantwortet und
der Serving-Pfad bekommt keinen Schutz. Das ist ein zulässiger Endzustand und
gehört dann so in `PROJECT_STATUS.md`.

## L1 — Friday Learning Controller v0.1 im Shadow-Modus

**Status:** Offline-Implementierung freigegeben am 2026-08-30 durch fortbestehenden Nutzerauftrag; Hardware- und Promotionspfad bleiben gate-basiert gesperrt

**Priorität:** nach Audit und Vereinheitlichung der vorhandenen Evidenz

**Freigabegrenze:** Die Offline-Implementierung der Control Plane ist freigegeben.
Neue reale Modellläufe bleiben ausschließlich manuell, am Netzteil, bei sicherer
Fremdlastfreiheit und sparsam mit maximal 30 Minuten erlaubt. Downloads und
Installationen bleiben ausgeschlossen. Eine automatische Produktaktivierung bleibt
bis zum bestandenen Promotionsgate gesperrt.

**Datenregel:** Echte vorhandene Daten und End-to-End-Tests haben Vorrang vor
synthetischen Daten. Synthetische Daten dürfen nur Rand- und Fehlerfälle abdecken;
sie begründen keine Performance-, Hardware- oder Modellbehauptung.

### Ziel

Eine kleine, begrenzt autonome Optimierungs-KI soll für **genau einen
vorregistrierten Operations- und Aktionsraum** aus historischen und kontrolliert
neu erhobenen Messungen lernen:

1. den aktuellen Hardware-/Workloadkontext beobachten,
2. ausschließlich erlaubte Kandidaten oder den nächsten sicheren Messpunkt
   priorisieren,
3. den bestehenden isolierten Worker und unveränderliche Correctness-, Ressourcen-
   und Benchmarkgates verwenden,
4. Ergebnis und Unsicherheit in ein versionssicheres Optimization Memory
   zurückschreiben und
5. im ersten Schritt nur eine Shadow-Empfehlung ausgeben. Baseline, Promotion und
   Rollback bleiben deterministisch und unabhängig vom Lernmodell.

Der minimale Erfolg ist nicht zwingend ein schnellerer Kandidat. Der Loop gilt
auch dann als funktionierend, wenn er falsche oder langsame Kandidaten zuverlässig
verwirft und reproduzierbar bei der Baseline bleibt.

### Nicht-Ziele für v0.1

- keine allgemein selbstprogrammierende oder sich selbst verändernde KI;
- keine freie Kernel- oder Sourcecode-Erzeugung;
- kein Lernen innerhalb produktiver Nutzeranfragen;
- keine automatische Änderung von Toleranzen, Baselines, Budgets oder
  Erfolgsschwellen;
- kein Reinforcement Learning und kein LLM als innerer Tuner oder Promotionrichter;
- kein Cross-Device-, Cross-Model- oder allgemeiner Self-Learning-Claim;
- keine autonome Produktaktivierung.

### Bereits vorhandener Unterbau

- getrennte SQLite-/JSON-Historien, Hashverkettung und lokale read-only UIs;
- Hardware-, Software-, Code-, Spec-, Modell- und Workloadbindungen in mehreren
  abgeschlossenen Studien;
- gepaarte Baseline-/Kandidatenmessungen mit Warmup, Wiederholungen,
  Correctness- und Ressourcengates;
- isolierte Worker, Timeout-/Ressourcenlogik, Circuit Breaker und serieller
  Fallback;
- N8/N10-Shadow-Router sowie eng freigegebene N10- und Head-Skip-Runtimepfade;
- positive, negative, fehlerhafte und abgebrochene Ergebnisse als Ausgangsmaterial.

Noch **nicht** vorhanden sind ein vereinheitlichter Lernkorpus, ein
leakage-sicherer Split, ein qualifiziertes Kosten-/Rankingmodell, kalibrierte
Unsicherheit, Drift-/Out-of-Distribution-Erkennung und ein Lerncontroller mit
reproduzierbarer Shadow-Auswertung. Die bestehenden Gemma-Planer sind keine
qualifizierte Lernkomponente; ihre strikten Ausgabe-/Vertragsgates sind
fehlgeschlagen.

### Vor jeder Implementierung zu entscheiden und freizugeben

- **Lernziel:** genau eine Startoperation beziehungsweise ein kurzer Teilgraph.
  Kandidaten sind ein geschlossener Planraum um N10/seriell, ein enger
  Head-Skip-Kontext oder eine Template-/Parameterfamilie einer einzelnen
  Operation. Verschiedene Studien dürfen nicht unbesehen vermischt werden.
- **Aktionsraum:** feste, typisierte Allowlist von Plan-/Template-IDs und
  Parametergrenzen; kein freier Sourcecode.
- **Zielfunktion:** genau eine primäre Metrik, zum Beispiel relative
  P50-Laufzeit oder TTFT, plus harte Correctness-, Peak-Memory-, RSS-, Swap-,
  Timeout- und Qualitätsgrenzen.
- **Autonomiegrenze:** zunächst Offline-Replay und Shadow-Empfehlung. Jeder reale
  neue Messlauf und jede spätere Aktivierung brauchen eine getrennte Freigabe.
- **Datenschema:** kanonische Schema- und Migrationsversion sowie Regeln für
  Retention, Quarantäne und Invalidierung.
- **Modell-/Bibliothekswahl:** zuerst einfache Regression/GBDT und Bayesian
  Optimization gegen Random/Grid. Vor einer neuen Abhängigkeit oder lokalen KI
  ist der vorhandene Bestand zu prüfen und eine ausdrückliche Installations-
  beziehungsweise Downloadfreigabe einzuholen.

### Benötigte Daten

Alle Felder müssen aus der Sicht **vor der Kandidatenmessung** als Feature oder
erst **nach der Messung** als Label gekennzeichnet sein. Sonst entsteht Leakage.

| Datengruppe | Pflichtdaten | Zweck / Gate |
| --- | --- | --- |
| Umgebung | Chip-/GPU-Familie, Kernzahl, RAM, OS-Build, Metal-/MLX-/Python-/Compiler-/Projektversion, Capability-Snapshot | exakter Gültigkeitsbereich, Cache-Invalidierung und Drift |
| Kalibrierung | A/A-Rohsamples, gemessene Bandbreiten-/Compute-Kalibrierung, Timer-/Sync-Scope | Rauschen und kleinsten auflösbaren Effekt bestimmen |
| Workload | Semantik-/Referenzhash, Operation, reale Shapes, Strides, Dtype, Batch-/Kontextbezug, Werteverteilung, Aliasregeln, NaN/Inf-Regeln, Toleranzen | fachlich gleiche Fälle erkennen und Hidden Correctness ermöglichen |
| Modell-/Inputbindung | Modell- und Quantisierungssnapshot, Tokenizer-/Prompt- oder Generatorversion, Seed und Input-/Outputhash; sensible Inhalte möglichst nicht speichern | Reproduzierbarkeit ohne unkontrollierte Inhaltsweitergabe |
| Baseline | Plan-/Source-/Artefakthash, Frameworkpfad, Version, Compileflags und Messvertrag | starke, unveränderte Referenz pro Vergleich |
| Kandidat/Aktion | Template-/Plan-ID, Parameter, Source-/Artefakthash, Compileflags, Math-Modus, Suchalgorithmus, Seed und Trialbudget | erklärbarer und replaybarer Aktionsraum |
| Kompilierung | Dauer, Status, Exit-/Signalgrund, Warnungs-/Loghash, Artefakthash, Peak-RSS und Timeout | ungültige Kandidaten und Compilekosten mitlernen |
| Correctness | Testset-/Holdout-ID, Seed, Token-/Byteidentität oder Fehlerstatistik, Invarianten, maximale Abweichung, Status | Machbarkeit ist ein eigenes Label und geht jeder Performancewertung voraus |
| Performance | sämtliche Warmup- und Rohsamples, randomisierte Blockreihenfolge, Median, Streuung, P95, Durchsatz/TTFT soweit relevant, Sync- und Readbackumfang | Effekt und Unsicherheit neu berechenbar halten |
| Ressourcen | MLX Active/Peak Memory, Prozess-Peak-RSS, Swap vor/nachher, Laufzeitbudget, Workerstatus | harte Sicherheits- und Produktguardrails |
| Systemzustand | Zeitstempel, Prozessfrische, thermischer Zustand, Energie-/Powerdaten nur wenn öffentlich und belastbar messbar, konkurrierende Last soweit beobachtbar | Störgrößen, Zensierung und Drift erkennen |
| Ergebnis | Baseline-/Kandidatenratio, Effektgröße, Konfidenzintervall, Feasible/Invalid/Censored, Abbruchgrund, Holdoutentscheid und Gültigkeitsbereich | Lernlabel, Promotionsevidenz und Negativresultat |
| Provenienz | Run-/Study-ID, Schema-, Code-, Spec-, Datensatz- und Modellhash, Parent-/Chain-ID, Erzeuger und unveränderlicher Zeitbezug | Replay, Audit und Schutz vor Datenvergiftung |

Wichtig: Erfolgreiche Kandidaten allein reichen nicht. Compilerfehler, falsche,
langsame, ressourcenintensive und kontrolliert abgebrochene Kandidaten müssen als
eigene Ergebnisse erhalten bleiben. Summaries ohne Rohsamples dürfen nicht als
gleichwertige Trainingslabels verwendet werden.

### Benötigter Datenkorpus und Coverage

- Zuerst wird ein read-only Inventar aller relevanten SQLite-, JSON- und
  Artefaktquellen erstellt. Formale, Engineering-, explorative und
  `legacy_summary`-Evidenz bleiben getrennte Qualitätsklassen.
- Alle Quellen werden auf das kanonische Schema abgebildet, nach Hash/Identität
  dedupliziert und mit expliziten Missing-/Censored-Markierungen versehen.
- Der erste Pipeline-Smoke darf vorhandene kleine Datenmengen verwenden, erzeugt
  aber **keinen Lern- oder Generalisationsclaim**.
- Vor einem qualifizierten GBDT-/Rankingclaim werden Hunderte bis Tausende saubere,
  vielfältige Messungen benötigt. Die genaue Mindestzahl wird nicht geraten,
  sondern nach A/A-Rauschmessung, Featurezahl, Shape-/Parametercoverage und
  vorregistrierter Power-/MDE-Betrachtung festgelegt.
- Training, Validation und Holdout werden mindestens nach Study/Run, Zeit,
  Shape-Familie und bei späterer Erweiterung nach Hardware/Operation gruppiert.
  Ein zufälliger Row-Split ist unzulässig.
- Hidden Correctness und Performance-Holdouts bleiben dem Planner und dem
  Featurebuilder verborgen. Nachbarshapes allein gelten nicht automatisch als
  Generalisationsbeleg.
- Pro terminalem Hardwareurteil sind unabhängige, randomisiert balancierte
  Sessions und ein Konfidenzintervall erforderlich; Einzelmesswerte sind keine
  Evidenz.

### Zu bauende Komponenten

1. **Corpus Auditor:** findet alle Quellen read-only, zählt verwendbare/fehlende
   Felder, erkennt Dubletten, Versionskonflikte, beschädigte Ketten und Leakage.
2. **Kanonisches Optimization Memory v2:** SQLite-Schema für Environment,
   Workload, Candidate, Compile-, Correctness-, Benchmark-, Profile- und
   Promotionrecords plus inhaltsadressierten Artefaktspeicher. Bestehende
   Evidenzdatenbanken bleiben unverändert; Import ist reproduzierbar und
   idempotent.
3. **Dataset Builder und Dataset Card:** erzeugt einen versionierten Snapshot mit
   Hash, Qualitätsklassen, Coverage, Missingness, Censoring, Splitdefinitionen
   und bekannten Grenzen.
4. **Feature Extractor:** deterministische, erklärbare Features aus Hardware,
   Operation, Shape/Stride/Dtype, Templateparametern, statischer
   Ressourcenabschätzung und Versionen. Sourcecode-Embeddings sind für v0.1
   ausgeschlossen.
5. **Deterministische Suchbaselines:** Random und Grid, bei geeignetem kleinen
   Raum zusätzlich Bayesian Optimization; identische Kandidaten-, Zeit- und
   Hardwarebudgets.
6. **Zweistufiges Lernmodell:** zunächst Machbarkeit/Fehlerklasse vorhersagen,
   danach relative Performance nur für zulässige Kandidaten ranken. Eine
   einfache Regression ist Sanity-Check; GBDT ist der erste Learned-Kandidat.
7. **Unsicherheits- und OOD-Gate:** unbekannte Fingerprints, Shapes, Versionen
   oder hohe Unsicherheit führen immer zu `no_recommendation` beziehungsweise
   Baseline/Shadow-Fallback.
8. **Offline Evaluator:** leakage-sichere Holdouts, Rank-Korrelation,
   Best-of-Budget/Regret, Kalibrierung, Fehlerklassen, Invalid-Suggestion-Rate
   und faire Ablation gegen Random/Grid/BO.
9. **Shadow Controller:** liest einen eingefrorenen Modell- und Datensatzhash,
   gibt nur eine Empfehlung mit Konfidenz/Begründung aus und verändert weder
   Runtime noch Promotionstatus.
10. **Worker-/Validator-Adapter:** nutzt die vorhandene Prozessisolation,
    Timeouts, Ressourcenlimits, Correctness, Readback und Fallbacks; Modelloutput
    wird als unvertrauenswürdige Eingabe behandelt.
11. **Promotion-/Rollback-Gate:** bleibt zunächst deaktiviert. Eine spätere
    Aktivierung benötigt vorregistrierte Schwellwerte, Canary, Circuit Breaker,
    sofortigen Baselinefallback und eine eigene Nutzerfreigabe.
12. **Lokale History-UI:** zeigt Datensatzcoverage, Datenqualität,
    Modell-/Datensatzversion, Training/Holdout, Prediction vs. Messung,
    Unsicherheit/OOD, Regret, Fehler-/Fallbackrate, Drift und jede Entscheidung
    chronologisch; read-only und loopback-only.
13. **Test- und Replay-Suite:** Schema-/Migrations-, Leakage-, deterministische
    Feature-, Modellartefakt-, OOD-, Manipulations-, Workercrash-, Circuit-
    Breaker-, UI-, Replay- und Regressionstests.

### Sicherheits- und Integritätsgrenzen

- Kandidaten, Compileroutput, Profilerlogs und Modelloutput sind unvertrauenswürdig.
- Lernmodell und Planner erhalten keinen direkten Schreibzugriff auf Spezifikation,
  Schwellwerte, Baseline, Rohhistorie, Promotiondaten oder ausführbare Dateien.
- Ausgeführter Source-, Binary-/Artefakt- und Messrecordhash müssen identisch
  gebunden sein; TOCTOU führt zum Abbruch.
- Neue Kandidaten laufen ohne Netzwerk, in einem kontrollierten Prozess mit
  Timeout, Ressourcenlimits, privatem temporärem Verzeichnis und terminaler
  Fehlerklassifikation.
- Datenimport ist append-only beziehungsweise snapshotbasiert, idempotent und
  prüft Ketten/Hashes. Änderungen an historischer Evidenz sind unzulässig.
- Kein Modell darf einen fehlgeschlagenen Correctness-/Ressourcengate durch einen
  vermeintlichen Performancegewinn überstimmen.

### Arbeitsphasen mit Gates

#### L1.0 — Read-only Daten- und Architektur-Audit

**Status:** Architektur freigegeben am 2026-08-30 für die Offline-Implementierung;
Hardwareläufe, automatische Aktivierung, Downloads und Installationen bleiben
blockiert. Der Architekturvorschlag steht in
[`docs/L1_GEMMA_OPTIMIZER_ARCHITEKTURVORSCHLAG.md`](docs/L1_GEMMA_OPTIMIZER_ARCHITEKTURVORSCHLAG.md).
Der geprüfte Q2-Handover reduziert die geplante Implementierungsduplikation:
IronMule `tune` wird über einen strikt gebundenen Adapter genutzt. Der
Architekturvorschlag ist entsprechend aktualisiert; Hardware und Promotion bleiben
separate Gates.

- Quellen, nutzbare Records, Feldcoverage, Qualitätsklassen und Datenlücken
  vollständig inventarisieren.
- Genau einen Startscope, Aktionsraum und eine primäre Zielfunktion als
  Architekturvorschlag vorlegen.
- Baseline des aktuellen Speicher-, Laufzeit-, Test- und UI-Standes dokumentieren.

**Gate:** reproduzierbares Inventar plus freigegebene Architektur.

**Kill/Pivot:** Sind relevante Rohsamples, Provenienz oder eine stabile Baseline
nicht rekonstruierbar, wird kein Lernmodell behauptet. Dann zuerst neue
vorregistrierte Daten erheben oder beim deterministischen Tuner bleiben.

#### L1.1 — Optimization Memory v2 und Corpus Builder

**Status:** abgeschlossen am 2026-08-30; offline materialisiert und ohne
Modell-/Hardwarelauf. Ergebnisse: `optimizer-v2.sqlite3` mit `401` Records,
Integritätskette `true`, sowie `optimizer-dataset-v1.json` mit `392` Records.
Der Datensatz bleibt `smoke_only/no_learning_claim` (`train=2`, `val=0`,
`holdout=0`); daraus wird kein Lern- oder Generalisationsclaim abgeleitet.

- Schema, idempotenten Import, Dataset Snapshot/Card, Split- und Leakageprüfungen
  umsetzen; alte Evidenz bleibt read-only.
- UI zeigt Coverage, Missingness, Censoring und Herkunft.

**Gate:** gleicher Input erzeugt byte-/hashidentischen Datensatz; fehlerhafte,
langsame und abgebrochene Kandidaten bleiben enthalten.

**Kill/Pivot:** Wenn Qualitätsklassen nicht sauber trennbar sind oder Labels nur
aus nicht replaybaren Zusammenfassungen stammen, werden diese Quellen vom
Training ausgeschlossen und die Datenerhebung neu geplant.

#### L1.2 — Offline Baselines und Learned Ranking

**Status:** deterministische Baselines und Shadow-Auswertung implementiert und
getestet; Learned Ranking, GBDT und BO bleiben offen und blockiert, weil der
aktuelle Korpus nur `2` Trainingsrecords, keine Validation und keinen Holdout
enthält.

**Mechanismus:** Erst echte, qualitätsklassifizierte Records in getrennten
Train-/Validation-/Holdout-Splits sammeln, dann deterministische Baselines gegen
Regression/GBDT/BO mit identischen Budgets vergleichen. Synthetische Daten dürfen
nur Rand- und Fehlerfälle prüfen.

**Kill/Pivot:** Solange `val=0` oder `holdout=0` gilt, bleibt der Status
`smoke_only/no_learning_claim`; kein Learned Model darf in Empfehlungen oder
Promotionentscheidungen eingehen. Bleibt ein Learned Model nach ausreichender
Coverage ohne reproduzierbaren Best-of-Budget-Vorteil, wird es entfernt und die
deterministische Suche bleibt der Pfad.

- Regression und GBDT gegen Random/Grid/BO unter identischen Splits und Budgets
  evaluieren; keine Runtimeintegration.
- Vorhersageunsicherheit, OOD und `no_recommendation` qualifizieren.

**Gate:** vorregistrierter Holdout, reproduzierbares Training, vollständige
Artefakt-/Datensatzbindung und mindestens Gleichstand mit der besten einfachen
Baseline ohne schlechtere Invalid-Suggestion-Rate.

**Kill/Pivot:** Liefert das Learned Model über mehrere Seeds/Holdouts keinen
Best-of-Budget- oder Trialvorteil, wird es aus dem Pfad entfernt. Optimization
Memory und deterministische Suche bleiben das Ergebnis.

#### L1.3 — Shadow Controller

**Status:** C0, Preregistration und Session-Plan sind abgeschlossen und geprüft;
die reale Adapter-/Modellausführung wurde noch nicht gestartet. Der Start wartet
weiter auf AC, stabile Idle-/Speicher-/Swap-Werte und `foreign=false`.
Reale Adapterausführung, Profilpromotion und Produktaktivierung bleiben blockiert.

- Modell empfiehlt nur; der bestehende Pfad bleibt unverändert.
- Empfehlung, Unsicherheit, tatsächliche Baselineentscheidung und späteres
  Messergebnis werden gemeinsam historisiert.

#### L1.3a — Gemma Multi-Modell-Portfolio (offline/read-only)

**Status:** in Arbeit; keine Hardwarefreigabe.

**Mechanismus:** Exakte, getrennte Identitätszellen für Gemma 1B, 4B, 12B und
27B verbinden bereits vorhandene lokale Cache-Identitäten mit ausschließlich
qualitätsklassifizierter Evidenz. Jede Zelle liefert genau einen Status:
`ready_for_experiment`, `waiting_readiness`, `missing_local_model`,
`insufficient_evidence` oder `unsupported`. Ein deterministischer
`next_safe_measurement` darf nur den nächsten erlaubten Messpunkt benennen;
er startet keinen Lauf und ersetzt weder Readiness noch Nutzerfreigabe.

Der aktuelle reale Cache enthält Gemma 1B, 4B und 12B; Gemma 27B fehlt lokal.
`legacy_summary`, Quarantäne, fehlende Identitätsfelder und Evidenz fremder
Hardware/Workloads bleiben aus einer Empfehlung ausgeschlossen. CLI und lokale
UI bleiben read-only; Modellload, Download, Aktivierung und Cross-Device- oder
Cross-Model-Speedclaims sind ausgeschlossen.

**Gate:** kanonischer, byteidentischer Portfolio-Snapshot; vollständige lokale
Cache-/Manifest-/Tokenizerbindung; keine Mischung zwischen Modell-, Hardware-
oder Workloadzellen; keine Pfad-/Prompt-/Rohlog-Leaks; echte Quellen vor
synthetischen Fixtures; Statusmatrix, deterministischer nächster Messpunkt,
CLI- und read-only-UI-Tests grün.

**Kill/Pivot:** Ein fehlender oder mehrdeutiger Cache wird positiv bewertet;
`legacy_summary` oder fremde Hardware wird als verwertbare Evidenz verwendet;
ein Resolver lädt/importiert ein Modell oder greift auf das Netzwerk zu; ein
Portfolio-Status startet automatisch einen Lauf; ein Snapshot ist nicht
reproduzierbar; Pfade, Prompts oder Rohlogs werden ausgeliefert; die Candidate-
Registry- oder Workloadbindung kann ohne Invalidierung geändert werden; oder
`unsupported`/`missing_local_model` erhält einen ausführbaren Messvorschlag.
Dann bleibt die betreffende Zelle bei Baseline und die Portfolio-Komponente
wird entfernt oder auf eine rein statische Inventaranzeige zurückgeführt.

**Gate:** keine Runtime-/Evidenzmutation, korrekter OOD-Fallback, reproduzierbarer
Replay und vorregistrierte Mindestwerte für Kalibrierung, Regret und
Invalid-Suggestion-Rate.

**Kill/Pivot:** Jede unerlaubte Aktivierung, Thresholdmutation, falsch als sicher
klassifizierte Correctnessverletzung oder nicht fail-closed behandelte
Ungewissheit stoppt den Controller sofort.

#### L1.4 — Begrenztes autonomes Experimentieren

Nur nach neuer ausdrücklicher Freigabe darf der Controller innerhalb der festen
Allowlist den **nächsten** Messkandidaten wählen. Worker, Validator und Promotion
bleiben deterministisch; Such-/Holdoutdaten bleiben getrennt.

**Gate:** feste Trial-/Zeit-/Speicherbudgets, Canary, Circuit Breaker,
Baselinefallback und Vorher-/Nachhermessung.

**Kill/Pivot:** GPU-/Systeminstabilität, nicht begrenzbare Hänger,
Datenbankinkonsistenz, wiederholtes Correctness-Overfitting oder Nutzen unterhalb
der Tuning-/Wartungskosten beendet die autonome Stufe.

#### L1.5 — Promotion oder Erweiterung

Automatische Promotion, mehrere Operationen, CPU/GPU-Placement, ein LLM-Outer-
Planner oder ein neuronales Modell sind jeweils eigene, spätere
Architekturentscheidungen und Vorregistrierungen.

**Kill/Pivot:** RL und direkter LLM-Tuner bleiben NO-GO, solange ein kleines
erklärbares Modell plus echte Messung nicht nachweislich unzureichend ist. Ein LLM
wird dauerhaft entfernt, wenn es Random plus BO/evolutionäre Suche unter gleichem
Budget nicht reproduzierbar schlägt.

### Definition of Done für L1

- ein freigegebener, exakt gebundener Scope und geschlossener Aktionsraum;
- ein reproduzierbarer, versionierter Lernkorpus mit Dataset Card und Hidden
  Holdout;
- mindestens Random/Grid beziehungsweise BO und Regression/GBDT fair verglichen;
- ein eingefrorenes Modellartefakt mit Datensatz-/Code-/Featurehash und
  Unsicherheits-/OOD-Verhalten;
- ein read-only Shadow Controller mit Baselinefallback, Replay und History-UI;
- vollständige Tests sowie dokumentierte Vorher-/Nachherwerte für Performance,
  Speicher, Laufzeit, Genauigkeit und Qualität;
- kein ungeprüfter Self-Learning-, Produkt-, Hardware- oder Generalisationsclaim;
- Ergebnisse, Fehler und verworfene Wege aus dem Backlog in Status, Journal und
  Studienakte übertragen; danach L1 aus dieser Datei entfernen oder nur den
  terminalen Dead-End-Verweis behalten.

---

Kandidaten-Studienakte: [`docs/KANDIDATENLISTE.md`](docs/KANDIDATENLISTE.md)
— kein zweites Backlog. Abgeschlossene Einträge stehen im
[Arbeitsjournal](docs/ARBEITSJOURNAL.md); die Repo-Hygiene M1 ist am
2026-09-02 vollständig erledigt worden.

## U1 — Rückbau der zwölf Dashboards (neu 2026-09-02)

**Status:** Ersatz steht, Rückbau offen. Eine Entscheidung, keine Messung.

**Befund.** `4431` Zeilen in zwölf `dashboard.py`, jede ein eigener
`ThreadingHTTPServer` mit eigenem HTML-Renderer. Der Ersatz ist gebaut:
`python tools/friday.py status`, zeilenorientiert, `--json`, `--plain`,
barrierefrei ab `60` Spalten (`friday_runtime_core/status.py`).

**Die Löschung zerfällt in zwei Gruppen, und nur eine ist frei.**

| | Zeilen | Pakete |
| --- | --- | --- |
| frei | `1533` | `friday_optimizer` (`1224`), `friday_phase1b` (`164`), `friday_avo_router` (`145`) |
| hashgebunden | `2898` | `friday_evidence`, `friday_h0`, `friday_h01`, `friday_h1`, `friday_n10`, `friday_n10_v2`, `friday_runtime`, `friday_runtime_n10`, `friday_head_skip_runtime` |

Die zweite Gruppe geht per `rglob` in `code_sha256` ein. Dort zu löschen
widerspricht `AGENTS.md` („Bestehende versiegelte Pakete bleiben wegen
Code-Hash-Bindung byteidentisch eingefroren"). Der Ausbauplan verlangt es
trotzdem. **Beides sind Nutzeranweisungen; der Widerspruch ist echt und gehört
entschieden, nicht stillschweigend in eine Richtung aufgelöst.**

**Mechanismus für den freien Teil.** Erst `status --decisions` bauen, das
`/api/decisions` aus `friday_optimizer/dashboard.py` ersetzt, dann die drei
freien Dateien löschen und ihre Tests auf die Renderfunktion umhängen. `-1533`
Zeilen ohne jedes Hashrisiko.

**Gate:** Vollsuite grün nach dem Rückbau; `status` und `status --json` liefern
denselben Snapshot; `status --plain` enthält kein Escape-Byte; bei `COLUMNS=60`
bleibt jede Zeile vollständig.

**Kill:** bleibt die Hashbindung bindend, entfallen `2898` der `4431` Zeilen
nicht, und der Nettogewinn ist `1533` statt `4000`. Das ist immer noch der
größte Einzelblock des Projekts, und es gehört so in `PROJECT_STATUS.md`.

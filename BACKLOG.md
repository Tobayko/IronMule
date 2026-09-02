# Backlog — Project Friday

Diese Datei enthält nur offene Arbeiten. Jeder Eintrag benennt Mechanismus,
Voraussetzungen, messbare Gates und ein Abbruch- oder Pivotkriterium. Erledigte
Einträge werden entfernt; Ergebnisse und verworfene Wege wandern in
`docs/ARBEITSJOURNAL.md`, `PROJECT_STATUS.md` oder die jeweilige Studienakte.

## F1 — Integrationsstudie: bestätigte Gewinne ernten (freigegeben, priorisiert)

**Status:** Nutzerfreigabe am 2026-09-01 erteilt. Vorregistrierung,
Analysebaustein und Projektion sind am 2026-09-01 fertig — Ergebnis im
Arbeitsjournal unter „2026-09-01 — F1". Offen ist nur noch die Ausführung.
Rahmen und Erfolgshebel: [`docs/FABLE_ERFOLGSPFAD.md`](docs/FABLE_ERFOLGSPFAD.md).

**Vorhanden:** [`docs/F1_INTEGRATION_VORREGISTRIERUNG.md`](docs/F1_INTEGRATION_VORREGISTRIERUNG.md)
(zwei Arme `cold`/`warm`, Schwellen `50 %`/`10 %`, MDE `5 %`, Tokenidentität
terminal), `friday_optimizer/integration.py` (`request_seconds`,
`evaluate_integration`) und `experiments/f1_integration/project_f1.py`.

**Erwartung, korrigiert:** `13,68 %` im warmen Arm, `70,05 %` im kalten. Die
frühere Lesart `21 %` unterstellte, dass Prefill- und Decodegewinn sich
multiplizieren; sie wirken auf verschiedene Phasen derselben Anfrage.

**Ausführungspfad geprüft (2026-09-02).** `session-plan` läuft read-only
durch und schließt korrekt fail-closed; die Blocker sind ausschließlich
Provenienzbindungen gegen veraltete Q2-Artefakte. Kein struktureller Defekt.
Der saubere Commit vom 2026-09-01 hat `optimizer_checkout_dirty` beseitigt.

**Offene Entscheidung — Workload.** Der gegatete IronMule-Pfad fährt `322`
Prompt-Token; F1s Evidenz für persistenten Prozess und Head-Skip stammt von
`897`. Die kombinierte Erwartung fällt damit von `13,68 %` auf `11,93 %`
(Abstand zur `10 %`-Schwelle: `1,93` statt `3,68` Punkte). Drei Wege,
Herleitung im Arbeitsjournal unter „2026-09-02 — F1s Ausführungspfad geprüft":

1. F1 auf `322`/`32` registrieren — nutzt die vorhandene auditierte
   Infrastruktur. **Empfohlen.**
2. Eigenen F1-Worker mit `897`-Prompt bauen wie bei P2 und W1.
3. Schwelle senken — ausgeschlossen.

**Power geprüft (2026-09-02).** Die Entscheidungsfunktion wurde gegen eine
bekannte Wahrheit simuliert. Bei dem Paar-Rauschen, das die versiegelte
Evidenz zeigt (`0,45 %`–`0,73 %`), qualifiziert F1 mit sechs Paaren in
`99,2 %`–`100 %` der Fälle; Falschqualifikation bei einer Wahrheit unterhalb
der Schwelle liegt überall bei höchstens `0,8 %`. Ab `2 %` Rauschen bricht die
Power ein (`65,2 %` bei sechs Paaren). Die Paarzahl ist deshalb jetzt an das
in den A/A-Sessions gemessene Rauschen gekoppelt — Regel in F1s
Vorregistrierung, Abschnitt 4b. Die Schwelle bleibt unverändert.

**Danach:**

1. Workload entscheiden, Fingerprint gegen den aktuellen HEAD sammeln, die
   Vorregistrierung mit Umgebungs-Hashes versiegeln.
2. A/A-Sessions je Arm zur MDE- **und Rauschbestimmung**, daraus die Paarzahl
   nach der vorregistrierten Regel, dann A/B — gegatet, manuell, AC-only,
   fremdlastfrei, maximal 30 Minuten je Lauf, einzeln bestätigt.

**Kill/Pivot:** unter Schwelle oder Identitätsbruch gilt Baseline; terminaler
Negativeintrag in die Studienakte, F1 wird hier gelöscht. Widerspricht die
Messung der Projektion, gewinnt die Messung und die Projektion wird korrigiert.

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

**Nicht gangbar:** F1s Sessions als Korpus mitzunutzen. F1s Kandidat ist
vorregistriert, jede Entscheidung hätte Propensity `1,0` und damit keine
Überlappung. Geprüft und verworfen am 2026-09-01.

**Gate:** OPE-Vorteil mit Konfidenzintervall und `conclusive=true`, keine
schlechtere Invalid-Suggestion-Rate als die deterministische Suche.

**Kill/Pivot:** bleibt der OPE-Vorteil über Seeds und Holdouts aus, bleibt es
bei Optimization Memory plus deterministischer Suche plus BO; RL bleibt NO-GO
und wird nicht als Abkürzung wiedereröffnet.

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

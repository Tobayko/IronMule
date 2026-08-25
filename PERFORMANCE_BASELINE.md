# Performance-Baseline

Stand: 25. August 2026, Zyklus 21 nach realer Runtime-Qualifikation, nach Zyklus 20,
Gemma 3 4B 4-bit g64 auf Apple M1 Max, MLX `0.32.0`,
mlx-lm `0.31.3`. Werte sind gemessen, sofern sie nicht ausdrücklich als Rechnung
markiert sind. Der einzige neue formale Claim ist die unten abgegrenzte
Prefill-Head-Skip-Studie; alle übrigen Werte dieser Baseline bleiben
`formal_claim=false`.

## Gerätemodell

```
ms_je_Token(Breite 1) = Layer × 0,16669 ms  +  Gewichte_GB × 2,79005 ms
```

Gefittet auf zwei Modellen, geprüft an vier zurückgehaltenen Konfigurationen
(Fehler `+0,9` bis `+15,5 %`). Effektive Bandbreite `358,4` GB/s = `89,6 %` der
veröffentlichten Spitze.

| Anteil am 4B-Einzeltoken | ms | Anteil |
| :--- | ---: | ---: |
| Dispatch (`34` Layer) | `5,67` | `48 %` |
| Gewichte lesen | `6,09` | `52 %` |

## TTFT nach Klassen

Klassen werden **nicht** in einem gemeinsamen Schätzer vermischt.

| Klasse | Wert | Bedingung |
| :--- | ---: | :--- |
| Modellladen | `1,47`–`1,76` s | Anteil an `cold_process` |
| `warm_uncached` | `1702,86` ms | `898`-Token-Prompt |
| `warm_prefix_hit` | `131,02` ms | `886` Token wiederverwendet — **korrektheitsungeprüft**, siehe unten |
| `warm_full_cache_hit` | nicht gemessen | – |

## Prospektiv bestätigter persistenter Modellprozess

Zyklus 13 verglich bei demselben lokalen Modell und sechs vorab festgelegten
`897`-Token-Prompts einen neuen Python-/Modellprozess je Anfrage mit einem Prozess,
der das Modell einmal lädt. Jede Anfrage verwendete weiterhin einen frischen
KV-Cache; Ausgabe waren jeweils `32` greedy Token.

| Endpunkt | Ergebnis | vorab festgelegte Grenze |
| :--- | ---: | ---: |
| A/A-Kalibrierung, Median | `0,961468` | `[0,90; 1,10]` |
| Charakterisierung, Median `warm/kalt` | `0,346142` | `≤ 0,50` |
| Validierung, Median `warm/kalt` | `0,347794` | `≤ 0,50` |
| alle sechs Paare, Median `warm/kalt` | **`0,346968`** | `≤ 0,50` |
| größtes einzelnes Verhältnis | `0,349647` | `≤ 0,65` |
| kalte TTFT, Median der sechs Werte | `5148,7741` ms | – |
| warme TTFT, Median der sechs Werte | `1785,1103` ms | – |
| greedy Tokenidentität | **`6/6` exakt** | Pflicht |
| warmes Peak-RSS | `3.763.077.120` Byte | `≤ 5 GiB` |
| warmes RSS-Wachstum | `0` Byte | `≤ 256 MiB` |
| Swap-Wachstum | `0` Byte | `≤ 0` |

Der Effekt **`−65,3032 %`** ist aus dem vorregistrierten Median der sechs
gepaarten Verhältnisse gerechnet; er ist kein separat gemessener Zeitwert. Kein
Paar und kein Ausreißer wurde entfernt. Der Lauf war am Netzteil und blieb mit
`41,586354` s Modellarbeit, höchstens `3,226913` s am Stück, Duty-Faktor `0,15`
und `576,933889` s Gesamtzeit innerhalb aller Schutzgrenzen.

Die Entscheidung lautet `engineering_gain_confirmed_exact_scope`. Sie bleibt
`formal_claim=false` und belegt nur diesen Prozess-Lebenszyklus auf diesem Gerät,
Modell-Snapshot und Workload. Ein normaler Dienst nutzt den Pfad noch nicht
automatisch.

## Gemma-4B-Planertest: gültiges negatives Ergebnis

Zyklus 14 war kein Geschwindigkeitstest, sondern prüfte einen einzigen festen
Planungsfall für das gewünschte selbstoptimierende System. Drei frische Prozesse
luden nacheinander denselben lokalen 4B-Snapshot und erhielten denselben
`322`-Token-Prompt. Erlaubt war nur ein JSON-Objekt mit einer Kandidaten-ID.

| Endpunkt | Ergebnis | vorab festgelegte Grenze |
| :--- | ---: | :--- |
| frische Prozesse | `3` | genau `3` |
| Ausgabe je Prozess | `23` greedy Token | höchstens `32` |
| Token und Text | **`3/3` exakt gleich** | Pflicht |
| Abschluss | **`3/3 stop`** | Pflicht |
| reines JSON ohne Zusatz | **`0/3`** | `3/3` |
| inhaltlich genannte erwartete ID | `3/3` | nur beschreibend, kein Ersatz-Gate |
| Rechenzeit, Median | `1,047670` s | – |
| gesamte Prozesszeit, Median | `4,858676` s | – |
| maximales Prozess-RSS | `3.764.961.280` Byte | höchstens `5 GiB` |
| maximaler MLX-Speicher | `3.021.085.374` Byte | höchstens `5 GiB` |
| Swap-Wachstum | `0` Byte | höchstens `0` |

Alle drei Antworten enthielten zwar
`persistent_service_qualification`, umgaben das Objekt aber mit einem
Markdown-Codeblock. Die vorab festgelegte Entscheidung lautet deshalb
`planner_contract_failed`. Der zusätzliche Rahmen wird nachträglich nicht
akzeptiert oder entfernt. Der Lauf bestätigt weder Selbstlernen noch eine neue
Performanceverbesserung und aktiviert nichts; `formal_claim=false`.

## Zyklus 15 — Zwei-Modell-Planner: gemessene Baseline

Die Studie `dual-model-evidence-planner-20260824-01` lief genau einmal mit sechs
balancierten Paaren und zwölf frischen seriellen Prozessen am Netzteil. Paare
`1–3` liefen `1b → 4b`, Paare `4–6` `4b → 1b`; jedes Modell wurde sechsmal
geladen. Der feste Vertrag verlangte ausschließlich
`{"candidate_id":"persistent_service_qualification"}`. Die Entscheidung ist
`no_planner_qualified`, `formal_claim=false`.

| Messgröße | Gemma 3 1B 4-bit | Gemma 3 4B 4-bit |
| --- | ---: | ---: |
| strikter Vertrag / Parser / erkannte ID | `0/6 / 0/6 / 0/6` | `0/6 / 0/6 / 0/6` |
| deterministische Token/Textläufe | `6/6` | `6/6` |
| Ausgabe / Abschlussgrund | `32 Token / length` | `23 Token / stop` |
| TTFT Median / MAD | `0,295451312 / 0,0005528535 s` | `0,796846125 / 0,0088023125 s` |
| Modellarbeit Median / MAD | `0,4608839165 / 0,0005743330 s` | `1,0487644165 / 0,0092854165 s` |
| Prozess-Walltime Median / MAD | `4,2468557705 / 0,0059329165 s` | `4,883630417 / 0,0182606455 s` |
| Peak-RSS | `1.937.965.056 B` | `3.765.420.032 B` |
| MLX-Peak | `1.012.548.526 B` | `3.021.085.374 B` |
| Swap-Delta | `0 B` | `0 B` |

Die 1B-Antwort verletzte den Vertrag mit Markdown, dem falschen Schlüssel
`persistent_service_id` und `<end_of_turn>`-Trailern. Die 4B-Antwort enthielt
die richtige ID, aber einen Markdown-Codeblock. In `0/6` Paaren waren die
dekodierten Modelltexte bytegleich. Diese Punkte sind rein formale Vertrags-
und Gleichheitsbefunde, keine qualitative Modellkritik.

Ressourcen-, Snapshot-, Pairing- und Budget-Gates bestanden. Gemessen wurden
`9,205052 s` Gesamt-Modellarbeit, maximal `1,151402 s` zusammenhängend und
`178,475444 s` Prozess-Walltime bei Duty-Faktor `0,15`; alle Swap-Deltas waren
`0` und es gab keinen Abbruch. Die folgenden Paarquotienten und Intervalle sind
berechnet (Bootstrap, `10.000` Resamples), nicht weitere Messungen:

| 1B / 4B | Median | Bootstrap-95-%-KI |
| --- | ---: | ---: |
| TTFT | `0,373014193` | `[0,365603946; 0,377539933]` |
| Modellarbeit | `0,439069434` | `[0,434598134; 0,444460794]` |
| Prozess-Walltime | `0,872042394` | `[0,864987297; 0,939562889]` |
| Tokenrate | `3,168801108` | `[3,130352029; 3,201472197]` |

Daraus sind ungefähr `12,8 %` kürzere 1B-Walltime und `48,5 %` weniger
1B-Peak-RSS berechnet. Diese Zahlen erlauben keine Präferenz, weil 1B und 4B
beide das Funktions-Gate verfehlten.

Evidenz: `experiments/dual_model_planner/results.json`, SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`; private
Startmarke SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`; die
Präregistrierung SHA-256
`246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`.

Die UI blieb read-only (GET/HEAD `200`, Schreibmethoden `405`, fremde Hosts
`421`) und änderte die Hashes nicht. Zyklus 15 speichert JSON-Rohdaten, keine
eigene Evidence-DB. Es gibt keinen echten Gemma-Matmul-Schalter und keinen
vollständigen „mit/ohne Matmul“-A/B-Pfad; dieser Vergleich wurde nicht gemessen
und bleibt ein separater künftiger vorregistrierungspflichtiger Kandidat.
Multi-Turn, parallele Requests, allgemeine Modellqualität, allgemeine
Planner-Fähigkeit, selbstlernende Runtime und Produktaktivierung bleiben offen.

## Prefill

| Blockgröße | Positionen/s | ms je Position |
| ---: | ---: | ---: |
| 256 | `505,4` | `1,979` |
| 512 | `530,7` | `1,884` |
| 1024 | `543,5` | `1,840` |
| 2048 | `544,1` | `1,838` |

Sättigt ab `1024`. Chat-Template und Tokenisierung liegen bei `0,044`–`0,649` ms und
sind kein Faktor.

## Formal bestätigter Prefill-Head-Skip

Zyklus 12 prüfte prospektiv genau einen Kandidaten: Beim greedy Prefill ohne
Prompt-Logprobs wird der LM-Head nur auf die tatsächlich gelesene letzte
Promptposition angewendet. Die versiegelte Studie verwendete einen lokalen,
revisionsgebundenen Modell-Snapshot, `897` Prompt-Token, Prefill-Chunk `256`, Batch
`1`, sechs A/A- und sechs A/B-Sessionprozesse sowie vier Messpaare je Session.

| Vorregistrierter Endpunkt | Ergebnis |
| :--- | ---: |
| A/A-Verhältnis | `1,002829` |
| A/A-95-%-KI | `[0,994931; 1,005964]` |
| eingefrorene MDE | `5 %` |
| A/B-Verhältnis gesamt | **`0,846385`** |
| Effekt | **`−15,3615 %`** |
| Charakterisierung, 95-%-KI | `[0,840544; 0,848452]` |
| Validierung, 95-%-KI | `[0,842683; 0,854941]` |
| Gesamt, 95-%-KI | `[0,843147; 0,851284]` |
| Greedy-Tokenidentität | **`12/12` Sessiongates** |

Das Verhältnis, der Effekt und die Intervalle sind die vorregistrierte Auswertung
der unverändert gespeicherten Messblöcke. Aus den gemessenen Sessionmedianen
abgeleitet lagen die Armmediane bei `1995,444239` ms und `1688,116333` ms. Beide
Arme meldeten denselben MLX-Peak von `3.213.903.666` Byte; der beobachtete RSS-Bereich
war `3.768.795.136` bis `3.769.696.256` Byte.

Der terminale Status lautet `head_skip_gain_confirmed`, die Aktion lediglich
`permit_bounded_architecture_review`. `formal_claim=true` gilt ausschließlich für
**ein Gerät, einen Modell-Snapshot, einen Prompt, einen Prefill-Plan und greedy ohne
Prompt-Logprobs**. Es gibt keine automatische Produktaktivierung und keinen
allgemeinen TTFT-, Modell- oder Cross-Device-Claim.

### Begrenzte Runtime-Qualifikation

Nach ausdrücklicher Architekturfreigabe wurde der bestätigte Kandidat als
getrennter, rückrollbarer Repository-Aufrufpunkt eingebaut. Die vorab eingefrorene
Engineering-Qualifikation ist **kein neuer Zyklus und kein neuer formaler Claim**.

| Endpunkt | Ergebnis | vorab festgelegte Grenze |
| :--- | ---: | ---: |
| einmaliger Evidenzload | `824,4602` ms | `< 5000` ms |
| gecachte Auswahl, Median | `0,0008647` ms | `≤ 0,025` ms |
| gecachte Auswahl, p95 | `0,0008725` ms | `≤ 0,050` ms |
| Zusatz zur direkten Auswahl | `0,0008395` ms | `≤ 0,020` ms |
| bisheriger Prefill, Median | `1806,4619` ms | – |
| schneller Prefill, Median | `1528,2070` ms | – |
| gepaartes Prefill-Verhältnis | **`0,845836`** | `≤ 0,95` |
| Prefill-Effekt | **`−15,4164 %`** | mindestens `5 %` schneller |
| Greedy-Tokenidentität | **hält** | Pflicht |
| MLX-Peak-Delta B−A | `−97.855.968` Byte | höchstens `+134.217.728` Byte |
| Swap-Delta | `0` Byte | kein Wachstum |

Die vier unverändert gespeicherten gepaarten Verhältnisse lauten `0,847804`,
`0,843327`, `0,843869` und `0,849053`. Korrektheitspaar und alle vier Messpaare
erzeugten dieselben 32 Token; Hash
`666dcfb103d263a12b29ed9a1c1ec496c6922f96c3a6e7cec083eab47fb5127c`.
Der Kandidatenpfad führte vier Prefill-Blöcke, aber genau einen LM-Head-Aufruf aus.

Der Lauf war am Netzteil, nutzte Duty-Faktor `0,15`, meldete `25,346709` s
Modellarbeit, höchstens `2,283342` s zusammenhängende Arbeit und `192,383843` s
erzwungene Pausen. Die private drei Einträge lange Runtime-Historie hat SHA-256
`6dcf6e4cb942b842dca6e9b0b071df8e7c6cb81ba28fdc5e0fdb05c414d20567` und
Kettenkopf `db4c98e892136930cc515be94417a294c960cfaa9a790a6ea05629d0b796b8f3`.
Der normale Repository-Aufruf autorisiert damit nur den exakt registrierten Fall;
jede Abweichung verwendet weiterhin den bisherigen Pfad.

## Decode-Durchsatz

| Modell | Einzelstrom | bester Batch | Faktor | Peak |
| :--- | ---: | ---: | ---: | ---: |
| 4B | `82,4` tok/s | `64` → `493,0` | `5,98x` | `6,55` GB |
| 1B | `225,1` tok/s | `256` → `3093,0` | `13,74x` | `3,60` GB |

Breitenkurve ist eine Treppenfunktion. Regressionen 4B: `6,7,8,9,48`; 1B: `48`.
Breite `48` regressiert in allen sechs geprüften Quantisierungskonfigurationen und ist
der einzige als konstant behandelbare Wert.

## Inter-Token-Latenz und KV-Reallokationen

Zyklus 11: `765` Prompt-Token, `48` Decodeschritte, Batch `1`, acht
Wiederholungen nach einem verworfenen Aufwärmlauf. Es wurden keine Ausreißer
verworfen.

| Endpunkt | ms |
| :--- | ---: |
| ITL p50 | `14,2670` |
| ITL p95 | `15,1385` |
| ITL p99 | `46,7879` |
| Minimum | `13,8230` |
| Maximum | `49,4430` |

Die Cacheformen änderten sich in allen acht Wiederholungen an denselben Stellen:

| Decodeschritt | Cacheklasse | Layer | gemessener Überschuss | vorab gerechnet |
| ---: | :--- | ---: | ---: | ---: |
| `1` | rotierend | `29` | `31,5853` ms | `0,7616` ms |
| `4` | global | `5` | `0,2968` ms | `0,1317` ms |

Die Summe der gemessenen Überschüsse entspricht `4,4263 %` der Decodezeit und
überschreitet die vorregistrierte `1-%`-Schwelle. Die vorhergesagten `0,13 %` waren
eine Bandbreitenrechnung, kein Messwert. Der große Ausschlag an Schritt `1` erklärt
den p99, ist aber zugleich mit sonstigen Kosten des ersten Decodeschritts konfundiert.
Die Messung lokalisiert daher einen Kandidaten; sie beweist noch nicht, dass eine
Cache-Änderung `4,4263 %` gewinnt. `formal_claim=false`.

## Kontextbasierte Spekulation

Auf echtem Projektinhalt, Median aus Wiederholungen nach Aufwärmlauf:

| Prompt | Akzeptanz | Speedup |
| :--- | ---: | ---: |
| Quelldatei umschreiben | `1,000` | `1,097` |
| Journal-Extraktion | `0,682` | `1,029` |
| Testausgabe | `0,375` | `0,994` |

Tiefe folgt der Trefferlänge: Übereinstimmungen ab `9` Token wurden `48/48` akzeptiert,
kürzere zu `53,6 %` (4B) bzw. `47,2 %` (1B).

## Erkannter Engpass

**Prompt Prefill.** Ein `898`-Token-Prompt kostet `1,70` s bis zum ersten Token,
gegenüber `12,1` ms je Ausgabetoken. Für einen Agenten mit stabilem Präfix dominiert
der Prefill die wahrgenommene Latenz um mehr als zwei Größenordnungen je Anfrage.

## Unsicherheiten und offene Punkte

- **Tokenidentität ist bei geänderter Prefill-Zerteilung nicht gegeben.** Siehe
  `OVERNIGHT_RESEARCH_LOG.md`, Zyklus 1. Das betrifft jede Optimierung, die die
  Blockstruktur verändert — Präfix-Cache und Batching eingeschlossen.
- `warm_full_cache_hit` ungemessen.
- Energie je Token ungemessen (benötigt Freigabe).
- Wired Memory, Memory Compression und Thermal State bisher nicht erfasst.
- Multi-Turn-Fortsetzung bisher nicht als Baseline gemessen.
- Mehrere parallele Requests bisher nicht als Baseline gemessen; `concurrent_32` in
  `EXPERIMENT_MATRIX.json` definiert nur den Workload.
- Alle Werte gelten für **ein** Gerät, **ein** Modell, **eine** Quantisierung.

## Zyklus 16 — Versiegelte Baseline und reales Ergebnis

Für die am 24.08.2026 freigegebene Studie
`matmul-compile-ab-20260824-01` ist der Kandidat
`fixed_cache_compiled_decode_v1` ist im lokalen Seal-Commit versiegelt und
gemessen. Die Arme waren
`standard_eager`, `fixed_eager` und `fixed_compiled`. In jedem Arm bleibt die
mathematische Matmul aktiv; nur Cache-Form und Compile-Umgebung werden verglichen.
Es gibt deshalb keinen Matmul-Aus-Pfad; die folgenden Werte betreffen nur die
Runtime-Organisation.

Modell, Gewichte und Quantisierung bleiben unverändert. Greedy Token- und
Textidentität muss exakt sein. Die alten Device-Model-Compile-Messungen sind
wegen falscher Token ab Position 2 ungültig und werden aus der Baseline
ausgeschlossen. `formal_claim=false`; ein negatives Ergebnis wäre gültig.
Präregistrierungs-SHA-256:
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`. Es gibt
Ergebnis-Hash: `fbcc2fc65ac5d255ed11039a74c34e9a02d942cec17b25a6ed863058e0073b57`;
Marker-Hash: `8adf6f9c2453524bd1e05f4973ee85f84a323e9461a3f9b996ec2d0f7fed3c2f`.

## Zyklus 16 — gemessene Runtime-Baseline und Kandidat

Seal `83ee3ea03f9fb303b8226ab8ad3189f07daec727`, Entscheidung
`runtime_compile_wins_exact_scope`, `formal_claim=false`. Sechs frische Prozesse
und 18 Arm-Ausführungen (3 × 6) erzeugten exakt gleiche Tokens und Texte.
Gemessene Decode-Mediane:

| Arm | Decode gesamt | Tokenrate | TTFT |
|---|---:|---:|---:|
| Standard | 0,399939187 s | 77,5131895/s | 0,638376521 s |
| Fixed-Eager | 0,3999597295 s | 77,5078153/s | 0,638425813 s |
| Fixed-Compiled | 0,371848789 s | 83,3672240/s | 0,6385446665 s |

Gemessene gepaarte Ratios: Fixed-Compiled/Standard `0,9295921887`, KI
`[0,9128789083; 0,9348209684]`; Fixed-Compiled/Fixed-Eager `0,9296309524`,
KI `[0,9256302629; 0,9327708433]`. Peak-RSS `3.771.564.032 B`, MLX-Peak
`3.476.049.782 B`, Swap-Delta `0 B`.

Nur berechnet: warm `0,9829777045` (rund 1,7022 % schneller), kalt
`1,0154895491` (rund 1,549 % langsamer), Break-even median rund 36,47
Decode-Schritte gegenüber 31 in diesem Lauf. Matmul war stets aktiv; die Studie
ändert nur Cacheform und MLX-Compile-Laufzeitorganisation.

## Zyklus 17 — historische Planung vor Hardware; inzwischen gemessen

Der Draft `fixed_compiled_batched_readback_n8_v1` vergleicht Readback `1` und
`8` ausschließlich auf dem qualifizierten Fixed-Compiled-4B-Pfad. Sechs
gepaarte frische Prozesse mit zwölf geplanten Arm-Ausführungen sind vorgesehen;
Modell, Gewichte, Quantisierung und Matmul bleiben unverändert. EOS-Tail wird
vollständig erfasst und getrimmt, exakte logische Token-/Textidentität ist
terminales Gate. Der Lauf ist inzwischen gemessen und endete mit
`no_clear_speedup_baseline_retained`; `formal_claim=false`. Cycle 7 `12,98 %`
bleibt explorativ.

## Zyklus 17 — historischer sealed_pre_hardware-Stand

`measured=false`, `formal_claim=false`, `authorization=reserved_not_consumed`.
Kein Modell-/GPU-/Hardwarelauf, Marker oder Resultat. Readback 1 versus 8 bleibt
die einzige Variable im identischen Fixed-Compiled-4B-Pfad; sechs frische Paare
und zwölf Arme sind geplant.

## Zyklus 17 — gemessener negativer Befund

Decode-Median/MAD: Readback 1 `0,2662952915/0,0002979590 s`, Readback 8
`0,2551394585/0,0002100205 s`; Readback `0,133416793`/`0,018026520 s`,
TTFT `0,640586125`/`0,7182502295 s`, Prefill `0,6383028335`/`0,638542875 s`,
Arm-Wall `1,7003978955`/`1,683269416 s`, Tokenrate `86,3702926`/`94,0662085 s⁻¹`.
Ratio-Median `0,9581074518`, KI `[0,9534714914;0,9598849359]`; 4,1893 % ist
berechnet. Readback 8 war stets schneller, die feste 5-%-Schwelle aber verfehlt;
Baseline retained, `formal_claim=false`. Der 8er-TTFT ist wegen der Boundary später.

## Zyklen 18–20 — terminale No-Load-Evidenz

Cycle 18 (`fused-greedy-compile-20260825-01`) lud kein Modell: Parent und Worker
hatten unterschiedliche Environment-Fingerprints; Status und Entscheidung
`resource_or_budget_failed`, `0` Läufe. Evidence-Commit
`dc2cdced58b629e6a39cb8ed870d847d8ee16c13`, Result-SHA
`ea644a912c9bb20a9fc992d7e24bfecfbb70285f2788ee83a15aeb4937503035`, Marker-SHA
`000bf298a3e03d51a84abe8087edfb51b173202451dbfed01bcac11607d3a6fd`.
Cycle 19 (`fused-greedy-compile-20260825-02`) blieb bei `load_count=0`, weil
die eigene Ergebnisdatei die Git-Bindung verletzte. Evidence-Commit
`59bbe9d698d978dcbd621fe89fb17bf98b286b8a`, Result-SHA
`4e02221975f6f1710e96dc70f69b4df6f48a1d93df859c6274ed83460dee0320`, Marker-SHA
`59525fe94e2705f56191f6ae6b9f0eb2f53ca36fa17442cd20fad70514df03e1`.
Cycle 20 (`fused-greedy-compile-20260825-03`) blieb ebenfalls bei `load_count=0`,
weil `dev` im Parent-Manifest, nicht aber im Worker-Manifest stand. Evidence-Commit
`78f983c71636637b7995eb90500fe689cbe53fee`, Result-SHA
`72e7e0692136766bcd5cea4147f3c106ad64de8ddadba855767d8908ae53200d`, Marker-SHA
`e2bbff9fad7aa6e3a8e1e16cb2d9ec884c05a1b8bc5a2fabc1225f06d5a0b9da`. Alle drei
bleiben `formal_claim=false`; es gab keinen Performancewert und keine Wiederholung.

## Zyklus 21 — gemessene Fused-Greedy-Baseline

Die Studie `fused-greedy-compile-20260825-04` wurde genau einmal gemessen:
Seal-Commit `ad4c92f32e608a8a0870b37e23a4dba0da1f666c`, Evidence-Commit
`4f89e51c3933aa9c9d42563393589da3c2e4a875`, Prereg-SHA
`a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf`, Result-SHA
`55bad770baad66cbebb804288845e9cf2785c0969c77355731ab8a23b3a43a2e`, Marker-SHA
`1c1dc10670c153c4c7430f3320671c08a3d56114e0fc5ee6af988c750ceb14e4`.
Sechs Paare und 12 Arm-Ausführungen bestanden alle Identitäts-, Ressourcen- und
Budgetgates. Gemessen wurden:

| Messgröße | External Greedy | Fused Greedy |
| :--- | ---: | ---: |
| Decode Median / MAD | `0,266399792 / 0,0005513755 s` | `0,2660886875 / 0,0001259585 s` |
| TTFT Median / MAD | `0,641516396 / 0,000482313 s` | `0,641348646 / 0,0006170835 s` |
| Modellzeit Median / MAD | `0,2659206645 / 0,00050935 s` | `0,265599773 / 0,0001226065 s` |
| Tokenrate Median / MAD | `86,3365071 / 0,1789603` | `86,4373498 / 0,0409238 Token/s` |

Die Fused-Einzelmedianmessung ist rund `0,117 %` niedriger, aber nicht das
gepaarte Entscheidungskriterium. Berechnet: Fused/External `1,000510010`
(`+0,0510 %` langsamer), Bootstrap-95-%-KI `[0,981178182; 1,004700679]`, Seed
`20260825`, 10.000 Resamples. Deshalb lautet die Entscheidung
`fused_greedy_compile_inconclusive`, Baseline retained, `formal_claim=false`.
Peak-RSS `3.769.974.784 B`, MLX-Peak `3.524.169.562 B`, Swap-Delta `0 B`.
Matmul blieb in allen Armen vollständig aktiv; Modell, Gewichte und Quantisierung
blieben unverändert. Ein Matmul-Aus-Vergleich wurde nicht durchgeführt und wäre
wegen geänderter Berechnung kein semantisch-identischer Kandidat.

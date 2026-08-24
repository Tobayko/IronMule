# Performance-Baseline

Stand: 24. August 2026, Vor-Hardware-Status Zyklus 16, nach Zyklus 15 und begrenzter Runtime-Qualifikation,
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

## Zyklus 16 — Versiegelte Vor-Hardware-Nulllinie (noch keine Messung)

Für die am 24.08.2026 freigegebene Studie
`matmul-compile-ab-20260824-01` ist der Kandidat
`fixed_cache_compiled_decode_v1` ist im lokalen Seal-Commit versiegelt, aber noch
nicht gemessen. Die geplanten Arme sind
`standard_eager`, `fixed_eager` und `fixed_compiled`. In jedem Arm bleibt die
mathematische Matmul aktiv; nur Cache-Form und Compile-Umgebung werden verglichen.
Es gibt deshalb keinen Matmul-Aus-Pfad und noch keinen Geschwindigkeitswert.

Modell, Gewichte und Quantisierung bleiben unverändert. Greedy Token- und
Textidentität muss exakt sein. Die alten Device-Model-Compile-Messungen sind
wegen falscher Token ab Position 2 ungültig und werden aus der Baseline
ausgeschlossen. `formal_claim=false`; ein negatives Ergebnis ist gültig.
Präregistrierungs-SHA-256:
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`. Es gibt
keine Ergebnisdatei und keine Startmarke.

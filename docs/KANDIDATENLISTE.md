# Kandidatenliste

Studienakte, kein Backlog: offene Arbeit steht ausschließlich in
[`../BACKLOG.md`](../BACKLOG.md). Diese Tabelle dokumentiert Kandidaten mit
ihren gemessenen bzw. terminalen Status (früher `EXPERIMENT_BACKLOG.md` im
Root; verschoben am 2026-09-01, Backlog M1 Punkt 2).

Stand: 25. August 2026, Zyklus 21 nach realer Runtime-Qualifikation, nach Zyklus 20.
Priorität nach erwarteter Wirkung je Aufwand, unter Berücksichtigung dessen, was
bereits gemessen ist.

| # | Kandidat | Mechanismus | Risiko | Status |
| ---: | :--- | :--- | :--- | :--- |
| 1 | exakte Präfix-/KV-Cache-Wiederverwendung | Prefill des stabilen Präfixes entfällt | Tokenidentität | **`candidate_correctness_failed`** (Zyklus 1) |
| 2 | Blockgrößen-Policy für Tokenidentität | Prefill nur in Breiten zerteilen, die die Ausgabe erhalten | kein längenunabhängig sicherer Wert gefunden | **`candidate_correctness_failed`** (Zyklus 2) |
| 3 | persistenter Modellprozess | Importe und Modellladen entfallen je Anfrage | gering | **`engineering_gain_confirmed_exact_scope`** (`−65,3032 %`, Zyklus 13) |
| 4 | deterministischer Warm-up | erster Lauf zahlt Allokation und Kernelaufbau | gering | teilweise umgesetzt in Messwerkzeugen |
| 5 | Prefill-Step-Size-Sweep | `2048` gegen `512` inkonsistent im Code | ändert Blockstruktur → Kandidat 2 zuerst | offen |
| 6 | Token-Cache für statische Präfixe | Tokenisierung `0,044`–`0,649` ms | – | **verworfen**: kein messbarer Anteil |
| 7 | Shape-Buckets Batch 1–32 | Breitenkurve ist Treppenfunktion | gemessen, Policy vorhanden | `candidate_characterized` |
| 8 | adaptives Microbatching | Plateau `8`–`32` nutzen | Tokenidentität | blockiert durch Zyklus 2 und 4 |
| 9 | Continuous Batching | Anfragen laufend ein- und ausklinken | hoch, Zustandsverwaltung | blockiert durch Zyklus 2 und 4 |
| 10 | N-Gram Speculative Decoding | Entwurf aus dem Kontext, kostenlos | Tokenidentität geprüft | `candidate_characterized`, umgesetzt |
| 11 | Draft-Model Speculative Decoding | 1B entwirft für 4B | – | **verworfen**: `0,560x` gemessen |
| 12 | `mx.compile` für Decode-Teilgraphen | `−23,8 %` Dispatch | **falsche Token ab Position 2** | **verworfen**; alte Device-Model-Compile-Messungen ungültig |
| 13 | KV-Cache fester Form | macht `mx.compile` gültig | Framework-Eingriff | offen, benötigt Cache-Neubau und Architekturfreigabe |
| 14 | Custom Metal Kernel | – | – | **verworfen**: Zyklus 9 lokalisiert keinen einzelnen Kernelengpass |
| 15 | vLLM-Metal-Vergleich | Paged KV, Prefix-Cache | – | `permission_required` |
| 16 | llama.cpp-Vergleich | zweite Referenz | andere Quantisierung | `permission_required` |
| 17 | Host-Readback aufschieben | vollständigen Token-Readback aus dem kritischen Pfad nehmen | kann ohne Readback nicht stoppen | **`candidate_recommended_for_preregistration`**, nur Obergrenze (Zyklus 6) |
| 18 | gebündelter Readback | Stop-Token nur alle `N` Schritte zum Host lesen | Überlauf bis `N-1` Token | **`no_clear_speedup_baseline_retained`** (Cycle 17; 4,1893 % berechnet, feste <5-%-Schwelle verfehlt) |
| 19 | LM-Head beim Prefill überspringen | nur die tatsächlich gelesene letzte Promptposition projizieren | unzulässig bei Prompt-Logprobs | **`engineering_go_exact_scope`** nach formalem Gewinn (`−15,3615 %`) und Runtime-Gate (`−15,4164 %`) |
| 20 | `logsumexp` bei greedy überspringen | argmax-invariante Normalisierung entfernen | isolierte Kosten sind nicht Grenzkosten | `candidate_characterized`, kein Gewinn (Zyklus 10) |
| 21 | KV-Cache-Reallokationen | Wachstumskopien im Decode vermeiden | erster Decodeschritt konfundiert; Cache-Neubau wäre Architekturänderung | **`candidate_recommended_for_preregistration`** (Zyklus 11) |
| 22 | lernendes Optimization Memory mit lokalem Planner | nutzt alle positiven und negativen Messungen für den nächsten Vorschlag | Selbstbestätigung und falsche Aktivierung | **`no_planner_qualified`** (Zyklus 15); 1B und 4B jeweils `0/6` im strikten Vertrag |
| 23 | Gemma-Matmul-A/B „mit/ohne“ | vollständigen Matmul-Optimierungspfad gegen unveränderten Pfad vergleichen | kein echter Schalter oder vollständiger A/B-Pfad vorhanden | **`open_future_preregistration`**; bisher nicht gemessen, neue Studie erforderlich |
| 24 | Fixed-Cache/Compile-Decode-A/B | Laufzeitumgebung mit festem KV-Cache vergleichen | Tokenidentität, Cache- und Compile-Vertrag | **`runtime_compile_wins_exact_scope`** (Zyklus 16; 18 Arm-Ausführungen gemessen) |
| 25 | Fused Greedy innerhalb der Compile-Umgebung | identisches greedy `argmax` innerhalb statt außerhalb des kompilierten Körpers | nur gepaarte Laufzeit entscheidet; Matmul bleibt aktiv | **`fused_greedy_compile_inconclusive`** (Zyklus 21; Baseline retained) |

## Begründung der Reihenfolge

Kandidat 2 war Voraussetzung für 1, 5, 8 und 9. Zyklus 2 fand keine zuverlässig
tokenidentische Blockgröße; Zyklus 4 zeigte, dass die Abweichungen Antworten ändern.
Die vier Kandidaten bleiben deshalb unter dem bestehenden Vertrag gesperrt.

Die priorisierte versiegelte Studie für **LM-Head beim Prefill überspringen** ist in
Zyklus 12 abgeschlossen: `R=0,846385`, 95-%-KI gesamt
`[0,843147; 0,851284]`, Effekt `−15,3615 %`, alle C-/V-/Gesamt-Gates und alle zwölf
Tokenidentitätsgates bestanden. Die anschließende begrenzte Runtime-Qualifikation
bestätigte `R=0,845836`, Effekt `−15,4164 %`, exakte Tokenidentität und alle fünf
Engineering-Gates. Der getrennte Repository-Aufruf ist nur für diesen exakten Fall
freigegeben; eine allgemeine oder automatische Produktaktivierung bleibt
ausgeschlossen.

## Zyklus 16 — finaler Vor-Hardware-Stand

Der Kandidat `fixed_cache_compiled_decode_v1` ist mit der eingefrorenen
Präregistrierung im lokalen Seal-Commit auf `sealed_pending_hardware` gesetzt.
SHA-256: `dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Es wurden keine Hardwarewerte gemessen; `results.json` und Startmarke fehlen.
Die Matmul bleibt in allen drei Armen aktiv. `formal_claim=false`.

## Zyklen 18–20 — terminale No-Load-Ergebnisse

Cycle 18 (`fused-greedy-compile-20260825-01`) endete vor dem Load wegen eines
Parent/Worker-Environment-Hash-Mismatch (`terminal provenance identity failed`),
`resource_or_budget_failed`, `0` Läufe; Evidence-Commit
`dc2cdced58b629e6a39cb8ed870d847d8ee16c13`, Result-SHA
`ea644a912c9bb20a9fc992d7e24bfecfbb70285f2788ee83a15aeb4937503035`, Marker-SHA
`000bf298a3e03d51a84abe8087edfb51b173202451dbfed01bcac11607d3a6fd`.
Cycle 19 (`fused-greedy-compile-20260825-02`) endete mit `load_count=0`, weil
der Worker seine eigene Ergebnisdatei als unerlaubte Git-Änderung sah; ebenfalls
`resource_or_budget_failed`, Evidence-Commit
`59bbe9d698d978dcbd621fe89fb17bf98b286b8a`, Result-SHA
`4e02221975f6f1710e96dc70f69b4df6f48a1d93df859c6274ed83460dee0320`, Marker-SHA
`59525fe94e2705f56191f6ae6b9f0eb2f53ca36fa17442cd20fad70514df03e1`.
Cycle 20 (`fused-greedy-compile-20260825-03`) endete mit `load_count=0`, weil
das Parent-Stat-Manifest `dev` enthielt und der Worker dieses Feld ausließ;
Evidence-Commit `78f983c71636637b7995eb90500fe689cbe53fee`, Result-SHA
`72e7e0692136766bcd5cea4147f3c106ad64de8ddadba855767d8908ae53200d`, Marker-SHA
`e2bbff9fad7aa6e3a8e1e16cb2d9ec884c05a1b8bc5a2fabc1225f06d5a0b9da`. Alle drei
bleiben `formal_claim=false` und wurden nicht wiederholt.

## Zyklus 21 — Fused-Greedy-Ergebnis

Der neue Kandidat `fixed_compiled_fused_greedy` wurde genau einmal mit sechs
Paaren auf dem unveränderten Gemma-4B-Snapshot gemessen. Seal-Commit
`ad4c92f32e608a8a0870b37e23a4dba0da1f666c`, Evidence-Commit
`4f89e51c3933aa9c9d42563393589da3c2e4a875`, Prereg-SHA
`a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf`, Result-SHA
`55bad770baad66cbebb804288845e9cf2785c0969c77355731ab8a23b3a43a2e`, Marker-SHA
`1c1dc10670c153c4c7430f3320671c08a3d56114e0fc5ee6af988c750ceb14e4`.
Status und Entscheidung sind `fused_greedy_compile_inconclusive`; die Baseline
bleibt retained, `formal_claim=false`. Gemessen: Decode-Median External/Fused
`0,266399792/0,2660886875 s`, TTFT `0,641516396/0,641348646 s`, Modellzeit
`0,2659206645/0,265599773 s`, Tokenrate `86,3365071/86,4373498 Token/s`.
Die einzelne Fused-Medianmessung ist rund `0,117 %` niedriger, aber das ist kein
gepaarter Gewinn. Berechnet: gepaartes Verhältnis `1,000510010` (`+0,0510 %`
langsamer), Bootstrap-KI `[0,981178182; 1,004700679]`, 10.000 Resamples,
Seed `20260825`. Alle 12 Arm-Ausführungen hatten identische physische,
logische, sichtbare Tokens und identischen Text; RSS-Peak `3.769.974.784 B`,
MLX-Peak `3.524.169.562 B`, Swap-Delta `0 B`.

Matmul blieb vollständig aktiv. Der Kandidat verschiebt nur den identischen
greedy Argmax in die Compile-Umgebung; Matmul-Aus wurde nicht getestet und wäre
keine semantisch identische Umgebung. Nächste Einzelkandidatenarbeit soll den
gemessenen Prefill/Kaltstart-Engpass und die fehlenden Multi-Turn- sowie
Parallel-Request-Baselines adressieren. Keine automatische Aktivierung.

## Zyklus 17 — Ergebnis

`measured=true`, Freigabe `consumed_exactly_once`, Entscheidung
`no_clear_speedup_baseline_retained`, `formal_claim=false`. Readback 8 war in
allen Paaren schneller, aber `0,9581074518` verfehlte die feste 5-%-Schwelle;
der 4,1893-%-Effekt ist berechnet. Der negative Befund ist gültig.

## Zyklus 17 — historischer sealed_pre_hardware-Stand

Offline versiegelt: `measured=false`, `formal_claim=false`,
`authorization=reserved_not_consumed`. Kein Modell-/Hardwarelauf, Marker oder
Resultat. Readback 1 versus 8 bleibt die einzige Variable im identischen
Fixed-Compiled-4B-Pfad; sechs frische Paare und zwölf Arme sind geplant.

## Zyklus 17 — historischer reservierter Pre-Hardware-Draft

`fixed-compiled-batched-readback-20260824-01` /
`fixed_compiled_batched_readback_n8_v1` ist `draft_pending_preflight`.
Die Antwort „Dann machen wir das mal“ reserviert genau einen Lauf; die Freigabe
ist noch nicht verbraucht. Nur Readback `1` versus `8` wird auf demselben
Fixed-Compiled-4B-Pfad variiert: sechs gepaarte frische Prozesse, zwölf geplante
Arm-Ausführungen. EOS-Tail wird vollständig gemessen und verworfen, exakte
logische Token-/Textidentität bleibt terminales Gate. Cycle 7 `12,98 %` bleibt
explorativ. Noch kein Marker, Ergebnis oder Modelllauf; `formal_claim=false`.

## Zyklus 16 — reales Ergebnis

`fixed_cache_compiled_decode_v1` erreicht im exakt begrenzten Fall die Entscheidung
`runtime_compile_wins_exact_scope`: sechs frische Prozesse, 18 Arm-Ausführungen
(3 × 6), identische Token und Texte. Gemessen wurden Decode-Medianen Standard
`0,399939187 s`,
Fixed-Eager `0,3999597295 s`, Fixed-Compiled `0,371848789 s`; die Ratios gegen
Standard und Fixed-Eager sind `0,9295921887` bzw. `0,9296309524` mit den
vorregistrierten Bootstrap-KIs. Peak-RSS `3.771.564.032 B`, MLX-Peak
`3.476.049.782 B`, Swap-Delta `0 B`.

Die warme/kalt-Projektion (`0,9829777045`/`1,0154895491`) und Break-even rund
`36,47` Schritte sind berechnet, nicht separat gemessen. Matmul blieb aktiv;
kein Gewichts-, Modell- oder Quantisierungswechsel. Kein allgemeiner Qualitäts-,
Selbstlern- oder Produktivclaim, keine automatische Aktivierung. Freigabe genau
einmal verbraucht; kein zweiter Lauf.

Die Vor-Hardware-Review behob drei P1-Ursachen: lazy Compile-/Fixed-Eager-Fehler
werden bis zur Synchronisierung korrekt als Kandidatenfehler klassifiziert,
Worker-Timeouts beachten die verbleibende Gesamt-Walltime, und beobachtete
Armzeit wird von akzeptierter Budgetbuchung getrennt. Keine dieser Korrekturen
ist ein Messergebnis oder eine Performanceaussage.

Der **persistente Prozess** ist in Zyklus 13 prospektiv bestätigt. Über sechs
vorgegebene Paare lag `warm/kalt` im Median bei `0,346968`, entsprechend einem aus
den Paaren gerechneten Effekt von `−65,3032 %`. Alle sechs greedy Tokenfolgen waren
exakt identisch. Der Median der gemessenen TTFT-Werte sank von `5148,7741` auf
`1785,1103` ms; Peak-RSS `3.763.077.120` Byte, kein RSS- oder Swap-Wachstum. Das ist
noch keine allgemeine Aktivierung und bleibt `formal_claim=false`.

Die neue Zwei-Modell-Studie `dual-model-evidence-planner-20260824-01` in Zyklus
15 ist ein gültiges negatives Ergebnis. Sie führte sechs balancierte Paare in
zwölf frischen seriellen Prozessen aus. Sowohl 1B als auch 4B waren intern `6/6`
deterministisch, erfüllten den strikten Vertrag aber jeweils `0/6`. 1B lieferte
Markdown, den falschen Schlüssel `persistent_service_id` und
`<end_of_turn>`-Trailer; 4B lieferte die richtige ID in einem unerlaubten
Markdown-Codeblock. Die dekodierten Texte waren zwischen den Modellen in `0/6`
Paaren bytegleich. Das sind Vertragsbefunde, keine qualitative Bewertung. Die
unveränderte Entscheidung ist `no_planner_qualified`; kein Kandidat wird
gestartet und nichts aktiviert. Cycle 14 bleibt separat und unverändert
`planner_contract_failed`.

Die Readback-Studien sind mit Cycle 17 abgeschlossen und treffen nur den Decode.
Zyklus 11 lokalisiert außerdem
`4,4263 %` korrelierte Decode-Grenzkosten, isoliert wegen der Überlagerung mit dem
ersten Decodeschritt aber noch keinen kausalen Gewinn. Das registrierte Zykluslimit
ist nun `17`; die Freigabe für Zyklus 17 ist exakt einmal verbraucht. Jede weitere
Hardwarestudie verlangt einen neuen expliziten Studien-/Zyklusvertrag und eine
neue ausdrückliche Freigabe.

Für das gewünschte eigenständige Lernen ist kein Download nötig: lokale 1B- und
4B-Snapshots sind vollständig vorhanden. Zyklus 15 zeigt im engen Fall, dass
beide Modelle den strikten Maschinenvertrag verfehlen; ein 1B- oder 4B-Planer
darf daraus nichts starten. Ein neuer Versuch müsste als eigener Kandidat eine
technisch erzwungene Auswahl aus festen IDs vorregistrieren. Messungen bleiben
alleiniger Richter; das Modell darf weder Schwellen ändern noch selbst aktivieren.

Ein echter Gemma-Pfad mit Matmul-Optimierungsschalter und vollständigem
„mit/ohne Matmul“-A/B-Vergleich existiert nicht. Dieser Vergleich wurde deshalb
nicht gemessen und wird nicht aus unabhängigen Matmul-Mikrobenchmarks abgeleitet.
Er bleibt als Kandidat 23 für eine separate, vorregistrierungspflichtige Studie
offen. Multi-Turn-Fortsetzung und mehrere parallele Requests sind ebenfalls noch
keine Messungen.

Noch ungemessen sind die verlangten Baseline-Workloads **Multi-Turn-Fortsetzung** und
**mehrere parallele Requests**. Der vorhandene Matrixeintrag `concurrent_32` ist nur
eine Workload-Definition, kein Messergebnis.

Kandidat 14 bleibt gesperrt: der Engpass ist Prefill, nicht ein Kernel-Hotspot, und der
Auftrag verbietet Kerneloptimierung ohne Profilerbeleg.

Zyklus 16 ist mit der Studie `matmul-compile-ab-20260824-01` und dem Kandidaten
`fixed_cache_compiled_decode_v1` genau für diesen runtime-only Vergleich
vorregistriert. Die mathematische Matmul bleibt in `standard_eager`,
`fixed_eager` und `fixed_compiled` aktiv; Modell, Gewichte und Quantisierung
bleiben unverändert. Exakte greedy Token- und Textidentität ist Pflicht,
`formal_claim=false`. Die Präregistrierung ist im lokalen Seal-Commit eingefroren
und es gibt noch keine Messung. Ein negatives Ergebnis zählt als gültiger Abschluss. Die
alten Device-Model-Compile-Werte werden wegen falscher Token ab Position 2
ausgeschlossen.

## Zyklus 16 — finaler Seal-Stand

Der lokale Commit mit dem final geprüften Stand ist der Seal-Commit; der Status
lautet `sealed_pending_hardware`. Präregistrierungs-SHA-256:
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Es wurden keine Hardwarewerte gemessen; `results.json` und Startmarke fehlen.
Die Matmul bleibt in allen drei Armen aktiv. `formal_claim=false`.

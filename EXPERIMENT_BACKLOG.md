# Kandidatenliste

Stand: 24. August 2026, nach Zyklus 15 und begrenzter Runtime-Qualifikation.
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
| 12 | `mx.compile` für Decode-Teilgraphen | `−23,8 %` Dispatch | **falsche Token** | **verworfen**, Ursache dokumentiert |
| 13 | KV-Cache fester Form | macht `mx.compile` gültig | Framework-Eingriff | offen, benötigt Cache-Neubau und Architekturfreigabe |
| 14 | Custom Metal Kernel | – | – | **verworfen**: Zyklus 9 lokalisiert keinen einzelnen Kernelengpass |
| 15 | vLLM-Metal-Vergleich | Paged KV, Prefix-Cache | – | `permission_required` |
| 16 | llama.cpp-Vergleich | zweite Referenz | andere Quantisierung | `permission_required` |
| 17 | Host-Readback aufschieben | vollständigen Token-Readback aus dem kritischen Pfad nehmen | kann ohne Readback nicht stoppen | **`candidate_recommended_for_preregistration`**, nur Obergrenze (Zyklus 6) |
| 18 | gebündelter Readback | Stop-Token nur alle `N` Schritte zum Host lesen | Überlauf bis `N-1` Token | **`candidate_recommended_for_preregistration`** (Zyklus 7) |
| 19 | LM-Head beim Prefill überspringen | nur die tatsächlich gelesene letzte Promptposition projizieren | unzulässig bei Prompt-Logprobs | **`engineering_go_exact_scope`** nach formalem Gewinn (`−15,3615 %`) und Runtime-Gate (`−15,4164 %`) |
| 20 | `logsumexp` bei greedy überspringen | argmax-invariante Normalisierung entfernen | isolierte Kosten sind nicht Grenzkosten | `candidate_characterized`, kein Gewinn (Zyklus 10) |
| 21 | KV-Cache-Reallokationen | Wachstumskopien im Decode vermeiden | erster Decodeschritt konfundiert; Cache-Neubau wäre Architekturänderung | **`candidate_recommended_for_preregistration`** (Zyklus 11) |
| 22 | lernendes Optimization Memory mit lokalem Planner | nutzt alle positiven und negativen Messungen für den nächsten Vorschlag | Selbstbestätigung und falsche Aktivierung | **`no_planner_qualified`** (Zyklus 15); 1B und 4B jeweils `0/6` im strikten Vertrag |
| 23 | Gemma-Matmul-A/B „mit/ohne“ | vollständigen Matmul-Optimierungspfad gegen unveränderten Pfad vergleichen | kein echter Schalter oder vollständiger A/B-Pfad vorhanden | **`open_future_preregistration`**; bisher nicht gemessen, neue Studie erforderlich |

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

Unter den noch unbestätigten empfohlenen Leistungskandidaten bleiben die beiden
Readback-Studien; sie treffen nur den Decode. Zyklus 11 lokalisiert außerdem
`4,4263 %` korrelierte Decode-Grenzkosten, isoliert wegen der Überlagerung mit dem
ersten Decodeschritt aber noch keinen kausalen Gewinn. Das registrierte Zykluslimit
ist nun `15`; die Freigabe für Zyklus 15 ist verbraucht. Jede weitere
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

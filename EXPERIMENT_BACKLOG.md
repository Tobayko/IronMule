# Kandidatenliste

Stand: 24. August 2026, nach Zyklus 11. Priorität nach erwarteter Wirkung je Aufwand, unter
Berücksichtigung dessen, was bereits gemessen ist.

| # | Kandidat | Mechanismus | Risiko | Status |
| ---: | :--- | :--- | :--- | :--- |
| 1 | exakte Präfix-/KV-Cache-Wiederverwendung | Prefill des stabilen Präfixes entfällt | Tokenidentität | **`candidate_correctness_failed`** (Zyklus 1) |
| 2 | Blockgrößen-Policy für Tokenidentität | Prefill nur in Breiten zerteilen, die die Ausgabe erhalten | kein längenunabhängig sicherer Wert gefunden | **`candidate_correctness_failed`** (Zyklus 2) |
| 3 | persistenter Modellprozess | Importe und Modellladen entfallen je Anfrage | gering | **`candidate_recommended_for_preregistration`** (Zyklus 5) |
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
| 19 | LM-Head beim Prefill überspringen | nur die tatsächlich gelesene letzte Promptposition projizieren | unzulässig bei Prompt-Logprobs | **`candidate_recommended_for_preregistration`** (Zyklus 8) |
| 20 | `logsumexp` bei greedy überspringen | argmax-invariante Normalisierung entfernen | isolierte Kosten sind nicht Grenzkosten | `candidate_characterized`, kein Gewinn (Zyklus 10) |
| 21 | KV-Cache-Reallokationen | Wachstumskopien im Decode vermeiden | erster Decodeschritt konfundiert; Cache-Neubau wäre Architekturänderung | **`candidate_recommended_for_preregistration`** (Zyklus 11) |

## Begründung der Reihenfolge

Kandidat 2 war Voraussetzung für 1, 5, 8 und 9. Zyklus 2 fand keine zuverlässig
tokenidentische Blockgröße; Zyklus 4 zeigte, dass die Abweichungen Antworten ändern.
Die vier Kandidaten bleiben deshalb unter dem bestehenden Vertrag gesperrt.

Von den empfohlenen Kandidaten hat **LM-Head beim Prefill überspringen** Vorrang für
eine versiegelte Studie: er trifft den gemessenen Engpass (`1,70` s Prefill gegen
`12,1` ms je Decodetoken). Der persistente Prozess trifft nur den Kaltstart, die
Readback-Kandidaten nur den Decode. Zyklus 11 lokalisiert zwar `4,4263 %` korrelierte
Decode-Grenzkosten, isoliert wegen der Überlagerung mit dem ersten Decodeschritt aber
noch keinen kausalen Gewinn.

Noch ungemessen sind die verlangten Baseline-Workloads **Multi-Turn-Fortsetzung** und
**mehrere parallele Requests**. Der vorhandene Matrixeintrag `concurrent_32` ist nur
eine Workload-Definition, kein Messergebnis.

Kandidat 14 bleibt gesperrt: der Engpass ist Prefill, nicht ein Kernel-Hotspot, und der
Auftrag verbietet Kerneloptimierung ohne Profilerbeleg.

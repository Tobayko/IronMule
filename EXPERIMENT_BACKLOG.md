# Kandidatenliste

Stand: 24. August 2026. Priorität nach erwarteter Wirkung je Aufwand, unter
Berücksichtigung dessen, was bereits gemessen ist.

| # | Kandidat | Mechanismus | Risiko | Status |
| ---: | :--- | :--- | :--- | :--- |
| 1 | exakte Präfix-/KV-Cache-Wiederverwendung | Prefill des stabilen Präfixes entfällt | Tokenidentität | **`candidate_correctness_failed`** (Zyklus 1) |
| 2 | Blockgrößen-Policy für Tokenidentität | Prefill nur in Breiten zerteilen, die die Ausgabe erhalten | offen, ob längenunabhängig lösbar | **offen, höchste Priorität** |
| 3 | persistenter Modellprozess | `1,47`–`1,76` s Ladezeit entfallen je Anfrage | gering | offen |
| 4 | deterministischer Warm-up | erster Lauf zahlt Allokation und Kernelaufbau | gering | teilweise umgesetzt in Messwerkzeugen |
| 5 | Prefill-Step-Size-Sweep | `2048` gegen `512` inkonsistent im Code | ändert Blockstruktur → Kandidat 2 zuerst | offen |
| 6 | Token-Cache für statische Präfixe | Tokenisierung `0,044`–`0,649` ms | – | **verworfen**: kein messbarer Anteil |
| 7 | Shape-Buckets Batch 1–32 | Breitenkurve ist Treppenfunktion | gemessen, Policy vorhanden | `candidate_characterized` |
| 8 | adaptives Microbatching | Plateau `8`–`32` nutzen | Tokenidentität | wartet auf BW1 |
| 9 | Continuous Batching | Anfragen laufend ein- und ausklinken | hoch, Zustandsverwaltung | offen |
| 10 | N-Gram Speculative Decoding | Entwurf aus dem Kontext, kostenlos | Tokenidentität geprüft | `candidate_characterized`, umgesetzt |
| 11 | Draft-Model Speculative Decoding | 1B entwirft für 4B | – | **verworfen**: `0,560x` gemessen |
| 12 | `mx.compile` für Decode-Teilgraphen | `−23,8 %` Dispatch | **falsche Token** | **verworfen**, Ursache dokumentiert |
| 13 | KV-Cache fester Form | macht `mx.compile` gültig | Framework-Eingriff | offen, benötigt Cache-Neubau |
| 14 | Custom Metal Kernel | – | – | **gesperrt**: kein Profilerbeleg für Kernelengpass |
| 15 | vLLM-Metal-Vergleich | Paged KV, Prefix-Cache | – | `permission_required` |
| 16 | llama.cpp-Vergleich | zweite Referenz | andere Quantisierung | `permission_required` |

## Begründung der Reihenfolge

Kandidat 2 steht oben, weil er **Voraussetzung** für 1, 5, 8 und 9 ist. Alle vier
verändern die Blockstruktur des Prefills, und Zyklus 1 hat gezeigt, dass genau das die
erzeugten Token verändern kann. Ohne eine Antwort auf Kandidat 2 kann keiner von ihnen
den Korrektheitsvertrag erfüllen.

Kandidat 14 bleibt gesperrt: der Engpass ist Prefill, nicht ein Kernel-Hotspot, und der
Auftrag verbietet Kerneloptimierung ohne Profilerbeleg.

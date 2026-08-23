# Nächster prospektiver Kandidat

Stand: 24. August 2026, nach Zyklus 5. Genau einer. Noch **kein** `formal_claim=true`.

## Empfehlung: persistenter Modellprozess

`candidate_recommended_for_preregistration` seit Zyklus 5. Der **einzige** Kandidat
dieses Auftrags, der ein Korrektheitsgate bestanden hat.

| Größe | Wert |
| :--- | ---: |
| `cold_process` TTFT | `5,053` s |
| warm TTFT | `1,747` s |
| entfernter Anteil | **`65,4 %`** |
| Tokenidentität | `3/3`, auch nach zwischengeschobenen Anfragen |
| RSS über fünf Anfragen | `3,77` GB, unverändert |

**Warum er als einziger besteht.** Er ändert nichts an der Numerik — keine
Blockgröße, keine Batchbreite, keine Cache-Struktur. Genau daran sind die drei
anderen Kandidaten gescheitert.

**Was zu registrieren wäre.** Eine eigene versiegelte Studie nach dem Muster von
H1-v2 und N10-v2: sechs A/A-Sessions, konservativer MDE-Boden, sechs A/B-Sessions mit
getrennter Charakterisierung und Validierung, eigene hashverkettete Historie, genau
ein Record mit `formal_claim`. Endpunkt ist `cold_process`-TTFT gegen `warm`-TTFT bei
sonst identischer Anfrage.

**Was er nicht löst.** Nach Entfernen der `3,31` s Startkosten bleibt der Prefill mit
`1,747` s als neuer Engpass. Dessen wirksamste Optimierung — Präfix-Wiederverwendung
mit `13,0x` — ist in Zyklus 4 dauerhaft gesperrt worden.

## Der Rest der Liste ist erschöpft

| Kandidat | Zustand |
| :--- | :--- |
| Präfix-Wiederverwendung | `correctness_failed` (Zyklus 1), endgültig durch Zyklus 4 |
| Blockgrößen-Policy | `correctness_failed` (Zyklus 2) |
| Prefill-Step-Sweep, Microbatching, Continuous Batching | gesperrt, ändern die Blockstruktur |
| `mx.compile` | `correctness_failed`, frühere Runde |
| Draft-Spekulation | `rejected`, `0,560x` |
| N-Gram-Spekulation | `characterized`, umgesetzt |
| Shape-Buckets | `characterized`, Policy vorhanden |
| Token-Cache | `rejected`, unter `0,1 %` der TTFT |
| Custom Metal | gesperrt, kein Profilerbeleg für Kernelengpass |
| KV-Cache fester Form | offen, Framework-Eingriff |
| vLLM, llama.cpp | `permission_required` |

Ohne Freigabe oder Framework-Arbeit bleibt **kein** weiterer Kandidat mit erwarteter
Wirkung, der den Korrektheitsvertrag erfüllen kann.

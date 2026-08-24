# Nächster prospektiver Kandidat

Stand: 24. August 2026, nach Zyklus 11. Noch **kein** `formal_claim=true`.

## Empfehlung: LM-Head beim Prefill überspringen

Fünf Kandidaten stehen inzwischen auf `candidate_recommended_for_preregistration`.
Dieser hat Vorrang, weil er als einziger den **gemessenen Engpass** trifft.

| Kandidat | Zyklus | Wirkung | trifft |
| :--- | ---: | ---: | :--- |
| **LM-Head beim Prefill überspringen** | 8 | **`15,3 %` des Prefills** | **TTFT** |
| Persistenter Prozess | 5 | `65,4 %` der `cold`-TTFT | Kaltstart |
| Gebündelter Readback | 7 | `12,98 %` je Token | Decode |
| Host-Readback (Obergrenze) | 6 | `15,3 %` | Decode, nicht abrufbar |
| KV-Cache-Reallokationen | 11 | `4,4263 %` korrelierter Decodeanteil | Decode, erster Schritt konfundiert |

Die Zyklen 1 bis 4 haben gezeigt, dass der Engpass der **Prefill** ist: `1,76` s
gegen `12,1` ms je Ausgabetoken. Ein Kandidat, der dort `15 %` holt, wiegt schwerer
als einer, der im Decode dasselbe holt.

**Was zu registrieren wäre.** Eine versiegelte Studie nach dem Muster von H1-v2 und
N10-v2: sechs A/A-Sessions, konservativer MDE-Boden, sechs A/B-Sessions mit getrennter
Charakterisierung und Validierung, eigene hashverkettete Historie, genau ein Record
mit `formal_claim`. Endpunkt sind Prefill-Sekunden bei identischen Token.

**Grenze, die mitregistriert werden muss.** Nur zulässig bei greedy Decoding ohne
Logprob-Ausgabe je Prompt-Token. Sobald Perplexität, Bewertung oder Logprobs verlangt
werden, ist der Kandidat nicht anwendbar.

## Die vier anderen empfohlenen Kandidaten

Sie schließen sich nicht aus und wirken an verschiedenen Stellen — Kaltstart, Prefill,
Decode. Eine gemeinsame Studie wäre allerdings **falsch**: sie würden sich in einem
gemeinsamen Endpunkt vermischen, und der Auftrag verlangt genau einen Kandidaten je
Studie. Der Zyklus-11-Kandidat benötigt für einen kausalen A/B-Test außerdem einen
Cache-/Framework-Eingriff und wartet deshalb auf die in `PERMISSION_REQUIRED.md`
beschriebene Architekturfreigabe.

## Was blockiert bleibt

| Kandidat | Grund |
| :--- | :--- |
| Präfix-Wiederverwendung (`13,0x`) | Korrektheit, Zyklus 1 und 4 |
| Blockgrößen-Policy | Korrektheit, Zyklus 2 |
| Prefill-Step-Sweep, Microbatching, Continuous Batching | ändern die Blockstruktur |
| `mx.compile` | Korrektheit, frühere Runde |
| Custom Metal | kein Profilerbeleg für einen Kernelengpass |
| KV-Cache fester Form | Framework-Eingriff |
| vLLM, llama.cpp, Energie, dichtes 7–9B | `permission_required` |

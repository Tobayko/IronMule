# Nächster prospektiver Kandidat

Stand: 24. August 2026, nach Zyklus 12. Das registrierte Zykluslimit `12` ist
erreicht; ohne neuen expliziten Studienvertrag wird kein weiterer Hardwarelauf
gestartet.

## Abgeschlossene Priorität: LM-Head beim Prefill überspringen

Die versiegelte Studie `head-skip-prefill-v1-20260824` bestätigte den priorisierten
Prefill-Kandidaten formal: `R=0,846385`, Effekt `−15,3615 %`, Gesamt-95-%-KI
`[0,843147; 0,851284]`, Charakterisierung und Validierung getrennt bestanden und
Greedy-Tokenidentität in `12/12` Sessiongates. Der Claim gilt nur für ein Gerät,
einen lokalen Modell-Snapshot, einen 897-Token-Prompt, Prefill-Chunk `256`, Batch `1`
und greedy ohne Prompt-Logprobs. Die begrenzte Integration wurde freigegeben und
bestand ihre vorregistrierte Runtime-Qualifikation mit `R=0,845836`, Effekt
`−15,4164 %` und identischen Token. Sie ist nur über den getrennten
Repository-Aufrufpunkt und nur im registrierten Fall zulässig; eine allgemeine
Produktaktivierung ist weiterhin nicht erlaubt.

## Empfehlung bei einem neuen Zyklusvertrag: persistenter Prozess

| Kandidat | Zyklus | bisherige Wirkung | trifft |
| :--- | ---: | ---: | :--- |
| **Persistenter Prozess** | 5 | `65,4 %` der `cold`-TTFT | Kaltstart |
| Gebündelter Readback | 7 | `12,98 %` je Token | Decode |
| Host-Readback (Obergrenze) | 6 | `15,3 %` | Decode, nicht direkt abrufbar |
| KV-Cache-Reallokationen | 11 | `4,4263 %` korrelierter Decodeanteil | Decode, erster Schritt konfundiert |

Der persistente Prozess hat unter den verbleibenden Kandidaten Vorrang, weil seine
isolierte Zerlegung bereits `65,4 %` der Kaltstart-TTFT lokalisiert und kein neuer
Modell- oder Softwaredownload nötig wäre. Das ist noch **kein Grenzkostengewinn**;
eine neue Mini-Vorregistrierung müsste einen echten frischen Prozess gegen einen
persistenten Prozess vergleichen, Greedy-Tokenidentität gaten und Kaltstart sowie
Warmzustand getrennt halten.

Die zwei Readback-Kandidaten betreffen nur den Decode. Der KV-Kandidat benötigt für
einen kausalen A/B-Test einen Cache-/Framework-Eingriff und bleibt bis zur expliziten
Architekturfreigabe gesperrt. Kandidaten dürfen nicht in einer gemeinsamen Studie
vermischt werden.

Vor einer weiteren Optimierungsstudie sind außerdem die verlangten Baseline-Workloads
**Multi-Turn-Fortsetzung** und **mehrere parallele Requests** offen. Auch sie benötigen
einen neuen expliziten Zyklus-/Messvertrag.

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

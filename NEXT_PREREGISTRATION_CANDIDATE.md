# Nächster prospektiver Kandidat

Stand: 24. August 2026, nach Zyklus 14. Das registrierte Zykluslimit `14` ist
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

## Abgeschlossener 4B-Planertest: Formatvertrag verfehlt

Gemma 4B wurde in Zyklus 14 genau einmal als eng begrenzter Planer geprüft. Drei
frische Prozesse nannten inhaltlich jeweils die hier ohnehin erwartete nächste ID
`persistent_service_qualification` und erzeugten exakt dieselben Token. Alle drei
Antworten enthielten aber einen nicht erlaubten Markdown-Codeblock. Damit gilt
`planner_contract_failed`, nicht ein Erfolg. Die Auswahl wurde nicht ausgeführt.

## Abgeschlossene Priorität: persistenter Prozess

Die prospektive Studie `persistent-process-20260824-03` verglich einen neuen
Python-/Modellprozess je Anfrage mit einem einmal geladenen Modellprozess. Der
Median der sechs vorgegebenen Paarverhältnisse betrug `0,346968`, entsprechend
gerechnet `−65,3032 %`; alle sechs greedy Tokenfolgen waren exakt identisch. Die
gemessenen TTFT-Mediane lagen bei `5148,7741` und `1785,1103` ms. Alle Ressourcen-
und Budgetgrenzen bestanden, `formal_claim=false`. Das bestätigt den Mechanismus,
aktiviert aber noch keinen normalen Dienst.

## Empfehlung bei neuem Architektur- und Zyklusvertrag: begrenzter persistenter Dienst

Der nächste Schritt mit größtem erwartbarem Nutzwert ist nicht noch ein anderer
Rechentrick, sondern der rückrollbare Einbau des bestätigten Prozess-Lebenszyklus.
Die Qualifikation soll **denselben einen Kandidaten** auf den bislang fehlenden
Workloads prüfen:

- mehrere aufeinanderfolgende Gesprächsrunden mit frischem Anfrage-Cache und
  unveränderter Gesprächssemantik;
- mehrere parallele Anfragen mit fester, speichersicherer Obergrenze;
- sichtbarer Rückfall auf den bisherigen Start-pro-Anfrage-Pfad bei Fehler oder
  Speicherdruck.

Dies benötigt zuerst die Architekturfreigabe aus `PERMISSION_REQUIRED.md`. Erst
danach darf eine neue Vorregistrierung Schwellen, Reihenfolge, Speichergrenzen und
Tokenidentitätsgates festschreiben. Die Zyklus-13-Werte werden dabei nicht als
Erfolgsschwelle übernommen.

## Danach verbleibende prospektive Kandidaten

| Kandidat | Zyklus | bisherige Wirkung | trifft |
| :--- | ---: | ---: | :--- |
| Gebündelter Readback | 7 | `12,98 %` je Token | Decode |
| Host-Readback (Obergrenze) | 6 | `15,3 %` | Decode, nicht direkt abrufbar |
| KV-Cache-Reallokationen | 11 | `4,4263 %` korrelierter Decodeanteil | Decode, erster Schritt konfundiert |

Die zwei Readback-Kandidaten betreffen nur den Decode. Der KV-Kandidat benötigt für
einen kausalen A/B-Test einen Cache-/Framework-Eingriff und bleibt bis zur expliziten
Architekturfreigabe gesperrt. Kandidaten dürfen nicht in einer gemeinsamen Studie
vermischt werden.

Das gewünschte selbstlernende Optimization Memory ist eine getrennte
Architekturarbeit, kein zusätzlicher Leistungskandidat in derselben Studie. Lokale
1B- und 4B-Snapshots sind vorhanden. Freie 4B-Textausgabe hat den Maschinenvertrag
bereits verfehlt. Ein neuer Planerkandidat müsste deshalb die Ausgabe technisch auf
genau eine feste ID begrenzen; ein nachträglich großzügigerer Parser wäre kein
Wiederholen derselben Studie. Messprogramm und feste Tabelle bleiben alleinige
Richter.

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

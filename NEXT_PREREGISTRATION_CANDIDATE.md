# Nächster prospektiver Kandidat

Stand: 25. August 2026, Zyklus 17 nach realer Runtime-Qualifikation. Die frühere Freigabe für
Zyklus 15 ist verbraucht. Für genau eine neue runtime-only Studie wurde am
24.08.2026 ausdrücklich freigegeben; ihre Präregistrierung ist im Arbeitsbaum
im lokalen Seal-Commit eingefroren. Die Freigabe ist verbraucht; ein weiterer
Lauf benötigt eine neue ausdrückliche Freigabe.

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

## Abgeschlossene Zwei-Modell-Studie: kein Planer qualifiziert

In Zyklus 15 wurden Gemma 3 1B und 4B in sechs balancierten Paaren und zwölf
frischen seriellen Prozessen genau einmal geprüft. Beide Modelle waren innerhalb
des Modells `6/6` deterministisch, erfüllten den strikten Vertrag aber jeweils
`0/6`; die direkte Textgleichheit zwischen den Modellen lag bei `0/6`. Beim 1B
verursachten Markdown, der falsche Schlüssel `persistent_service_id` und
`<end_of_turn>`-Trailer den Vertragsfehler. Beim 4B war die ID inhaltlich richtig,
aber ebenfalls von einem unerlaubten Markdown-Codeblock umgeben. Die Entscheidung
lautet `no_planner_qualified`, `formal_claim=false`; die Auswahl wurde nicht
ausgeführt. Zyklus 14 bleibt davon getrennt und unverändert
`planner_contract_failed`.

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

## Kandidatenstatus nach Cycle 17

| Kandidat | Zyklus | bisherige Wirkung | trifft |
| :--- | ---: | ---: | :--- |
| Gebündelter Readback | 7/17 | Cycle 17: `4,1893 %` berechnet, feste 5-%-Schwelle verfehlt | abgeschlossen; Baseline retained |
| Host-Readback (Obergrenze) | 6 | `15,3 %` | Decode, nicht direkt abrufbar |
| KV-Cache-Reallokationen | 11 | `4,4263 %` korrelierter Decodeanteil | Decode, erster Schritt konfundiert |

## Abgeschlossener Zyklus 16 — runtime-only Matmul-Umgebungs-A/B

Studie: `matmul-compile-ab-20260824-01`
Kandidat: `fixed_cache_compiled_decode_v1`
Status: `runtime_compile_wins_exact_scope`; 18 Arm-Ausführungen gemessen (3 × 6)
Claim: `formal_claim=false`

Präregistrierungs-SHA-256: `dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Ergebnis-Hash: `fbcc2fc65ac5d255ed11039a74c34e9a02d942cec17b25a6ed863058e0073b57`;
Marker-Hash: `8adf6f9c2453524bd1e05f4973ee85f84a323e9461a3f9b996ec2d0f7fed3c2f`.

Der Test ändert keine Modellgewichte, kein Modell und keine Quantisierung. Die
mathematische Matmul bleibt in allen Armen aktiv; verglichen werden nur
`standard_eager`, `fixed_eager` und `fixed_compiled`. Exakte greedy Token- und
Textidentität ist Pflicht. Alte Device-Model-Compile-Messungen werden wegen
falscher Token ab Position 2 nicht verwendet. Ein negatives Ergebnis ist gültig.

Der gebündelte Readback ist durch Cycle 17 abgeschlossen; nur die Host-Readback-
Obergrenze bleibt diagnostisch und nicht direkt abrufbar. Der KV-Kandidat benötigt für
einen kausalen A/B-Test einen Cache-/Framework-Eingriff und bleibt bis zur expliziten
Architekturfreigabe gesperrt. Kandidaten dürfen nicht in einer gemeinsamen Studie
vermischt werden.

Das gewünschte selbstlernende Optimization Memory ist eine getrennte
Architekturarbeit, kein zusätzlicher Leistungskandidat in derselben Studie. Lokale
1B- und 4B-Snapshots sind vorhanden. Zyklus 15 hat aber gezeigt, dass beide Modelle
im engen Vertrag nicht qualifizieren. Ein neuer Planerkandidat müsste deshalb die
Ausgabe technisch auf genau eine feste ID begrenzen; ein nachträglich großzügigerer
Parser wäre kein Wiederholen derselben Studie. Messprogramm und feste Tabelle
bleiben alleinige Richter. Allgemeine Modellqualität, allgemeine Planner-Fähigkeit,
selbstlernende Runtime und Produktaktivierung sind nicht belegt.

Ein echter Gemma-Pfad mit einem „mit/ohne Matmul“-Schalter oder vollständigem
Matmul-A/B-Vergleich existiert derzeit nicht. Er wurde nicht gemessen und bleibt
als separater, vorregistrierungspflichtiger Kandidat offen. Multi-Turn-Fortsetzung
und mehrere parallele Requests fehlen weiterhin als Baselines.

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

## Zyklus 16 — abgeschlossen im engen Runtime-Scope

Die Studie `matmul-compile-ab-20260824-01` ist mit Seal-Commit
`83ee3ea03f9fb303b8226ab8ad3189f07daec727` abgeschlossen:
`runtime_compile_wins_exact_scope`, `formal_claim=false`. In 18 Arm-Ausführungen
(3 × 6) waren
Token und Text exakt identisch. Die gemessenen Decode-Medianen waren Standard
`0,399939187 s`, Fixed-Eager `0,3999597295 s` und Fixed-Compiled
`0,371848789 s`; warm/kalt insgesamt sind nur berechnete Projektionen
`0,9829777045`/`1,0154895491`. Break-even: berechnet rund 36,47 Schritte bei
31 ausgeführten Schritten.

Matmul blieb in allen Armen aktiv; Modell, Gewichte und Quantisierung wurden
nicht verändert. Die Freigabe ist genau einmal verbraucht. Kein allgemeiner
Qualitäts-, Selbstlern- oder Produktivclaim und keine automatische Aktivierung.

## Zyklus 17 — historischer Pre-Hardware-Draft (abgeschlossen)

Studie `fixed-compiled-batched-readback-20260824-01`, Kandidat
`fixed_compiled_batched_readback_n8_v1`, Status `draft_pending_preflight`.
Die Antwort „Dann machen wir das mal“ reservierte genau einen neuen Lauf; die
Freigabe wurde anschließend exakt einmal verbraucht. Historisch geplant: sechs gepaarte
frische Prozesse, zwölf Arm-Ausführungen, Readback `1` versus `8` als einzige
Variable auf identischem Fixed-Compiled-4B-Pfad. EOS-Tail wird vollständig
getaktet und getrimmt; exakte logische Token und sichtbarer Text sind terminal.
Der Lauf ist inzwischen abgeschlossen; `formal_claim=false`; keine
Aktivierung, kein Dienst, kein Multi-Turn- und kein Qualitätsclaim. Cycle 7
`12,98 %` bleibt explorative Historie.

## Zyklus 17 — historischer sealed_pre_hardware-Stand

`measured=false`, `formal_claim=false`, `authorization=reserved_not_consumed`.
Kein Modell-/MLX-/Hardwarelauf, Marker oder Resultat. Readback 1 versus 8 bleibt
die einzige Variable im identischen Fixed-Compiled-4B-Pfad; sechs frische Paare
und zwölf Arme sind geplant.

## Zyklus 17 — Ergebnis und nächste Richtung

Entscheidung `no_clear_speedup_baseline_retained`, `measured=true`,
`formal_claim=false`, Freigabe exakt einmal verbraucht. Readback 8 war stets
schneller, aber die feste 5-%-Schwelle wurde verfehlt. Als nächste Richtung
bleiben nur Multi-Turn-Fortsetzung und begrenzte parallele Requests im
persistenten Lifecycle/Prefill-Pfad; neue Architekturfreigabe und neue
Hardware-Studie sind erforderlich. Nichts wird als Selbstlernen behauptet.

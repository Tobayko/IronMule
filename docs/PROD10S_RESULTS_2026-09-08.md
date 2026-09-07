# PROD10-S — einstündiger 12B-Serverlauf bestanden

**Der vorregistrierte Lauf ist bestanden:** 3.600,372 Sekunden Dauerphase,
798 vollständige HTTP-Anfragen, keine sichtbare Ausgabeabweichung und kein
ungeplanter Neustart. Einschließlich Referenz-/Integrationssetup wurden
833 vollständige Anfragen aufgezeichnet. Beide eigenen Worker endeten normal
mit Exitcode 0.

## Was tatsächlich geprüft wurde

Apple M1 Max, 32 GiB, eingefrorener Gemma-3-12B-4bit-Snapshot und unveränderte
installierte Runtime `47c416fb…edf7b7b`. Die Stunde enthält je 266 Anfragen für
long8, short32 und long32; JSON/SSE wechseln sich ab. 534 einzelne Anfragen
und 66 Vierer-Batches ergeben 798 Anfragen, davon 264 mit vier gleichzeitigen
Clients. Der Backendpfad serialisiert die Modellarbeit weiterhin; dies ist
kein Nachweis von vier gleichzeitig rechnenden GPU-Modellinstanzen.

Alle HTTP-Antworten stimmen mit der tatsächlichen Stock-Referenz in Text-Hash,
Prompt-/Completion-Zahl und Finish-Grund überein. Die öffentliche Schnittstelle
liefert keine Token-IDs; die vollständigen Token-Hash-Vergleiche gehören zur
vorgeschalteten direkten Stock-/Produktmatrix, nicht zu jedem HTTP-Soak-Record.

Am Ende meldet der Dienst: 809 abgeschlossene HTTP-Anfragen (11 Setup/Recovery
plus 798 Dauerphase), genau ein beabsichtigter Cancel, null Fehler, null aktive
oder wartende Anfragen und `ready=true`. Derselbe Produktworker PID 59238
blieb erhalten; Stock-PID 59147 und Produkt-PID schließen beide mit 0.
Der bereits separat nativ geprüfte Streaming-Disconnect-Fix wird damit nicht
durch einen versteckten Restart umgangen.

Vorher-/Nachher-Bindungen für Modell/Runtime/Hardware, Quellmanifest, Provider,
installierte Pakete und Modellmetadaten stimmen vollständig überein. Das
Hash-Ketten-Journal enthält für diesen Lauf 4.520 Ereignisse. Alle 833 Samples
stimmen mit dem Rohbericht überein; dessen kanonischer Digest stimmt mit dem
terminalen Journalereignis überein. Keine Ressourcenbeobachtung enthält Fehler.

## Speicher: aktuelle Werte, nicht nur Spitzenwerte

Die folgende Auswertung verwendet `physical_footprint_bytes` des Modellworkers.
Ein Lifetime-Highwater-Zähler wäre dafür ungeeignet. Die vier gleichen Zeitfenster
liegen zwischen erster und letzter vollständig protokollierter Soak-Antwort:
7. September 2026, 21:13:07–22:13:01 UTC, entsprechend 23:13:07–00:13:01
Europe/Berlin über den Tageswechsel. Das sind 3.593,722 s; die Zeit bis zur
ersten Antwort ist in dieser Trendtabelle nicht enthalten, wohl aber im
vollständigen Dauerlauf und seinen Rohdaten.

| Viertel | Beobachtungen | Median aktueller Modell-Footprint |
| --- | ---: | ---: |
| Q1 | 871 | 8,771 GB |
| Q2 | 871 | 8,781 GB |
| Q3 | 870 | 8,782 GB |
| Q4 | 865 | 8,782 GB |

Q4 minus Q1: **10.747.904 B = 10,25 MiB, etwa 0,123 %**. Das ist eine kleine
beobachtete Verschiebung in diesem Profil, kein allgemeiner Beweis für
Leak-Freiheit. Mediane sind über die protokollierten Messpunkte gebildet,
nicht zeitgewichtet. Der maximale Prozess-Footprint des gesamten Laufs betrug
9.112.590.456 B. RSS und Footprint werden nicht addiert.

Maximales beobachtetes **systemweites** Swapdelta: 258.736.128 B; am Ende
116.129.792 B. Swap war also nicht unverändert und kann nicht allein dem
Modell zugeschrieben werden. Es wurde gemäß Nutzerentscheid kein pauschales
RSS-/Swap-/Zeit-/Duty-Gate zur Beendigung angewendet.

## Bewertung und Grenzen

**Analyse-QA: mit den genannten Einschränkungen veröffentlichbar.** Die
Validierung korrigiert ausdrücklich die Verwechslung des aktuellen Footprints
mit dem Lifetime-Peak und prüft Zähler, Zeitfenster und Batch-Grundgesamtheit.
Der letzte Vierer-Batch umfasst Index 792–795, danach folgen nur 796 und 797
als einzelne Anfragen. Es wurden keine fehlgeschlagenen Soak-Antworten ausgefiltert.

Das Ergebnis gilt für ein Gerät, ein Modell und drei feste öffentliche
Workloads. Es belegt weder universelle Produktionsreife noch semantische
Modellqualität, Peak-Durchsatz, Multi-Mac-Betrieb oder einen Speedup. Server,
Controller und Testclients laufen im selben Elternprozess; dessen eigener
Speicher wurde nicht separat erfasst. Die obige Speicheraussage betrifft den
Modellworker, nicht pauschal die gesamte Serveranwendung. Eine solche Erweiterung
braucht eine getrennte Server-/Client-Prozessmessung.

GPU-Zeitstempel sowie die neue, noch deaktivierte Präfix-Wiederverwendung und
autonome Optimierung/RL bleiben getrennte offene Arbeit. Der Dauerlauf wird
nicht als Optimierungsgewinn oder Lernnachweis umetikettiert.

## Quellen und Reproduktion

- [Vorregistrierter Serverablauf](PROD10_SERVER_SOAK_SPEC.md)
- [Rohbericht](../research/raw/PROD10S_12B_soak_20260907_attempt1.json)
- [Reproduzierbarer QA-Bericht](../research/raw/PROD10S_12B_soak_20260907_attempt1_audit.json)
- [Read-only-Auswertung](../tools/product_soak_audit.py)
- [Vorgeschaltete 1B-/4B-/12B-Matrix](PROD10_RESULTS_2026-09-07.md)

Run-ID: `2c508d859f414ec9892c5a20cf1d0f35`. SHA-256 des Rohdateiinhalts:
`fa0b266f6bdb730fc7a585f76dca32c937c0ede8bfa9d7c31c83fa0eca29787e`.

# PROD12 — HTTP-Server und Modellworker getrennt nativ gemessen

Die getrennte 12B-Serverprüfung ist bestanden: **16 echte HTTP-Anfragen**
stimmen in Text-Hash, Prompt-/Completion-Zahl und Finish-Grund mit der
tatsächlichen Stock-Referenz überein. Server und Modell liefen in verschiedenen
Prozessen und endeten beide mit Exit0. Die Lastgeneratoren und das Journal
lagen außerhalb des Serverprozesses.

## Beobachtete Speicherwerte

| Prozess | Messpunkte | Höchster beobachteter aktueller Footprint | Prozess-Lifetime-Peak | Höchstes beobachtetes RSS |
| --- | ---: | ---: | ---: | ---: |
| HTTP-Server, PID76059 | 79 | 60.441.152 B | 62.489.152 B | 66.600.960 B |
| Modellworker, PID76061 | 73 | 9.106.823.312 B | 9.106.823.312 B | 1.096.335.360 B |

Der Server benötigt in diesem instrumentierten Profil also rund62,5MB
Footprint-Peak, der Modellworker rund9,11GB. RSS allein würde den Modellbedarf
hier deutlich unterschätzen. RSS, Footprint, MLX- und systemweite Swapzähler
werden nicht addiert. Auch die beiden Prozessmaxima liegen nicht zwingend
zum selben Zeitpunkt vor und belegen keinen Gesamt-RAM-Bedarf.

Die Modelltelemetrie wird vom Besitzer des tatsächlichen Workerprozesses
erhoben, als einzelne Metadatenframes übertragen und im Controller gespeichert.
Der Server hält nur Aggregate und kurzfristig eine Messzeile, keine anwachsende
Testhistorie. Die Diagnoseinstrumentierung ist trotzdem Teil des gemessenen
Serverprozesses. Sampling etwa einmal pro Sekunde; kein Last-/Zeit-/RSS-/Swap-
Abbruchgate und keine erneute einstündige Stabilitätsstudie.

## Korrektheit, Abschluss und Reichweite

Die drei Fälle `long_8`, `short_32`, `long_32` laufen je viermal (zuerst Warmup),
anschließend folgen vier gleichzeitig gestartete Clients. Die Modellarbeit
bleibt seriell. Final: `ready=true`, 16 abgeschlossene Anfragen, null Fehler,
null Cancels, null aktive oder wartende Anfragen. Die HTTP-Schnittstelle
exponiert keine Token-IDs; dieser Lauf behauptet keinen direkten Token-ID-
Vergleich sämtlicher HTTP-Antworten.

Alle fünf Vorher-/Nachherbindungen sind unverändert: Runtime-Identität,
installierte Module, Provider, Modellmetadaten und Quellmanifest. Der
installierte Code ist `a3d1b0e28b8c5e22496005f99d79e2befb13f495709ca88333386adbb28803f8`;
der enthaltene Präfixkandidat ist weiter ungenutzt. Nachherbindungen des
Serverkinds wurden nach Workerabschluss und innerhalb der Modelllease erhoben.

Der unabhängige, anschließend verschärfte Read-only-Audit bestätigt170
Journalereignisse:16 Requestsamples,152 Ressourcenbeobachtungen und je einen
Start/Abschluss. Alle Ressourcenzeilen entsprechen dem Rohbericht, alle
Prozesszuordnungen und Maxima wurden erneut geprüft. Der terminale
Berichtsdigest stimmt; keine beobachteten Telemetriefehler. Der Audit prüft
zusätzlich die gebundene Stock-Datei, jede HTTP-Ausgabe gegen diese Referenz,
alle PIDs statt nur des ersten Samples und private Pfad-/Inhaltsfelder.

**Mit diesen Grenzen teilbar:** Belegt ist die getrennte Prozessbeobachtung
dieses 12B-Profils auf dem lokalen M1 Max. Das ist kein Speedup, keine
Leak-Freiheitsgarantie, keine allgemeine Kapazitätszusage und kein Nachweis
für beliebige Langkontexte oder mehrere Macs. GPU-Engpassdiagnose und
autonome Optimierung/RL bleiben offen.

## Quellen

- [Spezifikation](PROD12_SERVER_MEMORY_SPEC.md)
- [Rohbericht](../research/raw/PROD12_12B_server_memory_20260908_attempt1.json)
- [Read-only-Auditor](../tools/product_server_memory_audit.py)
- [Verschärfter Auditexport](../research/raw/PROD12_server_memory_audit_20260908_v2.json)

Run: `507dcf6c589241e5be911b2ffeebe3d1`.
Rohdatei-SHA256: `217bee6a647f40cdd9365bfb8f5dd143db0d016d19b7f875c664d7c7b98d815c`.

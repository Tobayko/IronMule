# Kandidaten für IronMule — 2026-09-05

Status: Primärquellenrecherche und statische Einordnung, keine lokalen
Hardwareergebnisse. Der Nutzer hat neue Ansätze und Live-Tests freigegeben;
fehlender Metal-Zugriff blockiert in dieser Sitzung die Hardwareprüfung.

## 1. Interaktionsmodell und konservatives RL

Mechanismus: den gemeinsamen Einfluss von Workload, Modell, Cachezustand und
Konfiguration modellieren; eine konservative Policy wählt mehrstufige Versuche
einschließlich langsameren Zwischenkonfigurationen. Die vorhandene FQI ist ein
Ausgangspunkt, aber keine bereits qualifizierte Produktpolicy.

[Conservative Q-Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
behandelt die Überschätzung außerhalb der beobachteten Aktionsverteilung. Das
rechtfertigt Vorsicht bei unbekannten Konfigurationen, nicht die Behauptung,
352 lokale Wiederholungen reichten schon für allgemeines Hardwarelernen.

**Erster sinnvoller Vergleich:** nach neuer realer Kontext-/Kombinationsabdeckung
feste Aktionsmittelwerte, additive und interaktionsfähige Kostenmodelle sowie
FQI bei gleichem Messbudget und getrennten Bestätigungsdaten vergleichen.
**Kill:** kein unabhängiger Nettovorteil, fehlende Unterstützung oder instabile
Unsicherheit: deterministische Auswahl behalten. Aus R2 werden keine erfundenen
Mehrschrittübergänge oder Paarinteraktionen erzeugt.

## 2. Scheduling und Cache gemeinsam behandeln

[ORCA, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)
motiviert Scheduling auf Iterationsebene und selektive Bündelung. Seine
GPT-3-/NVIDIA-Ergebnisse sind keine Geschwindigkeitsvorhersage für einen M1 Max.

Mechanismus für IronMule: Queue-Wartezeit, Kapazitätswechsel, Prefix-Misses und
Compile-Cache gemeinsam reduzieren. Das unterscheidet sich von B25: dort wurde
bereits ausgeschlossen, dass der feste KV-Cache pro Token grundsätzlich wächst.

**Erster Versuch:** derselbe deklarierte Ausführungsplan mit kurzen/längeren
Anfragen und wechselnden Cachekapazitäten; Baseline und Kandidat alternieren.
StrictOneShot gegen ReusableSession ist kein Exact-Optimierungsvergleich, weil
der Planwechsel selbst die Ausgabe verändern kann.
**Kill:** Token-/Zustandsabweichung innerhalb des gleichen Plans, mehr Swap,
unbegrenzter Speicher oder kein Latenz-/Speichervorteil nach Eigenaufwand.

## 3. Begrenzte evolutionäre Kernelsuche

[Ansor, OSDI 2020](https://arxiv.org/abs/2006.06762) kombiniert einen
hierarchischen Suchraum, evolutionäre Suche und ein gelerntes Kostenmodell.
Die Publikation untersucht Intel-/ARM-CPUs und NVIDIA-GPUs; sie belegt keinen
Gemma-/Metal-Gewinn. Übernommen wird nur der Mechanismus einer strukturierten,
messgestützten Suche, nicht ein neuer Compiler oder eine fremde Leistungszahl.

**Erster Versuch:** erst einen konkreten Engpass nachweisen, dann wenige
versionierte Metal-Templates für genau eine reale Gemma-Tensoroperation und
feste Shapes vergleichen; den Sieger erst anschließend end-to-end bewerten.
**Kill:** keine Amortisierung der Compile-/Suchkosten, veränderte Ergebnisse
oder kein stabiler End-to-End-Vorteil. Bereits abgelehnte Residual+RMSNorm- und
Projektionsfusion-Versuche werden nicht unverändert wiederholt.

## 4. Gelernte Entwurfs-Heads statt Prompt-Lookup

[Medusa](https://openreview.net/pdf?id=PEpbUobfJv) und
[EAGLE, ICML 2024](https://icml.cc/virtual/2024/poster/35153)
untersuchen gelernte Mehrtokenentwürfe. Das ist ein anderer Entwurfsmechanismus
als die erfolglose n-Gramm-Suche dieses Projekts. Ein verwendbarer trainierter
Head für die lokal vorhandenen Gemma-Snapshots ist bisher nicht belegt.

**Voraussetzung:** passender Head samt Lizenz, Modell-/Quantisierungsbindung und
Speicherbudget. Erst danach Annahmerate, Kosten des Entwurfs/Verifizierens und
Ausgabevertrag messen. Die Break-even-Grenze folgt aus diesen Kosten, nicht aus
einer unverändert übernommenen Annahmequote eines anderen Modells.
**Kill:** fehlender kompatibler Head, nicht amortisierte Zusatzkosten oder
Verletzung des gewählten Vertrags. Breitere bf16-Forwards sind nicht automatisch
tokenidentisch; ohne Erhaltungsnachweis gehört der Pfad in Efficiency mit
separatem Nichtunterlegenheitsgate.

## Priorisierung

Zuerst korrekte Evidenz und portable Produktdiagnostik, dann Scheduling/Cache,
echte Kontext- und Kombinationsdaten und daraus Kostenmodell/RL. Kernelsuche ist
an einen belegten Engpass gebunden; Entwurfs-Heads bleiben eine getrennte
Voraussetzungsprüfung. Kein Ansatz erhält aus dieser Recherche eine Aktivierung.

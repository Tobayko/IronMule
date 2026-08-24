# Freigabepflichtige Kandidaten

Stand: 24. August 2026. Nur tatsächlich blockierte Punkte. Jeder wurde übersprungen,
der Loop lief weiter.

## 1. vLLM mit Metal-Backend

**Status:** nicht installiert (`import vllm` schlägt fehl).

**Nutzen:** Paged-KV und automatischer Prefix-Cache lösen genau das Problem, an dem
der MLX-Pfad in Zyklus 1 scheiterte — Wiederverwendung eines Präfixes ohne Neurechnung.
Ein Vergleich von TTFT, TPOT und Speicher gegen den bestehenden MLX-Pfad wäre die
direkteste Einordnung.

**Risiko:** großer Abhängigkeitsbaum; Gemma-3-Unterstützung auf Metal ist ungeprüft;
Tokenidentität gegenüber MLX ist **nicht** zu erwarten, da anderes Backend und andere
Kernel. Ein Vergleich wäre Engineering-Evidenz, kein Beweis semantischer Gleichheit.

**Benötigte Aktion:** ausdrückliche Installationsfreigabe.

## 2. llama.cpp

**Status:** nicht installiert, nicht im `PATH` (`llama-cli`, `llama-server` fehlen).

**Nutzen:** zweite unabhängige Referenz für Prefill- und Decode-Durchsatz auf
demselben Gerät.

**Risiko:** benötigt GGUF-Gewichte, die lokal **nicht** vorliegen. Ein Vergleich liefe
über eine andere Quantisierung und wäre damit doppelt inkommensurabel.

**Benötigte Aktion:** Installations- **und** Modelldownload-Freigabe.

## 3. Energiemessung

**Status:** `/usr/bin/powermetrics` vorhanden, verlangt aber ein Passwort
(`sudo -n true` schlägt fehl).

**Nutzen:** Energie je Token ist für Akkubetrieb die entscheidende Größe und bisher
vollständig ungemessen.

**Risiko:** keines für das System — reines Lesen von Zählern. Das Passwort gehört
jedoch nicht in einen Agentenlauf.

**Benötigte Aktion:** Der Nutzer führt selbst aus und übergibt die Ausgabe:

```
! sudo powermetrics --samplers gpu_power -i 500 -n 6
```

## 4. Dichtes 7–9B-Modell ohne Sliding Window

**Status:** lokal nicht vorhanden.

**Nutzen:** Gemma 3 begrenzt 29 von 34 Layern auf ein Fenster von 1024 Token, weshalb
`f_attention` bei 8–16K Kontext nur 12–17 % beträgt und jede Attention-Optimierung bei
`1,20x` deckelt. Bei voller Attention über 32K läge der Anteil plausibel bei 40–60 %.

**Risiko:** mehrere GB Download.

**Benötigte Aktion:** ausdrückliche Download-Freigabe.

## 5. Kausaler A/B-Test einer KV-Cache-Vorallokation

**Status:** Zyklus 11 lokalisierte Reallokationen an Decodeschritt `1` und `4` und
maß `4,4263 %` korrelierten Decodeanteil. Der große Ausschlag fällt zugleich auf den
ersten Decodeschritt; die Beobachtungsstudie kann ihn deshalb nicht kausal der Kopie
allein zuschreiben.

**Nutzen:** Ein A/B-Pfad mit vorallokiertem Cache könnte prüfen, welcher Anteil der
`31,8821` ms tatsächlich entfernbar ist. Erst damit ließe sich entscheiden, ob ein
Cache-Umbau lohnt.

**Risiko:** Der Kandidat greift in Cacheform, Allokationszeitpunkt und Speicherbedarf
von `mlx_lm` ein. Er kann Tokenidentität, Peak Unified Memory und Fallbackverhalten
ändern; die beobachteten `4,4263 %` dürfen nicht als erwarteter Gewinn übernommen
werden.

**Benötigte Aktion:** ausdrückliche Architekturfreigabe für einen isolierten,
rückrollbaren A/B-Messpfad mit eigenem Cachetyp, Correctness-Gate, Speichergrenzen
und `BudgetGuard`. Bis dahin wird der Kandidat übersprungen.

## 6. Begrenzte Integration des formal bestätigten Prefill-Head-Skips

**Status:** Der Nutzer hat die begrenzte Architektur am 24.08.2026 ausdrücklich
freigegeben. Zyklus 12 bestätigte im versiegelten Scope einen Prefill-Zeitquotienten
von `0,846385` beziehungsweise `−15,3615 %`; alle C-/V-/Gesamt-Gates und `12/12`
Greedy-Tokenidentitätsgates bestanden. Der getrennte Prototyp bestand anschließend
das vorregistrierte CPU-Gate und genau einen GPU-Lauf: `R=0,845836`, Effekt
`−15,4164 %`, identische Token, kein Swap-Wachstum und alle Engineering-Gates grün.
Damit ist nur der exakt registrierte Repository-Aufrufpunkt freigegeben.

**Nutzen:** Der Kandidat trifft den gemessenen Hauptengpass direkt und vermeidet beim
greedy Prefill die Projektion aller Promptpositionen, obwohl nur die letzte Position
für den ersten Ausgabetoken gelesen wird.

**Risiko:** Der Pfad ist unzulässig, sobald Prompt-Logprobs, Perplexität oder andere
Ausgaben pro Promptposition verlangt werden. Eine Integration in `mlx_lm` oder eine
lokale Runtime-Verzweigung verändert Architektur, API-Grenzen, Fehlerbehandlung und
Fallback. Der formale Einzelprompt-Claim belegt weder allgemeine TTFT-Wirkung noch
andere Promptlängen, Modelle, Quantisierungen oder Geräte.

**Erteilte Aktion:** begrenzter, rückrollbarer Runtime-Prototyp mit enger
Scope-Prüfung (`greedy`, keine Prompt-Logprobs), unverändertem Referenzpfad,
fail-closed Fallback, Tokenidentitäts-Regressionsgate, Speichergrenze und eigener
Historie. Die Freigabe umfasst keine breitere Aktivierung, Installation, Downloads
oder Erweiterung des formalen Claims.

## 7. Begrenzter persistenter Modell-Dienst

**Status:** Zyklus 13 bestätigte prospektiv im exakten Scope ein medianes
`warm/kalt`-Verhältnis von `0,346968`, entsprechend gerechnet `−65,3032 %` TTFT.
Alle sechs greedy Tokenpaare waren exakt identisch, Peak-RSS blieb unter `5 GiB`
und Swap wuchs nicht. Der Messworker ist noch kein normaler Dienst.

**Nutzen:** Wiederholte lokale Anfragen vermeiden Prozessstart und erneutes
Modellladen. Im gemessenen `897`-Token-Scope sank der Median der TTFT-Werte von
`5148,7741` auf `1785,1103` ms.

**Risiko:** Ein langlebiger Prozess verändert Lebensdauer, Fehlerbehandlung,
Speicherdruck, gleichzeitige Anfragen und Abschaltverhalten. Der Einzelrequest-
Versuch belegt weder Multi-Turn-Verhalten noch Parallelbetrieb. Ein hängender oder
fehlerhafter Dienst muss sichtbar auf den bisherigen Start-pro-Anfrage-Pfad
zurückfallen und darf die einmalige Studienmarke nicht als Betriebsevidenz deuten.

**Benötigte Aktion:** ausdrückliche Architekturfreigabe für einen getrennten,
rückrollbaren lokalen Dienst mit unverändertem Referenzpfad, festem lokalen
Modell-Snapshot, frischem Anfrage-Cache, Speichergrenze, Fehlerzähler, sichtbarem
Fallback und eigener Qualifikation für Multi-Turn sowie parallele Anfragen. Keine
allgemeine Aktivierung aus dem Zyklus-13-Ergebnis allein.

## 8. Selbstlernendes Optimization Memory mit kleinem lokalem Planner

**Status:** Das Nutzerziel „eigenständig optimieren“ ist erteilt; die genaue
Architekturfreigabe steht noch aus. Ein vollständiger lokaler
`mlx-community/gemma-3-1b-it-4bit`-Snapshot auf Revision
`2d44e83dc9e80843d22fb941d3d699a0b1351aa6` ist bereits vorhanden. Es ist kein
Download und keine Installation nötig. Der vorhandene `tools/model_loop.py` kann
Messungen innerhalb eines Laufs zurückfüttern, speichert dieses Lernen aber nicht
dauerhaft und verwendet derzeit das größere 4B-Modell.

**Nutzen:** Das System merkt sich Gewinne, Nullergebnisse und Fehler dauerhaft,
schlägt daraus genau den nächsten sinnvollen Versuch vor und wiederholt bekannte
Sackgassen nicht. Der 1B-Planer schont die 32-GB-Hardware und läuft getrennt vom
4B-Messworker, sodass beide Modelle nicht gleichzeitig im Speicher bleiben müssen.

**Risiko:** Ein Sprachmodell kann überzeugend klingende, aber falsche Vorschläge
machen und sich aus wenigen Messungen selbst bestätigen. Das Projekt besitzt noch
nicht die mehreren hundert sauberen Messungen, die laut technischem Konzept für
ein wirklich gelerntes Rangiermodell nötig sind. Modellgenerierter Code,
nachträglich geänderte Schwellen oder eine Aktivierung durch das Modell wären
unsicher und wissenschaftlich wertlos.

**Benötigte Aktion:** ausdrückliche Freigabe für diese enge Architektur:

1. lokale, private Historie mit **allen** Ergebnissen einschließlich Fehlern und
   Nullresultaten, jeweils an Hardware-, Modell-, Software- und Workload-Fingerprints
   gebunden;
2. bis zu mehreren hundert sauberen Messungen nur feste Regeln; der lokale 1B-Planer
   darf genau **einen** noch ungemessenen Kandidaten aus einer festen Liste als JSON
   vorschlagen, niemals Code oder Befehle;
3. weiterhin genau ein Kandidat je Zyklus, Vorregistrierung vor Hardware,
   `BudgetGuard`, Netzbetrieb, keine Wiederholung und zwingende Tokenidentität;
4. Messprogramm und feste Entscheidungstabelle urteilen. Der Planer darf keine
   Schwelle, kein Ergebnis, keine Korrektheit und keine Aktivierung bestimmen;
5. zunächst keine automatische Produktaktivierung. Ein bestätigter Kandidat erhält
   nur einen getrennten, rückrollbaren Integrationsvorschlag.

Nach dieser Freigabe wird zuerst nur die Historie plus 1B-Vorschlagsweg offline
gebaut und geprüft. Eine eigene Hardwarestudie folgt erst mit neuer
Vorregistrierung; beide Modelle laufen dabei nacheinander, nicht gleichzeitig.

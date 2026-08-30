# L1 Gemma Optimizer — Architekturvorschlag

**Status: OFFLINE-IMPLEMENTIERUNG FREIGEGEBEN; HARDWARE/PROMOTION GATE-BASIERT**
**Datum: 2026-08-30**
**Geltung: Offline-Implementierung freigegeben; kein realer Hardware-/Modelllauf, Download oder Installation ohne separates Gate**

## 1. Ziel und klare Grenzen

Der Nutzer möchte eine lokale Optimierungssteuerung für Apple-Silicon-Macs, zunächst
für Gemma 1B, 4B, 12B und 27B. Sie soll nur nach einem ausdrücklichen Start durch
den Nutzer eine insgesamt 5 bis 30 Minuten lange Sitzung ausführen. Die gewählte
Dauer ist eine harte Gesamtfrist; auch Warten auf Strom oder freie Hardware zählt
hinein.

Die Sitzung darf nur am Netzteil und nur bei sicher freier Maschine starten. Wenn
Claude, MLX, ein anderer Modellprozess oder eine andere schwere Fremdlast erkannt
wird, wartet sie. Fremdlast, die während der Sitzung erscheint, beendet die Sitzung
sicher und lässt die Baseline aktiv. Wenn die Maschine frei ist, darf die Sitzung
CPU, GPU und Unified Memory vollständig verwenden.

Das anfängliche Verhalten bleibt deterministisch und greedy. Ein Kandidat muss
dieselbe Ausgabe wie die unveränderte Baseline liefern. Gemessen werden mindestens
TTFT und Decode-only-Tokens/s. Ein bestätigtes Profil darf automatisch aktiviert
werden, aber der Nutzer kann zwischen automatischer Auswahl, Baseline und einem
festgehaltenen Profil wählen. Jede Verschlechterung oder Unsicherheit führt sofort
zur Baseline.

Nicht umfasst sind freie Code- oder Kernelgenerierung, ein Lernmodell als
Promotionsrichter, eine automatische Änderung von Schwellwerten, ein globales
Optimum, ein stiller Download, eine Installation oder eine Behauptung über Macs,
Modelle und Versionen, die nicht separat gemessen wurden.

## 2. Ehrliche Bedeutung von „Optimum“

Ein mathematisch garantiertes oder global für alle Macs gültiges Optimum ist nicht
erreichbar. Gemeint ist daher das **bestverifizierte lokale Profil** innerhalb einer
festen, geprüften Allowlist für genau einen Fingerprint aus:

```text
Hardware + macOS + MLX/mlx-lm + Runtime-Code + Modellrevision
+ Architektur + Quantisierung + Tokenizer + Workload + Strom-/Power-Modus
```

Ein Profil gilt nur für diese Identität und die festgelegte Workload-Bandbreite.
Historische Ergebnisse dürfen die Reihenfolge der Versuche verbessern, aber keine
unbelegte Übertragung ersetzen. Fehlt eine Identität, ein Messwert, eine freie
Maschine oder ein klares Konfidenzintervall, lautet die Entscheidung Baseline.

## 3. Ist-Evidenz und Gültigkeitsbereich

Der Audit ergibt folgende belastbare Ausgangslage:

| Bereich | Vorhandene Evidenz | Konsequenz |
| --- | --- | --- |
| Lokale Modelle | lokal auflösbare Snapshots nur Gemma 1B und 4B | 1B/4B können zuerst geprüft werden |
| 1B/4B/12B | gute Rohdaten aus getrennten Studien beziehungsweise Integrationsläufen; Q2 ist ein realer einmaliger 4B-Engineering-Smoke | nicht vermischen; jede Identität bleibt eigene Zelle |
| Q2 / IronMule `tune` | Allowlist-Screening, Paired Confirmation und Profilwrite liefen auf einem M1 Max mit Gemma 4B; nicht formal versiegelt, nicht produktiv und nicht global | nur als Verfahrensbeleg und Adapter-Unterbau verwenden; Friday-Gates bleiben strenger |
| 27B | keine Gemma-27B-Evidence und kein lokaler Gemma-27B-Snapshot; Qwen-27B darf nicht vermischt werden | kein 27B-Live-Test und kein 27B-Profil ohne neue Freigabe, Gemma-Snapshot und Messvertrag |
| Geräte | Evidenz überwiegend Apple M1 Max | keine Cross-Device-Aussage; jeder andere Mac beginnt bei A/A |
| Strom | AC-Messungen vorhanden; Akku und Low-Power verändern das Verhalten | AC-only, unbekannt bedeutet warten/abbrechen |
| Korrektheit | mehrere token- und byte-identische Studien, aber jeweils enger Scope | Baseline-/Kandidatengate pro Modell und Workload neu ausführen |
| Runtime | Executor, Fallbacks, Hash-History und lokale UIs vorhanden | nur als geprüfte Adapter wiederverwenden |

Die Root-Dokumente `PERFORMANCE_BASELINE.md`, `PROJECT_STATUS.md`,
`IMPLEMENTIERUNGSPLAN.md` und die vorhandenen Studien bleiben Evidenzquellen mit
ihren jeweiligen Qualitätsklassen. Eine Summary ohne Rohsamples wird niemals als
gleichwertiges Trainings- oder Promotionslabel behandelt.

Der Q2-Lauf in IronMule ist ein Engineering-Smoke: ein einmaliger M1-Max-/Gemma-4B-
Lauf mit Allowlist-Screening, gepaarter Bestätigung und Profilwrite. Die Übergabe
enthält dafür nicht die vollständigen Roh-PIDs, Exitcodes, Kandidaten-Outputs oder
das Zweitstart-Log. Das gespeicherte Profil trägt den Screening-Wert statt der
bestätigten Messung; der Fix `0de69b6` korrigiert dieses Verhalten erst für künftige
Läufe. Q2 beweist damit, dass der Ablauf einmal schließt, aber keinen formalen,
globalen oder produktiven Optimierungsclaim.

## 4. Platzierung und Besitz der Architektur

Die erste Implementierung soll als unabhängige Control Plane im Root entstehen. Sie
implementiert keinen zweiten Gemma-Tuner und keinen zweiten Modell-Worker, sondern
bindet den vorhandenen IronMule-Ausführungspfad über einen strikt commit- und
fingerprintgebundenen Adapter. Der aktuell von Claude verwendete Worktree darf
nicht als dauerhafter Adapter-Checkout verwendet werden; dafür ist später ein
separater sauberer Checkout mit festem Commit erforderlich.

```text
Project_Friday/
└── friday_optimizer/
    ├── corpus.py
    ├── memory.py
    ├── fingerprint.py
    ├── readiness.py
    ├── session.py
    ├── candidates.py
    ├── ironmule_adapter.py
    ├── evaluator.py
    ├── profiles.py
    ├── canary.py
    ├── history.py
    ├── dashboard.py
    └── cli.py
```

`friday_optimizer/` besitzt Sitzungsfreigabe, Ressourcenprüfung, Profilstatus,
kanonische History, Rollback und die Friday-Promotionentscheidung. `IronMuleTuneAdapter`
ruft `tune`/`revalidate` nur in einem isolierten, sauber gebundenen Checkout auf,
übernimmt dessen Allowlist und Correctness-Prüfungen und importiert Ergebnisse erst
nach eigener Prüfung. Die vorhandenen `friday_*`-Studien bleiben unverändert. Der
aktuell von Claude benutzte `.worktrees/ironmule-b7` bleibt vollständig unangetastet.

## 5. Komponenten und Verantwortungsgrenzen

| Komponente | Aufgabe | Darf nicht tun |
| --- | --- | --- |
| `CorpusAuditor` | SQLite-/JSON-/Artefaktquellen read-only inventarisieren, Qualitätsklasse, Missingness, Ketten und Dubletten melden | historische Evidenz ändern |
| `OptimizationMemoryV2` | unveränderliche, kanonische, versionierte Records für Environment, Workload, Candidate, Correctness, Benchmark, Profile und Promotion speichern | alte Records überschreiben; Modelloutput vertrauen |
| `HardwareFingerprint` | Chip, Kerne, RAM, GPU, macOS, MLX, Power- und Lastzustand binden | fremde Gerätewerte schätzen |
| `ModelFingerprint` | Revision, Manifest, Architektur, Quantisierung, Tokenizer und Artefakthash binden | Hub-ID ohne lokalen Snapshot ausführen |
| `WorkloadFingerprint` | Prompt-/Tokenizer-/Generatorversion, Shapes, Kontext, maximale Token und Modus binden | fremde Workloads stillschweigend übernehmen |
| `ReadinessLease` | AC, freie Maschine, Swap-/Speicherzustand und atomare eigene Besitzmarke verwalten | bei unklarer Lage freigeben |
| `SessionController` | manuelle Startfreigabe, 5–30-Minuten-Hard-Deadline und Zustandsmaschine steuern | Sitzungen selbständig oder auf Akku starten |
| `CandidateRegistry` | Friday-Allowlist und Grenzen führen und auf zulässige IronMule-Knobs abbilden | freien Python-, Metal- oder Kernelcode akzeptieren; IronMule-Suchraum unkontrolliert erweitern |
| `IronMuleTuneAdapter` | `tune`/`revalidate` in einem späteren sauberen Checkout mit Commit-, Modell- und Hardware-Fingerprint aufrufen | Claude-Worktree verwenden, Profil ungeprüft aktivieren oder eigene Tunerlogik duplizieren |
| `ProcessIsolationGate` | frischen Prozess pro Messblock, eindeutige PID, per-process Peak und Swap vor/nachher erzwingen | kumulative Interpreter-Peaks als unabhängige Messungen ausgeben |
| `CorrectnessEvaluator` | Token, Text, Stop-Grund, Anzahl und Hash vergleichen | Performance über Korrektheit stellen |
| `PerformanceEvaluator` | Warmup, A/A, AB/BA, Rohwerte, TTFT und Decode-only-Tokens/s auswerten | aus einem Einzelwert entscheiden |
| `AtomicProfileStore` | Baseline, Kandidat, aktiven Zeiger und vorherige Version atomar verwalten | halbgeschriebene Profile aktivieren |
| `RuntimeCanary/Rollback` | Revalidierung, Hysterese, Circuit Breaker und sofortige Baseline-Rückkehr | einen fehlerhaften Kandidaten weiterlaufen lassen |
| `History/UI` | append-only History und read-only Loopback-Anzeige bereitstellen | Netzwerkzugriff oder Schreib-API anbieten |

## 6. Geschlossene Kandidaten-Allowlist

Der erste Registry-Stand enthält nur überprüfbare, feste Kandidaten:

1. `baseline`: unveränderter Referenzpfad;
2. `persistent_process`: Modell einmal laden und wiederverwenden;
3. `fixed_compiled_cache`: feste Cacheform mit vorher gebundenem Compilepfad;
4. `head_skip_prefill`: nur wenn der exakt gebundene greedy-Prefillfall passt;
5. `readback_every_2`: nur im exakt gebundenen Q2-4B-/M1-Max-/MLX-Scope, nicht als
   allgemeine Readback-Empfehlung;
6. `combined_core_profile`: eine vorher einzeln qualifizierte Kombination, nicht
   freie Kombination beliebiger Schalter;
7. `throughput_width_2`, `throughput_width_3`, `throughput_width_4`: nur in einem
   getrennten Throughput-Scope, mit eigener Latenz- und Korrektheitsprüfung.

Kandidaten dürfen keine Gewichte, Modellarchitektur, Quantisierung oder Antwortlogik
ändern. Width 2/3/4 ist keine allgemeine Empfehlung für interaktive Einzelanfragen.
Width 8 oder höher und freie True-Batching-Experimente sind zunächst außerhalb des
Scopes. Ein LLM darf höchstens eine erlaubte Candidate-ID vorschlagen; Registry,
Worker und Evaluator bleiben die Ausführungsautorität.

## 7. Zustandsmaschine und Zeitgrenze

```text
requested
  -> waiting_for_ac_or_idle
  -> calibrating_aa
  -> testing_balanced_ab_ba_fresh_processes
  -> qualified | rejected | inconclusive
  -> atomic_activation
  -> canary
       -> active
       -> rollback_latched -> baseline
```

`waiting_for_ac_or_idle` prüft in kurzen, protokollierten Abständen erneut. Alle
Zustände teilen dieselbe vom Nutzer gewählte Gesamtdeadline von 5 bis 30 Minuten.
Wird die Deadline erreicht, ein Wert bleibt unklar oder eine Prüfung fällt aus,
endet die Sitzung sicher mit Baseline beziehungsweise `no_recommendation`.

Eine reale Hardware- oder Modellmessung beginnt erst nach einer separaten
ausdrücklichen Freigabe. Die Architekturfreigabe dieses Dokuments ist dafür nicht
ausreichend.

## 8. Readiness, Fremdlast und Lease

Die Freigabe muss gleichzeitig alle folgenden Bedingungen erfüllen:

- Stromquelle eindeutig `AC`; Akku und unbekannter Zustand blockieren.
- Low-Power-Modus aus oder eindeutig bekannt und zulässig.
- mehrere zeitlich getrennte Samples für CPU-/Systemlast, Speicher, Swap und
  Prozessbaum zeigen Ruhe; ein einzelner Momentwert genügt nicht.
- Prozessbaum und relevante Ressourcen werden für Claude, MLX, Python, Node und
  andere schwere Modellprozesse geprüft; nur ein Prozessname ist kein Beweis.
- Swap-Ausgangswert und Speicherwerte sind lesbar; fehlende Messbarkeit ist kein
  Freibrief.
- eine eigene atomare Lease wird erstellt und nach jedem Kontrollpunkt validiert.
- Bei großen oder service-/grouped Workloads läuft jeder Messblock in einem frischen
  Prozess; PID-Eindeutigkeit, per-process Peak sowie Swap vor und nach dem Block
  werden separat gebunden.

Es gibt keinen `--force`-Pfad in der autonomen Sitzung. Wird nach Lease-Erwerb
Fremdlast sichtbar, verliert die Lease ihre Gültigkeit, der Worker wird kontrolliert
beendet und die Sitzung geht zu Baseline. Die Lease muss vor Hardwareprobe,
Modellload, jedem Kandidatenblock und jeder Aktivierung gehalten werden.

## 9. Korrektheit und Messung

Zunächst ist ausschließlich deterministisches greedy/current behavior zulässig.
Jeder Kandidat muss gegenüber derselben Baseline alle folgenden Prüfungen bestehen:

- identische Token-IDs in derselben Reihenfolge;
- identischer dekodierter Text und identischer Antwort-Hash;
- identischer Stop-Grund;
- identische physische und sichtbare Tokenanzahl;
- gleicher Prompt-/Tokenizer-/Generatorvertrag;
- keine NaN-/Inf-, Cache-, Shape- oder Ressourcenverletzung.

Gemessen wird mit symmetrischem Warmup, mehreren Wiederholungen, randomisierter
AB/BA-Reihenfolge, frischen Prozessen pro Arm und gespeicherten Rohsamples. TTFT
wird vom vereinbarten Request-/Engine-Punkt getrennt ausgewiesen. Decode-only
Tokens/s enthält nur erzeugte Decode-Schritte; Prefill-Token, EOS und äußere
Prozessstartzeit werden nicht stillschweigend in diese Kennzahl gemischt.

Der Q2-Smoke verwendete eine End-to-End-Gesamtratio. Das ersetzt nicht das strengere
Friday-Gate: TTFT und Decode-only-Tokens/s werden getrennt bewertet; keine Größe darf
statistisch regressieren, und mindestens eine Größe muss die vorab bestimmte A/A-
Rauschgrenze beziehungsweise das MDE-Gate überschreiten.

## 10. Entscheidungsregel

Ein Kandidat darf nur aktiviert werden, wenn gleichzeitig gilt:

1. weder TTFT noch Decode-only-Tokens/s ist gegenüber der Baseline statistisch
   schlechter;
2. mindestens eine der beiden Zielgrößen verbessert sich über die aus A/A
   bestimmte Rauschgrenze beziehungsweise das vorab festgelegte MDE-Gate;
3. Correctness-, Speicher-, RSS-, Swap-, Timeout-, Temperatur- und Fremdlast-Gates
   bestehen;
4. das Konfidenzintervall ist ausreichend eng und die Messung reproduzierbar.

Bei widersprüchlichen Metriken, fehlendem A/A, hoher Unsicherheit oder unbekannter
Verteilung bleibt die Baseline. Kandidaten werden nach der konservativen
schlechtesten Konfidenzgrenze gerankt, nicht nach dem günstigsten Einzelwert.
Ein Rankingmodell darf nur zulässige Kandidaten sortieren; es darf keine Gate- oder
Schwellwertentscheidung überschreiben.

Forking ist dabei ein Memory- und Messintegritätsmechanismus, kein automatischer
Speed-Claim: Der korrigierte E15-Lauf hatte vier eindeutige Prozesse und flache
Peaks um 7,07 GB, war im Wall-Vergleich aber langsamer als der Vorher-Lauf.

## 11. Profile, Portabilität und Invalidierung

Profile werden niemals automatisch auf einen anderen Mac, ein anderes Gemma-Modell,
eine andere Revision, eine andere MLX-/mlx-lm-Version oder eine andere Quantisierung
übertragen. Ein neuer Fingerprint startet mindestens mit A/A und einer Baseline.

Das Profil wird ungültig bei Änderung von:

```text
Chip/GPU/RAM/Kernlayout, macOS, MLX, mlx-lm, Runtime-Code,
Modellrevision/Manifest, Architektur, Bits/Group-Size, Tokenizer,
Prompt-/Workloadvertrag, Power-Modus oder relevanter Cache-/Compilerkonfiguration
```

Die Hardware kann aus historischen Ergebnissen eine Versuchssortierung ableiten,
aber keine nicht gemessene Zahl interpolieren. Der lokale Gemma-27B-Fall bleibt bis
zu einem eigenen Snapshot und Messvertrag blockiert.

## 12. Automatische Aktivierung und Nutzer-Override

Es gibt drei sichtbare Betriebsarten:

- **Auto:** letztes qualifiziertes, kompatibles Profil nach erfolgreicher Canary
  verwenden;
- **Baseline:** unveränderten Pfad erzwingen;
- **Pinned:** ein vom Nutzer ausgewähltes, kompatibles Profil verwenden.

`Pinned` ist kein Freibrief: Correctness-, Readiness- und Ressourcen-Gates gelten
weiter. Ein Nutzer-Override verändert nicht die historische Evidenz. Aktivierung
erfolgt nur über einen atomaren aktiven Zeiger. Vorheriger Profilstand und Baseline
bleiben erhalten. Bei jeder Canary-Abweichung wird der Zeiger sofort auf Baseline
gedreht und ein Rollback-Latch gesetzt. Der Latch bleibt bis zu einer neuen
ausdrücklich gestarteten und bestandenen Sitzung aktiv.

## 13. Lokale UI

Die erste UI ist loopback-only und read-only. Sichtbar sind ausschließlich:

- aktueller Zustand und Wartegrund;
- Modell-/Hardware-Kurzbezeichnung und aktives Profil;
- TTFT und Decode-only-Tokens/s;
- Correctness-Ergebnis und finale Entscheidung;
- Screening- gegenüber Bestätigungswert, Konfidenzintervall, Tuner-Commit und
  Profilhash;
- PID-Eindeutigkeit, per-process Peak, Fork-/Prozessmodus und Swap-vor/nachher;
- chronologische History der Sitzungen.

Interne Sicherheitsdaten wie Lease-Proben, Prozessbaum-Snapshots, vollständige
Hashes, Fehlerklassifikation und Rollbackursache bleiben gespeichert, werden aber
nicht als unübersichtliche Rohdaten im Hauptfenster dargestellt. Keine POST-/PUT-/
DELETE-Route, kein externer Host und kein unkontrollierter Prompt- oder Modelltext
wird ausgeliefert.

## 14. Sicherheitsbedeutung in einfacher Sprache

Das System darf nur Dinge testen, die vorher erlaubt wurden. Es prüft zuerst, ob der
Mac am Strom hängt und niemand anderes die Maschine für KI benutzt. Wenn es das nicht
sicher weiß, wartet es. Jeder Test läuft in einer eigenen begrenzten Umgebung. Eine
falsche Antwort, ein Fehler oder eine auffällige Speicher-/Swap-Situation zählt als
Fehler, auch wenn der Kandidat schneller war. Dann wird sofort die bekannte Baseline
verwendet. Alte Ergebnisse werden nicht überschrieben, und kein Modell darf sich
selbst Regeln, Grenzen oder Freigaben geben.

## 15. Phasen und notwendige Freigaben

| Phase | Inhalt | Freigabe/Gate |
| --- | --- | --- |
| L1.0 | Read-only Inventar, Architektur und Daten-/Portabilitätsaudit | dieser Vorschlag; aktuell nicht freigegeben |
| L1.1 | `OptimizationMemoryV2`, CorpusAuditor, Dataset Card, idempotenter Import | Architekturfreigabe; keine Hardware nötig |
| L1.2 | Offline-Replay, A/A-Rauschmodell, deterministische CandidateRegistry | Daten-/Split-/Leakage-Gate |
| L1.3 | ReadinessLease, SessionController, `IronMuleTuneAdapter`, ProcessIsolationGate, Evaluator, UI | Sicherheitsreview und separate Freigabe vor erstem realen Hardwarelauf |
| L1.4 | erster manueller 5–30-Minuten-Lauf mit einem bereits lokalen 1B/4B-Snapshot | ausdrückliche Nutzerfreigabe für Hardwarelauf |
| L1.5 | Canary, atomare Aktivierung, Rollback-Latch und erneute Messung | eigene Promotionsfreigabe; zunächst kein autonomes Lernen in Nutzeranfragen |
| L1.6 | weitere Modelle/27B/andere Macs | eigener Snapshot, Vertrag, A/A/A-B, Ressourcenfreigabe und Evidenz; keine Übertragung |

Vor jedem Download oder jeder Installation eines Modells, einer Bibliothek oder
Software ist eine weitere ausdrückliche Freigabe nötig. Ohne diese Freigabe bleibt
der betreffende Kandidat blockiert.

## 16. Gates und Kill-Kriterien

Die Arbeit stoppt oder fällt auf Baseline zurück, wenn eines dieser Kriterien eintritt:

- AC, Low-Power- oder Fremdlaststatus ist unbekannt oder widersprüchlich;
- Lease fehlt, abläuft oder von einem zweiten Besitzer verletzt wird;
- die 5–30-Minuten-Gesamtdeadline wird erreicht;
- ein Kandidat verändert Token, Text, Stop-Grund, Anzahl oder Antwort-Hash;
- TTFT oder Decode-only-Tokens/s ist statistisch schlechter oder die MDE wird nicht
  erreicht;
- Swap wächst, Speicher-/RSS-/Temperatur-/Timeout-Grenzen werden überschritten;
- PID-Eindeutigkeit, per-process Peaks oder Swap-vor/nachher fehlen; ein Fork darf
  nicht als Geschwindigkeitsgewinn gewertet werden;
- Worker, Compiler, Modellidentität, Artefakthash oder Prozessbaum sind nicht
  reproduzierbar;
- ein Profil passt nicht exakt zum aktuellen Fingerprint;
- ein Lern-/Rankingmodell zeigt OOD, hohe Unsicherheit oder schlechtere
  Invalid-Suggestion-Rate als Random/Grid;
- ein Rollback ist nicht atomar nachweisbar oder ein UI-Endpunkt kann schreiben.

Bei drei unabhängigen, nicht reproduzierbaren Messungen wird nicht weiter optimiert;
das Ergebnis ist `inconclusive` und die Baseline bleibt. Wenn für ein Modell keine
Rohdaten, Identität oder freie Ressource verfügbar ist, wird kein Ersatzwert
behauptet.

## 17. Erteilter Freigabestatus

Am 2026-08-30 wurde die Offline-Implementierung der unabhängigen
`friday_optimizer/`-Control-Plane durch den fortbestehenden Nutzerauftrag
freigegeben. Diese Freigabe umfasst ausschließlich Code, Datenmodell,
Replay-/Fehlerfalltests und lokale read-only Dokumentation ohne Modell- oder
Hardwarelauf.

Weiterhin gesperrt bleiben:

- jeder reale Hardware- oder Modelllauf ohne separates Nutzer-Gate;
- Downloads und Installationen;
- automatische Produktaktivierung oder Promotion vor dem vollständigen Gate;
- Verwendung oder Änderung des aktuell von Claude verwendeten Worktrees
  `.worktrees/ironmule-b7`.

Ein späterer realer Lauf darf nur manuell, AC-only, bei sicherer Fremdlastfreiheit,
sparsam und mit einer harten Gesamtfrist von höchstens 30 Minuten gestartet werden.
Echte vorhandene Daten und End-to-End-Tests haben Vorrang; synthetische Daten sind
nur für Rand- und Fehlerfälle zulässig. Die bisher definierten Correctness-,
Fingerprint-, Lease-, Rollback- und Promotionsgrenzen werden durch diese
Offline-Freigabe nicht gelockert.

## 18. Verifizierter Offline-Stand

Die Offline-Control-Plane `friday_optimizer/` ist materialisiert und die
zugehörige CLI-, History-, Dashboard- und Shadow-Struktur ist implementiert und
getestet. Der Stand umfasst die Module `Memory`, `Corpus`, `Dataset`, `Bridge`,
`Fingerprint`, `Candidates`, `Evaluator`, `Readiness`, `Lease`, `Session`,
`Profile`, `History`, `Orchestrator`, `IronMuleAdapter`, `Dashboard` und `CLI`.
Es gibt weiterhin keinen `tune`-, `activate`-, Download- oder Installationspfad.

Materialisiert sind `.friday-data/optimizer-v2.sqlite3` mit `401` Records,
`1,212,416 B`, Modus `0600`, SHA-256 `5f5d286c...ab2aa` und bestätigter
Chain/Integrity sowie `.friday-data/optimizer-dataset-v1.json` mit `392` Records,
`2,208,967 B`, Modus `0600` und SHA-256 `79ce...f5c8d`. Der Dataset-Status ist
`smoke_only/no_learning_claim` mit `train=2`, `val=0`, `holdout=0`. Die
Offline-Evidence umfasst `406` Discovery-, `392` Normalized- und `2` Eligible-
Records (`Q2`/`B27d`) sowie `400` Bridge- und `1` History-Record.

L1.1 ist damit offline abgeschlossen. L1.2 bleibt für Learned Ranking, GBDT und
BO offen, bis echte Daten getrennte Validation-/Holdout-Splits ermöglichen;
deterministische Baselines und Shadow-Auswertung sind implementiert und getestet.
L1.3 ist als Offline-Control-Plane mit Shadow-Adapter umgesetzt, aber reale
Adapterausführung, Profilpromotion und Produktaktivierung bleiben bis zu ihren
eigenen Gates gesperrt. Die letzte Dashboard-Schemaänderung ist im Stand enthalten;
es gibt hier keinen neuen P0/P1-Sicherheitsaudit. Der Claude-Worktree
`.worktrees/ironmule-b7` bleibt unangetastet.

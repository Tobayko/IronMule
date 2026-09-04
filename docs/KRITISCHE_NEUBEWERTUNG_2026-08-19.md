# Kritische Neubewertung — 19. August 2026

**Status:** Forschungspivot empfohlen; Architektur/Umsetzung nicht freigegeben.

Dieses Dokument ist eine methodische Entscheidungsvorlage. Es ersetzt keine Nutzerfreigabe
für Worker, Custom-Metal-Code, Architektur, Downloads, Installationen oder lokale Modelle.
Es wurden für diese Neubewertung keine GPU-Läufe, Tests, Downloads oder Installationen
ausgeführt.

## 1. Executive Verdict

Das Projekt ist als enger Forschungs-PoC sinnvoll, aber seine belastbare Forschungsfrage ist
nicht „Kann ein LLM einen autonomen Hardware-Compiler bauen?“. Die tragfähige Reihenfolge ist:

1. **H0:** Ist der Mess-, Correctness- und Fallback-Harness vertrauenswürdig?
2. **H1:** Kann ein deterministischer, template-beschränkter Suchloop auf einem festgelegten
   Apple-Silicon-Fingerprint eine reale Operation netto verbessern oder sicher bei der Baseline
   bleiben?
3. **H2:** Liefert ein LLM-Planer unter identischem Messbudget einen inkrementellen Nutzen gegenüber
   Grid/Random/Bayesian/Evolutionary Search?

Die bisherige Phase 1A ist deshalb kein Optimierungsnachweis, sondern ein **H0-
Messsystem-Preflight**. Ihre `2048²`-Matmul ist als starkes Backend-/Regressionstest-Signal
wertvoll, aber als erste Forschung über Self-Optimization ungeeignet. Die konkrete
Optimierungsfrage sollte erst danach an einer fusionierten, LLM-relevanten und
template-beschränkten Operation untersucht werden, zum Beispiel residual-add + RMSNorm.

Der Forschungspivot lautet damit:

> Ein reproduzierbares, versions- und hardwaregebundenes Messsystem für sichere MLX/Metal-
> Such- und Validierungsstudien, das deterministisches template-constrained Tuning zuerst
> gegen starke Apple-Runtime-Baselines und danach einen optionalen LLM-Planer fair evaluiert.

Das ist enger als eine allgemeine Runtime, aber wissenschaftlich falsifizierbar und offen für
ein ehrliches Nullresultat.

## 2. Claim-Ledger

| Claim | Entscheidung | Präzise Formulierung |
|---|---|---|
| MLX/Metal kann auf dem Zielgerät kontrolliert vermessen werden | **beibehalten** | Nach H0-Verifikation für den konkreten Fingerprint, nicht als bereits gemessene Performance. |
| Ein geschlossener Loop kann Kandidaten validieren, ablehnen und zurückrollen | **beibehalten, aber noch unbewiesen** | Erst nach A/A-, absichtlich-falsch-, langsam-, Timeout- und Missing-Data-Kontrollen. |
| Der Loop ist „hardware-aware“ im allgemeinen Sinn | **einschränken** | Zunächst nur „hardware- und versionsspezifisches Tuning auf einem M1-Max-Fingerprint“. |
| `2048²`-Matmul ist der kleinste echte Optimierungsnachweis | **verwerfen** | Sie bleibt H0-/starke-Backend-Kontrolle; die erste Optimierungsoperation ist eine einzelne, fusionierte LLM-Operation. |
| `mx.compile` ist eine Negativkontrolle | **korrigieren** | `mx.compile` ist eine Framework-Vergleichsvariante; ein echter H0-Nullpfad ist A/A. |
| Gleiche Matmul in Holdout-Prozessen beweist Generalisierung | **verwerfen** | Das sind unabhängige Bestätigungsprozesse derselben Workload, keine unbekannten Workloads. |
| LLM-gestützte Kerneloptimierung ist neu | **verwerfen** | Metal-Sci, KernelBench und aktuelle Arbeiten zeigen direkte Prior Art. |
| Apple-Silicon-spezifische, offene Methodik kann einen Beitrag leisten | **beibehalten, aber offen** | Beitrag muss gegenüber Metal-Sci, bestehenden Runtimes und Suchsystemen empirisch abgegrenzt werden. |
| Ein kleines lokales Modell sollte jetzt geladen werden | **verwerfen** | Erst H0 und H1; danach nur ein freigegebener Planner-Ablationstest. |
| End-to-End-LLM-Gewinn folgt aus einem Kernel-Gewinn | **verwerfen** | End-to-End ist ein späteres, separates Gate mit TTFT, Tokens/s, Memory und Amortisation. |

## 3. Direkte Prior Art und Forschungsrisiko

### 3.1 Metal-Sci ist eine direkte Kollision

[Metal-Sci](https://arxiv.org/abs/2605.09708) beschreibt bereits einen Apple-Silicon-
Metal-Benchmark mit zehn Aufgaben, CPU-Referenzen, Roofline-gebundener Bewertung,
Held-out-Größen, Runtime-Kompilierung und LLM-gesteuerter `(1+1)`-Evolution. Das zugehörige
Repository ist [vicgalle/metal-sci-kernels](https://github.com/vicgalle/metal-sci-kernels).

Damit sind „LLM schlägt Metal-Kernel vor“, automatische Kompilierung, Fitnessfeedback und
Held-out-Generalisation keine hinreichende Neuheitsbehauptung mehr. Ein möglicher Beitrag
des Projekts muss sich auf die strengere Mess- und Promotionsmethodik, MLX-/Runtime-
Integration, versionssicheres Optimization Memory, Baseline-Fairness oder ein belastbares
Nullresultat konzentrieren. Metal-Sci ist selbst ein Preprint/Workshop-Artefakt; es wird hier
als Prior-Art-Evidenz, nicht als endgültiger Goldstandard behandelt.

### 3.2 Benchmark-Fingerprinting ist ein adversariales Validitätsrisiko

[Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) berichtet über zwei
GPU-Kernel-Suiten, dass **16/53 = 30 %** der In-Distribution-Gewinner nicht auf gehaltene
Konfigurationen übertragen wurden. Die berichteten Mechanismen umfassen Konfigurations-
Fingerprinting, Overfitting und Leakage in den Gate-Pfad. Die methodische Konsequenz ist
entscheidend: Holdouts müssen auf Achsen liegen, die der Optimierer nicht einfach
enumerieren kann; Correctness allein genügt nicht; Holdout-Performance muss tatsächlich
gemessen werden.

### 3.3 KernelBench-Verified verschärft die Baseline- und Memory-Anforderung

[KernelBench-Verified](https://arxiv.org/abs/2607.16241) führt eine realistischere Baseline,
vier verborgene Testverteilungen und explizite Speed/Memory-Metriken ein. Die Autoren
berichten für ihre geprüfte Einzelrunden-Evaluation einen besten geometrischen Mittelwert
von `0.88x` statt `1.43x` im Standardprotokoll; kein Modell schlug die realistische
PyTorch-Baseline konsistent, und 28 % der Kernel des besten Modells erhöhten den Peak-
Speicher. Das zugehörige Framework ist
[facebookresearch/kernel_bench_verified](https://github.com/facebookresearch/kernel_bench_verified).

Diese Werte werden nicht auf MLX oder M1 Max übertragen. Sie sind ein direktes Warnsignal
gegen sichtbare Tests, schwache Baselines und reine Warmzeit-Claims.

### 3.4 Apple-Runtime-Baselines müssen aktiv geprüft werden

MLX dokumentiert sowohl Graph-/Codekompilierung ([`mx.compile`](https://ml-explore.github.io/mlx/build/html/usage/compile.html))
als auch Custom-Metal-Kernel ([Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)).
Für RMSNorm existiert zudem eine spezialisierte MLX-Funktion
([`mx.fast.rms_norm`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html)).
Diese Pfade sind aktive starke Baselines und müssen vor jedem Custom-Kernel-Claim geprüft
werden.

[BaseRT](https://arxiv.org/abs/2607.00501) ist eine relevante native-Metal-Apple-Runtime,
aber zum Zeitpunkt dieser Neubewertung ein Preprint. Die bis zu `1.56x` gegenüber llama.cpp
und `1.35x` gegenüber MLX sind **Autorenclaims**, keine unabhängige Projektmessung und kein
direkter M1-Max-Nachweis. BaseRT darf deshalb als externe Referenz markiert werden, nicht
als bereits bestätigte Baseline oder als Installationsauftrag.

## 4. Risiken, die vor einer Umsetzung geschlossen werden müssen

### Fatal für die Kernbehauptung

- Phase 1A hat keinen Custom-Kandidaten und keinen Suchraum; sie prüft kein
  Self-Optimization.
- Die Performance-Holdout-Prozesse wiederholen dieselbe `2048²`-Workload und sind keine
  unbekannten Workloads.
- Ein einzelner M1-Max-Fingerprint identifiziert keine allgemeine Hardware-Awareness.

### Major

- Drei Prozesscluster sind ein Engineering-Gate, kein belastbarer Power-/CI-Nachweis.
- Eine aus Baselinefehlern abgeleitete Toleranz darf nicht die unabhängige FP64-Semantik-
  policy ersetzen.
- Sichtbare Shapes und Seeds erlauben adaptive Spezialisierung; verborgene Tests müssen
  auf nicht-enumerierbaren Achsen liegen.
- Die Kandidatengenerierung, Suchbudgets und primäre H1-Metrik sind vor Phase 1B noch nicht
  vollständig eingefroren.
- Ein LLM-Vergleich ist ohne fixiertes Modell-/Prompt-/Decoding-Protokoll und gleiche
  Hardware-, Zeit- und Trialbudgets nicht interpretierbar.
- Warmzeit ohne Tuning-/Compile-Amortisation kann einen praktisch negativen Kandidaten als
  Erfolg darstellen.

### Minor, aber zu protokollieren

- ±5-%-Warmup und 5-%-Promotionsschwelle brauchen einen Pilot-/Power-Bezug.
- Peak-Memory und `ru_maxrss` sind Beobachtungen, keine harte Unified-Memory-Isolation.
- Primäre Zielmetrik, Guardrails und Multiple-Comparisons-Regel müssen getrennt werden.
- Operation-/Shape-Auswahl darf nicht nach beobachteten Erfolgen nachträglich geändert werden.

## 5. Neue Forschungsfragen und Hypothesen

### H0 — Messsystem

Für eine eingefrorene Operation misst der Harness wiederholbar, erkennt absichtlich gleiche,
langsame, falsche, abstürzende und unvollständige Kandidaten und lässt bei jedem Fehler die
Baseline aktiv. H0 erlaubt keinen Performanceclaim.

H0-Kontrollen:

- A/A: zwei semantisch identische Baselinearme mit identischer Messpipeline;
- absichtlich langsamer, aber korrekter Kontrollarm;
- absichtlich falscher Kontrollarm;
- kontrollierter Timeout-/Crash-Arm nach freigegebenem Worker;
- Missing-Data-/abgebrochener-Prozess-Arm;
- Replay aus unveränderten Rohdaten.

### H1 — deterministischer template-constrained Tuner

Für eine vorab registrierte einzelne Operation, eine endliche Menge validierter Template-
Familien und einen festen Hardware-/Software-Fingerprint findet ein deterministischer
Suchloop auf versiegelten Workload-Holdouts eine stabile Nettoverbesserung gegen starke
Baselines oder verwirft alle Kandidaten korrekt.

### H2 — inkrementeller LLM-Nutzen

Unter gleichem Kandidatenraum, gleichem Hardware-Trialbudget, gleichem Wall-Clock-Budget,
identischer Validierung und demselben versiegelten Testset verbessert ein LLM-Planer die
Best-of-Budget-Leistung, die Zeit bis zum ersten gültigen Gewinner oder die Kosten pro
gültigem Gewinner gegenüber Random/Grid/BO/Evolutionary Search.

H2 darf erst nach bestandenem H0 und einem reproduzierbaren H1-Loop untersucht werden.
Correctness und Promotion bleiben deterministisch; das LLM darf nur deklarative
Suchraum-/Templatevorschläge liefern.

### 5.1 Formale H1-Sperre

Der A/A-Pilot schätzt ausschließlich die Standardabweichung der Session-Log-Ratios. Er
verwendet keine Compile- oder Kandidatendaten; Pilotdaten dürfen nie in die bestätigende
H1-Auswertung gelangen.

Vor jeder Kandidatensichtung wird eine vollständige H1-Vorregistrierung eingefroren:

- Mindestwirkung `5 %`, `alpha = 0.05`, Power `0.80`;
- feste Familien-, Cluster- und Analyseplanung;
- mindestens fünf unabhängige Sessions je Arm und Familie;
- eine vorab registrierte obere Machbarkeitsgrenze für die Sessionzahl, empfohlen `20`;
- falls die benötigte Zahl diese Grenze überschreitet: `H1 infeasible/no claim`, ohne
  Öffnen der Regeln oder Erweiterung des Testsets.

Jede Hypothese/Revision erhält ein frisches versiegeltes Testset. H2 darf kein durch H1
geöffnetes Testset wiederverwenden. Der versiegelte Test wird erst nach Freeze aus breiten,
vorab registrierten Shape-, Value- und Layout-Verteilungen gezogen. Sein 256-bit-Seed liegt
außerhalb des Repositories; vorab wird nur ein kryptographischer Commit-Hash dokumentiert.
Dies ist ein vorgeschlagenes Forschungsprotokoll und keine Architekturfreigabe.

## 6. Statistische Einheit und Datenaufteilung

### 6.1 Statistische Einheit

Die inferenzielle Einheit ist eine **Workload-/Shape-Familie**, nicht ein einzelner
Timingblock. Timingblöcke sind gepaarte technische Wiederholungen innerhalb einer Familie.
Prozesse/Sessions bilden Cluster; ihre Anzahl und Varianz müssen vor der H1-Auswertung aus
einem Pilot abgeleitet werden.

Die bisherigen drei Charakterisierungs- und drei Bestätigungsprozesse sind höchstens ein
Engineering-Gate. Ein hierarchisches Bootstrap mit 10.000 Resamples erzeugt bei drei
Clustern keine zusätzliche Unabhängigkeit und darf keine wissenschaftliche Präzision
vorspiegeln. Die vollständige H1-Powerplanung wird nach dem A/A-Pilot, aber vor jeder
Kandidatensichtung, mit festen Familien und Clustern eingefroren. Der Pilot bleibt aus der
bestätigenden H1-Auswertung ausgeschlossen.

### 6.2 Dreiteilige Aufteilung

| Split | Verwendung | Sichtbarkeit für Planner/LLM |
|---|---|---|
| Entwicklung | Template- und Suchraumdebugging, Harness-Kalibrierung | sichtbar |
| Validierung | Entscheidung über Suchstrategie und Zwischenparameter vor dem Freeze | nur nach festgelegtem Protokoll |
| Versiegelter Test | endgültige Promotion und Publikation | unsichtbar; nur Hash/Version |

Die Aufteilung muss nicht-enumerierbare Achsen enthalten: mindestens andere Shape-/Größen-
familien, mehrere Werteverteilungen und getrennte Correctness-Inputs. Für den versiegelten
Test liegt ein 256-bit-Seed außerhalb des Repositories; vorab wird ausschließlich dessen
kryptographischer Commit-Hash gespeichert. Nach dem Freeze zieht der Evaluator daraus die
vorab registrierten Shape-, Value- und Layout-Verteilungen. Ein neuer Testfall nach Sichtung
der Ergebnisse ist eine neue Experiment-Revision, kein stiller Holdout.

### 6.3 Adaptive-Overfitting und Multiple Comparisons

- Der versiegelte Test wird erst nach Kandidaten-, Prompt-, Suchraum- und Budget-Freeze
  geöffnet.
- Jede Sichtung eines Testresultats zählt als adaptive Analyse und wird protokolliert.
- Jede nachträgliche Kandidaten-, Shape-, Seed- oder Metrikwahl eröffnet eine neue Revision.
- Die primäre Metrik wird genau einmal vorab festgelegt; Memory, Correctness, Timeout und
  Safety sind Guardrails.
- Bei mehreren Operationen/Familien wird die Familienebene ausgewertet; explorative
  Einzelgewinne werden nicht als globale Erfolgsrate ausgegeben.
- Berichtspflichtig sind alle Versuche inklusive Compilerfehler, langsamer, falscher und
  verworfener Kandidaten (kein Survivorship Bias).

## 7. Mess- und Entscheidungsdesign

### 7.1 Starke Baselines

Mindestens zu prüfen sind MLX eager, MLX `compile`, vorhandene MLX-Fast-Primitives, eine
feste handgeschriebene Referenzvariante sowie Grid/Random Search. PyTorch MPS und BaseRT
sind externe Vergleichspunkte, sofern semantisch, quantisierungs- und hardwareseitig
vergleichbar; BaseRT bleibt als Preprint/Autorenclaim gekennzeichnet.

### 7.2 Amortisation

Ein Kandidat ist nur praktisch besser, wenn für die erwartete Aufrufzahl `N` gilt. Dabei
ist `T_baseline` die Zeit der stärksten Baseline; alle Zeiten müssen nichtnegativ sein,
dieselbe Einheit verwenden und denselben registrierten Workload-Mix und Scope abdecken:

```text
Tuningkosten + Compilekosten + N × Warmzeit_Kandidat
< N × Warmzeit_Baseline
```

`N` wird vorab als Anwendungsszenario registriert oder als Break-even-Kurve berichtet.
Cold-, Compile-, Warm-, Tuning- und End-to-End-Zeit werden getrennt gespeichert. Für das
Hard-Gate gilt außerdem `T_strongest_baseline = T_baseline`: Falls
`T_strongest_baseline <= T_candidate`, ist die Entscheidung `no_break_even`, der
Break-even-Wert unendlich und das Gate fällt. Andernfalls wird exakt
`ceil((T_tune + T_compile) / (T_strongest_baseline - T_candidate))` verwendet; er muss
`<= N` sein. Negative, null- oder fehlende Zeit-Artefakte sind ungültig und dürfen nicht
in die Formel gelangen.

### 7.3 Primäre und sekundäre Metriken

**H1-Primär:** `R = T_candidate / T_strongest_baseline`, kleiner ist besser. Pro Familie
wird der Session-Median gebildet; über vorab registrierte Familien wird das geometrische
Mittel auf der Log-Skala mit 95-%-Clusterintervall ausgewertet. H1-Erfolg erfordert Gesamt-
`R <= 0.95`, obere 95-%-Cluster-KI `< 1.0` sowie keine Familienregression `R >= 1.05`.
Correctness, Memory und Safety sind Guardrails.

**H1-Amortisation:** separates Hard-Gate mit
`N_break_even = ceil((T_tune + T_compile) / (T_strongest_baseline - T_candidate))` bei
positivem Nenner. Bei `T_strongest_baseline <= T_candidate` gilt `no_break_even`/unendlich
und das Gate fällt. Der Wert muss `<= N` des registrierten Aufrufbudgets sein;
Amortisation ist nicht still in die Primärmetrik zu integrieren.

**H2-Primär:** genau eine Metrik, die finale Best-Valid-Sealed-Test-Ratio nach festem
Hardware-Trialbudget `B`, gepaart über neue H2-Familien gegen die stärkste deterministische
Suche. Trials-to-first-win, Gesamtwalltime, Modellkosten und Invalid-Rate sind sekundär.
H2-Erfolgsschwelle und Power werden im Modellantrag vor jedem Download oder jeder
Installation eingefroren, nicht nach dem Ergebnis.

**Sekundär:** Erfolg/Reject-Rate, Trials bis zum gültigen Gewinner, Tuning- und Compilezeit,
Correctness-Fehler, Peak-/Active-Memory, RSS, Timeout-/Crashrate, Order-/Sessioneffekte,
Energie-/Thermikwerte nur bei nachgewiesener Messqualität.

## 8. Revidierter minimaler Versuchsweg

1. H0-A/A- und Kontroll-Fixtures in der vorregistrierten Matmul-Umgebung qualifizieren.
2. Keine Self-Optimization aus H0 ableiten.
3. Nach Architektur-/Sicherheitsfreigabe eine einzelne fused residual-add + RMSNorm-
   Operation mit eingefrorener Semantik und begrenzten Templatefamilien registrieren.
4. Entwicklung, Validierung und versiegelten Test auf Shape-/Value-Familien trennen.
5. Deterministische Grid/Random-Suche und starke MLX-Baselines messen.
6. A-priori Cluster-/Powerplan nach dem A/A-Pilot einfrieren.
7. Correctness, Memory, Safety, Amortisation und Holdout-Promotion gemeinsam auswerten.
8. Erst nach H1-Freeze optional eine LLM-Ablation mit einem noch nicht ausgewählten,
   ausdrücklich freigegebenen Open-Source-Modell planen.
9. Erst danach ein Modell als reine Integrationslast testen; End-to-End bleibt ein eigenes
   Gate.

## 9. Kill- und Pivot-Kriterien

- H0-A/A instabil oder Kontrollarme falsch klassifiziert: keine Performanceaussage; Harness
  reparieren.
- Correctness-, Timeout-, Missing-Data- oder Rollback-Gate verletzt: Candidate nicht
  promoten; bei wiederholtem Sicherheitsfehler Custom-Code stoppen.
- Kein Netto-H1-Gewinn auf versiegelten Familien nach dem registrierten Budget: Pivot zu
  Benchmark-/Nullresultat statt Suchraumerweiterung.
- Gewinner überträgt sich nicht auf nicht-enumerierbare Holdout-Achsen: Claim auf die
  Entwicklungsfamilie begrenzen; keine Generalisation.
- Tuning-/Compile-Break-even wird im registrierten Anwendungsszenario nicht erreicht:
  nur Offline-/Install-Time-Caching erwägen.
- LLM schlägt klassische Suche bei gleichem Budget nicht: LLM aus dem inneren Tuner
  entfernen; Planner-/Erklärrolle als negatives H2-Ergebnis dokumentieren.
- Rangfolge kippt nach OS-/MLX-/Compileränderung: Artefakt quarantänisieren und Fingerprint-
  Gültigkeit neu prüfen.

## 10. Dashboard-Mindestfelder

Die spätere lokale Historien-UI muss neben dem bestehenden Rohdatenbezug mindestens anzeigen:

- Revision, Fingerprint und Quellenstatus;
- H0/H1/H2, Split und Workload-/Shape-Familie;
- Baseline, Kandidat, Suchstrategie und Trial-/Zeitbudget;
- primäres Laufzeitverhältnis mit Clusterintervall;
- Cold-/Compile-/Warm-/Tuning-/Break-even-Werte;
- Correctness-, Memory-, Timeout-, Crash- und Rollbackstatus;
- sichtbare/versiegelte Testset-ID nur als Hash;
- Modell-/Prompt-ID nur bei freigegebenem H2-Lauf;
- Status `promoted`, `baseline_fallback`, `invalid`, `regression` oder `not_run`;
- Ursache, Messwert, Korrektur und erneute Verifikation bei jedem Fehler.

## 11. Explizites Live-Modell-Gate

**Jetzt kein Modelltest.** Es wird aktuell kein Gemma-, GLM- oder anderes Open-Source-Modell
geladen, installiert oder ausgeführt. Die Modellwahl ist absichtlich nicht festgeschrieben.

Ein späterer H2-Antrag muss vorab enthalten:

- Modellname, exakte Version, Parameter-/Quantisierungsgröße und Lizenz;
- erwarteten Disk-/Unified-Memory-Bedarf und thermische Messstrategie;
- Plannerrolle, erlaubte Ausgabeform und harte Validatorgrenzen;
- Prompt-/Decoding-/Tool-Protokoll, Modellzeit und identisches Hardwarebudget;
- Vergleich gegen Random/Grid/BO und Erfolgskriterium;
- Installations-, Sicherheits- und Rollbackplan.

Ohne diese Angaben und ausdrückliche Nutzerfreigabe bleibt H2 gesperrt. Ein lokales Modell
darf während eines GPU-Benchmarks nicht gleichzeitig auf derselben GPU laufen, da es Speicher,
Thermik und Queueing verfälschen würde.

## 12. Entscheidung

Empfohlen wird die Reklassifizierung von Phase 1A zu H0-Messsystem-Preflight und danach ein
Forschungspivot auf H1 deterministischer template-constrained Suche. Die bisherige
Architekturfreigabe bleibt unverändert offen. `IMPLEMENTIERUNGSPLAN.md`, `CODEX_START.md`,
`docs/PHASE1A_ARCHITEKTURFREIGABE.md`, `AGENTS.md` und `ProjectAtlas/` werden durch diese
Entscheidungsvorlage nicht geändert.

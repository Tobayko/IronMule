# Hardware-Aware Self-Optimizing AI Runtime

## Forschungs- und Technikkonzept für einen Apple-M1-Max-Proof-of-Concept

**Stand:** 15. August 2026  
**Zielplattform der ersten Untersuchung:** Apple M1 Max, 32 GB Unified Memory  
**Bewertungsprinzip:** Keine Optimierung ohne Messung; keine Behauptung ohne Baseline; keine Promotion ohne unabhängige Validierung.

> **Update 19.08.2026:** Die [kritische Neubewertung](KRITISCHE_NEUBEWERTUNG_2026-08-19.md)
> empfiehlt einen Forschungspivot: Phase 1A ist ein H0-Messsystem-Preflight, nicht der
> Nachweis von Self-Optimization. Metal-Sci, Benchmark-Fingerprinting und KernelBench-
> Verified verschärfen die Anforderungen an Prior Art, nicht-enumerierbare Holdouts und
> starke Baselines. Architektur, Worker, Custom Metal und lokale Modelltests bleiben nicht
> freigegeben.

> **Kurzurteil:** Ein echter, lokal laufender Proof of Concept ist heute möglich. Realistisch ist jedoch kein autonomer „KI-Compiler“, sondern ein sicher begrenzter, überwiegend offline arbeitender Autotuner: Er beobachtet einen definierten Workload, variiert zulässige Metal-Kernel- oder Ausführungsparameter, verwirft falsche Kandidaten und speichert statistisch bestätigte Gewinner. Ein LLM kann Hypothesen, neue Template-Varianten und Versuchspläne vorschlagen. Für die eigentliche Parametersuche sind klassische Suchverfahren und später ein gelerntes Kostenmodell voraussichtlich besser, billiger und reproduzierbarer. Die Apple Neural Engine ist dabei kein frei programmierbares Kernel-Ziel.

### Bewertungsstufen

| Stufe | Bedeutung in diesem Dokument |
|---|---|
| **Heute problemlos machbar** | Öffentliche APIs und etablierte Verfahren; geringe Forschungsunsicherheit. |
| **Mit vertretbarem Aufwand machbar** | Für einen engen PoC mit sorgfältiger Implementierung realistisch. |
| **Experimentell** | Technisch baubar, Nutzen oder Robustheit muss jedoch erst gemessen werden. |
| **Forschungsproblem** | Keine verlässliche allgemeine Lösung; Ergebnis kann negativ sein. |
| **Unrealistisch** | Für die beschriebene Plattform, eine Einzelperson oder den genannten Umfang nicht glaubwürdig. |
| **Technisch nicht sinnvoll** | Möglich oder teilweise möglich, aber gegenüber vorhandenen Lösungen der falsche Ansatz. |

---

# A. Executive Summary

Die Kernhypothese ist prüfbar:

> Kann ein Agent aus konkreten Hardware-, Workload- und Messdaten eine zulässige Ausführungsvariante auswählen oder erzeugen, sie kontrolliert testen und eine auf unbekannten Testfällen reproduzierbare Verbesserung gegenüber starken Baselines finden – oder korrekt erkennen, dass keine Verbesserung vorliegt?

Der entscheidende Gegenstand ist nicht das Sprachmodell, sondern der **geschlossene, evidenzbasierte Optimierungsprozess**. Dessen minimale Form besteht aus:

1. einer exakt beschriebenen Operation und Eingabeverteilung,
2. einer unveränderlichen Referenzimplementierung,
3. einer begrenzten Menge zulässiger Kernel- oder Scheduling-Entscheidungen,
4. einem isolierten Compiler- und Benchmark-Worker,
5. Correctness-Tests einschließlich unbekannter Testfälle,
6. statistisch belastbaren Vergleichsmessungen,
7. einer versionierten Datenbank aller Versuche und
8. einer Promotion- und Rollback-Regel.

Auf dem vorhandenen M1 Max bietet sich **MLX plus `mx.fast.metal_kernel`** an. MLX kann Metal-Kernel aus Python erzeugen, JIT-kompilieren und mit frei gewähltem Grid und Threadgroup ausführen. Die offizielle Dokumentation weist zugleich darauf hin, dass neue Kernel kompiliert werden müssen und dass die erste Kompilierung erheblich länger als eine warme Ausführung dauern kann ([MLX: Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [MLX: Installation/JIT](https://ml-explore.github.io/mlx/build/html/install.html)). Genau deshalb müssen Compile-Zeit und warme Laufzeit getrennt gemessen werden.

Als erste fachlich sinnvolle Operation wird **fused residual-add + RMSNorm** empfohlen. Sie ist wesentlich kleiner als eine vollständige Transformer-Inferenz, besitzt aber echte LLM-Relevanz und einen plausiblen Optimierungsraum bei Datenzugriffen, Reduktion, SIMD-Gruppen, Threadgroup-Größe und Vektorisierung. Eine Matrixmultiplikation sollte zunächst nur als **negative Kontrolle** dienen: MLX- und Systembibliotheken enthalten bereits stark optimierte GEMM-Pfade; ein selbst erzeugter Kernel ist dort für einen Anfänger besonders wahrscheinlich langsamer.

Die Rollenverteilung sollte lauten:

```text
LLM-Planer (optional)
  └─ formuliert Hypothesen, wählt Suchraum/Template-Familie, interpretiert Profile
             ↓
klassischer Tuner
  └─ Grid/Random/Bayesian/Evolutionary Search innerhalb harter Grenzen
             ↓
Compiler + isolierter Worker
  └─ kompiliert, validiert, misst und protokolliert
             ↓
Promotion Gate
  └─ Correctness + Holdout + Konfidenzintervall + Ressourcenlimits
             ↓
Optimization Memory / späteres Kostenmodell
```

Diese Arbeitsteilung entspricht dem Stand erfolgreicher Systeme: Triton besitzt eingebautes Konfigurations-Autotuning; TVM MetaSchedule kombiniert messungsbasierte Suche, evolutionäre Verfahren und Kostenmodelle; XLA kann Autotuning-Ergebnisse persistent wiederverwenden; TorchInductor vermisst alternative Kernel und speichert Gewinner im Cache ([Triton Autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html), [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html), [XLA Persisted Autotuning](https://openxla.org/xla/persisted_autotuning), [PyTorch Inductor GEMM Autotuning](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/)). Neu wäre daher nicht „Hardware-Awareness“ an sich.

Die aussichtsreichste Innovation liegt in einer **offenen, Apple-Silicon-spezifischen und methodisch strengen Kombination** aus MLX/Metal-Kerneloptimierung, sicherem Ausführungsprozess, versionsgenauem Optimization Memory, CPU/GPU-Platzierungsversuchen auf Unified Memory und einem fairen Vergleich von LLM-Planung gegen nichtsprachliche Suchverfahren. Ein glaubwürdiges Projekt publiziert auch Nullresultate und verworfene Kandidaten.

---

# B. Realitätscheck

## B.1 Machbarkeitsmatrix

| Teilidee | Bewertung | Begründung |
|---|---|---|
| Statische CPU-/GPU-/Speicherinformationen erfassen | **Heute problemlos machbar** | Betriebssystem- und Metal-APIs liefern Chip, Kerne, Speicher sowie konkrete Device- und Pipeline-Limits. Marketingwerte wie 400 GB/s sind nur theoretische Obergrenzen und müssen durch eigene Microbenchmarks ergänzt werden. |
| Tensor-Shapes, Dtypes, Strides, Batch und Kontext erfassen | **Heute problemlos machbar** | Die Runtime kennt diese Werte an der Operations- beziehungsweise Graphgrenze. |
| MLX-Laufzeit, aktive/Peak-GPU-Speichernutzung und Synchronisation messen | **Heute problemlos machbar** | MLX stellt Synchronisation sowie Active/Peak-Memory-APIs bereit ([MLX Streams](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html), [Active Memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html), [Peak Memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)). |
| Eigene MLX-Metal-Kernel kompilieren und testen | **Mit vertretbarem Aufwand machbar** | Öffentliche MLX-API; Grid, Threadgroup und Source sind steuerbar. Korrekte GPU-Reduktionen und gute Performance erfordern dennoch Metal-Wissen. |
| Begrenzte Parameter automatisch tunen | **Mit vertretbarem Aufwand machbar** | Grid/Random Search oder Bayesian Optimization über einen validierten Suchraum sind etablierte Verfahren. |
| Automatisch Bottlenecks eines eng definierten Kernels klassifizieren | **Mit vertretbarem Aufwand machbar** | Roofline-artige Messung, Profiler-Counter und kontrollierte Varianten können eine begründete Klassifikation liefern. Sie bleibt eine Hypothese mit Konfidenz, kein unfehlbares Etikett. |
| Beliebige Workloads automatisch korrekt klassifizieren | **Forschungsproblem** | Fusion, Caches, asynchrone Ausführung, Queueing und Datenabhängigkeiten machen eindeutige Ursachen oft unmöglich. |
| LLM erzeugt beliebigen Metal-Code und dieser wird „sicher“ ausgeführt | **Experimentell bis Forschungsproblem** | Prozessisolation, Timeouts und Input-Validierung reduzieren Risiken; GPU-Treiber, Compiler und geteilte GPU bleiben aber Teil der Angriffs- und Absturzfläche. Ein normaler Worker ist keine harte GPU-Virtualisierung. |
| LLM verbessert bestehende MLX-Kernel zuverlässig | **Forschungsproblem** | Einzelne Erfolge sind plausibel; verlässliche Verbesserung gegen starke, unbekannte Baselines über viele Operationen ist nicht belegt. |
| Erfolgreiche Konfigurationen persistent speichern und invalidieren | **Heute problemlos machbar** | Datenbank und inhaltsadressierte Artefakte sind Standardtechnik; schwierig ist die vollständige Cache-Key- und Validierungsdefinition. |
| Aus Benchmarkdaten ein Kosten-/Rankingmodell lernen | **Mit vertretbarem Aufwand machbar**, sobald Daten vorliegen | Für strukturierte Features ist ein Gradient-Boosting-Modell ein guter erster Kandidat. Ohne ausreichend vielfältige Daten lernt es nur die Suchhistorie auswendig. |
| CPU/GPU-Auswahl für einzelne Operationen messen | **Mit vertretbarem Aufwand machbar** | MLX unterstützt CPU- und GPU-Streams auf Unified Memory; kleine Operationen können auf der CPU schneller sein ([MLX Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)). |
| Einen dynamischen Gesamtgraphen optimal auf CPU/GPU verteilen | **Experimentell bis Forschungsproblem** | Lokale Einzeloperationsgewinne können durch Synchronisation, Cache- und Queue-Effekte verloren gehen. Der Suchraum wächst kombinatorisch. |
| Beliebige eigene Kernel auf der Neural Engine ausführen | **Unrealistisch / öffentlich nicht möglich** | Core ML nimmt Modellgraphen entgegen; Apple veröffentlicht keine allgemeine ANE-Kernel-Schnittstelle. |
| CPU/GPU/ANE über Core ML grob steuern | **Mit vertretbarem Aufwand machbar, aber begrenzt** | `MLComputeUnits` erlaubt Gerätemengen; die Runtime partitioniert. Es gibt kein „NeuralEngineOnly“ und keine präzise benutzerdefinierte ANE-Schedule ([Apple `MLComputeUnits`](https://developer.apple.com/documentation/coreml/mlcomputeunits)). |
| Online in einer produktiven Anfrage ständig neuen Low-Level-Code erzeugen | **Technisch nicht sinnvoll für Phase 1** | Kalte Kompilierung, Varianz, thermische Zustände und Zuverlässigkeitsrisiken dominieren. Besser: offline, bei Installation oder in einer kontrollierten Wartungsphase tunen. |
| Eigene IR in Phase 1 bauen | **Technisch nicht sinnvoll** | Ein typisiertes Execution-Plan-Schema reicht. MLIR/TVM besitzen bereits ausdrucksstarke IRs; eine eigene IR braucht erst einen nachgewiesenen Mehrbackend-Bedarf. |
| Plattformübergreifende Production Runtime als Einzelprojekt | **Unrealistisch** | Compiler-, Treiber-, Correctness-, Sicherheits- und Wartungsfläche entsprechen einem langfristigen Teamprojekt. |

## B.2 Was der PoC beweisen kann – und was nicht

Ein erfolgreicher PoC kann zeigen, dass der Loop:

- eine reale Operation anhand realer Messdaten optimiert,
- fehlerhafte oder langsamere Varianten zuverlässig aussortiert,
- eine Form-/Hardware-spezifische Konfiguration wiederfindet und im Cache speichert,
- Gewinne auf zuvor nicht zur Suche verwendeten Testeingaben bestätigt und
- Baselines stehen lässt, wenn sie besser sind.

Er beweist **nicht**, dass:

- ein LLM der Grund für den Gewinn war,
- die gefundene Variante auf anderen Chips, OS-Versionen oder Shapes besser ist,
- der Ansatz auf vollständige Modelle oder mehrere Plattformen skaliert,
- generierter Kernelcode produktionssicher ist oder
- allgemeine Compilerheuristiken ersetzt werden können.

Um den Zusatznutzen des LLM zu belegen, ist ein separates Experiment erforderlich: gleiche Kandidatenzahl, gleiches Zeitbudget und identische Validierung für LLM-Planer, Grid/Random Search und Bayesian beziehungsweise evolutionäre Suche. Ohne diese Ablation ist das Ergebnis nur ein Autotuning-, kein LLM-Ergebnis.

## B.3 Die realistische Produktform

Der sinnvolle Ausgangspunkt ist ein **hardwareabhängiger Optimierungsdienst außerhalb des Hot Path**:

- Er läuft nach Installation, Modellwechsel oder explizit im Labor.
- Er produziert signierte beziehungsweise inhaltsadressierte, getestete Artefakte und einen Execution Plan.
- Die produktive Runtime lädt ausschließlich promovierte Artefakte.
- Bei Versions-, Shape- oder Hardwareabweichung fällt sie auf die bekannte Baseline zurück.

Kontinuierliche Selbstmodifikation während einer Benutzeranfrage wäre zunächst weder wissenschaftlich sauber noch betrieblich robust.

---

# C. Existing Technology Analysis

## C.1 MLX und Metal

MLX deckt bereits mehrere Bausteine der Zielarchitektur ab:

- **Unified Memory und Streams:** Arrays müssen zwischen CPU und GPU nicht über eine explizite Kopie bewegt werden. MLX fügt Abhängigkeiten zwischen Streams ein. Das beseitigt jedoch weder Dispatch- und Synchronisationskosten noch Cache-Kohärenz, Page-Residency oder konkurrierende Bandbreitennutzung ([MLX Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)).
- **Graphoptimierung:** `mx.compile` kann unter anderem Operationen fusionieren und gemeinsame Teilausdrücke eliminieren. Der erste Aufruf beinhaltet Graph- und Codeerzeugung; neue Shapes können neue Kompilierung auslösen ([MLX Compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)).
- **Custom Kernel:** `mx.fast.metal_kernel` erzeugt aus Metal-Source, Ein-/Ausgabeinformationen und Startkonfiguration einen Kernel. Neue Kernel werden als Metal Library erstellt und gegebenenfalls JIT-kompiliert ([MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)).
- **Custom Extensions:** Wo die Python-API nicht reicht, lassen sich eigene CPU-/GPU-Operationen integrieren ([MLX Extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)).
- **Capture:** MLX kann eine `.gputrace`-Aufzeichnung starten, die anschließend in Apples Werkzeugen untersucht wird ([MLX Metal Capture](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.start_capture.html)).

Die MLX-Dokumentation zeigt selbst einen erheblichen Gewinn eines spezialisierten Custom Kernels gegenüber einer zusammengesetzten Implementierung. Das ist ein **Existenzbeispiel**, aber kein Beleg dafür, dass ein Agent beliebige MLX-Standardkernel schlagen wird. Gerade hochoptimierte Primitive sind die stärkste Baseline, nicht die schwächste Python-Formulierung.

## C.2 Metal-Compiler, Limits und Profiler

Metal bietet bereits die Compiler- und Laufzeitebene; ein eigener Compiler ist nicht nötig. Auf der M1-Familie gilt laut aktueller Apple-Capability-Tabelle die GPU-Familie Apple7. Für diese Familie nennt Apple unter anderem maximal 1.024 Threads pro Threadgroup und 32 KB Threadgroup-Speicher. Der für einen konkreten Kernel zulässige Wert kann niedriger sein und muss aus dem Pipeline State gelesen werden ([Metal Feature Tables](https://developer.apple.com/metal/capabilities/), [Pipeline-spezifisches Thread-Limit](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup)). Ein Agent darf deshalb keine statischen Maximalwerte blind als gültige Konfiguration verwenden.

Metal stellt programmatische Counter-Sample-Buffer bereit. Unterstützte Counter-Sets müssen jedoch pro Gerät abgefragt werden; mögliche Sets umfassen Zeitstempel-, Statistik- und Stage-Utilization-Counter. Sampling-Barrieren und Profiler verändern den Ablauf und gehören daher nicht in dieselben Runs wie die finale Zeitmessung ([GPU Counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers), [Counter Sampling](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers), [Common Counter Sets](https://developer.apple.com/documentation/metal/mtlcommoncounterset)).

Für die Kerneloptimierung sind vier Ebenen zu unterscheiden:

- Device-/Buffer-Zugriffe verwenden den gemeinsamen physischen Speicher, laufen aber weiterhin durch GPU-Caches und verbrauchen Bandbreite.
- Threadgroup Memory ist expliziter, schneller lokaler Scratch-Speicher mit hartem Kapazitätslimit; mehr Nutzung kann die Zahl gleichzeitig residenter Threadgroups reduzieren.
- Register beziehungsweise thread-lokaler Zustand sind nicht beliebig sichtbar steuerbar; starkes Unrolling und viele Zwischenwerte können Registerdruck und damit Parallelität verschlechtern.
- SIMD-Group-Primitiven können Reduktionen ohne eine Barriere nach jedem Schritt verkürzen. Die tatsächliche SIMD-Breite und Pipelinegrenzen werden aus dem kompilierten Pipeline State abgefragt, nicht geraten.

„Cache-Strategie“ bedeutet auf Metal deshalb primär Datenlayout, Zugriffslokalität, Tile-Größe und Wiederverwendung. Der Agent erhält keine allgemeine öffentliche Schnittstelle, mit der er Apples GPU-Caches frei partitioniert oder deren Replacement Policy programmiert.

Generische CUDA-/Triton-Begriffe dürfen nicht eins zu eins auf Metal übertragen werden:

| Generischer Hebel | Bedeutung auf dem Metal-PoC |
|---|---|
| Threadgroup-/Blockgröße | Anzahl Threads pro Threadgroup; gegen Pipeline- und Gerätegrenzen validieren |
| Tile Size | Compile-time-Parameter beziehungsweise eigene Templatevariante für die verarbeitete Datenkachel |
| „Anzahl Warps“ | kein separater Metal-Launchparameter; Zahl der SIMD-Groups ergibt sich aus Threadzahl und abgefragter SIMD-Breite |
| Vector Width | MSL-Vektortypen und gebündelte Loads/Stores; Alignment, Tail und tatsächlichen Codegen prüfen |
| Loop Unrolling | Template-/Compilerentscheidung; kann durch Codegröße/Registerdruck schaden |
| Pipeline Stages | in MSL gegebenenfalls manuell strukturierte Prefetch-/Double-Buffer-Variante, keine universelle Triton-Option |
| Memory Layout | Tensor-/Stride-Contract plus Indexierung; Layoutkonvertierung muss in den Gesamtbenchmark |
| Kernel Fusion | neue gemeinsame Semantik/Signatur und Correctness-Suite, nicht nur ein Launchflag |
| Precision/Math Mode | eigene numerische Policy mit Accuracy-Gate; kein kostenloser Performanceparameter |
| Cache Strategy | indirekt über Zugriffsmuster, Tiles und Wiederverwendung; keine frei programmierbare Cachepolitik |

Xcode GPU Capture, Metal Debugger, Metal System Trace und Counter-Analyse sind die primären Diagnosewerkzeuge ([Metal Developer Workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows), [Apple-GPU-Counteranalyse](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)). Eine GPU Trace ist geräte- und OS-spezifisch; sie ist ein Diagnoseartefakt, kein portabler Performancebeweis ([GPU Trace Replay](https://developer.apple.com/documentation/xcode/replaying-a-gpu-trace-file)).

## C.3 PyTorch MPS

PyTorch MPS setzt Operationen über MPS Graph und abgestimmte Metal-Pfade um ([PyTorch MPS Backend](https://docs.pytorch.org/docs/2.13/notes/mps.html)). In der aktuellen stabilen API kann PyTorch außerdem Metal-Shader kompilieren, Metallib-Dateien laden, synchronisieren, Speicherzustände abfragen und Metal Captures/Signposts erzeugen ([PyTorch MPS API](https://docs.pytorch.org/docs/2.13/mps.html), [`torch.mps.compile_shader`](https://docs.pytorch.org/docs/2.13/generated/torch.mps.compile_shader.html)).

PyTorch MPS ist daher nicht bloß eine passive Vergleichslaufzeit. Für den ersten PoC bleibt MLX sinnvoller, weil das Projekt explizit MLX-nahe lokale Inferenz anvisiert und dessen Custom-Kernel-Weg kompakt ist. Dennoch gehören semantisch identische PyTorch-MPS-Varianten als externe Baseline in die Messmatrix. Backend-Support und Compile-Verhalten müssen pro Operation auf genau dieser Installation geprüft werden.

## C.4 Core ML und Apple Neural Engine

Core ML kann einen Graphen auf CPU, GPU und Neural Engine partitionieren. Mit `MLComputeUnits` lässt sich nur eine **Menge zulässiger Geräte** wählen: alle Geräte, CPU-only, CPU+GPU oder CPU+Neural-Engine. Eine öffentliche Option „nur Neural Engine“ existiert nicht. Die tatsächliche Partitionierung hängt von Modell, Operatoren, Shapes, Hardware und Systemsoftware ab ([Apple `MLComputeUnits`](https://developer.apple.com/documentation/coreml/mlcomputeunits), [Core ML Typed Execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)).

`MLComputePlan` kann erwartete Geräteunterstützung, bevorzugte Geräte und geschätzte relative Operationskosten offenlegen. Das ist für spätere Planung nützlich, aber keine direkte Schedule-Kontrolle und kein Messersatz ([Apple `MLComputePlan`](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n), [WWDC: Core ML Performance Reports](https://developer.apple.com/videos/play/wwdc2024/10161/)). Core-ML-Custom-Layer sind zudem an das ältere NeuralNetwork-Backend gebunden; Apple empfiehlt zusammensetzbare Standardoperationen, damit der Compiler Hardwarepfade nutzen kann ([Core ML Custom Operators](https://apple.github.io/coremltools/docs-guides/source/custom-operators.html)).

Folglich ist die ANE für Phase 1 bewusst auszuschließen. Eine spätere Studie kann verschiedene Core-ML-Compute-Unit-Mengen und Graphformulierungen messen. Sie kann aber keine beliebigen ANE-Kernel erzeugen oder die ANE wie eine Metal-GPU programmieren.

## C.5 Autotuning, Scheduling und ML-Compiler

Wesentliche Teile der Idee existieren seit Jahren:

| System | Bereits vorhandene Idee | Bedeutung für dieses Projekt |
|---|---|---|
| **Triton** | Parameterkonfigurationen werden bei Schlüsseländerungen vermessen; eine Performance-Funktion kann Kandidaten vorfiltern. | Direktes Vorbild für begrenztes Kernel-Autotuning; offizieller Schwerpunkt CUDA/AMD, nicht Metal ([Triton Autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html)). |
| **TorchInductor** | Graph-/Kernel-Fusion, mehrere GEMM-Backends, Laufzeitbenchmarking und Cache des Gewinners. | Zeigt, dass Benchmark-gestützte Backendwahl bereits produktive Compilertechnik ist ([Inductor Fusion](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/), [Inductor GEMM Autotuning](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/)). |
| **TVM / MetaSchedule** | Suchräume aus Schedule-Transformationen, Builder/Runner, evolutionäre Suche, XGBoost-/MLP-Kostenmodelle und persistente Datenbank. | Der nächste direkte Vorläufer für Search + Measurement + Learned Cost Model ([TVM MetaSchedule Tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html), [MetaSchedule API](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html)). |
| **Ansor** | Hierarchische automatische Schedule-Suche mit gelerntem Kostenmodell. | Zeigt den Forschungsstand bei automatischem Scheduling, ohne ein LLM vorauszusetzen ([Ansor, OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng)). |
| **XLA** | Per-Fusion-Autotuning und persistenter Cache mit Hardware-/Versionsanforderungen. | Fast direktes Vorbild für Optimization Memory und dessen Invalidierung ([XLA Persisted Autotuning](https://openxla.org/xla/persisted_autotuning)). |
| **MLIR** | Erweiterbare Dialekte, GPU-Abstraktionen und Transform Dialect zur expliziten Steuerung von Transformationen. | Geeignete Infrastruktur für spätere Backends, aber weder Suchalgorithmus noch Kostenmodell an sich ([MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/), [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)). |
| **TVM Metal Codegen** | Erzeugung von Metal Shading Language und Metal-Target-Unterstützung. | Belegt, dass selbst Metal-Codegen/Autotuning keine völlig unbesetzte Nische ist ([TVM Codegen](https://tvm.apache.org/docs/arch/codegen.html), [TVM Targets](https://tvm.apache.org/docs/reference/api/python/target.html)). |

Eine eigene IR wäre erst gerechtfertigt, wenn ein konkretes, vorhandene IRs nicht gut abbildendes Problem nachgewiesen ist – etwa ein hardwareübergreifender Execution Plan mit Messprovenienz und Rollback-Semantik. Selbst dann sollte zunächst ein kleines Plan-Schema **oberhalb** vorhandener Compiler-IRs entstehen, keine neue Compiler-IR.

## C.6 LLM-generierte Kernel

Auch „LLM schreibt Kernel, Evaluator kompiliert und vermisst sie“ existiert bereits. KernelBench führte 250 PyTorch-Workloads und Feedbackschleifen für LLM-erzeugte Kernel ein; die ursprünglichen Ergebnisse zeigten zugleich, dass die damaligen Modelle die Baseline nur bei einem Minderheitsanteil erreichten ([KernelBench](https://arxiv.org/abs/2502.10517)).

Die 2026 veröffentlichte Untersuchung KernelBench-Verified ist für dieses Projekt besonders wichtig: Unter realistischeren Baselines, verborgenen Testverteilungen und Prüfungen gegen Reward Hacking sank die gemessene Leistung deutlich; die Autoren berichten zudem häufig erhöhte Peak-Memory-Nutzung ([KernelBench-Verified Paper](https://arxiv.org/abs/2607.16241), [zugehöriges Repository](https://github.com/facebookresearch/kernel_bench_verified)). Als Preprint ist das kein endgültiges Urteil, aber ein starkes Warnsignal: Sichtbare Correctness-Inputs und eine einzelne Performancezahl reichen nicht.

**Direkte Prior-Art-Korrektur:** [Metal-Sci](https://arxiv.org/abs/2605.09708) und das
[zugehörige Repository](https://github.com/vicgalle/metal-sci-kernels) kombinieren bereits
Apple-Silicon-Metal-Kernel, Runtime-Kompilierung, held-out Größen und LLM-gesteuerte
Evolution. [Gaming Without an Attacker](https://arxiv.org/abs/2608.08722) berichtet
`16/53 = 30 %` nicht übertragene In-Distribution-Gewinner und zeigt Benchmark-
Fingerprinting unter Selektionsdruck. Die Projektneuheit darf daher nicht mehr in
„LLM erzeugt Kernel mit Feedback“ liegen; sie muss als engere, reproduzierbare
Mess-/Promotionsmethodik gegen diese Systeme evaluiert werden.

AlphaEvolve zeigt umgekehrt, dass Sprachmodelle in Verbindung mit evolutionärer Suche und automatischen Evaluatoren echte algorithmische und Kernelverbesserungen finden können. Das System verwendet allerdings starke Modelle, große Evaluationsinfrastruktur und domänenspezifische Bewertungsfunktionen; es ist kein Beleg, dass ein kleines lokales 7B-Modell auf einem M1 Max dasselbe leistet ([Google DeepMind: AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)).

Die korrekte Schlussfolgerung lautet: **LLM-gestützte Kerneloptimierung ist plausibel, aber weder neu noch zuverlässig gelöst.** Robuste, Apple-spezifische Methodik kann trotzdem ein wertvoller Forschungsbeitrag sein.

---

# D. Novelty Analysis

## D.1 Was nicht neu ist

Nicht als Innovation beansprucht werden sollten:

- hardwareabhängige Auswahl von Kernelparametern,
- messungsbasiertes Autotuning,
- Kernel-Fusion und Layout-/Tile-Suche,
- learned cost models,
- persistente Autotuning-Caches,
- heterogene Graphpartitionierung,
- LLM-erzeugter Kernelcode mit Compiler-/Benchmarkfeedback,
- eine Compiler-IR oder ein GPU-Kernel-DSL als allgemeine Idee.

Diese Elemente sind in Triton, TVM/Ansor/MetaSchedule, XLA, TorchInductor, MLIR und aktueller LLM-Kernel-Forschung bereits vorhanden.

## D.2 Mögliche echte Beiträge

Ein ernstzunehmendes Projekt könnte dennoch neu beziehungsweise unterversorgt sein, wenn es die folgenden Punkte gemeinsam und offen löst:

1. **MLX-/Metal-spezifischer, reproduzierbarer Optimierungsloop.** Ein standardisierter Korpus echter Apple-Silicon-Operationen, Shapes, Baselines, Counterdaten und negativer Ergebnisse.
2. **Robuste Validierung gegen Optimizer-Overfitting.** Verborgene Inputverteilungen, semantische Hashes, Speichergrenzen, Mehrprozesswiederholungen und Promotion nur auf Holdout-Daten.
3. **Hybridarchitektur mit fairer Ablation.** LLM nur dort einsetzen, wo es Grid/Random/Bayesian/Evolutionary Search unter gleichem Budget messbar ergänzt.
4. **Execution-Plan-Lernen auf Unified Memory.** Empirische CPU/GPU-Platzierung einschließlich Übergangs-, Synchronisations- und Energieeffekten, nicht anhand nomineller FLOPS.
5. **Versions- und Kontext-sicheres Optimization Memory.** Nicht nur Gewinner speichern, sondern alle Messungen, Fehler, Compilerartefakte, thermischen Zustände und Gültigkeitsgrenzen.
6. **Ein transparentes Negativresultat.** Auch der belastbare Nachweis, dass ein LLM unter realistischen Bedingungen keinen Zusatznutzen gegenüber einem kleinen Tuner bietet, wäre wissenschaftlich wertvoll.
7. **Nicht-enumerierbare Evaluation.** Holdout-Performance, versteckte Wertverteilungen,
   Shape-Familien und Fingerprint-Erkennung müssen getrennt vom sichtbaren Suchsignal
   validiert werden; sichtbare Correctness allein ist kein Generalisationsbeleg.

## D.3 Die engste vertretbare Neuheitsbehauptung

Die frühere Formulierung „offenes, hardware- und versionsspezifisches Experimentalsystem
für sichere, messungsbasierte MLX/Metal-Kernel- und Execution-Plan-Suche ...“ wird als
**superseded** markiert: Metal-Sci und die aktuelle Benchmark-Fingerprinting-Literatur
belegen, dass diese Kombination allein keine Neuheitsbehauptung trägt.

Die engere, noch prüfbare Forschungsfrage lautet:

> Kann ein auf MLX/Metal und einen konkreten Apple-Silicon-Fingerprint begrenzter,
> template-constrained Mess- und Promotionsloop unter nicht-enumerierbaren Holdouts
> deterministisch sichere Nettoverbesserungen finden oder korrekt bei der starken Baseline
> bleiben, und liefert ein LLM unter gleichem Messbudget einen inkrementellen Nutzen?

Ob diese engere Frage publizierbar neu ist, bleibt offen und muss mit Metal-Sci,
KernelBench-Verified, Apple-Runtime-Baselines und einer systematischen Repository-Suche
abgegrenzt werden. Bis dahin sind „hardware-aware“, „sicher“ und „LLM-Verbesserung“ nur
Hypothesen, keine Projektclaims.

---

# E. Apple-M1-Max-Machbarkeit

## E.1 Verifizierter lokaler Ausgangszustand

Der historische Snapshot vom 15. August 2026 ist durch den autoritativen
[PROJECT_STATUS.md](../PROJECT_STATUS.md)-Stand vom 19. August 2026 superseded. Die
aktuelle lokale Prüfung ergab beziehungsweise bestätigte:

| Merkmal | Tatsächlicher Wert auf dem Zielgerät |
|---|---|
| Rechner | MacBook Pro, Modellkennung `MacBookPro18,2` |
| SoC | Apple M1 Max |
| CPU | 10 Kerne, davon 8 Performance- und 2 Efficiency-Kerne |
| GPU | **32 Kerne**; die Unsicherheit „24 oder 32“ ist für dieses Gerät damit aufgelöst |
| Neural Engine | 16 Kerne laut Apple-Spezifikation |
| Unified Memory | 32 GB |
| Nominelle Speicherbandbreite | 400 GB/s laut Apple-Spezifikation; nicht mit nachhaltig erreichbarer Workload-Bandbreite gleichzusetzen |
| Betriebssystem | macOS 26.5.2, Build 25F84, arm64 |
| Metal | Metal 4 gemeldet; M1-Serie entspricht Apple-GPU-Familie 7 |
| Entwicklungswerkzeuge | Xcode 26.6 (Build 17F113) und Command Line Tools vorhanden |
| Python-Pakete | MLX 0.32.0 und PyTorch im bestehenden `.venv`; API-Details siehe `PROJECT_STATUS.md` |

Die Hardwaredaten entsprechen Apples Spezifikation der 2021er M1-Max-Variante ([Apple MacBook Pro 16-inch 2021](https://support.apple.com/en-gb/111901), [Apple M1 Max Newsroom](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)). Die lokale Prüfung ist **kein Performancebenchmark**, sondern nur ein Preflight. Die detaillierten lokalen Mess- und Auditgrenzen bleiben in `PROJECT_STATUS.md` autoritativ.

## E.2 Welche Hardware- und Laufzeitinformationen tatsächlich verfügbar sind

| Informationsart | Direkt programmatisch im normalen Loop | Nur/primär über Profiler oder privilegiertes Tool | Statisch | Eingeschränkt oder nicht öffentlich |
|---|---:|---:|---:|---:|
| SoC, CPU-Kerne, RAM, OS-Build | ✓ |  | ✓ |  |
| CPU-ISA-/SIMD-Features | ✓ über unterstützte OS-/Compilerabfragen |  | ✓ | konkrete Intrinsics bleiben build- und architekturabhängig |
| GPU-Name/Familie, empfohlene Working-Set-Größe, Threadgroup-/Pipeline-Limits | ✓ über Metal |  | teilweise |  |
| CPU-Cacheangaben | teilweise | ✓ für Verhalten | teilweise | Apple-GPU-Cachetopologie/-größen sind nicht als vollständiger stabiler Runtimevertrag öffentlich |
| Tensor-Shapes, Strides, Dtype, Batch, Kontext, Operationsname | ✓ über Instrumentierung |  |  |  |
| End-to-End-Zeit, Kernelblock-Zeit bei expliziter Synchronisation | ✓ |  |  |  |
| MLX aktive und Peak-Speichernutzung | ✓ |  |  | Werte haben API-spezifische Definitionen; Cache-/Systemspeicher nicht automatisch enthalten |
| CPU-Auslastung und Prozess-RSS | ✓ über OS-APIs |  |  | Samplingauflösung begrenzt |
| Metal-Kernel-Timestamps und unterstützte GPU-Counter | ✓ in einem Low-Level-Metal-Harness | ✓ bequem in Xcode |  | Countermenge geräteabhängig; nicht alles über MLX-Python exponiert |
| Kernelbelegung, Cacheverhalten, Bandbreitenindikatoren, Stalls | teilweise | ✓ Metal Capture/Instruments |  | keine universelle, stabile „ein Wert“-API |
| GPU-Auslastung/Frequenz/Leistung auf Systemebene | teilweise über Systemwerkzeuge | ✓ `powermetrics`, Instruments |  | nicht zuverlässig einem kurzen einzelnen MLX-Kernel zuzuschreiben |
| ANE-Leistung und Aktivität | grob | ✓ `powermetrics`/Instruments, soweit unterstützt |  | keine freie Kerneltelemetrie oder Schedulingkontrolle |
| Thermalzustand | ✓ als grobe Stufe | ✓ ergänzende Diagnose |  | keine öffentliche stabile API für exakte Chiptemperaturen erforderlich/verfügbar |
| Exakter Energieverbrauch eines Mikro-Kernels |  | nur angenähert über lange Messfenster |  | nicht zuverlässig isoliert |
| Theoretische Bandbreite und veröffentlichte Kernausstattung |  |  | ✓ | sagt wenig über den konkreten Kernel aus |

`ProcessInfo.thermalState` liefert einen groben thermischen Zustand; macOS entscheidet selbst über Schutzmaßnahmen und Drosselung ([Apple `thermalState`](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.property)). Das auf dem Gerät vorhandene `powermetrics` kann CPU-, GPU-, ANE- und Thermal-Sampler anbieten. Seine eigene Hilfe bezeichnet durchschnittliche Leistungswerte als Schätzungen und warnt vor geräteübergreifenden Vergleichen. Es eignet sich daher als **sekundäre, lang laufende Same-Device-Metrik**, nicht als Wahrheitsquelle für die Energie eines 20-Mikrosekunden-Kernels.

Metal-Counter müssen zur Laufzeit enumeriert werden. Ein Dokumentationsname wie „stage utilization“ garantiert nicht, dass genau dieser Counter auf jeder OS-/Gerätekombination verfügbar oder störungsfrei messbar ist. Der PoC braucht daher eine Capability-Abfrage und explizit markierte fehlende Werte, keine erfundenen Nullen.

## E.3 Datentypen: Frameworksupport ist nicht Hardwarebeschleunigung

MLX führt unter anderem FP32, FP16, BF16 sowie mehrere Integer-Typen auf. FP64 ist auf der GPU nicht unterstützt und führt dort laut Dokumentation zu einem Fehler ([MLX Data Types](https://ml-explore.github.io/mlx/build/html/python/data_types.html)). Daraus folgt nicht, dass jede gelistete Ganzzahlbreite auf dem M1 Max durch eine gleichartige native Matrixeinheit beschleunigt wird.

Insbesondere sind „INT4-Modell“ und „native INT4-ALU“ verschiedene Aussagen: Quantisierte Modelle speichern Gewichte häufig gepackt und dequantisieren beziehungsweise verwenden spezialisierte Bibliotheksoperationen. Core ML kann Gewichte in niedriger Bitbreite komprimieren, doch Beschleunigung und Operatorpfad hängen von Gerät, OS, Format und Compute Unit ab ([Core ML Optimization Overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)). Für den M1 Max dürfen FP16/BF16/INT8/INT4-Leistungsannahmen nur aus gemessenen, konkreten Operationen stammen.

## E.4 Lokales Control-Plane-Modell

MLX-LM unterstützt lokale Generierung und quantisierte Modelle ([MLX-LM Repository](https://github.com/ml-explore/mlx-lm)). Ein Modell im Bereich 7B bis 14B mit 4-Bit-Gewichten kann nominal in 32 GB passen; tatsächlich konkurrieren jedoch Modellgewichte, KV-Cache, MLX-Allocator, Compilerartefakte, Zielworkload und Betriebssystem um denselben Speicher. Die Eignung ist deshalb mit einer festen Tool-Calling-/Code-Aufgabensuite zu qualifizieren, nicht aus der Parameterzahl abzuleiten.

Für Phase 1 ist ein starkes Cloud-Modell oder ein separat ausgeführtes lokales Modell methodisch sauberer. Läuft der Planer gleichzeitig auf derselben GPU, verändert er Speicherzustand, Temperatur und Queueing des zu benchmarkenden Systems. Daher gilt:

1. Vorschläge erzeugen und persistent ablegen.
2. Planermodell entladen beziehungsweise Prozess beenden.
3. definierte Abkühl-/Stabilisierungsbedingung abwarten.
4. Kandidaten in frischen Worker-Prozessen messen.

Damit bleibt die Hardwareoptimierung vollständig lokal, auch wenn die Control Plane anfänglich aus der Cloud kommt. Quellcode oder Messdaten dürfen nur bei ausdrücklich akzeptierter Datenpolitik an einen Cloud-Dienst gesendet werden.

Die Modellauswahl wird nicht durch Parameterzahl entschieden. Eine kleine Qualifikationssuite misst stattdessen Schema-/Tool-Calling-Treue, Anteil kompilierbarer Vorschläge, API-Halluzinationen, Fähigkeit zum begründeten „keine weitere Änderung“, Codeverständnis sowie Vorschlagslatenz und Peak Memory. Ein 7B-Modell, das zuverlässig im zulässigen Raum bleibt, ist geeigneter als ein 14B-Modell mit höherem Speicherverbrauch und mehr ungültigen Vorschlägen.

---

# F. Architektur

## F.1 Zielbild mit Vertrauensgrenzen

```text
                         CONTROL PLANE (nicht vertrauenswürdig)
  Workloadbeschreibung ──► Planner/LLM ──► Hypothese + deklarativer Suchraum
                                  │
                                  ▼
                         Policy-/Schema-Gate
                                  │ nur erlaubte Templates, Parameter,
                                  │ Ressourcen und Zielfunktionen
══════════════════════════════════╪══════════════════════════════════════════
                         EXECUTION PLANE (kontrolliert)
                                  ▼
      Hardware Inventory ──► Search Controller ──► Worker-Prozess
                                  │                   ├─ statische Prüfung
                                  │                   ├─ Kompilierung
                                  │                   ├─ Correctness
                                  │                   ├─ Warmup
                                  │                   ├─ Timing
                                  │                   └─ optionales Profiling
                                  ▼
                         Results + Raw Samples
                                  │
                     Promotion/Regression Gate
                         │                  │
                    akzeptiert          verworfen
                         │                  │
                         └──────► Optimization Memory
                                      │
                                      ▼
                         Production Registry
                         (nur promovierte Artefakte)
```

Das LLM hat keinen direkten Compiler-, Dateisystem-, Netzwerk- oder Ausführungszugriff. Es erzeugt zunächst ausschließlich ein typisiertes Proposal. Die Execution Plane entscheidet deterministisch, ob dieses Proposal zulässig ist. Selbst Compilerinput ist als unvertrauenswürdig zu behandeln, weil ein Compilerfehler oder Compilerbug bereits vor der Kernelausführung Schaden verursachen kann.

## F.2 Komponenten

### 1. Hardware Inventory

Erfasst bei jedem Tuning-Lauf:

- exakte Chip-/Modellkennung, GPU-Kernzahl und Unified-Memory-Größe,
- OS-Version und Build,
- Metal-Sprach-/GPU-Familien und tatsächlich abgefragte Features,
- maximale Threadgroup-Größe des kompilierten Pipeline States,
- verfügbare Counter-Sets,
- empfohlene Working-Set-Größe,
- CPU-Kerne und relevante CPU-Features,
- Versionen und Build-IDs von MLX, Python, Compiler und Projektcode.

Nominelle FLOPS, Cachegrößen oder Bandbreite aus Datenblättern werden als **statische Metadaten** markiert, nicht als gemessene Laufzeitfähigkeit.

### 2. Workload Registry

Eine Workloaddefinition enthält:

- mathematische Semantik und Referenzfunktion,
- Inputs/Outputs, Shapes, Dtypes, Strides und erlaubtes Aliasing,
- Wertebereiche und Verteilungen,
- Fehlertoleranz und besondere NaN/Inf-Regeln,
- zulässige Präzisionsmodi,
- Baselines und deren Versions-/Source-Hash,
- Optimierungsziel, Trialbudget und Ressourcenlimits.

Batch Size und Context Length sind auf Graphniveau nützlich; ein einzelner Kernel braucht zusätzlich die daraus tatsächlich resultierenden Tensorformen. Zwei Requests mit gleichem Kontext können durch Padding, KV-Cache-Layout oder Fusion unterschiedliche Kernelworkloads erzeugen.

### 3. Planner

Der Planner darf:

- vorhandene Messergebnisse und zusammengefasste Profile interpretieren,
- eine Template-Familie auswählen,
- einen begrenzten Parameterraum vorschlagen,
- einen gezielten Profiling-Lauf anfordern,
- eine neue, prüfbare Hypothese erzeugen und
- später einen Source-Patch vorschlagen, der erst durch eine Policyprüfung geht.

Er darf weder Resultate umetikettieren noch Toleranzen, Baselines oder Erfolgsmetriken nachträglich verändern. Diese gehören zum unveränderlichen Experimentmanifest.

### 4. Search Controller

Der Search Controller ist nicht sprachmodellbasiert. Er:

- dedupliziert Kandidaten anhand des kanonischen Hashes,
- wählt Grid-, Random-, Bayesian- oder evolutionäre Suche,
- setzt harte Trial-, Zeit- und Speicherbudgets,
- plant zufällige beziehungsweise randomisiert abwechselnde Baseline-/Kandidatenblöcke,
- beendet offensichtlich schlechte oder fehlerhafte Kandidaten früh und
- hält Search- und Holdout-Daten strikt getrennt.

### 5. Worker

Jeder neue Source-Kandidat wird in einem neuen, opferbaren Prozess kompiliert. Wiederholte Timing-Blöcke eines bereits validierten Artefakts dürfen in einem warmen Prozess laufen, müssen aber durch zusätzliche Frischprozess-Sessions bestätigt werden. Der Worker liefert strukturierte Ergebnisse; Logtext des Compilers wird als Daten behandelt, nicht als Anweisung an den Agenten.

### 6. Correctness Oracle

Für kleine Eingaben berechnet eine CPU-Referenz in höherer Präzision den Sollwert. Für große Eingaben kann zusätzlich die unveränderte MLX-Baseline dienen. Entscheidend ist nicht bloß `allclose`: aufgezeichnet werden mindestens Maximalfehler, relative Fehler, Quantile, NaN-/Inf-Muster und gegebenenfalls Norm-/Invarianzprüfungen.

### 7. Benchmark- und Profiler-Runner

Timing und tiefes Profiling sind getrennte Modi:

- **Timing-Modus:** minimale Instrumentierung, explizite Auswertung/Synchronisation, Rohsamples.
- **Profiler-Modus:** GPU Capture, Metal Counters oder Instruments; Ergebnisse erklären einen Bottleneck, gelten aber wegen Messperturbation nicht als finale Laufzeit.

### 8. Promotion Gate

Ein Kandidat wird nur aktiv, wenn er:

- alle sichtbaren und verborgenen Correctnessfälle besteht,
- Speicher-, Compile-, Laufzeit- und Crashlimits einhält,
- in unabhängigen Holdout-Messungen den vorab definierten Mindestgewinn erreicht,
- keine neue Speicherregression oberhalb des Limits verursacht und
- exakt zum Environment-/Workload-Fingerprint passt.

### 9. Production Registry

Die produktive Ausführung sieht keine Experimentkandidaten. Eine atomar umschaltbare Registry verweist entweder auf das letzte promovierte Artefakt oder auf die unveränderliche Framework-Baseline. Ein Fehlerzähler kann automatisch zur Baseline zurückfallen.

## F.3 Zustandsautomat des Optimierungsloops

```text
DISCOVER ─► FREEZE_MANIFEST ─► RUN_BASELINES ─► PROPOSE
                                                │
                         ┌──────────────────────┘
                         ▼
STATIC_VALIDATE ─► COMPILE ─► VISIBLE_CORRECTNESS ─► SEARCH_BENCHMARK
       │              │                 │                    │
       └─reject───────┴────reject───────┴────reject──────────┘
                                                             │
                                                             ▼
                                                   SELECT_CONTENDER
                                                             │
                                                             ▼
                       HIDDEN_CORRECTNESS ─► FRESH HOLDOUT SESSIONS
                               │                      │
                             reject        ┌──────────┴───────────┐
                                           ▼                      ▼
                                        PROMOTE                 REJECT
                                           │                      │
                                           └──── store all results┘
```

„Rollback“ bedeutet im PoC keine Quellcode-Rücksetzung. Der Produktionszeiger bleibt bis zur Promotion unverändert. Dadurch ist der normale Zustand bereits die sichere alte Version.

## F.4 Deklarativer Execution Plan

Für Phase 1 genügt ein JSON-/MessagePack-Schema, beispielsweise:

```json
{
  "schema_version": 1,
  "workload_id": "residual_rmsnorm:v1",
  "semantic_hash": "sha256:…",
  "target": {
    "chip": "Apple M1 Max",
    "gpu_cores": 32,
    "os_build": "25F84",
    "mlx_version": "…"
  },
  "shape_predicate": {"rows": 128, "hidden": 4096},
  "dtype": "float16",
  "backend": "mlx_metal_template",
  "template_id": "rmsnorm_simdgroup_v2",
  "parameters": {
    "threadgroup_size": 256,
    "vector_width": 4,
    "rows_per_group": 1,
    "math_mode": "safe"
  },
  "artifact_hash": "sha256:…",
  "fallback": "mlx_fast_rms_norm",
  "promotion_record": "runset:…"
}
```

Dieses Schema beschreibt Auswahl und Provenienz; es versucht nicht, Tensorprogramme wie MLIR oder TVM TIR darzustellen. Eine eigene Compiler-IR wird erst diskutiert, wenn mindestens zwei Backends und graphübergreifende Transformationen einen nachgewiesenen gemeinsamen Ausdruck benötigen.

## F.5 Optimization Memory

Eine lokale SQLite-Datenbank plus inhaltsadressierter Artefaktspeicher reicht zunächst. Sinnvolle logische Tabellen sind:

| Tabelle | Inhalt |
|---|---|
| `environment` | Chip, Kerne, RAM, OS-Build, Metal-/MLX-/Compiler-/Projektversionen, Capability-Snapshot |
| `workload` | Semantik-Hash, Referenz, Shape-/Stride-/Dtype-Domäne, Werteverteilung, Toleranzen, Zielmetrik |
| `candidate` | Template-/Source-Hash, Parameter, Compilerflags, Math-Modus, Planner-/Suchalgorithmus-Provenienz |
| `compile_run` | Dauer, Status, Warnungen, Artefakt-Hash, Prozessende, Ressourcenverbrauch |
| `correctness_run` | Testset-ID, Seed, Fehlerstatistiken, Invarianten, Status |
| `benchmark_run` | alle Rohsamples, Blockreihenfolge, Sync-Scope, Warmup, thermischer/Systemzustand, Speicherwerte |
| `profile_run` | Capture-/Counter-Metadaten und Artefaktlink; getrennt von finalem Timing |
| `promotion` | Baseline, Kandidat, Effektgröße, Konfidenzintervall, Holdout-IDs, Gültigkeitsbereich, Status |

Wichtige Regeln:

- **Alle** Kandidaten speichern, auch Compilerfehler, falsche und langsame Varianten. Sonst lernt ein späteres Modell aus Survivorship Bias.
- Rohsamples speichern; Median und Varianz sind abgeleitete Werte und jederzeit neu berechenbar.
- Seeds, Eingabegenerator und Testsetversion speichern, aber verborgene Testfälle nicht an den Planner geben.
- Ein Cache-Hit ist nur bei passendem Semantik-, Hardware-, Shape-, Dtype- und relevanten Softwarefingerprint gültig.
- Nach OS-, Framework-, Compiler-, Kernel-, Baseline- oder Treiberänderung wird nicht blind wiederverwendet. Der Datensatz bleibt für Lernen erhalten, das Artefakt wird jedoch quarantänisiert und neu validiert.
- Ein modellweiter Schlüssel mit „Context Length“ ersetzt nicht den operationsspezifischen Shape-/Stride-Key.

Später kann eine Konfiguration über eine Shape-Region gelten. Diese Region darf nicht geraten werden; sie wird mit Boundary- und Zufallstests empirisch validiert.

## F.6 Workload- und Bottleneck-Analyse

Die Begriffe müssen operationalisiert werden:

| Hypothese | Benötigte Evidenz | Geeignete Gegenprobe |
|---|---|---|
| **Compute-bound** | hohe Rechenauslastung, steigende Zeit mit FLOPs bei ähnlichem Datenvolumen, geringer Gewinn durch Layout-/Traffic-Reduktion | Präzision beziehungsweise Rechenmenge kontrolliert ändern; spezialisierte Matrix-/SIMD-Pfade vergleichen |
| **Bandwidth-bound** | geschätzte/aus Countern abgeleitete Bytes pro Zeit nahe der **gemessenen** nachhaltigen Bandbreite, Zeit skaliert mit Datenvolumen | Dtype/Bytes reduzieren, Fusion anwenden, Arbeitsmenge bei gleicher Rechenintensität variieren |
| **Memory-latency/cache-bound** | geringe Bandbreitenauslastung, aber stall-/cachebezogene Profilerindikatoren oder starkes Verhalten bei Zugriffsmusteränderung | zusammenhängendes gegen streuendes Layout, Working-Set- und Tile-Größe variieren |
| **Dispatch-bound** | viele sehr kurze Kernel, hoher Anteil CPU-Enqueue/Command-Buffer gegenüber GPU-Zeit | Operationen fusionieren oder mehrfach in einem Kernel bündeln |
| **Synchronization-bound** | Wartezeiten/Abhängigkeiten dominieren, CPU/GPU-Wechsel oder unnötige Barrieren sichtbar | synchronisationsarme Gesamtvariante beziehungsweise gleiche Geräteplatzierung messen |
| **End-to-end latency-bound** | Nutzerziel wird durch TTFT beziehungsweise einzelne serielle kritische Pfade begrenzt | Critical-Path-Analyse; Durchsatzoptimierung allein ist keine Lösung |

„Memory-bound“ sollte als Oberbegriff präzisiert werden: Kapazität, Bandbreite und Zugriffslatenz sind verschiedene Probleme. „Latency-bound“ ist meist eine Zielbeschreibung, keine physikalische Ursache.

Für eine Roofline-Näherung werden FLOPs und unvermeidbare Bytes aus der Semantik geschätzt, tatsächlich erreichte Bandbreite/Compute aber mit eigenen Microbenchmarks auf demselben Gerät kalibriert. Compilerfusion, Cachewiederverwendung und unnötige Transfers machen eine rein statische Bytezählung ungenau. Jede Klassifikation erhält deshalb eine Evidenzliste und Konfidenzstufe.

## F.7 CPU-, GPU- und spätere NPU-Orchestrierung

Eine Gerätewahl wird für den **gesamten Teilgraphen** bewertet:

```text
Gesamtkosten = Queue/Dispatch + Ausführung + Synchronisation
             + Layout/Konvertierung + Residency/Kohärenz
             + Auswirkung auf nachfolgende Operationen
```

Unified Memory kann eine physische Kopie vermeiden, aber nicht diese übrigen Kosten. Für kleine Operationen oder strikt serielle Steuerlogik kann die CPU schneller sein; für große, parallelisierbare Tensoroperationen meist die GPU. Die MLX-Dokumentation zeigt ausdrücklich, dass in einem Beispiel kleine Operationen auf CPU und eine große Matrixmultiplikation auf GPU schneller waren als „alles GPU“ – ein hilfreicher Hinweis, aber kein allgemeines Schedulinggesetz ([MLX Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)).

Phase 1 betrachtet CPU und GPU nur als zwei messbare Varianten einzelner Operationen oder eines sehr kurzen Teilgraphen. Später kann ein dynamisches Programm den Graphen partitionieren. Dazu sind Übergangskosten als Kanten und gemessene Operationskosten als Knoten zu modellieren; lokale Einzeloperationsminima reichen nicht.

Die Neural Engine wird nur über Core ML untersucht. Gemessen werden End-to-End-Varianten wie `.all`, `.cpuAndGPU` und `.cpuAndNeuralEngine` sowie der antizipierte `MLComputePlan`. Das System darf daraus keine Behauptung ableiten, eine konkrete Operation sei tatsächlich und ausschließlich auf der ANE gelaufen, sofern Apples Werkzeuge dies nicht belegen.

## F.8 Zielfunktionen

„Schneller“ muss vor dem Lauf definiert sein. Mögliche Profile:

- **Interactive:** primär TTFT/P50-Latenz, Guardrails für Speicher und Energie.
- **Throughput:** Tokens/s beziehungsweise Items/s bei festem Batch.
- **Memory-constrained:** Peak Memory unter hartem Limit, danach beste Laufzeit.
- **Energy-aware:** Energy-to-solution über ausreichend lange Messfenster, mit Latenzlimit.

Für mehrere Ziele sollte das System eine Pareto-Menge speichern. Ein einzelner gewichteter Score ist nur zulässig, wenn die Gewichte vor der Suche aus dem Anwendungsszenario stammen. Nachträgliches Umschalten der Zielfunktion wäre Ergebnisoptimierung.

## F.9 Einbettung in eine On-Device-AI-Architektur

Die langfristige Anwendungsebene sollte vom Optimizer entkoppelt bleiben:

```text
Anwendung
  ├─ deterministische lokale Funktionen
  ├─ lokaler Index/RAG
  ├─ lokales kleines LLM: Router, Tool Caller, einfache Agentenaufgaben
  ├─ promovierte lokale Execution Plans und Kernel
  └─ optionaler Cloud-Fallback für komplexe Planung
```

Tokenizer, Dateizugriff, RAG-Retrieval und kleine Kontrolloperationen bleiben typischerweise CPU-Aufgaben, bis Messungen etwas anderes zeigen; große Tensoroperationen sind GPU-Kandidaten. Ein Cloud-Fallback ist eine Anwendungs-/Datenschutzentscheidung, keine Hardwareoptimierung. Die produktive Inferenz verwendet nur bereits validierte lokale Artefakte – unabhängig davon, ob ein lokales oder Cloud-Modell sie vorgeschlagen hat. Damit kann das lokale LLM als Router oder Tool Caller dienen, ohne zugleich Sicherheitsrichter und GPU-Scheduler sein zu müssen.

---

# G. Technology Stack

## G.1 Phase-1-Stack

| Schicht | Empfehlung | Warum |
|---|---|---|
| Host | vorhandener M1 Max, macOS; Netzbetrieb und kontrollierter Systemzustand | reale Zielhardware |
| Entwicklungsumgebung | vollständiges Xcode mit Metal-Tools; separate Python-Umgebung | Compiler, Instruments, GPU Capture, reproduzierbare Abhängigkeiten |
| ML-Runtime | **MLX** | Apple-Silicon-fokussiert, CPU/GPU-Streams, Unified Memory, Custom Metal Kernels |
| Baseline 2 | **PyTorch MPS** | unabhängige etablierte Vergleichsimplementierung |
| Referenz | NumPy beziehungsweise kleine native CPU-Referenz in FP64/FP32, wo semantisch möglich | von GPU-Implementierungen unabhängiger Correctness Oracle |
| Control/Orchestrierung | Python | kompakter Harness, MLX-API, Such-/Statistikökosystem |
| Kernel | Metal Shading Language über `mx.fast.metal_kernel` | kein eigener Compiler, direkter PoC-Pfad |
| Low-Level-Telemetrie später | kleine Swift-/Objective-C++-/C++-Metal-Harness-Komponente | CounterSampleBuffer und Pipelineeigenschaften, die MLX-Python nicht direkt anbietet |
| Datenbank | SQLite mit Schema-Migrationen | lokal, transaktional, leicht prüfbar |
| Artefakte | inhaltsadressiertes lokales Verzeichnis | Source, Metallib/Captures, Manifeste und Logs unveränderlich zuordnen |
| Tests | `pytest` plus property-/seed-basierte Generatoren | wiederholbare sichtbare und verborgene Tests |
| Statistik | NumPy/SciPy oder äquivalent; eigener kleiner Analysecode | Bootstrap-Konfidenzintervalle, robuste Kennzahlen |
| Suche | zunächst eigene Grid/Random Search; danach Optuna oder scikit-optimize-ähnlicher BO-Tuner | geringe Phase-1-Komplexität; algorithmisch austauschbar |
| Agent | zunächst starkes Cloud-LLM **oder** qualifiziertes kleines MLX-LM-Modell | Agentenwert vom Runtime-PoC trennen |

Abhängigkeiten sind in einer Lockdatei zu fixieren. Container sind auf macOS kein Ersatz für nativen Metal-Zugriff und keine GPU-Sicherheitsgrenze. Für den ersten lokalen Worker ist Prozessisolation die praktikable Ebene; für eine robuste Distribution wäre später ein signierter, gehärteter Helper mit App-Sandbox-/Entitlement-Design zu prüfen.

## G.2 Noch nicht in Phase 1

- **Triton:** hervorragendes Vorbild und später für NVIDIA/AMD sinnvoll, aber kein offizielles Metal-Ziel des hier gewählten Pfads.
- **TVM:** besitzt Metal-Codegen und MetaSchedule; als Vergleich oder späteres Backend relevant. Es in den ersten MLX-PoC einzubauen würde aber zwei Compiler-/Runtimeprojekte gleichzeitig untersuchen.
- **MLIR:** sinnvoll als spätere Integrationsinfrastruktur; verfrüht für einen einzelnen Operator.
- **Core ML:** erst nach einem stabilen CPU/GPU-Loop; ANE-Steuerung bleibt absichtlich grob.
- **Kubernetes/Ray/verteilte Tuner:** für einen einzelnen M1 Max unnötige Betriebsfläche.

## G.3 Spätere NVIDIA-Variante

Auf NVIDIA kann dieselbe Kontrollarchitektur auf Triton/CUDA abgebildet werden. NVML liefert unter anderem rollierende GPU-/Speicherauslastung, Leistungs- und Taktinformationen; dessen Sampling liegt aber in wesentlich gröberer Größenordnung als ein einzelner kurzer Kernel ([NVML Utilization](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html), [NVML Device Samples](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceStructs.html)). Nsight Compute und CUPTI liefern deutlich detailliertere Kernelmetriken, können aber mehrere Replay-Pässe benötigen und die Ausführung perturbieren ([Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), [CUPTI](https://docs.nvidia.com/cupti/)).

Der NVIDIA-Pfad erleichtert die Diagnose, macht das Gesamtprojekt aber nicht automatisch einfach: CUDA-Version, Compute Capability, Treiber, Triton-/Compiler-Version und exakte GPU werden Teil des Cache-Keys. PTX sollte vom Compiler erzeugt werden; GPU-Assembly direkt zu schreiben bleibt außerhalb des Ziels.

## G.4 Spätere AMD- und CPU-Backends

AMD sollte als eigener Backendpfad über HIP/ROCm behandelt werden. HIP stellt Kernel-, Stream-, Event-, Speicher- und Runtime-Compilation-APIs bereit; ROCprofiler-SDK kann Dispatches, Speicherbewegungen und geräteabhängige Counter erfassen ([AMD HIP Runtime](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html), [ROCprofiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html)). Wie bei Metal und NVIDIA muss die Counterverfügbarkeit auf der konkreten GPU abgefragt werden. Eine CUDA-ähnliche API bedeutet nicht identische Performanceheuristiken.

Für CPU-Ziele sollten Compiler-Autovektorisierung und optimierte Bibliotheken die ersten Baselines sein. Erst danach sind explizite ARM-Neon- beziehungsweise x86-AVX/AMX-Varianten sinnvoll. Die offiziellen Arm C Language Extensions spezifizieren Neon-Intrinsics; MLIR besitzt unter anderem Vector-, Arm-Neon- und x86-Dialekte ([Arm ACLE](https://arm-software.github.io/acle/), [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/)). Diese Infrastruktur stärkt das Argument gegen eine eigene IR in frühen Phasen. Jede CPU-Variante braucht Laufzeit-Featurechecks beziehungsweise einen passend kompilierten Dispatchpfad; „x86“ oder „ARM“ allein ist kein ausreichender Cache-Key.

---

# H. Minimaler Proof of Concept

## H.1 Kernhypothese und bewusst enger Scope

**PoC-Hypothese H1:** Für mindestens eine festgelegte MLX-relevante Tensoroperation kann ein automatischer, hardwareabhängiger Loop aus mehreren zulässigen Metal-Konfigurationen entweder

1. eine auf unabhängigen Runs reproduzierbar schnellere korrekte Variante als alle eingefrorenen Baselines auswählen, **oder**
2. korrekt keine Custom-Variante promovieren, wenn die beste Baseline schneller ist.

**Separate Agentenhypothese H2:** Ein LLM-Planer verbessert bei gleichem Trial- und Zeitbudget die Suche gegenüber nichtsprachlichen Verfahren auf zuvor nicht verwendeten Workloads. H2 ist nicht Bestandteil des minimalen technischen Funktionsnachweises und darf bei einem erfolgreichen Autotuning-Loop negativ ausfallen.

Phase 1 ist ausdrücklich **Inference-only**. Gradienten, Training und ein rückwärtskompatibler Backward-Kernel würden den Correctness- und Suchraum erheblich vergrößern und sind erst nach einem stabilen Inferenzpfad sinnvoll.

## H.2 Operationswahl

Empfohlene Reihenfolge:

1. **Smoke Test: elementweise Fusion** – zum Prüfen von Compile, Crash Recovery, Timing und Datenbank; kein Forschungsresultat.
2. **Haupt-PoC: fused residual-add + RMSNorm** – reale LLM-Operation, überschaubarer Kernel, plausibel speicher-/reduktionsdominiert.
3. **Negative Kontrolle: Matrixmultiplikation** – zeigt, ob der Loop eine starke MLX-Baseline korrekt stehen lässt.

Die Hauptoperation sei beispielsweise:

```text
z = x + residual
r = mean(z², letzte Dimension)
y = z * rsqrt(r + epsilon) * weight
```

Semantik, Akkumulationspräzision, Broadcasting und Output-Dtype werden vor der Suche eingefroren. Eine Änderung von FP32-Akkumulation zu reduzierter Akkumulation ist keine bloße Optimierung, sondern eine neue numerische Variante mit eigener Accuracy Policy.

## H.3 Baselines

Für jede Shape-/Dtype-Kombination werden mindestens verglichen:

1. **CPU-Referenz** in höherer Präzision – primär Correctness, nicht zwingend Performance.
2. **MLX eager:** transparente Komposition aus Standardoperationen.
3. **MLX `mx.compile`:** gleiche Funktion kompiliert/fusioniert.
4. **MLX-Spezialprimitive**, soweit semantisch identisch, etwa `mx.fast.rms_norm` plus notwendige umgebende Operationen.
5. **PyTorch MPS:** semantisch äquivalente eager- und gegebenenfalls compile-/shaderfähige Variante.

Ein Custom Kernel gilt nur dann als Gewinn, wenn er gegen die **schnellste korrekte, semantisch gleiche Baseline** gewinnt. Ein Vergleich ausschließlich gegen absichtlich ungeschickten eager Python-Code wäre wertlos.

## H.4 Startsuchraum

Zunächst wird kein freier Sourcecode gesucht, sondern eine geprüfte Template-Familie:

- Reduktion über Threadgroup Memory versus SIMD-Group-Primitiven,
- Threadgroup-Größen aus einer validierten Menge, etwa 32/64/128/256, begrenzt durch den kompilierten Pipeline State,
- Vector Width 1/2/4/8, nur bei Alignment und Divisibilität; ansonsten expliziter Tailpfad,
- eine oder mehrere Zeilen pro Threadgroup,
- Anzahl partieller Reduktionen,
- safe versus relaxed math als **getrennte numerische Policy**,
- FP32-Akkumulation bei FP16/BF16-Input,
- optional getrennte Varianten für Decode-artige eine Zeile und Prefill-artige viele Zeilen.

Keine Konfiguration darf Grid-/Threadgrenzen, Speicherbedarf oder Outputgröße frei formulieren. Ein Host-Validator berechnet diese Werte aus Shape und Template.

Startshapes sollten reale LLM-Größen und Grenzfälle enthalten, zum Beispiel:

- Hidden Size: 3072, 4096 und 5120,
- Rows: 1, 16, 128 und 1024,
- FP16 zuerst, BF16 danach,
- zusätzliche nicht durch Vector Width teilbare und ungerade Hidden Sizes für Correctness und Grenzen.

Die optimale Konfiguration darf je Shape unterschiedlich sein. Ein Shape-spezifischer Gewinner muss nicht auf einem fremden Shape generalisieren; er darf dort schlicht keinen gültigen Cache-Hit erzeugen.

## H.5 Agentenrolle im ersten Versuch

Der Agent erhält:

- die mathematische Operation,
- Templatebeschreibungen und zulässige Parameter,
- Hardware-Capabilities,
- Baseline- und vergangene Kandidatenmessungen,
- kompakte Profilerhinweise, sofern angefordert.

Er liefert ein strukturiertes Proposal wie:

```json
{
  "hypothesis": "one-row decode is reduction/dispatch dominated",
  "template_id": "rmsnorm_simdgroup_v2",
  "search_space": {
    "threadgroup_size": [64, 128, 256],
    "vector_width": [2, 4],
    "rows_per_group": [1]
  },
  "requested_profile": null,
  "stop_rule": "budget_exhausted_or_dominated"
}
```

Der Controller darf das Proposal verkleinern oder ablehnen, aber nicht eigenmächtig erweitern. Erst nachdem der Template-PoC stabil ist, erhält der Agent die Möglichkeit, Source-Patches innerhalb einer kleinen, statisch geprüften MSL-Untermenge vorzuschlagen.

## H.6 Implementierungsablauf

### Schritt 0: Manifest und Vorregistrierung

- genaue Operation, Baselines, Shape-/Werteverteilungen und Dtypes festlegen,
- Correctness-Toleranzen aus der numerischen Semantik vor der Suche definieren,
- praktische Effektgrenze, Speicherlimit, Trialbudget und Abbruchregeln festschreiben,
- Git-Commit, Environment und Seeds erfassen.

### Schritt 1: Harness qualifizieren

- absichtlich richtiger, falscher, crashender, hängender und langsamer Testkernel,
- Nachweis, dass Fehlerklassen getrennt erkannt werden,
- Nachweis, dass ein Worker-Neustart den Controller nicht beschädigt,
- Timing eines bekannten Delays beziehungsweise kalibrierbaren Workloads,
- Nachweis, dass asynchrones Enqueue nicht fälschlich als Ausführungszeit gilt.

### Schritt 2: Baselines einfrieren

- Cold-Compile-/First-Call-Kosten separat erfassen,
- warme Laufzeit, Peak Memory und Correctness messen,
- schnellste gültige Baseline pro Shape/Dtype bestimmen,
- Source-/Binary-/Versionshash speichern.

### Schritt 3: Template-Search

- statisch ungültige Kandidaten verwerfen,
- in frischem Worker kompilieren,
- sichtbare Correctness-Suite ausführen,
- warm laufen lassen und Search-Timing erfassen,
- Kandidaten anhand vorab definierter Regel auswählen,
- sämtliche Resultate einschließlich Fehler speichern.

### Schritt 4: Unabhängige Promotion

- ausgewählten Kandidaten gegen die schnellste Baseline mit zuvor nicht sichtbaren Werten prüfen,
- unabhängige, randomisiert abwechselnde Timing-Sessions in frischen Prozessen ausführen,
- Speicher-, Thermal- und Systemzustand prüfen,
- nur bei vorab definierter Effektgröße und Konfidenz promoten.

### Schritt 5: Ablation

Den identischen Suchraum mindestens mit Grid/Random Search ausführen. Danach kann ein LLM den Raum beziehungsweise die Templatewahl steuern. Beide erhalten gleich viele erfolgreiche Compiler-/Benchmarkversuche und dasselbe Wall-Clock-Budget; sonst ist der Vergleich nicht interpretierbar.

## H.7 Minimaler Artefaktumfang

Ein vollständiger PoC liefert:

- reproduzierbares Environment- und Experimentmanifest,
- Hardware-/Capability-Report,
- Referenz- und Baselineimplementierungen,
- mindestens drei geprüfte Kernel-Templatevarianten,
- isolierten Worker mit Timeouts und Ressourcenlimits,
- sichtbare und verborgene Correctness-Suite,
- Benchmarkharness mit Rohdaten,
- SQLite-Optimization-Memory,
- Promotion-/Fallback-Mechanismus,
- einen automatisch generierten Bericht mit Gewinn, Konfidenz, Speicher und Compile-Amortisation,
- dokumentierte Null-/Fehlerresultate.

## H.8 Was als Ergebnis zulässig ist

Es gibt drei gleichwertig ehrlich berichtete Resultatklassen:

1. **Bestätigter Gewinn:** Ein Custom Kernel besteht Holdouts und unabhängige Sessions und schlägt die stärkste Baseline.
2. **Korrekte Baselinebeibehaltung:** Custom-Kandidaten funktionieren, sind aber nicht besser; der Loop verwirft sie zuverlässig.
3. **Infrastruktur nicht valide:** Timing, Isolation oder Correctness ist nicht ausreichend stabil. Dann ist keine Performanceaussage zulässig, bis der Harness repariert ist.

Nur Klasse 1 belegt eine konkrete Performanceoptimierung. Klassen 1 und 2 können die technische Loop-Hypothese stützen. Keine davon belegt ohne Ablation einen Nutzen des LLM.

---

# I. Entwicklungsphasen

## Phase 0 – Mess- und Sicherheitsfundament

**Ziel:** Noch keine intelligente Optimierung, sondern ein vertrauenswürdiger Versuchsstand.

- vollständiges Xcode und MLX in fixierter Umgebung installieren,
- Hardware-/Softwarefingerprint erfassen,
- isolierten Worker, strukturierte RPC und Watchdog bauen,
- lazy/asynchrone MLX-Ausführung korrekt synchronisieren,
- Rohdaten, Manifest und Artefakte persistieren,
- Fehler-, Crash-, Timeout- und Correctness-Fixtures qualifizieren,
- MLX eager/compiled/fast sowie PyTorch MPS baselinen.

**Gate:** Wenn der Harness einen absichtlich falschen oder nur asynchron eingereihten Kernel nicht erkennt, darf Phase 1 nicht beginnen.

## Phase 1 – Begrenzter Template-Autotuner

**Ziel:** H1 für residual-add + RMSNorm prüfen.

- drei oder mehr MSL-Templatefamilien,
- Grid/Random Search über validierte Parameter,
- statistische Search- und Holdout-Messung,
- Optimization Memory und Promotion/Fallback,
- Matrixmultiplikation als negative Kontrolle.

**Gate:** Reproduzierbarer Gewinn oder korrekte Baselinebeibehaltung. Ein instabiler Benchmark ist kein Nullresultat, sondern ein Infrastrukturfehler.

## Phase 2 – Zweite echte Operation und begrenzte Source-Mutation

**Ziel:** Generalität des Loops über mehr als eine handoptimierte Operation prüfen.

Geeignete Kandidaten sind RoPE, Softmax oder Quantize/Dequantize. Der Agent darf kleine AST-/Template-Patches vorschlagen, jedoch weiterhin keinen beliebigen Hostcode. Hidden Tests und Prozessisolation werden verschärft.

**Gate:** Mindestens zwei Operationsfamilien mit vollständig reproduzierbarem Versuch; nachweislich keine Toleranz-/Testfallausnutzung.

## Phase 3 – LLM-Ablation und lokale Modellpipeline

**Ziel:** H2 prüfen, nicht voraussetzen.

- LLM gegen Grid, Random, Bayesian/evolutionär unter identischem Budget,
- mehrere Seeds und zuvor nicht verwendete Workloads,
- Cloud- und lokales Modell getrennt qualifizieren,
- Plannerprozess von Benchmarkprozessen zeitlich und thermisch trennen,
- optional kleines MLX-LM-Modell profilieren und eine ausgewählte Operation austauschen.

End-to-End-Kennzahlen sind nun TTFT, Promptverarbeitung, Generationstempo und Peak Memory. Ein schneller Mikro-Kernel ohne messbaren Modellgewinn gilt nicht als Inferenzverbesserung.

## Phase 4 – CPU/GPU-Execution-Plan-Suche

**Ziel:** Nicht nur einen Kernel, sondern kurze Teilgraphen platzieren.

- Operations- und Übergangskosten messen,
- CPU-/GPU-Varianten über MLX-Streams vergleichen,
- dynamische Programmierung oder Graph Search über einen kleinen Graphen,
- Layout-, Synchronisations- und Fusionsentscheidungen berücksichtigen,
- End-to-End statt Summe isolierter Operationszeiten optimieren.

Core ML/ANE kann parallel als **separate Black-/Gray-Box-Studie** beginnen, nicht als frei programmierbares Backend.

## Phase 5 – Learned Optimization System

**Ziel:** Kandidatenzahl durch ein Kosten-/Rankingmodell reduzieren.

- Datenqualität, Censoring und Versionsverschiebungen analysieren,
- einfache Regression und GBDT-Ranking als Baselines,
- aktive Auswahl/Bayesian Optimization,
- Hardware-/Workload-Features und Unsicherheit,
- strikt zeitliche beziehungsweise hardwarebezogene Holdouts.

Ein Modell wird nur eingesetzt, wenn es im Vergleich zu Random Search bei gleichem Messbudget bessere Kandidaten findet und bei Unsicherheit auf Messung zurückfällt.

## Phase 6 – NVIDIA-Backend

**Ziel:** Architektur auf eine telemetriereichere Plattform übertragen.

- NVML für grobe Geräte-/Prozesszustände,
- Triton für Kernel und Autotuning,
- Nsight Compute/CUPTI für Diagnose,
- CUDA-/Treiber-/Compute-Capability-sicheres Caching,
- Vergleich mit TorchInductor, cuBLAS/CUTLASS und Frameworkbaselines.

Die gemeinsame Schicht bleibt zunächst das Experiment-/Execution-Plan-Schema. Metal- und CUDA-Kernel benötigen getrennte Backendimplementierungen.

## Phase 7 – Weitere Hardware und IR-Entscheidung

**Ziel:** Erst jetzt AMD ROCm, ARM NEON und x86 SIMD prüfen.

Nach zwei realen Backends wird entschieden:

- Reicht eine graphbasierte Plan-IR über bestehenden Compiler-IRs?
- Kann MLIR/TVM TIR direkt genutzt oder erweitert werden?
- Welche Semantik ist tatsächlich plattformgemeinsam und welche backend-spezifisch?

Eine neue allgemeine Compiler-IR ist nur zulässig, wenn konkrete Use Cases mit MLIR/TVM nicht sauber darstellbar sind. „Einheitlichkeit“ allein rechtfertigt die jahrelange Compilerwartung nicht.

---

# J. Benchmarkstrategie

## J.1 Messmanifest

Vor jedem Experiment wird ein unveränderliches Manifest erstellt:

- Hardware-/Softwarefingerprint,
- Netz- oder Batteriebetrieb, Display-/Hintergrundzustand,
- Operation, Shapes, Strides, Werteverteilung und Dtypes,
- Baseline- und Kandidatenhashes,
- Synchronisations- und Timingumfang,
- Warmup-/Kalibrierregel,
- Sample-/Blockzahl oder adaptives Stoppkriterium,
- Correctness-Toleranzen,
- Mindest-Effektgröße,
- Trial-, Compile-, Speicher- und Laufzeitbudget,
- primäre und sekundäre Metriken.

Änderungen erzeugen ein neues Experiment; sie überschreiben keine frühere Definition.

## J.2 MLX korrekt messen

MLX wertet Operationen lazy aus. Das Messen von Python-Funktionsaufruf bis Rückkehr kann daher nur Graphaufbau beziehungsweise Enqueue messen. Vor dem Endzeitpunkt ist das Ergebnis explizit auszuwerten und der relevante Stream zu synchronisieren ([MLX Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [MLX Streams](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html)).

Es werden zwei Zeitumfänge getrennt berichtet:

1. **Steady-state Kernel-/Blockzeit:** viele Wiederholungen eines bereits kompilierten Kandidaten, einmal pro Block synchronisiert; reduziert Hosttimer-Overhead.
2. **Synchronous end-to-end operation latency:** Dispatch, Abhängigkeiten und abschließende Synchronisation pro realistischer Operation beziehungsweise Teilgraph.

Eine Optimierung darf nicht deshalb gewinnen, weil ihre Arbeit außerhalb des gemessenen Intervalls liegt. Ebenso muss entschieden werden, ob Layout-/Contiguous-Kopien Teil des Scopes sind. `ensure_row_contiguous` in einem MLX-Custom-Kernel kann eine Kopie verursachen; diese darf nicht außerhalb des Kandidaten verborgen, aber innerhalb der Baseline mitgemessen werden ([MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)).

## J.3 Cold-, Warm- und Amortisationsmessung

Separat erfassen:

- Prozessstart und Model Load,
- erste Graph-/Kernelkompilierung,
- erster GPU-Aufruf,
- warme stabile Ausführung,
- Cache-Hit nach Prozessneustart,
- Invalidierung nach relevanter Änderung.

Der Break-even lautet:

```text
benötigte Wiederverwendungen = zusätzliche Tuning- und Compile-Kosten
                               / eingesparte Zeit pro Ausführung
```

Ein Kernel mit 10 % warmer Beschleunigung kann betrieblich nutzlos sein, wenn er vor Ablauf seines Gültigkeitsfensters nicht oft genug ausgeführt wird.

## J.4 Warmup und Stichproben

Keine fixe Zahl ist universell richtig. Der Harness verwendet daher:

1. einen Kalibrierlauf zur Wahl einer Iterationszahl, sodass ein Timingblock deutlich länger als Timer-/Sync-Overhead ist,
2. mehrere Warmup-Blöcke bis Kompilierung, Cachefüllung und Frequenzzustand nicht mehr den dominanten Trend bilden,
3. randomisiert abwechselnde Baseline-/Kandidatenblöcke,
4. ausreichend viele Blöcke für ein enges Konfidenzintervall,
5. mindestens drei frische Prozesssessions für einen Promotionkandidaten.

Als praktikabler Startwert kann ein Block etwa 50–200 ms Nutzarbeit enthalten; die genaue Dauer wird kalibriert. Sehr lange Blöcke erhöhen thermische Drift, sehr kurze Blöcke Host- und Timerrauschen.

## J.5 Statistische Auswertung

Primär berichtet werden:

- Median pro Block und pro Session,
- Median Absolute Deviation beziehungsweise IQR,
- gepaarte Kandidat/Baseline-Verhältnisse,
- Bootstrap-95-%-Konfidenzintervall über randomisierte, gepaarte Blöcke,
- Rohsamplezahl, Ausreißerregel und alle verworfenen Runs.

Ein sinnvoller anfänglicher Promotionsstandard ist:

- mindestens **5 %** niedrigere Medianzeit als die stärkste Baseline,
- 95-%-Konfidenzintervall des gepaarten Zeitverhältnisses vollständig unter 1,0,
- kein unabhängiger Sessionpunkt klar regressiv,
- gleiche Aussage in einer neuen Holdout-Messung nach Ende der Suche.

Die 5 % sind keine Naturkonstante. Sie sind eine vorab festgelegte praktische Schwelle und dürfen für einen anderen Einsatzzweck geändert werden, aber nicht nach Sichtung der Ergebnisse. Bei Mikrosekundenkerneln kann eine strengere Schwelle nötig sein. P99 wird erst berichtet, wenn die Stichprobe seine Schätzung trägt; aus einigen Dutzend Läufen wird kein belastbares P99 konstruiert.

## J.6 Winner's Curse und Holdout

Wenn Hunderte verrauschte Varianten getestet werden, ist die beste beobachtete Variante häufig zufällig zu gut. Deshalb:

- Search-Messung wählt nur einen Contender,
- Promotion verwendet neue Prozesse, neue Blockreihenfolgen und noch nicht verwendete Seeds,
- Correctness nutzt verborgene Werte-/Grenzfallverteilungen,
- eine endgültige Replikation erfolgt nach einem Neustart oder in einer zeitlich getrennten Session,
- Resultate werden nicht aus demselben Sample gleichzeitig ausgewählt und bestätigt.

Für shape-spezifisches Tuning bleibt der Performance-Holdout beim gleichen Shape; die neue Zufallsreihenfolge und Session prüfen Reproduzierbarkeit. Eine behauptete Shape-Generalisation benötigt dagegen ausdrücklich nicht zur Suche verwendete Shapes.

## J.7 Correctness-Protokoll

Die Suite umfasst je nach Operation:

- mehrere deterministische Zufallsseeds und Wertebereiche,
- Nullwerte, sehr kleine/große endliche Werte, gemischte Vorzeichen,
- NaN/Inf-Verhalten, falls von der Semantik unterstützt,
- ungerade und nicht vektorteilbare Dimensionen,
- minimale und maximale erlaubte Shapes,
- nichtkontiguierliche Eingaben, sofern der Contract sie erlaubt,
- Aliasing-/Overlap-Regeln,
- Referenz in höherer Präzision für kleine Fälle,
- Vergleich gegen die Frameworkbaseline für große Fälle,
- Invarianten, etwa Normalisierungsstatistik.

Atol/rtol werden pro Dtype und Operationssemantik vorab festgelegt. Zusätzlich muss das Fehlerprofil des Kandidaten innerhalb eines definierten, an der Referenz gemessenen Envelopes liegen; ein lockeres `allclose=True` allein genügt nicht. Fast/relaxed math wird nie stillschweigend mit safe math gleichgesetzt.

Der Planner sieht Testgeneratorvertrag und sichtbare Beispiele, aber nicht Seeds und konkrete Inhalte des Promotion-Sets. Dadurch wird hardcodierter testfallspezifischer Code erschwert.

## J.8 Speicher

Mindestens berichtet werden:

- MLX Active Memory,
- MLX Peak Memory nach definiertem Reset-/Prozesszustand,
- Prozess-RSS beziehungsweise Footprint, soweit verfügbar,
- Output-/Workspace-Größen aus dem Contract,
- OOM und Memory-Limit-Verstöße.

API-Definitionen sind zu benennen: MLX Active Memory schließt gecachte, nicht aktive Buffer aus; auch PyTorchs `current_allocated_memory` bildet Tensorallokationen, nicht notwendigerweise den gesamten Backendcache, ab ([MLX Active Memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html), [PyTorch MPS Memory](https://docs.pytorch.org/docs/2.13/generated/torch.mps.current_allocated_memory.html)). Werte unterschiedlicher APIs dürfen nicht als identisch bezeichnet werden.

## J.9 Profiler und Counter

Profile dienen der Hypothesenerzeugung:

- Metal System Trace für CPU/GPU-Queueing, Dispatch und Synchronisation,
- GPU Capture/Debugger für Encoder, Pipeline, Ressourcen und Kernelablauf,
- verfügbare Metal Counter für Zeit, Auslastung und Statistik,
- Instruments für CPU-Pfade und Systeminteraktion.

Profilerläufe werden mit eigenem `profile_run` gespeichert. Eine Variante gewinnt nicht aufgrund einer profilierten Zeit. Nach jeder profilergeleiteten Änderung folgt ein sauberer unprofilierter Benchmark.

## J.10 Energie und Thermik

Für Phase 1 ist Latenz die Primärmetrik; Energie ist explorativ:

- nur ausreichend lange, wiederholte Arbeitsblöcke messen,
- Baseline/Kandidat randomisiert alternieren,
- groben `thermalState`, Power-Source und Hintergrundlast erfassen,
- `powermetrics`-Schätzungen über das gesamte Messfenster integrieren,
- nur auf demselben Gerät und derselben Konfiguration vergleichen,
- keine einzelne Kernelenergie oder exakte Chiptemperatur behaupten.

MetricKit liefert aggregierte Felddiagnostik, keine geeignete unmittelbare Mikrobenchmark-Rückkopplung ([Apple MetricKit](https://developer.apple.com/documentation/metrickit)). Bei thermischer Drift wird die Session verworfen oder nach einer vorab definierten Stabilisierung wiederholt.

## J.11 Spätere LLM-End-to-End-Metriken

Definitionen werden festgeschrieben:

- **TTFT:** vom Eintritt des bereits tokenisierten oder nicht tokenisierten Requests – genau eine Variante wählen – bis zum ersten verfügbaren Token; Model Load separat.
- **Prompt Processing Speed:** verarbeitete Prompttokens pro Sekunde nach klarer Warm-/Cold-Regel.
- **Generation Speed:** generierte Tokens nach dem ersten Token pro Sekunde, bei festem Sampling und Seed.
- **Peak Memory:** nach benannter API und Prozessphase.
- **Qualität:** identische Logits beziehungsweise definierte numerische/Tasktoleranz; Performance darf Modellqualität nicht still verändern.

Prompts, Kontextlängen, Cachezustand, Samplingparameter und Stopregeln sind identisch. Eine Mikro-Kernelverbesserung wird nur als LLM-Verbesserung bezeichnet, wenn diese End-to-End-Metriken sie bestätigen.

---

# K. Sicherheitskonzept

## K.1 Bedrohungsmodell

Zu behandeln sind nicht nur absichtliche Angriffe, sondern wahrscheinliche Agentenfehler:

- ungültige Pointer-/Indexlogik und Out-of-bounds-Zugriffe,
- GPU-Hang, sehr lange Schleifen oder extreme Ressourcenallokation,
- Compilercrash oder Ausnutzung eines Compilerfehlers,
- numerisch falscher, aber auf sichtbare Tests optimierter Code,
- absichtliche Test-/Timer-Manipulation,
- Prompt Injection aus Compilerlogs oder externen Artefakten,
- Datenexfiltration über Netzwerk/Dateisystem,
- Vergiftung von Optimization Memory oder Promotionmetadaten,
- Race/TOCTOU zwischen validiertem Source und ausgeführtem Artefakt,
- Denial of Service für Desktop/GPU und mögliche Systeminstabilität.

## K.2 Verteidigung in Schichten

### Stufe 1: Template-only

Der erste PoC akzeptiert nur bekannte Template-IDs und Werte aus typisierten, begrenzten Mengen. Grid, Buffergrößen und Threadgroup-Speicher werden hostseitig berechnet. Das ist die wichtigste Sicherheits- und Correctnessmaßnahme.

### Stufe 2: Statische und semantische Prüfung

Bei späteren Source-Patches:

- erlaubte MSL-Untermenge beziehungsweise AST-Transformationen,
- keine Includes, Host-APIs, dynamischen Dateizugriffe oder beliebigen Compilerflags,
- berechenbare Outputgrößen und Ressourcenobergrenzen,
- Bounds-Guards und Tailpfade,
- kanonisierter Source-Hash vor und nach Kompilierung,
- Compilerwarnungen und generierte Signatur speichern.

Statische Prüfung kann GPU-Code nicht vollständig beweisen. Sie ist ein Filter, keine Sicherheitsgarantie.

### Stufe 3: Opferbarer Worker

- eigener Prozess mit minimaler RPC-Oberfläche,
- frisches temporäres Arbeitsverzeichnis und nur benötigte Read-only-Artefakte,
- Netzwerkzugriff gesperrt,
- keine Nutzergeheimnisse oder generischen Credentials im Environment,
- harte Input-/Output-, Speicher-, Compile- und Wall-Clock-Limits,
- externer Watchdog und Prozessgruppenbeendigung,
- strukturierte Crashklassifikation und anschließender frischer Prozess.

Ein Prozesslimit allein garantiert keine Begrenzung aller Metal-/Unified-Memory-Ressourcen. MLX-/Metal-Limits und beobachteter Systemfootprint müssen zusätzlich kontrolliert werden.

### Stufe 4: Numerische und speicherseitige Validierung

- kleine High-Precision-Referenzfälle zuerst,
- sichtbare, dann verborgene zufällige und adversariale Fälle,
- Canaries/Guardregions, soweit der gewählte Low-Level-Harness dies belastbar ermöglicht,
- keine Performanceausführung nach Correctnessfehler,
- mehrere Shapes und Speicherlayouts,
- Peak-Memory- und Leak-Prüfung über Wiederholungen.

### Stufe 5: Promotion und Betrieb

- Experimentartefakte sind nie automatisch produktiv,
- Promotionrecord bindet exakten Artefakt- und Environmenthash,
- atomarer Registrywechsel,
- unveränderliche bekannte Baseline,
- Laufzeitguard und automatischer Fallback,
- append-only Auditlog beziehungsweise manipulationssichtbare Hashkette,
- Revalidierung nach relevanter Software-/Hardwareänderung.

## K.3 Ehrliche Grenze des Sandboxing

Das Beenden eines Worker-Prozesses beendet nicht zwingend sofort bereits an die GPU eingereichte Arbeit. Metal-Treiber und GPU werden mit dem Host geteilt; ein normaler macOS-Prozess ist keine hart virtualisierte GPU-Sandbox. Ein besonders fehlerhafter Kernel kann die GPU zurücksetzen oder den Rechner beeinträchtigen.

Für einen lokalen Forschungs-PoC sind Templatebeschränkung, dedizierter Worker, Watchdog, Limits und häufiges Persistieren vertretbar. Für beliebigen fremden Sourcecode oder einen Mehrbenutzerdienst reicht das nicht. Dann wären ein dedizierter Testrechner, ein gehärteter/signierter Helper mit Apples unterstütztem Sandboxmodell oder eine andere echte Isolationsarchitektur zu evaluieren. Veraltete beziehungsweise private macOS-Schnittstellen dürfen nicht als Produktionssicherheitsgrenze verkauft werden.

## K.4 Was explizit verboten bleibt

- direkte GPU-ISA-/Machine-Code-Erzeugung,
- private Apple-APIs und ANE-Reverse-Engineering,
- ungeprüfte Compilerflags oder dynamische Bibliotheken,
- Netzwerk-/Shellzugriff aus dem Kernelworker,
- selbstständiges Lockern von Toleranzen oder Limits,
- Überschreiben der Baseline,
- Promotion nach einem Einzelrun,
- Fortsetzen nach wiederholter Systeminstabilität ohne menschliche Freigabe.

---

# L. Risiken und technische Sackgassen

| Risiko | Einschätzung | Konsequenz | Gegenmaßnahme / Entscheidungssignal |
|---|---|---|---|
| **Starke Baseline lässt keinen Spielraum** | hoch wahrscheinlich bei GEMM und etablierten Primitiven | Custom Kernel ist langsamer | stärkste MLX-Varianten einschließlich `mx.compile` verwenden; Fusion realer Operationsketten untersuchen; Nullresultat publizieren |
| **Scheinbeschleunigung durch Lazy Execution** | ohne guten Harness sehr wahrscheinlich | falsches Ergebnis | explizite Auswertung/Synchronisation; absichtliche Timing-Fixtures; zwei definierte Timing-Scopes |
| **Thermische/Frequenz- und Hintergrunddrift** | wahrscheinlich | Gewinner wechselt zwischen Sessions | randomisierte Paarung, kurze Blöcke, frische Sessions, thermische Zustände und Systemlast erfassen |
| **Winner's Curse durch viele Trials** | sicher ohne Holdout | zufälliger Spitzenwert wird promoted | Search-/Holdout-Trennung, neue Seeds/Prozesse, Bootstrap-Intervall |
| **Numerische Überanpassung/Reward Hacking** | relevant, besonders bei freiem LLM-Code | schnell, aber falsch außerhalb sichtbarer Tests | verborgene Verteilungen, High-Precision-Oracle, Invarianten, keine veränderbaren Toleranzen |
| **Out-of-bounds, Hang oder Treiberreset** | möglich | Daten-/Systeminstabilität | Template-only zuerst, Worker/Watchdog/Limits, dedizierter Forschungsrechner für freie Mutationen |
| **Compiler ist Teil der Angriffsfläche** | real | Crash bereits bei Compilation | Compilation ebenfalls im Worker, Source-/Flag-Allowlist, keine Secrets/Netzwerkrechte |
| **Telemetrie reicht nicht für eindeutige Diagnose** | auf Apple wahrscheinlich | Bottleneck nur indirekt bestimmbar | Capability-Abfrage, kontrollierte Gegenexperimente, Low-Level-Metal-Harness; Unsicherheit offen berichten |
| **Profiler verändert Performance** | sicher | Counterlauf nicht mit normalem Timing vergleichbar | Profiling ausschließlich zur Diagnose, anschließend sauberer Benchmark |
| **Unified Memory wird als kostenlos missverstanden** | wahrscheinlich | schlechte CPU/GPU-Platzierung | Übergangs-, Sync-, Cache- und Gesamtlatenz mitmessen |
| **Lokales Agentenmodell verunreinigt Messung** | sehr wahrscheinlich bei gleichzeitigem Lauf | weniger Speicher, Wärme, Queueing | Proposal- und Benchmarkphase trennen; Agentprozess entladen; Cloud zunächst zulässig |
| **Suchraum explodiert** | hoch ab Source-/Graphsuche | unvertretbare Messkosten | hierarchische Suche, Compilerregeln, Pruning, Trialbudget, später Kostenmodell |
| **Learned Cost Model überfitten** | hoch bei kleiner Datenbank | schlechte Vorschläge auf neuen Shapes/Versionen | Random-/zeitliche/HW-Holdouts, Unsicherheit, Fallback auf Messung; Negativdaten speichern |
| **Cache ist nach Update ungültig** | sicher über längere Zeit | Regression oder Crash | exakter Fingerprint, Quarantäne und Revalidierung statt blindem Cache-Hit |
| **Mikrogewinn verschwindet End-to-End** | häufig | kein Nutzerwert | Modell-/Teilgraphbenchmark als separates Gate; Amdahl-Effekt quantifizieren |
| **Compile-/Tuningkosten amortisieren sich nie** | relevant bei variablen Shapes | Nettoverschlechterung | Break-even und erwartete Wiederholungszahl in Promotion einbeziehen |
| **LLM ist schlechter als Random/BO** | plausibel | „AI“-Anteil ohne Nutzen | faire Ablation; LLM entfernen oder auf Hypothesen/Profilerklärung begrenzen |
| **ANE-Kontrolle wird überschätzt** | sehr wahrscheinlich ohne harte Scopegrenze | Architektur basiert auf nicht existierender API | ausschließlich öffentliche Core-ML-Abstraktion; separate Black-/Gray-Box-Studie |
| **Plattformabstraktion verliert relevante Details** | hoch | mittelmäßige Kernel überall | gemeinsame Plan-/Datenebene, backend-spezifische Kernel und Capabilities |
| **Abhängigkeit von MLX-/OS-API-Änderungen** | mittel bis hoch | Reproduktion bricht | Versionierung, Contract-Tests, kleine Adaptergrenze, gespeicherte Manifeste |
| **Energiebehauptungen sind nicht belastbar** | hoch bei kurzen Kerneln | irreführende Resultate | nur lange Same-Device-Vergleiche; Energie zunächst sekundär; Unsicherheit angeben |
| **Generierter Code/Lizenzen unklar** | relevant bei Veröffentlichung | rechtliches/Projekt-Risiko | Herkunft/Prompt/Modell protokollieren, Lizenzprüfung, keine unbekannten Codefragmente übernehmen |

Die größte fachliche Sackgasse wäre, den Agenten so lange Baseline, Toleranz, Input oder Messfenster verändern zu lassen, bis eine Verbesserung erscheint. Das wäre Benchmarkoptimierung, keine Hardwareoptimierung.

---

# M. Forschungsfragen und Rolle des LLM

## M.1 Methodenvergleich

| Methode | Stärken | Schwächen | Sinnvolle Rolle im Projekt |
|---|---|---|---|
| **Compilerheuristiken** | sehr schnell, deterministisch, kodieren harte Constraints und Domänenwissen | passen sich nicht automatisch an alle Chips/Shapes an | immer zuerst: ungültige Räume eliminieren, Defaultkonfiguration und Fallback liefern |
| **Grid Search** | vollständig und leicht prüfbar bei kleinem diskretem Raum | skaliert exponentiell, verschwendet Trials | Phase 1 für 10–100 zulässige Kombinationen und als Ground Truth kleiner Räume |
| **Random Search** | einfach, parallel, überraschend starke Baseline bei wenigen relevanten Dimensionen | nutzt frühere Messungen nicht gezielt | obligatorische Vergleichsbasis für jede „intelligente“ Suche |
| **Bayesian Optimization** | wählt bei teuren Messungen anhand Modell und Unsicherheit gezielt neue Punkte | gemischte, bedingte und stark diskontinuierliche Räume schwierig; Modellierungsaufwand | Phase 1/2 für niedrige bis mittlere Parameterdimension, etwa Tile-/Threadgroup-Werte |
| **Evolutionary/Genetic Search** | geeignet für strukturierte, diskrete und bedingte Schedules/Codevarianten; gut parallelisierbar | viele Evaluierungen; Mutationen brauchen gültige Repräsentation | später für Template-/Schedule-/AST-Suche; in TVM-artiger Kombination mit Kostenmodell |
| **Klassische Regression** | transparente, billige Baseline | bildet Interaktionen und Unstetigkeiten nur begrenzt ab; Unsicherheit schwach | erster Sanity-Check für Laufzeitprognose, nicht alleiniger Optimierer |
| **Gradient Boosting / GBDT** | stark auf tabellarischen, gemischten Hardware-/Shape-/Parameterfeatures; relativ datenarm | extrapoliert schlecht; Unsicherheit nicht automatisch kalibriert | **empfohlenes erstes learned cost/ranking model**, sobald Hunderte bis Tausende saubere Messungen vorliegen |
| **Kleines neuronales Modell** | flexible Repräsentation, gemeinsame Embeddings über große Räume möglich | braucht wesentlich mehr vielfältige Daten; schwieriger zu erklären/kalibrieren | erst bei großer Multi-Operator-/Multi-Hardware-Datenbasis gegen GBDT rechtfertigen |
| **Reinforcement Learning** | kann Folgen von Graph-/Scheduleentscheidungen optimieren | extrem sample-ineffizient auf realer Hardware, verrauschter Reward, schwer sicher zu explorieren | nicht für frühe Phasen; allenfalls später mit Simulator/Cost Model und abschließender Realmessung |
| **LLM-Planung** | kann Profilertexte, Code, Semantik und diskrete Strategiewechsel gemeinsam verarbeiten; erzeugt neue Hypothesen/Templates | nicht deterministisch, teuer, schwach bei feiner numerischer Suche, halluziniert APIs/Erfolge | äußerer Planner, Code-/Templatevorschläge und Diagnose; nicht alleiniger Tuner oder Promotionrichter |

Die empfohlene Hierarchie ist:

```text
Compilerregeln und Capabilities
       ↓ begrenzen den gültigen Raum
Random/Grid oder Bayesian/Evolutionary Search
       ↓ erzeugt messbare Kandidaten
realer Builder/Runner auf Hardware
       ↓ liefert unverfälschte Labels
GBDT-Ranking-/Kostenmodell
       ↓ reduziert spätere Kandidaten
LLM-Planer
       └─ wählt bei Bedarf neue Strategie oder Templatefamilie,
          entscheidet aber nie über Correctness oder Promotion
```

TVM MetaSchedule nutzt bereits eine verwandte Kombination aus evolutionärer Suche, realem Builder/Runner und XGBoost-Kostenmodell ([TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)). Die Beweislast liegt daher beim LLM: Es muss einen messbaren Zusatznutzen liefern, nicht umgekehrt.

## M.2 Experimentell zu beantwortende Fragen

| Forschungsfrage | Experiment | Widerlegung beziehungsweise negatives Ergebnis |
|---|---|---|
| **RQ1: Funktioniert der geschlossene Loop zuverlässig?** | absichtlich richtige/falsche/langsame/crashende Kandidaten plus echte Templates; Replay aus Datenbank | falscher Kandidat wird promoted, Baseline nicht reproduzierbar oder Zustand nach Crash korrupt |
| **RQ2: Gibt es auf M1 Max messbaren Spielraum über starke MLX-Baselines?** | RMSNorm-Fusion, RoPE/Softmax und GEMM-Kontrolle über reale Shapes | keine Variante erreicht vorregistrierten Holdout-Gewinn; das ist ein valides Nullresultat |
| **RQ3: Trifft die Bottleneck-Klassifikation nützliche Vorhersagen?** | vorab Strategie aus Profil/Counter ableiten; gezielte Gegenmaßnahmen testen | Klassifikation sagt weder Richtung noch relative Wirkung besser als einfache Heuristik voraus |
| **RQ4: Liefert ein LLM Zusatzwert?** | gleiche Budgets für LLM, Random, BO/evolutionär; mehrere Seeds und unbekannte Workloads | LLM erreicht nicht signifikant bessere Best-of-Budget-Leistung oder verbraucht mehr Zeit bei gleichem Ergebnis |
| **RQ5: Sind freie Codeänderungen wertvoller als Templates?** | Template-only gegen eingeschränkte AST-/Source-Mutation bei identischer Validierung | keine zusätzlichen Gewinne oder unverhältnismäßig mehr Fehler/Sicherheitskosten |
| **RQ6: Wie stark ist Shape-Spezialisierung?** | Konfigurationen auf Nachbar- und Holdout-Shapes testen | Gewinner generalisiert nicht; Cache muss exakt shape-spezifisch bleiben |
| **RQ7: Kann Optimization Memory Trials sparen?** | kalter Tuner gegen Cache/GBDT-Start auf zeitlich späteren Workloads | weniger Trials führen zu schlechteren Gewinnern oder Cache-Invalidierung dominiert |
| **RQ8: Wann ist CPU trotz GPU schneller?** | kleine bis große Operationen/Teilgraphen, inklusive Übergangs-/Sync-Kosten | GPU dominiert im gesamten geprüften Bereich; dann wird CPU-Platzierung für diesen Workload verworfen |
| **RQ9: Überlebt ein Mikrogewinn die Modellintegration?** | unveränderte MLX-LM-Pipeline gegen Kernelersatz bei festen Prompts | TTFT/Tokens/s/Memory zeigen keinen relevanten Gewinn; Kernel bleibt Forschungsartefakt |
| **RQ10: Ist energieorientiertes Tuning auf dieser Plattform belastbar?** | lange, randomisiert abwechselnde Runs mit powermetrics und Thermalzustand | Messunsicherheit größer als Effekt; Energie wird nicht als Optimierungsziel verwendet |
| **RQ11: Was bleibt über OS-/MLX-Versionen stabil?** | Quarantäne und Revalidierung nach Update | Gewinner oder Rangfolge kippt; gelerntes Modell benötigt Versionsfeatures oder Neustart |
| **RQ12: Welchen realen Einfluss kann Core ML/ANE zulassen?** | `.all`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, Compute Plan und End-to-End-Profil | keine belastbare Zuordnung/Verbesserung; ANE bleibt außerhalb der Runtimeoptimierung |

## M.3 Datensatz- und Modellstrategie

Vor einem Learned Optimizer werden zunächst Datenqualitätsfragen gelöst:

- Ist die gleiche Konfiguration unter gleichem Fingerprint stabil?
- Wie groß ist Messrauschen gegenüber Unterschieden zwischen Kandidaten?
- Sind fehlgeschlagene/abgebrochene Runs korrekt als zensierte oder kategoriale Ergebnisse repräsentiert?
- Welche Features stehen zur Entscheidung **vor** der Messung zur Verfügung?
- Ist der Split nach Zeit, Hardware, Operation und Shape streng genug gegen Leakage?

Das erste Modell sollte einen Kandidaten **ranken** oder seine relative Laufzeit zur Baseline schätzen. Features können enthalten:

- Hardwarefamilie, Kerne, gemessene Bandbreiten-/Compute-Kalibrierung,
- Operation/Templatefamilie,
- Shape, Strides, Dtype und geschätzte arithmetische Intensität,
- Threadgroup-/Tile-/Vector-/Stageparameter,
- statische Ressourcenabschätzungen,
- relevante Compiler-/Frameworkversion.

Der Sourcecode selbst wird in der ersten Learned-Phase nicht in ein neuronales Modell eingebettet. Ein strukturierter, erklärbarer Featurevektor ermöglicht bessere Fehleranalyse. Vorhersagen steuern nur die Reihenfolge der echten Messungen; sie ersetzen die Promotionmessung nie.

---

# N. Erfolgs- und Abbruchkriterien

## N.1 Getrennte Erfolgsstufen

### Erfolg 0 – Vertrauenswürdiger Benchmarkharness

- Lazy Execution und Synchronisation korrekt behandelt,
- absichtliche Fehler, Timeouts und Abstürze werden erkannt,
- Rohdaten und Environment erlauben Replay,
- die Unsicherheit ist klein genug, um den vorregistrierten praktischen Effekt zu erkennen.

### Erfolg 1 – Funktionierender Optimization Loop

- mindestens drei zulässige Kernelvarianten automatisch kompiliert, validiert und verglichen,
- falsche und langsamere Varianten verworfen,
- Gewinner oder Frameworkbaseline deterministisch ausgewählt,
- Entscheidung nach Neustart reproduzierbar,
- Optimization Memory und Fallback funktionieren.

Dies ist der **minimale PoC-Erfolg**, auch wenn die Frameworkbaseline gewinnt.

### Erfolg 2 – Reale Hardwareoptimierung

- mindestens ein Custom-/Execution-Plan-Kandidat ist gegenüber der stärksten semantisch gleichen Baseline im Median mindestens um die vorregistrierte Schwelle besser,
- 95-%-Intervall und getrennte Sessions bestätigen die Richtung,
- verborgene Correctnessfälle bestehen,
- Memory-/Accuracy-Guardrails bestehen,
- Compile-/Tuningkosten amortisieren sich im deklarierten Wiederverwendungsszenario.

### Erfolg 3 – Nachgewiesener LLM-Beitrag

- mehrere, vorab getrennte Operations-/Workloadfälle,
- identische Trial- und Zeitbudgets,
- wiederholte Seeds,
- LLM-Planung schlägt Random und mindestens ein starkes nichtsprachliches Verfahren bei Best-of-Budget oder benötigten Hardwaremessungen,
- Gewinn bleibt auf unbekannten Workloads bestehen.

Ein einzelner spektakulärer Kernel erfüllt Erfolg 3 nicht.

### Erfolg 4 – Research Prototype

- mehrere Operationsfamilien und kurze Graphen,
- versionssicheres Optimization Memory,
- gelerntes Rankingmodell mit Unsicherheits-/Fallbackverhalten,
- CPU/GPU-Planung mit End-to-End-Gewinn,
- reproduzierbarer öffentlicher Benchmarkkorpus und dokumentiertes Sicherheitsmodell.

## N.2 Stop-, Reparatur- und Pivotkriterien

**Sofort reparieren, keine Ergebnisse berichten**, wenn:

- bekannte falsche Kernel die Promotion bestehen,
- Timing ohne Arbeitssynchronisation möglich ist,
- Baselinewiederholungen die praktische Effektschwelle nicht auflösen können,
- Datenbank/Artefakte nach einem Workercrash inkonsistent sind oder
- Source, Binary und Messrecord nicht eindeutig gebunden sind.

**Freie Kernelgenerierung stoppen**, wenn:

- wiederholt GPU-/Systemresets oder nicht begrenzbare Hänger auftreten,
- Compile-/Workerisolation die Maschine oder Nutzerdaten nicht ausreichend schützt,
- Correctness-Overfitting trotz Hidden Tests wiederholt auftritt.

Dann bleibt Template-/Parameterautotuning der sichere Projektkern.

**Von Custom Kernels zu Placement/Fusion/Dataset pivotieren**, wenn nach einem vorregistrierten, fachlich sinnvollen Budget über mindestens zwei echte Operationsfamilien keine Variante die stärksten MLX-`compile`-/Fast-Baselines reproduzierbar schlägt. Der Umfang des Budgets wird vorab festgelegt – beispielsweise mehrere Templatefamilien und einige Dutzend bis niedrige Hunderte gültige Konfigurationen pro Operation – und nicht nachträglich verlängert, bis zufällig ein Gewinn entsteht.

**Das LLM aus dem Optimierungspfad entfernen**, wenn es über mehrere Workloads und Seeds nicht besser als Random plus BO/evolutionäre Suche ist oder sein Rechen-/Latenzaufwand den Suchgewinn übersteigt. Das ist kein Scheitern der Runtime, sondern eine nützliche Architekturentscheidung.

**Gesamtprojekt beenden oder wesentlich neu formulieren**, wenn:

- weder Kernel-/Fusions- noch CPU/GPU-Placementversuche einen messbaren End-to-End-Nutzen zeigen,
- Savings die Tuning-/Wartungs-/Invalidierungskosten in realistischen Nutzungshorizonten nicht amortisieren,
- die benötigte Telemetrie nur über private/reverse-engineerte Schnittstellen erreichbar wäre oder
- der einzige scheinbare Nutzen aus schwachen Baselines, gelockerten Toleranzen oder nicht reproduzierbaren Einzelruns stammt.

---

# O. Aufwand

## O.1 Aufwandsmaßstab

Kalenderzeit wäre ohne Kenntnis von Erfahrung, verfügbarer Fokuszeit und Qualitätsziel spekulativ. Deshalb werden relative Entwicklungsgrößen verwendet:

- **S:** isolierte, bekannte Komponente mit wenigen Schnittstellen,
- **M:** mehrere integrierte Komponenten und systematische Tests,
- **L:** multidisziplinäres Subsystem mit erheblichem Experiment-/Debugginganteil,
- **XL:** längerfristiges Forschungs- oder Teamprogramm mit dauernder Wartung.

KI-Coding-Agents können Implementierung, Testgenerierung und Dokumentationsrecherche beschleunigen. Sie reduzieren nicht die benötigten Hardwaremessungen, statistische Wiederholungen oder das Risiko subtiler numerischer/GPU-Fehler.

## O.2 Phasenbewertung

| Phase | Schwierigkeit | Benötigtes Wissen | Entwicklungsaufwand | Hardwarebedarf |
|---|---|---|---|---|
| **0: Harness** | mittel bis hoch | Python, MLX-Lazy/Streams, Prozesse/RPC, robuste Statistik, macOS/Metal-Tools | **M** | vorhandener M1 Max; vollständiges Xcode |
| **1: Template-Autotuner** | hoch, aber für Einzelperson realistisch | MSL, Threadgroups/SIMD-Groups, Reduktionen, Numerik, Benchmarkdesign | **L** | vorhandener M1 Max genügt |
| **2: zweite Operation/Source-Mutation** | hoch bis sehr hoch | GPU-Algorithmen, statische Validierung, Fuzz-/Property-Tests, Compilerfehler | **L–XL** | M1 Max; idealerweise separater Testrechner bei freien Mutationen |
| **3: LLM-Ablation/MLX-LM** | hoch | Experimentdesign, Agentenprotokolle, lokale Inferenz, Modellqualifikation | **L** | M1 Max; Cloud optional; zusätzlicher Speicherrechner hilfreich, nicht nötig |
| **4: CPU/GPU-Graphplanung** | sehr hoch | Graphalgorithmen, Scheduling, Cache/Kohärenz, MLX-Streams, End-to-End-Profiling | **XL** | M1 Max; weitere Apple-Silicon-Varianten für Generalität später nötig |
| **Core-ML-/ANE-Studie** | mittel bis hoch, aber begrenzte Kontrolle | Core ML Conversion, Compute Units/Plan, Instruments | **M–L** | M1 Max; neuere SoCs nötig, wenn Aussagen darüber gemacht werden sollen |
| **5: Learned Optimizer** | hoch | Datenqualität, Ranking/Regression, BO/Active Learning, Drift/OOD | **L–XL** | M1 Max reicht für GBDT; vor allem viele saubere Messdaten und Storage nötig |
| **6: NVIDIA-Backend** | sehr hoch | CUDA, Triton, NVML, Nsight/CUPTI, NVIDIA-Baselines | **XL** | separate unterstützte NVIDIA-GPU; für Generalität mehrere Generationen |
| **7: Multi-Hardware/IR** | extrem | Compilerbau, MLIR/TVM, mehrere ISA-/Runtimeökosysteme, Release Engineering | **XL/Teamprogramm** | Flotte aus Apple/NVIDIA/AMD/x86/ARM, CI und langfristige Wartung |
| **Production Runtime** | extrem | alle obigen plus Security Engineering, Observability, Compatibility, Incident Response | **mehrjähriges Teamprogramm, nicht Einzel-PoC** | breite Testmatrix und dedizierte Infrastruktur |

## O.3 Wissenslücken, die der PoC schließen muss

Ein Entwickler muss nicht bereits GPU-Compilerexperte sein, sollte aber gezielt lernen:

1. Metal Shading Language, Grid/Threadgroup/SIMD-Group und Speicherhierarchie,
2. parallele Reduktionen, Alignment, Vektorisierung und numerische Stabilität,
3. MLX-Lazy-Evaluation, Streams, Compilation und Memory APIs,
4. reproduzierbares Microbenchmarking und robuste Statistik,
5. Prozessisolation, Watchdogs und sichere Artefaktprovenienz,
6. Profilerinterpretation ohne Counterwerte zu überdeuten.

Die Reihenfolge ist wichtig: Erst einen existierenden Beispielkernel verstehen und manuell variieren; dann den Tuner automatisieren; erst danach ein LLM Sourcecode-Änderungen vorschlagen lassen.

---

# Persönliche Machbarkeit

## PoC

**Ja.** Ein technisch interessierter Einzelentwickler kann mit modernen Coding-Agents einen echten Demonstrator auf diesem M1 Max bauen, auch ohne anfängliche GPU-Compiler-Spezialisierung. Voraussetzung ist, den Scope auf eine Operation, wenige geprüfte Templates und saubere Mess-/Correctness-Infrastruktur zu begrenzen. Der schwierigste Teil ist nicht das Aufrufen eines Metal-Kernels, sondern ein Ergebnis zu erzeugen, dem man glauben darf.

Coding-Agents können:

- MLX-/Metal-Beispiele adaptieren,
- Worker, Datenbank und Tests implementieren,
- Compilerfehler erklären,
- Profilerhypothesen und Varianten vorschlagen.

Sie können nicht:

- reale Hardwaremessung ersetzen,
- unbekannte Metal-/ANE-Fähigkeiten erfinden,
- numerische Korrektheit außerhalb der Tests garantieren,
- einen Treiberhang sicher „wegargumentieren“ oder
- aus einem verrauschten Einzelwert einen Performancebeweis machen.

## Research Prototype

**Ja, aber nur eng und mit erheblicher Lernkurve.** Ein Einzelentwickler kann einen guten MLX/Metal-Research-Prototype mit zwei bis einigen Operationsfamilien, Hybridtuner, Datenbank und fairer LLM-Ablation bauen. Freie Kernelmutation, Graphplanung, robuste Sicherheit und veröffentlichungsfähige Evaluation machen daraus ein langfristiges Forschungsprojekt. Externe Reviews durch GPU-/Numerikexperten wären sehr wertvoll.

## Production Runtime

**Nein, nicht in der beschriebenen Breite als Einzelperson.** Eine robuste Runtime für viele Modelle, Shapes, OS-Versionen und Hardwaregenerationen benötigt eine große Correctnessmatrix, Sicherheitsengineering, Compiler-/Treiberkompatibilität, Telemetrie, Rollout/Rollback und kontinuierliche Wartung. Ein schmaler produktiver Tuner für eine kontrollierte Anwendung wäre denkbar; eine allgemeine Runtime nicht.

## Konkurrenz zu CUDA, XLA oder TorchInductor

**Nein.** Diese Systeme sind Plattformen mit großen Teams, jahrelanger Compilerarbeit, Hardwarezugang und enormer Testabdeckung. Sinnvoll wäre Konkurrenz nicht auf ihrer gesamten Ebene, sondern ein komplementäres, enges Open-Source-Projekt: Apple-Silicon-Benchmarkkorpus, MLX/Metal-Autotuner, sichere Agentenevaluation oder Optimization-Memory-Format. Ein erfolgreicher Prototyp könnte später in bestehende Compiler-/Runtimeprojekte integriert werden, statt sie zu ersetzen.

---

# Antworten auf die drei finalen Fragen

## Frage 1: Kann heute auf einem Apple M1 Max mit 32 GB ein echter KI-gesteuerter Hardware-Optimization-Loop gebaut werden?

**Ja – als kontrollierter Offline-/Install-Time-PoC.** MLX kann eigene Metal-Kernel JIT-kompilieren und auf CPU/GPU ausführen; Metal/Xcode liefern Diagnosemöglichkeiten; ein Controller kann Correctness, Warmup, wiederholte Messungen, Promotion, Cache und Rollback automatisieren. „KI-gesteuert“ sollte zunächst bedeuten, dass ein LLM Hypothesen oder den Suchraum vorschlägt. Die eigentliche Suche und Entscheidung bleiben deterministisch und messungsbasiert. Freie ANE-Kernel, beliebige sichere Selbstmodifikation und eine allgemeine Multi-Hardware-Runtime sind damit nicht gegeben.

## Frage 2: Welche kleinste Implementierung beweist oder widerlegt die Kernhypothese?

Die kleinste überzeugende Implementierung ist:

1. **eine** reale Operation: fused residual-add + RMSNorm,
2. **vier starke Baselines:** MLX eager, MLX `compile`, passende MLX-Fast-Primitive und PyTorch MPS,
3. **drei geprüfte Metal-Templatefamilien** mit begrenzter Threadgroup-/Vector-/Reduktionssuche,
4. ein isolierter Worker mit Compile-/Runtime-Timeouts,
5. CPU-/MLX-Referenz und sichtbare plus verborgene Correctnessfälle,
6. explizite Warmup-/Synchronisationslogik, randomisierte Paarmessungen und unabhängiger Holdout,
7. SQLite-Optimization-Memory und atomarer Baselinefallback,
8. zunächst Grid/Random Search; danach optional ein LLM-Planer mit exakt gleichem Budget.

Der Loop ist technisch bestätigt, wenn er zuverlässig den tatsächlichen Gewinner auswählt oder alle Custom-Varianten zugunsten der Baseline verwirft. Eine **LLM-spezifische** Hypothese ist erst bestätigt, wenn der LLM-Planer auf mehreren unbekannten Workloads klassische Verfahren unter gleichem Budget schlägt.

## Frage 3: Was ist nach einem erfolgreichen PoC der sinnvollste nächste Schritt?

Nicht sofort ein vollständiges LLM oder eine eigene IR bauen. Der sinnvollste Schritt ist ein **offener, reproduzierbarer MLX/Metal-Operator-Benchmark und Tuning-Korpus**:

- eine zweite und dritte reale Operation ergänzen,
- Test-/Messprotokoll und Optimization-Memory-Schema stabilisieren,
- alle Gewinner, Verlierer und Fehler veröffentlichen,
- LLM, Random, Bayesian und evolutionäre Suche fair ablatieren,
- die beste Variante in einen kleinen MLX-LM-End-to-End-Test integrieren,
- erst danach CPU/GPU-Teilgraphplatzierung und ein GBDT-Kostenmodell hinzufügen.

Damit entsteht ein überprüfbares Open-Source-Forschungsprojekt mit klarer Nische. Liefert das LLM keinen Zusatznutzen, wird es aus dem Tuningkern entfernt; das Projekt bleibt als hardwareabhängiger Autotuner und Datensatz sinnvoll. Liefert selbst der Tuner keinen End-to-End-Gewinn, sollte das Ergebnis veröffentlicht und der Schwerpunkt auf Messinfrastruktur, Placement oder Compilerintegration verlagert werden.

---

# Ausgewählte Primärquellen

Die Quellen wurden für dieses Konzept am 15. August 2026 geprüft. Versionsabhängige APIs und Capability-Angaben sind beim tatsächlichen Implementierungsstart erneut gegen die installierten Versionen zu verifizieren.

## Apple, Metal und Core ML

- [Apple M1 Max – offizielle Ankündigung und Spezifikationen](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)
- [Metal Feature Set Tables](https://developer.apple.com/metal/capabilities/)
- [Metal GPU Counters and Counter Sample Buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
- [Metal Developer Workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows)
- [Metal `recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- [Core ML `MLComputeUnits`](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [Core ML `MLComputePlan`](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)

## MLX und PyTorch

- [MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [MLX Compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX Devices and Streams](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/2.13/notes/mps.html)
- [PyTorch MPS API](https://docs.pytorch.org/docs/2.13/mps.html)

## Compiler und Autotuning

- [Triton `autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [TVM MetaSchedule Tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng)
- [XLA Persisted Autotuning](https://openxla.org/xla/persisted_autotuning)
- [MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)

## Agentische Kerneloptimierung

- [KernelBench](https://arxiv.org/abs/2502.10517)
- [KernelBench-Verified](https://arxiv.org/abs/2607.16241)
- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

## NVIDIA für eine spätere Variante

- [NVML API](https://docs.nvidia.com/deploy/nvml-api/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [CUPTI Documentation](https://docs.nvidia.com/cupti/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

## AMD und CPU-Vektorisierung

- [AMD HIP Runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html)
- [AMD ROCprofiler-SDK Quick Reference](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html)
- [Arm C Language Extensions / Neon](https://arm-software.github.io/acle/)
- [MLIR Dialectübersicht einschließlich Vector, Arm Neon und x86](https://mlir.llvm.org/docs/Dialects/)

---

## Gesamturteil

Die Idee ist als **enger, hybrider und messungszentrierter Forschungs-PoC sinnvoll**. Sie ist als Behauptung eines selbstverständigen autonomen KI-Compilers nicht tragfähig. Der größte technische Wert entsteht wahrscheinlich nicht durch ein LLM, das fortlaufend Low-Level-Code schreibt, sondern durch die Kombination aus sauberem Hardwarefeedback, gut begrenzten Suchräumen, existierenden Compilern, verlässlicher Validierung, Optimization Memory und einem späteren Kostenmodell. Ob ein LLM darüber hinaus wirklich hilft, ist eine offene Forschungsfrage – und sollte die erste Hypothese sein, die das Projekt bereit ist zu verwerfen.

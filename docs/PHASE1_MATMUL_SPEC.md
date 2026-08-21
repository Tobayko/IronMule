# Phase 1A — H0-Messsystem-Preflight (Matmul)

**Status:** H0-Messsystem-Preflight; Offline-Harness, SQLite v1, fester Worker Option A und
read-only-Loopback-Dashboard sind implementiert. Bisherige Live-Läufe waren ungültig oder
fail-closed; es gibt noch keine gültige abgeschlossene H0-Performanceentscheidung.

**Zweck:** Diese Datei friert den kleinsten reproduzierbaren MLX-Harness-Preflight und den
festen Offline-Vertrag ein. Die produktive H0-Historie verwendet SQLite v1 unter
`.friday-data/h0.sqlite3`; das Dashboard liest sie ausschließlich read-only auf
`127.0.0.1`. Sie definiert Mess- und Entscheidungsregeln, nicht deren Hardware-Ergebnisse.

**Forschungsgrenze:** H0 prüft Messbarkeit, Correctness-, Fehler- und Fallbackverhalten für
eine feste Operation. H0 ist kein Nachweis von Self-Optimization, Hardware-Generalisation,
LLM-Nutzen, Transformer-/End-to-End-Performance oder Cross-Device-Übertragbarkeit.

**Phasenabgrenzung:** Alle Akzeptanzkriterien und Geltungsbehauptungen dieser Datei gelten
ausschließlich für Phase 1A. Der vollständige Phase-1-DoD aus `IMPLEMENTIERUNGSPLAN.md` ist
damit nicht erfüllt: Danach ist eine separate Phase 1B mit einem begrenzten Custom-MLX-Metal-
Kandidaten und einem isolierten Worker erforderlich. Phase 1B ist nicht Bestandteil dieser
Vorregistrierung, nicht implementiert und benötigt eine separate Sicherheits- und
Architekturfreigabe.

## 1. Geltungsbereich und Nicht-Ziele

1. Der Versuch beantwortet nur, ob die feste Operation auf der vorhandenen Installation
   reproduzierbar gemessen und gegen ihre Baseline entschieden werden kann.
2. Er ist kein Beleg für Transformer-, LLM-, ANE-, Multi-Hardware- oder End-to-End-
   Performance.
3. Es wird kein eigener Metal-Kernel, kein generierter Kernelcode, keine eigene IR und
   kein Compiler eingeführt.
4. Die einzige Performance-Workload ist die kanonische `2048²`-FP16-Matmul aus Abschnitt 2.
   `mx.compile` ist eine sichere Framework-Vergleichsvariante; es ist weder ein echter
   A/A-Nullpfad noch ein Custom-Kernel-Kandidat oder Phase-1B-Nachweis.
5. Eine Messung darf erst starten, wenn alle hier als verpflichtend markierten Werte
   entweder erhoben oder mit `null` und einem Grund protokolliert werden.

## 2. Feste Workload

| Feld | Festgelegter Wert |
|---|---|
| Operation | `Y = mx.matmul(A, B)` |
| A-Shape | `(2048, 2048)` |
| B-Shape | `(2048, 2048)` |
| Y-Shape | `(2048, 2048)` |
| Dtype A/B/Y | `float16` |
| Layout | C-contiguous, row-major; keine implizite Transposition |
| Ausführung | MLX, derselbe Stream je gepaartem Block |
| Operationen pro Auswertung | genau eine Matmul |
| Zufallsverteilung | uniform `[-1, 1)` auf Host-`float32` |
| Generator | PCG64 |
| Seed | vollständig vorab im Manifest festgelegte Prozess-/Fixture-/Reihenfolge-Seeds; siehe Abschnitt 7 |

Die Hostwerte werden zunächst als `float32` erzeugt und danach deterministisch zu
`float16` konvertiert. Genau diese konvertierten Werte sind die Eingabe der MLX-
Messung und der Oracle-Berechnung. Zufällige Erzeugung innerhalb von MLX ist nicht
zulässig, weil sie Generator-, Stream- und Versionszustand in den Versuch einführen
würde.

Für die kanonische Fixture werden SHA-256-Digests von A, B und der vollständigen
Fixture-Metadaten gespeichert. Der Digest wird über die kanonische Byte-Repräsentation
der tatsächlich verwendeten FP16-Werte gebildet, mit expliziter Byteorder und ohne
Pickle- oder plattformabhängige Serialisierung.

## 3. Rechen- und Speicherinvarianten

Die FLOP-Zahl ist vorab festgelegt:

`2 × 2048 × 2048 × 2048 = 17,179,869,184 FLOP = 17.179869184 GFLOP`.

Die Nutzdaten bestehen aus A, B und Y, jeweils `2048² × 2` Bytes. Damit sind exakt
`3 × 2048² × 2 = 25,165,824 Bytes` A+B+Y-Nutzdaten zu protokollieren.

Diese Zahl umfasst weder MLX-Allocator-Overhead noch Cache, temporäre Buffer,
Python-Objekte, Prozess-RSS oder Compilerartefakte. Nutzdaten und jede beobachtete
Speichermetrik bleiben getrennt.

## 4. Referenz und Correctness

Der Oracle berechnet auf dem Host mit FP64 aus den exakten, bereits konvertierten
FP16-Werten. Der Oracle darf nicht die ursprünglichen FP32-Zufallswerte verwenden.
Die Referenz ist damit unabhängig von MLX-Auswertung, GPU-Rundung und MLX-Backend.

Die Performanceauswertung verwendet ausschließlich die kanonische `2048²`-Fixture aus
Abschnitt 2. Correctness-Fälle werden separat und niemals in eine Performanceaggregation
aufgenommen.

Für die kanonische Performance-Fixture muss jede Variante vor einer Laufzeitentscheidung
folgende Checks bestehen:

- Shape exakt `(2048, 2048)` und Dtype exakt `float16`;
- keine NaN- oder Inf-Werte in Eingaben, Ergebnis oder Oracle;
- absoluter und relativer Fehler je Element sowie `q50`, `q95`, `q99` und `max`;
- normierter Fehler, definiert als `||Y - Oracle||₂ / max(||Oracle||₂, tiny)`;
- identische Fixture-Digests und identische Ausgabedimensionen.

Quantile und Maximum werden auf dem vollständigen Ergebnis berechnet, ohne nachträgliche
Ausreißertrimmung. Ein Correctness-Fehler disqualifiziert den gesamten Block und wird
mit Ursache, Variante, Seed und Prozess-ID gespeichert.

Ein analytischer Fehler-Envelope darf nur verwendet werden, wenn der verwendete MLX-
Akkumulationscontract für diese Version und Dtype durch öffentliche Dokumentation oder
einen reproduzierbaren Contract-Test belegt ist. Unabhängige FP64-Hard-Caps und die
eingefrorene Semantik bleiben das Zulässigkeits-Gate. Ohne solche unabhängigen Caps ist
jede Kandidatenmessung ungültig; es gibt dann nur einen Baseline-Referenzlauf.
Baseline-abgeleitete Faktoren dürfen zusätzlich als Regressionsdiagnose gespeichert
werden, definieren aber nicht die zulässige Semantik.

### 4.1 Baseline-only Correctness-Kalibrierung

Die Correctness-Kalibrierung wird ausschließlich mit der Eager-Baseline durchgeführt.
Der Sicherheitsfaktor ist fest `2.0`; die harten Caps sind `abs_max <= 1.0`,
`rel_q99 <= 0.05` (nur für Elemente mit `abs(Oracle) >= 1.0`) und
`normalized_l2 <= 0.01`. Für absolute und relative Fehler werden jeweils `q50`,
`q95`, `q99` und `max` gespeichert; zusätzlich werden `normalized_l2` und
`scaled_normalized_inf = ||Y - Oracle||∞ / max(||Oracle||∞, tiny)` gespeichert.

Überschreitet ein Baseline-Lauf eines der harten Caps, ist der Harness ungültig. Die
Baseline-Kalibrierung bleibt diagnostisch und darf nach Beginn der Kandidatenmessung nicht
als Zulässigkeitsdefinition geöffnet, verschoben oder aus Kandidatendaten abgeleitet
werden. Metriken ohne sinnvollen unabhängigen harten Cap werden als diagnostisch markiert
und sind kein Gate.

### 4.2 Nicht getimte Correctness-Matrix

Die folgende Matrix ist ein separater, nicht getimter Correctness-only-Vertrag. Sie prüft
dieselbe `mx.matmul(A, B)`-Semantik, verwendet keine nichtkontiguierlichen Inputs und keine
anderen Dtypes, weil der Contract C-contiguous FP16 festlegt. Alle FP16- und FP64-Oracles
werden aus den exakt erzeugten und anschließend deterministisch konvertierten FP16-Werten
gebildet; die ursprünglichen FP32-Zufallswerte sind kein Oracle-Eingang.

| Gruppe | A-Shape | B-Shape | Seed | Fixture |
|---|---:|---:|---:|---|
| visible | `64×64` | `64×64` | `0xC0DE0001` | uniform `[-1,1)` |
| visible | `17×31` | `31×13` | `0xC0DE0002` | uniform `[-1,1)` |
| visible | `33×65` | `65×7` | `0xC0DE0003` | RHS exakt null |
| visible | `31×47` | `47×19` | `0xC0DE0004` | uniform `[-2^-10,2^-10)` |
| visible | `31×47` | `47×19` | `0xC0DE0005` | uniform `[-4,4)` |
| holdout | `23×37` | `37×29` | `0xC0DE1001` | uniform `[-1,1)` |
| holdout | `65×33` | `33×9` | `0xC0DE1002` | uniform `[-4,4)` |

Jeder Case speichert Shape, Dtype, C-contiguous-Layout, Seed, Fixture-Digest, NaN/Inf-
Status sowie elementweise absolute und relative Fehler, `q50`, `q95`, `q99`, `max` und
den normierten L2-Fehler gegen das FP64-Oracle. Das Zero-RHS-Ergebnis muss exakt null sein.
Zusätzlich wird die Sign-Invariante `(-A)@B = -(A@B)` für `64×64` innerhalb des
eingefrorenen Envelopes geprüft. Die Matrix ist ein Correctness-Gate und liefert keine
Zeit-, Throughput- oder Performanceaggregation.

## 5. Vergleichsvarianten

### 5.1 Eager-Baseline

Die Baseline ist die direkte Funktion `lambda a, b: mx.matmul(a, b)` ohne Compile-
Wrapper. Eingaben, Stream, Synchronisation und Output-Auswertung sind ansonsten gleich.

### 5.2 Compile-Vergleichsvariante

Die sichere, bekannte Framework-Vergleichsvariante ist exakt:

```python
mx.compile(lambda a, b: mx.matmul(a, b), shapeless=False)
```

Die Signatur, Shapes und Dtype werden nicht verändert. Kein Customcode, keine Fast-
Primitive und keine fusionierte Nebenoperation dürfen in Phase 1A hinzukommen. Diese
Variante ist nicht der in `IMPLEMENTIERUNGSPLAN.md` geforderte Custom-MLX-Metal-Kandidat.

### 5.3 H0-Kontrollarme

#### 5.3.1 A/A-GPU-Nullkontrolle

Der echte GPU-Nullkontrollpfad verwendet exakt drei Charakterisierungs- und drei
Bestätigungsprozesse mit jeweils 30 gepaarten Blöcken. In jedem Block werden zwei separat
erzeugte, aber semantisch identische Eager-Callables mit identischer Fixture, identischem
Stream und identischer Synchronisation gemessen. Die A/A-Seeds sind fest:

- Charakterisierung: Fixture `0xAA1A2026+i`, Reihenfolge `0xAA0D2026+i`;
- Bestätigung: Fixture `0xAA1A2126+i`, Reihenfolge `0xAA0D2126+i`, jeweils `i=0..2`.

Die Richtung ist für alle Ratios `R = T_candidate / T_baseline`, kleiner ist besser. Für
jedes Set (Charakterisierung und Bestätigung) werden die Session- und Set-Schätzer exakt
so gebildet:

```text
R_s  = exp(median_b(log(t_B / t_A)))
R_AA = exp(median_s(log(R_s)))
```

Dabei sind `A` und `B` die zwei separat erzeugten identischen Eager-Callables, `b` die 30
gepaarten Blöcke und `s` die drei Sessions des jeweiligen Sets. Das hierarchische
10.000er-Perzentil-Bootstrap resampelt Sessions und innerhalb jeder gezogenen Session
Blöcke, rekonstruiert bei jedem Resample exakt `R_AA` und verwendet feste Seeds
`0xAA052026` (Charakterisierung) und `0xAA052126` (Bestätigung). Das ist ein
Engineering-Äquivalenzgate, kein wissenschaftlicher Äquivalenznachweis.

Diese beiden Bootstrap-Seeds gehören ausschließlich zum A/A-Gate und sind im
Manifest-v1-Contract als geschlossene, validierte Felder umgesetzt; die `aa_gpu`-Manifeste
binden die `AA05`-Seeds an Set und Index. Der rückwärtsinkompatible Contractfix ist damit
abgeschlossen und `aggregation_contract_ready=true`. Die echte A/A-GPU-Ausführung bleibt
trotzdem bis zum separat angekündigten Go/No-Go durch `live_execution_authorized=false`
gesperrt. Die Statistikformeln und die registrierte Bootstrap-Methode ändern sich dadurch
nicht.

Das A/A-Gate klassifiziert nur `tie` und ist H0-invalid, wenn eine Bedingung verletzt ist:

- `R_AA` **und das vollständige 95-%-Bootstrap-KI** liegen in `[0.98, 1.02]`;
- das 95-%-KI enthält zusätzlich `1.0`;
- keine Session-Ratio `R_s` liegt außerhalb `[0.95, 1.05]`.

Der A/A-Pilot schätzt ausschließlich die Standardabweichung der Session-Log-Ratios. Er
verwendet keine Compile-, Custom-Kernel- oder Kandidatendaten und liefert keinen H1-
Performancebeweis.

#### 5.3.2 Deterministische Analyse-Fixtures (keine GPU-Zeit)

Die Analyse-Fixtures prüfen die Klassifikation und Aggregation rein arithmetisch. Sie dürfen
keine Sleeps, Timer oder GPU-Aufrufe verwenden und sind kein Performancebeleg. Für drei
Cluster `p=0..2` und 30 Paare `b=0..29` lautet die Baseline in Nanosekunden exakt:

```text
baseline_ns(p,b) = 1_000_000 + 1_000 * (((17*p + 13*b) mod 11) - 5)
```

Die Slow-Fixture verwendet exakt `candidate_ns = 1.10 × baseline_ns` und muss als
`regression` mit `baseline_fallback` klassifiziert werden. Eine optionale Known-Win-
Fixture verwendet exakt `candidate_ns = 0.90 × baseline_ns` und darf nur den Analysepfad
`promoted` prüfen; sie ist kein GPU- oder Forschungsresultat.

Die feste Falsch-Fixture verwendet die kleine `64²`-Correctness-Fixture mit Seed
`0xBAD02026`; das Ergebnis ist `zeros_like(matmul)` statt des Matmul-Ergebnisses. Sie muss
als `invalid: correctness` klassifiziert und niemals getimt werden.

Die Missing-Data-Fixture lässt das Pflichtfeld `rss_peak_bytes` ohne `missing_reason` weg
und muss als `invalid: missing_required_field` klassifiziert werden. Ein Replay derselben
kanonischen Bytes muss denselben Decision-Hash erzeugen.

#### 5.3.3 Geschlossene Worker-Control-Fixtures

Der freigegebene feste Worker Option A kennt zwei feste, nicht frei parametrisierbare
Kontrollmodi. Ihr Offline-Vertrag und die Fallback-Persistenz sind implementiert:

- `control_timeout`: überschreitet die 120-s-Gesamtdeadline ohne GPU-Arbeit; erwartet wird
  die Fehlerklasse `timeout` und Baseline-Fallback;
- `control_exit_70`: beendet den Worker mit Exit-Code `70`; erwartet wird die Fehlerklasse
  `worker_exit` und Baseline-Fallback.

Ein realer GPU-Timeout-/Crash-Lauf bleibt bis zum separat angekündigten H0-Go/No-Go
gesperrt; die Offline-Kontrollen sind kein Hardware-Nachweis. Es gibt keine freie
Command-, Source- oder Codeeingabe.

## 6. Zeitmessung

Alle Zeitstempel kommen aus `time.perf_counter_ns()`. Für jeden Output ist exakt ein
`mx.eval(out)` auszuführen; unmittelbar vor dem Ende jedes Zeitfensters ist exakt
`mx.synchronize()` auszuführen. Eine gemessene Auswertung umfasst Aufruf, `mx.eval(out)`
und die erforderliche Stream-Synchronisation. Enqueue-Zeit allein ist kein Ergebnis.
Die API-Namen `time.perf_counter_ns`, `mx.eval` und `mx.synchronize` werden unverändert
im Manifest gespeichert.

Die folgenden Zeiten werden getrennt gespeichert:

1. `compile_wrapper_setup_ns`: reine Zeit zum Erzeugen des `mx.compile`-Wrappers;
   sie ist keine Compile-Zeit und wird nicht als solche bezeichnet;
2. `first_eval_compile_inclusive_ns`: erste Auswertung inklusive tatsächlicher erster
   Kompilierung und vollständiger Synchronisation;
3. Cold-eval-Zeit nach definierter Prozess-/Fixture-Initialisierung;
4. Warm-eval-Zeit jeder Wiederholung nach Warmup.

Für Laufzeitstatistiken wird eine Zweierpotenz von Wiederholungen gewählt. Die kleinste
Zweierpotenz, deren Messfenster zwischen 50 und 200 ms liegt, wird verwendet; maximal
4096 Wiederholungen sind erlaubt. Liegen auch 4096 Wiederholungen außerhalb des Fensters,
wird die Session `invalid`; es wird keine Performanceaussage daraus abgeleitet.

Vor der Statistik gibt es je Variante zunächst exakt acht Warmup-Blöcke. Warmups werden
nicht als Ergebnisstatistik ausgewertet, aber jeder Fehler invalidiert die Variante. Ein
Warmup-Block läuft, bis seine äußere gemessene Dauer `block_ns >= 50_000_000` ns erreicht.
Die Auswertungszahl wird dabei auf höchstens 4096 begrenzt. Wird diese Grenze erreicht,
ohne die Mindestdauer zu erreichen, bricht der Lauf fail-closed mit
`repetition_window_unreachable` ab.

Der Gate-Wert eines Blocks ist ausschließlich der gerundete ganzzahlige Wert
`round(block_ns / evaluations)`. Das ist dieselbe Batch-Statistik wie im produktiven
Messpfad; der Median einzelner Evals ist kein Warmup-Gate. Stabil ist eine Variante genau
dann, wenn die Gate-Werte der letzten fünf Blöcke jeweils innerhalb von ±5 % ihres
Medians liegen. Andernfalls wird jeweils genau ein weiterer Block angefordert, bis
maximal 16 Blöcke erreicht sind; wird dann keine Stabilität erreicht, ist die Variante
`invalid` mit `warmup_unstable`, nicht „bereinigt“.

Pro Block werden höchstens ein persistierter Gate-Sample und ein geschlossener Summary
persistiert. Der Summary enthält `block_index`, `evaluations`, `block_ns`, `per_eval_ns`
und `median_eval_ns`, `min_eval_ns`, `max_eval_ns` der Einzeleval-Zeiten. Eine ungebundene
Liste aller Einzelevals wird nicht persistiert.

Die neue `warmup_unstable`-Diagnose wird strikt als Schema v2 erzeugt. Schema-v1-Readback
mit den historischen Einzelwerten bleibt ausschließlich für bestehende Daten kompatibel;
neue Läufe dürfen keine v1-Diagnose mehr erzeugen.

## 7. Prozess- und Blockdesign

Es gibt drei Charakterisierungsprozesse und drei frische unabhängige Bestätigungsprozesse
für dieselbe kanonische `2048²`-Workload. Diese Bestätigungsprozesse sind kein unbekannter
Workload-Holdout und belegen keine Generalisation auf andere Shapes oder Workload-Familien.
Jeder Prozess ist unabhängig startbar und erhält ein gespeichertes Manifest mit Revision, Code-Hash,
Umgebungs-Hash, Fixture-Digest und Seeds.

In jedem Prozess werden 30 gepaarte Blöcke ausgeführt. Ein Block misst Baseline und
Kandidat mit derselben Fixture und demselben Seed. Die Reihenfolge wird pro Block
randomisiert, aber über die vollständige Folge randomisiert-ausgeglichen, sodass jede
Variante gleich oft zuerst und zuletzt läuft. Die Blockreihenfolge, Reihenfolge,
Seed-Liste und Prozess-ID werden gespeichert.

Die Seeds und Reihenfolgen werden vollständig vorab im Manifest festgelegt: für
Charakterisierung erhält Prozess `i=0..2` die Fixture-Seeds `0xF17A2026+i` und die
Reihenfolge-Seeds `0xB10C2026+i`; für Bestätigung erhält Prozess `i=0..2` die Fixture-Seeds
`0xF17A2126+i` und die Reihenfolge-Seeds `0xB10C2126+i`. Es gibt keine
fresh-generated Mehrdeutigkeit. Charakterisierungsdaten dürfen nur Stabilitäts- und
Kalibrierungsentscheidungen liefern. Nachträgliches Umsortieren oder Entfernen von
Blöcken ist verboten.

## 8. Statistik und Unsicherheit

Für jede Variante und jedes Prozess-Set werden Median, MAD und IQR der gepaarten
Warmzeiten berechnet. Zusätzlich werden gepaarte Ratios `candidate / baseline` pro Block
und deren Median, MAD und IQR gespeichert.

Für spätere Kandidaten-/H1-Datensätze wird das 95-%-Konfidenzintervall je Datensatz getrennt
hierarchisch mit jeweils 10.000 Bootstrap-Resamples berechnet: Charakterisierung verwendet
den festen Seed `0xB0052026`, Bestätigung den festen Seed `0xB0052126`. Diese `B005`-Seeds
sind ausdrücklich **nicht** die A/A-Seeds und werden nicht für den A/A-Nullpfad verwendet.
Die Hierarchie resampelt Prozesse
und darin Blöcke; Charakterisierung und Bestätigung werden nie vermischt. Drei Prozess-
cluster sind dabei nur ein Engineering-Gate, kein belastbarer wissenschaftlicher CI- oder
Power-Nachweis. Ein Bootstrap mit 10.000 Resamples erzeugt keine zusätzliche Unabhängigkeit.
Nach dem A/A-Pilot muss H1 für die Workload-/Shape-Familien eine cluster-level
Powerplanung mit vorab eingefrorener Mindestwirkung und Clusterzahl registrieren.
Implementierung, jeweiliger Seed, Replikationszahl und Intervallmethode werden im Manifest
versioniert.

### 8.1 Formale H1-Sperre nach dem H0-Pilot

Der A/A-Pilot schätzt ausschließlich die Standardabweichung der Session-Log-Ratios. Pilot-
daten dürfen weder Compile- noch Kandidatendaten enthalten und werden niemals in die
bestätigende H1-Auswertung übernommen.

Vor jeder Kandidatensichtung muss eine vollständige H1-Vorregistrierung eingefroren sein:

- Mindestwirkung `5 %`, `alpha = 0.05`, Power `0.80`;
- feste Workload-/Shape-Familien, feste Clusterzahl und feste Analyse;
- mindestens fünf unabhängige Sessions je Arm und Familie;
- vorab registrierte obere Machbarkeitsgrenze für die Sessionzahl (empfohlen `20`);
- falls die benötigte Zahl die Grenze übersteigt: H1 ist `infeasible/no claim`, die Regeln
  werden nicht geöffnet und das Testset nicht erweitert.

Für jede Hypothese und jede Revision wird ein frisches versiegeltes Testset verwendet. H2
darf kein durch H1 geöffnetes Testset wiederverwenden. Der versiegelte Test zieht erst nach
dem Freeze aus breiten, vorab registrierten Shape-, Value- und Layout-Verteilungen. Sein
256-bit-Seed liegt außerhalb des Repositories; vorab wird nur ein kryptographischer
Commit-Hash veröffentlicht. Dieses Protokoll ist ein Forschungsvorschlag und keine
Architektur- oder Sicherheitsfreigabe.

Es gibt keine nachträgliche Ausreißertrimmung. Fehlerhafte oder fehlende Blöcke werden
nicht als Ausreißer behandelt, sondern machen den betroffenen Prozess oder die Variante
nach den Gate-Regeln ungültig.

## 9. Promotion- und Fallback-Gates

Ein Kandidat gewinnt nur, wenn alle Bedingungen in Charakterisierung und Bestätigung jeweils
separat erfüllt sind; die Datensätze werden nie zu einem gemeinsamen Set vermischt:

- Median-Ratio `<= 0.95` in Charakterisierung und Bestätigung;
- obere Grenze des hierarchischen 95-%-KI `< 1.0` in Charakterisierung und Bestätigung;
- keine Session weist in Charakterisierung oder Bestätigung eine Ratio `>= 1.05` auf;
- Correctness, Memory und Safety sind grün;
- alle erforderlichen Messwerte und Hashes sind vorhanden.

Andernfalls wird das Ergebnis als `regression`, `tie`, `invalid` oder `baseline_fallback`
klassifiziert. Bei Regression, Tie, fehlender Bestätigung, Correctness-/Memory-/Safety-
Fehler oder unvollständiger Evidenz bleibt die Eager-Baseline aktiv. Ein Nullresultat ist
ein gültiges Forschungsergebnis.

### 9.1 H1- und H2-Metrikvertrag für spätere Phasen

H1 verwendet `R = T_candidate / T_strongest_baseline`, kleiner ist besser. Pro Familie wird
der Session-Median gebildet; über die vorab registrierten Familien wird das geometrische
Mittel auf der Log-Skala ausgewertet. H1-Erfolg erfordert Gesamt-`R <= 0.95` und eine obere
95-%-Cluster-KI `< 1.0`. Correctness, Memory und Safety sind Guardrails; keine Familie darf
eine Regression `R >= 1.05` zeigen.

Die Amortisation ist ein separates Hard-Gate. `T_baseline` ist immer die Zeit der stärksten
Baseline. Alle `T_*`-Zeiten müssen nichtnegativ sein, dieselbe Einheit verwenden und
denselben vorab registrierten Workload-Mix und Scope abdecken. Mit `T_tune`, `T_compile`,
`T_baseline` und `T_candidate` gilt:

```text
if T_strongest_baseline <= T_candidate:
    N_break_even = infinity      # decision = no_break_even
else:
    N_break_even = ceil((T_tune + T_compile) /
                        (T_strongest_baseline - T_candidate))
```

`T_strongest_baseline` bezeichnet dabei dasselbe `T_baseline` im registrierten Scope.
Bei `no_break_even`/unendlich fällt das Hard-Gate unabhängig von `N`; negative, null- oder
fehlende Zeit-Artefakte dürfen nicht in die Formel gelangen. Andernfalls muss der
aufgerundete Wert kleiner oder gleich dem vorab registrierten Aufrufbudget `N` sein.

H2 besitzt genau eine Primärmetrik: die finale Best-Valid-Sealed-Test-Ratio nach einem
festen Hardware-Trialbudget `B`, gepaart über neue H2-Workload-/Shape-Familien gegen die
stärkste deterministische Suche. Trials bis zum ersten Gewinn, Gesamtwalltime,
Modellkosten und Invalid-Rate sind sekundär. H2-Erfolgsschwelle und Power werden mit dem
Modellantrag vor jedem Download oder jeder Installation eingefroren, niemals nach dem
Ergebnis.

## 10. Memory-Messung

Folgende Metriken werden getrennt erfasst; unterschiedliche APIs dürfen nicht als
identisch bezeichnet werden:

- MLX active memory;
- MLX peak memory;
- MLX cache/reserved memory;
- Prozess-RSS.

Jede Metrik enthält API-Name, Einheit, Zeitpunkt, Reset-/Prozesszustand und `null` plus
Grund, falls sie auf dieser Installation nicht verfügbar ist. Das Memory-Gate ist
separat je Domäne: `Peak_candidate <= 1.05 × Peak_baseline + 1 MiB` für MLX und ebenso
für RSS, sofern beide Werte vorhanden sind. Ein fehlender Wert besteht das Gate nicht.

## 11. Safety und Ressourcenlimits

Vorgesehene Limits sind: 10 s für First-eval inklusive Compile, 5 s für jeden
Synchronisationsaufruf und 120 s für einen Gesamtprozess. Zielgrenzen sind 1 GiB MLX-
Speicher und 2 GiB RSS. Die konkrete Durchsetzung durch Worker, Timeout, Prozessgruppe
und Rollback ist eine separate Sicherheits-/Architekturentscheidung und darf nicht aus
dieser Spezifikation als bereits implementiert abgeleitet werden.

Ein Timeout, Crash, Compilefehler, Speicherüberschreitung oder nicht reproduzierbarer
Zustand stoppt den Prozess, markiert die Variante ungültig und aktiviert die Baseline.
Keine Netzisolation wird behauptet. Die spätere kontrollierte Ausführung muss ihre
Isolation und ihre Grenzen ausdrücklich nachweisen.

## 12. Revision, Identität und fehlende Werte

Der Projekt-Root ist kein Git-Repository. Deshalb wird `revision: null` mit Grund
`project root is not a Git repository` gespeichert. Zusätzlich werden Code-Hash,
Manifest-Hash, Spezifikations-Hash und Umgebungs-Hash gespeichert. Änderungen an diesen
Hashes eröffnen einen neuen Messlauf; Ergebnisse werden nicht still zusammengeführt.

Jedes fehlende Feld wird als `null` gespeichert und erhält ein maschinenlesbares
`missing_reason`. Vermutete Werte, Schätzungen und stillschweigende Defaults sind nicht
zulässig.

## 13. Artefakte und implementierter Offline-Unterbau

Der implementierte H0-Unterbau speichert Fixture, Seeds, Rohzeiten, Correctness, Memory,
Safety, Bootstrap-Details, Hashes und Entscheidungen transaktional in SQLite v1 unter
`.friday-data/h0.sqlite3`. Der feste Worker Option A und der Adaptervertrag bleiben
geschlossen; unbekannte Pfade, Module, Flags oder freie Codeeingabe sind nicht zulässig.
Aggregate müssen aus den unveränderten Rohdaten rekonstruierbar sein.

Die lokale Historien-Dashboard-UI ist als read-only Loopback-Dashboard auf `127.0.0.1`
vorgesehen. Sie liest nur aus der SQLite-Historie und verändert weder Messlogik noch
Daten. Dies ist keine Behauptung vollständiger Netzwerk-, Datei- oder Prozessisolation.
Produktive H0-Rohdaten existieren noch nicht; Dashboard-Tests verwenden ausschließlich
temporäre Testdatenbanken.

## 14. Reihenfolge des kontrollierten Versuchs

1. Manifest und Hashes erzeugen; erst dann Fixture und Oracle erstellen.
2. Contract- und API-Verfügbarkeit prüfen und protokollieren; `mx.compile` dient nur als
   sichere Framework-Vergleichsvariante gemäß `CODEX_START.md`; A/A ist der echte Nullpfad.
3. FP64-Baseline-Kalibrierung und acht initiale Warmup-Blöcke durchführen.
4. Charakterisierungsblöcke mit gespeicherten Seeds ausführen.
5. Stabilitäts-, Correctness-, Memory- und Safety-Gates auswerten.
6. Nur mit eingefrorener Semantik und unabhängigen FP64-Hard-Caps Kandidat und
   Bestätigungsprozesse messen.
7. Statistik, hierarchisches Bootstrap und Promotion-Entscheidung schreiben.
8. Rohdaten unverändert behalten und eine kurze Ergebniszusammenfassung erzeugen.

Jeder Schritt ist reproduzierbar, abbrechbar und journalpflichtig. Bei einem Fehler wird
Ursache, beobachteter Messwert, Korrektur und erneute Verifikation im Arbeitsjournal
ergänzt; die ursprüngliche Rohmessung bleibt erhalten.

## 15. Akzeptanzkriterien für Phase 1A

- Diese Vorregistrierung bleibt unverändert, solange der Versuch läuft.
- Eine unabhängige Phase-1A-Ausführung kann kanonische Performance-Fixture, Correctness-
  Matrix, Seeds, Hashes und alle Gates rekonstruieren.
- Baseline und sichere `mx.compile`-Vergleichsvariante sind semantisch identisch und
  enthalten keinen Customcode.
- A/A-GPU-Nullkontrolle verwendet exakt drei Charakterisierungs- und drei Bestätigungs-
  prozesse mit je 30 Paaren, den festgelegten `AA1A`-/`AA0D`-Seeds und dem Gate `tie`,
  Median-Ratio `[0.98,1.02]`, KI-Inklusion von `1.0` und Session-Ratios in `[0.95,1.05]`.
- Deterministische Analyse-Fixtures klassifizieren Slow `1.10x` als Regression/Fallback,
  optional Known-Win `0.90x` nur analytisch als promoted, Falschdaten als
  `invalid: correctness`, Missing Data als `invalid: missing_required_field` und Replay
  identisch; keine dieser Fixtures verwendet Sleep oder GPU-Zeit.
- Timeout-/Crash-Kontrollen heißen ausschließlich `control_timeout` und `control_exit_70`,
  bleiben bis Worker-/Architekturfreigabe unimplementiert und werden nicht als H0 bestanden
  ausgegeben.
- Warmup-Blöcke, `compile_wrapper_setup_ns`, `first_eval_compile_inclusive_ns`, Cold, Warm,
  Memory und Safety sind getrennt belegt.
- Drei Charakterisierungs- und drei unabhängige Bestätigungsprozesse mit je 30 gepaarten
  Blöcken sind für
  die kanonische Performance-Workload vollständig oder explizit invalidiert.
- Keine Performanceentscheidung erfolgt ohne Ratio-, KI-, Bestätigungs-, Correctness-, Memory-
  und Safety-Nachweis; die Correctness-Matrix bleibt aus der Performanceaggregation.
- Ein negatives oder nicht reproduzierbares Ergebnis wird als gültiges Ergebnis archiviert.

Der vollständige Phase-1-DoD ist erst nach separater Phase 1B mit begrenztem Custom-MLX-
Metal-Kandidaten und isoliertem Worker erreicht. Phase 1B ist hier weder erfüllt noch
freigegeben. H1 benötigt zusätzlich eine separate Workload-/Shape-Familienaufteilung und
eine cluster-level Powerplanung; die drei H0-Prozesscluster ersetzen beides nicht.

Die formale H1-Sperre aus Abschnitt 8.1 und der H1/H2-Metrikvertrag aus Abschnitt 9.1 sind
vor jeder Kandidatensichtung beziehungsweise jedem Modellantrag einzuhalten.

## 16. Revision und Ausführungsstatus

Diese methodische Revision vom 20. August 2026 ändert die Entscheidungsgrenzen und den
Offline-Implementierungsstatus des H0-Preflights. Der SQLite-/Dashboard-/Worker-Vertrag
und der rückwärtsinkompatible Manifest-v1-Contractfix für die expliziten A/A-Bootstrap-
Seeds `0xAA052026` und `0xAA052126` sind umgesetzt; die `aa_gpu`-Manifeste binden diese
Seeds, und `aggregation_contract_ready=true`. Bis zum separat angekündigten Go/No-Go
bleibt die A/A-GPU-Ausführung durch `live_execution_authorized=false` gesperrt. Die
Statistik und ihre 10.000er-Bootstrapdefinition bleiben unverändert.

In diesem Dokumentationsschritt wurden keine Tests, GPU-Läufe, Worker-Läufe, Downloads,
Installationen oder Messungen ausgeführt; die bisherigen Live-Läufe waren ungültig oder
fail-closed, und es gibt noch keine gültige abgeschlossene H0-Performanceentscheidung.
Ein begrenzter `eager_baseline`-Canary ist nach der W1v3-/Output-Umsetzung einmalig
vorgesehen; `aa_gpu` bleibt bis zu einer separaten Freigabe gesperrt.

# Vorregistrierung: begrenzter H1-Runtime-Prototyp

Status: **vor Implementierung der Live-Messungen eingefroren**

Runtime-ID: `h1-runtime-dispatch-n8-20260821-01`

Datum: 2026-08-21

## 1. Zweck und zulässige Aussage

Der Prototyp überführt ausschließlich die formal bestätigte H1-v2-Entscheidung in einen
kleinen lokalen Ausführungspfad. Er untersucht, ob eine evidenzgebundene Policy den bereits
bestätigten Dispatch-Plan sicher und mit vernachlässigbarem Steuerungsaufwand auswählen kann.

Zulässig ist höchstens die Aussage:

> Auf genau dem versiegelten Gerät, in der versiegelten Softwareumgebung und für acht
> FP16-Matmul-Operationen mit 2048 × 2048 Matrizen kann die Runtime den formal bestätigten
> Batch-Dispatch-Plan auswählen, ohne dessen Korrektheit oder gemessenen Vorteil durch den
> Policy-Pfad aufzuheben.

Nicht zulässig sind Modell-, End-to-End-, ANE-, Kernel-, Compiler-, Cross-Shape- oder
Cross-Device-Aussagen. Die Runtime-Messung ist Engineering-Validierung, keine zweite formale
Hypothesenprüfung.

## 2. Unveränderliche H1-Bindung

Die optimierte Auswahl ist nur erlaubt, wenn die vollständige H1-Historie mit dem versiegelten
H1-v2-Protokoll erfolgreich wiedergegeben wird und alle folgenden Identitäten exakt stimmen:

- Study-ID: `h1v2-dispatch-n8-20260821-01`
- terminaler Record: `f508fc9e2b1f44a1b60084bdbeca581024f1f3599535b3dd662a9305c99a9357`
- Decision-SHA-256: `5b022a1dcc127cba05dc86c427dafcc0b8a629e479cc1d29d742514555a5baa5`
- Preregistration-SHA-256: `50baafba71656e1786f120098e1d4f47933c9ab532c8891c39aa6d248561b550`
- H1-Provenienz-SHA-256: `e08732640516712818fd1872411acdcbfdf7fb91849a588ee1101a8007e7d7e3`
- Status: `h1_gain_confirmed`
- Aktion: `permit_bounded_runtime_prototype`
- Claim-Scope: `one-device-one-workload-one-execution-plan`
- Gates: Byte-Identität und Gain in Gesamt-, Charakterisierungs- und Validierungssplit.

Zusätzlich müssen die aktuellen H1-Code-, H1-Spec-, Umgebungs- und Hardware-Fingerprints den
versiegelten Fingerprints entsprechen und der Projekt-Worktree sauber sein. Die Git-Revision
darf sich wegen des separat hinzugefügten Runtime-Pakets unterscheiden; H1-Code und H1-Spec
dürfen sich nicht unterscheiden.

## 3. Geschlossener Workload

| Feld | Wert |
|---|---|
| Operation | `matmul` |
| Datentyp | `float16` |
| LHS/RHS/Output | jeweils `2048 × 2048` |
| RHS-Anzahl | `8` |
| Baseline | pro Operation `eval` und `synchronize` |
| Kandidat | acht Operationen einreihen, einmal gemeinsam `eval` und `synchronize` |

Die Runtime leitet Shape und Datentyp aus den tatsächlichen Tensoren ab. Eine vom Aufrufer
behauptete Kennzeichnung autorisiert keinen Batch-Pfad.

## 4. Fail-closed Policy

Das H1-Artefakt wird beim Erzeugen eines Controllers genau einmal vollständig und read-only
geladen. Danach liegt eine unveränderliche Policy-Projektion im Speicher; der Hot Path liest
keine SQLite-Datei.

Jede fehlende, beschädigte, fremde oder nicht exakt passende Evidenz sowie jeder unbekannte
Workload führt zur seriellen Baseline. Schlägt ein autorisierter Batch-Aufruf fehl, wird er im
selben Aufruf **nicht** automatisch seriell wiederholt. Damit entstehen keine verdeckten
Doppelwirkungen. Stattdessen wird ein prozessweiter Circuit Breaker verriegelt; alle
Folgeaufrufe wählen seriell. Der fehlerhafte Aufruf bleibt sichtbar als Fehler.

## 5. Policy-Overhead-Messung (CPU)

- fünf Warmup-Blöcke;
- 21 Messblöcke mit abwechselnder Reihenfolge A/B und B/A;
- 20.000 Aufrufe pro Arm und Block;
- A: direkter Zugriff auf den bereits bekannten unveränderlichen Plan;
- B: Tensor-Metadaten beobachten und die gecachte Policy auswählen;
- Median, MAD und interpoliertes p95 werden gespeichert;
- Rohsummen und Nanosekunden pro Aufruf werden für alle Blöcke gespeichert.

Vorab eingefrorene Freigabegates:

- Policy-Median ≤ 25.000 ns;
- Policy-p95 ≤ 50.000 ns;
- gepaarter zusätzlicher Median ≤ 20.000 ns;
- Auswahl bleibt `batched`; Circuit Breaker bleibt offen.

## 6. MLX/GPU-Engineering-Validierung

- Netzbetrieb ist Pflicht;
- derselbe deterministische H1-Fixture- und Operand-Seed;
- eine ungemessene Korrektheitsausführung je Arm vor Timing;
- zwei Warmup-Paare;
- zwölf Messblöcke, alternierend A/B und B/A;
- Median und MAD für beide Pläne, Median der gepaarten B/A-Verhältnisse;
- vollständige Output-Digests und maximaler absoluter Fehler;
- bestehende GPU-, kontinuierliche Last-, Duty-Cycle- und Walltime-Budgets bleiben aktiv.

Freigabegates:

- alle acht Outputs byte-identisch und maximaler absoluter Fehler `0.0`;
- Runtime wählt tatsächlich den Batch-Plan;
- Median der gepaarten B/A-Verhältnisse ≤ `0.95`;
- Circuit Breaker bleibt offen.

Ein negatives oder nicht reproduzierbares Ergebnis sperrt die Optimierung, ist aber ein gültiges
Forschungsergebnis.

## 7. Persistenz und UI

Runtime-Messungen werden getrennt von der terminalen H1-Datenbank in
`.friday-data/runtime.sqlite3` gespeichert. Die Datenbank ist privat (`0600`), append-only und
verknüpft jeden Datensatz kryptografisch mit dem Vorgänger. Lesen verifiziert Schema,
kanonisches JSON, Projektionen, Provenienz, Record-Identität und die vollständige Hash-Kette.

Die lokale UI bindet ausschließlich an `127.0.0.1:8769`, öffnet die Datenbank read-only und
zeigt Historie, Status, Kennzahlen und Revision. Sie besitzt keinen Mutationsendpunkt.

## 8. Freigabe für den nächsten Schritt

Erst wenn Policy- und GPU-Gates gemeinsam bestanden sind, darf ein getrenntes H2-Experiment
prüfen, ob ein bereits lokal vorhandenes Gemma-Modell innerhalb eines geschlossenen
Kandidatenraums nützliche Planvorschläge liefern kann. H1 bleibt dabei die alleinige
Performance-Evidenz; Modellvorschläge erhalten nie direkte Ausführungs- oder Installationsrechte.

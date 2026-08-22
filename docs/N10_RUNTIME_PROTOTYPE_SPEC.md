# Vorregistrierung: begrenzter N10-Runtime-/AVO-lite-Prototyp

Status: **vor Implementierung der Live-Messungen eingefroren**

Runtime-ID: `n10-runtime-dispatch-20260822-01`

Datum: 2026-08-22

## 1. Zweck und zulässige Aussage

Der Prototyp überführt ausschließlich den formal positiven N10-v2-Entscheid in
einen getrennten lokalen Ausführungspfad. Er prüft, ob eine evidenzgebundene
Policy den bereits bestätigten Dispatch-Plan sicher, korrekt und mit begrenztem
Steuerungsaufwand auswählen kann.

Zulässig ist höchstens die Aussage:

> Auf genau dem versiegelten Gerät, in der versiegelten Softwareumgebung und für
> zehn FP16-Matmul-Operationen mit 2048 × 2048 Matrizen kann die getrennte
> Runtime den formal bestätigten Batch-Dispatch-Plan auswählen, ohne dessen
> Korrektheit oder gemessenen Vorteil durch den Policy-Pfad aufzuheben.

Die Runtime-Validierung ist Engineering-Evidenz, keine zweite formale
Hypothesenprüfung. Modell-, End-to-End-, ANE-, Kernel-, Compiler-, Cross-Shape-
und Cross-Device-Aussagen sind ausgeschlossen. Die bestehende N8-Runtime und
ihre Datenbank werden nicht verändert.

## 2. AVO-lite als geschlossene Zustandsmaschine

AVO-lite bedeutet hier ausschließlich:

1. **Observe:** Shape, Datentyp und Operandenzahl aus realen Tensoren ableiten;
2. **Verify:** den exakten terminalen N10-v2-Store, seinen Vorgänger und die
   versiegelte Geräte-/Softwareidentität einmalig read-only prüfen;
3. **Optimize:** genau einen fest registrierten Batch-Dispatch-Plan auswählen;
4. **Validate:** Hot-Path-Overhead, GPU-Effekt, Byteidentität und Budgets messen;
5. **Fallback:** bei Unsicherheit seriell ausführen; bei einem Batchfehler den
   aktuellen Aufruf nicht wiederholen und den Circuit Breaker verriegeln.

Es gibt keinen LLM-Aufruf, keine freie Aktion, keine Codegenerierung, keinen
Kandidatensuchraum und kein selbständiges Nachtrainieren.

## 3. Unveränderliche N10-v2-Bindung

Die optimierte Auswahl ist nur erlaubt, wenn alle Identitäten exakt stimmen:

- Study-ID: `h2n10-dispatch-confirmation-20260822-02`;
- terminaler Store: `.friday-data/n10-v2.sqlite3`, SHA-256
  `54e9c57ca6b76fa671b94f748b7ee471575b7dd7445bad00ae3cab38f691fc4f`;
- Snapshot-Revision:
  `9c9a94a8f799f2eb29b9e03c4e1b6e681aa945199753158cf8fc8c317b06090d`;
- terminaler Record:
  `47283e73eb6eefa01dc0f2e1760a2a2d350ca51019b8ddfa3a297d4b695e1249`;
- Decision-SHA-256:
  `99f08dbb92730ec68a8867f15b4aeff4297a06284e7ffde4a63a76152420adf2`;
- Preregistration-SHA-256:
  `771c715520d3289cd2fbf051d469228d5686ae921eed1104e154914ac2a85ac8`;
- N10-v2-Provenienz:
  `17d0dd505e349a4bbb7ffde3c291a3a44226d0fce79c235ce2ce890289e0c9ef`;
- Status `n10_gain_confirmed`, Aktion
  `permit_bounded_n10_runtime_prototype`, Claim
  `n10_batched_dispatch_is_faster_beyond_mde`;
- genau 16 Records, genau ein formaler Claim, Byteidentität und Gain in Gesamt-,
  Charakterisierungs- und Validierungssplit.

Der N10-v1-Vorgänger muss weiterhin exakt terminal und timingfrei replayen.
Zusätzlich müssen aktuelle N10-Code-, N10-Spec-, Umgebungs- und
Hardwarefingerprints den versiegelten Fingerprints entsprechen:

- Code `727f1faa52f22595ef506b8194588a49a9f2bbd07355adb52a01cbe465660efe`;
- Spec `9c9e28f0d36213051654746b386a77e00eb116a8ae503a99ef1d2c987312ea65`;
- Umgebung
  `6ef07ef1a2976e4dfc5a0fb7a65b1535a28372c145f8d488c7cd5d0a33ff6624`;
- Hardware
  `ee157aaa01de24f2fcb3057bf6cacbfbc361257d2a192eadc3fd75f33f3133b3`.

Die Root-Git-Revision darf sich wegen des getrennten Runtime-Pakets und der
Ergebnisdokumentation unterscheiden. Der Worktree muss bei Live-Messungen sauber
sein; N10-Code und gebundene N10-Spec bleiben bytegleich.

## 4. Geschlossener Workload

| Feld | Wert |
| --- | --- |
| Operation | `matmul` |
| Datentyp | `float16` |
| LHS/RHS/Output | jeweils `2048 × 2048` |
| RHS-Anzahl | `10` |
| Baseline | pro Operation `eval` und `synchronize` |
| Kandidat | zehn Operationen einreihen, einmal gemeinsam `eval` und `synchronize` |

Die Runtime leitet die Metadaten aus den tatsächlichen Tensoren ab. Caller-Labels
oder ähnliche Behauptungen autorisieren den Batchpfad nicht.

## 5. Fail-closed Policy

Der Controller lädt die N10-v2-Evidenz genau einmal vollständig und read-only;
der Hot Path greift danach nur auf die unveränderliche Projektion im Speicher zu.
Die Cold-Load-Laufzeit wird separat gespeichert und muss unter `10 s` bleiben.

Jede fehlende, beschädigte, fremde, zu offen berechtigte oder nicht exakt
passende Evidenz sowie jeder unbekannte Workload führt zur seriellen Baseline.
Ein autorisierter Batchfehler wird im selben Aufruf nicht seriell wiederholt.
Stattdessen wird ein prozessweiter Circuit Breaker verriegelt; Folgeaufrufe sind
seriell, der fehlerhafte Aufruf bleibt sichtbar.

## 6. Policy-Overhead-Messung (CPU)

- fünf Warmup-Blöcke;
- 21 Messblöcke mit alternierender A/B- und B/A-Reihenfolge;
- 20.000 Aufrufe pro Arm und Block;
- A: direkter Zugriff auf den bekannten unveränderlichen Plan;
- B: zehn Tensor-Metadaten beobachten und die gecachte Policy auswählen;
- Median, MAD, interpoliertes p95, Rohsummen und ns/Aufruf speichern.

Vorab eingefrorene Gates:

- Cold Evidenzload ≤ `10.000.000.000 ns`;
- Policy-Median ≤ `25.000 ns`;
- Policy-p95 ≤ `50.000 ns`;
- gepaarter zusätzlicher Median ≤ `20.000 ns`;
- Auswahl bleibt `batched`, Circuit Breaker bleibt offen.

## 7. MLX/GPU-Engineering-Validierung

- Netzbetrieb ist Pflicht;
- exakt die registrierte N10-v2-Fixture und deren Operand-Seed;
- eine ungemessene Korrektheitsausführung je Arm;
- zwei Warmup-Paare;
- zwölf Messblöcke, alternierend A/B und B/A;
- Median/MAD beider Pläne und Median der gepaarten B/A-Verhältnisse;
- Output-Digests, maximaler absoluter Fehler, CPU/RSS/MLX- und Budgetwerte.

Gates:

- alle zehn Outputs byteidentisch, maximaler absoluter Fehler `0.0`;
- Runtime wählt tatsächlich den Batchplan;
- Median der gepaarten B/A-Verhältnisse ≤ `0.95`;
- alle Ressourcenbudgets bestanden, Circuit Breaker offen.

Ein negatives oder nicht reproduzierbares Ergebnis sperrt N10 in der Runtime und
bleibt ein gültiges Ergebnis. Es gibt keinen automatischen Retry.

## 8. Persistenz und UI

Engineering-Messungen werden getrennt in
`.friday-data/runtime-n10.sqlite3` gespeichert. Application-ID `FRN1`, Modus
`0600`, geschlossenes SQLite-v1-Schema, append-only Trigger, kanonisches JSON und
vollständige Hashkette sind Pflicht. Jeder Report bindet den formalen
N10-Decision-Record und trägt `formal_claim=false`.

Die lokale UI bindet nur an `127.0.0.1:8772`, öffnet die Datenbank read-only und
zeigt Historie, Status, Kennzahlen und Revision. Sie besitzt keinen
Mutationsendpunkt und soll `KeyboardInterrupt` beim manuellen Stop ohne
Traceback behandeln.

## 9. Autorisierte Reihenfolge

```text
ProjectAtlas-Kontext und unveränderte Baselines
→ Implementierung und Offline-Tests
→ sauberer lokaler Commit
→ read-only Policy-Load
→ benchmark-policy --run-id n10-policy-overhead-20260822-01 --execute
→ nur bei bestandenem CPU-Gate:
  validate-gpu --run-id n10-runtime-validation-20260822-01 --execute
→ read-only Replay, UI-, Hash- und Baselineprüfung
```

Vor dem sauberen Commit sind keine neuen Runtime-Livemessungen zulässig.
Freie Codegenerierung, Custom Metal, weitere Modellrunden und ein breiterer
Suchraum bleiben geschlossen.

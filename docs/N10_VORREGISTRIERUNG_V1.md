# N10-v1 — prospektive Ein-Kandidaten-Vorregistrierung

**Freigabe:** 22. August 2026

**Study-ID:** `h2n10-dispatch-confirmation-20260822-01`

**Status dieses Dokuments:** Designvertrag; eine Messung ist erst nach sauberem
Implementierungscommit und persistiertem `preregistration`-Record zulässig.

## 1. Forschungsfrage und Abgrenzung

Für exakt zehn unabhängige FP16-Matrixmultiplikationen der Form
`[2048,2048] @ [2048,2048]` auf dem einen lokal gebundenen Apple-M1-Max-Gerät
wird geprüft, ob ein gemeinsamer MLX-Eval-/Synchronisationspunkt schneller ist
als zehn serielle Eval-/Synchronisationspunkte.

Der Kandidat `N=10` wurde vor dieser Studie einmal explorativ ausgewählt. Die
Auswahl stammt aus dem nativen Research-Record
`5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a`:
Das lokale Gemma 3 4B schlug in genau einer geschlossenen Runde `3,10,16` vor;
der deterministische Harness rangierte `N=10` an erster Stelle. Diese alten
Daten dürfen in keine Schätzung oder Entscheidung dieser Studie eingehen.
Gemma nimmt an Kalibrierung, Bestätigung und Entscheidung nicht erneut teil.
Der Auswahlzeitpunkt ist zusätzlich durch den damaligen Research-DB-Hash
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`
und die Snapshot-Revision
`c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`
gebunden.

Der Claim ist beschränkt auf ein Gerät, eine Tensorform, FP16, genau zehn
Operationen und genau die beiden registrierten Dispatchpläne. Kein Claim gilt
für vollständige Modelle, andere Shapes, Dtypes, Geräte, Kernel, Compiler oder
den Apple Neural Engine.

## 2. Primärer Endpunkt und Entscheidungsregionen

Primärer Endpunkt ist das gepaarte Verhältnis

`R = Dauer(B) / Dauer(A)`

aus synchronisierten Wall-Clock-Dauern. Die kleinste relevante Effektgröße
(`MDE`) wird ausschließlich aus frischen A/A-Daten bestimmt und hat einen
prospektiven Floor von `5 %` sowie einen Cap von `15 %`.

Für Charakterisierung, Validierung und alle sechs Bestätigungssessions werden
getrennte hierarchische 95%-Bootstrapintervalle gebildet:

- **Gain:** alle drei oberen Intervallgrenzen liegen strikt unter `1 − MDE`;
- **Regression:** alle drei unteren Intervallgrenzen liegen strikt über
  `1 + MDE`;
- **Äquivalenz:** alle drei Intervalle liegen vollständig innerhalb
  `[1 − MDE, 1 + MDE]`;
- **sonst:** inkonklusiv.

Ein positiver Entscheid autorisiert nur die Entwicklung und erneute Live-
Validierung eines begrenzten N10-Runtime-Prototyps. Er ändert die bestehende
N8-Policy nicht automatisch.

## 3. Workload und Kandidat

| Feld | Versiegelter Wert |
| --- | --- |
| Operation | `matmul` |
| LHS/RHS/Output | jeweils `[2048,2048]` |
| Dtype | `float16` |
| Anzahl RHS/Operationen | `10` |
| Fixture-Seed | `8754882193294599646` |
| Operand-Seed | `7421913553926890024` |
| Arm A | jede Operation einzeln `eval` + `synchronize` |
| Arm B | zehn Operationen enqueue, dann ein `eval` + `synchronize` |
| Korrektheit | Referenz, A und B müssen byte-identisch sein; max. Fehler `0` |

Die Seeds wurden vor Messbeginn deterministisch aus SHA-256 über die Domain
`project-friday:h2-n10-v1:<label>` abgeleitet: Die ersten acht Digest-Bytes
werden als Big-Endian-Integer gelesen und das höchstwertige Bit gelöscht. Die
Ergebnisse sind danach als 63-Bit-Konstanten eingefroren. Die Labels lauten
`fixture`, `operands`, `session:C0` bis `session:V2`,
`bootstrap:calibration`, `bootstrap:characterization`,
`bootstrap:validation` und `bootstrap:all`.

Versiegelte Session-Seeds:

- `C0`: `5060361785459989569`; `V0`: `883950215809699703`;
- `C1`: `2323802873345837297`; `V1`: `483519612603395666`;
- `C2`: `5893687926320354209`; `V2`: `5188879407004767969`.

## 4. Zweistufiges Studiendesign

### 4.1 Frische A/A-Kalibrierung

Arm A und Arm B führen beide den seriellen Plan über getrennte Callables aus.
Es gibt sechs getrennte Prozesse in der festen Reihenfolge
`C0,V0,C1,V1,C2,V2`. `C*` bildet die Charakterisierungs-, `V*` die
Validierungskohorte. Pro Prozess gelten:

- zwei gepaarte Warmupblöcke;
- 24 gepaarte Messblöcke;
- exakt zwölf `AB`- und zwölf `BA`-Blöcke in deterministisch gemischter Folge;
- Netzbetrieb;
- mindestens 20 Sekunden echte Pause zwischen Prozessen;
- kein optionales Stoppen und kein Retry.

Kalibrierung besteht nur, wenn das Gesamtintervall `1` enthält, der aggregierte
Bias höchstens `5 %` beträgt, die abgeleitete MDE höchstens `15 %` beträgt und
alle Ausgaben byte-identisch sind. Danach werden MDE, Kalibrierungshash und
Sessionhashes in einem separaten `confirmation_seal` eingefroren.

### 4.2 Frische A/B-Bestätigung

Erst nach dem Confirmation-Seal folgen sechs neue Prozesse in derselben
`C0,V0,C1,V1,C2,V2`-Reihenfolge. Arm A ist seriell, Arm B ist der N10-Batchplan.
Warmup-, Block-, Reihenfolge-, Pausen- und Ressourcenregeln bleiben identisch.
Alle sechs Sessions sind für den terminalen Entscheid erforderlich.

## 5. Statistik

- Session-Schätzer: `exp(median(log(B_ns/A_ns)))`;
- Studienschätzer: Median der Session-Log-Mediane;
- Intervall: deterministischer hierarchischer Perzentil-Bootstrap;
- Draws: `10.000`;
- Konfidenz: `0,95`;
- Kandidatenzahl und primäre Endpunkte: jeweils `1`, daher keine zusätzliche
  Multiplizitätskorrektur.

Versiegelte Bootstrap-Seeds:

- Kalibrierung gesamt: `777255143216008523`;
- Bestätigung Charakterisierung: `7159943182929271886`;
- Bestätigung Validierung: `8989465731481879185`;
- Bestätigung gesamt: `4114342224181825282`.

## 6. Ressourcen-, Sicherheits- und Fehlervertrag

Je Session gelten maximal 120 Sekunden erfasste GPU-Arbeit, sechs Sekunden
kontinuierliche GPU-Arbeit, 25 % Duty Cycle in 60 Sekunden und 20 Minuten Wall.
Die gemeinsame BudgetGuard-Policy verlangt bei Bedarf vier Sekunden Pause und
kennt einen 60-Sekunden-Kandidaten-Cooldown. CPU-Zeit, RSS sowie verfügbare
MLX-Speicherzähler werden gespeichert.

Jeder Fehler nach bestandenem Preflight erzeugt – soweit die versiegelte
Provenienz noch unverändert ist – einen terminalen `session_failure`-Record.
Fehlgeschlagene Sessions werden nicht wiederholt. Änderungen an Code, Spec,
Umgebung, Hardwareidentität oder Git-Revision schließen den Lauf fail-closed.

Nicht Bestandteil dieser Studie sind Custom Metal, generierter Code, freie
Modellaktionen, ein Kandidatensuchraum, AVO-Supervision oder eine Änderung der
produktiven Runtime. Diese Schritte benötigen nach dem Studienentscheid einen
eigenen Vertrag.

## 7. Provenienz und Persistenz

Die Vorregistrierung bindet:

- den sauberen Root-Git-Commit und einen leeren Root-Diff;
- alle Python-/SQL-Dateien unter `friday_n10/`;
- `tools/run_n10_v1.py`, den Budgetvertrag und den Fixture-Generator;
- dieses Dokument, `docs/PHASE1_MATMUL_SPEC.md`,
  `docs/H1H2_EVIDENZ_ARCHITEKTUR.md` und die eingefrorenen Requirements;
- Python- und Paketversionen sowie öffentliche Hardware-/macOS-Merkmale.

Die separate Datei `.friday-data/n10-v1.sqlite3` verwendet eine eigene
Application-ID, Modus `0600`, ein festes Schema, append-only Trigger und einen
vollständigen Replay vor jedem Insert. Die UI auf Loopback-Port 8770 öffnet die
Datenbank ausschließlich read-only.

## 8. Autorisierte Reihenfolge

```text
self-check
→ Implementierung vollständig testen
→ Implementierung und dieses Dokument committen
→ seal --execute
→ run-stage --stage calibration --execute
→ seal-confirmation --execute
→ run-stage --stage confirmation --execute
→ read-only Replay, UI- und Hashprüfung
```

Vor dem persistierten Seal ist keine neue N10-Messung zulässig. Nach einem
terminalen Fehler oder Entscheid sind keine weiteren Records zulässig.

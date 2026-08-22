# N10-v2 — prospektive Ein-Kandidaten-Vorregistrierung

**Freigabe:** 22. August 2026

**Study-ID:** `h2n10-dispatch-confirmation-20260822-02`

**Status dieses Dokuments:** Designvertrag; eine Messung ist erst nach sauberem
Implementierungscommit und persistiertem `preregistration`-Record zulässig.

**Vorgänger:** N10-v1 ist terminal beendet und wird weder fortgesetzt noch
überschrieben. Dessen C0-Versuch stoppte vor jeder Timingmessung am korrekt
arbeitenden H0-Fixture-Guard, weil der dort eingefrorene neue Seed keine
registrierte Produktionsidentität besaß. Der unveränderte Fehlerrecord ist
`3ce4477adf3ca13d30207f37d98f21e36c316c82e3d102abfacf61c091492e49`,
der terminale DB-Hash
`e0b5f4af62c128938e1e12e388c16b344a66e18eebf9e0568c7ebe34c5a4f0d5`
und die Snapshot-Revision
`bbc75d60b5cfc61a1037c0a104e117a89561ec63e13240ca9b84f1bc98c08976`.
N10-v2 ist eine neue Study-ID mit neuer DB und frischen Seeds, kein Retry.

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
| Fixture-Seed | `4051312678` (`0xF17A2026`) |
| Fixture-A-SHA-256 | `33043be0345487a8a41b522df292e5288914b9c6c6c4dc823dbec72b9146bf86` |
| Fixture-Identität | registrierter H0-Produktionsvertrag; alle vier Digests werden fail-closed geprüft |
| Operand-Seed | `8108914365621233760` |
| Arm A | jede Operation einzeln `eval` + `synchronize` |
| Arm B | zehn Operationen enqueue, dann ein `eval` + `synchronize` |
| Korrektheit | Referenz, A und B müssen byte-identisch sein; max. Fehler `0` |

Der Fixture-Seed wird bewusst aus dem bereits kryptographisch registrierten
H0-Produktionsvertrag übernommen. Alle übrigen Seeds wurden vor Messbeginn
deterministisch aus SHA-256 über die Domain
`project-friday:h2-n10-v2:<label>` abgeleitet: Die ersten acht Digest-Bytes
werden als Big-Endian-Integer gelesen und das höchstwertige Bit gelöscht. Die
Ergebnisse sind danach als 63-Bit-Konstanten eingefroren. Die Labels lauten
`operands`, `session:C0` bis `session:V2`,
`bootstrap:calibration`, `bootstrap:characterization`,
`bootstrap:validation` und `bootstrap:all`.

Versiegelte Session-Seeds:

- `C0`: `5694182798642334346`; `V0`: `4016037479549399342`;
- `C1`: `4702616514600041353`; `V1`: `5448993668583962080`;
- `C2`: `6937834284092508076`; `V2`: `3319947694069614818`.

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

- Kalibrierung gesamt: `3420748623931472299`;
- Bestätigung Charakterisierung: `968347539867383741`;
- Bestätigung Validierung: `2471101842785840228`;
- Bestätigung gesamt: `1603501775215485335`.

## 6. Ressourcen-, Sicherheits- und Fehlervertrag

Je Session gelten maximal 120 Sekunden erfasste GPU-Arbeit, sechs Sekunden
kontinuierliche GPU-Arbeit, 25 % Duty Cycle in 60 Sekunden und 20 Minuten Wall.
Die gemeinsame BudgetGuard-Policy verlangt bei Bedarf vier Sekunden Pause und
kennt einen 60-Sekunden-Kandidaten-Cooldown. CPU-Zeit, RSS sowie verfügbare
MLX-Speicherzähler werden gespeichert.

Bereits Self-Check, Seal und jeder Session-Preflight prüfen, dass Fixture-Seed
und alle vier erwarteten Digests im H0-Produktionsvertrag registriert sind.
Seal, jeder Session-Preflight und jede Mutation eines abgeleiteten Records
(Kalibrierungszusammenfassung, Confirmation-Seal und terminaler Entscheid)
prüfen außerdem N10-v1 read-only vollständig gegen dessen terminalen DB-Hash,
Snapshot-Revision, zwei Records und null Timing-Sessions; jede Abweichung
sperrt V2.
Jeder Fehler nach bestandenem Preflight erzeugt – soweit die versiegelte
Provenienz noch unverändert ist – einen terminalen `session_failure`-Record.
Ein vorhandener stabiler Benchmark-Fehlercode wird ohne Fehlermeldung im
begrenzten Typfeld erhalten. Fehlgeschlagene Sessions werden nicht wiederholt.
Änderungen an Code, Spec, Umgebung, Hardwareidentität oder Git-Revision
schließen den Lauf fail-closed.

Nicht Bestandteil dieser Studie sind Custom Metal, generierter Code, freie
Modellaktionen, ein Kandidatensuchraum, AVO-Supervision oder eine Änderung der
produktiven Runtime. Diese Schritte benötigen nach dem Studienentscheid einen
eigenen Vertrag.

## 7. Provenienz und Persistenz

Die Vorregistrierung bindet:

- den sauberen Root-Git-Commit und einen leeren Root-Diff;
- alle Python-/SQL-Dateien unter `friday_n10_v2/`;
- den unveränderten N10-v1-Reader und dessen Vorregistrierungsdokument zur
  vollständigen Vorgängerprüfung;
- `tools/run_n10_v2.py`, den Budgetvertrag, den Fixture-Generator und dessen
  unveränderliche Korrektheitsidentitäten;
- dieses Dokument, `docs/PHASE1_MATMUL_SPEC.md`,
  `docs/H1H2_EVIDENZ_ARCHITEKTUR.md` und die eingefrorenen Requirements;
- Python- und Paketversionen sowie öffentliche Hardware-/macOS-Merkmale.

Die separate Datei `.friday-data/n10-v2.sqlite3` verwendet eine eigene
Application-ID, Modus `0600`, ein festes Schema, append-only Trigger und einen
vollständigen Replay vor jedem Insert. Die UI auf Loopback-Port 8771 öffnet die
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

## 9. Nachtrag nach terminalem Abschluss

Dieser Abschnitt entstand **nach** dem formalen Lauf und ist nicht Teil der
versiegelten Provenienz. Die exakt ausgeführte Fassung dieses Dokuments liegt im
Root-Commit `959df09b9d197edbd0a0984eda25092997b4ab23`; der dort berechnete
Provenienz-Hash lautet
`17d0dd505e349a4bbb7ffde3c291a3a44226d0fce79c235ce2ce890289e0c9ef`.

Die autorisierte Reihenfolge wurde vollständig eingehalten. Der
Präregistrierungsrecord lautet `343bbbd1…f556f94`. Die sechs A/A-Sessions
ergaben aggregiert `R=0,999586`, 95%-KI `[0,998764; 1,000443]`; die rohe MDE
war `0,0857 %`, der vorregistrierte Floor blieb `5 %`. Der Confirmation-Seal
lautet `d6402bb9…5404487` und bindet
`confirmation_seal_sha256=7ad8e461…a7813`.

Alle sechs A/B-Sessions waren byteidentisch und bestanden ihre Budgets. Der
terminale Entscheid ergab insgesamt `R=0,874912`, 95%-KI
`[0,871768; 0,875614]`, entsprechend `12,509 %` weniger Zeit. Die getrennten
Splits bestanden ebenfalls: Charakterisierung `R=0,875216`, 95%-KI
`[0,869739; 0,876217]`; Validierung `R=0,874608`, 95%-KI
`[0,871695; 0,875607]`. Record `47283e73…e1249` trägt als einziger
`formal_claim=true`, Status `n10_gain_confirmed`, und erlaubt ausschließlich
`permit_bounded_n10_runtime_prototype`.

Der terminale Store enthält 16 replaybare Records, Modus `0600`, Größe
`180.224 B`, SHA-256
`54e9c57ca6b76fa671b94f748b7ee471575b7dd7445bad00ae3cab38f691fc4f`
und Snapshot-Revision
`9c9a94a8f799f2eb29b9e03c4e1b6e681aa945199753158cf8fc8c317b06090d`.
Die read-only UI lieferte GET/HEAD `200`, wies POST mit `405` ab und ließ den
Dateihash unverändert. Ihr vollständiger Replay benötigte `3,42–3,44 s` je
Snapshot; ein manueller `Ctrl-C`-Stop beendet den Prozess zwar, zeigt derzeit
aber einen `KeyboardInterrupt` und Exit `1`. Diese UI-Lifecycle-/Latenzbefunde
werden erst außerhalb des versiegelten Studiencodes bearbeitet.

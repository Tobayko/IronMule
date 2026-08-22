# Vorregistrierung: evidenzgebundener N8/N10-Shadow-Router

Status: **vor Implementierung und vor jeder Live-Messung eingefroren**

Router-ID: `avo-shadow-router-20260822-01`

Datum: 2026-08-22

## 1. Freigabe und Ziel

Der Nutzer hat am 22.08.2026 die nächste Runde ausdrücklich freigegeben:
notwendige Software darf installiert werden, neue Modelle dürfen nicht
installiert oder heruntergeladen werden, und ein späterer Kernelpfad darf unter
hohen Sicherheitsanforderungen geprüft werden.

Dieser Vertrag umfasst ausschließlich einen getrennten, **shadow-only**
arbeitenden Router. Er verbindet die bereits formal beziehungsweise technisch
bestätigten N8- und N10-Runtime-Policies, ohne einen neuen Ausführungsplan zu
aktivieren oder die bestehenden Pakete `friday_runtime/` und
`friday_runtime_n10/` zu verändern.

## 2. Fester Scope

Der Router kennt genau drei Entscheidungen:

1. N8: FP16-`2048²`, acht rechte Operanden, bestehender versiegelter N8-Plan;
2. N10: FP16-`2048²`, zehn rechte Operanden, bestehender versiegelter N10-Plan;
3. alles andere: serieller Fallback.

Die Entscheidung wird ausschließlich aus tatsächlichen Tensor-Metadaten
abgeleitet. Vom Aufrufer gelieferte Labels, Modellnamen oder Planwünsche dürfen
niemals autorisieren. Eine optimierte Shadow-Empfehlung ist nur zulässig, wenn
**beide** zugrunde liegenden Policies ihre exakte Evidenz, Umgebung, Hardware
und aktuelle Quellidentität autorisieren. Eine teilweise verfügbare Policy
sperrt den gesamten kombinierten Router fail-closed.

Der Router führt in dieser Phase keinen optimierten Plan aus. Sein erzwungener
Ausführungsplan bleibt immer `serial_shadow_only`; er protokolliert lediglich
die Empfehlung der bestehenden Policy. Damit kann der bestehende N8- oder
N10-Runtime-Pfad nicht durch den Router verändert werden.

## 3. Geschlossene Zustandsmaschine

```text
LOAD_BOTH_EVIDENCE
  -> VERIFY_EXACT_IDENTITIES
  -> OBSERVE_REAL_TENSORS
  -> SELECT_N8 | SELECT_N10 | SERIAL
  -> COMPARE_WITH_DIRECT_POLICY
  -> RECORD_SHADOW_DECISION
  -> SERIAL_SHADOW_ONLY
```

Jede Ausnahme, unbekannte Form, unbekannter Datentyp, andere Operandenzahl,
schmutzige Git-Arbeitskopie, fremde DB, abweichende Hashkette oder
Identitätsabweichung führt zu `serial_shadow_only`. Es gibt keine freie Suche,
keine Modellaktion, keine Codegenerierung und keinen Retry einer fehlgeschlagenen
Live-Messung.

## 4. Unveränderliche Messfälle

Nach einem sauberen lokalen Implementierungscommit sind genau zwei neue Läufe
zulässig:

- `avo-router-policy-20260822-01`: gepaarter CPU-Overhead-Vergleich zwischen
  direkter N8/N10-Policywahl und Shadow-Router;
- `avo-router-shadow-20260822-01`: einmalige Prüfung mit echten MLX-Tensoren
  für N8, N10 und feste Negativfälle.

Vor dem sauberen Commit dürfen beide Befehle nur ihre Ausführungssperre testen
und keine Router-DB anlegen.

CPU-Messvertrag:

- 5 Warmup-Blöcke;
- 21 balancierte A/B-Blöcke mit alternierender Reihenfolge;
- 10.000 Entscheidungen je Arm und Block;
- N8 und N10 werden deterministisch abwechselnd geprüft;
- Baseline und Kandidat müssen in jedem Aufruf dieselbe Empfehlung liefern;
- Cold Load beider Policies `<=15 s`;
- Router-Median `<=30 µs`;
- Router-p95 `<=60 µs`;
- gepaarter zusätzlicher Median gegenüber direkter Wahl `<=15 µs`.

Shadow-Validierung:

- echte MLX-FP16-Tensoren mit Form `2048²`;
- exakt acht und exakt zehn RHS-Tensorreferenzen empfehlen den jeweils
  registrierten Batch-Plan, führen aber weiterhin nur
  `serial_shadow_only` aus;
- Operandenzahl neun, falsche Form und falscher Datentyp fallen seriell zurück;
- direkte Policy und Router müssen in allen Fällen übereinstimmen;
- keine Matmul-Ausführung und kein Modelllauf.

Ein negatives Gate ist ein gültiges terminales Engineering-Ergebnis und wird
nicht durch Wiederholung ersetzt.

## 5. Persistenz und UI

Die eigene SQLite-v1-Datei lautet `.friday-data/avo-router.sqlite3`, verwendet
Application-ID `FRR1`, Modus `0600` und eine unveränderliche Metadatenzeile.
Records werden kanonisch serialisiert, append-only hashverkettet und vor jedem
Snapshot vollständig replayt. Zulässige Record-Arten sind ausschließlich:

- `policy_overhead`;
- `shadow_validation`;
- `router_failure`.

Jeder Record trägt `formal_claim=false`, beide zugrunde liegenden
Decision-Record-IDs, Messwerte, vollständige Router-Provenienz und den
Vorgängerhash. Die read-only Loopback-UI läuft ausschließlich auf
`127.0.0.1:8773`, erlaubt GET/HEAD und weist mutierende Methoden ab.

## 6. Abbruch- und Promotionsregel

Nur wenn beide Läufe alle Gates bestehen, erhält der Router den Status
`shadow_router_validated`. Das erlaubt weiterhin keine produktive Aktivierung.
Es erlaubt lediglich die getrennte Vorregistrierung eines einzelnen statischen
Custom-Metal-Kandidaten mit isoliertem Worker, Timeout, Ressourcenlimits,
Correctness-Oracle und Rollback.

Neue Modelle, adaptive Kandidatensuche, produktive Integration, Cross-Device-
Claims, eine eigene GPU-ISA, ein Compiler oder eine neue IR bleiben außerhalb
dieses Vertrags.

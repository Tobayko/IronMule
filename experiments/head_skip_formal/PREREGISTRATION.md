# Zyklus 12 — versiegelte LM-Head-Prefill-Studie

**Study-ID:** `head-skip-prefill-v1-20260824`

**Kandidaten-ID:** `prefill-head-skip-20260824-02`

**Status:** prospektiver Designvertrag; vor dem ersten Zyklus-12-Hardwarewert
geschrieben. Eine Messung ist erst nach Implementierungscommit und persistiertem
`preregistration`-Record zulässig. `formal_claim=false` bis zum einzigen terminalen
`study_decision`-Record.

## 1. Auswahl und Abgrenzung

Zyklus 8 charakterisierte explorativ, dass der LM-Head auf allen Promptpositionen
`14,79` bis `16,05 %` des Prefills beansprucht, obwohl nur die letzte Logitzeile
gelesen wird. Diese Daten wählen den Kandidaten aus, gehen aber in keine Schätzung,
Kalibrierung oder Entscheidung dieser Studie ein.

Der Kandidat hat Vorrang vor persistentem Prozess und Readback-Bündelung, weil der
gemessene Engpass der warme Prefill ist (`~1,70 s` gegen `~12,1 ms` je
Decodetoken). Geprüft wird genau ein Kandidat und ein primärer Endpunkt.

Der Claim gilt nur für:

- Apple M1 Max, 32 GB;
- MLX `0.32.0`, mlx-lm `0.31.3`;
- die projektlokal gebundene Snapshot-Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2` von
  `mlx-community/gemma-3-4b-it-4bit`;
- greedy Decode ohne Prompt-Logprobs;
- Promptlänge `897`, Prefill-Chunk `256`, Batch `1`.

Kein Claim gilt für Logprob-/Perplexity-Pfade, andere Modelle, Quantisierungen,
Chunkgrößen, Hardware oder eine produktive Aktivierung.

## 2. Variable und Endpunkt

**Arm A — Referenz:** Jeder Prefill-Block läuft durch Modellkörper und LM-Head auf
allen Positionen; jeder Block wird wie im unveränderten Referenzpfad ausgewertet und
synchronisiert.

**Arm B — Kandidat:** Dieselben Token, Caches und Blockgrenzen laufen durch denselben
Modellkörper. Bei nichtletzten Blöcken wird nur der Hidden State ausgewertet; beim
letzten Block wird der LM-Head ausschließlich auf `hidden[:, -1:, :]` angewandt.

Der primäre Endpunkt ist pro gepaartem Block

`R = Prefilldauer(B) / Prefilldauer(A)`.

Die Dauer endet zwingend **vor** jedem Budget-`charge()` und jeder Guard-Pause.
Sekundär werden absolute Prefilldauer, Prozess-RSS und öffentliche
MLX-Speicherzähler gespeichert. Es gibt keine isolierte LM-Head-Mikromessung und
keinen daraus abgeleiteten Gewinn.

## 3. Workload

Der Promptinhalt ist die exakt im Harness eingefrorene Zeichenfolge mit SHA-256
`73675a7043bd40e61586757d8252cf1fb69bfb53b8747ff47f1c08d5fb8f69e5`.
Nach dem gebundenen Chat-Template muss er exakt `897` Token ergeben; andernfalls
fällt die Studie vor Timing geschlossen aus.

| Feld | Wert |
| :--- | ---: |
| Batch | `1` |
| Prefill-Chunk | `256` |
| Korrektheitshorizont | `32` greedy Token |
| Warmup-Paare je Session | `2` |
| Messpaare je Session | `4` |
| Sessions je Stufe | `6` |

Es werden keine Ausreißer verworfen und keine Sessions optional gestoppt.

## 4. Korrektheitsgate

Vor jedem Timing vergleicht der Prozess einen vollständigen Referenzlauf und den
Kandidatenlauf über denselben festen 32-Token-Horizont. Pflicht sind:

- exakt identische Token-IDs;
- identischer Finish-Status des festen Horizonts;
- unveränderter Prompt ohne Trunkierung;
- kein stiller Fallback und exakt der registrierte Kandidatenpfad.

Ein Mismatch erzeugt `correctness_failed_terminal`. Die Session und der Kandidat
werden nicht wiederholt; kein Zeitwert dieser Session darf in die Statistik eingehen.

## 5. Stufe 1 — A/A-Kalibrierung

Sechs getrennte Prozesse laufen in der festen Reihenfolge
`C0,V0,C1,V1,C2,V2`. Beide Timingarme sind getrennte Callables des unveränderten
Arm-A-Pfads. Jede Session enthält zwei verworfene Warmup-Paare und vier Messpaare,
je zwei `AB` und `BA` in einer vorab SHA-256-gemischten Reihenfolge.

Aus den sechs Session-Ratios wird vor A/B berechnet:

```text
raw_MDE = 2 × sd(session_ratio) × sqrt(2/3)
MDE     = max(0,05, raw_MDE)
```

A/B wird nur geöffnet, wenn:

- das hierarchische 95-%-Intervall den Nullwert `1,0` enthält;
- der Punktschätzer höchstens `5 %` Bias zeigt;
- `MDE ≤ 15 %`;
- alle sechs Korrektheitsgates halten.

MDE, Kalibrierungs- und Sessionhashes werden in einem eigenen
`confirmation_seal` gebunden.

## 6. Stufe 2 — A/B-Bestätigung

Nach dem Confirmation-Seal folgen sechs neue Prozesse in derselben Reihenfolge.
Arm A ist der Referenzpfad, Arm B der LM-Head-Skip. C-Sessions bilden die
Charakterisierung, V-Sessions die Validierung. Für C, V und alle sechs Sessions
werden getrennte deterministische hierarchische 95-%-Bootstrapintervalle mit je
`10.000` Ziehungen gebildet.

Versiegelte Seeds (unsigned 64 bit):

| Zweck | Seed |
| :--- | ---: |
| Session `C0` | `15510830734782369641` |
| Session `V0` | `13859906320662629798` |
| Session `C1` | `3290811032693642639` |
| Session `V1` | `14366515575250128902` |
| Session `C2` | `13587802099656419680` |
| Session `V2` | `12362147029480673024` |
| Bootstrap Kalibrierung | `3434287716142173047` |
| Bootstrap Charakterisierung | `16895945304681056598` |
| Bootstrap Validierung | `16493265756820087568` |
| Bootstrap gesamt | `7407874620929745004` |

## 7. Vorab festgelegte Entscheidungstabelle

Die Schranken sind `[1 − MDE, 1 + MDE]`. Alle Vergleiche sind strikt wie angegeben.

| Bedingung in C, V und gesamt | Status | Aktion |
| :--- | :--- | :--- |
| alle oberen Grenzen `< 1 − MDE` | `head_skip_gain_confirmed` | begrenzte Architekturprüfung zulässig; keine automatische Aktivierung |
| alle unteren Grenzen `> 1 + MDE` | `head_skip_regression_confirmed` | Kandidat verwerfen |
| alle drei Intervalle vollständig innerhalb der Schranken | `head_skip_equivalent_within_mde` | Kandidat verwerfen |
| sonst | `head_skip_inconclusive` | ohne Promotion stoppen |

Jeder terminale Ausgang wird akzeptiert. Schwellen, Workload und Deutung werden nach
der Messung nicht geändert.

## 8. Budgets und Ausführung

Jede Hardwarearbeit läuft über `BudgetGuard` mit einer eigenen Policy:

| Grenze | Wert |
| :--- | ---: |
| GPU-Arbeit je Prozess | `120 s` |
| kontinuierliche GPU-Arbeit | `6 s` |
| Duty-Cycle je 60-s-Fenster | `0,15` |
| Pacing-Ziel | `0,14` |
| Pflichtpause | mindestens `16 s` je erfasster GPU-Operation, in `4-s`-Schritten |
| Wall-Clock je Prozess | `1.200 s` |
| Netzbetrieb | Pflicht |

Zwischen getrennten Sessions liegen zusätzlich `20` reale Sekunden. Ein
fehlgeschlagener Hardwareprozess wird nicht erneut gestartet.

## 9. Versiegelung, Historie und UI

Die neue, isolierte Datei `.friday-data/head-skip-v1.sqlite3` besitzt eine
append-only Hashkette. Jeder Append replayt Schema, vollständige Historie,
Payload-Selbsthash und Vorgängerhash; SQL-Update und -Delete sind per Trigger
gesperrt. Bestehende Evidence-Datenbanken werden nicht geöffnet oder geändert.

Das Seal bindet sauberen Git-Commit, leeren Diff, Code-/Spec-Hashes, Python- und
Paketversionen, Hardwareidentität sowie lokale Modellrevision. Jede Session prüft
diese Provenienz erneut. Die integrierte kleine UI öffnet die Datenbank ausschließlich
read-only und zeigt Verlauf, Status, Ratios, MDE und terminale Intervalle.

Genau der terminale `study_decision`-Record trägt `formal_claim=true`; alle
vorherigen Records tragen `false`. Ein Fehlerrecord ist terminal, aber kein positiver
Performanceclaim.

## 10. Autorisierte Reihenfolge

```text
self-check
→ Implementierung und diese Vorregistrierung testen
→ beides committen
→ seal --execute
→ run-stage --stage calibration --execute
→ seal-confirmation --execute
→ run-stage --stage confirmation --execute
→ read-only verify/snapshot/UI-Prüfung
```

Vor dem Seal ist jede Zyklus-12-Hardwaremessung untersagt. Nach terminalem Record
sind keine weiteren Appends zulässig.

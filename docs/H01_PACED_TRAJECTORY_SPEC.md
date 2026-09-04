# H0.1 — Replicated Paced Engineering Envelope, Design A v2

Status: vorregistriert vor der ersten H0.1-Live-Datenerhebung. Design A v1 ist
wegen unzureichender Result-Replay-Bindung, indexbasierter Zeitachse und einer
nicht gerechtfertigten Einzelsession-/Bootstrap-Interpretation verworfen. Es
existieren keine v1-Live-Daten. v2 ist ein neuer, deterministischer
Engineering-Envelope-Vertrag und kein nachträgliches Grenzwerttuning.

Diese Studie ändert weder H0, dessen Messvertrag, historische Ergebnisse noch
SQLite v1. Kein H0.1-Status darf einen H0-Lauf reklassifizieren, eine Promotion
auslösen oder eine H0-Schwelle verändern.

## Forschungsfrage und Aussagegrenze

Design A v2 prüft ausschließlich, ob Laufzeittrajektorien einer festen Operation
unter zwei vorgegebenen Pacing-Abständen in sechs vollständig replizierten,
provenienzgleichen Sessions innerhalb eines deterministischen Engineering-
Envelopes liegen. Pro Session werden Metriken und Gates nur charakterisiert; ein
gültiges Session-Resultat heißt immer `h01_session_complete`, unabhängig davon,
ob Gates bestehen. Eine Stationaritätsaussage entsteht ausschließlich auf
Study-Ebene und nur, wenn alle Gates aller sechs Sessions bestehen.

Es werden keine p-Werte, keine `p_cp`-Größe, kein Bootstrap-Konfidenzintervall und
keine probabilistische Inferenz berechnet. Der maximale Median-Changepoint bleibt
eine deterministische Effektgröße mit festem Engineering-Gate.

## Identität, Replikation und Provenienz

Es gibt exakt sechs getrennte Prozesse in exakt dieser Reihenfolge:
`C0, V0, C1, V1, C2, V2`. `C` und `V` sind ausschließlich vorregistrierte
Replikationslabels; sie besitzen keine asymmetrische wissenschaftliche Bedeutung.
Fehlende, doppelte, vertauschte oder selektiv abgebrochene Sessions machen die
Study ungültig.

Jede Session besitzt einen festen, disjunkten signed-Int64-Schedule-Seed. Der
Scheduler bleibt `sha256_fisher_yates_v1`: SHA-256-Counterblöcke, unbiased
Rejection Sampling und Fisher-Yates. Python `random`, NumPy und MLX gehören nicht
zum Core-Vertrag.

Jedes Manifest bindet kanonisch:

- Study-Spec-SHA-256 und Code-SHA-256,
- Environment-SHA-256,
- alle Fixture-Komponenten samt Fixture-Aggregat,
- unveränderte H0-Parent-Lineage,
- Sessionidentität, vollständigen Schedule und feste Budgets/Gates.

Die Study akzeptiert nur sechs Manifeste mit identischen Spec-, Code-,
Environment-, Fixture- und Parent-Lineage-Werten.

## Fester Ablauf und reale Pacing-Zeit

- 32 aufgezeichnete Burn-in-Samples in acht Viererblöcken.
- Danach exakt 20 Sekunden angeforderter Cooldown; beobachtet mindestens 20
  Sekunden.
- 80 Main-Samples in zwanzig Viererblöcken.
- Jeder Viererblock enthält exakt zweimal `short_50ms` und zweimal
  `long_750ms`; die Reihenfolge ist materialisiert und gehasht.
- Kein adaptiver Stopp, keine Budgetanpassung, keine Ausreißerlöschung und keine
  nachträgliche Sampleauswahl.

Jedes Sample speichert `requested_gap_ns`, `gap_start_ns`, `gap_end_ns`,
`start_ns` und `duration_ns`. `actual_gap_ns` wird ausschließlich als
`gap_end_ns - gap_start_ns` abgeleitet und nicht redundant gespeichert.
`gap_end_ns == start_ns` gilt exakt. Ab Sample 1 beginnt der Gap exakt am Ende
des vorherigen Samples; beim Burn-in/Main-Übergang beginnt er exakt nach dem
beobachteten Cooldown. Für das erste Sample wird der Gap rückwärts vom Start
gebunden.

Pacing-Adherence gilt nur bei
`requested_gap_ns <= actual_gap_ns <= requested_gap_ns + 250_000_000 ns`.
`250 ms` ist vor Live-Daten als konservative obere Scheduling-Overshoot-Grenze
registriert. Jede größere Zusatzpause, negative/rebound Zeitfolge, inkonsistente
Gap-Grenze oder falsche Pacing-Bindung macht die Session `h01_invalid`; Labels
werden außerhalb dieser Adherence niemals analytisch verwendet.

## Feste Budgets

| Größe | Wert |
| --- | ---: |
| Schema | 2 |
| Burn-in-Samples | 32 |
| Cooldown | 20.000.000.000 ns |
| Main-Samples | 80 |
| Main-Blöcke | 20 |
| Samples je Block | 4 |
| Short-Pacing | 50.000.000 ns |
| Long-Pacing | 750.000.000 ns |
| Maximaler Gap-Overshoot | 250.000.000 ns |
| Changepoint-Splits | 8 bis 72 einschließlich |
| ACF-Lags | 1 bis 4 |
| Sessions | 6, exakt `C0,V0,C1,V1,C2,V2` |

## Deterministische Session-Analyse

Nur die 80 Main-Samples gehen in Metriken ein; alle 112 Samples bleiben in der
Bilanz. Dauern werden natürlich logarithmiert. Innerhalb jedes Pacing-Stratums
wird der Median der Log-Dauern abgezogen. Ohne Löschung oder Winsorization werden
berechnet:

1. Theil-Sen-Slope der Residuen gegen die reale Zeitachse
   `(start_ns - first_main_start_ns) / 1e9`. Der Trajektorieneffekt ist
   `exp(slope_per_second * observed_main_span_seconds) - 1`.
2. Maximaler absoluter Median-Changepoint-Effekt über Splits `8..72`.
3. Spearman-ACF der Residuen für Lags `1..4` und
   `ESS = 80 / (1 + 2 * sum(max(0, rho_lag)))`, begrenzt auf `1..80`.
4. Pacing-Effekt als `exp(median(log long) - median(log short)) - 1`.
5. Tail-Ratio als Maximum der pacing-normalisierten Dauer geteilt durch deren
   Median.
6. SHA-256 der pacing-stratifizierten Residuen und der 80 abgeleiteten realen
   Main-Gaps sowie maximaler realer Gap-Overshoot als Replay-Evidenz.

## Feste Session-Gates

Alle Vergleiche schließen die Grenze ein.

| Gate | PASS-Bedingung |
| --- | --- |
| Trend | `abs(Theil-Sen-Trajektorieneffekt) <= 0,05` |
| Changepoint | `abs(maximaler Median-Effekt) <= 0,05` |
| ACF | `max(abs(rho_1..rho_4)) <= 0,50` |
| ESS | `ESS >= 40` |
| Pacing | `abs(Pacing-Effekt) <= 0,03` |
| Tail | `Tail-Ratio <= 1,20` |

Ein gültiges vollständiges Session-Resultat trägt unabhängig von den Gates
`h01_session_complete` und `session_characterized`. Ein Schema-, Pacing-, Hash-,
Correctness-, Reihenfolge-, Zeit-, Int64- oder Endlichkeitsfehler trägt
`h01_invalid` und keine Teilmetriken.

## Study-Level-Entscheidung

Die Study validiert und replayt Manifest, Trace und Resultat jeder Session selbst.
Gelieferte Metriken, Gates, Status und Decision-Hash werden nicht vertraut:
`analyze_trace` wird aus unverändertem Manifest und Trace erneut ausgeführt und
das vollständige kanonische Resultat muss bytegleich sein. Eine gemeinsam
veränderte Metrik-/Gate-/Status-/Decision-Hülle wird verworfen.

Exakte terminale Study-Statuswerte:

- `h01_stationarity_supported`: genau sechs gültige vollständige Sessions und
  jedes Gate jeder Session ist `pass`.
- `h01_complete_unresolved`: genau sechs gültige vollständige Sessions, aber
  mindestens ein Gate ist `fail`.
- `h01_invalid`: irgendein Contract-, Safety-, Correctness-, Provenienz-,
  Vollständigkeits-, Reihenfolge- oder Replayfehler.

Andere Study-Statuswerte sind unzulässig. Study und Sessions verwenden stets `no_h0_conclusion`;
`h0_reclassification` und `promotion_applicable` sind exakt `false`.

## Telemetrie, Grenzen und Änderungsregel

Thermal- und Power-Telemetrie sind rein beschreibend. Fehlende Werte sind nur als
geschlossenes XOR-Paar `{value: null, missing_reason: <registrierter Grund>}`
zulässig; `0` ist kein Ersatz. Der stdlib-only Core importiert weder NumPy noch
MLX und führt keine Hardwarearbeit aus. Worker, Storage, Dashboard und Live-
Ausführung gehören nicht zu dieser Implementierungsphase.

Schwellen, Seeds, Samplezahl, Pacing, Overshoot-Grenze, Statussemantik oder
Replikationszahl dürfen nach Sichtung von Daten nicht verändert werden. Jede
spätere Designänderung benötigt neue Spezifikation, neue Provenienz und neue
Run-/Study-IDs.

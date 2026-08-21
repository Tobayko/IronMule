# Forschungsentscheid nach Evidenzaudit — 21.08.2026

## Entscheid

| Pfad | Aktueller Entscheid | Begründung | Bedingung für erneute Prüfung |
| --- | --- | --- | --- |
| Begrenzte evidenzgebundene Runtime | **GO im exakten H1-Scope** | formales H1-v2-Gain-Gate sowie CPU-Overhead- und MLX/GPU-Runtime-Gates bestanden; Korrektheit byte-identisch | jede Evidenz-, Code-, Spec-, Environment-, Hardware- oder Workload-Abweichung fällt weiterhin seriell zurück |
| Geschlossener H2-Modellvorschlag | **GO für eine explorative Runde** | Nutzer hat vorhandenes Gemma freigegeben; das Modell darf nur bis zu drei Integer aus `2..16` vorschlagen, Harness und Guard entscheiden allein | lokaler validierter Snapshot, kein Netzwerk, kein Code, keine freie Ausführung, `formal_claim=false` |
| Phase 1B / Custom MLX-Metal | **NO-GO** | formales H1 bestätigt nur Dispatch-Amortisation; reale Roofline deutet weiter auf Speicherlimit, und eine separate Custom-Kernel-Sicherheits-/Architekturfreigabe fehlt | freigegebene isolierte Worker-/Rollback-Architektur, eigener prospektiver Vertrag und neuer expliziter Nutzerentscheid |
| Cross-Device | **NO-CLAIM / derzeit blockiert** | es existiert Evidenz von genau einem M1 Max; ein zweites Zielgerät ist nicht verfügbar | mindestens ein unabhängiges Gerät, identischer versiegelter Workload und vollständige Provenienz |
| breiterer Live-Suchraum | **NO-GO** | mehr Kandidaten würden den explorativen Charakter und Multiple-Testing-/Winner's-Curse-Risiken vergrößern | neue H1-Vorregistrierung mit Familien/Splits, Kandidatenbudget, Powerplanung und frischen IDs |
| Offline-Protokoll-/Testarbeit | **GO** | verändert keine Hardwareevidenz und schließt bekannte Reproduzierbarkeitslücken | weiterhin keine stillschweigende Freigabe eines Live-Laufs |

## Evidenzbasis

1. H0 ist technisch persistent, aber die aktuelle DB enthält `9` `aa_gpu`-Runs:
   drei ältere Charakterisierungen und eine spätere 3+3-Generation. Mindestens ein
   Prozess der relevanten Kalibrierung war `warmup_unstable`; der registrierte
   Loader verlangt global exakt sechs Runs und kann die append-only DB daher nicht
   vertragskonform aggregieren.
2. Das Arbeitsjournal hält unmittelbar vor dem ersten A/B-Lauf fest: kein formales
   A/A-Gate, kein hierarchisches Bootstrap und `MDE` noch nicht eingefroren. Die
   anschließend verwendeten `5 %` sind deshalb eine explorative Schwelle und
   können nicht rückwirkend vorregistriert werden.
3. Dispatch-, Loop- und H2-Codegen-Effekte sind intern gepaart, repliziert und
   correctness-geprüft. Sie bleiben wertvolle technische Beobachtungen, sind aber
   kein formaler H1/H2-Nachweis nach `PHASE1_MATMUL_SPEC.md` Abschnitt 8.1/9.1.
4. Die Roofline-Zusammenfassung zeigt auf dem beobachteten Gerät deutlich höhere
   Bandbreiten- als Rechenauslastung. Das begrenzt die Motivation für einen
   reinen „näher an der ISA“-Pfad, beweist aber keine allgemeine Hardwareaussage.
5. Die Fusionsprüfung liefert ein belastbares Negativergebnis für den realen
   Generierungspfad: Der cache-freie Forward-Pass ist nicht der produktive
   KV-Cache-Pfad; End-to-End blieben nur `−0,5 %`/`−0,1 %`.
6. H1-v2 wurde anschließend prospektiv auf Commit `1fbe73c` geschlossen. Sechs
   A/A-Sessions bestimmten den konservativen MDE-Floor `5 %`; sechs frische
   A/B-Sessions bestätigten byte-identisch `R=0,879718`, 95%-Intervall
   `[0,877045; 0,880403]`, mit bestandenen Charakterisierungs- und
   Validierungssplits. Nur der terminale Record trägt `formal_claim=true`.
7. Der daraus erlaubte begrenzte Runtime-Prototyp bestand auf Commit `0b0a893`
   sein CPU-Gate (`11,045 µs` Policy-Median) und sein gepaartes GPU-Gate
   (`R=0,879209`, `−12,079 %`, byte-identisch). Die zwei Engineering-Records
   sind getrennt hashverkettet; sie erweitern den formalen Claim nicht.

## Nächster kleinster wissenschaftlich zulässiger Schritt

Nicht Phase 1B und kein freier Suchraum. Der nächste kleinste Schritt ist genau
**eine** explorative H2-Runde mit dem bereits lokal validierten Gemma 3 4B:

- Das Modell sieht nur den geschlossenen FP16-`2048²`-Dispatch-Kontext und darf
  höchstens drei noch ungetestete Batchgrößen als Integer aus `2..16` nennen.
- Prosa, Code, Shelltext, Floats, Boolwerte, Duplikate und Werte außerhalb der
  Allowlist werden verworfen und erhalten keine Ausführungswirkung.
- Der deterministische Harness prüft Korrektheit, misst die bekannten seriellen
  und gebatchten Pläne unter BudgetGuard und bestätigt einen Leader separat.
- Der vorhandene Snapshot wird per lokalem Ref, Revision, Tokenizer und real
  gelesenen MLX-Gewichten validiert. Offline-Umgebungsvariablen schließen einen
  Netzwerkfallback zusätzlich aus. Es gibt keinen Download und keine Installation.
- Der Bericht wird mit Rohdaten in SQLite v1 gespeichert und bleibt ausdrücklich
  `formal_claim=false`; er kann H1 oder die Runtime-Freigabe weder ändern noch
  auf andere Shapes, Geräte oder Modelle übertragen.

Ein unbrauchbarer Modelloutput oder ein negativer Kandidat ist ein gültiges
Ergebnis. Weitere Runden, Custom Metal oder Modellcode erfordern danach einen
neuen Entscheid.

## Umsetzungsstand des Audits

Die freigegebene Offline-Arbeit ist abgeschlossen: Root-Provenienz, gemeinsame
H1/H2-Budgets, SQLite-v1-Persistenz, expliziter Legacy-Import und read-only
Historien-UI sind implementiert. Nach einer späteren ausdrücklichen
Rechenfreigabe enthält die produktive Research-DB zusätzlich drei native
Schema-v1-Ereignisse: einen Dispatch-Rohbericht, einen sanitisierten
Roofline-Guard-Abbruch und einen erfolgreichen Gemma-1B/4B-Rohbericht. Die
vollständige Suite bestand nach H1-v2 und Runtime-Prototyp mit `468` Tests und
`2.463` Subtests. Der formale H1-Store enthält 16 Records; die getrennte
Runtime-Historie zwei bestandene, hashverkettete Engineering-Records.

Die neuen Messungen sind durch Schema v1 ausdrücklich `formal_claim=false`. Der
Dispatch-Befund liegt explorativ jenseits der 5%-Schwelle; die neue Roofline-
Messung klassifiziert beide vorhandenen Gemma-Modelle erneut als
speicherbegrenzt. Der formale H1- und der bestandene Runtime-Befund ändern
deshalb nur den eng begrenzten Runtime-/H2-Vorschlagspfad; Phase 1B,
Cross-Device und breiterer Live-Suchraum bleiben NO-GO/NO-CLAIM.

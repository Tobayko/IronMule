# Forschungsentscheid nach Evidenzaudit — 21.08.2026

## Entscheid

| Pfad | Aktueller Entscheid | Begründung | Bedingung für erneute Prüfung |
| --- | --- | --- | --- |
| Begrenzte evidenzgebundene Runtime | **GO im exakten H1-Scope** | formales H1-v2-Gain-Gate sowie CPU-Overhead- und MLX/GPU-Runtime-Gates bestanden; Korrektheit byte-identisch | jede Evidenz-, Code-, Spec-, Environment-, Hardware- oder Workload-Abweichung fällt weiterhin seriell zurück |
| Geschlossener H2-Modellvorschlag | **eine Runde abgeschlossen; kein weiterer Lauf** | Gemma schlug `3,10,16` vor; Harness bestätigte explorativ `N=10`, aber Kandidatenselektion und Ergebnis sind Schema v1 mit `formal_claim=false` | vor weiterer Runde oder Runtime-Erweiterung neue prospektive Studie und explizite Architekturfreigabe |
| Prospektive N10-Bestätigung | **V1 terminal; V2-Gain formal bestätigt** | V1 stoppte vor Timing; V2 lief auf sauberem Commit mit registrierter Fixture, frischen Seeds, sechs A/A- und sechs A/B-Sessions bis `n10_gain_confirmed` | genau ein begrenzter N10-Runtime-Prototyp mit fester Allowlist und eigenen Engineering-Gates; keine direkte Produktivänderung |
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
8. Genau eine offline erzwungene H2-Runde auf Commit `99267d3` ließ Gemma 3 4B
   die Batchgrößen `3,10,16` vorschlagen. Der Harness wählte `N=10`; drei frische
   Bestätigungsreplikate ergaben `R=0,671573`, 95%-Intervall
   `[0,648895; 0,731190]`. Der persistierte Bericht enthält Rohdaten, bleibt aber
   wegen Modellselektion und Schema v1 ausdrücklich `formal_claim=false`.
9. N10-v2 wurde anschließend prospektiv auf Commit `959df09` geschlossen.
   Sechs A/A-Sessions ergaben `R=0,999586`, 95%-Intervall
   `[0,998764; 1,000443]`; sechs frische A/B-Sessions bestätigten
   byteidentisch `R=0,874912`, 95%-Intervall `[0,871768; 0,875614]`.
   Charakterisierung und Validierung bestanden die 5%-Gewinngrenze getrennt.
   Der terminale Record `47283e73…e1249` ist der einzige formale Claim.

## Freigegebener nächster wissenschaftlicher Schritt

Nicht Phase 1B, keine zweite Modellrunde und keine direkte Änderung der
bestehenden Runtime. Die N10-Ein-Kandidaten-Studie ist positiv terminal.
Freigegeben ist jetzt genau ein getrennter N10-Runtime-/AVO-lite-Prototyp:

- feste Allowlist für exakt FP16-`2048²`, zehn Matmuls und den bestätigten
  Batch-Dispatch-Plan;
- exakte Bindung an den 16-Record-N10-v2-Store und dessen terminalen Claim;
- serieller Fallback, Circuit Breaker, read-only Historie und eigene
  Baseline-/Nachher-Gates für Policy-Overhead, GPU-Effekt und Korrektheit;
- kein freier Kandidatensuchraum, keine weitere Modellaktion, keine
  Codegenerierung und kein Custom Metal.

Die bestehende N8-Runtime bleibt unverändert, bis der getrennte N10-Prototyp
seine eigenen Engineering-Gates bestanden hat.

Die explizite Nutzerfreigabe für Vertrag, lokale CPU-/GPU-Tests und Ausführung
liegt seit 22.08.2026 vor. Commit, Seal und terminale Bestätigung sind
abgeschlossen. Weitere Modellrunden, Custom Metal, freie Codegenerierung und
ein breiterer Suchraum bleiben NO-GO.

## Umsetzungsstand des Audits

Die freigegebene Offline-Arbeit ist abgeschlossen: Root-Provenienz, gemeinsame
H1/H2-Budgets, SQLite-v1-Persistenz, expliziter Legacy-Import und read-only
Historien-UI sind implementiert. Nach einer späteren ausdrücklichen
Rechenfreigabe enthält die produktive Research-DB zusätzlich vier native
Schema-v1-Ereignisse: einen Dispatch-Rohbericht, einen sanitisierten
Roofline-Guard-Abbruch, einen erfolgreichen Gemma-1B/4B-Roofline-Bericht und den
einrundigen H2-Modellloop. Die
vollständige Suite bestand nach H1-v2 und Runtime-Prototyp mit `468` Tests und
`2.463` Subtests. Der formale H1-Store enthält 16 Records; die getrennte
Runtime-Historie zwei bestandene, hashverkettete Engineering-Records. Die
Research-DB umfasst 14 verifizierte Zeilen und ist read-only replaybar.

Der prospektive N10-v1-Vertrag wurde auf Commit `c3e582c` versiegelt. C0
stoppte terminal mit `BenchmarkError`, bevor eine Timing-Session oder ein
formaler Claim entstehen konnte: Der neue Fixture-Seed war im wiederverwendeten
H0-Produktionsvertrag nicht registriert. Die V1-DB bleibt mit zwei Records
unverändert. N10-v2 verwendet eine bereits registrierte Fixture-Identität,
frische übrige Seeds sowie eine read-only Bindung an den exakten V1-Endzustand.
Der echte `2048²`-CPU-Fixture-Replay, `22` fokussierte Tests und die vollständige
Suite mit `508` Tests und `2.480` Subtests bestanden. V2 lief danach auf Commit
`959df09` bis zum positiven terminalen Entscheid durch: `R=0,874912`, 95%-KI
`[0,871768; 0,875614]`, byteidentisch, formaler Claim genau einmal. Der
16-Record-Store hat SHA-256 `54e9c57c…fc4f`; die N8-Runtime wurde dabei nicht
verändert.

Die neuen Messungen sind durch Schema v1 ausdrücklich `formal_claim=false`. Der
Dispatch-Befund liegt explorativ jenseits der 5%-Schwelle; die neue Roofline-
Messung klassifiziert beide vorhandenen Gemma-Modelle erneut als
speicherbegrenzt. Der formale H1- und der bestandene Runtime-Befund ändern
deshalb nur den eng begrenzten Runtime-/H2-Vorschlagspfad; Phase 1B,
Cross-Device und breiterer Live-Suchraum bleiben NO-GO/NO-CLAIM.

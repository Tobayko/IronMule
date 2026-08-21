# Forschungsentscheid nach Evidenzaudit — 21.08.2026

## Entscheid

| Pfad | Aktueller Entscheid | Begründung | Bedingung für erneute Prüfung |
| --- | --- | --- | --- |
| Begrenzte evidenzgebundene Runtime | **GO im exakten H1-Scope** | formales H1-v2-Gain-Gate sowie CPU-Overhead- und MLX/GPU-Runtime-Gates bestanden; Korrektheit byte-identisch | jede Evidenz-, Code-, Spec-, Environment-, Hardware- oder Workload-Abweichung fällt weiterhin seriell zurück |
| Geschlossener H2-Modellvorschlag | **eine Runde abgeschlossen; kein weiterer Lauf** | Gemma schlug `3,10,16` vor; Harness bestätigte explorativ `N=10`, aber Kandidatenselektion und Ergebnis sind Schema v1 mit `formal_claim=false` | vor weiterer Runde oder Runtime-Erweiterung neue prospektive Studie und explizite Architekturfreigabe |
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

## Nächster kleinster wissenschaftlich zulässiger Schritt

Nicht Phase 1B, keine zweite Modellrunde und keine direkte Runtime-Erweiterung.
Der nächste wissenschaftlich saubere Schritt wäre eine neue prospektive
Ein-Kandidaten-Studie für `N=10`:

- `N=10` wird als einziger aus Vorwissen ausgewählter Kandidat vor frischen Daten
  eingefroren; Gemma nimmt an dieser Bestätigung nicht mehr teil.
- Eine neue Study-ID trennt Modellselektion, Charakterisierung und Validierung;
  MDE, Splits, Seeds, Armreihenfolge, Cooldowns, Budgets und terminale Fehler
  werden vorab versiegelt.
- Frische A/A- und A/B-Daten müssen Korrektheit und Effekt unabhängig von der
  hier beobachteten Winner's-Curse-/Multiple-Testing-Auswahl bestätigen.
- Erst ein positiver terminaler Entscheid dürfte eine **eigene** N=10-Policy
  autorisieren. Die bestehende N=8-Runtime bleibt unverändert und fällt für
  N=10 weiterhin seriell zurück.

Dieser neue Architektur-/Studienvertrag benötigt vor Implementierung eine
explizite Nutzerfreigabe. Weitere Runden, Custom Metal und Modellcode bleiben
bis dahin NO-GO.

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

Die neuen Messungen sind durch Schema v1 ausdrücklich `formal_claim=false`. Der
Dispatch-Befund liegt explorativ jenseits der 5%-Schwelle; die neue Roofline-
Messung klassifiziert beide vorhandenen Gemma-Modelle erneut als
speicherbegrenzt. Der formale H1- und der bestandene Runtime-Befund ändern
deshalb nur den eng begrenzten Runtime-/H2-Vorschlagspfad; Phase 1B,
Cross-Device und breiterer Live-Suchraum bleiben NO-GO/NO-CLAIM.

# Forschungsentscheid nach Evidenzaudit — 21.08.2026

## Entscheid

| Pfad | Aktueller Entscheid | Begründung | Bedingung für erneute Prüfung |
| --- | --- | --- | --- |
| Phase 1B / Custom MLX-Metal | **NO-GO** | separate Sicherheits-/Architekturfreigabe fehlt; formaler H1-Unterbau ist nicht geschlossen; Roofline deutet bei realer Inferenz auf Speicher- statt Rechenlimit | freigegebene Worker-/Rollback-Architektur, reproduzierbare Baseline, neuer expliziter Nutzerentscheid |
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

## Nächster kleinster wissenschaftlich zulässiger Schritt

Nicht Phase 1B und nicht mehr Kandidaten. Zuerst ist eine neue, prospektive
H1-Studie für weiterhin genau **eine** Tensoroperation zu registrieren. Sie muss
vor jeder GPU-Zeit, die formalen H1-Phasenfortschritt begründen soll, festlegen:

- welche A/A-Generation aus einer neuen, konfliktfreien Study-ID stammt;
- wie `MDE`, Kandidatenzahl und Familien/Splits aus der Kalibrierung abgeleitet
  und anschließend versiegelt werden;
- welche Rohdaten in `.friday-data/research.sqlite3` landen;
- welche Abbruch-, Duty-, Cooldown- und Correctness-Gates gelten;
- welche Aussage bei negativem oder nicht reproduzierbarem Ergebnis zulässig ist.

Erst nach dieser Schließung kann ein neuer Live-Lauf, der formalen H1-
Phasenfortschritt begründen soll, separat angekündigt und freigegeben werden.
Dieser Entscheid selbst führte keinen GPU-, Modell-, Worker-, Download- oder
Installationslauf aus.

## Umsetzungsstand des Audits

Die freigegebene Offline-Arbeit ist abgeschlossen: Root-Provenienz, gemeinsame
H1/H2-Budgets, SQLite-v1-Persistenz, expliziter Legacy-Import und read-only
Historien-UI sind implementiert. Nach einer späteren ausdrücklichen
Rechenfreigabe enthält die produktive Research-DB zusätzlich drei native
Schema-v1-Ereignisse: einen Dispatch-Rohbericht, einen sanitisierten
Roofline-Guard-Abbruch und einen erfolgreichen Gemma-1B/4B-Rohbericht. Die
vollständige Suite bestand danach mit `439` Tests und `2.447` Subtests.

Die neuen Messungen sind durch Schema v1 ausdrücklich `formal_claim=false`. Der
Dispatch-Befund liegt explorativ jenseits der 5%-Schwelle; die neue Roofline-
Messung klassifiziert beide vorhandenen Gemma-Modelle erneut als
speicherbegrenzt. Damit steigt die technische Evidenzqualität, aber keiner der
obigen NO-GO-/NO-CLAIM-Entscheide ändert sich.

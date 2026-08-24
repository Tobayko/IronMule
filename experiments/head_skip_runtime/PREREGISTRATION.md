# Mini-Vorregistrierung: Head-Skip-Runtime-Qualifikation

Status: **vor Implementierung der Live-Messung und vor jeder Runtime-Hardwaredatei
eingefroren**

Datum: 2026-08-24

Qualification-ID: `head-skip-runtime-qualification-20260824-01`

Runtime-ID: `head-skip-runtime-20260824-01`

`formal_claim=false`: Diese Qualifikation prüft nur den freigegebenen Einbau des
bereits formal bestätigten Kandidaten. Sie ist kein Zyklus 13 und erweitert den
Einzelworkload-Claim nicht.

## Hypothesen

- **H1 Korrektheit:** Baseline und eingebauter schneller Pfad erzeugen bei greedy
  exakt dieselben 32 Token; der schnelle Pfad wird nachweislich ausgeführt.
- **H2 Auswahl:** Nur der exakt registrierte Request wählt den schnellen Pfad. Jede
  einzelne Scopeabweichung, fehlende Evidenz und ein verriegelter Circuit Breaker
  wählen die Baseline.
- **H3 Einbauwirkung:** Der Median des gepaarten Zeitquotienten
  `schnell/bisherig` liegt bei höchstens `0,95`.
- **H4 Steuerungsaufwand:** Die gecachte Entscheidung kostet im Median höchstens
  `25.000 ns`, im p95 höchstens `50.000 ns` und gepaart höchstens `20.000 ns`
  zusätzlich gegenüber einem direkten Planabruf. Der einmalige Evidenzload bleibt
  unter `5 s`.
- **H5 Ressourcen:** Der schnelle Pfad benötigt höchstens `128 MiB` mehr gemeldeten
  MLX-Peak als die Baseline; Budget, Netzbetrieb und Prozessgrenzen halten.

## Workload

- lokaler Snapshot `mlx-community/gemma-3-4b-it-4bit`, Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`;
- Promptinhalt-SHA
  `73675a7043bd40e61586757d8252cf1fb69bfb53b8747ff47f1c08d5fb8f69e5`;
- `897` gerenderte Prompt-Token, Chunk `256`, Batch `1`;
- greedy, Temperatur `0`, keine Prompt-Logprobs;
- feste Ausgabe von `32` Token ohne vorzeitigen Stopp;
- A: voller LM-Head für jeden Prefill-Block;
- B: LM-Head nur für letzte Position des letzten Prefill-Blocks.

## Stufe 1: Offline- und CPU-Gates

Vor MLX/GPU müssen bestehen:

- vollständiger Replay der unveränderten 16-Record-Studienhistorie;
- exakte DB-, Decision-, Preregistration-, Modell-, Software- und
  Hardwareidentität;
- Scope-, Rückfall-, Circuit-Breaker-, Persistenz-, UI- und CLI-Tests;
- fünf CPU-Warmup-Blöcke;
- 21 CPU-Messblöcke mit alternierender A/B- und B/A-Reihenfolge;
- 20.000 Entscheidungen je Arm und Block;
- alle H2- und H4-Schwellen.

Ein CPU-Gatefehler beendet die Qualifikation als `runtime_disabled`; dann wird kein
GPU-Lauf gestartet.

## Stufe 2: genau ein kontrollierter MLX/GPU-Lauf

Der Lauf verwendet einen frischen Prozess. Reihenfolge:

1. ein ungemessenes Referenz-/Kandidatenpaar für H1;
2. ein ungemessenes Warmup-Paar in Reihenfolge B/A;
3. vier Messpaare in fester Reihenfolge A/B, B/A, A/B, B/A;
4. Median und MAD je Arm sowie Median der vier gepaarten B/A-Quotienten;
5. Speicher-, RSS-, Prozesszeit- und `BudgetGuard`-Zusammenfassung.

Jede einzelne Prefill-/Decode-Dauer endet vor dem zugehörigen `charge()`. Für den
Duty-Faktor `0,15` wird pro aufgezeichneter GPU-Operation mindestens bis zu einer
Gesamtperiode von `Dauer / 0,14` gepaced; dadurch werden Guard-Ruhezeiten nicht in
den Endpunkt gemessen.

Es werden keine Ausreißer verworfen. Ein fehlgeschlagener Hardwarelauf wird weder im
selben noch in einem neuen Prozess wiederholt.

## Endpunkte und Rechnungen

Gemessen und unverändert gespeichert werden:

- Rohdauer A/B jedes Messpaars;
- erzeugte Token beider Correctness-Arme;
- tatsächliche Pfadmarker;
- MLX-Peak, aktiver/cache-Speicher, RSS und Prozesszeit;
- vollständige Guard-Zusammenfassung.

Daraus werden berechnet:

- `ratio_i = B_ns / A_ns` je Paar;
- primär `median(ratio_i)`;
- Armmediane, MAD und prozentuale Änderung `100 × (R − 1)`;
- Speicher-Delta B minus A.

## Vorab festgelegte Entscheidungstabelle

| Korrektheit H1 | Auswahl H2 | Wirkung H3 | CPU H4 | Ressourcen H5 | Entscheidung |
| :---: | :---: | :---: | :---: | :---: | :--- |
| hält | hält | hält | hält | hält | `engineering_go_exact_scope` |
| **verfehlt** | beliebig | beliebig | beliebig | beliebig | `correctness_failed_terminal` |
| hält | **verfehlt** | beliebig | beliebig | beliebig | `runtime_disabled` |
| hält | hält | **verfehlt** | beliebig | beliebig | `baseline_fallback` |
| hält | hält | hält | **verfehlt** | beliebig | `baseline_fallback` |
| hält | hält | hält | hält | **verfehlt** | `baseline_fallback` |

Nur `engineering_go_exact_scope` erlaubt die Nutzung über den getrennten
Repository-Aufrufpunkt und nur im registrierten Scope. Jede andere Zeile lässt den
Referenzpfad aktiv. Kein Ergebnis erzeugt einen neuen formalen Claim.

## Abbruchregeln

- fehlender Netzbetrieb oder Budgetverstoß;
- veränderte versiegelte Evidenz, Spec, Code oder Modellrevision;
- Tokenmismatch oder nicht exakt ausgeübter Kandidatenpfad;
- nicht endliche oder nicht positive Dauer;
- Swap-Wachstum, Prozessfehler oder Persistenzfehler;
- jede Änderung am Workload nach dieser Vorregistrierung.

Schwellen und Entscheidungstabelle werden nach dem ersten Messdatum nicht geändert.

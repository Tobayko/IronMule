# H1-v2 — formale Vorregistrierung der Dispatch-Studie

**Status:** freigegeben, prospektiv und vor dem ersten H1-v2-Messwert versiegelt.
**Study-ID:** `h1v2-dispatch-n8-20260821-01`
**Freigabe:** Nutzerauftrag vom 21.08.2026, den vorgeschlagenen formalen H1-v2-Pfad
umzusetzen und weiter zu testen. Diese Freigabe umfasst die Architekturänderung und
die bereits erlaubte Nutzung von CPU/GPU; sie umfasst keinen Download und keine
Installation.

## Forschungsfrage und Grenze

Auf genau einem vorhandenen Apple-Silicon-Gerät wird geprüft, ob acht unabhängige
FP16-Matmuls mit gemeinsamem linken Operand schneller ausgeführt werden, wenn alle
Operationen vor einem einzigen abschließenden `eval`/`synchronize` eingereiht werden,
anstatt nach jeder Operation zu synchronisieren.

Der einzige primäre Endpunkt ist das gepaarte Verhältnis
`B_ns / A_ns`. Es gibt genau einen Kandidaten und genau einen Workload:

- Shape `2048 × 2048`, FP16;
- acht rechte Operanden, feste PCG64-Seeds;
- A: `serial_per_op_eval_and_sync`;
- B: `enqueue_all_then_single_eval_and_sync`;
- keine Kernelgenerierung, kein Custom Metal und kein Modell-End-to-End-Claim.

Frühere Dispatch-Messungen dienten ausschließlich der Kandidatenauswahl. Sämtliche
H1-v2-Entscheidungsdaten müssen nach dieser Vorregistrierung neu erhoben werden.

## Versiegelter Ablauf

H1-v2 besteht aus zwei unvertauschbaren Stufen.

### Stufe 1: A/A-Kalibrierung

Sechs getrennte Prozesse laufen in der festen Reihenfolge
`C0,V0,C1,V1,C2,V2`. Beide Arme sind getrennte Callables desselben seriellen Plans.
Jede Session verwirft zwei gepaarte Warmup-Blöcke und speichert danach 24 gepaarte
Messblöcke. Die Reihenfolge A/B oder B/A ist pro Session exakt ausgeglichen und wird
mit einem vorab festgelegten SHA-256-Fisher-Yates-Schedule bestimmt.

Aus den sechs Session-Ratios wird vor A/B abgeleitet:

```text
raw_MDE = 2 × sd(session_ratio) × sqrt(2/3)
MDE     = max(0,05, raw_MDE)
```

Die Kalibrierung öffnet A/B nur, wenn das hierarchische 95-%-Intervall den Nullwert
`1,0` enthält, der Punktschätzer höchstens 5 % Bias zeigt und `MDE ≤ 15 %` bleibt.
Die Kalibrierungszusammenfassung und die daraus abgeleitete MDE werden in einem
separaten Bestätigungssiegel gebunden.

### Stufe 2: A/B-Bestätigung

Nach dem Siegel folgen sechs neue Prozesse in derselben festen Reihenfolge und mit
demselben Blockdesign. Charakterisierung (`C0,C1,C2`) und Validierung (`V0,V1,V2`)
werden getrennt sowie gemeinsam mit je 10.000 deterministischen hierarchischen
Bootstrap-Ziehungen ausgewertet.

Ein Gewinn gilt nur als bestätigt, wenn die Obergrenzen aller drei Intervalle
unter `1 − MDE` liegen. Eine Regression benötigt spiegelbildlich drei Untergrenzen
über `1 + MDE`. Äquivalenz erfordert, dass alle drei vollständigen Intervalle
innerhalb `[1 − MDE, 1 + MDE]` liegen. Alles andere ist formal inkonklusiv.

Es gibt kein optionales Stoppen, keine Wiederholung fehlgeschlagener Sessions und
keine Erweiterung der Kandidatenfamilie. Jede Änderung erfordert eine neue Study-ID.

## Korrektheit, Ressourcen und Historie

Vor Timingbeginn müssen Referenz, A und B für alle acht Ausgaben byteidentisch sein;
der maximale absolute Fehler muss exakt null sein. Erfasst werden Rohzeiten,
Armreihenfolge, GPU- und Wall-Zeit, CPU-Zeit, RSS sowie verfügbare MLX-Speicherwerte.

Pro Session gelten fail-closed:

| Budget | Grenze |
| --- | ---: |
| GPU-Arbeit | höchstens 120 s |
| kontinuierliche GPU-Arbeit | höchstens 6 s |
| Pflichtpause | mindestens 4 s |
| GPU-Duty-Cycle pro 60-s-Fenster | höchstens 25 % |
| Wall-Clock | höchstens 20 min |
| Kandidaten-Cooldown | mindestens 60 s |
| Stromquelle | Netzbetrieb |

Zwischen getrennten Sessions vergehen zusätzlich 20 reale Sekunden. Jeder Lauf ist
an einen sauberen Git-Stand, Datei-Hashes, Python-/Paketumgebung und Hardwareidentität
gebunden. Der neue Store `.friday-data/h1-v2.sqlite3` ist append-only; Update, Delete
und Ersetzen sind per Schema gesperrt. Jede Leseoperation replayt die vollständige
Studienhistorie. Die lokale UI öffnet den Store ausschließlich read-only.

## Formale Aussagegrenze

Auch ein positives Ergebnis gilt nur für dieses Gerät, diesen FP16-Workload und
diesen Ausführungsplan. Es belegt weder einen Vorteil auf anderen Geräten noch
Transformer-End-to-End-Performance, Modellqualität, Self-Optimization im Allgemeinen
oder einen Vorteil selbstgenerierter Kernel.

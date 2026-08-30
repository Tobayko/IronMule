# Gemma auf Apple Silicon: Portabilitäts- und Evidenzvertrag

## Zweck

Dieses Dokument beschreibt, wie Project Friday lokale Gemma-Läufe auf Apple
Silicon sicher und ehrlich auf ein einzelnes Gerät trimmt. Es ist kein
Performanceclaim und überträgt keine historische Geschwindigkeit auf ein
anderes Gerät, Modell oder eine andere Softwareversion.

Ein Profil ist nur für die vollständige lokale Identität gültig:

```text
Chip/GPU + RAM/Kerne + macOS + Python + MLX/mlx-lm + Friday/IronMule-Code
+ Modell-ID/Revision/Manifest + Architektur + Quantisierung + Tokenizer
+ Workload/Generator/Power-Modus
```

Fehlt ein Feld oder ändert sich ein Hash, wird das Profil nicht wiederverwendet.
Der sichere Startwert ist die unveränderte Baseline.

## Modell- und RAM-Realität

| Modell | Aktueller Evidenzstand | Portabilitätsregel |
| --- | --- | --- |
| Gemma 1B | lokaler Snapshot/Evidence vorhanden | erster kleiner Realtest nach lokalem Fingerprint und A/A |
| Gemma 4B | lokaler Snapshot und Q2-Verfahrensbeleg vorhanden | eigener A/A-/A/B-Test; Q2-Speedup nicht übernehmen |
| Gemma 12B | getrennte Evidenz, aber kein allgemeiner Friday-Claim | eigener Snapshot, RAM-/Swap-Gate und neue Messung erforderlich |
| Gemma 27B | kein belastbarer lokaler Gemma-27B-Messvertrag | blockiert, bis Snapshot, Identität und Ressourcenlauf vorhanden sind |

Gewichte, KV-Zustand, temporäre Arrays, Python-/MLX-RSS und Betriebssystem-
Overhead konkurrieren um Unified Memory. Eine rechnerische Gewichtsgröße ist
keine Zusage, dass ein Lauf ohne Swap oder sicherem Peak-Gate passt. Deshalb
entscheidet jeder Lauf mit beobachtetem MLX-Peak, Prozess-RSS und Swap-Delta.
12B/27B werden nicht aus Qwen-, anderen Gemma- oder fremden Gerätewerten
interpoliert.

## Per-device Self-Calibration

Für jedes Gerät entsteht ein neuer exakter Fingerprint. Er bindet mindestens
Chip/GPU, RAM, Kernlayout, macOS, Python, MLX, mlx-lm, Runtime-/Source-Commit,
Modellrevision, vollständigen Manifest- und Tokenizer-Hash, Quantisierung,
Generator, Prompt-/Workloadvertrag und Power-Modus. Historische Daten dürfen
die Reihenfolge erlaubter Versuche vorschlagen, aber keine Messzahl ersetzen.

Der v0.1-Ablauf ist manuell und einmalig:

1. Der Nutzer startet ausdrücklich eine Sitzung und wählt 5 bis maximal 30
   Minuten. Warten auf Readiness zählt in diese harte Gesamtfrist.
2. Readiness verlangt AC, Low-Power aus, stabile Last/Speicher/Swap-Werte,
   keine fremde Modell-/Claude-Last und eine gültige Lease. Unbekannt bedeutet
   warten oder abbrechen.
3. Calibration führt einen echten A/A-Lauf mit identischer Baseline in
   ausgeglichenen AB-/BA-Prozessen aus. Er bestimmt Rauschen und bindet Raw-
   Evidence; er testet noch keinen Kandidaten.
4. Test führt den gebundenen IronMule-Tuner einmal aus. Die bereits von
   `tune.confirm` ausgeführte A/B-Bestätigung wird genau einmal abgegriffen;
   kein zweiter Confirmation-Lauf und keine freie CLI-Option sind erlaubt.
5. Baseline und Kandidat müssen Token-, Count-, Text- und source-derived
   Stop-Equivalence bestehen. Engine-TTFT und Decode-only-Tokens/s werden mit
   Rohserien, Medianen, Streuung und Konfidenzintervallen ausgewertet.
6. Nur eine vollständig belegte Shadow-Empfehlung darf entstehen. Aktivierung,
   Promotion und Runtime-Änderung bleiben in v0.1 deaktiviert.

## Correctness, Ressourcen und Rollback

Korrektheit steht vor Geschwindigkeit. Token-IDs werden nur transient zum Gate
verglichen. Persistiert werden ausschließlich redigierte numerische Serien und
Token-/Count-/Text-Equivalence-Hashes; Prompt, Text, Token-IDs, PIDs, lokale
Pfade und stdout/stderr gehören nicht in das Session-Result.

Der Text-Hash bindet den exakten Tokenizer und die Tokenfolge. Die Stop-
Equivalence ist source-derived aus identischer Folge/Anzahl sowie gebundenem
Max-Token-, Capacity- und EOS-Vertrag; eine nicht beobachtete Stop-Ursache wird
nicht behauptet.

Der Ressourcenvertrag prüft MLX-Peak, Prozess-RSS, Swap-vorher/nachher,
Timeout/Crash, Power und fremde Last. Für den aktuellen Shadow-Scope gelten
maximal 12 GiB für MLX-Peak und RSS sowie `swap_delta=0`, zusätzlich zur
gewählten Sitzungsfrist. Ein unbekannter oder widersprüchlicher Wert ist kein
Pass.

Bei jedem Fehler, Mismatch, Timeout, Swap-Wachstum, Speicherüberschreitung,
Leaseverlust oder Readiness-Problem wird die Empfehlung verworfen und die
Baseline beibehalten. Eine spätere Aktivierung müsste atomar, canary-geprüft,
versioniert und mit sofortigem Baseline-Rollback ausgestattet sein; sie ist
nicht Teil dieses Dokuments.

## Andere Macs

Ein anderer Apple-Silicon-Mac startet immer mit Baseline und einem neuen A/A.
Auch derselbe Chip mit anderem RAM, macOS, MLX, Python, Power-Modus,
Modellrevision, Quantisierungs- oder Tokenizer-Hash ist eine neue Domäne.
`M1 Max`-Ergebnisse werden nicht als Beleg für M1, M2, M3 oder M4 verwendet.
Ein Speedclaim gilt nur, wenn die lokale Identität, der Workload, die Baseline,
die Raw-Samples und alle Sicherheitsgates gemeinsam vorliegen.

## Kill-Kriterien

Ein Kandidat wird dauerhaft verworfen oder die Sitzung bleibt bei Baseline,
wenn eines davon eintritt:

- Token, Count, Text-Hash, source-derived Stop-Contract oder Zustandsidentität
  weichen ab.
- A/A fehlt, ist unbalanciert, nicht reproduzierbar oder überschreitet die
  registrierte Unsicherheitsgrenze.
- TTFT/Decode-Metrik, Raw-Serie, Konfidenzintervall, Modellidentität oder
  Ressourcenbeleg fehlt.
- Swap wächst, MLX-Peak/RSS überschreitet 12 GiB, ein Timeout/Crash/Fremdprozess
  tritt auf oder die Lease/Readiness wird ungültig.
- Ein Source-, Interpreter-, Workload-, Modell- oder Registry-Hash weicht ab.
- Ein Test versucht, lokale Pfade, freie Flags, Downloads, Installationen oder
  automatische Promotion einzuschleusen.

Ein negatives oder inconclusive Ergebnis ist ein gültiges Ergebnis. Es wird
nicht durch synthetische Daten, eine andere Modellgröße oder einen anderen Mac
„aufgefüllt“.

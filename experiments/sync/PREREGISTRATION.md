# Mini-Vorregistrierung — Kosten der Host-Synchronisation je Decode-Schritt

**Kandidaten-ID:** `host-sync-20260824-01`
**Zyklus:** 6 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Warum dieser Kandidat

Der Auftrag nennt unter Schritt 2 ausdrücklich „unnötige CPU-GPU-Synchronisationen"
und „vollständige Logit-Readbacks". Beides wurde in fünf Zyklen nicht systematisch
geprüft.

Jeder Decode-Schritt liest das gesampelte Token zum Host — für die Stop-Token-Prüfung
und fürs Streaming. Das erzwingt eine Synchronisation je Token.

**Er ist der letzte verbliebene Kandidat, der die Numerik nicht berührt.** Gerechnet
wird dasselbe; verändert wird nur, *wann* das Ergebnis gelesen wird. Damit hat er
dieselbe Ausgangslage wie der persistente Prozess aus Zyklus 5, der als einziger ein
Korrektheitsgate bestanden hat.

## Hypothesen

**H1 (Korrektheit).** Ein Loop ohne Host-Readback je Schritt erzeugt dieselben Token
wie einer mit.

**H2 (Wirkung).** Der Readback kostet mindestens `3 %` der Schrittzeit.

Die Schwelle ist bewusst niedrig: bei `12,1` ms Schrittzeit wären `3 %` rund
`0,36` ms, und alles darunter verschwindet in der Streuung, die frühere Zyklen mit
`0,4`–`1,8 %` gemessen haben.

## Genau geänderte Variable

Ob je Schritt ein Host-Readback stattfindet. Sonst nichts.

## Arme

| Arm | Verhalten |
| :--- | :--- |
| `readback` | je Schritt `int(y[0,0])`, wie der heutige Pfad |
| `deferred` | Token bleiben auf dem Gerät, ein einziger Readback am Ende |
| `eos_check` | je Schritt Readback **und** Stop-Token-Prüfung, wie ein echter Generator |

Der dritte Arm ist nötig, weil ein Dienst nicht nur liest, sondern auf das Gelesene
reagiert. Ihn wegzulassen würde die Ersparnis überschätzen.

## Workload

`128` Decode-Schritte nach einem `~900`-Token-Prefill, greedy, Batch `1`.
Drei Wiederholungen je Arm, Arme abwechselnd, Median. Aufwärmlauf vorab, dessen
Ergebnis verworfen wird.

## Primärer Endpunkt

Verhältnis der Sekunden je Token, `deferred` gegen `readback`.

## Sekundäre Endpunkte

`eos_check` gegen `readback`; Tokenidentität aller drei Arme.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im selben
Prozess. Keine nachträgliche Änderung der Schwelle.

## Vorab festgelegte Deutung

| H1 | H2 | Entscheid |
| :--- | :--- | :--- |
| hält | hält | `candidate_recommended_for_preregistration` |
| hält | verfehlt | `candidate_characterized` — der Readback ist nicht der Engpass |
| scheitert | egal | `candidate_correctness_failed`, terminal |

Ein Scheitern von H1 wäre besonders aussagekräftig: es hieße, dass selbst der
Zeitpunkt des Lesens die Ausgabe beeinflusst, und damit wäre auf dieser Plattform
**gar kein** semantisch äquivalenter alternativer Pfad mehr konstruierbar.

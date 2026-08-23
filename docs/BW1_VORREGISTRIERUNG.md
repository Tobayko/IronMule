# BW1 — prospektive Ein-Kandidaten-Vorregistrierung: Batch-Decode

**Study-ID:** `bw1-batch-decode-20260823-01`

**Status dieses Dokuments:** Designvertrag. Eine Messung ist erst nach sauberem
Implementierungscommit und persistiertem `preregistration`-Record zulässig.

**Vorgeschichte, ausdrücklich ohne Anspruch:** Die Breiten-, Segmentierungs- und
Spekulationsmessungen vom 23. August 2026
([`DECODE_WIDTH_BEFUND`](DECODE_WIDTH_BEFUND_2026-08-23.md),
[`GERAETEMODELL`](GERAETEMODELL_2026-08-23.md),
[`POLICY_GELTUNGSBEREICH`](POLICY_GELTUNGSBEREICH_2026-08-23.md)) sind durchgehend
`formal_claim=false`. Sie waren nicht prospektiv registriert, hatten kein A/A-Gate und
keine vor der Messung eingefrorene MDE. Sie begründen diese Studie, ersetzen sie nicht
und dürfen nicht als deren Vorergebnis zitiert werden.

## 1. Forschungsfrage und Abgrenzung

Erbringt gebündeltes Decodieren von `32` gleichzeitigen Anfragen auf genau einem Gerät,
für genau ein Modell und genau eine Quantisierung einen Durchsatzvorteil jenseits einer
vorab eingefrorenen Mindestwirkung — **ohne die erzeugten Token zu verändern**?

Ausdrücklich **nicht** Gegenstand:

- eine Aussage über andere Modelle, Quantisierungen oder Geräte;
- eine Aussage über **Latenz**. Der Endpunkt ist Durchsatz. Eine einzelne Anfrage
  wird durch Bündelung nicht schneller beantwortet, und diese Studie behauptet das nicht;
- die Breiten-Policy als Ganzes. Registriert ist **eine** Breite, nicht eine Kurve;
- kontextbasierte Spekulation. Sie ist ein getrennter Kandidat und bleibt hier aus.

## 2. Primärer Endpunkt und Entscheidungsregionen

Primärer Endpunkt ist das **gepaarte Verhältnis** `R` der Sekunden je Sample-Token:

```
R = t_batch / t_seriell
```

gemessen über balancierte Blöcke, in denen beide Arme dieselben Prompts in derselben
Reihenfolge abarbeiten.

| Region | Bedingung | Entscheid |
| :--- | :--- | :--- |
| Gewinn | oberes Ende des 95-%-Intervalls von `R` liegt unter `1 − MDE` | `bw1_gain_confirmed` |
| kein Gewinn | sonst | `bw1_inconclusive` |

Ein Entscheid ist nur gültig, wenn **alle** Korrektheitsgates aus Abschnitt 3 bestanden
sind. Ein Gewinn bei verändertem Text ist kein Gewinn, sondern ein anderes Modell.

## 3. Korrektheitsgates, vor der Zeitmessung

**Tokenidentität.** Jede der `32` gebündelten Sequenzen muss Token für Token dem
entsprechen, was dieselbe Anfrage allein unter greedy Sampling erzeugt.

Das ist **nicht** vorab garantiert. Bündelung ändert die Form der Kernelaufrufe, und
eine frühere, nicht registrierte Messung dieser Sitzung fand genau daran eine
Abweichung an einem Punkt, an dem der Logit-Abstand `0,344` betrug — also kein
Rundungsartefakt. Scheitert dieses Gate, endet die Studie mit
`bw1_correctness_failed`; das ist ein gültiges terminales Ergebnis und wird **nicht**
durch Lockerung des Kriteriums umgangen.

**Vollständigkeit.** Jede Sequenz muss die volle registrierte Tokenzahl erzeugen oder
an einem Stop-Token enden. Ein abgeschnittener Lauf ist ungültig, nicht kurz.

## 4. Workload und Kandidat

Eingefroren, genau einer:

| Größe | Wert |
| :--- | :--- |
| Modell | `mlx-community/gemma-3-4b-it-4bit`, Revision aus dem Projekt-Snapshot |
| Quantisierung | `4 bit`, Gruppengröße `64` |
| Gerät | Apple M1 Max, Fingerprint wie in `friday_evidence.provenance` |
| Kandidat | Batch-Decode, Breite `32`, segmentiert |
| Kontrolle | `32` einzelne Decodeläufe, Breite `1`, dieselben Prompts |
| Prompts | `32` verschiedene, je `64 ± 4` Token, fester Seed |
| Ausgabe | `128` Token je Sequenz, greedy (`temp=0`) |
| Segmentlänge | aus gemessenen Schrittkosten und dem Continuous-Limit abgeleitet |

Gleiche Promptlängen sind Absicht: Padding ist ein **eigener** Effekt, und ihn hier
hineinzumischen würde den Endpunkt verwässern. Die ungleiche Variante ist ein späterer,
getrennter Kandidat.

## 5. Zweistufiges Studiendesign

### 5.1 Frische A/A-Kalibrierung

Sechs getrennte Prozesse, beide Arme **seriell**. Der Zweck ist, die Streuung des
Messsystems zu bestimmen, bevor ein Unterschied behauptet wird.

Gates:

1. aggregiertes `R` der A/A-Läufe muss `1,0` einschließen;
2. kein einzelner Lauf darf mehr als `5 %` vom Median abweichen;
3. alle sechs Läufe am Netzteil, in getrennten Prozessen, mit realem Cooldown;
4. Tokenidentität zwischen allen sechs Läufen.

Aus der A/A-Streuung wird die MDE abgeleitet. **Der konservative Boden von `5 %` gilt
in jedem Fall**, auch wenn die rohe Ableitung kleiner ausfällt — so wie bei H1-v2
(`0,0752 %` roh) und N10-v2 (`0,0857 %` roh).

### 5.2 Frische A/B-Bestätigung

Sechs weitere getrennte Prozesse, danach, ohne Wiederverwendung der A/A-Daten. Die
ersten drei bilden die Charakterisierung, die letzten drei die Validierung; beide
müssen das Gain-Gate **getrennt** bestehen.

## 6. Statistik

- Schätzer: Median je Block, hierarchischer Bootstrap über Sessions und Blöcke;
- `10.000` Replikate, Seeds aus `friday_h0.constants.AA_BOOTSTRAP_SEEDS`;
- 95-%-Intervall, zweiseitig;
- kein Nachziehen von Blöcken, kein Verwerfen von Ausreißern, keine nachträgliche
  Änderung der Schwelle.

## 7. Ressourcen-, Sicherheits- und Fehlervertrag

- `BudgetGuard` mit der bestehenden Policy: `120 s` GPU je Prozess, `6 s` kontinuierlich,
  `25 %` Duty über `60 s`, `20 min` Wall, `60 s` Kandidaten-Cooldown;
- Segmentierung nach `friday_hardware.HardwareProfile.steps_per_segment`, damit die
  effiziente Breite das Continuous-Limit nicht verletzt;
- kein Download, keine Installation, kein Modellwechsel;
- ein Fehler wird **nicht** im selben Prozess wiederholt. Er wird persistiert und
  beendet den Lauf.

## 8. Provenienz und Persistenz

- eigene Datei `.friday-data/bw1.sqlite3`, Application-ID `FRB1`, Modus `0600`;
- append-only, hashverkettet, Schema v1;
- gebunden werden Root-Git-Revision, Code-Fingerprint über die beteiligten Pakete,
  Spec-Fingerprint über **dieses** Dokument, Umgebungs- und Hardware-Fingerprint;
- genau ein terminaler Record darf `formal_claim=true` tragen;
- read-only UI auf einem eigenen Port, wie bei den bestehenden Speichern.

**Hinweis zur Benennung.** Der Spec-Fingerprint bindet Pfad und Inhalt dieses
Dokuments. Es ist nach dem Seal weder umzubenennen noch zu editieren — auch nicht
für Tippfehler.

## 9. Autorisierte Reihenfolge

1. Implementierung, vollständige Offline-Suite grün;
2. sauberer Commit, keine ungetrackten Änderungen;
3. `preregistration`-Record schreiben, Seal bilden;
4. Korrektheitsgates (Abschnitt 3);
5. sechs A/A-Prozesse, Kalibrierungs-Replay, MDE einfrieren, Bestätigungssiegel;
6. sechs A/B-Prozesse;
7. terminaler Entscheid, genau einmal.

Jeder Schritt ist erst nach dem vorherigen zulässig. Wird die Reihenfolge verletzt,
ist der Lauf ungültig, unabhängig vom Ergebnis.

## 10. Abbruch- und Promotionsregel

Die Studie endet terminal bei:

- gescheitertem Korrektheitsgate → `bw1_correctness_failed`;
- gescheitertem A/A-Gate → `bw1_calibration_failed`;
- verfehltem Gain-Gate → `bw1_inconclusive`;
- Budget- oder Umgebungsverletzung → sanitisierter Fehlerrecord.

Ein bestandener Entscheid erlaubt **ausschließlich**
`permit_bounded_batch_decode_prototype` — also einen begrenzten Ausführungspfad im
exakt registrierten Scope, wie ihn N8 und N10 bekommen haben. Er erlaubt **nicht**:
eine andere Breite, ein anderes Modell, eine andere Quantisierung, ein anderes Gerät,
ungleiche Promptlängen, Spekulation oder produktive Aktivierung.

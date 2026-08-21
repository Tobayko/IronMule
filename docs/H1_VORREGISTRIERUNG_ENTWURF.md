# H1 — Vorregistrierung (ENTWURF, nicht freigegeben)

> **Auditnachtrag vom 21.08.2026:** Dieser Entwurf wurde vor den historischen
> H1/H2-Läufen weder vollständig ausgefüllt noch freigegeben. Insbesondere waren
> das formale A/A-Gate, der hierarchische Bootstrap und die MDE vor dem ersten
> A/B-Lauf nicht versiegelt. Die damaligen Ergebnisse können dieses Dokument
> nicht rückwirkend erfüllen und werden nur als explorative Legacy-
> Zusammenfassungen geführt. Für einen künftigen formalen H1-Lauf ist eine neue,
> prospektive Version mit neuer Study-ID erforderlich.

**Status:** Entwurf. Dieses Dokument ist **nicht** freigegeben und autorisiert keinen
Lauf, keine Installation und keinen Download. Es friert die Zahlen ein, die *vor* der
ersten H1-Messung feststehen müssen. Eine Freigabe durch den Nutzer ist erforderlich,
bevor irgendetwas davon ausgeführt wird.

**Historischer Zweck:** H0.1 hatte gezeigt, dass der naheliegende Messmodus für H1
untauglich ist. Die beabsichtigte Schließung vor Sichtung von H1-Daten gelang nicht;
dieser Entwurf dokumentiert den Zielvertrag, ist aber keine Vorregistrierung der
bereits ausgeführten Läufe.

**Forschungsgrenze:** H1 prüft, ob eine agentengesteuerte Änderung an einer festen
Operation einen *messbaren* Effekt erzeugt. H1 ist kein Nachweis von
Self-Optimization über mehrere Operationen, keine Aussage über Modelle, keine
Cross-Device-Aussage und keine End-to-End-Performanceaussage.

## 1. Was aus H0.1 feststeht

Diese Werte sind gemessen, nicht angenommen. Sie stammen aus der abgeschlossenen
Study `h01-study-1812a894…c39ca` mit sechs Sessions und 480 Main-Samples.

| Größe | Wert | Quelle |
| --- | ---: | --- |
| Session-Median, gepact | `6,34`–`8,67 ms` | 6 Sessions |
| Between-Session-CV, gepact | `13,2 %` | 6 Session-Mediane |
| Nachweisgrenze 3v3, gepact | rund `21 %` | `2 × SE(diff)` |
| Within-Batch-CV, dicht | `7,0 %` | Run22, 32 Evals |
| Anteil Samples über `1,5 ×` Median | `16,7 %` | 480 Main-Samples |
| Erstes Sample nach `20 s` Cooldown | `11,4`–`15,2 ms` | 6/6 Sessions |

**Konsequenz:** Der gepacte Einzelmodus kann Effekte unter `21 %` nicht von
Untergrundrauschen unterscheiden. Typische Kernel-Optimierungen liegen unter diesem
Wert. H1 darf in diesem Modus daher nicht primär entschieden werden.

## 2. Was aus H0.1 ausdrücklich **nicht** feststeht

Der dichte Batch-Modus ist *innerhalb* eines Batches achtfach stabiler. Die dafür
entscheidende Größe ist jedoch die **Streuung zwischen getrennten Läufen**, und die
ist nicht gemessen. Die vier vorhandenen `eager_baseline`-Runs besitzen
unterschiedliche `code_sha256` und sind deshalb nicht als Replikate verwendbar.

Aus `CV = 7,0 %` innerhalb eines Batches folgt **keine** Nachweisgrenze für H1. Wer
das gleichsetzt, verwechselt Messpräzision mit Reproduzierbarkeit.

## 3. Fester Ablauf: A/A vor A/B

H1 läuft in zwei getrennten, nacheinander freizugebenden Stufen.

**Stufe 1 — A/A-Kalibrierung.** Zwei Gruppen von je `k` Läufen mit **identischer**
Konfiguration, identischer Fixture und identischer Provenienz. Es wird nichts
optimiert. Ergebnis ist ausschließlich die Between-Run-Streuung im Batch-Modus.

Aus ihr wird die Mindest-Effektgröße `MDE` abgeleitet und **eingefroren**, bevor
Stufe 2 beginnt:

```
MDE = 2 × SE(diff) = 2 × s_ratio × sqrt(2/k)
```

`s_ratio` ist die Standardabweichung der **gepaarten Session-Ratios**
`R_s = exp(median_b(log(t_B / t_A)))` über die Blöcke `b` eines Prozesses.

**Korrektur gegenüber der ersten Fassung dieses Entwurfs.** Ursprünglich stand hier
`s_between` als Standardabweichung der *ungepaarten Lauf-Mediane*. Das war ein Fehler
dieses Entwurfs, kein Freiheitsgrad: Der gepaarte Schätzer `R_s` ist in
`docs/PHASE1_MATMUL_SPEC.md` Abschnitt 5.3.1 seit dem 19.08.2026 vorregistriert, also
lange vor jeder A/A-Messung. Die Korrektur bringt diesen Entwurf mit der bestehenden
Vorregistrierung in Übereinstimmung; sie wählt keinen Schätzer nach Datenlage aus.

Der Unterschied ist erheblich und inhaltlich der Kern der Sache: Beide Arme eines
Blocks erleben denselben Störuntergrund, weshalb sich dieser im Quotienten
herauskürzt. Ungepaart bleibt er vollständig stehen.

**Stufe 2 — A/B.** Erst wenn `MDE` eingetragen ist, wird ein Kandidat gegen die
Baseline gemessen. Ein Effekt unterhalb `MDE` ist ein **Nullbefund**, kein
"tendenzieller Gewinn". Ein Effekt oberhalb `MDE` ist ein Kandidat für Replikation,
noch kein bestätigter Gewinn.

**Abbruchregel gegen Fischen:** Die Anzahl der in Stufe 2 geprüften Kandidaten wird
vor Stufe 2 festgelegt. Wird sie erhöht, ist das eine neue Vorregistrierung mit neuer
`study_id`.

## 4. Bekannte Störgröße: der Cooldown-Effekt

Das erste Sample nach einer längeren Pause ist in allen sechs H0.1-Sessions um rund
das Doppelte erhöht. Ursache nicht gemessen.

Regel für H1: Jeder Lauf verwirft eine feste, vorab registrierte Anzahl von
Warmup-Samples nach jeder Pause. Diese Zahl steht vor Stufe 1 fest und ist für
Baseline und Kandidat **identisch**. Ein asymmetrisches Warmup zwischen den Armen
macht den Lauf ungültig.

## 5. Hardwareschonung — verbindliche Budgets

H0.1 belastete das Gerät mit `5,26 s` GPU-Arbeit über `6,6 min`, also einem
Duty-Cycle von `1,33 %`; die längste ununterbrochene Last war eine einzelne Matmul
mit `26,8 ms`. H1 sucht über Kandidaten und wiederholt Messungen, kann also deutlich
mehr Last erzeugen. Deshalb gelten harte, prüfbare Obergrenzen.

| Budget | Grenze | Prüfung |
| --- | ---: | --- |
| GPU-Arbeit je Lauf | `≤ 120 s` | Summe gemessener `duration_ns` |
| Ununterbrochene GPU-Last | `≤ 6 s` | dann Pflichtpause |
| Pflichtpause nach einem Lastblock | `≥ 4 s` | reale Wartezeit |
| Duty-Cycle über jedes `60 s`-Fenster | `≤ 25 %` | gleitendes Fenster |
| Wall-Clock je Lauf | `≤ 20 min` | harter Abbruch |
| Cooldown zwischen Kandidaten | `≥ 60 s` | reale Wartezeit |
| Netzbetrieb | verpflichtend | `power_source == ac_power` |

**Netzbetrieb ist eine Messanforderung, nicht nur Schonung:** Auf Akku begrenzt macOS
das GPU-Power-Budget. Ein Lauf auf Batterie ist weder schonend noch vergleichbar und
wird vor der ersten Messung abgelehnt.

Die Grenze für ununterbrochene Last wurde vor dem ersten H1-Lauf von `2 s` auf `6 s`
korrigiert. Grund: Ein A/A-Prozess erzeugt strukturbedingt rund `4,1 s` zusammenhängende
Last (30 gepaarte Blöcke à `68 ms` über zwei Arme). Der ursprüngliche Wert war ohne
Kenntnis dieser Blockstruktur gewählt. Die Korrektur erfolgte **vor** jeder H1-Messung
und ohne Sichtung von H1-Daten; nach dem ersten Lauf wäre sie nachträgliches Tuning und
damit unzulässig.

**Fail-closed:** Jede Überschreitung bricht den Lauf ab und verwirft ihn, statt ihn
zu kürzen. Ein abgebrochener Lauf ist ein gültiges Ergebnis vom Typ "nicht
durchgeführt", niemals ein Teilergebnis.

**Nicht verfügbar:** Eine Temperaturschwelle ist nicht registrierbar.
`ProcessInfo.thermalState` hat keine stdlib-Bindung, und `powermetrics` benötigt
erhöhte Rechte. Die Budgets oben sind der Ersatz: Sie begrenzen die *Ursache* von
Wärmeeintrag statt eine nicht messbare Wirkung.

## 6. Änderungsregel

`MDE`, `k`, Warmup-Anzahl, Kandidatenzahl, Messmodus und sämtliche Budgets aus
Abschnitt 5 dürfen nach Sichtung von Daten nicht verändert werden. Jede spätere
Änderung benötigt eine neue Spezifikation, neue Provenienz und neue Run-/Study-IDs.

H0 und H0.1 bleiben unverändert. Kein H1-Ergebnis reklassifiziert einen H0-Lauf,
löst eine Promotion aus oder verändert eine H0- oder H0.1-Schwelle.

## 7. Offene Entscheidungen für den Nutzer

1. Freigabe für Stufe 1 (A/A-Kalibrierung) — reine Messung, keine Optimierung.
2. `k` ist **nicht** frei wählbar. Der A/A-Nullpfad ist in
   `docs/PHASE1_MATMUL_SPEC.md` Abschnitt 5.3.1 bereits mit exakt drei
   Charakterisierungs- und drei Bestätigungsprozessen und festen Seeds
   vorregistriert. Damit gilt `k = 3` je Set. Eine Änderung wäre ein Bruch der
   bestehenden H0-Vorregistrierung.
3. Ob der Cooldown-Effekt vorab als eigene kleine Studie isoliert wird. Er ist der
   einzige reproduzierbare Befund aus H0.1 und wäre billig zu charakterisieren.

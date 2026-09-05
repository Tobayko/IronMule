# Schlussfolgerungen nach der R2-Kampagne — 05.09.2026

**Status: Vorlage, nicht Entscheid.** Alles hier ist meine Einschätzung nach den
`352` Messpunkten vom 4./5. September. Die Entscheidung unter „Was zu entscheiden
ist" hat der Nutzer noch nicht getroffen. Bis dahin gilt der Stand von
`BACKLOG.md` R2 unverändert.

Belege: `docs/R2_VORREGISTRIERUNG.md` samt Amendments A und B,
`docs/ARBEITSJOURNAL.md` Eintrag 2026-09-05,
`experiments/r2_campaign/evaluate_corpus.py`.

---

## 1. Was jetzt belegt ist

`352` von `352` messbaren Punkten, `0` zensiert, `0` Tokenidentitätsbrüche,
Hash-Kette verifiziert. Gepaart AB/BA, sechs Paare je Punkt.

| Knopf | Messungen | Gewinn | Streuung | Urteil |
| --- | ---: | ---: | --- | --- |
| `head_skip_prefill` | `214` | **`+12,40 %`** | `[0,8670; 0,8870]` | trägt |
| `fixed_compiled_cache` | `46` | `+1,55 %` | `[0,9756; 0,9933]` | trägt knapp |
| `baseline` (Kontrolle) | `43` | `−0,02 %` | `[0,9929; 1,0045]` | schließt Null ein, wie es muss |
| `readback_every_2` | `49` | `−0,14 %` | `[0,9923; 1,0082]` | kein Gewinn |

**Drei unabhängige Bestätigungen**, und das ist der eigentliche Wert dieser
Zahlen:

1. `head_skip_prefill` reproduziert über `214` Punkte das Geräteprofil vom
   2026-09-03 (`0,8778`, KI `[0,8668; 0,8888]`) — gemessen über einen anderen
   Aufrufweg.
2. Die A/A-Kontrolle (`baseline` gegen sich selbst) landet auf `1,0002`. Eine
   Kontrolle, die nahe `1` liegen muss, tut es.
3. `readback_every_2` zeigt keinen Gewinn — dieselbe Antwort, die die
   Nachmessung vom 2026-09-03 gab, als sie `bundled_readback` im Geräteprofil
   durchfallen ließ.

Ein Ergebnis, das sich auf drei getrennten Wegen gleich zeigt, ist kein Artefakt.

## 2. Was nicht belegt ist — und warum

Die Leitfrage von R2 lautete: **lohnt sich eine lernende Auswahl, die selbst
entscheidet, welcher Knopf gedrückt wird?**

**Diese Frage bleibt unbeantwortet.** Nicht, weil die Antwort nein wäre, sondern
weil der vorregistrierte Test in dieser Korpusgröße nicht durchführbar ist.

Das Tor verlangte `conclusive=true` auf dem Holdout — den letzten `20 %` der
versiegelten Reihenfolge. Dort erreichen drei von vier Zielpolicies die
ESS-Untergrenze `30` nicht (`7`, `7`, `5`).

**Der Fehler liegt in meiner Vorregistrierung, nicht in der Messung.** Die
Größentabelle des Trockenlaufs — `400` Punkte ergeben `5/5` belastbare
Schätzungen — gilt für den **gesamten** Korpus. Wer `20 %` als Holdout
abschneidet, behält `70` Punkte, und bei `p(andere) = 0,12` zieht eine nicht
gehintete Aktion darin rund **acht** Mal. Ich habe einen Holdout für ein Tor
eingefroren, das er nie tragen konnte.

**„Nicht bestanden" und „nicht prüfbar" sind nicht dasselbe.** Das ist die
Kernaussage dieses Dokuments. R2 ist nicht widerlegt worden; R2 ist mit dieser
Datenmenge nicht entscheidbar.

Holdout und ESS-Untergrenze werden **nicht** nachträglich angepasst. Genau für
diesen Moment standen sie vorher fest.

## 3. Meine Empfehlung: die Frage schließen, nicht weiter messen

Der Weg zum stellbaren Tor kostet rund **`1250` gemessene Punkte**, also gegenüber
`352` weitere etwa **`41` Stunden**. Ich rate davon ab, aus drei Gründen:

**Erstens: das Projekt hat die Frage fachlich schon beantwortet.**
`docs/FABLE_ERFOLGSPFAD.md` (2026-09-01) legt dar, warum RL hier das falsche
Werkzeug ist — einstufige Wahl aus einer festen Liste ist ein Contextual Bandit,
kein sequentielles Problem; RL braucht `10⁴`–`10⁶` Interaktionen. Der Aktionsraum
hat **vier** messbare Einträge. Vier Dinge misst man einmal durch und weiß es
dann. Genau das ist gerade passiert.

**Zweitens: die erwartete Antwort ist nein.** `41` Stunden Messzeit auszugeben,
um ein erwartetes Nein formal zu bestätigen, ist ein schlechter Tausch — solange
nichts davon abhängt.

**Drittens: der Korpus trägt den nützlicheren Zweck bereits.** `352` gelabelte
Punkte mit `27` Kontextfeldern reichen für ein Kostenmodell samt Bayesian
Optimization; derselbe Erfolgspfad nennt dafür `50`–`100` saubere Punkte als
Untergrenze. Der teure Teil ist getan.

## 4. Was ich stattdessen vorschlage, nach Nutzen sortiert

1. **Den einen tragenden Knopf ausreizen.** `head_skip_prefill` ist mit `+12,4 %`
   der einzige echte Gewinn und im Serving-Pfad bereits aktiv. Offen ist, ob er
   auch bei anderen Modellgrößen und längeren Antworten trägt — das ist eine
   kurze, gezielte Messung, keine Kampagne.
2. **Das Kostenmodell bauen.** Aus den `352` Punkten lernen, welcher Knopf unter
   welchen Bedingungen hilft. Konkreter Nutzen: das Einmessen eines neuen Macs
   wird kürzer, weil das Modell die Reihenfolge der Versuche vorschlägt. Das ist
   der Weg, den `FABLE_ERFOLGSPFAD` als tragfähig bezeichnet.
3. **R2 schließen** — mit dem Befund aus Abschnitt 2, nicht mit einem Nein.

## 5. Was zu entscheiden ist

| Option | Kosten | Was sie liefert |
| --- | --- | --- |
| **A — R2 schließen, Kostenmodell bauen** *(empfohlen)* | keine weitere Messzeit | nutzbares Modell aus vorhandenen Daten; R2 dokumentiert als nicht prüfbar |
| **B — Korpus auf ~`1250` ausbauen** | ~`41` h, rund vier Nächte | das Tor wird stellbar; erwartete Antwort bleibt nein |
| **C — Beides, B später** | wie A, B optional | Modell zuerst; die Kampagne bleibt wieder aufnehmbar, weil der Cursor am Korpus hängt |

Option C ist gangbar, weil der Supervisor über vorhandene Outcome-Records
fortsetzt: eine spätere Fortsetzung derselben Kampagne ist ohne Zusatzaufwand
möglich, solange `campaign_id` und Siegel unverändert bleiben.

## 6. Was hier ausdrücklich nicht steht

- **Kein Lernclaim.** Es wurde keine Policy trainiert und keine aktiviert;
  `learning_claim=false` und `no_activation=true` gelten unverändert.
- **Kein Cross-Device- und kein Cross-Model-Claim.** Alles gilt für 4B, diese
  Maschine, den angepinnten Engine-Commit `03e884cb`.
- **Kein Widerruf des RL-NO-GO und keine Bestätigung.** Der Status bleibt
  NO-GO, jetzt mit einer belegten Begründung statt einer offenen Frage.
- **Keine nachträgliche Anpassung** von Holdout, ESS-Untergrenze oder
  Reward-Metrik.

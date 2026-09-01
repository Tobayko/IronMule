# F1 — prospektive Vorregistrierung der Integrationsstudie

**Freigabe:** 1. September 2026 (Nutzerfreigabe, festgehalten in
[`FABLE_ERFOLGSPFAD.md`](FABLE_ERFOLGSPFAD.md), Abschnitt „Nutzerfreigabe vom
2026-09-01")

**Study-ID:** `f1-integration-warmcold-20260901-01`

**Status dieses Dokuments:** Designvertrag. Eine Messung ist erst zulässig nach
sauberem Implementierungscommit, gebundenem Fingerprint und persistiertem
`preregistration`-Record. Dieses Dokument enthält bewusst noch keine Hashes der
Messumgebung; sie werden beim Versiegeln eingetragen.

## 1. Forschungsfrage und Abgrenzung

Das Projekt hat drei Gewinne einzeln bestätigt und keinen davon integriert.
Geprüft wird, ob die gemeinsame Anwendung von **persistentem Modellprozess**,
**Prefill-Head-Skip** und **`fixed_compiled`** auf einem realen
End-to-End-Anfragepfad von Gemma 3 4B (4-bit, MLX) auf dem einen gebundenen
Apple-M1-Max-Gerät die **Anfragezeit** messbar senkt.

Gemessen wird die Anfrage als Ganzes, nicht mehr eine Phase:

```
request_seconds = ttft_seconds + tokens / decode_tps
```

Der Claim bleibt beschränkt auf ein Gerät, einen gebundenen Gemma-4B-Snapshot,
die registrierte Promptfamilie, Batch `1`, greedy ohne Prompt-Logprobs. Kein
Claim gilt für andere Geräte, Modelle, Quantisierungen oder Promptlängen.

## 2. Zwei Arme

| Arm | Baseline | Kandidat | Was der Arm isoliert |
| --- | --- | --- | --- |
| `cold` | frischer Prozess je Anfrage, Knobs auf Default | persistenter Prozess + Head-Skip + `fixed_compiled` | Modellladen und Kaltstart |
| `warm` | bereits geladener Prozess, Knobs auf Default | derselbe Prozess + Head-Skip + `fixed_compiled` | reine Rechenhebel ohne Ladeeffekt |

Die Arme unterscheiden sich ausschließlich in der Prozesswiederverwendung.
Beide laufen auf demselben Snapshot, derselben Engine und derselben
Promptfamilie.

## 3. Projektion — was erwartet wird und warum

Abgeleitet offline aus versiegelter Evidenz durch
[`experiments/f1_integration/project_f1.py`](../experiments/f1_integration/project_f1.py),
Datenquelle `experiments/persistent_process/results.json` (sechs Paare,
Prompt `897` Token, `32` generierte Token):

| Größe | Wert |
| --- | --- |
| warme Baseline | TTFT `1,7851 s`, Decode `0,4367 s` (`70,99` Token/s), gesamt `2,2218 s` |
| kalte Baseline | gesamt `5,5969 s` |
| Prefill-Anteil der warmen Anfrage | `79,84 %` |

| Kandidat | Ratio | End-to-End-Gewinn |
| --- | --- | --- |
| nur Head-Skip | `0,877356` | `12,26 %` |
| nur `fixed_compiled` | `0,985805` | `1,42 %` |
| beide, Arm `warm` | `0,863161` | **`13,68 %`** |
| beide + persistenter Prozess, Arm `cold` | `0,299489` | **`70,05 %`** |

**Zentraler Befund dieser Projektion — Phasen multiplizieren nicht.** Die
naive Lesart „`15,4 %` plus `7 %`" ergäbe `0,786793`, also `21,32 %`. Das ist
falsch. Head-Skip wirkt auf Prefill, `fixed_compiled` auf Decode; die
zusammengesetzte Wirkung ist das zeitgewichtete Mittel, nicht das Produkt. Die
korrekt zusammengesetzte Erwartung liegt bei `13,68 %`, nicht bei `21 %`.

**Zweiter Befund — `fixed_compiled` ist in diesem Arm fast wirkungslos.** Bei
`32` generierten Token trägt es `1,42 %` bei. Das liegt unter der sonst
verwendeten MDE von `5 %`; sein Einzelbeitrag ist in einer End-to-End-Studie
grundsätzlich nicht bestätigungsfähig. Es bleibt im Kandidatenprofil, weil es
kostenlos mitläuft, aber die Studie darf keinen Einzelnachweis dafür
beanspruchen.

**Sensitivität — der Gewinn sinkt mit der Antwortlänge:**

| generierte Token | Prefill-Anteil | Ratio | Gewinn |
| --- | --- | --- | --- |
| `8` | `94,06 %` | `0,851326` | `14,87 %` |
| `32` | `79,84 %` | `0,863161` | `13,68 %` |
| `128` | `49,75 %` | `0,888198` | `11,18 %` |
| `512` | `19,84 %` | `0,913084` | `8,69 %` |

Die registrierte Antwortlänge ist deshalb Teil des Vertrags: `32` Token, wie in
der persistenten Prozessstudie.

**Geltungsbereich der Zahl, ergänzt am 2026-09-02.** Der Gewinn des warmen
Arms hängt an der registrierten Antwortlänge und fällt mit ihr:

| generierte Token | kombinierter warmer Gewinn |
| --- | --- |
| `32` (registriert) | `13,68 %` |
| `128` | `11,18 %` |
| `256` | `9,80 %` — **unter der Schwelle dieser Studie** |
| `512` | `8,69 %` |

Bei `256` Token unterschritte derselbe Kandidat die eigene `10 %`-Schwelle und
die Studie meldete korrekt `below_threshold`. F1 bleibt damit gültig, ist aber
ausdrücklich eine Aussage über das kurze Antwortregime. Ob dieses Regime das
richtige Ziel ist, klärt die getrennte Studie W1
(`experiments/w1_regime/PREREGISTRATION.md`); F1 wird dafür nicht erweitert.

## 4. Vorregistrierte Schwellen

| Arm | Mindestgewinn `min_gain` | Begründung |
| --- | --- | --- |
| `warm` | `10 %` | Projektion `13,68 %`, Abstand zur Schwelle deckt Rauschen und Antwortlängen bis `256` Token |
| `cold` | `50 %` | Projektion `70,05 %`, großer Abstand, weil der Ladeeffekt robust ist |

MDE für die Gegenrichtung: `5 %`, wie in allen bisherigen Studien; sie wird
aus A/A-Sessions dieser Studie bestimmt und **nicht** aus der Projektion.

Entscheidungsregel, implementiert in
[`friday_optimizer/integration.py`](../friday_optimizer/integration.py) als
`evaluate_integration`:

- Konfidenzintervall der Ratio vollständig unter `1 − min_gain` → `qualified`;
- Intervall vollständig über `1 + mde` → `rejected` (bestätigte Regression);
- Intervall schneidet `1 + mde` → `inconclusive`;
- sonst → `below_threshold`, Baseline bleibt.

Weniger als sechs gültige Paare je Arm → `inconclusive`, ohne Zahl.

## 5. Gates

1. **Tokenidentität ist terminal.** Jede Anfrage vergleicht `token_sha256`
   zwischen Baseline und Kandidat. Eine einzige Abweichung beendet die Studie
   mit Baseline-Fallback; es gibt keine Toleranz und keine Nachverhandlung.
2. **Ressourcen- und Sicherheitsgate** wie in allen bisherigen Sessions:
   AC-Betrieb, kein Low-Power, kein Fremdlast, RSS-/Swap-Grenzen, Hardtimeout.
3. **Paarung** nach den Regeln des bestehenden Evaluators: balancierte
   `AB`/`BA`-Reihenfolge, eindeutige Paar-/Session-Identität, keine Duplikate.
4. **Provenienz:** sauberer Checkout, gebundener Fingerprint, Code-Manifest,
   Registry-Hash. `optimizer_identity()` verweigert bei dirty checkout — das
   ist Vorbedingung, kein Hinweis.

## 6. Kill- und Pivotkriterien

- Bleibt ein Arm über die geplanten Sessions unter seiner Schwelle, gilt
  Baseline; der Befund wandert als terminaler Negativeintrag in die Studienakte
  und F1 wird aus dem Backlog entfernt.
- Bricht Tokenidentität, endet die Studie sofort und terminal.
- Widerspricht die gemessene warme Ratio der Projektion um mehr als das
  Konfidenzintervall, ist **die Projektion** falsch und wird korrigiert — die
  Messung gewinnt. Der Widerspruch ist selbst ein Ergebnis und gehört ins
  Journal.

## 7. Was diese Studie ausdrücklich nicht ist

Kein Cross-Device-, Cross-Model- oder Qualitätsclaim. Keine automatische
Produktaktivierung: ein bestandenes Gate erlaubt Integration, nicht Aktivierung.
Kein Lernclaim — der Optimizer wählt hier nichts, das Kandidatenprofil ist
vorregistriert.

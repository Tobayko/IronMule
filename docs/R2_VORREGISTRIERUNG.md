# R2 — Vorregistrierung: der Explorationskorpus und was er beantworten darf

**Studien-ID:** `r2-corpus-20260904-01`
**Registriert:** 2026-09-04, vor dem ersten Messpunkt.
**Status bei Registrierung:** `optimizer-v2.sqlite3` enthält `404` Zeilen, davon
`402` mit `phase=feature` und **`2` mit `phase=label`**; `393` sind importierte
Altlasten. Der Lernkorpus ist leer, jeder Schätzer meldet `insufficient_data`.

---

## Die Frage

R2 im Backlog verlangt einen Korpus, auf dem eine gelernte Policy gegen
Random/Grid/BO per Off-Policy-Evaluation bewertet werden kann. Der Korpus
existiert nicht, und ohne ihn ist RL nicht widerlegt, sondern **unbeantwortet**.

Diese Studie erzeugt ihn: `400` gemessene Punkte über fünf zulässige Aktionen,
gezogen nach einer vor dem ersten Punkt versiegelten Regel.

**Sie beantwortet nicht, ob RL funktioniert.** Sie stellt die Datenlage her, auf
der diese Frage überhaupt entscheidbar wird — und liefert dabei den Korpus, den
das Kostenmodell und die Bayesian Optimization ohnehin brauchen.

## Die ehrliche Erwartung, vorab hingeschrieben

`docs/FABLE_ERFOLGSPFAD.md` (2026-09-01) begründet, warum RL hier das falsche
Werkzeug ist: einstufige Kandidatenwahl aus fester Allowlist ist ein Contextual
Bandit, kein MDP; Policy-RL braucht `10⁴`–`10⁶` Interaktionen, hier gibt es
`400`; der Reward ist teuer, verrauscht und zensiert. Ein negativer OPE-Befund
ist damit der **erwartete** Ausgang, nicht der überraschende.

Er wird trotzdem gemessen, weil das Kill-Kriterium sonst nie feuern kann und RL
auf unbestimmte Zeit als offene Möglichkeit weiterläuft. Und er ist billig zu
tragen: derselbe Korpus trägt die Auswertung, die dasselbe Dokument als den
tragfähigen Weg beschreibt.

---

## Aktionsraum — eingefroren

Die fünf für diesen Fingerprint zulässigen Kandidaten aus der versiegelten
Registry (`friday_optimizer/candidates.py`, Eignung in `_default_specs`):

| Aktion | Eignungsbedingung |
| --- | --- |
| `baseline` | immer zulässig |
| `head_skip_prefill` | interactive, greedy, keine Prompt-Logprobs |
| `persistent_process` | interactive, throughput |
| `fixed_compiled_cache` | interactive, throughput |
| `readback_every_2` | zusätzlich gebunden an gemma-3-4b, M1 Max, MLX `0.32.0` |

`combined_core_profile` bleibt draußen (unerfüllte Prerequisites), die
`throughput_width_*` ebenfalls (throughput-only). **Der Aktionsraum wird während
der Kampagne nicht erweitert.**

Der Aktionsraum von `friday_serve/rl_controller.py` ist ein **anderer** (sechs
Knopfbündel) und hat mit dieser Studie nichts zu tun.

## Logging-Policy — eingefroren

| Parameter | Wert |
| --- | --- |
| Regel | `epsilon_greedy` |
| ε | `0,6` |
| Hint | `head_skip_prefill` |
| `seed_base` | `20260904` |
| `campaign_id` | `r2-corpus-20260904-01` |
| Punkte | `400` |
| Modell | `mlx-community/gemma-3-4b-it-4bit` |

**Warum ε = 0,6.** Der Trockenlauf vom 2026-09-02 ist darauf kalibriert, und die
Größentabelle im Backlog hängt daran. Bei fünf Aktionen ergibt das
`p(Hint) = 0,52` und `p(andere) = 0,12`, also erwartete Ziehungen von `208` für
den Hint und je `≈48` für die übrigen — alle deutlich über der ESS-Untergrenze
`30`. Genau unter dieser Bedingung lieferte der Trockenlauf `5/5` belastbare
Schätzungen bei einem Medianfehler von `0,21` Punkten.

**Die Reihenfolge ist versiegelt, nicht die Aktion.** Jeder Punkt zieht seinen
Seed aus `campaign_hash` und seinem Index (`CampaignPlan.seed_for`). Die Sequenz
ist reproduzierbar und **nicht nachträglich umsortierbar**. Ein abgebrochener
Punkt wird beim Index neu gezogen und ergibt dieselbe Aktion.

## Reward — eingefroren

**Genau eine Metrik: `ratio_median`.** Gebildet aus der rohen `total_ns`-Serie,
die das Sessionergebnis je Paar und Arm ohnehin speichert:

```
je Paar   ratio = median(candidate.total_ns) / median(baseline.total_ns)
Reward    ratio_median = median(ratio über die Paare)
```

Gepaart zuerst, dann aggregiert. Kleiner als `1` heißt schneller.

**Nicht zulässig:** die Metrik aus den Phasenverhältnissen des Evaluators
(`ttft`, `decode_tps`) zusammensetzen. Das ist E04, die gemessene Sackgasse —
Phasenverhältnisse multiplizieren nicht, sie mitteln sich zeitgewichtet.

**Nicht zulässig:** `decode_ratio` als Reward. `replay.default_reward` rechnet
`1 − reward`, **ohne** `reward_metric` anzusehen; das ist für ein Zeitverhältnis
richtig und für einen Durchsatz vorzeichenverkehrt.

## Zensierung — eingefroren

Fehlgeschlagene Läufe werden geschrieben, nicht verworfen: sie zu entfernen
verzerrt jede spätere Schätzung zugunsten der Aktionen, die zufällig überlebten.

| Ursache | Wert |
| --- | --- |
| Readiness abgelehnt, Plan blockiert | `censored_gate_failed` |
| Deadline überschritten | `censored_timeout` |
| sonstiger Fehler, unbrauchbare Evidenz | `censored_error` |

**Erwartete Verzerrung, vorab benannt:** `ReplayEnv` führt zensierte Schritte mit
`censored_reward = 0,0`. Jede Schätzung wird dadurch um rund
`Zensurrate × wahrem Gewinn` zur Null gezogen — bei `5 %` Zensur und `13 %`
wahrem Gewinn also um etwa `0,65` Prozentpunkte. Das ist eine bekannte
Konservativität, kein Befund.

## Readiness — hergeleitet und eingefroren

`ReadinessGate` hat auf dieser Maschine nach Aktenlage nie bestanden. Gemessen
am 2026-09-04 mit der Vorgabepolicy, alle vier Gründe gleichzeitig:

```
foreign_workload_or_unknown, load_too_high, cpu_too_high, memory_reserve_too_low
```

Drei davon sind **Einheitenfehler auf macOS**, keine belegte Maschinenlast:

| Prüfung | Vorgabe | Was sie hier misst | Befund |
| --- | --- | --- | --- |
| `max_load_1m` | `0,75` roh | `0,75` auf zehn Kernen = `7,5 %` Auslastung | Leerlauf liegt bei `4,0`–`6,0` |
| `max_cpu_percent` | `35,0` | Summe über Prozesse, `100` = ein Kern | liegt unter **einem** belegten Kern |
| `min_memory_available_fraction` | `0,05` | freie Seiten aus `vm_stat` | macOS hält `2,14 %`, weil es freie Seiten als Cache ausgibt |
| `workload_active` | unbedingt | *jeder* aktive Nutzerprozess | trifft `modelmanagerd`, `SystemUIServer`, `BTLEServerAgent`, Browser-Renderer |

Gemessene Verteilung über `141` Proben in `12` Minuten
(`experiments/r2_readiness/idle.json`). **Das Fenster war nicht ruhig** — die
Testsuite lief mit; `min` und `p05` sind daher die Leerlaufschätzung, `median`
zeigt, wie belegt dieses Fenster war:

| Größe | min | p05 | Median | max |
| --- | ---: | ---: | ---: | ---: |
| Last **je Kern** | `0,460` | `0,486` | `0,691` | `3,652` |
| CPU summiert (`100` = ein Kern) | `105,9` | `127,5` | `178,8` | `963,5` |

**Eingefrorene Kampagnen-Policy:**

| Feld | Wert | Herleitung |
| --- | --- | --- |
| `max_load_1m` | `0,75` | **unverändert**, nur in der gemeinten Einheit |
| `normalize_load_by_cpus` | `True` | macht `0,75` zur Last je Kern |
| `max_cpu_percent` | `400,0` | vier Kerne — über dem Hintergrund dieser Maschine (`p05 127,5`, Median `178,8`), unter einer gesättigten (`963,5`) |
| `require_idle_workload` | `False` | die Erkennung trifft macOS-Dienste; die numerischen Grenzen tragen stattdessen |
| `min_memory_available_fraction` | `0,0` | Swap-**Wachstum** bleibt die Speichergrenze und wird getrennt geprüft |

**Das ist keine Absenkung, damit die Studie läuft.** G1 verbietet das
ausdrücklich, und `max_load_1m` bleibt bei seiner ursprünglichen Zahl. Geändert
wird die Einheit, in der sie gelesen wird, und zwei Prüfungen, die auf macOS
etwas anderes messen als ihren Namen. `ReadinessPolicy`s Vorgabewerte bleiben
unverändert; die Abweichung gilt nur für diese Kampagne und steht hier.

**Nachprüfbar:** mit dieser Policy meldet `check_readiness` auf dieser Maschine
`ready=True, reasons=[]`; mit der Vorgabe die vier Gründe oben.

## Holdout — eingefroren

Die **letzten 20 % der versiegelten Ziehungsreihenfolge**, Indizes `320`–`399`.

Die Reihenfolge steht vor jeder Messung fest, also ist der Split per Konstruktion
leckfrei und braucht keinen Umweg über `dataset.py`. Er wird **nicht**
nachträglich verschoben, auch nicht bei ungünstiger Aktionsverteilung im
Holdout.

## Entscheidungsregel — eingefroren

**Das Tor tragen `ips` und `doubly_robust`.** Beide liefern ein
Bootstrap-Konfidenzintervall.

`snips` wird als Stabilitätsvergleich berichtet und **entscheidet nichts**:
`replay._estimate` vergibt kein Intervall, sobald ein `normaliser` gesetzt ist,
und `snips` ist der einzige Schätzer mit einem. Das R2-Gate im Backlog verlangt
ein Konfidenzintervall; mit `snips` allein wäre es nicht erfüllbar. Diese
Festlegung löst den Widerspruch vor dem ersten Punkt, nicht nach der Auswertung.

`conclusive` prüft nur `status == "ok"` und einen vorhandenen Wert — **nicht**
das Intervall. Das Intervall wird deshalb getrennt berichtet und getrennt
gelesen.

**Bestanden**, wenn beide Bedingungen gleichzeitig gelten:

1. `ips` **und** `doubly_robust` melden `conclusive=true` auf dem Holdout, und
   das Konfidenzintervall der gelernten Policy liegt vollständig über dem der
   besten Vergleichspolicy aus Random/Grid/BO bei identischem Budget.
2. Die Invalid-Suggestion-Rate ist nicht schlechter als die der
   deterministischen Suche.

## Kill-Kriterium

Bleibt der OPE-Vorteil über Seeds und Holdouts aus, bleibt es bei Optimization
Memory plus deterministischer Suche plus BO. **RL bleibt NO-GO und wird nicht
als Abkürzung wiedereröffnet.**

Zusätzlich, vor der Kampagne: liefert der erste Block keine `10` gelabelten
Punkte, wird nicht skaliert. Die Ursache wird gefunden, bevor `39` weitere
Blöcke sie vervielfachen.

## Was diese Studie ausdrücklich nicht ist

- **Kein Lernclaim.** Ein Korpus ist keine Aussage. Bis zur Auswertung bleibt
  `learning_claim=false`.
- **Keine Aktivierung.** `no_activation` bleibt gesetzt; keine Policy schaltet
  einen Knopf.
- **Kein Cross-Device- und kein Cross-Model-Claim.** 4B, diese Maschine, dieser
  Engine-Commit.
- **Keine Absenkung einer Sicherheitsgrenze.** Budget, Duty-Cycle, Netzbetrieb
  und Tokenidentität bleiben unverändert, ebenso die Vorgabewerte von
  `ReadinessPolicy`.
- **Kein CQL/IQL.** Im Repository existieren `ips`, `snips`, `doubly_robust` und
  `replayer` — sonst nichts. Die Policy-Klasse bleibt klein und erklärbar;
  tiefe Netze sind unter `10⁴` Samples nicht begründbar.

---

# Amendment A — 2026-09-04, nach dem Pilotpunkt, vor der Kampagne

Zwei Pilotpunkte auf echter Hardware haben drei Annahmen der obigen Fassung
widerlegt. Es ist **nichts von der Kampagne gemessen**; die Änderungen sind
folgenlos und stehen hier, bevor der erste gezählte Punkt läuft.

## A1 — Messpfad: Kalibrier-Harness statt Session-Runner

Die obige Fassung ging von `friday_optimizer session` aus. Der Pfad kann keinen
Kampagnenpunkt fahren:

- `real_session.py:46` — `ALLOWED_CANDIDATE = "combined_core_profile"`,
  hartkodiert an drei Stellen (`:688`, `:870`, `:1212`). Er misst **einen**
  Kandidaten, und zwar den, den dieser Aktionsraum ausschließt.
- `--prereg` verlangt ein maschinenlesbares Artefakt mit sechzehn Hash-Feldern,
  nicht dieses Dokument.
- Der Pfad ist nie gelaufen; sein Readiness-Tor hat nie bestanden.

**Stattdessen:** `friday_calibrate.runner.build_runner` und `paired_arms` — der
Code, der das einzige echte Geräteprofil dieser Maschine erzeugt hat. Gepaart
AB/BA, Tokenidentität je Paar, Budget-Guard und Netzteil-Tor eingeschlossen.

Damit entfällt die Readiness-Policy aus dem Abschnitt „Readiness" für den
Messpfad: `paired_arms` durchläuft `ReadinessGate` nicht. Die dort hergeleiteten
Grenzen und die Stellschraube `require_idle_workload` bleiben gültig und
dokumentiert, werden von dieser Kampagne aber **nicht benutzt**. Was stattdessen
gilt: `require_ac_power()` und `BudgetGuard`, wie bei jeder Messung dieses
Projekts.

## A2 — `persistent_process` wird gezogen, aber nicht gemessen

`persistent_process` ist **kein Engine-Knopf** — `ironmule.runtime.Knobs` führt
es nicht. Und das Kalibrier-Harness hält den Prozess ohnehin über alle Paare
offen. Gegen `baseline` gemessen verglichen es zwei identische Konfigurationen
und lieferte per Konstruktion eine Ratio nahe `1,0`: ein **fabriziertes**
Null-Ergebnis, keine Messung.

Der Aktionsraum bleibt trotzdem unverändert, und das Siegel damit auch. Die
Ziehung kommt aus der versiegelten Registry und bildet die ehrliche
Zulässigkeit ab; sie zu beschneiden hieße, den Fingerprint zu verbiegen.

**Stattdessen:** ein gezogener `persistent_process`-Punkt wird **nicht gemessen
und bekommt keinen Outcome-Record.** Kein `not_run`, kein `censored_*` — gar
keinen.

Der Unterschied ist nicht kosmetisch. `ReplayEnv.reward_of` liefert `None` nur
dann, wenn ein Schritt überhaupt keinen Outcome hat, und `_weights` überspringt
genau solche Schritte (`replay.py:333`). Ein zensierter Record dagegen zählt als
Reward `0,0`, geht in `samples` ein und verwässert damit jede IPS-Schätzung um
den Anteil dieser Punkte — bei `≈48` von `400` also um rund `12 %`. Ein Record zu
schreiben hieße zu behaupten, es sei nichts beobachtet worden; kein Record zu
schreiben sagt korrekt, dass es keine Beobachtung gibt.

**Ergänzend eingefroren:** über `persistent_process` wird **keine Zielpolicy
ausgewertet.** Die vorregistrierten Ziele sind deterministische Punktmassen je
messbarer Aktion; für sie ist `distribution["persistent_process"] = 0`, ein
gezogener Punkt trüge also ohnehin das Gewicht null.

| Aktion | Knöpfe | gemessen |
| --- | --- | :---: |
| `baseline` | `{}` | ja |
| `head_skip_prefill` | `{head_skip_prefill: True}` | ja |
| `fixed_compiled_cache` | `{compiled_fixed_cache: True}` | ja |
| `readback_every_2` | `{readback_every: 2}` | ja |
| `persistent_process` | — | **nein** |

Gezogene Verteilung über die `400` versiegelten Punkte, nachgerechnet:
`head_skip_prefill 214`, `readback_every_2 49`, `persistent_process 48`,
`fixed_compiled_cache 46`, `baseline 43`. Gemessen werden also **`352` Punkte**,
und jede messbare Aktion bleibt deutlich über der ESS-Untergrenze `30`.

## A3 — Reward: die Paarung des Integrationspfads

Die Erstfassung nannte `median(candidate.total_ns) / median(baseline.total_ns)`
aus der Ergebnisdatei des Session-Runners. Ohne diesen Pfad gibt es die Serie
nicht. Stattdessen `integration.paired_request_ratios`: je Probe
`request_seconds = ttft + tokens / decode_tps`, je Paar das Verhältnis, darüber
der Median. Das sind die Paarungsregeln des Evaluators, dieselben, die F1 benutzt
hat.

Unverändert: **`ratio_median` bleibt die einzige Metrik**, kleiner als `1` heißt
schneller, `decode_ratio` bleibt gesperrt, und Phasenverhältnisse werden nicht
zusammengesetzt (E04).

## A4 — Gate-Schätzer: `doubly_robust` braucht ein Kostenmodell

`replay.evaluate` führt `doubly_robust` nur aus, wenn ein `reward_model`
übergeben wird, und `friday_optimizer replay` übergibt keines. Über die CLI ist
der Schätzer also nicht erreichbar.

**Präzisiert:** das Tor trägt **`ips`** — der einzige über die CLI erreichbare
Schätzer mit Bootstrap-Intervall. `doubly_robust` tritt hinzu, sobald die
Auswertung in Phase 3 ein Kostenmodell stellt; kann sie das nicht, ruht das Tor
auf `ips` allein, und das wird berichtet. `snips` bleibt Stabilitätsvergleich
ohne Stimmrecht.

Nachgeprüft: `_interval` liefert `(None, None)` erst unterhalb von **zwei**
Proben. Bei `400` Punkten hat `ips` sein Intervall; im Pilotlauf mit einer Probe
erwartungsgemäß nicht.

## A5 — Gemessene Punktkosten

`campaign.MEASURED_POINT_SECONDS` nimmt `167,0 s` an. Zwei Pilotpunkte auf dieser
Maschine, sechs Paare, 4B: **`179,3 s`** und **`178,5 s`**. Davon sind rund
`26,5 s` GPU-Arbeit und `148,3 s` vorgeschriebene Pause — die Kapazität hängt an
der Abkühlung, nicht am Takt.

**Nachtrag nach dem ersten Fenster (68 Punkte):** gemessener Median **`188 s`**.
Der Aufschlag gegenüber den Piloten ist erklärbar — während des Fensters liefen
CI-Reparaturen auf derselben Maschine. Die Paarung fängt das ab (beide Arme eines
Paares laufen unter denselben Bedingungen); für die Wandzeitplanung gilt der
höhere Wert.

Nach A2 werden `352` der `400` Punkte gemessen. Bei `188 s` sind das rund
**`18,4` Stunden**. Die Blockzahl `40` bleibt die Obergrenze.

`MEASURED_POINT_SECONDS` selbst bleibt **unverändert**: die Konstante geht über
`CampaignPlan.as_dict` in den `campaign_hash` ein, und ein geänderter Wert wäre
ein anderes Siegel — die bereits gemessenen Punkte gehörten dann zu keiner
Kampagne mehr. Die gemessene Zahl steht hier, nicht im Code.

## A6 — Die Pilotpunkte bleiben Kampagnenpunkte

Weil A2 das Siegel nicht ändert, sind beide Pilotpunkte gültige Punkte dieser
Kampagne.

- Punkt `0` (`readback_every_2`, `ratio_median 0,9842`) wurde gemessen und
  **nicht** geschrieben. Er wird im regulären Lauf an seinem Index neu gezogen —
  derselbe Seed, dieselbe Aktion — und dann geschrieben.
- Punkt `1` (`r2-corpus-20260904-01.0001`, `head_skip_prefill`,
  `ratio_median 0,8760`) ist geschrieben und zählt.

Punkt `1` ist zugleich eine unabhängige Gegenprobe: das Geräteprofil führt
`head_skip` mit Ratio `0,8778`, KI `[0,8668; 0,8888]`, gemessen am 2026-09-03
durch das Kalibrier-CLI. Der hier über den Kampagnenpfad gemessene Wert `0,8760`
liegt in diesem Intervall. Zwei getrennte Aufrufwege, dasselbe Ergebnis.

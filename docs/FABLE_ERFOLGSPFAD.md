# Fable-Erfolgspfad

Festgehalten am 2026-09-01 auf Nutzerauftrag. Zweck: andere Sessions sollen
ohne Neuherleitung wissen, wo das Projekt steht, was freigegeben ist und
welche Hebel zum Erfolg führen. Startpunkt bleibt `../BACKLOG.md` (Eintrag F1
verweist hierher).

## Standortbestimmung (2026-09-01)

- Als Methodik-/Evidenzprojekt bereits erfolgreich: Vorregistrierung, Seals,
  Hashketten, ~1400 Tests, belastbare Negativbefunde.
- Als Performance-Projekt real, aber gedeckelt: Inferenz ist speicherbegrenzt
  (36–58 % Bandbreitennutzung laut Roofline). Kleine Compute-Hebel sind
  abgegrast (Custom Kernel `1,87 %`, gebündelter Readback `4,19 %`,
  Fused-Greedy inconclusive, Draft-Spec `0,56x`).
- Als Self-Learning-Optimizer (L1) stand jetzt unrealistisch: Korpus
  `train=2, val=0, holdout=0`; benötigt Hunderte saubere Messungen bei
  manuellem 30-Minuten-Durchsatz. Beide Gemma-Planer scheiterten zweimal am
  Format-Vertrag (`0/6`).
- Kernbefund: die großen Gewinne sind formal bestätigt, aber nicht
  integriert — persistenter Prozess `−65 %`, Head-Skip `−15,4 %`,
  `fixed_compiled` `−7 %` Decode, Batch-Dispatch `−12 %`. Das Projekt misst
  exzellent und erntet nicht. Der nächste Erfolg liegt in Integration, nicht
  in neuer Einzelmessung.

## Nutzerfreigabe vom 2026-09-01 (F1 — Integrationspfad)

Der Nutzer gibt den Integrationspfad frei, damit das Projekt zum Erfolg
kommt: **eine vorregistrierte Integrationsstudie**, die die bereits einzeln
bestätigten Gewinne — persistenter Modellprozess, Prefill-Head-Skip und
`fixed_compiled` — gemeinsam auf einem realen End-to-End-Anfragepfad
(Gemma 4B) misst.

Rahmen der Freigabe (unverändert gültige Grenzen):

- Offline-Implementierung, Preregistrierung und Tests sind freigegeben.
- Reale Hardwareläufe bleiben gate-basiert: manuell gestartet, AC-only,
  fremdlastfrei, maximal 30 Minuten je Lauf, jeder Lauf einzeln bestätigt.
- Downloads und Installationen bleiben gesperrt.
- Exakte Token-/Textidentität ist terminales Gate; jede Verletzung fällt auf
  Baseline zurück.
- Automatische Produktaktivierung bleibt bis zum bestandenen Promotionsgate
  gesperrt; diese Freigabe hebt kein bestehendes NO-GO außerhalb von F1 auf.

Erfolgsdefinition: eine End-to-End-Zahl statt weiterer Einzelstudien —
TTFT und Decode-Tokens/s eines realen 4B-Chat-Requests, integriert gegen
Standardpfad, mit vorregistrierter Mindestschwelle. Erwartung aus den
Einzelbefunden: kombiniert 20–70 % je nach Szenario (warm/kalt).

Kill/Pivot für F1: bleibt der integrierte Pfad über vorregistrierte Sessions
unter der Schwelle oder bricht Tokenidentität, gilt Baseline und der Befund
wandert als terminaler Negativeintrag in die Studienakte.

## Weitere Hebel Richtung Erfolg (priorisiert)

1. **Ernten vor Neumessen.** Erst F1 abschließen, dann neue Kandidaten.
2. **Bereits empfohlene Kandidaten aus der Studienakte**
   (`KANDIDATENLISTE.md`): Host-Readback aufschieben (Kandidat 17,
   `candidate_recommended_for_preregistration`) und
   KV-Cache-Reallokationen (Kandidat 21, ebenfalls empfohlen). KV-Cache
   fester Form (Kandidat 13) ist durch Zyklus 16 faktisch vorvalidiert.
3. **Bandbreiten-Hebel statt Compute-Hebel.** Die Roofline sagt
   speicherbegrenzt: KV-Cache-Quantisierung und kleinere KV-Dtypes sind die
   Klasse mit theoretischem Spielraum; als eigene vorregistrierte Studie mit
   Qualitäts- statt nur Identitätsgate (verändert Zahlenwerte).
4. **Messdurchsatz erhöhen, sonst bleibt L1 tot.** Gebatchte Freigaben:
   ein manuell gestarteter, gegateter 30-Minuten-Block darf mehrere
   vorregistrierte Messpunkte seriell abarbeiten statt einen. Gleiche
   Guards, mehr Datenpunkte pro Freigabe — einziger realistischer Weg zu
   einem Lernkorpus mit `val>0`/`holdout>0`.
5. **Studienkosten senken.** Gemeinsame Lib `friday_evidence` (Regel steht
   in `AGENTS.md`) plus ein Studien-Template, damit eine neue Studie Tage
   statt Wochen kostet.
6. **Planner nur mit constrained decoding.** Der zweimalige `0/6`-
   Vertragsfehler ist ein Sampling-Problem, kein Modellproblem: Ausgabe
   per Grammatik/JSON-Schema beim Sampling erzwingen, nicht per Prompt
   erbitten. Ohne das bleibt jeder LLM-Planner NO-GO.
7. **L1 ehrlich halten.** Solange der Korpus leer ist: deterministische
   Suche ist das Ergebnis, kein Learned Ranking behaupten (Kill-Kriterium
   steht im Backlog).

## Reinforcement Learning — fachliche Einordnung (2026-09-01)

Das bestehende RL-NO-GO im Backlog ist fachlich richtig. Begründung und die
Bedingungen, unter denen es je kippen dürfte:

**Warum klassisches RL hier das falsche Werkzeug ist**

1. **Falsche Problemstruktur.** Kandidatenauswahl aus fester Allowlist mit
   sofortiger Messung ist ein *Contextual Bandit* (Ein-Schritt-Entscheid),
   kein sequentielles MDP. Es gibt keinen Zustandsübergang, den eine Policy
   lernen müsste — Bellman/Policy-Gradient lösen ein Problem, das hier nicht
   existiert. Die korrekte Formalisierung ist Bandit + Bayesian Optimization,
   und BO steht bereits im L1-Plan.
2. **Sample-Effizienz.** RL braucht typischerweise 10⁴–10⁶ Interaktionen.
   Der Korpus hat `train=2`; der Durchsatz sind manuell freigegebene
   30-Minuten-Läufe. GP-BO/GBDT liefern ab ~50–100 sauberen Punkten Nutzen,
   Policy-RL in diesem Regime nie.
3. **Reward ist teuer, verrauscht, zensiert.** Messrauschen nahe der MDE,
   Timeouts und Compilerfehler erzeugen zensierte Rewards; RL degeneriert
   damit zu Random Search mit Extraschritten. Zudem Reward-Hacking-Risiko:
   eine Policy lernt Messartefakte (Thermik, Reihenfolge) auszunutzen —
   genau die Störgrößen, die das Messprotokoll mühsam kontrolliert.
4. **Stand der Praxis im Autotuning.** AutoTVM, Ansor, TenSet und Verwandte
   nutzen gelernte *Kostenmodelle plus Suche*, nicht Policy-RL. Ein Surrogat
   (GBDT/GP) mit Akquisition (Thompson Sampling oder UCB) liefert die
   Exploration, die man von RL erwartet, mit Größenordnungen weniger Samples
   und bleibt erklärbar.

**Was stattdessen funktioniert (der Lernpfad, der Erfolg hat)**

- Zweistufig, wie im L1-Plan: erst Machbarkeit klassifizieren, dann relative
  Performance ranken; zensierte Läufe als Censored-Label behandeln, nicht
  verwerfen.
- Akquisition statt Policy: Thompson Sampling oder UCB über dem
  Kostenmodell wählt den nächsten Messpunkt — das ist die seriöse Form von
  „Exploration" für dieses Budget.
- Features shape-/parametergenerisch bauen, damit jede teure Messung für
  viele künftige Entscheidungen zählt (Transfer innerhalb des Geräts; kein
  Cross-Device-Claim).
- Messdurchsatz zuerst (Hebel 4): ohne `val>0`/`holdout>0` ist jede
  Lernaussage leer.

**Wann RL doch — drei Bedingungen, alle gleichzeitig**

1. Bandit+BO plateauiert nachweislich über mehrere Seeds/Holdouts (das
   bestehende Kill-Kriterium, umgekehrt gelesen);
2. es existiert ein echter sequentieller Aktionsraum, in dem Entscheidung A
   den Raum von B verändert, nicht nur eine flache Kandidatenliste;
3. Interaktionen sind billig, weil ein validiertes Kostenmodell oder ein
   Simulator den Großteil der Rollouts trägt und echte Hardware nur
   verifiziert.

**Wo RL in diesem Projekt gut wäre (konkrete Einsatzorte)**

- **Mehrstufige Fusions-/Compile-Ketten:** erst wenn Kandidaten aus
  sequentiellen Entscheidungen bestehen (Op-Grouping → Template →
  Parameter, wobei jeder Schritt den Folgeraum verändert — die
  IronMule-Grouping-Achse ist der natürlichste Kandidat). Das ist ein
  echtes MDP mit kurzem Horizont (3–5 Schritte).
- **Messbudget-Allokation:** welcher Messpunkt als nächstes bei knappem
  Restbudget (Optimal Stopping über eine Kampagne). Grenzfall zwischen
  Bandit und RL; lohnt erst bei Kampagnen mit vielen Punkten je Freigabe.
- **CPU/GPU-Placement über Teilgraphen** (L1.5-Erweiterung): sequentielle
  Placement-Entscheide mit Interaktionseffekten.
- **Nicht geeignet bleiben:** flache Einzelkandidatenwahl (Bandit/BO),
  Promotionentscheide (deterministisch per Vorregistrierung, niemals
  gelernt), Toleranz-/Schwellwertwahl (verboten per Sicherheitsgrenze).

**Wie RL umgesetzt werden müsste (phasiert, jede Phase mit Gate)**

- **R0 — heute, kostenlos, ohne RL zu bauen:** jede Entscheidung des
  deterministischen Tuners RL-ready loggen: Kontext-Features, vollständige
  Kandidatenmenge, gewählte Aktion, Auswahlregel und — bei stochastischer
  Auswahl — die Auswahlwahrscheinlichkeit (Propensity), dazu Reward mit
  Zensierungsstatus. Ohne Propensities ist spätere Off-Policy-Evaluation
  unmöglich; mit ihnen entsteht der Offline-RL-Korpus als Nebenprodukt
  jeder normalen Messung. Gehört als Schemafeld ins Optimization Memory v2.
- **R1 — Replay-Umgebung, offline:** Gym-artige Env über dem Optimization
  Memory (Zustand = Kontext-Features, Aktionen = Allowlist mit
  Action-Masking, Reward = gemessene Ratio; zensierte/fehlgeschlagene Läufe
  als eigene terminale Rewards, nie verworfen). Kein Hardwarelauf; Env-Tests
  wie jede Studienkomponente.
- **R2 — Offline-RL auf dem Korpus:** konservative Verfahren, die ohne
  Live-Exploration auskommen (CQL/IQL-Klasse); Policy-Klasse klein und
  erklärbar (linear oder Baum über den vorhandenen Features — tiefe Netze
  sind unter 10⁴ Samples nicht begründbar). Bewertung ausschließlich per
  Off-Policy-Evaluation (Doubly-Robust/gewichtete Importance-Sampling-
  Schätzer) gegen Random/Grid/BO unter identischem Budget, auf
  vorregistriertem Holdout. Gate: OPE-Vorteil mit Konfidenzintervall,
  keine schlechtere Invalid-Suggestion-Rate.
- **R3 — modellbasierte Rollouts:** das qualifizierte Kostenmodell aus L1
  dient als Simulator (Dyna-Stil): Rollouts fast vollständig im Modell,
  echte Hardware verifiziert nur die Top-K-Empfehlungen. Voraussetzung:
  kalibrierte Unsicherheit des Kostenmodells; hohe Unsicherheit maskiert
  die Aktion.
- **R4 — Shadow, dann gegatete Realmessung:** identisch zur bestehenden
  L1-Architektur — Policy empfiehlt nur, deterministische Correctness-,
  Ressourcen- und Promotionsgates führen aus; OOD/Unsicherheit erzwingt
  `no_recommendation`; Circuit Breaker und Baselinefallback unverändert.
  Jede reale Messung bleibt einzeln freigegeben.

**Umsetzungsstand (2026-09-01).** R0 und R1 sind offline gebaut, getestet und
bedienbar; Details im Arbeitsjournal unter „2026-09-01 — R0 und R1".

- R0 → `friday_optimizer/decisions.py`: `DecisionEvent` (Kontext,
  Kandidatenmenge, Aktion, Regel, Propensity, Seed) und `OutcomeEvent`
  (Reward plus Zensierungsstatus) als Feature-/Label-Records in Optimization
  Memory v2. Kein neuer SQL-`kind` — die Schema-SQL wird byteexakt geprüft,
  darum versioniertes Payload-Schema unter `RecordKind.SYSTEM`.
- R1 → `friday_optimizer/replay.py`: `ReplayEnv` mit striktem Action-Masking,
  nicht imputierten Kontrafakten und nie positivem Zensierungsreward; dazu
  `ips`, `snips`, `doubly_robust`, `replayer`, effektive Stichprobengröße und
  ein Ehrlichkeitsgate bei `DEFAULT_MIN_SAMPLES = 30`.
- Bedienung: CLI `decide` / `outcome` / `replay`, Orchestrator-Methoden
  `select` / `record_outcome` / `replay` / `evaluate_policy`, read-only
  Dashboard-Panel unter `/api/decisions`.
- Offen bleibt R2 (Backlog-Eintrag). Blocker ist der Korpus, nicht der Code:
  ohne Überlappung meldet jeder Schätzer `insufficient_data`. Eine rein
  deterministische Loggingpolicy erzeugt keine Überlappung — ein Teil der
  Entscheidungen muss mit `epsilon_greedy` und protokolliertem Seed fallen.

**Invarianten für jede RL-Stufe** (aus den bestehenden Sicherheitsgrenzen):
Policy-Output ist unvertrauenswürdige Eingabe; kein Schreibzugriff auf
Schwellwerte, Baselines oder Historie; Action-Masking strikt auf die
versiegelte Allowlist; Reward ausschließlich aus gateten Messungen — nie aus
Modellschätzungen — für jeden Promotionsentscheid.

Bis R0–R2 belegt sind, gilt: Optimization Memory + deterministische Suche +
BO ist der Lernpfad; RL bleibt NO-GO und wird nicht als Abkürzung
wiedereröffnet. Einzig R0 (Propensity-Logging) sollte sofort in jedes neue
Schema, denn es kostet nichts und hält alle späteren Optionen offen.

## Nicht wieder anfassen (gemessene Dead-Ends)

Draft-Model Speculative Decoding (`0,56x`), Custom Metal Kernel (kein
Einzelengpass), Token-Cache für Präfixe (kein Anteil), altes
Device-Model-`mx.compile` (falsche Token ab Position 2), gebündelter
Readback (`4,19 %` unter Schwelle), Fused-Greedy-Compile (inconclusive),
freier Prompt-basierter Planner-Vertrag (`0/6`, zweimal).

# Mini-Vorregistrierung — Gemma-4B als evidenzgebundener Planer

**Kandidaten-ID:** `gemma-4b-evidence-planner-20260824-01`  
**Zyklus:** `14`  
**Status:** vor jeder Hardwaredatei geschrieben; Schwellen ab jetzt unveränderlich  
**Claim:** `formal_claim=false`

## Genau ein Kandidat und enger Scope

Geprüft wird ausschließlich, ob das bereits lokal vorhandene Gemma-3-4B-Modell aus
einer festen, aus bisherigen Project-Friday-Ergebnissen gebildeten Liste genau den
nächsten sinnvollen Versuch auswählen kann. Das Modell schlägt nur eine vorhandene
Kandidaten-ID vor. Es erzeugt keinen Code, ändert keine Datei, startet keinen
Kandidaten und entscheidet weder über Korrektheit noch Aktivierung.

Nicht geprüft werden der 1B-Planer, ein Modellvergleich, Modelltraining,
Gewichtsänderungen, automatische Produktaktivierung oder die Leistung des vom
Planer ausgewählten Kandidaten.

## Fester Planungsfall

- Gerät: lokaler Apple M1 Max mit 32 GB, Netzbetrieb Pflicht.
- Software: projektlokale `.venv`, MLX `0.32.0`, mlx-lm `0.31.3`.
- Modell: `mlx-community/gemma-3-4b-it-4bit`, ausschließlich lokaler Snapshot
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2` über
  `tools/_bench.py:resolve_local_model_snapshot`.
- Sampling: greedy, `temperature=0`, höchstens `32` Ausgabetoken.
- Drei frische Python-/Modellprozesse erhalten bytegleich denselben fest
  eingebetteten Planungsfall. Zwischen ihnen wird kein Modell- oder KV-Zustand
  geteilt.
- Der Planungsfall enthält genau diese vier Möglichkeiten:
  `persistent_service_qualification`, `batched_readback`,
  `host_readback_upper_bound`, `kv_cache_preallocation_ab`.
- Fest eingebettete Evidenz: Der persistente Prozess sparte im prospektiven
  Einzelrequest-Scope gerechnet `65,3032 %` bei exakten Token; Multi-Turn und
  parallele Anfragen fehlen. Gebündelter Readback lokalisierte `12,98 %` je
  Decodetoken, kann aber zusätzliche Token erzeugen. Host-Readback ist nur eine
  Obergrenze. KV-Vorallokation lokalisiert `4,4263 %` korrelierte Decodezeit und
  braucht eine eigene Architekturfreigabe.
- Feste Prioritätsregel im Prompt: bestätigten größten End-to-End-Hebel wählen,
  der zugleich eine fehlende Pflicht-Workload schließt; reine Obergrenzen und
  freigabepflichtige Cacheeingriffe nicht wählen.
- Erlaubte Antwort ist ausschließlich das strikte JSON-Objekt
  `{"candidate_id":"persistent_service_qualification"}`.

Die erwartete Auswahl ist vorab festgelegt und wird nach Sicht der Modellantwort
nicht geändert. Der Test zeigt nur, ob dieser eine geschlossene Planungsfall
funktioniert; er belegt keine allgemeine Planungsfähigkeit.

## Vorab festgelegter Ablauf

1. Vor dem ersten Modellprozess wird eine einmalige private Startmarke geschrieben.
   Existiert sie bereits, verweigert das Werkzeug jeden weiteren Hardwarelauf.
2. Der Elternprozess startet nacheinander genau drei frische Worker. Jeder Worker
   prüft Modell-ID und Snapshot, lädt das Modell genau einmal, erzeugt greedy und
   gibt Token-IDs, dekodierten Text, Abschlussgrund, Laufzeit und Speicherwerte als
   streng geprüftes JSON-Ereignis zurück.
3. Der Elternprozess stoppt jede Modelldauer vor dem anschließenden
   `BudgetGuard.record_gpu()`. Ruhezeiten des Guards gehören nie zur Modelldauer.
4. Alle drei Rohantworten bleiben erhalten. Es werden keine Ausreißer verworfen,
   gekürzt, ersetzt oder nachträglich umgedeutet.

## Hypothesen und feste Schwellen

**H1 (greedy Korrektheit und Wiederholbarkeit).** Alle drei frischen Prozesse
liefern exakt dieselbe Folge von höchstens `32` Token, denselben Text und denselben
Abschlussgrund `stop`. Jede PID ist verschieden, jeder Worker lädt das gebundene
Modell genau einmal. Ein Tokenmismatch ist `correctness_failed` und terminal.

**H2 (Antwortvertrag).** Jede der drei Antworten ist ohne Markdown, Vorspann oder
Zusatztext als striktes JSON parsebar. Das Objekt besitzt genau den Schlüssel
`candidate_id`; dessen Wert ist eine der vier fest eingebetteten IDs. Ein Verstoß
ist `planner_contract_failed`, terminal.

**H3 (Priorität).** Alle drei Antworten wählen exakt
`persistent_service_qualification`. Eine andere gültige ID ist
`planner_priority_failed`, ein gültiges negatives Ergebnis.

**H4 (Ressourcen).** Jeder Worker bleibt bei höchstens `5 GiB` Prozess-RSS und
höchstens `5 GiB` MLX-Peak. Der systemweite Swap darf nicht wachsen; ein unbekannter
Swapwert gilt als nicht bestanden.

**H5 (Schutzbudget).** Jeder Modellabschnitt läuft unter `BudgetGuard`, am
Netzteil, mit Duty-Faktor `0,15`, höchstens `6 s` zusammenhängender Modellarbeit,
höchstens `120 s` Modellarbeit und höchstens `1200 s` Gesamtzeit.

## Abbruchregeln

- Fehlende lokale Dateien, falsche Revision, schmutziger eigener Git-Arbeitsstand,
  Akkubetrieb oder vorhandene Startmarke beenden vor der Hardware.
- Workerfehler, Timeout, ungültiges Ereignis, Tokenmismatch, unbekannter Swap oder
  Budgetverletzung beenden den Lauf terminal; Teilergebnisse und Fehlergrund
  bleiben erhalten.
- Ein gestarteter Hardwarelauf wird weder im selben Prozess noch mit derselben
  Studie wiederholt.
- Antwortschema, erwartete ID, Prozesszahl, Schwellen und Entscheidungstabelle
  werden nach Sicht der Antworten nicht geändert.

## Vorab festgelegte Entscheidungstabelle

| H1 | H2 | H3 | H4/H5 | Entscheidung |
| :--- | :--- | :--- | :--- | :--- |
| scheitert | egal | egal | egal | `correctness_failed`, terminal |
| hält | scheitert | egal | egal | `planner_contract_failed`, terminal |
| hält | hält | scheitert | hält | `planner_priority_failed` |
| hält | hält | hält | scheitert | `resource_or_budget_failed`, terminal |
| hält | hält | hält | hält | `planner_4b_qualified_exact_case` |

Auch die letzte Zeile bleibt `formal_claim=false`. Sie erlaubt weder eine
automatische Kandidatenausführung noch eine Produktaktivierung.

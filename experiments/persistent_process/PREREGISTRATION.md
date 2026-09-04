# Mini-Vorregistrierung — persistenter Modellprozess

**Kandidaten-ID:** `persistent-process-20260824-03`  
**Zyklus:** `13`  
**Status:** vor jeder Hardwaredatei geschrieben; Schwellen ab jetzt unveränderlich  
**Claim:** `formal_claim=false`

## Ein Kandidat und enger Scope

Geprüft wird ausschließlich, ob das bereits verwendete Modell zwischen Anfragen im
Speicher bleiben darf. Arm A startet für jede Anfrage einen frischen Pythonprozess,
lädt das Modell, beantwortet genau eine Anfrage und endet. Arm B lädt dasselbe
Modell einmal und beantwortet danach mehrere Anfragen. Beide Arme verwenden für
jede Anfrage einen frischen KV-Cache und denselben unveränderten Modellpfad.

Nicht Teil dieses Zyklus sind Head-Skip, Präfixwiederverwendung, Readback-Bündelung,
KV-Cache-Umbau, ein anderes Modell, Quantisierungsänderungen oder automatische
Produktaktivierung.

## Fester Workload

- Gerät: lokaler Apple M1 Max mit 32 GB, Netzbetrieb Pflicht.
- Software: projektlokale `.venv`, MLX `0.32.0`, mlx-lm `0.31.3`.
- Modell: `mlx-community/gemma-3-4b-it-4bit`, ausschließlich Snapshot
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2` über
  `tools/_bench.py:resolve_local_model_snapshot`.
- Prompts: die im Worker fest eingebetteten Schlüssel `P`, `Q`, `R`; gemeinsamer
  fester Vorspann mal `40`, danach je eine feste technische Frage. Erwartet werden
  je `897` Prompt-Token. Der Warm-up-Prompt `S` wird nie ausgewertet.
- Ausgabe: `32` Token, greedy, `temperature=0`, keine Prompt-Logprobs.
- Prefill-Chunk: `256`; Batch: `1`.
- Für jede Anfrage wird ein frischer KV-Cache erzeugt.

## Vorab festgelegter Ablauf

1. Vor dem ersten Worker wird eine einmalige Startmarke geschrieben. Existiert sie
   bereits, verweigert das Werkzeug jeden weiteren Hardwarelauf.
2. A/A-Kalibrierung: zwei Paare aus je zwei frischen Prozessen, Prompts `P`, `Q`.
   Es werden keine Werte verworfen.
3. Charakterisierung: ein neuer warmer Worker, ein nicht gewerteter Warm-up `S`,
   danach drei gepaarte Vergleiche `P`, `Q`, `R` in Reihenfolge `AB`, `BA`, `AB`.
4. Validierung: ein zweiter neuer warmer Worker, ein nicht gewerteter Warm-up `S`,
   danach drei gepaarte Vergleiche `R`, `P`, `Q` in Reihenfolge `BA`, `AB`, `BA`.
5. Arm A startet für jeden Messpunkt einen neuen Prozess. Arm B bleibt nur innerhalb
   seiner vorab benannten Phase bestehen. Zwischen Anfragen darf kein Cache geteilt
   werden.

TTFT wird außerhalb des Workers gestoppt: bei Arm A vom Start des frischen Prozesses,
bei Arm B vom Absenden der Anfrage, jeweils bis zum Empfang des ersten Tokens. Die
gesamte Modellarbeitsdauer wird ebenfalls vor dem anschließenden
`BudgetGuard.record_gpu()` gestoppt. Ruhezeiten des Guards gehören nie zur TTFT.

## Hypothesen und feste Schwellen

**H0 (A/A-Kalibrierung).** Alle A/A-Token sind identisch. Jedes Verhältnis des
zweiten zum ersten kalten Prozess liegt in `[0,80; 1,25]`; der Median liegt in
`[0,90; 1,10]`.

**H1 (Korrektheit und Pfad).** Alle sechs kalten/warmen Paare liefern exakt dieselben
`32` Token. Jeder kalte Messpunkt hat eine neue PID. Je Phase lädt der warme Worker
das Modell genau einmal, meldet vor der ersten gewerteten Anfrage Bereitschaft und
erzeugt für jede Anfrage einen frischen KV-Cache. Ein Tokenmismatch ist terminal.

**H2 (Wirkung).** Das Verhältnis `warm_TTFT / kalt_TTFT` ist in Charakterisierung,
Validierung und über alle sechs Paare jeweils im Median höchstens `0,50`. Zusätzlich
muss jedes einzelne Paar höchstens `0,65` erreichen. Keine Ausreißer werden
verworfen oder ersetzt.

**H3 (Ressourcen).** Der warme Worker bleibt bei höchstens `5 GiB` Prozess-RSS. Sein
RSS darf nach dem nicht gewerteten Warm-up bis zum letzten gewerteten Request um
höchstens `256 MiB` wachsen. Der systemweite Swap darf nicht wachsen; ein unbekannter
Swapwert gilt als nicht bestanden.

**H4 (Schutzbudget).** Jeder Hardwareabschnitt läuft unter `BudgetGuard`, am
Netzteil, mit Duty-Faktor `0,15`, höchstens `6 s` zusammenhängender Modellarbeit,
höchstens `120 s` Modellarbeit und höchstens `1200 s` Gesamtzeit.

## Abbruchregeln

- Fehlende lokale Dateien, falsche Revision, schmutziger eigener Git-Arbeitsstand,
  Akkubetrieb oder eine bereits vorhandene Startmarke beenden vor der Hardware.
- Scheitert H0, werden Charakterisierung und Validierung nicht gestartet.
- Tokenmismatch, Protokollfehler, Workerfehler, Timeout, unbekannter Swap oder
  Budgetverletzung beenden den Lauf terminal.
- Ein fehlgeschlagener Hardwarelauf wird weder im selben Prozess noch mit derselben
  Studie wiederholt. Teilergebnisse und Fehlergrund bleiben erhalten.
- Schwellen, Prompts, Reihenfolgen und Stichprobenzahlen werden nach Sicht der Werte
  nicht geändert.

## Vorab festgelegte Entscheidungstabelle

| H0 | H1 | H2 Charakterisierung | H2 Validierung | H3/H4 | Entscheidung |
| :--- | :--- | :--- | :--- | :--- | :--- |
| scheitert | egal | nicht gestartet | nicht gestartet | egal | `calibration_failed`, terminal |
| hält | scheitert | egal | nicht fortsetzen | egal | `correctness_failed`, terminal |
| hält | hält | scheitert | nicht gestartet | hält | `candidate_characterized_no_gain` |
| hält | hält | hält | scheitert | hält | `candidate_not_confirmed` |
| hält | hält | hält | hält | scheitert | `resource_or_budget_failed`, terminal |
| hält | hält | hält | hält | hält | `engineering_gain_confirmed_exact_scope` |

Auch die letzte Zeile bleibt `formal_claim=false`. Eine spätere Aktivierung in einem
normalen Dienst ist eine getrennte Architekturentscheidung.

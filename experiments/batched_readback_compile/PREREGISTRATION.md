# Zyklus 17 — gebündelter Readback auf dem validierten Fixed-Compiled-Pfad

**Studien-ID:** `fixed-compiled-batched-readback-20260824-01`  
**Lauf-ID:** `fixed-compiled-batched-readback-validation-20260824-01`  
**Kandidat:** `fixed_compiled_batched_readback_n8_v1`  
**Status:** vor jeder Messung geschrieben; `formal_claim=false`

Diese Präregistrierung bindet einen einzigen neuen Kandidaten. Sie wird vor dem
ersten Hardware- oder Modellstart geschrieben und danach nicht mehr verändert.
Es gibt keinen Download, keine Installation, keine Gewichts-, Modell- oder
Quantisierungsänderung, keinen Push, keinen Produktivdienst und keine automatische
Aktivierung. Die erteilte Nutzerfreigabe gilt genau für diese eine Studie.

## Ausgangspunkt und Abgrenzung

Cycle 6 maß explorativ `15,3 %` für vollständig entfallenden Host-Readback. Dieser
Arm konnte nicht anhalten und ist daher nur eine Obergrenze. Cycle 7 maß explorativ
bei `N=8` `12,98 %` gegenüber `N=1`; beide Läufe verwendeten den normalen
MLX-Cache, feste Schrittzahl und keine Cycle-16-Compile-Umgebung. Diese Werte sind
Vorwissen, keine neue Schwelle und kein Beleg für Cycle 17.

Cycle 17 prüft ausschließlich die Readback-Häufigkeit innerhalb des bereits
validierten Cycle-16-`fixed_compiled`-Pfades. Matmul bleibt aktiv. Die beiden Arme
verwenden denselben Fixed-Cache, dieselbe `mx.compile`-Funktion, dieselbe
Mathematik und dieselbe MLX-/Modellkonfiguration. Die einzige experimentelle
Variable ist, ob der erzeugte Token je Schritt (`N=1`) oder nach jeweils acht
Schritten (`N=8`) zum Host gelesen und auf EOS geprüft wird.

## Hypothesen

**H1 — Geschwindigkeit.** Der Kandidatenarm `fixed_compiled_readback_8` ist im
gesamten Decode-Kritischen-Pfad schneller als der Baselinearm, wenn der gepaarte
Ratio-Median `candidate / baseline` höchstens `0,95` beträgt und die obere
gepaarte Bootstrap-95-%-Konfidenzgrenze strikt unter `1,0` liegt.

**H2 — Identität.** Beide Arme erzeugen dieselbe logische greedy-Tokenfolge bis
einschließlich EOS beziehungsweise bis zur physischen Obergrenze sowie exakt
denselben sichtbaren UTF-8-Text vor EOS. Die physische Tail-Folge des Kandidaten
darf nach EOS wegen der späteren Stopprüfung länger sein, muss aber innerhalb
des jeweiligen Arms über alle sechs Prozesse deterministisch bleiben.

**H0 / Negativannahme.** Es gibt keinen klaren Gewinn nach H1. Ein plausibler
Verlust durch EOS-Tail, Blockgrenze oder zusätzliche Readback-/Blockarbeit gilt
als möglicher Ausgang; er wird nicht durch die alte Cycle-7-Zahl von `12,98 %`
weginterpretiert. Diese Zahl ist ausschließlich Orientierung und keine
Erwartungsschwelle für Cycle 17.

## Fest gebundene Ausführung

- Modell: lokaler `mlx-community/gemma-3-4b-it-4bit`-Snapshot, Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`.
- Snapshot- und Gewichtsidentität werden aus dem bestehenden lokalen Resolver
  gebunden; erwartete Cycle-16-Werte sind Snapshot-SHA-256
  `e6edcd46c52b4cf5580f095185a94858565896df7f31c23522294e8f73b3edae` und
  Gewicht-SHA-256
  `94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3`.
- Apple M1 Max, 32 GB Unified Memory, Netzteilbetrieb; Hardware- und
  GPU-Gerät-Gates bleiben zwingend.
- Fixed-Cache-Kapazität: `512` Positionen; Cacheform und Zustandsbaum bleiben
  unverändert zum Cycle-16-Arm.
- Sampling: greedy, `temperature=0`; keine Quantisierung oder Präzisionsänderung.
- Prompt: bytegleich zum Cycle-14/16-Planerprompt, ohne Newline oder sonstige
  Ergänzung. Erwarteter Rohprompt-SHA-256:
  `c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b`.
  Erwarteter Prompt-Token-SHA-256:
  `80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad`.
  Erwarteter gerenderter Prompt-SHA-256:
  `9e18d10b7b101bda3d28593190e622544d474655872aed826c9cbc44211a2cca`.
- Promptlänge: `322` Token; höchstens `32` physisch erzeugte Tokens.
- Beide Arme erhalten für Warmup und Messung dieselben Promptbytes, Tokenbytes,
  Sampling-Einstellungen und Cachekapazität.

Die zwei Arme sind:

| Arm | Readback | Stop-Prüfung |
| :--- | :--- | :--- |
| `fixed_compiled_readback_1` | nach jedem erzeugten Token | nach jedem Token |
| `fixed_compiled_readback_8` | nach höchstens acht erzeugten Tokens | einmal je Readback-Block |

Beide Arme laufen in einem frischen Python-Prozess seriell. Jeder Prozess lädt das
Modell genau einmal; jeder Arm erhält danach einen eigenen frischen Prefill- und
KV-Cache. Kein Modell-, Token- oder KV-Zustand wird zwischen Armen geteilt. Beide
Arme verwenden acht identische Warmup-Forwards; Warmup wird verworfen, aber in
Ressourcen- und Budgetevidenz aufgenommen.

## Vorab festgelegte Reihenfolge

Es gibt sechs frische Prozesse mit je einem gepaarten Armvergleich. Die Reihenfolge
ist vorab festgelegt und wird nicht permutiert:

1. `fixed_compiled_readback_1 → fixed_compiled_readback_8`
2. `fixed_compiled_readback_8 → fixed_compiled_readback_1`
3. `fixed_compiled_readback_1 → fixed_compiled_readback_8`
4. `fixed_compiled_readback_8 → fixed_compiled_readback_1`
5. `fixed_compiled_readback_1 → fixed_compiled_readback_8`
6. `fixed_compiled_readback_8 → fixed_compiled_readback_1`

Ein fehlgeschlagener Prozess wird nicht wiederholt. Teilergebnisse werden erhalten;
ein neuer Versuch bräuchte eine neue Studie und neue Freigabe.

## Physische, logische und sichtbare Ausgabe

Jeder physisch gesampelte Token wird gezählt und zeitlich vollständig erfasst.
`physical_tokens` sind alle bis zur Readback-Grenze erzeugten Tokens, höchstens
32. Der erste EOS-Token gehört zur logischen Folge. `logical_tokens` sind die
physisch erzeugten Tokens bis einschließlich des ersten EOS; fehlt EOS, sind sie
die vollständige physische Folge. Alle Tokens nach dem ersten EOS werden als Tail
verworfen, aber ihre Erzeugungs-, Readback- und Stopentscheidungszeit bleibt in
der primären Zeit enthalten. `overproduced_tokens` zählt genau diesen Tail.

`visible_tokens` sind die logischen Tokens vor EOS; bei fehlendem EOS entsprechen
sie der logischen Folge. Der sichtbare Text wird ausschließlich aus diesen Tokens
mit der gebundenen Tokenizer-Konfiguration dekodiert. Markdown ist kein Qualitäts-
oder Vertragskriterium und wird nicht bereinigt; die sichtbaren Textbytes müssen
zwischen Armen und Prozessen trotzdem exakt gleich sein. Die Kandidaten-Cache-
Zustände werden nach einer EOS-Entscheidung zwingend verworfen. Diese Studie macht
keine Multi-Turn-Aussage und verwendet keinen Cache über eine Antwort hinaus.

Der Baseline-Arm prüft EOS nach jedem Token. Der `N=8`-Arm darf innerhalb eines
Blocks bis zu sieben Tokens nach einem EOS erzeugen, bevor er die Entscheidung
kennt. Dieses Überproduzieren ist der erwartete Preis des Kandidaten, wird nicht
versteckt und zählt vollständig zur kritischen Zeit. Ein fehlender EOS ist kein
Fehler; in diesem Fall müssen alle bis zu 32 physischen Tokens identisch sein.
Die vorhandene Cycle-16-Evidenz setzt keinen EOS-Treffer voraus: Die EOS-Position
darf daher `null` sein, muss bei einem Treffer aber ein nichtnegativer physischer
Index sein; ein negativer Index ist ein Schemafehler.

## Korrektheits- und Identitätsgates

Die Studie bewertet keine allgemeine Modellqualität, keine Planerfähigkeit, keine
Markdown-Qualität und keinen semantischen Nutzen. Sie bewertet nur die Identität
des greedy-Ausführungsergebnisses:

- Prompt-Rohbytes, Prompt-Tokenbytes und gerenderte Promptbytes müssen über alle
  zwölf Arm-Ausführungen identisch sein.
- `physical_tokens` müssen innerhalb jedes einzelnen Arms über alle sechs Prozesse
  exakt gleich sein, einschließlich eines eventuell erzeugten EOS-Tails.
- Zwischen den Armen müssen `logical_tokens`, `visible_tokens` und der sichtbare
  UTF-8-Text über alle sechs Prozesse byte-/wertgleich sein. Wegen des erlaubten
  EOS-Tails wird keine armübergreifende Gleichheit der physischen Folge nach EOS
  verlangt.
- Bei fehlendem EOS gilt zusätzlich zwischen den Armen Identität der vollständigen
  `physical_tokens`; ein Unterschied vor EOS ist immer ein terminaler
  Korrektheitsfehler.
- Determinismus, Greedy-Sampler, Temperatur und Tokenizer-Dekodierung müssen in
  beiden Armen identisch bleiben.

Ein Token- oder Text-Mismatch ist `correctness_failed`, terminal. Es wird weder
getrimmt außer dem vorab definierten EOS-Tail noch durch ein Qualitätsmaß ersetzt.

## Primäre Messgröße und Statistik

Die primäre Messgröße ist die gesamte `decode_critical_path_ns` je Arm. Der Timer
beginnt unmittelbar vor der ersten post-prefill Decode-/Sampling-Aktion und endet
unmittelbar nach der Stopentscheidung einschließlich letzter Readback-Operation,
EOS-Prüfung, sichtbarer Folgeableitung, Tail-Verwerfen und Cache-Discard. Prefill,
Fixed-Cache-Konvertierung, Compile-Wrapper und Compile-Kaltstart werden separat
gemessen; sie sind nicht Teil dieser primären Readback-Entscheidung.

Der Timer wird vor jeder Budget-`charge()` gestoppt. Budgetpausen und Guard-Ruhezeit
gehören nicht zur Decode-Zeit.

Für jede Arm-Ausführung werden mindestens gespeichert:

- physische, logische und sichtbare Tokenzahl sowie Overproduction;
- EOS gefunden, EOS-Token-ID, physische Position, Blocknummer und Stopzeit;
- `readback_interval`, Readback-Anzahl, Blockgrößen, einzelne Readbackzeiten und
  Summe der Readbackzeiten;
- `cache_discarded`, Abschlussgrund, Prefill, TTFT, Compile-/Warmupwerte und
  kritische Decode-Zeit;
- Token-/Text-SHA-256 für physische, logische und sichtbare Folge;
- Prompt-, Prompt-Token- und gerenderter Prompt-Hash;
- monotone Host-Verfügbarkeitszeit je Readback-Block und je physischem Token,
  Readback-Grenze, Blockgröße und `block_latency_ns`; alle Tokens desselben
  Blocks erhalten absichtlich denselben Host-Zeitpunkt;
- Modell-ID, Revision, Snapshot-/Gewichtshashes, Git-Revision und Dirty-State,
  Code-/Spec-/Prompt-/Umgebungsfingerprints;
- PID, Load-Zähler, AC-/Hardware-/GPU-Daten, RSS-/MLX-Peak, Swap vor/nach/Delta,
  Budget- und Abbruchdaten sowie `stderr` nur als begrenzte bereinigte Evidenz.

Die sechs gepaarten Verhältnisse sind
`fixed_compiled_readback_8 / fixed_compiled_readback_1`. Berichtet werden pro Arm
Median und MAD der kritischen Decode-Zeit, Tokenrate, TTFT, Readbackzeit,
Host-Verfügbarkeitszeit und Blocklatenz sowie Ressourcen. Eine physische
Token-Inter-Arrival-Zeit wird nicht behauptet: alle Tokens innerhalb eines
Readback-Blocks erhalten denselben Boundary-Zeitpunkt. Stattdessen werden die
Abstände zwischen aufeinanderfolgenden Host-Verfügbarkeitsgrenzen mit p50/p95/p99
als report-only-Sekundärdaten ausgewiesen. Die Entscheidung verwendet ausschließlich
die kritische Decode-Gesamtzeit.

Bootstrap: gepaartes Median-Ratio-Bootstrap, Perzentil-95-%-Intervall, Seed
`20260824`, `10000` Resamples, keine Ausreißerentfernung. Die alte Cycle-7-Zahl
`12,98 %` wird nicht als Messwert eingesetzt und nicht in die neue Statistik
eingemischt.

## Vorab festgelegte Entscheidungstabelle

Priorität ist von oben nach unten bindend:

| Bedingung | Entscheidung |
| :--- | :--- |
| AC-/Hardware-/GPU-/Snapshot-/Integritätsgate, Swap, Budget, Timeout oder Ressourcenfehler | `resource_or_budget_failed` |
| Readback-/Fixed-Compile-API ist nicht korrekt ausführbar, ohne Ressourcenfehler | `candidate_not_runnable` |
| Logische Token-, sichtbare Text-, Prompt- oder Determinismusabweichung | `correctness_failed` |
| Ratio-Median `≤ 0,95` und obere gepaarte Bootstrap-95-%-Grenze `< 1,0`; alle Gates bestanden | `runtime_readback8_wins_exact_scope` |
| Untere gepaarte Bootstrap-95-%-Grenze `> 1,0`; alle Korrektheitsgates bestanden | `readback8_regression_baseline_retained` |
| Korrekt, aber kein eindeutiger Gewinn oder keine eindeutige Regression | `no_clear_speedup_baseline_retained` |

Keine Zeile erlaubt Kandidatenausführung, Produktaktivierung, einen Dienst, eine
Gewichtsänderung oder einen allgemeinen Modell-/Qualitätsclaim. Jede Entscheidung
bleibt `formal_claim=false`.

## Budget, Ressourcen und Fail-Closed-Regeln

- AC-Pflicht; Offline-Umgebung mit `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` und `PYTHONNOUSERSITE=1`.
  Der Snapshot wird ausschließlich über den bestehenden lokalen Resolver geladen.
- `BudgetGuard`: Duty-Faktor höchstens `0,15`, zusammenhängende Modellarbeit je
  Arm höchstens `6 s`, gesamte GPU-Modellarbeit höchstens `120 s`, gesamte
  Studie höchstens `1200 s`.
- Nach jedem akzeptierten Arm sind vorab mindestens `13` Blöcke à `4 s`
  `required_break()` festgelegt; zusätzlich gilt die registrierte Duty-Formel.
  Pausen werden nicht in die gestoppte Decode-Zeit eingerechnet. Die gesamte
  beobachtete und die tatsächlich akzeptierte Guard-Zeit werden getrennt belegt.
- Peak-RSS höchstens `5 GiB`, MLX-Peak höchstens `5 GiB`, Swap-Delta `0`.
- Frischer Prozess je Paar, genau ein Modellload, strikt serielle Arme, begrenzter
  Output und ein einzelnes striktes JSON-Ereignis. Worker-Timeout, Outputreader,
  Join und Prozessabbruch teilen eine harte Deadline; nach Ablauf wird fail-closed
  beendet. Kein Retry.
- Privates Marker-Verzeichnis Modus `0700`, private Markerdatei Modus `0600`.
  Vorhandener Marker oder vorhandene Ergebnisdatei verhindert jeden Hardwarelauf.
- Fehlender Snapshot, falsche Revision, falsche Hardware, fehlende AC-Versorgung,
  Speicher-/Swap-Wachstum, Budgetverletzung, Timeout, Prozesskill oder Schemafehler
  bewahren Teilergebnisse und führen zu einem geschlossenen Ressourcen-/Budget-
  Ergebnis mit Vorrang vor späteren Deutungen.

## Zulässige Aussagen nach der Studie

Ein positives Ergebnis darf ausschließlich lauten, dass Readback-Intervall `N=8`
im exakt registrierten Fixed-Compiled-Decode-Fall schneller oder nicht schneller
war. Es sagt nichts über andere Prompts, andere Antwortlängen, Multi-Turn,
parallele Requests, andere Modelle, allgemeine Gemma-Qualität, selbstlernende
Optimierung oder einen Matmul-Aus-Pfad aus. Die mathematische Matmul, das Modell,
die Gewichte und die Quantisierung bleiben unverändert.

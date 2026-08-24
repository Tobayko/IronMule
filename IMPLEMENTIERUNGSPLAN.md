# Implementierungsplan

## Auditierter Planstand — 24.08.2026, Zyklus 16 nach Hardwaremessung

Die frühere Abfolge wurde durch den Evidenzaudit enger gefasst. Historische
Dispatch-, Loop-, Modell- und Codegen-Läufe sind explorative
`legacy_summary`-Beobachtungen: Das formale A/A-Gate war nicht geschlossen, die
MDE vor A/B nicht versiegelt und H1/H2-Rohblöcke nicht persistent. Sie dürfen
keinen Phasenfortschritt zu Phase 1B, Cross-Device oder einem breiteren Suchraum
begründen.
Die danach prospektiv geschlossenen H1-/N10-/Runtime-/Shadow-Gates und die neue
explizite Nutzerfreigabe erlauben inzwischen genau den in Punkt 10 beschriebenen
statischen Ein-Kernel-Versuch; sie erlauben weiterhin keine breite Phase 1B.

Aktuelle Reihenfolge:

1. **Root-Provenienz und H1/H2-Evidenzschicht — umgesetzt und offline
   verifiziert.** Root-Git, geschlossenes SQLite-v1-Schema, native/Legacy-
   Trennung, gemeinsame Budgets, fail-closed Persistenz und read-only Historien-
   UI. Historische Werte werden ohne erfundene Rohdaten herabgestuft importiert.
   Produktionsstand: `10` Legacy-Zusammenfassungen und `4` native Ereignisse
   (drei mit Rohmessungen, ein sanitisiertes Guard-Fehlerereignis),
   idempotenter Import und verifizierter read-only Snapshot.
2. **Neuen prospektiven H1-Vertrag erstellen — implementiert, offline
   verifiziert und formal ausgeführt.** Genau eine Tensoroperation; neue Study-ID
   `h1v2-dispatch-n8-20260821-01`; sechs getrennte A/A-Sessions,
   deterministische MDE-Ableitung, genau ein A/B-Kandidat, getrennte
   Charakterisierungs-/Validierungssplits, symmetrisches Warmup, Byteidentität,
   Ressourcenbudgets und terminale Fehlerregeln. Die 16 neuen Offline-Tests und
   die vollständige Suite sind grün. Die terminale 16-Record-Studie endete mit
   `h1_gain_confirmed`, `R=0,879718`, 95%-Intervall
   `[0,877045; 0,880403]`, Byteidentität und bestandenem 5%-MDE-Gate.
3. **Aktuelle Rechenfreigabe wurde für begrenzte v1-Exploration genutzt.** Ein
   Dispatch-Lauf und ein offline erzwungener Gemma-1B/4B-Roofline-Lauf sind mit
   Rohdaten persistiert; sie ersetzen den fehlenden v2-Vertrag nicht. Vor einer
   neuen formalen A/A→A/B-Studie wird die versiegelte Spezifikation zur
   Bestätigung vorgelegt. Keine Installation und kein Download erfolgten.
4. **H1-v2-Ausführung — abgeschlossen.** Präregistrierung, sechs A/A-Prozesse,
   Kalibrierungs-Replay, konservativer MDE-Floor, Bestätigungssiegel, sechs
   frische A/B-Prozesse und terminaler Split-Entscheid wurden auf dem sauberen
   Commit `1fbe73c` ohne Retry ausgeführt. Nur der terminale Record trägt
   `formal_claim=true`.
5. **Begrenzten Runtime-Prototyp validieren — abgeschlossen.**
   Exakte H1-/Workload-/Hardwarebindung, serieller Fallback, latched Circuit
   Breaker, separate append-only Hash-Ketten-Historie und read-only UI sind
   implementiert. Auf Commit `0b0a893` bestanden CPU-Policy-Overhead
   (`11,045 µs` Median, `11,078 µs` p95) und gepaarte MLX/GPU-Validierung
   (`R=0,879209`, `−12,079 %`, byte-identisch) alle Gates. Die Vorregistrierung
   steht in [`docs/RUNTIME_PROTOTYPE_SPEC.md`](docs/RUNTIME_PROTOTYPE_SPEC.md).
6. **Kleinsten geschlossenen H2-Vorschlagslauf ausführen — abgeschlossen.**
   Das bereits lokale Gemma 3 4B durfte höchstens drei noch
   ungetestete ganzzahlige Batchgrößen aus `2..16` vorschlagen. Parser und Harness
   behalten die Ausführungsautorität; Modellcode, freie Parameter, Download und
   Installation blieben ausgeschlossen. Eine Runde schlug `3,10,16` vor und
   bestätigte explorativ `N=10` mit `R=0,671573`, 95%-Intervall
   `[0,648895; 0,731190]`; das Ergebnis bleibt Schema-v1-Evidenz mit
   `formal_claim=false`.
7. **Prospektive N10-Ein-Kandidaten-Studie — abgeschlossen, Gain bestätigt.**
   N10-v1 wurde auf Commit `c3e582c` versiegelt und
   stoppte beim ersten C0-Versuch vor jeder Timingmessung am korrekt arbeitenden
   H0-Fixture-Guard: Der neu abgeleitete Fixture-Seed hatte keine registrierte
   Produktionsidentität. Die zwei V1-Records bleiben unverändert; es gibt keinen
   Retry. N10-v2 ist eine neue Study-ID/DB mit registrierter H0-Fixture-
   Identität, frischen Operand-/Session-/Bootstrap-Seeds und eigener
   Vorgängerprüfung. Nach `508` Tests und `2.480` Subtests wurde V2 auf Commit
   `959df09` versiegelt. Sechs A/A-Sessions kalibrierten die konservative MDE
   auf `5 %`; sechs byteidentische A/B-Sessions bestätigten
   `R=0,874912`, 95%-KI `[0,871768; 0,875614]`. Der 16-Record-Store endet mit
   `n10_gain_confirmed` und genau einem formalen Claim.
8. **Begrenzten N10-Runtime-/Runtime-lite-Pfad prüfen — abgeschlossen; exakter
   Scope bestanden.** Der getrennte Prototyp mit fester Allowlist, seriellem
   Fallback, Circuit Breaker, vollständiger Provenienz und eigener
   Baseline-/Nachher-Messung wurde auf Commit `5eaad38` versiegelt. `17`
   fokussierte Tests sowie die Vollsuite mit `525` Tests bestanden. Das
   einmalige CPU-Gate bestand mit `12,372 µs` Policy-Median und `12,448 µs`
   p95; der danach zulässige einmalige MLX/GPU-Lauf bestand mit
   `R=0,875753`, `−12,425 %`, Byteidentität und `max_abs_error=0`. Die eigene
   Historie enthält genau zwei gültige, hashverkettete Engineering-Records;
   ihre read-only UI wurde auf Port `8772` geprüft. Die bestehende N8-Runtime
   blieb unverändert. Das Ergebnis ist ein Engineering-GO nur für exakt
   FP16-`2048²`, zehn Matmuls und den versiegelten Batch-Plan. Freie
   Codegenerierung und Custom Metal sind nicht autorisiert; Phase 1B bleibt
   **NO-GO**, Cross-Device **NO-CLAIM**, weitere Modellrunden und ein breiterer
   Live-Suchraum bleiben **NO-GO**.
   Der aktuelle Entscheid steht in
   [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md).
9. **Evidenzgebundenen N8/N10-Shadow-Router prüfen — abgeschlossen; alle Gates
   bestanden.** Der getrennte Router wurde auf dem sauberen Commit `70bc451`
   versiegelt. Er verlangt gleichzeitig die exakte N8- und N10-Evidenz,
   beobachtet reale Tensor-Metadaten, besitzt keine `execute`-Methode und
   erzwingt immer `serial_shadow_only`. `19` fokussierte Tests sowie die
   Vollsuite mit `544` Tests bestanden; ein vollständiger Security-Diff-Scan
   endete mit kompletter Coverage und null reportablen Findings. Das einmalige
   CPU-Gate bestand mit `13,719 µs` Router-Median, `13,815 µs` p95 und
   `1,585 µs` zusätzlichem Median. Die danach einmalig ausgeführte
   MLX-Shadow-Validierung bestand alle fünf Gates; N8/N10 wurden nur empfohlen,
   drei Negativfälle routeten seriell und es wurde keine Matmul ausgeführt.
   Die eigene DB enthält genau zwei hashverkettete Engineering-Records; Port
   `8773` wurde read-only geprüft. Das ist ein **GO ausschließlich für die
   nächste getrennte Vorregistrierung**, nicht für produktive Aktivierung.
10. **Einen statischen Custom-Metal-Kandidaten vorregistrieren — als Nächstes.**
   Nach der ausdrücklichen Nutzerfreigabe und dem bestandenen Shadow-Router darf
   genau ein statischer Fusionskandidat für Residual-Add plus RMSNorm gegen die
   starke MLX-Referenz spezifiziert werden. Vor jeder Kompilierung oder
   Timingmessung werden Shape, Precision, Correctness-Grenzen, A/A-Gate,
   A/B-Reihenfolge, Warmup, Wiederholungen, Timeout, Ressourcenlimits und
   terminale Abbruchregeln eingefroren. Ausführung ist nur in einem
   kontrollierten Worker zulässig. Eine Installation oder ein neues Modell ist
   dafür nicht freigegeben oder erforderlich; produktive Integration und
   adaptive Kernelsuche bleiben **NO-GO**.

## Zyklus 15 — enge Zwei-Modell-Studie, finaler Stand

Die neu erteilte Nutzerfreigabe ist ausschließlich auf die vorregistrierte Studie
`dual-model-evidence-planner-20260824-01` begrenzt. Sie erlaubt keinen allgemeinen
Planner, keine automatische Aktivierung und keine Ausweitung früherer Studien. Es
existiert keine Matmul-On/Off-Integration; ein Matmul-On/Off-Vergleich ist nicht
Bestandteil dieses Zyklus und wird nicht als Ergebnis behauptet.

Der geschlossene Studienvertrag bindet die bereits lokal vorhandenen Snapshots
`mlx-community/gemma-3-1b-it-4bit` (Revision
`2d44e83dc9e80843d22fb941d3d699a0b1351aa6`) und
`mlx-community/gemma-3-4b-it-4bit` (Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`) an dasselbe Gerät, dieselbe
Offline-Umgebung und denselben gerenderten Prompt. Es gibt sechs feste Paare mit
zwölf frischen seriellen Prozessen: `1b → 4b` für die Paare `1–3`, danach
`4b → 1b` für die Paare `4–6`; jedes Modell läuft sechsmal. Der einzige erlaubte
Planerwert ist `persistent_service_qualification`; die Studie misst keine
allgemeine Qualität und bleibt `formal_claim=false`.

Der Vor-Hardware-Snapshot ist über diese Dateihashes reproduzierbar gebunden:
Präregistrierung `246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`,
Worker `b1db90d306d5de5c6ff466d046c5c617c5dd42cdaee3f6f7b4bcd5bf2a024bc0`,
Harness `59691f50a1f33d4930b36ccce24ec701af74ebd0f9f095912a75e15a28978470` und
read-only UI `5db9bf832c17470c0899ee0fd4062b42d524904e1ee3224894e87a7bed049607`.
Der finale fokussierte Offline-Stand ist `47` Tests plus `42` Subtests, Exit `0`;
auch `py_compile` endete mit Exit `0`. Diese unabhängige Test-Luna-Verifikation
wurde in diesem Dokumentationsschritt nicht erneut ausgeführt.

Der vollständige Vor-Hardware-Preflight ist abgeschlossen: Worker `17/17` und
Harness `25/25`, jeweils Exit `0`; Defaultaufruf Exit `78` ohne Startmarke oder
Resultat; `compileall`, `git diff --check`, AST-Parsing und
`xcodebuild -checkFirstLaunchStatus` jeweils Exit `0`. Die fokussierte Suite lief
mit Exit `0` in `3,36 s` bei `60.801.024 B` Peak-RSS, die vollständige
`pytest`-Suite mit Exit `0` in `45,43 s` bei `200.523.776 B` Peak-RSS.
ProjectAtlas-Runtime und -Konfiguration bestanden mit Exit `0` auf Version
`0.4.5-rc1`.

Die read-only Umgebungsprüfung bestätigte MLX `0.32.0`, mlx-lm `0.31.3` und
`Device(gpu, 0)`. Der Resolver bestätigte ohne Modellload den 1B-Snapshot auf
Revision `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` mit `732.577.304 B`
Gewichten sowie den 4B-Snapshot auf Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2` mit `3.400.569.562 B`
Gewichten. Die Präregistrierung liegt nach dem reinen Formatfix bei
`246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`.
Ignorierte `__pycache__`-Verzeichnisse sind vorhanden, werden aber nicht Teil des
Commits. Keine dieser Prüfungen führte Hardwarearbeit, GPU-Rechnung oder einen
Modellload aus.

Der staged Diff-Check hatte zuvor wegen genau drei Trailing-Spaces in der
Präregistrierung Exit `2` geliefert. Diese drei Formatzeichen wurden vor Hardware
entfernt; Semantik, Studienvertrag, Schedule, Gates und Claim änderten sich nicht.

Zum Zeitpunkt des Preflights gab es noch keine Studienergebnisse, keine Startmarke
und keine `results.json`. Dieser historische Vor-Hardware-Stand wurde danach durch
den folgenden einzigen zulässigen Lauf abgeschlossen. Der Zyklus-14-
Dokumentationsaudit ist auf `ee12bb5` verankert.
Die vorab geschlossenen Korrekturen (Duplicate-Paare, unvollständige Erfolgs-
aggregation, per-Run-Content-Hashing-Bias, Snapshot-/Prompt-Bindung, fail-safe
Partialpfade, stdout-Limit, UI-Whitelist und Ressourcenprüfungsreihenfolge) bleiben
verbindlich. Der finale Stand erhält ein bereits validiertes Event auch bei einem
nachfolgenden Ressourcenabbruch, bindet die UI zusätzlich an die feste Run-ID und
eine geschlossene Decision-Allowlist und behandelt einen minimalen Fehlerreport
ohne `metrics` kontrolliert als Fehler statt als Erfolg.

### Zyklus 15 — reales Ergebnis und Entscheidung

Der Studienlauf `dual-model-evidence-planner-validation-20260824-01` wurde genau
einmal am Netzteil ausgeführt: sechs balancierte Paare, zwölf frische serielle
Prozesse, drei Paare `1b → 4b` und drei Paare `4b → 1b`. Beide Modelle wurden
jeweils sechsmal geladen, nie gleichzeitig und nie wiederholt. Die unveränderte
Entscheidungstabelle gab `no_planner_qualified` zurück (`formal_claim=false`),
weil beide Modelle den strikten Maschinenvertrag in `0/6` Läufen erfüllten.

| Messwert | 1B | 4B |
| --- | ---: | ---: |
| Vertrag / Parser / `candidate_id` | `0/6 / 0/6 / 0/6` | `0/6 / 0/6 / 0/6` |
| Determinismus | `6/6` | `6/6` |
| TTFT Median / MAD | `0,295451312 / 0,0005528535 s` | `0,796846125 / 0,0088023125 s` |
| Modellarbeit Median / MAD | `0,4608839165 / 0,0005743330 s` | `1,0487644165 / 0,0092854165 s` |
| Prozess-Walltime Median / MAD | `4,2468557705 / 0,0059329165 s` | `4,883630417 / 0,0182606455 s` |
| Peak-RSS / MLX-Peak | `1.937.965.056 / 1.012.548.526 B` | `3.765.420.032 / 3.021.085.374 B` |
| Swap-Delta | `0 B` | `0 B` |

Die 1B-Ausgabe enthielt Markdown, den falschen Schlüssel
`persistent_service_id` und `<end_of_turn>`-Trailer. Die 4B-Ausgabe hatte die
richtige ID, aber einen Markdown-Codeblock. Die direkte dekodierte Textgleichheit
zwischen den Modellen lag bei `0/6`; innerhalb jedes Modells waren Text und Token
`6/6` deterministisch gleich. Das sind Vertragsbefunde, keine qualitative
Bewertung der Modelle.

Alle Ressourcen-, Budget-, Snapshot- und Pairing-Gates bestanden. Gemessen wurden
`9,205052 s` Gesamt-Modellarbeit, maximal `1,151402 s` zusammenhängend und
`178,475444 s` Walltime bei Duty-Faktor `0,15`; es gab keine Abbrüche und kein
Swap-Wachstum. Die paarweisen Verhältnisse mit Bootstrap-95-%-KI sind berechnet:
TTFT `0,373014193 [0,365603946; 0,377539933]`, Modellarbeit
`0,439069434 [0,434598134; 0,444460794]`, Walltime
`0,872042394 [0,864987297; 0,939562889]`, Tokenrate
`3,168801108 [3,130352029; 3,201472197]`. Die daraus berechneten ungefähr
`12,8 %` kürzere 1B-Walltime und `48,5 %` geringerer 1B-Peak-RSS können wegen des
Funktions-Gatefehlers keine Modellpräferenz oder Aktivierung begründen.

Die Rohdaten liegen unter
`experiments/dual_model_planner/results.json` (SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`), die
private Startmarke hat SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`.
Die UI-Prüfung ergab GET/HEAD `200`, Schreibmethoden `405`, fremde Hosts `421`
und unveränderte Evidenzhashes. Zyklus 15 hat eine JSON-Datei, keine eigene
SQLite-Evidence-DB.

Es existiert weiterhin kein vollständiger Gemma-Matmul-A/B-Pfad und kein
„mit/ohne Matmul“-Schalter. Dieser Vergleich wurde nicht gemessen und bleibt ein
separater künftiger vorregistrierungspflichtiger Kandidat. Allgemeine
Modellqualität, allgemeine Planner-Fähigkeit, selbstlernende Runtime,
Produktaktivierung, Multi-Turn-Fortsetzung und parallele Requests sind durch
diesen Zyklus nicht belegt. Die Freigabe ist verbraucht; weitere Hardwarearbeit
braucht neue Freigabe und einen neuen Zyklus.

### Zyklus-15-Postflight

Nach dem einzigen Hardwarelauf bestand die fokussierte Suite mit `47` Tests und
`42` Subtests bei Exit `0`; die vollständige Suite sammelte `744` Tests und
endete mit Exit `0`. `compileall`, die strikte JSON-Prüfung von
`results.json`, `verification.json` und `EXPERIMENT_MATRIX.json`, `json.tool`,
AST, `git diff --check` und `xcodebuild -checkFirstLaunchStatus` endeten jeweils
mit Exit `0`. ProjectAtlas meldete zunächst `refresh_required`; genau ein
inkrementeller Refresh war erfolgreich. Runtime `0.4.5-rc1` und die
projektlokale MCP-Konfiguration waren gültig. MLX `0.32.0`, mlx-lm `0.31.3` und
`Device(gpu, 0)` wurden read-only geprüft; es gab keine Modell- oder GPU-Arbeit.

Nach dem Postflight blieben Ergebnis-SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`,
Verifikations-SHA-256
`24696c679de567519e8f2b3b034f0833de8122569072b71feeae794c05bbf4e6` und
Marker-SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327` sowie alle
DB-Hashes unverändert. Die Verifikation hatte leere Abweichungen und bestätigte
`no_planner_qualified` bei `formal_claim=false`. ProjectAtlas hatte keine
getrackten Änderungen; bestehende untracked Fixture-`.gradle`-Verzeichnisse
blieben unangetastet.

## Historischer H0-Pivot und Freigabestatus — 20.08.2026

`JA — Ich gebe den Forschungspivot H0 → H1 → H2 und die Implementierung von Phase 1A/H0 mit SQLite v1, read-only Loopback-Dashboard und festem Worker Option A frei. Keine Downloads, Installationen, Custom-Metal-Kernels oder Modellgewichte.`

Der Offline-Unterbau ist umgesetzt: SQLite v1 unter `.friday-data/h0.sqlite3`, fester
Worker Option A und ein read-only Dashboard auf `127.0.0.1`. H0 bleibt eine einzelne
FP16-`2048²`-Matmul und ist kein Modelltest. Der vollständige Offline-Pre-Live-Adapter
ist **GO**; im finalen Offline-Adapter-Scope sind keine offenen P0/P1/P2 verblieben.
Die lokale H0-DB enthält inzwischen `22` Runs. Der lexikalische Launcherpfad und W1v3
wurden umgesetzt; Run22 ist ein abgeschlossener einzelner `eager_baseline`-Reference-
Lauf mit produktiven Matmul-Rohzeiten, aber weiterhin kein Vergleichs-, A/A- oder
Optimierungsnachweis. `run_mlx` und `mlx-run --execute` sind implementiert; ohne
`--execute` bleibt `EXIT_MLX_LOCKED=78`/`not_released` vor Runner-/Worker-/Benchmark-/
MLX-Import.

Der separate H0.1-Unterbau ist ebenfalls implementiert: vorregistrierter
Paced-Trajectory-Core, append-only SQLite v1 und read-only Historien-Dashboard. Ein
atomarer H0→H0.1-Adapter-Execute importierte exakt drei verifizierte historische
Warmup-Beobachtungen; eine runtime-unavailable Generation blieb ausgeschlossen. Das
ist **GO für Evidenzmigration**, aber kein Stationaritätsbefund: Die H0.1-DB enthält
noch `0` Paced-Sessions und `0` Paced-Studies.

Diese Reihenfolge ist am 21.08.2026 abgeschlossen: Ausführungspfad geschlossen,
Zielgeräte-Preflight bestanden, `C0,V0,C1,V1,C2,V2` ohne nachträgliche
Schwellenänderung ausgeführt, vollständiger Study-Replay bytegleich. Ergebnis:
`h01_complete_unresolved`, `failed_gate_count=23` — replizierte Stationarität ist
**nicht** unterstützt. Das ist ein gültiges negatives Ergebnis; der vorregistrierte
Envelope wird deutlich verfehlt, dominierend über die Tail-Ratio `2,53`–`3,13` bei
Grenze `1,20`.

Die damals als nächste offene Entscheidung bezeichnete H1-Stufe ist ohne neue Nutzerfreigabe nicht zu
treffen, weil ein nicht stationärer Messuntergrund jede Vorher/Nachher-Aussage
begrenzt: Eine Optimierung müsste einen Effekt zeigen, der größer ist als die
beobachtete Streuung derselben unveränderten Operation. `aa_gpu`, H2-Modelle und
Custom Metal bleiben außerhalb dieses Schritts. Für den späteren Modelltest ist
**Gemma 3** bestimmt, gestuft `1B` vor `4B`; es wurde nichts heruntergeladen oder
installiert.

Aktuell: Live-Pfad `45/45` (Wall `4.022908 s`, U/S `3.149974/0.200018 s`, Peak-RSS
`42,139,648 B`, keine belegte Self-/Child-Trennung); `get_cache_memory`-Fix `16/16`
(Wall `0.086906 s`, U/S `0.140900/0.054489 s`, Peak-RSS `49,938,432 B`, ebenfalls
ohne belegte Trennung); Nicht-Live-Suite `133/133` (Wall `23.720160 s`, U/S
`22.722187/0.559409 s`, Self-/Child-RSS `71,368,704/23,642,112 B`); unabhängiger
Replay `133/133` (Wall `23.588426 s`, U/S `22.769535/0.504137 s`, Self-/Child-RSS
`60,342,272/23,707,648 B`); socketfrei `4/4` plus `3` Setup-Subtests (Wall
`0.001793 s`, U/S `0.001437/0.000137 s`, Self-/Child-RSS `31,457,280/0 B`).

Historische, anders enumerierte Offline-Verifikation: Hauptsuite ohne Dashboard `177` bestanden, `3` Windows-
Skips und `12` Subtests (Wall `26.034290 s`, Total U/S `23.373336/1.227233 s`, Self-/
Child-Peak-RSS `15,499,264/74,186,752 B`); socketfreies Dashboard `4/4` plus `3`
Setup-Subbranches (Wall `0.002041 s`, RSS `31,260,672 B`). Die frühere autorisierte
HTTP-Prüfung `13/13` bleibt die letzte vollständige HTTP-Evidenz; der spätere `16`-er
Scope wurde wegen Sandbox-/Usage-Limit nicht final wiederholt.

## Phase 0 — Reproduzierbarer lokaler Unterbau

**Status:** Offline-Unterbau und Pre-Live-Adapter implementiert/verifiziert; historische
fail-closed Canaries und der erfolgreiche Run22-Reference-Lauf sind archiviert. Der
separate H0.1-Evidenzspeicher ist produktiv angelegt; Paced-Ausführung steht noch aus.

- Python 3.12-Umgebung und feste Abhängigkeiten dokumentieren.
- Xcode/Command-Line-Tools, MLX Metal und PyTorch MPS prüfen.
- ProjectAtlas initialisieren, MCP-Verbindungen prüfen, `.projectatlas` nicht committen.
- Hardware-/Software-Fingerprint als JSON erzeugen.

Der Zielgeräte-Smoke bestätigte MLX `0.32.0` mit einer 1-Element-Operation, war aber
keine Matmul. Der Canary lief außen `0.166578416 s`, erzeugte keine Rohsamples,
Correctness-Zeilen oder A/A-Session und erhöhte die DB auf `16` Runs. Zwei stabile
read-only Snapshots bestätigten `snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
`source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
`run_count=16` und `query_only=1`.

**Abbruchkriterium:** Wenn MLX und Metal auf dem Gerät nicht reproduzierbar erreichbar sind, den
Apple-spezifischen PoC stoppen oder auf CPU/portable Messungen zurückbauen.

## Phase 1 — Kleinster echter Proof of Concept (nach H0-Go/No-Go)

- eine feste Matrixmultiplikation oder eine andere einzelne Tensoroperation; H0 selbst
  bleibt zunächst der offline validierte FP16-`2048²`-Matmul-Vertrag;
- MLX-Standardbaseline;
- mindestens ein klar begrenzter Custom-MLX-Metal-Kandidat erst nach separater Phase-1B-
  Sicherheits-/Architekturfreigabe;
- isolierter Worker-Prozess mit Timeout;
- Correctness gegen Referenz/High-Precision-Ausgabe;
- Warmup, Wiederholungen, Median, Streuung und Mindestverbesserung;
- SQLite-v1-Speicher für Hardware, Workload, Konfiguration und Ergebnis;
- Rollback bei Compilefehler, Crash, Timeout oder Verschlechterung.

**Erfolg:** Der Loop findet reproduzierbar eine bessere Konfiguration oder erkennt reproduzierbar,
dass die Baseline für den getesteten Fall nicht geschlagen wird.

## Phase 2 — Eine echte ML-Operation

RMSNorm, Softmax, RoPE oder Quantization/Dequantization auswählen. Auswahl nur anhand eines
Profiling-Befunds, nicht nach Attraktivität. Den gleichen Sicherheits- und Statistikrahmen aus Phase 1
wiederverwenden.

## Phase 3 — Kleines MLX-LM-Modell (erst nach H1)

Ein quantisiertes 7B–14B-Modell nur als Integrationslast verwenden. Erst Profiling, dann eine einzelne
Hotspot-Operation optimieren. Tokens/s, TTFT, Prompt-Verarbeitung, Generation, Peak Memory und
thermische Langzeiteffekte getrennt messen. Eine Kernelverbesserung gilt erst, wenn sie die
End-to-End-Metrik nicht verschlechtert.

## Phase 4 — Gelerntes Cost Model

Zuerst Grid-/Random-/Bayesian Search als Referenz. Danach Gradient Boosting oder Regression für die
begrenzte Parametermenge evaluieren. RL, evolutionäre Suche und ein LLM als direkter innerer
Optimierer sind Forschungsoptionen, keine Voraussetzungen. Das Cost Model schlägt Kandidaten vor; ein
echter Benchmark entscheidet.

## Phase 5 — Portable Backend-Schicht

Ein gemeinsames Control-Plane-Datenmodell definieren und Backend-Adapter für MLX/Metal, später Triton/
CUDA und eventuell ROCm ergänzen. Keine eigene IR bauen, solange MLIR/Graph-/Compiler-IR die benötigte
Abstraktion liefern. Android/iOS zuerst als vorab validierte/AOT-Kandidaten betrachten; kein Anspruch
auf frei programmierbare mobile GPU-/NPU-Kernel.

## Phase 6 — Forschungs-/Open-Source-Projekt

- veröffentlichte Benchmark-Suite mit Hardware-/Treiber-/Framework-Fingerprint;
- reproduzierbare Worker-Sandbox und Regressionstests;
- versionierte Optimization Memory mit Ablauf-/Kompatibilitätsregeln;
- klare Trennung zwischen gemessenen Fakten, Heuristiken und Forschungsannahmen;
- erst danach weitere Hardware und Cloud-Fallbacks.

## Abbruch oder Umbau

Das Projekt wird auf Benchmark-/Cost-Model-Forschung zurückgebaut, wenn ein LLM keinen Mehrwert
gegenüber deterministischer Suche liefert, wenn Custom-Kernel keinen stabilen Vorteil zeigen oder
wenn Plattform-APIs das notwendige Feedback nicht zuverlässig liefern. Das ist ein valides Ergebnis,
kein Scheitern des Projekts.

## Finaler Contract- und Run21-Nachtrag — 20.08.2026

Der finale Offline-Contract ist mit Core `175/0`, Dashboard `4/4` und `0` offline MLX-
Imports belegt. Provenienz: `575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`;
Code `aae3245e…` (nur als übergebenes Präfix), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 lief genau einmal: Exit `10`, Wall `1.14 s`, User/System `0.98/0.16 s`, Peak-RSS
`369,573,888 B`. Fail-closed Ergebnis: `invalid/invalid/baseline_fallback`, Diagnose
`warmup_unstable` nach `16` Warmups. `all`: Median `2,391,354.5 ns`, MAD `287,125 ns`,
IQR `582,260.25 ns`; `last5`: Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`,
Min/Max `2,067,916/2,677,583 ns`, Stabilität `false`. Es gab `0` Rohsamples, `0`
Correctness-Zeilen, `3` Scalars und `1` Artifact; kein `aa_gpu`, keine Performance- oder
Correctness-Aussage.

Die eingefrorene Regel `8 → maximal 16` Warmups und letzte fünf Werte innerhalb `±5 %`
entspricht dem Code; es liegt kein Implementierungsdefekt vor. Ursache bleibt OS-/Thermik-/
MLX-unspezifisch. Es gab keine nachträgliche Schwellenänderung und keinen Retry. Die
DB-/Bundle-/Result-/Payload-/Evidence-Hashes sind als `c9a521…`, `420b7c…`, `027908…`,
`ac4a82…`, `cd409d…`, `837841…` nur in der übergebenen Kurzform verfügbar.

Der `python`-Alias und der Dashboard-`self.path`-Fehler werden als Harnessfehler getrennt
vom Projektbefund geführt. Die Konvergenzregel verlangt Reproduktion und unabhängigen
Readback, bevor ein Harnessbefund die Bewertung beeinflusst; insbesondere wird kein
Threshold nachträglich angepasst. Der statische Dashboard-Check bestätigt automatische
read-only-Historienlektüre und Sichtbarkeit des `invalid`-Status; Server/Sockets wurden in
diesem Dokumentationsnachweis nicht gestartet.

## Zyklus 16 — Vor-Hardware-Schritt: runtime-only Fixed-Cache/Compile-A/B

Für den am 24.08.2026 ausdrücklich freigegebenen Einzelversuch ist die Studie
`matmul-compile-ab-20260824-01` mit dem Kandidaten
`fixed_cache_compiled_decode_v1` vorregistriert. Die Studie ändert weder
Modellgewichte noch Modellarchitektur noch Quantisierung. Die mathematische
Matmul bleibt in allen Armen aktiv; gemessen werden ausschließlich die drei
Laufzeitpfade `standard_eager`, `fixed_eager` und `fixed_compiled`.

Vor dem Hardwarelauf muss die Präregistrierung noch lokal versiegelt werden.
Der aktuelle Stand ist daher: im Arbeitsbaum vorregistriert, noch nicht
versiegelt und noch nicht gemessen. Greedy Token- und Textidentität müssen exakt
gleich bleiben; die alten Device-Model-Compile-Werte sind wegen falscher Token
ab Position 2 ungültig. `formal_claim=false`. Ein negatives Ergebnis beendet
den Kandidaten gültig und führt nicht zu einer automatischen Aktivierung.

## Zyklus 16 — Seal vor Hardware

Der lokale Commit mit diesem final geprüften Stand ist der Seal-Commit; der
Status lautet `sealed_pending_hardware`. Die Präregistrierung
`matmul-compile-ab-20260824-01` ist mit SHA-256
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf` eingefroren.
Es bleiben keine Ergebnisdatei und keine Startmarke vorhanden; `formal_claim=false`.

Die drei behobenen P1-Ursachen waren lazy MLX-Materialisierung außerhalb der
Fehlerklassifikation, ein Worker-Timeout ohne Bezug zur verbleibenden Gesamt-Walltime
und die Vermischung von beobachteter Armzeit mit akzeptierter Budgetbuchung. Die
Vor-Hardware-Lösung klassifiziert Materialisierungsfehler korrekt, verwendet eine
harte Deadline und speichert observed/charged/accepted getrennt. Arm- und
Fehlerfelder bleiben dabei streng validiert.

Verifiziert: fokussierte Tests 34 passed/Exit 0, vollständige Suite/Exit 0,
compileall/Exit 0, Worker-Selfcheck 21/Exit 0, Harness-Selfcheck 18/Exit 0,
UI-Selfcheck/Exit 0, Default ohne Ausführung/Exit 78 ohne Marker oder Ergebnisse,
`git diff --check`/Exit 0, Xcode-Check/Exit 0. Atlas 0.4.5-rc1 mit MCP,
M1 Max/32 GiB/AC, `Device(gpu,0)`, MLX 0.32.0 und mlx-lm 0.31.3 sind verifiziert.

## Zyklus 16 — Ergebnis, Scope und Abschluss

Seal-Commit: `83ee3ea03f9fb303b8226ab8ad3189f07daec727`; Studie:
`matmul-compile-ab-20260824-01`; Entscheidung:
`runtime_compile_wins_exact_scope`; `formal_claim=false`. Evidence-Commit
`cc6d2ea012a0cd6a858acc9a66d4754e95c421b7`, Result
`fbcc2fc65ac5d255ed11039a74c34e9a02d942cec17b25a6ed863058e0073b57`, Verification
`09b1b53841a59bad3c4b1b9a0ef62fb659668b472358c10fa9188cad158f0038`, Marker
`8adf6f9c2453524bd1e05f4973ee85f84a323e9461a3f9b996ec2d0f7fed3c2f`, Präregistrierung
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Die Freigabe ist genau einmal verbraucht.

Gemessen: sechs Prozesse, 18 Arm-Ausführungen (3 × 6), exakt gleiche Tokens und
Texte. Decode-Median/
TTFT: Standard `0,399939187 s`/`0,638376521 s`, Fixed-Eager
`0,3999597295 s`/`0,638425813 s`, Fixed-Compiled `0,371848789 s`/
`0,6385446665 s`. Die gemessenen Decode-Ratios sind `0,9295921887`
(`[0,9128789083; 0,9348209684]`) gegen Standard und `0,9296309524`
(`[0,9256302629; 0,9327708433]`) gegen Fixed-Eager.

Berechnet: warm `0,9829777045`, kalt `1,0154895491`, Break-even median rund
`36,47` Decode-Schritte bei 31 gemessenen Schritten. Matmul blieb aktiv;
Modell, Gewichte und Quantisierung blieben unverändert. Die Aussage gilt nur
für diesen lokalen Runtime-Fall und aktiviert keinen Produktivpfad.

## Zyklus 16 — Post-Hardware-Verifikation

Nach dem einmaligen Lauf wurde die Evidenz unabhängig geprüft; ein weiterer
Hardware- oder Modelllauf fand nicht statt. Die vollständige Suite bestand mit
`787 Tests in 71 Dateien`, Exit 0. Der fokussierte Test
`test_matmul_compile_ab` bestand mit `43 passed, 60 subtests`, Exit 0. Compileall,
Worker-Selfcheck 21, Harness-Selfcheck 18, Dashboard-Selfcheck 0,
`xcodebuild -checkFirstLaunchStatus`, `jq` und `git diff --check` endeten mit
Exit 0. Der Standardaufruf lieferte erwartungsgemäß Exit 78 ohne Mutation;
Der Harness-`--show`-Aufruf lief einmal mit Exit 0, stderr blieb leer und er
lieferte genau eine gültige JSON-Zeile.

Die read-only UI antwortete auf GET/HEAD mit `200`, auf Schreibmethoden mit
`405` und auf einen fremden Host mit `421`; `no-store` war gesetzt und kein
unbereinigter Modelltext wurde gerendert. Cycle-16- und Cycle-15-Evidence sowie
12 SQLite-Datenbanken waren vor und nach dem Test identisch; die private
Startmarke blieb `0600`, und alle geprüften Hashes blieben unverändert.

ProjectAtlas wurde genau einmal inkrementell per `watch_once` aktualisiert:
ein Zyklus, 967 indexierte Textkandidaten, 11 geparste und 732 unveränderte
Symbole. Runtime `0.4.5-rc1` und die projektlokale MCP-Konfiguration waren gültig.
Getrackte ProjectAtlas-Dateien wurden nicht verändert; ein bereits vorhandenes
verschachteltes `.gradle`-Untracked blieb unberührt.

Ursache des Lifecycle-Selfcheck-Fehlers war die falsche Annahme, Evidence müsse
fehlen. Die Korrektur prüft fehlende und vorhandene Evidence read-only, verlangt
reguläre Dateien ohne Symlink, Marker-Modus `0600` sowie unveränderte Hashes und
Dateimodi. Der Arbeitsbaum-Harness weicht deshalb vom versiegelten Code ab; die
Evidence erhält die Code-Fingerprints des Seal-Stands. `formal_claim=false`.

## Zyklus 17 — Draft vor Preflight

`fixed-compiled-batched-readback-20260824-01` /
`fixed_compiled_batched_readback_n8_v1`, Status `draft_pending_preflight`.
„Dann machen wir das mal“ reserviert genau einen neuen Lauf; die Freigabe ist
noch nicht verbraucht. Readback `1` versus `8` ist die einzige Variable auf
identischem Fixed-Compiled-4B-Pfad. Sechs gepaarte frische Prozesse und zwölf
Arm-Ausführungen sind geplant; Modell, Gewichte, Quantisierung und Matmul bleiben
invariant. EOS-Tail wird vollständig getaktet und getrimmt, exakte logische
Token/Textidentität ist terminal. Noch kein Marker, Resultat oder Modelllauf;
`formal_claim=false`, keine Aktivierung, kein Dienst, kein Multi-Turn- oder
Qualitätsclaim.

## Zyklus 17 — sealed_pre_hardware

Offline-Preflight abgeschlossen: `measured=false`, `formal_claim=false`,
`authorization=reserved_not_consumed`; kein Modell-/MLX-/Hardwarelauf, Marker
oder Resultat. Readback 1 versus 8 bleibt die einzige Variable im identischen
Fixed-Compiled-4B-Pfad; sechs frische Paare und zwölf Arme sind geplant.

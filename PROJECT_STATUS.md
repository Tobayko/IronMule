# Projektstatus

**Stand:** 25. August 2026, Zyklus 17 nach realer Hardwaremessung
**Zielgerät:** Apple M1 Max, 32 GB Unified Memory, 10-Core CPU, 32-Core GPU

## Auditierter aktueller Stand

| Bereich | Verifizierbarer Stand | Zulässige Aussage |
| --- | --- | --- |
| Root-Provenienz | Root-Git-Repository; Zyklus 15 auf Seal-Commit `23cff19` (Ergebnis-/Dokumentationsabschluss folgt getrennt), Zyklus 14 auf Vor-Hardware-Commit `8067dc6` und separatem Ergebnis-/Dokumentations-Commit `8923467`, Zyklus 13 auf `f954617`, Zyklus 11 auf `1dff9a5`, formale Head-Skip-Studie auf `9466bb9`, Phase 1B auf `ea8f959`, N8/N10-Shadow-Router auf `70bc451`, N10-Runtime auf `5eaad38`, N10-v2 auf `959df09`, N10-v1 auf `c3e582c`, formaler H1-v2-Code auf `1fbe73c`; `ProjectAtlas/` als gepinntes, unverändertes Gitlink | formale und native Läufe sind an konkrete Root-Revisionen gebunden |
| H0 | `.friday-data/h0.sqlite3` mit `28` Runs, darunter `9` `aa_gpu`-Runs | H0-Rohhistorie vorhanden; **kein** formal geschlossenes A/A-Gate |
| H0.1 | `3` Legacy-Beobachtungen, `6` Paced-Sessions, `1` Study mit `h01_complete_unresolved` | replizierte Stationarität nicht unterstützt; gültiger Negativbefund |
| H1/H2 historisch | zehn rekonstruierbare Zusammenfassungen, keine Rohblöcke und keine vollständige historische Provenienz | ausschließlich `legacy_summary`; formale H1/H2-Claims `false` |
| H1/H2 künftig | SQLite-v1-Evidenz, saubere Git-/Code-/Spec-/Environment-Bindung, gemeinsame Budgets und read-only Historien-UI implementiert; vier native Ereignisse vorhanden | prospektive Exploration möglich; formale Claims bleiben in v1 ausdrücklich `false` |
| H1-v2 formal | terminale 16-Record-Historie: versiegelte Präregistrierung, sechs bestandene A/A-Sessions, MDE `5 %`, sechs frische A/B-Sessions und Split-Entscheid `h1_gain_confirmed` | für genau ein Gerät, FP16-`2048²`, acht Matmuls und den Batch-Dispatch-Plan ist der Gain jenseits der MDE formal bestätigt; kein Modell-/Cross-Device-Claim |
| Begrenzte Runtime | exakte H1-Bindung, tensorbasierte Scope-Prüfung, serieller Fallback, Circuit Breaker, Hash-Ketten-Historie und read-only UI; CPU- und MLX/GPU-Gates auf sauberem Commit bestanden | Batch ist nur für den exakt registrierten Workload freigegeben; Policy-/Runtime-Befund ist Engineering-Validierung, kein neuer formaler oder Modell-Claim |
| N10-Runtime / Runtime-lite | getrenntes Paket `friday_runtime_n10/` mit exaktem 16-Record-/DB-/Snapshot-Claim, N=10-Allowlist, Cold-Load-Gate, Circuit Breaker und eigener DB/UI; CPU- und MLX/GPU-Gates auf sauberem Commit bestanden | **Engineering-GO nur im exakten N10-Scope**; jede Evidenz-, Code-, Spec-, Umgebungs-, Hardware- oder Workload-Abweichung fällt seriell zurück; N8 bleibt unverändert |
| N8/N10-Shadow-Router | getrenntes Paket `friday_avo_router/`; beide versiegelten Policies müssen autorisieren, reale Tensor-Metadaten bestimmen die Empfehlung, erzwungener Plan bleibt `serial_shadow_only`; CPU-, MLX-Metadaten-, History-, UI- und Security-Gates auf sauberem Commit bestanden | **Shadow-GO** nur für Beobachtung und als Gate zur getrennten Ein-Kernel-Vorregistrierung; keine optimierte Ausführung, keine produktive Integration und kein neuer formaler Claim |
| Phase 1B Residual+RMSNorm | statischer, quellhashgebundener Custom-Metal-Kandidat auf Commit `ea8f959`; Security-Diff, 566er Vollsuite, Qualification, A/A, A/B, Persistenz und UI abgeschlossen | Correctness und Messsystem bestanden; nur `1,870 %` Gain bei vorregistrierten `5 %` Mindestgewinn: **keine Promotion**, `baseline_fallback`, gültiger Negativbefund |
| H2 Gemma-Minimallauf | eine offline erzwungene Gemma-4B-Runde schlug `N=3,10,16` vor; Harness bestätigte explorativ `N=10` mit frischen drei Replikaten | nützliche Modellselektion beobachtet, aber Schema v1 bleibt `formal_claim=false`; keine Runtime-Erweiterung und keine zweite Runde |
| Prefill-Head-Skip formal | terminale 16-Record-Historie: versiegelte Präregistrierung, sechs bestandene A/A-Sessions, eingefrorene MDE `5 %`, sechs frische A/B-Sessions, `R=0,846385`, Gesamt-KI `[0,843147; 0,851284]`, `12/12` Tokenidentitätsgates | Gain ist nur für ein Gerät, einen gebundenen Gemma-4B-Snapshot, einen 897-Token-Prompt, Chunk `256`, Batch `1` und greedy ohne Prompt-Logprobs formal bestätigt; Integration bleibt freigabepflichtig |
| Persistenter Modellprozess | prospektiver Zyklus 13 mit zwei bestandenen A/A-Paaren, drei Charakterisierungs- und drei Validierungspaaren; Gesamtmedian `R=0,346968`, `6/6` exakte Tokenpaare, kein RSS-/Swap-Wachstum | **Engineering-Gain nur im gemessenen Scope**, gerechnet `−65,3032 %`, `formal_claim=false`; normaler Dienstpfad und automatische Aktivierung bleiben freigabepflichtig |
| Gemma-4B-Planer | prospektiver Zyklus 14 mit drei frischen Prozessen; `3/3` exakt gleiche greedy Antworten und richtige ID im Rohtext, aber `0/3` Antworten ohne Markdown-Rahmen | **`planner_contract_failed`**; keine Selbstoptimierung, kein Geschwindigkeitsgewinn und keine Aktivierung, `formal_claim=false` |
| Zwei-Modell-Planner | prospektiver Zyklus 15 mit sechs balancierten Paaren und zwölf frischen, seriellen Prozessen; 1B und 4B jeweils deterministisch `6/6`, aber strict contract/parser/candidate `0/6` | **`no_planner_qualified`**; 1B: Markdown, falscher Schlüssel `persistent_service_id`, End-of-turn-Trailer; 4B: Markdown-Codeblock trotz richtiger ID; `formal_claim=false` |
| N10-v1 / N10-v2 formal | V1 auf `c3e582c` vor Timing terminal; V2 auf `959df09` mit registrierter Fixture-Identität versiegelt und mit 6 A/A- plus 6 A/B-Sessions terminal abgeschlossen | `N=10`-Batch-Dispatch ist für genau ein Gerät, FP16-`2048²`, zehn Matmuls und den festen Plan jenseits 5 % bestätigt; nur ein begrenzter N10-Runtime-Prototyp ist freigegeben |

## Zyklus 15 — Zwei-Modell-Studie: Vor-Hardware-Vertrag und Ergebnis

Die aktuelle Nutzerfreigabe ist exakt auf die Studie
`dual-model-evidence-planner-20260824-01` begrenzt. Sie erlaubt ausschließlich die
vorregistrierte Prüfung der zwei bereits lokal vorhandenen Modelle in einem festen
Planungsfall und hebt keine früheren NO-GO-Grenzen auf. Es gibt **keine Matmul-On/Off-
Integration**; ein solcher Vergleich wurde weder durchgeführt noch erfunden.

Die Vorregistrierung bindet:

| Schlüssel | Modell | Snapshot-Revision |
| --- | --- | --- |
| `1b` | `mlx-community/gemma-3-1b-it-4bit` | `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` |
| `4b` | `mlx-community/gemma-3-4b-it-4bit` | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |

Zyklus `15` nutzt sechs feste Paare und zwölf frische, serielle Prozesse: Paare
`1–3` laufen `1b → 4b`, Paare `4–6` `4b → 1b`; jedes Modell läuft genau sechsmal.
Der einzige akzeptierte Planerfall ist die JSON-Auswahl
`persistent_service_qualification`. Die Studie bewertet keine allgemeine
Modellqualität, kein Lernen, keinen Code und keine Aktivierung. Der Claim bleibt
`formal_claim=false`.

Der eingefrorene Vor-Hardware-Code ist über die folgenden SHA-256-Werte gebunden:

- Präregistrierung: `246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`
- Worker: `b1db90d306d5de5c6ff466d046c5c617c5dd42cdaee3f6f7b4bcd5bf2a024bc0`
- Harness: `59691f50a1f33d4930b36ccce24ec701af74ebd0f9f095912a75e15a28978470`
- read-only UI: `5db9bf832c17470c0899ee0fd4062b42d524904e1ee3224894e87a7bed049607`

Der staged Diff-Check meldete vor Hardware wegen drei Trailing-Spaces in der
Präregistrierung Exit `2`. Ausschließlich diese Formatzeichen wurden entfernt.
Studienvertrag, Modelle, Zeitplan, Gates, Entscheidungstabelle und
`formal_claim=false` blieben semantisch unverändert; es gab keine
Vertragsänderung und keine Hardwareausführung.

Der finale fokussierte Offline-Stand umfasst `47` Tests und `42` Subtests mit
Exit `0`; auch `py_compile` endete mit Exit `0`. Diese unabhängige Test-Luna-
Verifikation wurde in diesem reinen Dokumentationsschritt nicht erneut ausgeführt.
Der Zyklus-14-Dokumentationsaudit ist auf Commit `ee12bb5` verankert.

### Zyklus 15 — einziger realer Hardwarelauf

Der eingefrorene Vertrag wurde nach dem Preflight genau einmal ausgeführt:
`dual-model-evidence-planner-validation-20260824-01`. Der Lauf bestand aus sechs
balancierten Paaren und zwölf frischen, seriellen Python-Prozessen; Paare `1–3`
liefen `1b → 4b`, Paare `4–6` `4b → 1b`. Kein Prozess lud beide Modelle, jeder
Prozess lud genau einmal. Es gab keine Wiederholung, keinen Download, keine
Installation und keinen Push. Der Lauf endete mit Exit `1`, weil die unveränderte
Entscheidungstabelle `no_planner_qualified` zurückgab; das ist ein gültiges
funktionales Negativergebnis, kein Messabbruch. `formal_claim=false`.

| Messung | Gemma 3 1B 4-bit | Gemma 3 4B 4-bit |
| --- | ---: | ---: |
| strikter Vertrag / Parser / `candidate_id` | `0/6` / `0/6` / `0/6` | `0/6` / `0/6` / `0/6` |
| Determinismus innerhalb des Modells | `6/6` | `6/6` |
| Ausgabe / Abschlussgrund | `32 Token / length` | `23 Token / stop` |
| TTFT Median / MAD | `0,295451312 / 0,0005528535 s` | `0,796846125 / 0,0088023125 s` |
| Modellarbeit Median / MAD | `0,4608839165 / 0,0005743330 s` | `1,0487644165 / 0,0092854165 s` |
| Prozess-Walltime Median / MAD | `4,2468557705 / 0,0059329165 s` | `4,883630417 / 0,0182606455 s` |
| Peak-RSS | `1.937.965.056 B` | `3.765.420.032 B` |
| MLX-Peak | `1.012.548.526 B` | `3.021.085.374 B` |
| Swap-Delta | `0 B` | `0 B` |

Die 1B-Antwort scheiterte wegen Markdown, des falschen Schlüssels
`persistent_service_id` statt `candidate_id` und mehrerer
`<end_of_turn>`-Trailer. Die 4B-Antwort enthielt die richtige ID, aber ebenfalls
einen unerlaubten Markdown-Codeblock. Das sind ausschließlich Vertragsbefunde;
keine allgemeine Qualitätsbewertung. Die dekodierten Texte der beiden Modelle
waren direkt in `0/6` Paaren bytegleich.

Alle Ressourcen-, Budget-, Snapshot-, Pairing- und Fresh-Process-Gates bestanden.
Gemessene Guarddaten: `9,205052 s` Gesamt-Modellarbeit, maximal `1,151402 s`
zusammenhängend, `178,475444 s` Walltime, Duty-Faktor `0,15`, keine Abbrüche,
kein Swap-Wachstum. Die gepaarten Verhältnisse und Bootstrap-KIs sind berechnet,
nicht zusätzliche Messungen: TTFT `0,373014193`
`[0,365603946; 0,377539933]`, Modellarbeit `0,439069434`
`[0,434598134; 0,444460794]`, Walltime `0,872042394`
`[0,864987297; 0,939562889]`, Tokenrate `3,168801108`
`[3,130352029; 3,201472197]`. Daraus folgen berechnet ungefähr `12,8 %`
kürzere 1B-Walltime und `48,5 %` weniger 1B-Peak-RSS. Weil beide Modelle das
Funktions-Gate verfehlten, entsteht daraus keine Präferenz und keine Aktivierung.

Evidenz: [`experiments/dual_model_planner/results.json`](experiments/dual_model_planner/results.json)
SHA-256 `7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`;
private Startmarke `.friday-data/dual-model-planner/attempt.json`, SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`;
Präregistrierung SHA-256
`246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`.
Die Markerdatei liegt in einem Verzeichnis mit Modus `0700` und selbst mit
Modus `0600`. Die lokale read-only UI bestätigte GET/HEAD `200`, schreibende
Methoden `405` und fremde Host-Header `421`; Ergebnis- und Markerhash blieben
unverändert. Zyklus 15 verwendet JSON-Rohdaten, keine eigene Zyklus-15-SQLite-DB.

Es gibt weiterhin keinen echten Gemma-Pfad mit einem Matmul-Optimierungsschalter
„mit/ohne“ und keinen vollständigen Matmul-A/B-Vergleich. Das wurde nicht gemessen
und wird nicht behauptet; es bleibt ein separater, künftig vorregistrierungs-
pflichtiger Kandidat. Ebenso sind allgemeine Modellqualität, allgemeine
Planner-Fähigkeit, selbstlernende Runtime und Produktaktivierung nicht belegt.
Multi-Turn-Fortsetzung und mehrere parallele Requests bleiben ungemessen. Die
Freigabe für den einzelnen Zyklus-15-Lauf ist verbraucht; jeder weitere
Hardwarelauf benötigt eine neue ausdrückliche Freigabe und einen neuen Zyklus.

### Vollständiger Zyklus-15-Preflight

Der finale, rein lokale Preflight wurde ohne Hardware- oder Modellausführung
geschlossen:

- Worker-Selbsttest `17/17`, Exit `0`; Harness-Selbsttest `25/25`, Exit `0`;
- Defaultaufruf ohne Ausführungsfreigabe: Exit `78`; weder private Startmarke noch
  `results.json` wurden erzeugt;
- `py_compile` und `compileall`: jeweils Exit `0`;
- fokussierte Suite: `47` Tests plus `42` Subtests, Exit `0`, Wall `3,36 s`,
  Peak-RSS `60.801.024 B`;
- vollständige `pytest`-Suite: Exit `0`, Wall `45,43 s`, Peak-RSS
  `200.523.776 B`;
- `git diff --check`, AST-Parsing und `xcodebuild -checkFirstLaunchStatus`:
  jeweils Exit `0`;
- ProjectAtlas-Runtime- und Konfigurationsprüfung: jeweils Exit `0`, Runtime
  `0.4.5-rc1`;
- read-only Versions-/Geräteprüfung: MLX `0.32.0`, mlx-lm `0.31.3`, Defaultgerät
  `Device(gpu, 0)`; dies war nur Introspektion und keine GPU-Rechnung;
- read-only Resolverprüfung: 1B-Revision
  `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` mit `732.577.304 B`
  Gewichten sowie 4B-Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2` mit `3.400.569.562 B`
  Gewichten; kein Snapshot wurde geladen;
- die Präregistrierung liegt nach dem Formatfix bei SHA-256
  `246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe`;
- ignorierte `__pycache__`-Verzeichnisse sind vorhanden, gehören aber weder zum
  Studienartefakt noch zu einem späteren Commit.

Nach allen Prüfungen blieben
`experiments/dual_model_planner/results.json` und
`.friday-data/dual-model-planner/attempt.json` abwesend. Es gab keine
Hardwarearbeit, keine GPU-Rechnung, keinen Modellload und keinen Commit.

Zum Zeitpunkt des dokumentierten Preflights blieben die Ergebnisse absichtlich
leer: Es existierten noch keine Hardwareevidenz, keine private Startmarke und keine
`results.json`. Der nachfolgende Hardwareabschnitt dokumentiert den einzigen
zulässigen Lauf.

### Zyklus-15-Postflight

Nach dem Hardwarelauf bestand die fokussierte Suite mit `47` Tests und `42`
Subtests bei Exit `0`; die vollständige Suite sammelte `744` Tests und endete
ebenfalls mit Exit `0`. `compileall`, die strikte JSON-Prüfung von Ergebnis-,
Verifikations- und Matrixdatei, `json.tool`, AST-Prüfung,
`git diff --check` und `xcodebuild -checkFirstLaunchStatus` endeten jeweils mit
Exit `0`. ProjectAtlas meldete zunächst `refresh_required`; genau ein
inkrementeller Refresh war anschließend erfolgreich. Runtime `0.4.5-rc1` und die
projektlokale MCP-Konfiguration waren gültig. MLX `0.32.0`, mlx-lm `0.31.3` und
`Device(gpu, 0)` wurden nur read-only geprüft; es gab keine Modell- oder
GPU-Arbeit.

Die Evidenz blieb unverändert: Ergebnis-SHA-256
`7c87c8cfd884b302641d77f2edb186e402d20a2a2f9a108c896ba88062d8523d`,
Verifikations-SHA-256
`24696c679de567519e8f2b3b034f0833de8122569072b71feeae794c05bbf4e6`,
Marker-SHA-256
`ed4e97d61d0fa43ee31dc551c3de7c74d65001080d4f7bb55dca7da3d0774327`;
alle DB-Hashes blieben unverändert. Die Verifikation meldet leere
Abweichungen, Entscheidung `no_planner_qualified` und `formal_claim=false`.
ProjectAtlas hatte keine getrackten Änderungen; bestehende untracked Fixture-
`.gradle`-Verzeichnisse wurden nicht angefasst.

Vorregistrierte Fehler und erfolgreiche Lösungen, die vor Folgeschritten gelten:

- doppelte Paare/Run-Positionen werden durch explizite Paar- und Schedule-IDs sowie
  Duplicate-Rejection ausgeschlossen;
- unvollständige Läufe können nicht mehr als Erfolg aggregiert werden; partielle
  Rohereignisse bleiben erhalten und der Entscheid bleibt fail-safe;
- per-Run-Content-Hashing wurde aus den gemessenen Zeitfenstern entfernt und als
  einmaliger Parent-Pre-/Post-Manifestcheck außerhalb der Messung umgesetzt;
- Snapshot-Revision, Snapshot-/Gewichts-Hashes und der gerenderte Prompt werden
  vom Parent an den Worker gebunden und vor/nach dem Load geprüft;
- Partial-/Fehlerpfade speichern den Teilstand, brechen terminal ab und erzeugen
  keinen erfolgreichen Claim;
- ein bereits validiertes Worker-Event bleibt auch dann in der partiellen Evidenz
  erhalten, wenn erst die anschließende Ressourcenprüfung terminal abbricht;
- eine feste stdout-Obergrenze von `1.000.000` Byte wird vom Parent überwacht;
- die UI akzeptiert nur die feste Run-ID sowie die geschlossenen Model-,
  Kandidaten- und Decision-Allowlisten und bleibt read-only;
- ein minimaler Fehlerreport ohne `metrics` wird als kontrollierte Fehlerform
  verarbeitet und nicht als erfolgreicher Studienreport aufgewertet;
- Ressourcen- und Budgetprüfungen laufen in fester Reihenfolge vor einer
  erfolgreichen Aggregation; ein Ressourcenabbruch gewinnt vor Korrektheits- und
  Vertragsauswertung.

ProjectAtlas meldete beim initialen Start zunächst `refresh_required`; beim
anschließenden Refresh trat einmalig ein SQLite-Lock auf. Der Retry war erfolgreich.
ProjectAtlas selbst und das eingebundene Upstream-Repository wurden nicht geändert.

Die produktive Research-DB enthält `10` verifizierte `legacy_summary`-Zeilen und
`4` native Ereignisse: drei gültige Berichte mit Rohmessungen sowie einen
sanitisierten Guard-Abbruch. Datei: `118.784 B`, Modus `0600`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision `c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Ein zweiter Import war idempotent (`0` neu, `10` bereits vorhanden) und ließ den
damaligen Dateihash unverändert. Der vollständige Offline-Testlauf nach
Implementierung des Runtime-Prototyps bestand mit `468` Tests und `2.463`
Subtests in `34,58 s` (Wall des umgebenden Prozesses `34,87 s`, User `135,36 s`,
System `3,38 s`, Exit `0`). Der letzte H1-v2-Implementierungsstand davor lag bei
`455` Tests und `2.447` Subtests in `33,04 s`; die H1-v2-Baseline bei `439`
Tests in `32,01 s`.

Die eigenständige Head-Skip-Evidenz liegt read-only in
`.friday-data/head-skip-v1.sqlite3`: `16` hashverkettete Records, genau ein
terminaler Record mit `formal_claim=true`, Modus `0600`, Größe `77.824 B`, SHA-256
`15ee462bbad5a8f757373f093fdf2ccfb8bdd0048c03447c1cb635acd38ec8d9` und
Kettenkopf `8a568e61f0e087794b1997f273e580c72e7f5abaa1eb8bad7954b303dd38a2d4`.
Die UI lieferte ihren realen Verlauf per GET aus; der read-only Replay bestand und
ließ die DB unverändert. Es wurde keine Runtime aktiviert.

Die Zyklus-13-Evidenz liegt in
`experiments/persistent_process/results.json`, SHA-256
`3925d83139cb6278c2b0aa103716e36a33f550f852bd30758976090fa0f7024`.
Die einmalige Startmarke liegt privat unter
`.friday-data/persistent-process/attempt.json`; der Hardwareprozess lief genau
einmal. Die lokale read-only UI zeigte acht Verlaufszeilen und denselben Entscheid.
Es wurde kein persistenter normaler Dienst aktiviert.

Die Zyklus-14-Evidenz liegt in `experiments/planner_4b/results.json`, SHA-256
`64a72331d1a415ae1dac191fecdf9c69cd43f5c11566c2df5ec091cf50a60975`.
Der Hardwarelauf lief am Netzteil genau einmal von Vor-Commit `8067dc6`; die
private Startmarke hat SHA-256
`6e741162f6d02ec69ee74ad7670b8e1a5046a3bc1430b946d7511b1248a6d573`.
Peak-RSS `3.764.961.280` Byte, MLX-Peak `3.021.085.374` Byte und
Swap-Delta `0` hielten alle Gerätegrenzen. Die lokale read-only UI zeigte drei
Verlaufszeilen und denselben negativen Entscheid. Das Ergebnis wurde nicht
wiederholt oder nachträglich gelockert.

### Unabhängiges Audit von Zyklus 14

Das Audit bestätigte die getrennte lokale Provenienz: Vor dem Hardwarelauf lag
`8067dc6c1fb175f0df539394b2e4dad5894b14b8` vor; der unveränderte Ergebnis- und
Dokumentationsabschluss liegt in `8923467c57d61d3599c430687b949052e397a95c`.
Die Präregistrierung blieb bei SHA-256
`0fa346db7985cdd4dfa49015b395ee0f9d56a097a06f3828b0c161c45e53e5ec`, die
Ergebnisdatei bei `64a72331d1a415ae1dac191fecdf9c69cd43f5c11566c2df5ec091cf50a60975`
und die private Startmarke bei
`6e741162f6d02ec69ee74ad7670b8e1a5046a3bc1430b946d7511b1248a6d573`.
Die drei PIDs waren verschieden, jeder Prozess meldete einen Modellload, und
Tokenfolge sowie Rohtext waren `3/3` identisch. Der Rohtext enthielt die richtige
ID, aber einen unerlaubten Markdown-Codeblock; die unveränderte Entscheidung ist
`planner_contract_failed`, `formal_claim=false`.

Unabhängig erneut geprüft wurden die 17 Offline-Tests (Exit `0`), der Worker-
Selbsttest `11/11`, der Harness-Selbsttest `9/9`, AST-/JSON-Parsing,
`git diff --check` (Exit `0`) und `xcodebuild -checkFirstLaunchStatus` (Exit `0`).
Die read-only UI antwortete auf GET mit `200`, auf den Snapshot-GET mit `200`
und auf einen fremden Host mit `421`; die Ergebnis-SHA blieb dabei unverändert.
HEAD wurde nicht als erfolgreich dokumentiert und antwortete im Audit mit `501`.

Offen und absichtlich unverändert bleibt, dass `results.json` keinen
Gewichts-SHA speichert. Der lokal separat verifizierte SHA von
`model.safetensors` ist `94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3`,
ist aber keine nachträgliche Ergänzung der alten Evidenz. Die drei absichtlichen
Markdown-Trailing-Spaces in der versiegelten Präregistrierung werden wegen des
eingefrorenen Hashes nicht entfernt. Es gab keinen erneuten Hardwarelauf und
keinen Security-Check.

Abschlussprüfung am 24.08.2026: ProjectAtlas-Refresh ohne Timeout, Runtime
`0.4.5-rc1`, gültige projektlokale MCP-JSON-Konfiguration,
`xcodebuild -checkFirstLaunchStatus` Exit `0`, MLX-GPU-Gerät Apple M1 Max und
vollständige `.venv/bin/python -m pytest -q`-Suite bei `100 %` mit Exit `0`.

Das Evidenzaudit korrigiert die frühere Statussprache: Der dokumentierte formale
A/A-Loader verlangt global genau sechs kompatible Prozesse, die append-only H0-DB
enthält aber neun `aa_gpu`-Runs aus mehreren Generationen. Mindestens ein relevanter
Prozess war `warmup_unstable`; zudem fehlte allen historischen Runs eine Root-Git-
Revision. Unmittelbar vor dem ersten A/B-Lauf waren hierarchischer Bootstrap,
formales A/A-Gate und MDE noch nicht geschlossen. Spätere gepaarte, replizierte
H1/H2-Zahlen bleiben technisch wertvoll, können aber nicht rückwirkend
vorregistriert werden.

Aktueller Entscheid: Der **begrenzte N8-Runtime-Prototyp** hat seine Gates
bestanden; die genau eine H2-Gemma-Runde ist abgeschlossen. N10-v1 bleibt ein
gültiger terminaler Engineering-Negativlauf ohne Timingdaten und wird nicht
wiederholt. Der separate korrigierte N10-v2-Vertrag band diesen V1-Endzustand,
lief ohne Gemma oder adaptive Auswahl vollständig durch und bestätigte den
festen N10-Dispatch-Plan formal. Der davon getrennte, allowlist-basierte
N10-Runtime-/Runtime-lite-Prototyp hat nun auch sein Cold-Load-/CPU-Gate und sein
gepaartes MLX/GPU-Gate bestanden. Der danach getrennt implementierte N8/N10-
Shadow-Router hat ebenfalls alle vorregistrierten CPU-, reale Tensor-,
Persistenz-, UI- und Sicherheitsgates bestanden. Das ist weiterhin keine
produktive Integration und kein allgemeiner Agenten-, Modell- oder
Hardwareclaim. Die bestehenden N8-/N10-Runtimes blieben unverändert.
Der ausdrücklich freigegebene einzelne statische Custom-Metal-Kandidat wurde
inzwischen isoliert geprüft. Er bestand Correctness, Speicher und A/A, erreichte
gegen `fast_rms_norm` aber nur `1,870 %` statt der vorregistrierten `5 %` und
wurde deshalb nicht promoviert. Produktive Phase 1B, adaptive Kernelsuche und
breiterer Live-Suchraum bleiben **NO-GO**, Cross-Device bleibt **NO-CLAIM** und
weitere Modellrunden bleiben **NO-GO**.
Details: [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md),
Persistenzvertrag: [`docs/H1H2_EVIDENZ_ARCHITEKTUR.md`](docs/H1H2_EVIDENZ_ARCHITEKTUR.md).
Der initiale Auditlauf installierte nichts, lud nichts herunter und führte keinen
GPU- oder Modelllauf aus. Nach späterer ausdrücklicher Rechenfreigabe wurden die
unten dokumentierten lokalen Läufe ausgeführt; auch dabei gab es weder Download
noch Installation.

N10-v1 enthält genau zwei replaybare Records: Präregistrierung
`3233c5ee…8985b` und terminalen Fehler `3ce4477a…92e49`; keine Timing-Session und
kein formaler Claim. Datei: Modus `0600`, SHA-256
`e0b5f4af62c128938e1e12e388c16b344a66e18eebf9e0568c7ebe34c5a4f0d5`,
Snapshot-Revision
`bbc75d60b5cfc61a1037c0a104e117a89561ec63e13240ca9b84f1bc98c08976`.
Der V2-Produktions-Fixture-Test reproduzierte alle vier registrierten Digests.
`22` fokussierte Tests und `10` Subtests bestanden in `40,28 s`; die nach dem
abschließenden Dateihash- und Mutationsgrenzen-Hardening wiederholte
vollständige Suite bestand mit `508` Tests und `2.480` Subtests in `207,82 s`.
V2-Spec-Fingerprint:
`66a01028b5c7ba6cd7b05faef1f3100413d793c6b4d7e3982bea671fb9bba6cd`.
Auf dem sauberen Commit `959df09b9d197edbd0a0984eda25092997b4ab23` band der
Seal die Provenienz `17d0dd505e349a4bbb7ffde3c291a3a44226d0fce79c235ce2ce890289e0c9ef`.
Die sechs A/A-Sessions ergaben `R=0,999586`, 95%-KI
`[0,998764; 1,000443]`; rohe MDE `0,0857 %`, konservativ eingefroren auf `5 %`.
Die sechs A/B-Sessions waren byteidentisch und ergaben insgesamt
`R=0,874912`, 95%-KI `[0,871768; 0,875614]`, entsprechend `12,509 %` weniger
Zeit. Charakterisierung (`R=0,875216`) und Validierung (`R=0,874608`)
bestanden das Gain-Gate getrennt. Der terminale Record `47283e73…e1249`
trägt als einziger `formal_claim=true` und erlaubt
`permit_bounded_n10_runtime_prototype`.

`.friday-data/n10-v2.sqlite3` enthält 16 vollständig replaybare Records, Modus
`0600`, Größe `180.224 B`, SHA-256
`54e9c57ca6b76fa671b94f748b7ee471575b7dd7445bad00ae3cab38f691fc4f`,
Snapshot-Revision
`9c9a94a8f799f2eb29b9e03c4e1b6e681aa945199753158cf8fc8c317b06090d`.
Die UI auf Port 8771 lieferte GET/HEAD `200` und wies POST mit `405` ab; der
vollständige Replay kostet derzeit `3,42–3,44 s` je Snapshot. Ein manueller
`Ctrl-C`-Stop beendet den Server, erzeugt aber noch einen sichtbaren
`KeyboardInterrupt`/Exit `1`. Während N10-v2 lief kein Modell und es gab weder
Download noch Installation.

Der getrennte N10-Runtime-/Runtime-lite-Pfad verwendet Runtime-ID
`n10-runtime-dispatch-20260822-01`, Application-ID `FRN1`, DB
`.friday-data/runtime-n10.sqlite3` und UI-Port `8772`. Der Controller prüft den
exakten formalen DB-Hash und Snapshot, replayt N10-v1 als Vorgänger, vergleicht
die versiegelten N10-Code-/Spec-/Umgebungs-/Hardwarefingerprints, beobachtet
reale Tensoren und cached danach nur die unveränderliche Policy. Unsicherheit
fällt seriell zurück; ein Batchfehler wird nicht wiederholt und verriegelt den
Circuit Breaker. `17` fokussierte Tests und `9` Subtests bestanden in `3,54 s`;
die abschließende vollständige Regression bestand mit `525` Tests und `2.489`
Subtests in `211,66 s`. Der Implementierungsstand wurde lokal auf `main` als
Commit `5eaad38ec0f5da4b01bd9d64237d3736f548ff14` versiegelt; es erfolgte kein
Push. Die saubere Runtime-Provenienz lautet
`02784bd7108767008c9951724421cc3f841390d463a8c6b153b059c5c497e22c`.

Der einmalige CPU-Lauf bestand alle Gates: Cold Load `3,482664083 s`,
Policy-Median `12,372 µs`, p95 `12,448 µs` und zusätzlicher Median
`12,343 µs`. Der danach zulässige einmalige MLX/GPU-Lauf bestand ebenfalls:
zwölf balancierte Blöcke, Baseline-Median `20,797459 ms`, Kandidaten-Median
`18,220750 ms`, `R=0,875753`, Effekt `−12,425 %`, byteidentisch und
`max_abs_error=0`. Die zwei Engineering-Records
`f140083d…89306` und `d6143fca…c979f` sind append-only hashverkettet und
erweitern den formalen Studienclaim nicht.

Die finale Runtime-Datei enthält genau zwei Records, Modus `0600`, Größe
`53.248 B`, SHA-256
`81286ffa2af11a814ffe4e11cdd67ce7fa5804ff42f4efd094cf161dbae22cd5`,
Snapshot-Revision
`a7b9352b913e62b9faf1e59cec2f5531435121d716e08cf2e7f8f24075f6327e`.
Der read-only Replay ließ den Dateihash unverändert. Die echte UI lieferte auf
Port `8772` GET/HEAD `200`, wies POST mit `405` ab und beendete sich per
`Ctrl-C` mit Exit `0` ohne Traceback. Es lief kein Modell; weder Download noch
Installation fanden statt.

## Evidenzgebundener N8/N10-Shadow-Router

### Benennung: warum `avo` in Bezeichnern stehen bleibt

Der Konzeptname wurde am 23.08.2026 durchgängig auf **Runtime-lite** und
**Shadow-Router** geändert. Drei Klassen von Vorkommen blieben bewusst unverändert:

- **Bezeichner in versiegelten Records**: `avo-shadow-router-20260822-01`,
  `avo-router-policy-20260822-01`, `avo-router-shadow-20260822-01`. Sie stehen so in
  `metadata.router_id` und `records.entity_key` der append-only Datei und beschreiben,
  was tatsächlich gespeichert ist.
- **Pfade in `provenance_json.code_files`**: die zehn Dateien unter
  `friday_avo_router/` sowie `tools/run_avo_router.py`. Der Record bindet Pfad *und*
  Inhalt an `code_sha256`; ein Umbenennen ließe die Provenienzprüfung ins Leere laufen
  und der Router fiele dauerhaft seriell zurück.
- **Gebundene Spezifikationen**: `docs/AVO_SHADOW_ROUTER_SPEC.md` steht in
  `spec_files` mit `spec_sha256`, `docs/N10_RUNTIME_PROTOTYPE_SPEC.md` ebenso für die
  N10-Runtime. Auch eine reine Textänderung darin bricht den Hash.

Eine vollständige Umbenennung ist damit kein Suchen-und-Ersetzen, sondern eine neue
Vorregistrierung mit neuer ID, neuer Datei und wiederholten Gates. Sie würde die
bestehenden terminalen Records entwerten und ist **nicht** erfolgt.

Der neue Router verwendet ID `avo-shadow-router-20260822-01`, Application-ID
`FRR1`, DB `.friday-data/avo-router.sqlite3` und UI-Port `8773`. Er lädt die
unveränderten N8-/N10-Policies, autorisiert eine Empfehlung nur, wenn beide
exakten Evidenzpfade autorisieren, und leitet den Scope ausschließlich aus
realen Tensor-Metadaten ab. Der Router besitzt keine `execute`-Methode. Auch bei
einer N8-/N10-Empfehlung bleibt der tatsächlich erzwungene Plan
`serial_shadow_only`; falsche Form, falscher Datentyp und nicht registrierte
Operandenzahl melden auch auf Routerebene `route=serial`.

Die Implementierung wurde lokal auf `main` als Commit
`70bc451f764d36e75de0a1c9ac61849717e577e8` versiegelt; es erfolgte kein Push.
`19` fokussierte Tests bestanden in `0,135 s`. Die vollständige Regression
bestand mit `544` Tests in `210,574 s` (`210,84 s` außen, `199,23 s` User,
`1,47 s` System, maximales RSS `76.496.896 B`, keine Swaps). Der zusätzlich
versiegelte Security-Diff-Scan prüfte zehn Produktionsdateien sowie Tests und
Dokumentation als manuelle Add-backs mit kompletter Coverage und null
reportablen Findings. Sein lesbarer Report hat SHA-256
`0c5a558d908d45b9e6561a5caf90e8bc5d929856ba4e441ecda44ccb282d983b`.

Der einmalige CPU-Lauf bestand alle sechs Gates: Cold Load `7,176239584 s`,
direkter Policy-Median `12,138946 µs` (MAD `0,042467 µs`), Router-Median
`13,719000 µs` (MAD `0,044638 µs`), Router-p95 `13,815279 µs` und gepaarter
zusätzlicher Median `1,585208 µs`. Alle `21` balancierten Blöcke verwendeten
je `10.000` Entscheidungen pro Arm; beide direkten Policies und der Router
stimmten überein. Der terminale Record lautet
`a1a1c1a08eb22c41e442becfe7d6a6a2feb67c2322596eaf1d9fc0a595b253fd`.

Die genau einmal zulässige MLX-Shadow-Validierung bestand danach alle fünf
Gates. Exakt acht beziehungsweise zehn FP16-`2048²`-RHS-Tensorreferenzen
empfahlen N8/N10, während Operandenzahl neun, falsche Form und FP32 seriell
blieben. Direkte Policy und Router stimmten in allen Fällen überein;
`no_matmul_executed=true`. MLX meldete `33.554.432 B` aktiven Speicher,
`33.554.436 B` Peak und `8 B` Cache; maximales Prozess-RSS war `51.560.448 B`.
Der Record lautet
`19e36e7b32209d62afa5eae54973e2dc326a1bd0efaa0d8b8a73737463384c6c`.

Die finale Router-Datei enthält genau diese zwei vollständig replaybaren,
hashverketteten Engineering-Records, Modus `0600`, Größe `36.864 B`, SHA-256
`128c090de37a79606f35c564d19035f0bcffedcea4b4018fa618cffedc58c6f8`
und Snapshot-Revision
`b1c0832c0957e5a2d0e88bda1409f8d4b04be036a5edd32a20ea8c2d57b2c758`.
Die echte UI lieferte GET/HEAD `200`, POST `405`, die vorregistrierten
Security-Header und beendete sich per `Ctrl-C` mit Exit `0`. Ihr Replay ließ
den DB-Hash bytegleich. Es lief kein Modell, keine Matmul und kein Custom-
Metal-Kernel; es gab weder Download noch Installation.

## Phase 1B: statischer Residual-Add-plus-RMSNorm-Kandidat

Der einzelne Kandidat wurde vor jeder Compilation in
[`docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md`](docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md)
gebunden, durch zwei vollständige Security-Diff-Snapshots geprüft und lokal als
Commit `ea8f95980ac6da513c374aa658b4d2d4cc4a9d20` versiegelt. Maßgeblicher
Security-Snapshot:
`codex-security-snapshot/v1:sha256:232c04fe0b4cfbb896120961d91f779b582e04c71b54cbad6bedcaca4c88fa26`;
komplette Coverage, acht Flächen, null Findings. Die vollständige Regression
bestand `566/566` Tests in `212,552 s`. Es erfolgte kein Push.

Die einmalige Qualification bestand sechs vollständige Correctnessfixtures und
alle Vergleiche gegen vier MLX-Baselines. Der maximale Kandidatenfehler war
`0,001953125`, die erste Compilation plus Eval dauerte `193,044 ms`, MLX-Peak
war `89.153.552 B`. Record: `1ff03f1b…56d6e`.

Die drei A/A-Prozesse bestanden mit `R=1,003445`, 95%-KI
`[0,997767; 1,009240]`. Die feste Tie-Regel wählte `fast_rms_norm` aus den zwei
innerhalb `0,5 %` liegenden spezialisierten Baselines. Drei frische A/B-Prozesse
ergaben `R=0,981298`, 95%-KI `[0,972124; 0,985900]`, also rund `1,870 %`
Gewinn. Correctness, jede Einzelsession und Speicher bestanden, aber das
vorregistrierte 5-%-Gate scheiterte. Terminaler Record: `f051b1f8…595a6`;
Status `candidate_inconclusive`, Aktion `baseline_fallback`, kein formaler Claim
und keine Aktivierung.

Die Phase-1B-Historie enthält genau diese zwei Records, Modus `0600`, Größe
`86.016 B`, SHA-256
`4ba0cbd679083683b2504dbf174691402aa851967b88befadeb4035145558452`,
Snapshot-Revision
`45a65df53f27ad8c79cfb6583be3566a1b4bb41f160e68e530bfeec5a1ab031b`.
Die read-only UI auf Port `8774` lieferte GET/HEAD `200`, POST `405`, wies einen
fremden Host mit `421` ab und ließ den DB-Hash unverändert. Es lief kein Modell;
nichts wurde heruntergeladen oder installiert.

## Formales H1-v2-Ergebnis und begrenzte N8-Runtime

Die formale Studie lief vollständig auf dem sauberen Commit
`1fbe73c69cedeb69284a264c5e3f45e3e393b822`. Die Präregistrierung bindet Code,
Spezifikation, Python/MLX-Umgebung und Apple-M1-Max-Hardware; alle zwölf Sessions
liefen am Netzteil in getrennten Prozessen mit realem Inter-Session-Cooldown.

Die sechs A/A-Sessions ergaben ein aggregiertes Verhältnis `1,000109` mit
95%-Intervall `[0,999193; 1,000540]`. Die rohe kalibrierte MDE war rund
`0,0752 %`; prospektiv blieb deshalb der konservative Floor von `5 %` maßgeblich.
Alle vier Kalibrierungsgates bestanden. Die sechs anschließenden A/B-Sessions
waren byte-identisch und ergaben insgesamt `R=0,879718`, 95%-Intervall
`[0,877045; 0,880403]`, Effekt `−12,028 %`. Charakterisierung
(`R=0,879415`) und Validierung (`R=0,880044`) bestanden das Gain-Gate getrennt.
Der terminale Record
`f508fc9e2b1f44a1b60084bdbeca581024f1f3599535b3dd662a9305c99a9357`
trägt als einziger `formal_claim=true` und erlaubt nur
`permit_bounded_runtime_prototype`.

Die formale Datei `.friday-data/h1-v2.sqlite3` enthält `16` vollständig
replaybare Records, Modus `0600`, Größe `163.840 B`, SHA-256
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`.
Ein erneuter read-only Runtime-Preflight ließ diesen Hash unverändert.

Der getrennte Prototyp ist in `friday_runtime/` und
[`docs/RUNTIME_PROTOTYPE_SPEC.md`](docs/RUNTIME_PROTOTYPE_SPEC.md) definiert.
Er autorisiert Batching nur bei exakt derselben terminalen H1-Entscheidung,
unverändertem H1-Code/Spec-Fingerprint, derselben Umgebung/Hardware, sauberem
Worktree und aus tatsächlichen Tensoren abgeleitetem Workload. Alle anderen
Fälle wählen seriell. Ein Batch-Fehler wird nicht im selben Aufruf wiederholt,
sondern verriegelt alle Folgeaufrufe seriell. Im absichtlich schmutzigen
Entwicklungsstand verifizierte der reale Preflight alle `16` H1-Records und fiel
korrekt mit `worktree_dirty` auf seriell zurück. Eine Live-Messung erfolgte vor
dem sauberen Runtime-Commit bewusst noch nicht.

Auf dem anschließend sauberen Commit
`0b0a893f58e9c757a0aa7b49565a8b1c1eb2a561` autorisierte derselbe Preflight den
exakten Scope. Das CPU-Gate (5 Warmups, 21 balancierte Blöcke, je 20.000 Aufrufe)
ergab Policy-Median `11.045 ns`, p95 `11.078 ns` und gepaarten zusätzlichen
Median `11.017 ns`; Record
`a9c08e2b4d79590e1cfa1d5270c53a80a69b1ff1f39507f003fcd6d8d2be1815`.
Alle Grenzen von `25/50/20 µs` bestanden.

Die anschließende MLX/GPU-Validierung (2 Warmup-Paare, 12 balancierte Blöcke)
ergab seriell `20,360 ms`, Runtime-Batch `17,643 ms`, gepaartes
`R=0,879209` und Effekt `−12,079 %`. Die acht Outputs waren byte-identisch,
maximaler absoluter Fehler `0,0`; Circuit Breaker blieb offen. GPU-Arbeit
`0,667252 s`, Wall im Guard `1,059850 s`, maximale kontinuierliche Last
`0,667252 s`, MLX-Peak `209.715.200 B`, RSS-Peak `440.401.920 B`. Record:
`643af8606c83cbcd0a591ba63bebb8745ddf5d4a346971c1d733c8d2b566c2dc`.
Der Policy-Median entspricht rund `0,063 %` der Kandidatenlaufzeit.

`.friday-data/runtime.sqlite3` enthält damit zwei vollständig replaybare,
hashverkettete Records, Modus `0600`, `45.056 B`, SHA-256
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`;
Snapshot-Revision
`a53e6b31c8266b1881ebebfc4dca8c28e9a4177d7648496863fc2b6d4cd6eb3f`.
Read-only UI-Snapshot und H1-Readback änderten keine Datei.

## Geschlossener H2-Gemma-Minimallauf

Auf dem sauberen Dokumentationscommit
`99267d3422f5a8573cad0f53e7009a4cf8f52198` lief genau eine Runde des bereits
implementierten `model-loop`. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` und
der projektlokale Resolver schlossen einen Netzwerkfallback aus; geladen wurde
ausschließlich der vorhandene Snapshot
`mlx-community/gemma-3-4b-it-4bit` Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, eine MLX-Gewichtsdatei mit
`3.400.569.562 B`. Es gab keinen Download und keine Installation.

Die Modellantwort war ausschließlich `[3, 10, 16]`; Parser und Allowlist ließen
genau diese drei Integer-Kandidaten zu. Explorative 20-Block-Messungen:

| Batchgröße | B/A-Ratio | 95%-Intervall |
| ---: | ---: | ---: |
| `3` | `0,849019` | `[0,797567; 0,903789]` |
| `10` | `0,784921` | `[0,741686; 0,830676]` |
| `16` | `0,889566` | `[0,881424; 0,897784]` |

Der Harness wählte `N=10` und bestätigte es separat mit drei Replikaten
`0,6649/0,6716/0,7014`: hierarchisch `R=0,671573`, 95%-Intervall
`[0,648895; 0,731190]`, explorativer Effekt `−32,84 %`. Korrektheitsgates und
5%-Schwelle bestanden. Der Bericht bleibt explizit `formal_claim=false`, weil
das Modell drei Kandidaten aus vorhandener Evidenz auswählte und diese Studie
nicht prospektiv als formale N=10-Bestätigung registriert war. Die produktive
Runtime bleibt deshalb auf `N=8`; `N=10` fällt dort seriell zurück.

Der Guard verbuchte `9,908610 s` GPU-Arbeit, maximal `1,730481 s`
kontinuierlich, `180,024674 s` Kandidaten-Cooldown, `16,022076 s` Pflichtpausen
und `212,268826 s` Wall. Evidenz-ID:
`5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a`.
Die Research-DB enthält nun `14` verifizierte Zeilen, davon `4` native und eine
native `model-loop`-Zeile mit Rohdaten; Modus `0600`, `118.784 B`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision
`c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Der read-only Replay ließ den Hash unverändert. Es wurde keine zweite Runde
gestartet.

## Neue native v1-Exploration nach Rechenfreigabe

Alle folgenden Läufe waren an einen sauberen Root-Commit gebunden, liefen am
Netzteil unter dem gemeinsamen Guard und wurden vor stdout append-only
persistiert. Schema v1 erzwingt weiterhin `formal_claim=false`.

**Einzeloperation vor dem Modelltest:** FP16-Matmul `2048²`, acht Operationen,
drei Replikate mit je 25 gepaarten Blöcken. Batched Dispatch war byte-identisch
und erreichte `R=0,780054`, hierarchisches 95%-Intervall
`[0,765530; 0,877456]`, Effekt `−21,995 %`. GPU-Arbeit `2,803 s`, Wall
`11,468 s`; Evidenz-ID `b866022a…a92eb6`. Das überschreitet explorativ die
fixierte 5%-Schwelle, ist aber noch kein formaler H1-v2-Nachweis.

**Lokaler Gemma-Roofline-Lauf:** Die Werkzeuge lösen ausschließlich validierte
Snapshots im Projektcache auf; Repository-ID, Revision und Gewichtsumfang stehen
im Bericht. Fünf Messwiederholungen je Modell, 369 Prompt-Token:

| native v1 | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Snapshot | `2d44e83d…` | `93724907…` |
| Folge-Token | `5,012 ms` / `199,5 Token/s` | `10,949 ms` / `91,3 Token/s` |
| Prefill | `0,3271 s` / `1.128,0 Token/s` | `0,8735 s` / `422,4 Token/s` |
| geschätzte Bandbreitennutzung | `36,53 %` | `58,47 %` |
| geschätzte FP16-Rechennutzung | `2,78 %` | `4,45 %` |
| exploratives Roofline-Urteil | speicherbegrenzt | speicherbegrenzt |

Gesamtbudget: `10,360 s` GPU-Arbeit, maximal `1,129 s` kontinuierlich,
`52,123 s` verifizierte Pausen und `68,111 s` Wall. Evidenz-ID
`31c20b1e…647c36`, Commit `faa4f88`. Ein erster Versuch auf Commit `29a2b74`
wurde korrekt vor 4B abgebrochen, weil Warmups/Wiederholungen ohne Zwischenpause
die `6-s`-Kontinuierlichkeitsgrenze überschritten; Fehler-ID
`ffe98ffa…1a0ac4`. `pace_generation` und drei Regressionstests schließen diese
Lücke.

Die neue Rohmessung reproduziert die Richtung der historischen Roofline-
Zusammenfassung, wertet sie aber nicht formal auf. **Auf diesem damaligen Stand**
blieb Phase 1B/Custom Metal **NO-GO**. Das zuvor fehlende Schema/Protokoll v2 wurde anschließend unter
`friday_h1/` implementiert, offline verifiziert und auf dem sauberen Commit
`1fbe73c` formal ausgeführt; das Ergebnis steht im vorherigen Abschnitt.

## Historisches Arbeitsprotokoll

Die folgenden Abschnitte erhalten die damaligen Messungen und Entscheidungen. Wo
sie „bestätigt“ sagen, ist das die **historische explorative Klassifikation**, nicht
der aktuelle formale Evidenzgrad.

## Historischer H0-Implementierungsstand

- H0 ist eine einzelne FP16-`2048²`-Matmul-Workload, kein Modelltest und kein
  Self-Optimization- oder Hardware-Generalisation-Nachweis.
- Der Offline-Unterbau umfasst SQLite v1 unter `.friday-data/h0.sqlite3`, festen Worker
  Option A und ein read-only Dashboard auf `127.0.0.1`. Die lokale Datenbank enthält
  `22` Runs einschließlich Run22; Run22 ist eine abgeschlossene H0-eager-baseline
  reference. Das ist kein Modell-, A/A- oder Self-Optimization-Nachweis.
- `run_mlx` und der statische Schalter `mlx-run --execute` sind implementiert. Ohne
  `--execute` endet der Befehl vor Runner-/Worker-/Benchmark-/MLX-Import weiterhin mit
  `EXIT_MLX_LOCKED=78`/`state=not_released`.
- W1v3 wurde in Benchmark, Worker, Runner und Aggregation umgesetzt und offline geprüft:
  äußere Warmup-Blöcke von mindestens `50 ms`, maximal `4096` Evals mit
  `repetition_window_unreachable` bei Nichterreichen, Gate `round(block_ns/evals)`,
  `8..16` Blöcke, ±`5 %` für die letzten fünf Gate-Werte, geschlossene bounded
  Block-Summaries sowie Warmup-Fehlerdiagnose Schema v2 mit v1-Readback. Das tote
  `_Timed.output`-Feld ist entfernt.
- Run22 lief genau einmal. Der Common-Wrapper ist `completed/measurement_complete/
  baseline_fallback` mit `error=null`; gemäß Worker-/Runner-Vertrag sind die verschachtelte
  `benchmark_classification=baseline_reference`, `benchmark_action=not_run` und
  `aggregation_required=false` die erfolgreiche eager-baseline-reference. Die anfängliche
  Operator-Deutung von `baseline_fallback` als Fehler wurde anhand dieses Vertrags korrigiert;
  es liegt kein Produktfehler vor.
- H0-Baseline läuft. Weder `aa_gpu` noch Optimierung oder Self-Optimization sind bewiesen.

## Historischer H0.1-Stand — separate Stationaritätsforschung

- H0.1 ist strikt von H0 getrennt. Es besitzt einen vorregistrierten Paced-Trajectory-
  Vertrag mit exakt sechs vorgesehenen Sessions `C0,V0,C1,V1,C2,V2`, einen
  stdlib-only Analyse-/Study-Core, eine eigene append-only SQLite-v1-Datenbank und ein
  read-only Dashboard. Am 21.08.2026 wurden `6` Paced-Sessions und `1` Paced-Study
  auf dem Zielgerät ausgeführt; der H0.1-Stationaritätsentscheid liegt damit vor und
  lautet `h01_complete_unresolved`.
- Vier historische H0-Generationen wurden vollständig über den öffentlichen H0-
  Bundle-Verifier inventarisiert und durch exakte provenance-/strukturgebundene Adapter
  geschlossen behandelt. A (`runtime_unavailable`) ist eine erkannte Exklusion. B, C
  und D wurden am 20.08.2026 in genau einem atomaren Produktions-Execute als rein
  deskriptive `legacy_h0_warmup_observation`-Bundles importiert. Es gab keine
  H0-Reklassifikation und keine Leistungs- oder Stabilitätsaussage.
- `.friday-data/h01.sqlite3`: `3` verifizierte Legacy-Bundles, Größe `53,248 B`,
  Mode `0600`, SHA-256
  `fd2c6e56d5f108d6670745a338930d6050c38b03eac8cc050170a466818d9d57`.
  Execute-Report-SHA-256:
  `4e73ab2d7b0aa0bf0cb7e559550de254ddadfa41c0f31ee86b92b9203bef788f`;
  H0 blieb bytegleich bei SHA-256
  `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`.
- Das read-only H0.1-Dashboard zeigt Total `3`, Kind
  `legacy_h0_warmup_observation=3`, Status `legacy_observation=3` und Revision
  `d9bc6e5ab430b68e16c9b9dfa62463896c9ad9d64ef003a4a862460378b2af3f`.
  Der finale socketfreie Snapshot lief in `0.04000666690990329 s` bei Peak-RSS
  `29,261,824 B`. Die reale read-only HTML-/API-Grenze wurde danach auf
  `http://127.0.0.1:8766/` mit Exit `0` geprüft; der Server läuft in Session `40690`.
- Finale H0.1-Verifikation: `57/57` Tests und `2,244/2,244` Subtests,
  `0` Failures/Errors/Skips, Wall `25.714773458894342 s`, Self-User/System
  `22.572449/0.502031 s`, Peak-RSS `43,368,448 B`; NumPy-/MLX-Importe und
  Socketkonstruktionen jeweils `0`.
- Forschungsgrenze: Der Legacy-Import ist **GO** als Evidenzmigration. Der
  Sechs-Session-Paced-Vertrag wurde am 21.08.2026 auf dem Zielgerät ausgeführt und
  vollständig replayt; siehe den folgenden Abschnitt. Das war kein Modelltest; es
  wurde kein Modell geladen oder installiert.

## Störprozess aufgeklärt (21.08.2026)

- Der Untergrund, der ungepaarte Messung wertlos macht, ist charakterisiert:
  **unimodal mit langem rechtem Schwanz**, **zufällig verteilt** (Runs-Test über
  sechs Sessions: beobachtet ≈ erwartet), **blockweit** (`22` von `150` Blöcken
  treffen beide Arme, erwartet wären `4,1`) und mit einer **Zeitskala von rund
  `340 ms`** (Autokorrelation `+0,576` bei `68 ms`, `0,000` bei `408 ms`).
- Ein langsam variierender, gerätweiter Prozess — plausibel OS-Scheduling und
  fremde Last. Nicht eliminierbar, aber erklärend: Beide Arme eines Blocks liegen
  in derselben Störungsepisode, weshalb sich die Störung im Quotienten herauskürzt.
- Erklärt vier bisher getrennte Beobachtungen als ein Phänomen: das ungelöste
  H0.1, das zu breite A/A-Bootstrap-Intervall, die nicht funktionierende
  Cutoff-Metrik und die Wertlosigkeit ungepaarter Messung.
- **Neue harte Messregel:** Vergleichsarme innerhalb von rund `340 ms` messen.

## Roofline — die Inferenz ist speicherbegrenzt (21.08.2026)

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Bandbreite genutzt | `31,9 %` | `51,2 %` |
| Rechenwerke genutzt | `2,4 %` | `3,9 %` |
| Prefill je Token schneller | `7,3x` | `5,4x` |

- **Faktor `13`** zwischen beiden Auslastungen, in beiden Modellen `memory_bound`.
  Zwei unabhängige Wege (Auslastungsrechnung und Prefill-Vergleich) sagen dasselbe.
- **Konsequenz:** Code „näher an der Maschinensprache" optimiert den Anteil, der
  mit `2,4`–`3,9 %` ohnehin leerläuft. Wirksam sind nur weniger Bytes
  (Quantisierung — bei 4-bit-Modellen bereits eingelöst) und weniger Durchgänge
  (Kernel-Fusion).
- **Obergrenze:** Bei `51,2 %` Bandbreitenauslastung bringt selbst eine perfekte
  Optimierung ohne Gewichtsverkleinerung höchstens rund `2x`.
- Spitzenwerte `400 GB/s` und `21 TFLOPS` sind Herstellerangaben, nicht gemessen.

## Fusions-Layer — geprüft und verworfen (21.08.2026)

- Ein `mx.compile`-Wrapper über den Forward-Pass zeigte `−12,4 %` (1B) und
  `−15,0 %` (4B) bei bytegleichen Logits. **Praktisch wertlos:** die
  End-to-End-Messung an der echten Generierungsschleife ergibt `−0,5 %` und
  `−0,1 %`.
- Ursache belegt: Die Generierung übergibt bei jedem Aufruf einen KV-Cache
  (`18` Aufrufe mit Cache, `0` ohne), `mx.compile` kann `RotatingKVCache` nicht
  entgegennehmen, und **`mlx-lm` fusioniert bereits selbst**
  (`@partial(mx.compile, shapeless=True)` in `gemma3_text.py` und
  `activations.py`).
- Eigener Messfehler auf dem Weg: `model.__call__` auf der **Instanz** gesetzt —
  Python löst `obj()` über `type(obj).__call__` auf und ignoriert das
  Instanzattribut, der Patch war wirkungslos.
- Wert des Ergebnisses: Ein ganzer Lösungsweg ist **mit Begründung**
  ausgeschlossen. Eine wirksame Layer müsste unterhalb ansetzen — Cache-Layout,
  Speicherverwaltung oder eigene fusionierte Kernel.
- Werkzeuge `tools/measure_roofline.py` und `tools/measure_fusion_layer.py`;
  letzteres misst ausdrücklich den cache-freien Forward-Pass, **nicht** einen
  Generierungsgewinn.

## Historische explorative H2-Codegen-Beobachtung (21.08.2026)

- Nutzerfreigabe für Ausführung modellgenerierten Codes und erhöhtes GPU-Budget
  erteilt. Kein zweites Gerät, Cross-Device bleibt offen.
- **Bestätigt: `R = 0,8838`, `−11,62 %`, `95%-KI [0,8676, 0,8975]`**, drei
  Replikate. Fünf Pläne geschrieben, fünf gemessen, drei über der Schwelle.
- Drei Schutzschichten: heute semantisch begrenzte AST-Plansprache (ein
  Iterationslevel, höchstens `32` statisch gewichtete Matmuls, keine freien
  Allokationsprimitive), Prozessisolation mit Timeout/CPU-Grenze/bereinigter
  Umgebung sowie ein Correctness-Gate. Die MLX-Speichereinstellung ist nur eine
  Richtlinie, kein hartes OS-Limit.
- Zwei Anläufe scheiterten an eigenen Fehlern: Der Prompt zeigte die Baseline zu
  prominent (Modell kopierte sie viermal), und der Validator blockierte
  `out.append(x)` — genau die gesuchte Optimierung. Beides korrigiert und
  getestet.
- Werkzeuge `tools/plan_sandbox.py` und `tools/codegen_loop.py`, in der CLI als
  `codegen`. Gesamtsuite `387` Tests / `2.377` Subtests grün.
- Grenze: Der Plan bleibt eine Umsortierung derselben festen Rechnung. Keine
  Kernel, keine Algorithmenwahl, keine Numerikänderung — die Allowlist lässt das
  nicht zu.

## Vollständiger Testlauf und Selbstoptimierung (21.08.2026)

- **Testsuite `90 s` → `31 s`** (Faktor `2,9`) über `pytest-xdist`, in `pytest.ini`
  festgelegt. Untere Schranke ist ein `17,6 s`-Test mit `16` vollen Bootstraps;
  mehr Worker bringen nichts (Amdahl). Das Bootstrap zu beschleunigen wurde
  **verworfen**, weil `friday_h0/aggregation.py` in der geschlossenen Code-Liste
  steht und eine Änderung die Provenienz aller H0-Läufe brechen würde.
- **Sicherheitsfund: `aa` besaß kein `--execute`-Gate.** Der Prüfaufruf startete
  real einen A/A-Lauf; nur der Resume-Mechanismus verhinderte eine Aufzeichnung.
  Gate nachgerüstet, alle vier messenden Werkzeuge gesperrt. `ReleaseGateTest`
  prüft jetzt, dass jedes registrierte Werkzeug einer Gruppe zugeordnet ist —
  ein neues Werkzeug ohne Einordnung lässt die Suite fehlschlagen.
- **Entdoppelung:** `require_ac_power` (vier Kopien) und das Release-Gate (drei
  Kopien) liegen jetzt in `tools/_bench.py`. Dateiübergreifender Duplikatscan
  findet nichts mehr.
- `aa` hat jetzt auch `--self-check`; das README-Versprechen gilt damit für alle
  vier Werkzeuge.
- **`337` Tests / `2.322` Subtests grün in `31,3 s`**, Guard `pass`, Provenienz
  beider Phasen unverändert, End-to-End nach dem Refactoring bestätigt.

## Einsatzreife für Dritte (21.08.2026)

- **Einziger Einstieg `tools/friday.py`:** `list`, `doctor` und Durchreichen an die
  fünf Werkzeuge. `doctor` prüft Python, MLX/Metal, NumPy, `mlx-lm`, Netzbetrieb
  und Plattenplatz.
- **`docs/ERGEBNISSE.md`** fasst alle Befunde, sieben Nullbefunde, Grenzen und
  sechs Messregeln auf einer Seite zusammen; jeder Befund nennt sein
  Reproduktionskommando. Ersetzt für Einsteiger die `2.540` Journalzeilen.
- **README neu:** Kernbefund im ersten Absatz (ungepaart messen ist hier nahezu
  wertlos, Beispiel `mx.compile`), Schnellstart, Werkzeuge, Messregeln, Budgets,
  Grenzen.
- Keine absoluten Pfade im Code; `requirements-apple-silicon.txt` führt `mlx-lm`
  als optionale Zeile mit Dry-Run-Hinweis.
- `tests/test_friday_cli.py` (`12` Tests, `34` Subtests) prüft Werkzeugregistrierung,
  Release-Gates, Self-Checks und dass jeder Dokumentationslink auflöst.
- Gesamtsuite `326` Tests / `2.312` Subtests grün, H0.1-Guard `pass`.

## Historische explorative Self-Optimization-Loop-Beobachtung (21.08.2026)

- **`3` von `3` Läufen: `optimization_confirmed`.** Gewählt `N=8` (`−13,60 %`),
  `N=6` (`−11,13 %`), `N=6` (`−14,11 %`). Der Loop konvergiert auf `N=6`–`8`.
- **Autonomer Fund:** `N=6` und `N=7` kamen in der manuellen Suche nicht vor
  (dort nur `2,4,8,16`). Der Loop schlug sie selbst vor, maß sie und bestätigte
  einen davon unabhängig.
- Drei Runden: `explore` über eine feste Kandidatenmenge, `refine` mit selbst
  vorgeschlagenen Nachbarn, `confirm` mit `3` Replikaten und hierarchischem
  Bootstrap. Correctness-Gate vor jeder Zeitmessung: bytegleich oder verworfen.
- **Erst nicht reproduzierbar (`1/3`), Ursache Winner's Curse.** Rangfolge nach
  dem Punktschätzer wählte den glücklichsten Ausreißer (`0,750`, `0,741`), der bei
  unabhängiger Nachmessung auf `0,87`–`0,96` regressierte. Korrigiert auf
  Rangfolge nach Konfidenzobergrenze; die Schwelle `MDE = 5 %` blieb unverändert.
- Werkzeug `tools/optimization_loop.py` (`--execute`-Gate, Netzbetrieb, GPU- und
  Wall-Budget, `--self-check`), Tests `tests/test_optimization_loop.py` (`15`).
- Grenze: fester, von Hand definierter Suchraum. Kein Codegenerieren, keine
  Kernel — das bleibt H2 mit eigener Sicherheitsfreigabe.

## Historische explorative H2-Vorstufe — Modelltests Gemma 3 (21.08.2026)

- Nutzerfreigabe für Download und Installation erteilt, Auflage Projektordner
  eingehalten: `HF_HOME` auf `.friday-data/models`, Pakete im lokalen `.venv`.
  Belegt sind `3,9 GB`; `16 GB` bleiben frei.
- **Provenienz ungebrochen.** `uv pip install mlx-lm` zog `24` Pakete, ließ aber
  `mlx 0.32.0` und `numpy 2.5.2` unverändert. `environment_sha256` bleibt
  `74ca2dac…`, `code_sha256` H0 `101cdadf…` und H0.1 `f66e4b5a…` ebenfalls.
  Alle früheren Läufe bleiben vergleichbar.
- **Stufe 1, `gemma-3-1b-it-4bit`:** `737 MB`, TTFT `205 ms` ohne Pause,
  Folge-Token `5,0 ms` (rund `200 Token/s`).
- **Stufe 2, `gemma-3-4b-it-4bit`:** `3,40 GB` auf Disk, davon werden nur
  `2.560,8 MB` geladen. Folge-Token `11`–`12,8 ms` (rund `85 Token/s`),
  TTFT `304,8 ms` ohne Pause.
- **Widerlegte Annahme: es gibt keinen Vision-Tower-Offset im Speicher.** Der
  SigLIP-Tower (`833,7 MB`) und der Projektor (`5,9 MB`) liegen im Repo, werden von
  `mlx_lm.load` aber **nicht geladen**. Die früher geplante Bestimmung über die
  Peak-RSS-Differenz beider Stufen ist damit gegenstandslos; der Anteil wurde direkt
  aus dem safetensors-Index quantifiziert.
- **Cooldown-Effekt bei 4B bestätigt:** `R = 1,414` (`+41 %`),
  `95%-KI [1,209, 1,653]`, `10` Paare im direkten Wechsel. Betrifft nur die TTFT.
- **Der Effekt skaliert nicht mit der Modellgröße:** `1B` `1,37x`, `4B` `1,414x`.
  Trotz vierfacher Parameterzahl praktisch gleich — er ist eine Eigenschaft des
  Geräts, nicht der Arbeitslast, konsistent mit dem Matmul-Befund ohne jedes Modell.
- Zwei eigene Zwischenzahlen wurden durch sorgfältigere Messung nach unten
  korrigiert (`503,9 ms` bei 1B, `4,16x` bei 4B). Abgeleitete Regel: kein Befund
  aus weniger als zehn Wiederholungen, und Behandlungsarme im direkten Wechsel statt
  frei randomisiert, wenn die Behandlung eine Zeitkomponente hat.

## Cooldown-Effekt — isoliert und erklärt (21.08.2026)

- **Dosis-Wirkungs-Beziehung nachgewiesen.** Das erste Sample nach einer Pause ist
  verlangsamt, monoton mit der Pausenlänge: `0,94x` bei `0 s`, `1,89x` bei
  `0,25 s`, `3,67x` bei `2 s`, `4,12x` bei `20 s`. Sättigung bei rund `4x` ab
  etwa `2 s`. Ohne Pause ist der Exzess exakt `0,00`.
- **Ursache überwiegend GPU-Taktung.** Eine Idle-Pause von `5 s` ergibt `R = 4,02`;
  dieselbe Pause mit periodischer Mini-Matmul nur `R = 2,53` (Verhältnis `0,487`,
  `95%-KI [0,311, 0,762]`). Der MLX-Allocator scheidet aus: der Cache bleibt über
  die Pause konstant bei `8,6 MB`.
- **Keep-Alive ist keine brauchbare Optimierung.** Sieben Dosierungen gemessen; die
  beste senkt das erste Sample von `10,05 ms` auf `5,11 ms`, kostet aber `14,44 ms`
  eigene GPU-Zeit. Netto `−9,50 ms`. Alle Varianten netto negativ.
- **Der Effekt erklärt H0.1 nicht.** Post-hoc deskriptiv: ohne die ersten sechs
  Main-Samples bestünde `trend` `1/6`, `changepoint` `0/6`, `tail` `0/6` — identisch
  zum realen Ergebnis. Die H0.1-Instabilität stammt von über die Session verteilten
  Ausreißern. Zwei unabhängige Phänomene. Study unverändert
  `h01_complete_unresolved`.
- Praktische Konsequenz für künftige Messungen: Nach einer Pause gehen bis zu
  `5,12` Sample-Äquivalente verloren. Wer nach einer Pause misst, ohne den Anlauf
  zu verwerfen, verzerrt ein 80-Sample-Mittel um bis zu `5,8 %` — mehr als die
  H1-Nachweisschwelle von `5 %`.
- Werkzeuge: `tools/measure_cooldown_effect.py` (`--execute`-Gate, Netzbetrieb,
  GPU-/Wall-Budget, `--self-check`) und `tests/test_cooldown_effect.py` (`15` Tests).

## Historische explorative H1-Beobachtung — Dispatch-Plan (21.08.2026)

- **Ergebnis: `−14,7 %` bei `N = 8`, Optimum `−17,4 %` bei `N = 4`.** Gepaart
  gemessen, über fünf Replikate repliziert, `95%-KI [0,8263, 0,8777]`, Correctness
  `byte_identical`. Verdikt `effect_confirmed` gegen die vorab eingefrorene
  Schwelle `MDE = 5 %`.
- Der Kandidat ist eine **Ausführungsplanänderung**, keine Kernel-Optimierung:
  `N` Matmuls mit einer einzigen Synchronisation statt `N` einzelner
  Synchronisationen. Identische Arithmetik, bytegleiche Ergebnisse.
- `serial 2,572 ms/Matmul` gegen `batched 2,212 ms/Matmul`; Ersparnis `0,360 ms`
  je Matmul. GPU-Arbeit `5,8 s` gegen Budget `120 s`.
- **Ohne die A/A-Vorarbeit wäre das Ergebnis falsch gewesen.** Ungepaart erschien
  `mx.compile` mit `−27,6 %`; gepaart ergibt derselbe Kandidat `R = 1,0019`,
  `KI [0,9990, 1,0047]` — kein Effekt. Der scheinbare Gewinn war reines Rauschen.
- Ausgeschlossene Fehlerquellen: Deduplizierung identischer Teilausdrücke (alle
  Messungen mit paarweise verschiedenen Operanden), Ergebnis-Caching, veränderte
  Arithmetik.
- Geprüfte Nullbefunde: prätransponiertes `B` `+3,1 %`, `mx.einsum` `+0,6 %`,
  eigener GPU-Stream `±0 %`, echter 3D-Batch-Matmul `−3,9 %`/`−1,8 %` mit
  `1,0` im Konfidenzintervall. Die Optimierung ist damit ausgereizt.
- Reichweite: gilt für **unabhängige** Operationen. Keine Aussage über Modelle,
  Transformer-Inferenz oder andere Geräte. H0 und H0.1 bleiben unverändert.
- Werkzeuge: `tools/measure_dispatch_plan.py` (`--execute`-Gate, Netzbetriebspflicht,
  GPU-Budget, Correctness-Gate, `--self-check`) und `tests/test_dispatch_plan.py`
  (`12` Tests, offline).

## H0.1 — ausgeführte Sechs-Session-Paced-Study (21.08.2026)

- Nutzerfreigabe: ausdrückliche Freigabe für den GPU-Lauf. Keine Installation, kein
  Download, kein Modell. H0.1 misst die feste `2048²`-FP16-Matmul, nicht ein Modell.
- Zuvor fehlte der Ausführungspfad vollständig: `build_trace` nahm bereits
  aufgezeichnete `durations_ns` entgegen, aber niemand erzeugte sie. Neu sind
  `friday_h01/provenance.py`, `friday_h01/runner.py`, `friday_h01/cli.py` und
  `tests/test_h01_runner.py`. Der GPU-Pfad liegt hinter demselben
  `--execute`-Release-Gate wie H0 (`state=not_released`, Exit `78`).
- Preflight: `preflight_ok`, Parent Run22, `fixture_seed=4051312678`, Netzbetrieb
  `ac_power`, `thermal_state=api_unavailable`, Proben `24.60/4.80/3.86 ms`.
  Eingefrorener `code_sha256=f66e4b5a2444643fb375a098398bbd3829d717a7b956e62f46a6a54617986e94`.
- Lauf: sechs getrennte Prozesse `C0,V0,C1,V1,C2,V2`, alle `h01_session_complete`,
  Wall je Session `66.00`–`66.37 s`.
- Gate-Bilanz: `changepoint` 6/6 fail, `tail` 6/6 fail, `trend` 5/6 fail,
  `pacing` 5/6 fail, `ess` 1/6 fail, `acf` 0/6 fail; Summe `23`.
- Study: `h01-study-1812a894…c39ca`, `status=h01_complete_unresolved`,
  `conclusion=replicated_stationarity_not_supported`, `session_count=6`,
  `failed_gate_count=23`, `action=no_h0_conclusion`, `h0_reclassification=false`,
  `promotion_applicable=false`.
- **Ergebnis: replizierte Stationarität ist nicht unterstützt.** Das ist ein gültiges
  negatives Ergebnis, kein fehlgeschlagener Lauf. Der Envelope wird deutlich und nicht
  knapp verfehlt. Dominierend ist die Tail-Ratio `2,53`–`3,13` bei Grenze `1,20`;
  `acf` besteht überall und das Trendvorzeichen wechselt zwischen Sessions, die Daten
  sind also von sporadischen Ausreißern geprägt und nicht von gerichtetem Drift.
- Beobachtung am gleichen Messpunkt, ohne Ursachenbehauptung: Run22 misst rund
  `2,07 ms` je Matmul innerhalb eines dichten 32er-Batches, H0.1 misst dieselbe
  Operation einzeln mit `50 ms`/`750 ms` Pacing bei einem Median um `6,97 ms`. Die
  Ursache ist nicht gemessen. Festhaltbar ist nur, dass die H0-Baseline einen dicht
  gepackten Batch-Zustand charakterisiert und nicht die isoliert gepacte Einzeloperation.
- Verifikation: alle sechs Sessions und die Study unabhängig neu berechnet und
  bytegleich; Gesamtsuite `266` Tests / `2.265` Subtests grün; Dashboard-Snapshot
  Total `10` (`paced_session=6`, `paced_study=1`, `legacy_h0_warmup_observation=3`),
  Revision `a2d1b2469e21f01de04e03b747ac897bb602059d5ec5ceeb098dcba5b03b4e1b`.
- Betriebsrisiko: Eine aufgezeichnete Session ist nicht wiederholbar. Die `run_id` ist
  deterministisch aus Provenienz abgeleitet, die Messdaten sind es nicht; der
  append-only Store antwortet mit `StorageConflict`. Ein Abbruch mitten in der Study
  macht die bereits aufgezeichneten Sessions dieser Provenienz unbrauchbar. Ein
  Wiederholungslauf ist eine neue Study, kein Patch.
- Grenze: H0 bleibt unverändert. Keine Reklassifikation, keine Promotion, keine
  Performanceaussage, kein Nachweis von Self-Optimization oder Generalisation.

## Run22 — abgeschlossener eager-baseline-Reference-Lauf (20.08.2026)

- Nutzerfreigabe und Umfang: begrenzte W1v3-/Output-Fix-Umsetzung plus genau ein
  `eager_baseline`-Canary; kein `aa_gpu`, keine Installation und kein Retry. Der einzige
  Live-Befehl lief mit dem freigegebenen `eager_baseline`-Pfad. Run-ID:
  `h0-eager_baseline-characterization-0-14d435dcc2170feec70d8baaa712860e59a6148ca3f211aad98eff1c9d7cf0ff`.
- Ergebnis: äußerer `real=3.79 s`, Exit `10`, DB danach `22` Runs. Der Common-Wrapper ist
  `completed/measurement_complete/baseline_fallback`, `error=null`. Der verschachtelte
  Vertrag meldet `benchmark_classification=baseline_reference`,
  `benchmark_action=not_run`, `aggregation_required=false`; damit ist der eager-baseline
  Reference-Lauf erfolgreich abgeschlossen. Die anfängliche Operator-Deutung von
  `baseline_fallback` als Fail wurde anhand des Worker-/Runner-Vertrags korrigiert und ist
  kein Produktfehler.
- Warmup und Baseline: `8` stabile Warmups; Gate-Werte
  `[2566556,2179783,2188775,2143891,2155069,2174895,2195533,2192185]`, Median der
  letzten fünf `2174895 ns`; `30` Messblöcke mit jeweils `32` Reps; Calibration
  `68155792 ns`; Baseline-Median `2138574.859375 ns`, MAD `17041.671875 ns`, IQR
  `35343.0859375 ns`, Minimum `2105915.34375 ns`, Maximum `2210087.25 ns`.
- Correctness: Gate bestanden; `9/9` Cases und `86/86` Metrics. `abs_max=
  0.0310508173`, `normalized_l2=0.0002074681`, `abs_q99=0.0110023008`,
  `rel_q99_abs_oracle_ge_1=0.0004333980`.
- Speicher: active `16777216 B`, peak `25165824 B`, cache `8422698 B`, RSS-Peak
  `369655808 B`. Der Memory-Gate ist `not_evaluable_missing_required_metric`,
  `hard_limit=false`. Retention-Nachprobe nach dem Fix: `67108864 B` erzeugt,
  `0` Payloads/`0 B` retained; `_Timed` enthält nur `duration_ns`, `evaluation_ns`
  und `synchronize_ns`.
- Freeze und Artefakte: Code
  `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`, Spec
  `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; kein Git-Root.
  Manifest `73058165244fe505035182f0044dc5ab8bd16ef523ebfc44b44d5b6f616e239e`, Result
  `bda3d23d56e49c2d26bf7c3e73d52b61c3ea022c3fb61ab0719bfedef58a6d09`, Evidence
  `edaf6cae5a98185f183fd368189a8be3a56c194540e4f64300903cff42d1a6a0`, Projection
  `a51aa4b3cadf00dc5338eee199206b2b8f876c4fd3aeaaa2d5261364254ed790`, Bundle
  `a566c912032efab919dddf5ca7f67b986f29464a655abf15617733aeb6947c49`.
- Read-only Dashboard-Snapshot: `snapshot_id=325afcc9a45311ba716f64a51e7395cd7f2cf1c872c9a3f349c6daf9361398de`,
  `source_revision=7cdad7edcb6099894d588bb9927de322bd4f7ce02d256673768647db54131c73`,
  `run_count=22`, `completed`; der Dashboard-Socket war frei.
- Nach Abschluss wurde der read-only Dashboard-Server erneut auf
  `http://127.0.0.1:8765/` gestartet: `state=serving`, Session `4414`. Die integrierte
  Browseransicht traf vor dem Serverstart zunächst `connection refused` und blockierte
  danach den lokalen Reload per Browser-URL-Policy. Es gab keine Umgehung und keine
  Datenänderung; der socketfreie Snapshot bleibt die verifizierte UI-Evidenz, der direkte
  lokale Link ist verfügbar.
- Umgebung und Verifikation: voller einmaliger Pytest-Lauf Exit `0`, Wall `66.837 s`,
  `228 passed`, `2211` Subtests in `66.24 s`. Ein engerer Unittest-Lauf wurde nach
  `30.018 s` mit Exit `124` absichtlich gestoppt, nachdem `103` Marker grün und keine
  Fehler/Fehlschläge sichtbar waren; er wird nicht als Fehler verschwiegen. CLI-Lock:
  Exit `78`; Usage-Fehler: Exit `64`; `xcodebuild -checkFirstLaunchStatus`: Exit `0`.
  ProjectAtlas `0.4.5-rc1`; Python `3.12.13`, NumPy `2.5.2`, MLX `0.32.0`, macOS
  `26.5.2 arm64`. Der sandboxed Import meldete kein Metal-Gerät; es lief dabei keine
  MLX-Operation.
- Vorher/Nachher: vorher `206 passed + 47 subtests`, DB `21`, Retention `64` Payloads /
  `67108864 B` lebend, Code `aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c`,
  Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; nachher die
  Freeze- und Testwerte dieses Run22-Abschnitts.

Dieser eine Baseline-Lauf ist keine vergleichende Performanceaussage. H0-Baseline ist
damit ausführbar und referenziert; A/A, Optimierung und Self-Optimization sind nicht
bewiesen. Weitere Ausführung benötigt eine neue ausdrückliche Freigabe.

## Historischer Live-Pfad-/Preflight-Stand vor Run22 — 20.08.2026

Die folgenden Abschnitte dokumentieren frühere Preflight-, Run20- und Run21-Stände.
Sie sind historisch und nicht der aktuelle Projektstatus.

- Live-Pfad `45/45 OK`: Wall `4.022908 s`, User/System `3.149974/0.200018 s`, Peak-RSS
  `42,139,648 B`; keine Self-/Child-Aufteilung belegt. Nach dem Fix auf die reale
  `get_cache_memory`-API: `16/16 OK`, Wall `0.086906 s`, User/System
  `0.140900/0.054489 s`, Peak-RSS `49,938,432 B`, ebenfalls ohne belegte Aufteilung.
- Aktuelle Nicht-Live-Suite: `133/133`, Wall `23.720160 s`, User/System
  `22.722187/0.559409 s`, Self-/Child-Peak-RSS `71,368,704/23,642,112 B`;
  unabhängiger Replay `133/133`, Wall `23.588426 s`, User/System
  `22.769535/0.504137 s`, Self-/Child-Peak-RSS `60,342,272/23,707,648 B`. Ein echter
  Importguard belegte, dass dabei keine MLX-Matmul-/GPU-Workload lief.
- Socketfreies Dashboard: `4/4` plus `3` Setup-Subtests, Wall `0.001793 s`, User/System
  `0.001437/0.000137 s`, Self-/Child-Peak-RSS `31,457,280/0 B`, null Socketaufrufe.
  Historisch bestand eine autorisierte HTTP-Prüfung `13/13`; spätere Sandbox-Bindefehler
  und der nicht wiederholte `16`-er Scope sind weder Produktfehler noch finaler Grünnachweis.
- Der Sandbox-Preflight hatte kein Metal. Der danach autorisierte Zielgeräte-Smoke
  bestätigte MLX `0.32.0` und eine 1-Element-Operation in Tool-Wall `1.741108708 s`;
  das war keine Matmul und kein H0-Ergebnis. Eine spätere reine API-Prüfung bestätigte
  `get_cache_memory`, ohne die API oder eine GPU-Workload aufzurufen.
- Canary: äußerer Wall `0.166578416 s`; Child User/System `0.106607/0.040468 s`, Child-
  Peak-RSS `28,442,624 B`, gespeicherter Worker-RSS `23,150,592 B`. Äußere Self-User/
  System/RSS wurden nicht separat gemessen. Ergebnis
  `invalid/runtime_unavailable/baseline_fallback`, Fehler
  `NumPy import unavailable: ModuleNotFoundError`: `0` Rohsamples, `0` Correctness-
  Zeilen, `3` Supervisor-Scalars und `1` Projection-Artifact; Performance, Ratio/KI,
  Warmup/Repetitions und MLX active/peak/cache fehlen. Kein `aa_gpu`, keine Promotion.
- Canary-Hashes: Code `246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b`,
  Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`, Manifest
  `11ac87fb704169e58ac506eda5d0549a91ad19e8ff52b43c5bb7f28e61d982c1`, Result
  `cb97e223fd26c87aa1f1e3a87e56b4c61c76c5b69e7d0420721392727e31aa02`, Evidence
  `406c42b4a99f72703b9623fd8ba5e5c0e68c46495f5a7bd0db1cef1674e0499d`, Projection
  `d9071855d3b1dc6318aa8c832c66c368314ef9ce4ff790911dd4a96939fdaf24`, Bundle
  `1de0c11763c38462420bb74277d8018b2db1517f9eb17e234938b27681a8c41b`.
- Ursache: `Path(sys.executable).resolve()` kollabiert den lexikalischen Launcher
  `/Users/tobiasburandt/Project_Friday/.venv/bin/python` zum Basisinterpreter
  `/opt/homebrew/Cellar/python@3.12/3.12.13_2/Frameworks/Python.framework/Versions/3.12/bin/python3.12`;
  dadurch fehlt in der bereinigten Worker-Umgebung die venv-Paketsuche.
- Minimaler Vorschlag: den fest erwarteten absoluten, aber lexikalischen venv-Launcher an
  `Popen` übergeben und Launcher, Parent und Ziel vor/nach Spawn eng über Owner, Modus,
  Typ, Device und Inode prüfen. Restgrenze: pfadbasiertes `Popen` ist nicht fd-gebunden;
  vollständige TOCTOU-Schließung erfordert Helper/`fexecve` und eine neue Architektur.
  Status: **AWAITING USER APPROVAL**.
- DB: `16` Runs (`15` unveränderte Offline-Controls + Canary). Zwei stabile read-only
  Dashboard-Snapshots: `snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
  `source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
  `run_count=16`, `returned_count=16`, `truncated=false`, `query_only=1`. Die UI zeigt
  die Historie ohne Schreibzugriff; nur der Beobachtungszeitpunkt variiert.

## Historischer Offline-Pre-Live-Nachweis — anderer Scope

- Hauptsuite ohne Dashboard: `177` bestanden, `3` Windows-Skips und `12` Subtests;
  Wall `26.034290 s`, Total User/System `23.373336/1.227233 s`, Self-Peak-RSS
  `15,499,264 B`, Child-Peak-RSS `74,186,752 B`.
- Socketfreie Dashboard-Prüfung: `4/4` plus `3` Setup-Subbranches; Wall `0.002041 s`,
  Peak-RSS `31,260,672 B`. Eine frühere autorisierte HTTP-Prüfung bestand `13/13`;
  der spätere `16`-er HTTP-Scope wurde nach der letzten Härtung wegen Sandbox-/Usage-
  Limit nicht final wiederholt. Es wird kein grüner finaler `16`-er HTTP-Lauf behauptet.
- Der CLI-Lock wurde bestätigt: `mlx-run` endet mit Exit `78`, ohne Runner oder Worker zu
  importieren.
- Dritte Offline-Control-Generation:

  | Lauf | Wall | Exit | Ergebnis |
  |---|---:|---:|---|
  | slow | `0.191745 s` | `10` | `regression` |
  | known | `0.156192 s` | `0` | nur synthetisch |
  | wrong | `0.157021 s` | `10` | `correctness` |
  | missing | `0.157268 s` | `10` | `missing` |
  | exit70 | `0.145334 s` | `10` | `worker_exit` |
  | replay | `0.159542 s` | — | `idempotent` |

  Sequenz: `0.967681 s`, Self-Peak-RSS `16,334,848 B`, Child-Peak-RSS
  `28,819,456 B`; Provenance `5745e93f…39d57`, Replay-Bundle
  `6ae4a453…b7335`.
- Finale DB-Evidenz: `15` Runs (`3 × 5`), jeder mit genau einem verifizierten
  `common_result`; die älteren `10` Runs blieben unverändert. Snapshot:
  `source_revision=3b70324f…ab658d`, `id=512934c9…b5b52`, `run_count=15`, nicht
  abgeschnitten. DB-Größe `229,376 B`, Datei `0600`, Verzeichnis `0700`,
  `query_only=1`. Das sind Offline-Controls, keine H0-Hardwarewerte.

Der `177`-er Scope ist historisch und anders enumeriert; er ist mit dem aktuellen
`133`-er Scope nicht als Regression oder Zuwachs vergleichbar.

## Verifiziert

- macOS 26.5.2 (Build 25F84)
- Xcode 26.6 (Build 17F113) unter `/Applications/Xcode.app`
- `xcode-select` zeigt `/Applications/Xcode.app/Contents/Developer`
- `xcodebuild -checkFirstLaunchStatus` erfolgreich
- Python 3.12.13 und uv 0.11.19
- MLX 0.32.0 im bestehenden `.venv`; die aktuelle Luna-Read-only-Introspection bestätigte
  `mx.matmul`, `mx.eval`, `mx.synchronize`, `mx.compile` sowie
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`
- ProjectAtlas-Runtime 0.4.5-rc1 unter `/Users/tobiasburandt/.local/bin/projectatlas`
- ProjectAtlas-Codex-Plugin 0.4.5-rc1 installiert; offizieller Marketplace auf `v0.4.5-rc1`
- Codex-MCP-Server `projectatlas` aktiviert und auf die Project-Friday-Datenbank versioniert
- Historischer Setup-Snapshot (nicht aktuell): `projectatlas init` und anschließender
  `watch --once` indizierten 543 Dateien und 257 Ordner; der damalige lokale Index meldete
  281 offene Purpose-Hinweise.
- `scripts/verify_environment.sh` erfolgreich: Xcode, ProjectAtlas, MLX Metal, PyTorch MPS und
  alle drei MCP-JSON-Dateien geprüft

## Heutige Read-only-Audits

- Erster Sandbox-Lauf: Exit 1 wegen `RuntimeError: No Metal device available`; Ursache ist der
  fehlende GPU-Zugriff innerhalb der Sandbox.
- Genehmigter Lauf außerhalb der Sandbox: Exit 0; Tool-Walltime 1.741108708 s.
- Der aktuelle Luna-Read-only-Introspektionslauf im bestehenden `.venv` bestätigte MLX 0.32.0
  und die APIs `mx.matmul`, `mx.eval`, `mx.synchronize`, `mx.compile` sowie
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`. Die Sandbox hatte kein Metal; daher fand in diesem Lauf
  kein GPU-Lauf statt.
- Es wurden keine lokalen KI-, Modell- oder Software-Installationen und keine Downloads ausgeführt.

## Projektintegration

- Repository: `/Users/tobiasburandt/Project_Friday/ProjectAtlas`
- Projektlokale ProjectAtlas-Daten: `/Users/tobiasburandt/Project_Friday/.projectatlas/`
- generierte MCP-Dateien: `projectatlas.mcp.json`, `projectatlas.claude.mcp.json`,
  `projectatlas.opencode.json`
- vollständiges Konzept kopiert nach `docs/TECHNISCHES_KONZEPT.md`
- Phase-1A/H0-Vorregistrierung ergänzt: `docs/PHASE1_MATMUL_SPEC.md` (Matmul-
  Messsystem-Preflight;
  kanonische FP16-`2048²`-Performance-Workload, separate Correctness-Matrix, Correctness-/
  Memory-/Safety-Gates, Prozess-/Bootstrap-Regeln und Fallback).

## Atlas- und Indexstände

- `543 Dateien / 257 Ordner / 281 Purpose-Hinweise` ist ein historischer,
  definitionsgebundener Stand aus dem früheren `init`-/`watch --once`-Audit; die ursprüngliche
  Verifizierungs-Bullet oben bezeichnet ebenfalls nur diesen damaligen Snapshot. Er wird nicht
  mit späteren Atlas-Zählungen zusammengeführt.
- Der aktuelle Post-Edit-Atlas-Snapshot meldet Generation `22`, `549 Dateien` und
  `257 Ordner`.
- Die aktuelle Atlas-Overview meldet `280` fehlende Purpose-Angaben; das ist eine separate
  Overview-/Coverage-Metrik.
- Der aktuelle Session-Brief meldet `805` Blocker. Dieser Wert ist eine separate
  Session-/Health-Metrik und nicht identisch mit Datei-, Ordner- oder Overview-Zahlen.

## Offen nach Run22

- optional: ProjectAtlas-Purpose-Queue gezielt für die wenigen Projektdateien kuratieren; die große
  Upstream-Codebasis muss nicht manuell mit erfundenen Zwecken versehen werden;
- Der Offline-Unterbau für SQLite v1, das read-only `127.0.0.1`-Dashboard und Worker
  Option A einschließlich Pre-Live-Adapter ist offline implementiert und final geprüft.
  Die feste DB enthält `22` Runs; Run22 ist ein einzelner Baseline-Reference-Lauf, kein
  Hardware-Optimization-Loop.
- Der separate H0.1-Unterbau und die historische Evidenzmigration sind implementiert
  und verifiziert. Nächster wissenschaftlicher Schritt ist nicht ein weiterer H0-Retry,
  sondern die einmalige Durchführung des vorregistrierten Sechs-Session-Paced-Protokolls
  und dessen geschlossener Study-Replay. Bis dahin bleibt H0.1 `unresolved`.
- Die kritische Neubewertung reklassifiziert Phase 1A zu H0: Sie darf nur Messsystem-,
  Correctness-, Kontrollarm- und Fallbackverhalten belegen, keine Self-Optimization oder
  Hardware-Generalisation. Ein Forschungspivot auf H1 deterministic template-constrained
  tuning ist freigegeben, aber erst nach H0-Go/No-Go und A/A-3+3-Aggregation
  wissenschaftlich zu planen.
- `docs/PHASE1A_ARCHITEKTURFREIGABE.md` dokumentiert SQLite v1, read-only Loopback-
  Historien-Dashboard und Worker Option A als `approved/implemented-offline`. Nach der
  späteren allgemeinen Nutzerfreigabe wurde ausschließlich der dokumentierte H0.1-
  Legacy-Import ausgeführt; `aa_gpu`, Custom Metal, Modelle und weitere Optimierung
  wurden nicht ausgeführt.

## Phase-1A-Readiness

- Vorregistrierte Operation: `Y = mx.matmul(A, B)`, FP16 C-contiguous `2048²`, exakt
  17.179869184 GFLOP und 25,165,824 Bytes A+B+Y-Nutzdaten.
- Correctness-only-Matrix separat vorregistriert: sichtbare Seeds `0xC0DE0001` bis
  `0xC0DE0005` sowie Holdout-Seeds `0xC0DE1001` und `0xC0DE1002`; sie ist nicht Teil der
  Performanceaggregation. Zero-RHS muss exakt null sein; die `64²`-Sign-Invariante wird
  innerhalb des eingefrorenen Fehler-Envelopes geprüft.
- Timing-Contract: pro Output `mx.eval(out)` und vor Zeitfensterende `mx.synchronize()`;
  `time.perf_counter_ns`, `mx.eval` und `mx.synchronize` werden im Manifest benannt.
- MLX 0.32.0 und die aufgeführten Matmul-, Eval-, Compile- und Memory-APIs wurden in
  einem früheren read-only Introspektionslauf im bestehenden `.venv` bestätigt; die
  Sandbox hatte kein Metal, daher gab es keinen GPU-Lauf.
- `mx.compile` ist als sichere Framework-Vergleichsvariante eingeordnet; A/A ist der echte
  H0-Nullpfad. Es ist kein Custom-MLX-Metal-Kandidat.
- Keine Tests, Downloads, Installationen, GPU-Läufe oder Modelltests in diesem
  Dokumentationsschritt; die früheren fokussierten Offline-Testmetriken sind im
  append-only Arbeitsjournal vermerkt. Es wurden keine Modelle ausgewählt oder
  festgeschrieben.

## Offen: isolierter Ein-Kernel-Versuch und vollständiger Phase-1-DoD

Der vollständige DoD aus `IMPLEMENTIERUNGSPLAN.md` ist mit Phase 1A/H0 nicht erfüllt. Offen
bleiben eine separate Phase 1B mit begrenztem Custom-MLX-Metal-Kandidaten sowie dessen
Ausführung in einem isolierten Worker-Prozess mit Timeout, Ressourcenlimits, Correctness-
Test und Rollback. Die separate Sicherheits-/Architekturfreigabe und das vorgeschaltete
Shadow-Router-Gate liegen inzwischen vor; Vorregistrierung, Worker-Implementierung und
Messung des genau einen statischen Kandidaten sind noch offen. Eine breitere Phase 1B ist
nicht freigegeben. Die H1-Workload-/Shape-Familienaufteilung und eine cluster-level
Powerplanung bleiben ebenfalls offen.

## Grenzen dieses Status

Die Smoke-Tests beweisen nur Erreichbarkeit und einfache Korrektheit. Sie beweisen keine stabile
Performanceverbesserung, keine optimale Kernelkonfiguration, keine Neural-Engine-Kontrolle und keine
Übertragbarkeit auf iOS, Android, NVIDIA oder Rechenzentren.

## Historischer finaler Vertragsabschluss und Run21-Canary — 20.08.2026

Dieser Abschnitt ist der historische Run21-Stand; der vorstehende Run22-Abschnitt ist
maßgeblich.

Der finale Contract-Stand ist dokumentiert und offline geprüft: Core `175/0`, Dashboard
`4/4`, `0` offline MLX-Imports. Die zugehörige Provenienz ist
`575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`; Code-Hash
`aae3245e…` (im übergebenen Evidenzsatz nur als Präfix vorhanden), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac` und Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 wurde genau einmal ausgeführt und fail-closed beendet: Exit `10`, Wall `1.14 s`,
User `0.98 s`, System `0.16 s`, Peak-RSS `369,573,888 B`. Der gespeicherte Befund lautet
exakt `invalid/invalid/baseline_fallback` mit `warmup_unstable` nach `16` Warmups. Die
Agent-Statistik zeigt für `all` Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR
`582,260.25 ns`; für `last5` Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`,
Minimum `2,067,916 ns`, Maximum `2,677,583 ns`, Stabilität `false`. Persistiert wurden
`0` Rohsamples, `0` Correctness-Zeilen, `3` Scalars und `1` Artifact. Es gab kein `aa_gpu`
und daraus folgt weder eine Performance- noch eine Correctness-Aussage.

Die DB-Evidenz vor Run20 trägt den übergebenen Hash `c9a521…`; die Run21-DB den Hash
`420b7c…`. Bundle `027908…`, Result `ac4a82…`, Payload `cd409d…` und Evidence
`837841…` sind im Evidenzsatz nur verkürzt übergeben; die Ellipsen werden nicht durch
erfundene Vollhashes ersetzt. Die lokale UI liest die SQLite-Historie automatisch
read-only; die statische Prüfung von `friday_h0/dashboard.py` bestätigt Run-Auflistung und
Statusübernahme einschließlich `invalid`. In diesem Nachweis wurden weder Server noch
Socket gestartet.

Wissenschaftliche Entscheidung: Der eingefrorene Vertrag `8 → maximal 16` Warmups mit
Stabilität der letzten fünf Werte innerhalb `±5 %` entspricht dem Code. Es liegt kein
Implementierungsdefekt vor. Die Ursache der instabilen Messung bleibt als OS-/Thermik-/MLX-
Unsicherheit offen; Schwelle und Daten wurden nicht nachträglich geändert und Run21 wurde
nicht wiederholt. Der `python`-Aliasfehler und der Dashboard-`self.path`-Fehler sind separat
als Harnessfehler klassifiziert, nicht als Projektfehler. Konvergenzregel: Harnessbefunde
werden nur nach reproduzierbarer Wiederholung und unabhängigem Readback bewertet; sie ändern
keine wissenschaftliche Schwelle und ersetzen keinen Canary.

## Begrenzter Head-Skip-Runtime-Prototyp qualifiziert — 24.08.2026

Der Nutzer hat den zuvor in `PERMISSION_REQUIRED.md` blockierten kleinen
Runtime-Prototyp ausdrücklich freigegeben. Der Vertrag
`docs/HEAD_SKIP_RUNTIME_SPEC.md` und die Mini-Vorregistrierung
`experiments/head_skip_runtime/PREREGISTRATION.md` wurden vor jeder neuen
Runtime-Messung bytegenau eingefroren. Die Qualifikation ist Engineering-Evidenz,
kein Zyklus 13 und bleibt `formal_claim=false`.

Das getrennte Paket `friday_head_skip_runtime/` ist offline implementiert. Es
replayt die unveränderte formale 16-Record-Evidenz, prüft Modell, Prompt,
Tokenizergebnis und alle erlaubten Requestwerte und wählt den schnellen Pfad nur
im exakt bestätigten Einzelfall. Jede Unsicherheit wählt den Referenzpfad. Ein
Fehler des schnellen Pfads wird nicht im selben Aufruf wiederholt und verriegelt
den schnellen Pfad für den Prozess. Der MLX-Adapter lädt ausschließlich den lokal
gebundenen Snapshot. Eine eigene private, append-only SQLite-Historie und eine
read-only Oberfläche auf `127.0.0.1:8775` sind enthalten.

Vor dem neuen Code bestand die unveränderte relevante Baseline mit Exit `0` in
`4,44 s` außen, `11,08 s` User, `0,58 s` System, maximal `50.380.800 B` RSS,
Peak-Footprint `34.308.912 B` und ohne Swaps. Nach der Implementierung bestanden
die `20` neuen fokussierten Offline-Tests, die Bytecode-Kompilation und
`git diff --check`. Der absichtlich im schmutzigen Vor-Commit-Stand aufgerufene
Policy-Check replayte alle `16` formalen Records und fiel erwartungsgemäß auf
`worktree_dirty` zurück; es entstand keine Runtime-Datenbank und keine GPU-Arbeit.

Vor dem Live-Lauf wurde die Aktivierung zusätzlich an genau einen bestandenen
CPU-Record, eine unveränderliche Startmarke und genau einen bestandenen GPU-Record
gebunden. Abweichende Lauf-IDs oder Datenbankpfade können keinen zweiten Versuch
öffnen; unbekannter Swap-Verbrauch schließt das Ressourcengate. Die vollständige
Projekttestsammlung bestand mit Exit `0` in `38,50 s`, maximal `192.921.600 B` RSS,
Peak-Footprint `46.007.136 B` und ohne Swaps. Diese Softwaretest-Laufzeit ist kein
Modell-Geschwindigkeitswert.

Die vollständige lokale Quellprüfung ist unter
`/private/tmp/codex-security-scans/Project_Friday/c7db74f_20260824T084223Z`
versiegelt: alle `12` Quellzeilen wurden vollständig geprüft, Abdeckung
`complete`, keine offene Arbeit und `0` berichtspflichtige Sicherheitsbefunde.
Der nachgeschaltete Zwei-Dateien-Delta nach Ergänzung des CPU-BudgetGuards ist unter
`/private/tmp/codex-security-scans/Project_Friday/5b17bfb_20260824T085130Z`
ebenfalls mit vollständiger Abdeckung und `0` Befunden versiegelt.

Das CPU-Gate bestand mit Policy-Median `864,7 ns`, p95 `872,473 ns`, Zusatz
`839,473 ns` und Evidenzload `824,460209 ms`. Danach wurde genau ein, nicht
wiederholter MLX-Prozess gestartet. Baseline und Kandidat erzeugten im
Korrektheitspaar und in allen vier Messpaaren dieselben `32` Token. Der gemessene
Prefill-Median sank von `1806,4618545 ms` auf `1528,206979 ms`; das vorab
festgelegte gepaarte Verhältnis beträgt `0,8458362745`, entsprechend gerechnet
`−15,4164 %`. Peak-Delta `−97.855.968 B`, Swap-Delta `0`, Netzbetrieb und Duty
`0,15` hielten. Alle H1–H5-Gates bestanden; Entscheidung
`engineering_go_exact_scope`, weiterhin `formal_claim=false`.

Die Runtime-Historie enthält genau CPU-Record, Startmarke und GPU-Record, Modus
`0600`, SHA-256
`6dcf6e4cb942b842dca6e9b0b071df8e7c6cb81ba28fdc5e0fdb05c414d20567`,
Kettenkopf `db4c98e892136930cc515be94417a294c960cfaa9a790a6ea05629d0b796b8f3`.
Der normale Policy-Aufruf autorisiert den schnellen Pfad jetzt nur im exakten
registrierten Fall; die reale read-only Loopback-Abfrage lieferte HTTP `200` und
die verifizierte Dreierhistorie. Ungeprüfte Prompts, Modelle, Logprob-Anfragen,
Multi-Turn- und Parallel-Workloads bleiben auf der Baseline.

## Zyklus 16 — Vor-Hardware-Status: runtime-only Matmul-Umgebungs-A/B

Am 24.08.2026 wurde genau eine neue lokale Studie freigegeben:
`matmul-compile-ab-20260824-01` mit dem Kandidaten
`fixed_cache_compiled_decode_v1`. Die Freigabe betrifft ausschließlich die
Laufzeitumgebung. Modellgewichte, Modellarchitektur und Quantisierung werden
nicht geändert.

Die mathematische Matmul-Operation bleibt in allen drei Armen aktiv. Verglichen
werden nur `standard_eager`, `fixed_eager` und `fixed_compiled` — also der
Standardpfad, ein fester KV-Cache ohne Compile und ein fester KV-Cache mit
Compile. Es gibt keinen echten Matmul-Aus-Schalter und keinen Vergleich mit
veränderten Gewichten.

Die Präregistrierung ist im lokalen Seal-Commit eingefroren und noch nicht
gemessen. Exakte greedy Token- und
Textidentität ist ein Pflicht-Gate; ein Unterschied ist ein terminaler
Korrektheitsfehler. Die alten Device-Model-Compile-Messungen sind wegen falscher
Token ab Position 2 ungültig und werden nicht als Baseline oder Gewinn verwendet.
`formal_claim=false`. Es gibt noch keine Zyklus-16-Ergebnisdatei oder
Hardwareevidenz. Ein negatives Ergebnis ist gültig; bis zum Seal und den
vorgesehenen Messungen wird nichts automatisch ausgeführt oder aktiviert.

## Zyklus 16 — final geprüfter Seal-Stand vor Hardware

Der lokale Commit, der diesen final geprüften Stand enthält, ist der Seal-Commit.
Status danach: `sealed_pending_hardware`. Die Präregistrierung der Studie
`matmul-compile-ab-20260824-01` ist unverändert eingefroren; SHA-256
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Es gibt weiterhin keine Messung, keine `results.json` und keine private
Startmarke. `formal_claim=false`.

Vor-Hardware-Fixes aus dem Review: (1) lazy MLX-Fehler im kalten Compiled- und
Fixed-Eager-Pfad werden bis zur Materialisierung als `candidate_not_runnable`
erfasst, außer Ressourcen-/OOM-Fehlern; (2) der Parent begrenzt jeden Worker
auf die verbleibende harte Gesamt-Walltime und bricht sicher ab; (3) beobachtete
Armzeit und akzeptierte/gebuchte GPU-Zeit sind getrennt, einschließlich der
theoretischen Duty-Pause, sodass Budgetablehnungen als Teil-Evidenz erhalten
bleiben. Zusätzlich werden Arm-Längen, Finish-Grund und Messstatistiken streng
geprüft; minimale Worker-Fehler bleiben als bereinigtes Terminalereignis erhalten.

Verifikation: fokussiert `34 passed` (Exit 0), vollständige Pytest-Suite (Exit 0),
Compileall (Exit 0), Worker-Selfcheck 21 (Exit 0), Harness-Selfcheck 18 (Exit 0),
UI-Selfcheck (Exit 0), Standardaufruf ohne Hardware (Exit 78; keine Artefakte),
`git diff --check` (Exit 0), `xcodebuild -checkFirstLaunchStatus` (Exit 0).
ProjectAtlas 0.4.5-rc1 und MCP sind verfügbar. Gerät: Apple M1 Max, 32 GiB,
Netzbetrieb, `Device(gpu,0)`; MLX 0.32.0 und mlx-lm 0.31.3. Snapshot-SHA
`e6edcd46...eda`, Gewicht-SHA `94d3d701...74af3`.

## Zyklus 16 — reales Ergebnis und unabhängige Einordnung

Die Studie `matmul-compile-ab-20260824-01` wurde im Seal-Commit
`83ee3ea03f9fb303b8226ab8ad3189f07daec727` ausgeführt. Evidence-Commit
`cc6d2ea012a0cd6a858acc9a66d4754e95c421b7`; Ergebnis-Hash
`fbcc2fc65ac5d255ed11039a74c34e9a02d942cec17b25a6ed863058e0073b57`,
Verifikations-Hash
`09b1b53841a59bad3c4b1b9a0ef62fb659668b472358c10fa9188cad158f0038`, Marker-Hash
`8adf6f9c2453524bd1e05f4973ee85f84a323e9461a3f9b996ec2d0f7fed3c2f`,
Präregistrierungs-Hash
`dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf`.
Die Nutzerfreigabe ist damit genau einmal verbraucht; es gibt keinen zweiten Lauf.

Gemessen wurden sechs frische Prozesse mit drei Armen, also 18 Arm-Ausführungen
(3 × 6). Token und Text waren in allen 18 Arm-Ausführungen exakt gleich. Die
Decode-Gesamtzeit-Mediane und
Tokenraten waren:

| Arm | Decode-Median | Tokenrate | TTFT-Median |
|---|---:|---:|---:|
| `standard_eager` | 0,399939187 s | 77,5131895/s | 0,638376521 s |
| `fixed_eager` | 0,3999597295 s | 77,5078153/s | 0,638425813 s |
| `fixed_compiled` | 0,371848789 s | 83,3672240/s | 0,6385446665 s |

Der gemessene gepaarte Ratio `fixed_compiled/standard_eager` beträgt
`0,9295921887`, Bootstrap-95-%-KI `[0,9128789083; 0,9348209684]` — rechnerisch
7,0408 % schnellere Decode-Phase. Gegen `fixed_eager` beträgt der Ratio
`0,9296309524`, KI `[0,9256302629; 0,9327708433]`. Peak-RSS war
`3.771.564.032 B`, MLX-Peak `3.476.049.782 B`, Swap-Delta `0 B`.

Berechnet, nicht separat gemessen: warme End-to-End-Projektion `0,9829777045`
(rund 1,7022 % schneller insgesamt), kalte One-off-Projektion `1,0154895491`
(rund 1,549 % langsamer) und Break-even median rund 36,47 Decode-Schritte;
der Lauf umfasste 31 Schritte. Matmul wurde nie abgeschaltet. Modell, Gewichte
und Quantisierung blieben unverändert; geändert wurden nur feste Cacheform und
MLX-Compile-Laufzeitorganisation. Das ist kein allgemeiner Qualitäts-,
Selbstlern- oder Produktivclaim; automatische Aktivierung bleibt verboten.

Der Lifecycle-Selfcheck-Bug wurde erst nach der Messung entdeckt und getrennt
behoben; er wird nicht rückwirkend als Messwert umgedeutet. Die read-only UI
lieferte GET/HEAD `200`, Schreibmethoden `405`, fremden Host `421`; die Hashes
blieben unverändert. `formal_claim=false`.

## Post-Hardware-Verifikation — Zyklus 16 und historische Evidenz

Die abgeschlossene Studie wurde nach dem Lauf unabhängig verifiziert; ein neuer
Hardware- oder Modelllauf fand nicht statt. Die vollständige Suite meldete `787
Tests in 71 Dateien`, Exit 0. Der fokussierte Lauf
`test_matmul_compile_ab` meldete `43 passed, 60 subtests`, Exit 0. Compileall,
Worker-Selfcheck 21, Harness-Selfcheck 18, Dashboard-Selfcheck 0, Xcode-Check,
`jq` und `git diff --check` endeten jeweils mit Exit 0. Der Standardaufruf blieb
bei erwarteter Exit 78 und veränderte keine Artefakte. Der Harness-`--show`-
Aufruf lief einmal mit Exit 0, stderr blieb leer und er lieferte genau eine
gültige JSON-Zeile.

Die reale read-only UI-Verifikation ergab GET/HEAD `200`, alle Schreibmethoden
`405`, einen fremden Host `421` und `no-store`. Unbereinigter Modelltext wurde
nicht ausgeliefert. Die Cycle-16- und Cycle-15-Evidence sowie alle 12 SQLite-
Datenbanken waren vor und nach der Prüfung identisch; die private Startmarke
blieb auf Modus `0600`. Die Ergebnis- und Datenbank-Hashes blieben unverändert.

ProjectAtlas wurde genau einmal inkrementell mit `watch_once` aktualisiert:
`cycles=1`, `indexed=967` Textkandidaten, `parsed=11`, `unchanged=732` Symbole;
Runtime `0.4.5-rc1` und die projektlokale MCP-Konfiguration waren gültig.
Getrackte ProjectAtlas-Dateien blieben unangetastet; das bereits vorhandene
verschachtelte `.gradle`-Untracked blieb ebenfalls unangetastet.

Der nachträglich gefundene Lifecycle-Selfcheck-Fehler lag in der Annahme, die
Evidence müsse fehlen. Die Korrektur prüft nun read-only sowohl fehlende als auch
vorhandene Evidence, verbietet Symlinks, verlangt Modus `0600` für die Marke und
vergleicht Hashes sowie Modi vor und nach der Prüfung. Der Arbeitsbaum-Harness
unterscheidet sich dadurch vom versiegelten Code; die Evidence bewahrt jedoch die
Code-Fingerprints des Seal-Stands. `formal_claim=false`; die Freigabe ist
verbraucht und es wurde nichts automatisch aktiviert.

## Zyklus 17 — historischer Pre-Hardware-Draft

Studie `fixed-compiled-batched-readback-20260824-01`, Kandidat
`fixed_compiled_batched_readback_n8_v1`, Status `draft_pending_preflight`.
Die Nutzerantwort „Dann machen wir das mal“ reserviert genau einen neuen Lauf;
die Freigabe ist reserviert, aber noch nicht verbraucht. Noch kein Marker,
Ergebnis oder Modelllauf.

Einzige Variable ist Readback `1` versus `8` auf identischem Fixed-Compiled-4B-
Pfad; Modell, Gewichte, Quantisierung und Matmul bleiben unverändert. Geplant
sind sechs gepaarte frische Prozesse (zwölf Arm-Ausführungen). EOS-Tail wird
vollständig zeitlich erfasst und getrimmt; exakte logische Token und sichtbarer
Text sind terminales Gate. `formal_claim=false`; keine Aktivierung, kein Dienst,
kein Multi-Turn- und kein Qualitätsclaim. Cycle 7 `12,98 %` bleibt explorativ.

## Zyklus 17 — historischer sealed_pre_hardware-Stand (25.08.2026)

Offline-Preflight abgeschlossen: `measured=false`, `formal_claim=false`,
`authorization=reserved_not_consumed`; kein Modell-/MLX-/GPU-/Hardwarelauf,
kein Marker, kein Resultat, kein Dienst und keine Autoaktivierung. Readback 1
versus 8 bleibt die einzige Variable auf identischem Fixed-Compiled-4B-Pfad;
sechs gepaarte frische Prozesse und zwölf Arm-Ausführungen sind geplant.
Prereg `74f63c36ddd141c4b4666d9f15d7b17d3ac9294e2d63cb29f6d9e35a80db21b1`,
Worker `fecf712b44e6d1a8c46565dda59569fa11cdc762fc49917307874435e4a2efde`,
Harness `9c0689be97a1ee5022f7c4b4623af9bd4a9906411291d9c4295b4c16184c7ff0`,
Protocol `a58b6298a22e676b9213cc0e4b8fc22ecdc7e0adb25eb07a58f663d268164c30`, Dashboard `ccbaf05368f21acfa2c627a33f3ee9c5629d335d45f76abb5a74d1399fbeaaee`.
Selfchecks, 30/30, 817/817 (~195 s), Compileall, 89 JSON, Xcode und Diff
bestanden; ProjectAtlas Runtime `0.4.5-rc1` gut, Session-Brief
`refresh_required` wegen `dependency_closure_limit`. Swap 19359465472 Bytes,
Preflight-Delta 0; Snapshot/Gewicht/Generation-Config- und Hardware-Gates
bestanden. Die Nutzerfreigabe bleibt reserviert und ungenutzt.

## Zyklus 17 — Ergebnis

`measured=true`, Freigabe `consumed_exactly_once`, Entscheidung
`no_clear_speedup_baseline_retained`, `formal_claim=false`. Alle 6 Paare/12
Arme waren korrekt; Readback 8 war in jedem Paar schneller, aber Ratio-Median
`0,9581074518`, KI `[0,9534714914;0,9598849359]` verfehlte die feste 5-%-
Schwelle. Der 4,1893-%-Effekt ist berechnet; Baseline retained. Keine Modell-,
Gewichts-, Quantisierungs- oder Matmuländerung, keine Qualitäts- oder
Aktivierungsaussage. Evidence-Audit `evidence_valid=true`.
Result-SHA `d2eb29fe31dcf47fe294d2b0bec2d724fe6cc6e6f6d88d2035783613338fbddd`,
Verification-SHA `be91320e44507573601b81b90016ff0b9c8bee9458465cebff5ae438e6bf9214`,
Marker-SHA `18801fdfca677eb1b0d6a82ac95954990ac05476876447ca1170a42e754710dc`.

## Zyklen 18–21 — Fused-Greedy-Compile: Fehlerkette und gültiger Abschluss

Zyklen 18 bis 20 waren terminale Vorläufe ohne Performance-Evidence. Alle
blieben `formal_claim=false` und luden kein Modell (`load_count=0`):

| Zyklus | Resultat-SHA256 | Seal-Commit | Evidence-Commit | Ursache |
|---|---|---|---|---|
| 18 | `ea644a912c9bb20a9fc992d7e24bfecfbb70285f2788ee83a15aeb4937503035` | `f1383587d585620f75e3c1e9bd40a71cbd0e8af9` | `dc2cdced58b629e6a39cb8ed870d847d8ee16c13` | Parent/Worker-Environment-Fingerprint verschieden; Provenienzprüfung vor Paar 0. |
| 19 | `4e02221975f6f1710e96dc70f69b4df6f48a1d93df859c6274ed83460dee0320` | `7278bda3281161cebdcf395fd4aa50df5de5124e` | `59bbe9d698d978dcbd621fe89fb17bf98b286b8a` | Worker sah sein eigenes absichtlich erzeugtes Resultat als unerlaubte Git-Änderung. |
| 20 | `72e7e0692136766bcd5cea4147f3c106ad64de8ddadba855767d8908ae53200d` | `6c0e18b17b2febf184fda0fc09552b5613c49dc0` | `78f983c71636637b7995eb90500fe689cbe53fee` | Parent-Manifest enthielt `dev`, Worker-Manifest nicht; Snapshot-Binding vor dem Laden. |

Die Fehler wurden fail-closed erhalten. Korrekturen waren identische
Environment-Fingerprints, eine Allowlist nur für das eigene Resultat und ein
gemeinsamer Snapshot-Statvertrag inklusive `dev`. Es gab keinen Matmul-
On/Off-Nebenclaim.

Zyklus 21 wurde mit Seal `ad4c92f32e608a8a0870b37e23a4dba0da1f666c` und
Evidence `4f89e51c3933aa9c9d42563393589da3c2e4a875` abgeschlossen. Prereg-SHA
`a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf`, Resultat-
SHA `55bad770baad66cbebb804288845e9cf2785c0969c77355731ab8a23b3a43a2e`,
Marker-SHA `1c1dc10670c153c4c7430f3320671c08a3d56114e0fc5ee6af988c750ceb14e4`.

Gemessen wurden sechs frische serielle 4B-Prozesse und zwölf Arme. Token und
Texte waren exakt gleich. Decode-Median: external `0,266399792 s`, fused
`0,266088688 s`. Der gepaarte Ratio-Median ist `1,000510010`, Bootstrap-95-%-
KI `[0,981178182; 1,004700679]`, seed `20260825`, 10.000 Resamples, ohne
Ausreißerentfernung. Die Entscheidung lautet
`fused_greedy_compile_inconclusive`, also kein klarer Gewinn; der erwartete
`--execute`-Exit für diesen Nicht-Gewinn ist `1`.

Ressourcen- und Budget-Gates bestanden; RSS maximal `3.769.974.784 B`,
MLX-Peak `3.524.169.562 B`, Swap-Delta überall `0 B`. Rohzeiten und
Identitäts-Hashes sind gemessen; Mediane, Perzentile, Prozentänderungen und
Konfidenzintervalle sind berechnet. Modell, Gewichte, Quantisierung und
Matmul blieben unverändert. Kein Qualitäts-, Selbstlern-, Produkt- oder
Aktivierungsclaim.

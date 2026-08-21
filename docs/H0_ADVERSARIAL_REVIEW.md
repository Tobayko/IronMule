# H0 — adversariales Abschlussreview

**Stand:** 20.08.2026 · **Entscheidungsstatus:** Offline-Pre-Live-Adapter **GO**;
Live-H0 vom Nutzer freigegeben, erster `eager_baseline`-Canary fail-closed **NO-GO**;
weitere Live-Ausführung **AWAITING USER APPROVAL** für den Launcher-Sicherheitsfix.
Dieses Dokument ist eine Kontroll- und Übergabevorlage, kein Hardware-Ergebnis.

## Kurzurteil

Der Forschungspivot H0 → H1 → H2 und die Implementierung von Phase 1A/H0 sind mit
SQLite v1, read-only Loopback-Dashboard und festem Worker Option A freigegeben. H0
belegt derzeit nur, dass ein geschlossenes Mess-, Kontroll-, Persistenz- und
Fallback-Protokoll offline reproduzierbar vorbereitet ist. Es belegt weder einen
MLX-/GPU-Performancegewinn noch Self-Optimization, Hardware-Generalisation,
Correctness auf realer Hardware, Memory-Sicherheit oder Prozessisolation.

Der externe wissenschaftliche Anlass bleibt die in der kritischen Neubewertung
dokumentierte Prior-Art und Transfer-Risikoanalyse (unter anderem Metal-Sci,
`Gaming Without an Attacker` mit 16/53 = 30 % Transfer-Failures, KernelBench-Verified
und aktive MLX-/Apple-Runtime-Baselines). BaseRT ist dort ausdrücklich nur als
Preprint-/Autorenclaim behandelt: [kritische Neubewertung](KRITISCHE_NEUBEWERTUNG_2026-08-19.md).

## Befunde, Kontrollen und Status

| Angegriffene Behauptung / P0-P1-Risiko | Reproduzierbare Kontrolle | Fixstatus und verbleibende Grenze |
|---|---|---|
| Einzelprozess oder ein einzelner Block darf eine Promotion auslösen | Common Result `measurement_complete` ist neutral; Promotion/Aggregation ist ein separater Schritt. Analyse-`known_win` ist nur eine Offline-Fixture. | **Behoben offline.** Kein H0-Live-Ergebnis darf aus einem Prozess promoted werden. |
| Reihenfolge war nicht wirklich gepaart | A/A verwendet zwei separat erzeugte Eager-Callables, 30 gepaarte Blöcke, registrierte balancierte Reihenfolge und gespeicherte Order-Seed-/Sample-Identität. Fehlende oder unvollständige Paare invalidieren. | **Behoben im Vertrag.** Erst ein MLX-Lauf kann die Kontrolle ausführen. |
| Memory-Werte vermischen Cache, Peak, RSS oder unavailable | Active/peak/cache und Prozess-RSS werden getrennt mit API, Einheit, Wert oder `missing_reason` gespeichert. Das 1-GiB-Limit ist best effort, keine harte Unified-Memory-Grenze. | **Abgegrenzt.** Kein Memory-Safety-Gate ohne tatsächliche Telemetrie; RSS bleibt best effort. |
| A/A ist nur behauptet oder hat unklare Seeds | Exaktes Gate: 3 Charakterisierungs- und 3 Bestätigungsprozesse, je 30 Paare, Session-/Set-Ratio und hierarchischer 10.000er-Bootstrap. A/A-Bootstrap: `0xAA052026`/`0xAA052126`; Gate `tie`, Band/KI/Sessiongrenzen. | **Contractfix abgeschlossen.** `aa_gpu`-Manifeste binden die `AA05`-Seeds; `aggregation_contract_ready=true`. Das historische Offline-Aggregat trug `live_execution_authorized=false`; dies ist nicht mit der späteren Nutzerfreigabe gleichzusetzen. A/A wurde dennoch nicht gestartet. |
| Correctness kann durch Performanceaggregation verdeckt werden | FP64-Oracle aus exakten FP16-Werten, Null-RHS, Vorzeicheninvariante und unabhängige harte Caps; Correctness-Matrix bleibt außerhalb der Performanceaggregation. Falsche Fixture wird nie getimt. | **Behoben offline.** Kein realer Hardware-Correctness-Nachweis vorhanden. |
| Adapter verliert Samples, Missing Values oder RSS-Evidenz | Common Result/Storage behalten Rohsamples, skalare Werte, Correctness-Metriken, Statusereignisse und Missing-Gründe; unavailable wird nicht als `0` kodiert. | **Behoben offline.** Die geschlossene Projektion validiert und persistiert die erlaubten Felder oder invalidiert fail-closed; sie autorisiert keinen Live-Lauf. |
| Test-/Snapshot-Zahlen werden als Forschungsstichprobe gelesen | Inferenz-Einheit ist Workload-/Shape-Familie; Timingblöcke sind Wiederholungen. Drei H0-Prozesscluster sind ein Engineering-Gate, kein Power-Nachweis. Versiegelter Test und nicht-enumerierbare Achsen bleiben H1/H2-Pflicht. | **Methodisch behoben.** Keine Generalisation aus H0. |
| Persistenz ist bei Fehlern teilweise oder Replay nicht identisch | SQLite v1 nutzt Transaktion, Append-only-Status/Rohdaten, Bundle-/Manifest-Hashes und idempotenten Replay; `created_at_unix_ns` ist nicht replaybestimmend. Beschädigte DB-/Trigger-/Hashfälle werden fail-closed geprüft. | **Behoben offline.** Python-`sqlite3` ist nicht vollständig fd-gebunden TOCTOU-frei; diese Grenze bleibt offen dokumentiert. |
| Neutraler Common Result wird als Erfolg missverstanden | Zulässige neutrale Klasse: `completed` + `baseline_fallback` + `measurement_complete` + `error=null`; sie autorisiert keine Aggregation oder Promotion. | **Behoben.** Analyse-/Control-Klassen bleiben geschlossen und modusgebunden. |
| Seed-Familien kollidieren semantisch | A/A ausschließlich `AA05`; Eager-/Compile-Sessions verwenden ihre manifestgebundenen F17A/B10C-Seedfamilien; B005 ist späterer Kandidaten-/H1-Kontext und nicht A/A. | **Behoben im Vertrag.** `aa_gpu`-Manifestbindung und Replayvertrag sind abgeschlossen; `aggregation_contract_ready=true`. `live_execution_authorized=false` bezeichnet nur den historischen Offline-Aggregatstand; der aktuelle Stop folgt aus dem Canary-NO-GO und dem offenen Security-Fix. |
| Auflösung des Pythonpfads zerstört die venv-Semantik | Lexikalischen und aufgelösten `sys.executable`-Pfad vergleichen; fehlende Runtime muss fail-closed persistieren. | **Canary deckte P1 auf.** `.resolve()` wählte den Basisinterpreter; NumPy war im bereinigten Worker nicht sichtbar. Kein Matmul lief. Minimaler Fix **AWAITING USER APPROVAL**; pfadbasiertes `Popen` bleibt nicht vollständig TOCTOU-frei. |

## Finales Adapterreview und Fixkette

Die letzte adversariale Runde band den Projection-Hash an vollständige originale
Evidence, vollständiges Result, Manifest und normalisierte Core-Arrays. Genau ein
deklaratives `normalization_projection_v1`-Artifact trägt den Hash und die Zählwerte.
Der Adapter prüft den exakten Worker-/Benchmarkvertrag und die eindeutige Verlinkung von
`correctness.performance` und `correctness.sign_invariant` zu den neun Pflichtfällen.
Der read-only Storage-Verifier prüft die gespeicherten Child-Rows; positive Timings,
Warmup-/Ratio-Rekonstruktion, `measured_at > 0` sowie Median-/Probe-Bindung sind
fail-closed abgesichert.

Die Adapteraufgabe wurde wegen ausbleibender Statusmeldung zweimal kontrolliert
unterbrochen; der sichtbare Teilstand wurde anschließend gezielt triagiert. Beim ersten
Abschluss sanken die fokussierten Fehlerstände reproduzierbar von `7 → 6 → 5 → 2`.
Ursachen waren unter anderem die Abweichung `probes` versus `probe_raw` und ein
Einrückungsfehler; Schlüsselbindung und Einrückung wurden korrigiert. Alle Befunde sind
geschlossen. Im finalen Offline-Adapter-Scope bestehen keine offenen P0/P1/P2; das
separate Live-Gate ist keine Adapterlücke.

Historischer, anders enumerierter Teststand: Hauptsuite ohne Dashboard `177` bestanden, `3` Windows-Skips,
`12` Subtests, Wall `26.034290 s`, Total U/S `23.373336/1.227233 s`, Self-Peak-RSS
`15,499,264 B`, Child-Peak-RSS `74,186,752 B`. Socketfreies Dashboard: `4/4` plus
`3` Setup-Subbranches, Wall `0.002041 s`, RSS `31,260,672 B`. Der CLI-Lock wurde mit
Exit `78` verifiziert; Runner und Worker wurden dabei nicht importiert.

## Aktuelle Pre-Live-Verifikation und Canary

Der Live-Pfad (`run_mlx`, statisches `mlx-run --execute`) bestand `45/45` in Wall
`4.022908 s`, U/S `3.149974/0.200018 s`, Peak-RSS `42,139,648 B`; keine Self-/Child-
Aufteilung ist belegt. Nach dem Fix auf die reale `get_cache_memory`-API bestand der
Fokusscope `16/16` in Wall `0.086906 s`, U/S `0.140900/0.054489 s`, Peak-RSS
`49,938,432 B`, ebenfalls ohne belegte Aufteilung. Ohne `--execute` bleibt Exit `78`
vor Runner-/Worker-/Benchmark-/MLX-Import.

Die aktuelle Nicht-Live-Suite bestand `133/133` (Wall `23.720160 s`, U/S
`22.722187/0.559409 s`, Self-/Child-RSS `71,368,704/23,642,112 B`); der unabhängige
Replay bestand `133/133` (Wall `23.588426 s`, U/S `22.769535/0.504137 s`, Self-/
Child-RSS `60,342,272/23,707,648 B`). Ein echter Importguard belegte null MLX-Matmul-/
GPU-Arbeit. Der zuerst verwendete MetaPath-Guard verwechselte Metadaten-Probes mit
Imports; ein Guard auf echte Importaufrufe löste den False Positive. Socketfrei liefen
`4/4` plus `3` Setup-Subtests (Wall `0.001793 s`, U/S `0.001437/0.000137 s`, Self-/
Child-RSS `31,457,280/0 B`). Sandbox-HTTP-Bindefehler sind kein Produktresultat; die
letzte autorisierte vollständige HTTP-Evidenz bleibt `13/13`.

Der Sandbox-Preflight hatte kein Metal. Der autorisierte Zielgeräte-Smoke bestätigte
MLX `0.32.0` mit einer 1-Element-Operation, aber keine Matmul. Eine historische Spec-
Assertion gehörte zu einer früheren Provenienzgeneration; gelöst wurde dies durch
Generationstrennung, nicht durch Umschreiben alter Evidenz.

Der Nutzer gab den angekündigten Live-H0-Lauf frei. Sol begrenzte ihn danach auf einen
`eager_baseline`-Canary und stoppte bei dessen NO-GO vor `aa_gpu`. Äußerer Wall
`0.166578416 s`; Child U/S `0.106607/0.040468 s`, Child-RSS `28,442,624 B`, gespeicherter
Worker-RSS `23,150,592 B`; äußere Self-U/S/RSS nicht separat gemessen. Das Ergebnis
`invalid/runtime_unavailable/baseline_fallback` enthält `0` Rohsamples, `0` Correctness-
Zeilen, `3` Supervisor-Scalars und `1` Projection-Artifact. Performance, Ratio/KI,
Warmup/Repetitions, Correctness und MLX-Memory fehlen; es gab keine Promotion.

Die DB enthält jetzt `16` Runs. Zwei stabile read-only Snapshots meldeten
`snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
`source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
`run_count=16`, `returned_count=16`, `truncated=false`, `query_only=1`.

## Offline-Evidenz ohne Hardwarebehauptung

Die folgenden Werte sind historische Zwischenstände, die durch den definitiven
Teststand oben ergänzt, nicht als Regression mit ihm verglichen werden. Der konsolidierte
erste Lauf meldete `129 collected`, `106 pass`, `7` veraltete
Seed-Fixture-Abweichungen und `16` Sandbox-Socketfehler. Nach dem Fixturefix in
`tests/test_protocol.py` liefen Protocol/Worker/Supervisor mit `24/24` (Wall
`0.823154 s`, Self U/S `0.257013/0.080812 s`, RSS `46,301,184 B`). Die gesamte Suite
ohne Dashboard lief mit `113/113` (Wall `20.778518 s`, Self U/S
`19.526546/0.333315 s`, Self-RSS `64,405,504 B`, Child-RSS `23,609,344 B`). Diese
Zählungen stammen aus unterschiedlichen Scopes und Umgebungen und sind nicht als
Regression gegeneinander zu lesen.

Die Offline-Historie initialisierte die DB mit Exit 0 in `0.106884 s`. Fünf Runs und
ein idempotenter Replay wurden persistiert: slow `0.195217 s`, Exit 10,
`regression`; known-win `0.164255 s`, Exit 0, analytisch `promoted`; wrong
`0.151175 s`, Exit 10, `correctness`; missing `0.158439 s`, Exit 10,
`missing`; exit70 `0.141642 s`, Exit 10, `worker_exit`; Replay `0.153464 s`,
`idempotent`. Die Sequenz dauerte `0.964355 s`, Self-RSS `15,859,712 B`, Child-RSS
`28,524,544 B`. Die DB war `118,784 B`, Modus `0600`, UID `501`, Elternverzeichnis
`0700`, `application_id=1179797552`, `user_version=1`, fünf Runs, je genau ein
Common Result, keine Rohsamples, nicht abgeschnittene Snapshot-Ansicht und stabile
Revision-/Identitätshashes. Diese Werte sind Offline- und Analyse-Evidenz, keine
MLX-/GPU-Messung.

Die dritte und finale Offline-Control-Generation ergab:

| Lauf | Wall | Exit | Ergebnis |
|---|---:|---:|---|
| slow | `0.191745 s` | `10` | `regression` |
| known | `0.156192 s` | `0` | nur synthetisch |
| wrong | `0.157021 s` | `10` | `correctness` |
| missing | `0.157268 s` | `10` | `missing` |
| exit70 | `0.145334 s` | `10` | `worker_exit` |
| replay | `0.159542 s` | — | `idempotent` |

Gesamt: `0.967681 s`, Self-Peak-RSS `16,334,848 B`, Child-Peak-RSS
`28,819,456 B`, Provenance `5745e93f…39d57`, Replay-Bundle
`6ae4a453…b7335`. Die finale DB enthält `15` Runs (`3 × 5`), jeden mit genau einem
verifizierten `common_result`; die älteren `10` blieben unverändert. Snapshot:
`source_revision=3b70324f…ab658d`, `id=512934c9…b5b52`, `run_count=15`, nicht
abgeschnitten; DB `229,376 B`, Datei `0600`, Verzeichnis `0700`, `query_only=1`.
Auch das sind ausschließlich Offline-Controls und keine H0-Hardwarewerte.

Vorherige autorisierte Dashboard-HTTP-Verifikation lief mit `13/13`, vor der finalen
Finite-/Cleanup-Härtung. Danach liefen `3/3` reine Finite-/Cleanup-Units; der letzte
16er-HTTP-Scope war wegen Sandbox-/Usage-Limit nicht wiederholbar. Es gibt keinen
behaupteten grünen finalen HTTP-Gesamtlauf.

`xcodebuild -checkFirstLaunchStatus` endete mit Exit 0. ProjectAtlas ist wegen des
dokumentierten Nutzungslimits derzeit nicht erneut refreshbar; der letzte bekannte
verschachtelte ProjectAtlas-Gitstatus war sauber. Der Project-Friday-Root ist kein
Git-Worktree.

## Stop-Gates vor dem nächsten Live-H0-Lauf

1. Entscheidung über den minimalen Launcher-Fix einholen: **AWAITING USER APPROVAL**.
   Vorgeschlagen ist der feste absolute, lexikalische venv-Launcher mit engen Owner-/
   Mode-/Typ-/Device-/Inode-Prüfungen vor/nach Spawn. Pfadbasiertes `Popen` bleibt
   dennoch nicht vollständig TOCTOU-frei; fd-Bindung wäre eine Architekturentscheidung.
2. Nur feste `mx.matmul`-Eager-/Compile-Vergleichsarme, kein Custom Metal, kein Modell.
   Zuerst den `eager_baseline`-Canary wiederholen; A/A erst nach vollständigem Gate.
3. Erst bei vollständiger Correctness-, Memory-, Safety-, Timeout-, Missing-Data-
   und A/A-Evidenz den H0-Befund als Messsystemergebnis archivieren. Ein negativer
   oder nicht reproduzierbarer Effekt ist gültig und beendet jede Optimierungsbehauptung.
4. Nach dem A/A-Pilot H1-Powerplanung vor jeder Kandidatensichtung einfrieren. H2
   darf kein bereits geöffnetes H1-Testset verwenden; ein Modellantrag friert
   Erfolgsgrenze und Kosten vorher ein.

## Ehrliche Systemgrenzen

RSS und Memory-Limit sind best effort. Es gibt keine harte Garantie gegen Unified-
Memory-Erschöpfung, Netzwerk-/Dateisystemangriffe, Treiberhänger oder Parent-Tod.
Python-`sqlite3` erlaubt keine vollständig fd-gebundene TOCTOU-Garantie. Die Hash-
Domain-Separation wurde bewusst nicht still verändert. Keine Aussage dieses Reviews
setzt Modelle, Downloads, Installationen oder Custom-Metal-Kernels voraus. Qwen 3.8
27B war nur die Nutzerpräferenz für einen späteren Modelltest und wurde weder verwendet
noch heruntergeladen.

## Finaler Offline-Contract und Run21-NO-GO — 20.08.2026

Der finale Contract-Stand ist offline mit Core `175/0`, Dashboard `4/4` und `0` MLX-
Imports belegt. Provenienz: `575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`;
Code `aae3245e…` (nur als übergebenes Präfix), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 wurde exakt einmal ausgeführt und mit Exit `10` fail-closed beendet: Wall `1.14 s`,
User/System `0.98/0.16 s`, Peak-RSS `369,573,888 B`. Der Persistenzbefund ist
`invalid/invalid/baseline_fallback`, Diagnose `warmup_unstable` nach `16` Warmups.
Agent-Statistik: `all` Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR `582,260.25 ns`;
`last5` Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`, Min/Max
`2,067,916/2,677,583 ns`, Stabilität `false`. Rohsamples `0`, Correctness-Zeilen `0`,
Scalars `3`, Artifact `1`. Kein `aa_gpu`; keine Performance- oder Correctness-Aussage.

Die Regel ist unverändert und codekonform: Warmup `8 → maximal 16`, Stabilität der letzten
fünf Werte innerhalb `±5 %`. Die Messung weist keinen Implementierungsdefekt nach; OS-,
Thermik- oder MLX-Ursache bleibt unbekannt. Es gab keine post-hoc-Schwellenänderung und
keinen Retry. DB vor Run20 `c9a521…`, Run21-DB `420b7c…`, Bundle `027908…`, Result
`ac4a82…`, Payload `cd409d…`, Evidence `837841…`; diese Hashes sind nur in der übergebenen
Kurzform verfügbar.

Der `python`-Alias und der Dashboard-`self.path`-Fehler sind Harnessfehler und keine
Projektbefunde. Die Konvergenzregel trennt sie vom wissenschaftlichen Ergebnis: erst
reproduzieren und unabhängig read-backen, dann bewerten; keine Schwelle wird nachträglich
an einen instabilen Einzel-Canary angepasst. Der statische Dashboard-Check bestätigt,
dass die read-only SQLite-Historie Runs automatisch auflistet und `_status` auch `invalid`
sichtbar macht. Es wurden dabei kein Dashboard-Server und keine Sockets ausgeführt.

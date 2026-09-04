# Phase 1A — Architekturfreigabe (approved; implemented; Live-Canary gestoppt)

**Status:** approved/implemented-offline; Live-Pfad implementiert, Canary fail-closed
gestoppt; Stand 20.08.2026.

Dieses Dokument beschreibt den freigegebenen Offline-Unterbau für den H0-
Messsystem-Preflight. Implementiert und offline verifizierbar sind SQLite v1, der feste
Worker Option A, das read-only-Loopback-Dashboard und die feste Datenbank
`.friday-data/h0.sqlite3`. Der Live-Pfad ist implementiert; der erste freigegebene
Canary endete vor NumPy-/MLX-Benchmarksetup. Eine echte Matmul-Hardwaremessung ist daher
weiterhin nicht erfolgt.

Der freigegebene Stand autorisiert keine Downloads, Installationen, Custom-Metal-Kernels
oder Modellgewichte. `run_mlx` und `mlx-run --execute` sind implementiert; ohne
`--execute` bleibt `EXIT_MLX_LOCKED=78`/`state=not_released` vor Runner-, Worker-,
Benchmark- und MLX-Import aktiv.

## Finaler Offline-Pre-Live-Gate

**Entscheid:** Der Offline-Pre-Live-Adapter ist **GO**. Der Nutzer gab den angekündigten
lokalen H0-MLX-Lauf frei; Sol begrenzte ihn danach auf einen `eager_baseline`-Canary
und stoppte nach dessen **NO-GO** vor `aa_gpu`. Weitere Live-Ausführung ist bis zur
Launcher-Sicherheitsentscheidung gesperrt: **AWAITING USER APPROVAL**. Ohne `--execute`
endet der CLI-Lock weiter mit Exit `78`, ohne Runner, Worker, Benchmark oder MLX zu
importieren.

Das finale Adapterreview schloss die vollständige Evidence-Bindung über Projection-Hash
und deklaratives Artifact, den exakten Adaptervertrag und die Correctness-Verlinkung,
den read-only Storage-Verifier einschließlich Child-Rows sowie positive Timingwerte,
Warmup-/Ratio-Rekonstruktion, `measured_at > 0` und Median-/Probe-Bindung. Im
abgegrenzten Offline-Adapter-Scope bestehen keine offenen P0/P1/P2.

Aktuelle Evidenz: Live-Pfad `45/45` (Wall `4.022908 s`, U/S
`3.149974/0.200018 s`, Peak-RSS `42,139,648 B`, keine belegte Self-/Child-Aufteilung),
Cache-API-Fix `16/16` (Wall `0.086906 s`, U/S `0.140900/0.054489 s`, Peak-RSS
`49,938,432 B`, ebenfalls ohne belegte Aufteilung), Nicht-Live-Suite `133/133` (Wall
`23.720160 s`, U/S `22.722187/0.559409 s`, Self-/Child-RSS
`71,368,704/23,642,112 B`) und unabhängiger Replay `133/133` (Wall `23.588426 s`, U/S
`22.769535/0.504137 s`, Self-/Child-RSS `60,342,272/23,707,648 B`). Socketfrei:
`4/4` plus `3` Setup-Subtests, Wall `0.001793 s`, U/S `0.001437/0.000137 s`, Self-/
Child-RSS `31,457,280/0 B`.

Die historische, anders enumerierte Hauptsuite ohne Dashboard bestand `177` Tests bei `3` Windows-Skips und `12`
Subtests (Wall `26.034290 s`, Total U/S `23.373336/1.227233 s`, Self-/Child-Peak-RSS
`15,499,264/74,186,752 B`). Das socketfreie Dashboard bestand `4/4` plus `3` Setup-
Subbranches (Wall `0.002041 s`, RSS `31,260,672 B`). Die letzte vollständige autorisierte
HTTP-Evidenz bleibt `13/13` vor der abschließenden Finite-/Cleanup-Härtung; der spätere
`16`-er HTTP-Scope wurde wegen Sandbox-/Usage-Limit nicht final wiederholt.

Die feste DB enthält `15` synthetische Offline-Control-Runs und einen fail-closed
Canary, jeweils mit genau einem verifizierten `common_result`. Der Canary enthält keine
Rohsamples oder Correctness-Zeilen und keine Performance-, Memory- oder A/A-Evidenz.

## Live-Canary, Ursache und Restgrenze

Der autorisierte Zielgeräte-Smoke bestätigte MLX `0.32.0` mit einer 1-Element-Operation;
er war keine Matmul. Der Canary lief außen `0.166578416 s`; Child U/S
`0.106607/0.040468 s`, Child-Peak-RSS `28,442,624 B`, gespeicherter Worker-RSS
`23,150,592 B`. Äußere Self-U/S/RSS wurden nicht separat gemessen. Ergebnis:
`invalid/runtime_unavailable/baseline_fallback`, weil `Path(sys.executable).resolve()`
den repo-lokalen venv-Launcher auf den Basisinterpreter auflöste und NumPy in der
bereinigten Worker-Umgebung nicht sichtbar war. Es lief weder MLX-Matmul noch A/A.

Vorgeschlagen ist, den fest erwarteten absoluten, lexikalischen `.venv/bin/python`-
Launcher an `Popen` zu übergeben und Launcher, Parent und Ziel eng vor/nach Spawn zu
prüfen. Pfadbasiertes `Popen` bleibt dennoch nicht vollständig TOCTOU-frei; fd-Bindung
würde Helper/`fexecve` und eine neue Architekturentscheidung erfordern. Daher:
**AWAITING USER APPROVAL**.

Zwei stabile read-only Dashboard-Snapshots nach dem Canary meldeten
`snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
`source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
`run_count=16`, `returned_count=16`, `truncated=false`, `query_only=1`.

## Sol-Empfehlung: ein zusammenhängendes Phase-1A-Paket

### 1. Messdatenspeicher

Verwendet wird SQLite aus der vorhandenen Python-Standardbibliothek (`sqlite3`) mit einer
versionierten Migration `v1`. Schreibvorgänge werden transaktional ausgeführt. Rohsamples,
Manifest, Umgebungs-/Code-Hashes, Status, Fehlerursache, Messwerte und Testergebnisse werden
vollständig gespeichert; aggregierte Werte sind daraus reproduzierbar. Das Schema muss eine
append-only Messhistorie ermöglichen, wobei Migrationen explizit versioniert und im Journal
dokumentiert werden.

JSONL ist für den produktiven H0-Pfad nicht ausgewählt. Die SQLite-Historie bleibt
append-only; Aggregate müssen aus den gespeicherten Rohdaten rekonstruierbar sein.

### 2. Lokales Historien-Dashboard

Das Dashboard ist read-only gegenüber den Messdaten und verwendet ausschließlich Python-
Standardbibliothek. Es bindet ausschließlich an `127.0.0.1`, lädt keine externen Assets und
führt keine Netzwerk- oder Schreiboperationen aus. Ergebnisse, Baseline/Kandidat, Status,
Zeitstempel, Shapes, Precision, Warmup-/Wiederholungsparameter, Median/Streuung sowie
Fehler-/Fallbackgründe werden mit einer kleinen Historie übersichtlich angezeigt.

HTTP-Anfragen, Antwortgröße, Datensätze pro Abfrage und Laufzeit erhalten feste Grenzen.
Fehlerhafte oder unbekannte Parameter werden abgewiesen; es gibt keine beliebigen Datei-
oder Pfadparameter. Das Dashboard ist nur eine lokale Beobachtungsoberfläche und darf keine
Sicherheitsgrenzen des Workers ersetzen.

### 3. Phase-1A-Worker, Option A

Der Worker wird als fester `python -m`-Entrypoint gestartet. Vor dem Start wird ein
geschlossenes Manifest doppelt validiert: gegen ein festes Schema und unmittelbar vor der
Ausführung gegen die erlaubte Laufzeitkonfiguration. Unbekannte Felder, Pfade, Quellen,
Flags und Module werden abgewiesen. Es gibt keine freie Code- oder Kommandozeileninjektion.

Der Parent startet den Worker mit `start_new_session=True`. Ein monotonic Watchdog begrenzt
die Laufzeit; bei Timeout wird die Prozessgruppe mit `killpg` beendet. Das Warten ist
begrenzt. `stdout` und `stderr` werden während der Ausführung aktiv drainiert und durch ein
festes Bytebudget begrenzt; auch das Ergebnisobjekt ist größenbegrenzt. Der Worker führt nur
die eingefrorene `mx.matmul`-/`mx.compile`-Konfiguration der Phase-1A-Spezifikation aus.
Bei Fehler, Timeout, ungültigem Ergebnis oder nicht bestandener Correctness wird sicher auf
die Baseline zurückgefallen. Jeder Fallback wird mit Ursache und vollständigem Kontext
protokolliert.

Harte, im Prozessvertrag durchsetzbare Eigenschaften:

- geschlossenes, doppelt validiertes Manifest ohne unbekannte Felder/Pfade/Source/Flags/Module;
- fester `python -m`-Entrypoint und feste erlaubte Operationen;
- neue Prozesssession, monotonic Timeout, Prozessgruppenabbruch per `killpg` und bounded wait;
- aktiv drainendes `stdout`/`stderr` mit Bytebudget und begrenztes Ergebnis;
- Correctness-Gate, Fehlerstatus und Baselinefallback;
- feste `mx.matmul`-/`mx.compile`-Ausführung gemäß vorregistrierter Spezifikation.

Best effort, nicht als harte Sicherheitsgrenze zu behaupten:

- `mx.metal.set_memory_limit(1GiB)` als MLX-Hinweis/Best-Effort-Limit;
- Parent-RSS-Ziel von 2 GiB mit Polling;
- bereinigte Environment-Variablen und Arbeitsverzeichnis sowie `close_fds`.

Nicht behaupten oder aus diesen Maßnahmen ableiten:

- Netzwerk- oder Filesystemisolation;
- eine harte Grenze des Apple-Unified-Memory-Systems;
- Schutz vor GPU- oder Driver-Hang;
- eine garantierte Parent-Death-/Supervisor-Garantie.

Der Worker-Review bestätigt für das vorhandene Python 3.12.13 auf Darwin, dass `setsid` und
`killpg` verfügbar sind. `RLIMIT_AS` und `RLIMIT_RSS` sind auf diesem Zielsystem beide
Resource-ID 5; RSS ist daher nur Präferenz/Beobachtung und keine belastbare harte Grenze.
`ru_maxrss` ist erst nach dem Prozessende verfügbar. Die MLX-Speicherbegrenzung ist nicht als
harte Unified-Memory-Grenze belegt.

### 4. Phase 1B bleibt ausdrücklich gesperrt

Custom Metal ist in Phase 1A nicht freigegeben. Für Phase 1B wäre höchstens Option B — ein
signierter App-Sandbox-Helper — zu prüfen. Entitlements, Deployment-Ziel, Signierung und
jede Installation eines Helpers sind eine separate Sicherheits-/Architekturentscheidung
und benötigen jeweils ausdrückliche Nutzerfreigabe. Phase 1B wird durch diese Dokumentation
weder implementiert noch vorbereitet.

## Stop-Kriterien aus dem Worker-Review

Die Ausführung ist vor jedem Messlauf abzubrechen, wenn Manifest- oder Entrypoint-Validierung
fehlschlägt, ein unbekannter Parameter/Modul/Pfad erkannt wird, Byte- oder Zeitbudget
überschritten wird, Prozessgruppenabbruch nicht sicher bestätigt werden kann, das Ergebnis
unbegrenzt oder nicht parsebar ist, Correctness oder unabhängige FP64-Hard-Caps fehlschlagen oder die
MLX-/Hardware-Voraussetzungen nicht reproduzierbar nachgewiesen sind. Ebenso stoppt Phase 1A,
wenn ein vermeintliches Limit als harte Isolation oder harte Unified-Memory-Grenze ausgegeben
werden müsste. Ein negativer oder nicht reproduzierbarer Effekt bleibt ein gültiges Ergebnis;
es gibt dann keinen Optimierungsanspruch.

## Explizite Nutzerfreigabe

Am 19.08.2026 wurde exakt freigegeben:

`JA — Ich gebe den Forschungspivot H0 → H1 → H2 und die Implementierung von Phase 1A/H0 mit SQLite v1, read-only Loopback-Dashboard und festem Worker Option A frei. Keine Downloads, Installationen, Custom-Metal-Kernels oder Modellgewichte.`

Die Freigabe autorisiert keine Phase 1B, keinen Custom-Metal-Code, keinen signierten Helper
und keine Entitlement-/Deployment-Änderung. Sie autorisiert außerdem keine reale
Performance-, Correctness-, Memory- oder Safety-Aussage ohne separat angekündigten und
bestandenem H0-Go/No-Go-Lauf.

Am 20.08.2026 gab der Nutzer den zuvor angekündigten echten lokalen H0-MLX-Lauf frei und
nannte Qwen 3.8 27B als Präferenz für einen späteren echten lokalen KI-Test. Qwen wurde
nicht verwendet oder heruntergeladen. Die Begrenzung auf einen Canary war Sols
nachgelagerte Sicherheits-/Wissenschaftsentscheidung, nicht eine Verengung der Freigabe.

## Finaler Contract-Nachweis und Run21-Canary — 20.08.2026

Der Offline-Contract ist final mit Core `175/0`, Dashboard `4/4` und `0` offline MLX-
Imports belegt. Provenienz `575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`,
Code `aae3245e…` (nur Präfix übergeben), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 lief genau einmal und endete mit Exit `10`, Wall `1.14 s`, User/System `0.98/0.16 s`
und Peak-RSS `369,573,888 B`. Der fail-closed Befund ist
`invalid/invalid/baseline_fallback`, `warmup_unstable` nach `16` Warmups. Für `all` wurden
Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR `582,260.25 ns` gemeldet; für `last5`
Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`, Min/Max
`2,067,916/2,677,583 ns`, Stabilität `false`. Rohsamples `0`, Correctness `0`, Scalars
`3`, Artifact `1`; kein `aa_gpu` und keine Performance-/Correctness-Aussage.

Die vorregistrierte Regel `8 → maximal 16` Warmups und letzte fünf Werte innerhalb `±5 %`
ist codekonform. Es wurde kein Implementierungsdefekt gefunden; die Ursache bleibt
OS-/Thermik-/MLX-unspezifisch. Keine nachträgliche Schwellenänderung und kein Retry. DB
vor Run20 `c9a521…`, Run21-DB `420b7c…`, Bundle `027908…`, Result `ac4a82…`, Payload
`cd409d…`, Evidence `837841…` (jeweils nur übergebene Kurzform).

Die Dashboard-Implementierung liest die SQLite-Historie read-only und übernimmt den
Status einschließlich `invalid`; dies wurde statisch geprüft. Es wurde kein Server und
kein Socket gestartet. Der `python`-Alias und der Dashboard-`self.path`-Fehler sind
Harnessfehler, keine Projektfehler. Die Konvergenzregel verlangt Reproduktion plus
unabhängigen Readback und verbietet post-hoc Threshold-/Retry-Anpassungen.

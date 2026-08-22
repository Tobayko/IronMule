# Implementierungsplan

## Auditierter Planstand — 22.08.2026

Die frühere Abfolge wurde durch den Evidenzaudit enger gefasst. Historische
Dispatch-, Loop-, Modell- und Codegen-Läufe sind explorative
`legacy_summary`-Beobachtungen: Das formale A/A-Gate war nicht geschlossen, die
MDE vor A/B nicht versiegelt und H1/H2-Rohblöcke nicht persistent. Sie dürfen
keinen Phasenfortschritt zu Phase 1B, Cross-Device oder einem breiteren Suchraum
begründen.

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
8. **Begrenzten N10-Runtime-/AVO-lite-Pfad prüfen — offline implementiert,
   Live-Gates noch geschlossen.** Der positive N10-Entscheid erlaubt einen getrennten Prototyp
   mit fester Allowlist, seriellem Fallback, Circuit Breaker, vollständiger
   Provenienz und eigener Baseline-/Nachher-Messung. Die bestehende N8-Runtime
   bleibt unverändert. Paket, Tests, eigene Persistenz/UI und Vorregistrierung
   sind vorhanden; `17` fokussierte Tests sowie die Zwischen-Vollsuite mit
   `525` Tests bestanden. Nach sauberem Commit folgen einmalig
   Cold-Load-/CPU-Overhead- und danach nur bei Erfolg GPU-Gate. Freie
   Codegenerierung und Custom Metal sind nicht autorisiert; Phase 1B bleibt
   **NO-GO**, Cross-Device **NO-CLAIM**, weitere Modellrunden und ein breiterer
   Live-Suchraum bleiben **NO-GO**.
   Der aktuelle Entscheid steht in
   [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md).

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

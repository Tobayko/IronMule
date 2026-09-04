# Experimente

## Aktueller Stand — 20.08.2026

Der H0-Offline-Unterbau, der Pre-Live-Adapter und die Dashboard-Prüfungen sind
implementiert. Die lokale `.friday-data/h0.sqlite3` enthält `15` synthetische
Offline-Control-Runs (`3 × 5`) und einen fail-closed `eager_baseline`-Canary. Er endete
vor NumPy-/MLX-Benchmarksetup mit `runtime_unavailable`, erzeugte keine Rohsamples oder
Correctness-Zeilen und ist keine H0-Hardwaremessung. `experiments/` enthält weiterhin
keine produktiven MLX-/GPU-Rohdaten; `aa_gpu` wurde nicht ausgeführt. Ohne `--execute`
endet `mlx-run` weiter mit Exit `78`. Der nächste Live-Lauf ist bis zur Launcher-
Sicherheitsentscheidung gesperrt: **AWAITING USER APPROVAL**.

Das lokale read-only Historien-Dashboard kann bei Bedarf mit
`./.venv/bin/python -m friday_h0.cli dashboard --port 8765` gestartet werden. Es bindet
nur an `127.0.0.1` und schreibt nicht in die Messdatenbank. Diese Dokumentation behauptet
nicht, dass der Server aktuell läuft. Die letzte vollständige autorisierte HTTP-Evidenz
ist `13/13`; der spätere `16`-er Scope wurde nach weiterer Härtung wegen Sandbox-/Usage-
Limit nicht final wiederholt.

Zwei read-only Snapshots nach dem 16. Run waren stabil:
`snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
`source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
`run_count=16`, `returned_count=16`, `truncated=false`, `query_only=1`. Die lokale UI
zeigt damit die Historie ohne Schreibzugriff; diese Dokumentation behauptet nicht, dass
der Server aktuell läuft. Qwen 3.8 27B wurde weder heruntergeladen noch verwendet.

Jeder Lauf erhält einen eigenen Unterordner oder eine JSONL-Datei. Mindestens speichern:

- Zeitstempel, Git-/Code-Revision und Tool-Versionen;
- Hardware-/OS-/Treiber-Fingerprint;
- Operation, Tensor-Shapes, Batch, Context, Precision und Speicherlage;
- Baseline- und Kandidatenkonfiguration;
- Warmup-Anzahl, Wiederholungsanzahl, Rohzeiten, Median, Varianz/Perzentile;
- Correctness-Ergebnis, Timeout/Crash/Compilerfehler und Entscheidung;
- Energie-/Temperaturdaten nur mit Quelle und Messgrenzen.

Keine zusammengefasste „Verbesserung“ ohne die Rohmesswerte ablegen. Große Rohdaten und lokale
SQLite-Dateien sind per `.gitignore` ausgeschlossen; veröffentlichte, bereinigte Benchmarkdaten
werden später bewusst ausgewählt.

## Finaler Contract- und Run21-Nachtrag — 20.08.2026

Der finale Offline-Contract ist mit Core `175/0`, Dashboard `4/4` und `0` offline MLX-
Imports belegt. Provenienz `575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`;
Code `aae3245e…` (nur Präfix), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 wurde genau einmal ausgeführt: Exit `10`, Wall `1.14 s`, User/System `0.98/0.16 s`,
Peak-RSS `369,573,888 B`; Ergebnis `invalid/invalid/baseline_fallback`, Diagnose
`warmup_unstable` nach `16`. `all`: Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR
`582,260.25 ns`; `last5`: Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`,
Min/Max `2,067,916/2,677,583 ns`, Stabilität `false`. Persistenz: Rohsamples `0`,
Correctness `0`, Scalars `3`, Artifact `1`. Es gab kein `aa_gpu` und keine Performance-
oder Correctness-Aussage.

Der Code entspricht dem eingefrorenen Warmupvertrag `8 → maximal 16` mit `±5 %`-Band
für die letzten fünf Werte. Die Messung ist kein Implementierungsdefekt; OS-/Thermik-/MLX-
Ursache bleibt unbekannt. Keine nachträgliche Schwellenänderung und kein Retry. Die
übergebenen verkürzten Evidenz-Hashes sind DB vor Run20 `c9a521…`, Run21-DB `420b7c…`,
Bundle `027908…`, Result `ac4a82…`, Payload `cd409d…`, Evidence `837841…`.

Die UI liest die SQLite-Historie automatisch read-only; statisch ist die Sichtbarkeit des
`invalid`-Status bestätigt. Für diesen Nachweis wurden weder Server noch Socket gestartet.
`python`-Alias und Dashboard-`self.path` werden als Harnessfehler getrennt vom Projekt-
befund geführt. Konvergenz bedeutet hier reproduzierbare Wiederholung plus unabhängigen
Readback; ein Harnessfehler ändert keine Schwelle und löst keinen post-hoc Retry aus.

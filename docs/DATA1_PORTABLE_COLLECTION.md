# DATA1: portable Datenerhebung für IronMule

Stand: 2026-09-07. Implementiert ist die opt-in Datenerhebung mit MLX-, CUDA-T4-
und JAX-TPU-Adaptern, privater Kaggle-Paketierung, Quotenjournal, verifiziertem
Import, versioniertem Split-Protokoll und Offline-Lernauswertung. Die produktive
Inferenz wird durch keinen dieser Befehle umgestellt.

## Vorrangiges Leistungsziel — Nutzerkorrektur 2026-09-07

Das Hauptziel ist maximale End-to-End-Beschleunigung, mindestens die bisherige
Bestleistung in der jeweils vergleichbaren Workload. Geringerer Messaufwand ist
sekundär; er ersetzt keinen Laufzeitgewinn. Die Kaggle-Gratisgrenzen gelten weiter.

| Vergleichbare Workload | Historische Zielmarke | Einschränkung |
| --- | --- | --- |
| E13: 4B, acht Fragen mit gemeinsamem Dokument | Sessionratio <=0,2039695604; rund 4,90x | Sekundäre Performancekennzahl; 20/352 Antworten divergierten. Aktuelle Exact-/Efficiency-Qualität muss neu nachgewiesen werden. |
| D5: 1B, 897 Prompt-/32 Ausgabetokens | Anfrageratio <=0,6959773071; rund 1,44x | Tokenidentisch beobachtet, explorativ, `formal_claim=false`; neue Qualifikation erforderlich. |
| B39d: 12B, sechs Anfragen à 48 Tokens | Serverwallratio <=0,8194867050; Tokenrate >=1,2202787058 | Präregistriert qualifiziert, aber keine automatische Aktivierung oder Übertragung auf andere Hardware. |

Die Quellen stehen in `research/raw/E13_summary.json`,
`experiments/serve_gain/gain_1b_32_b.json` und
`research/raw/B39d_public_summary_20260828.json`. Gegenübergestellt werden künftig
das native Stock-Backend und die beste kompatible IronMule-Konfiguration mit
denselben Modell-/Quantisierungs-/Workloadbedingungen. TTFT, Tokenrate und
Gesamtzeit bleiben getrennte Kennzahlen. Für NVIDIA/TPU sind dies Zielmarken
passender Workloadklassen, keine bereits erbrachten Hardware-Nachweise.

Die prospektive Lernpolicy v2 wählt auf Validation nach der Laufzeit der wirklich
ausgewählten Variante; Vorhersage-RMSE ist nur ein nachgeordnetes Kriterium.
Der Matmul-Proxy muss das beobachtete Optimum erreichen (`oracle_ratio <=1,0`);
die frühere 5-%-Langsamer-Toleranz entfällt. Das 20-%-Messkostenziel wird nur noch
sekundär ausgewiesen. Auch ein bestandener Proxy setzt `runtime_goal_met` nicht
auf true: Dafür fehlen eigenständige vollständige Inferenzbenchmarks.
Historische Messschwellen, Ergebnisse und verbrauchte Holdouts bleiben unverändert.

## Bedienung

Alle Beispiele laufen im Checkout mit `.venv/bin/python ironmule_cli.py`;
nach Installation entspricht das dem Befehl `ironmule`. Status/Planung benötigen
weder einen Modellimport noch ein geöffnetes GPU-Gerät.

```sh
ironmule data status
ironmule data capture --partition train --case-index 0
ironmule data capture --partition train --case-index 0 --limit 1 --execute
ironmule data plan --capture /absolute/path/capture.json --backend mlx --output /absolute/path/spec.json
ironmule data run --spec /absolute/path/spec.json --data-dir /absolute/path/captured/data
ironmule data run --spec /absolute/path/spec.json --data-dir /absolute/path/captured/data --execute
ironmule data import --spec /absolute/path/spec.json --report /absolute/path/report.json
ironmule data dataset
ironmule data train
ironmule data dashboard --port 8789
```

`--state-dir` steht vor dem Unterbefehl. Default ist `.friday-data/portable`
im aktuellen Arbeitsverzeichnis. `capture` und `run` starten nur mit `--execute`.
`plan` ist standardmäßig ein Smoke; `--mode measure` registriert den gepaarten
Messlauf. Eine geänderte Implementierung braucht eine neue Spec und Versuch-ID.
Zurückgestellte, fehlgeschlagene und abgebrochene Versuche bleiben erhalten.

Für die zwölf vorgesehenen Fälle gibt es je Partition `train`, `validation`,
`holdout` die Indizes 0–3. Ein eigener Prozess je Fall verhindert, dass alle Fälle
über eine gemeinsame Capture-Session zu einer einzigen Leakage-Gruppe werden.
`--limit 4` dient dem Pipeline-Piloten; vier gemeinsam erfasste Fälle sind keine
vier unabhängigen Sessions. Replikationen derselben Gewichte/Eingaben erzeugen
keine neuen unabhängigen Trainingsfälle. Die Studie bleibt bei fehlender Coverage
ausdrücklich `no_learning_claim`.

## Kostenloses Kaggle

Das Werkzeug nutzt die offizielle Kaggle-CLI 2.2.4. Sie liegt hier isoliert unter
`.friday-data/tooling/kaggle`; die MLX-Projektumgebung wurde nicht verändert.
Optionales CPU-Training nutzt scikit-learn (`ironmule[data]`); ohne ausreichend
qualifizierte Daten wird diese Abhängigkeit nicht einmal importiert.

```sh
ironmule data quota --resource gpu --use-codex-kaggle-credentials
ironmule data quota --resource tpu --use-codex-kaggle-credentials
ironmule data run --spec /absolute/path/cuda-spec.json --data-dir /absolute/path/data \
  --owner tobayko --accelerator NvidiaTeslaT4 \
  --account-preflight /absolute/path/account-preflight.json \
  --use-codex-kaggle-credentials --execute
```

Der Credential-Schalter liest ausschließlich den vorhandenen Kaggle-Eintrag für
`https://www.kaggle.com/mcp` und hält den Token im Prozessspeicher. Er kopiert ihn
weder in Ergebnisse noch in Cloud-Artefakte. Er kann den hier falsch im Feld
`bearer_token_env_var` abgelegten Literal-Token nach ausdrücklicher Nutzerfreigabe
nutzen. Die globale Codex-Konfiguration wird dadurch nicht geändert. Ein normal
gesetztes `KAGGLE_API_TOKEN` benötigt diesen Schalter nicht.

`account-preflight.json` enthält tatsächlich geprüfte Kontofakten, keine
automatisch gesetzten Freigaben: `free_account_verified`,
`no_paid_linkage_verified`, `active_session_verified` (keine anderen aktiven Jobs),
`account_checked_at_unix_s`, `session_checked_at_unix_s`, `additional_usage: false`
und `supported_free_skus`. Zeitstempel müssen aktuell sein. Fehlende Browser- oder
Providerbelege sperren den Lauf; die normale Quotenabfrage beweist diese Fakten
nicht. Insbesondere sind freie TPU-SKUs nicht aus einer allgemeinen SKU-Liste
ableitbar.

- Geldbudget 0 EUR; keine GCP-/Colab-Pro-Erweiterung, keine bezahlten APIs.
- Je GPU-/TPU-Resetfenster maximal min(10 % Gratisgesamtquote, 7200 Sekunden).
- Erster Smoke maximal 180 Sekunden, regulär maximal 900 Sekunden; Gast-Arbeit
  endet spätestens nach 60 beziehungsweise 720 Sekunden.
- Eine globale Projektreservierung gleichzeitig. Vor dem Absenden werden
  Laufzeit plus 120 Sekunden Puffer reserviert; 7200 Sekunden Restquote bleiben frei.
- Das append-only Quotenjournal zählt konservativ die vollständige Reservierung,
  auch wenn ein Job schneller endet. Es verrechnet beobachtete Quotenänderungen,
  Rundungsunsicherheit und tatsächliche Orchestrierungszeit.
- Unklare Übertragung, unbestätigtes Ende, Reset über eine offene Sitzung,
  Mehrverbrauch oder fehlgeschlagener Job frieren Folgestarts ein. Kein Auto-Retry.
- Dataset und Notebook sind privat, der Notebook-Internetzugang ist abgeschaltet.
  Ein eigener Slug je Versuch verhindert Verwechslungen mit einer neueren Version.

Die CLI gibt Quoten in auf 0,01 Stunden gerundeten Werten aus. Dieser Vertrag ist
an Version 2.2.4 gebunden; die Rundungsauflösung wird konservativ berücksichtigt.
Kaggle dokumentiert den genauen Startpunkt der Quoten-/Timeout-Uhr nicht. Die
Zulassungskontrolle ist deshalb kein Versprechen sekundengenauer Providerabrechnung.

## Evidenz und Lernen

`friday_evidence.portable` ergänzt die gemeinsame Evidenzbibliothek. Alte L1-/R2-
Verträge und versiegelte Pakete werden nicht umgeschrieben. R2s 352 beobachtete
Ausgänge gehören zu einer einzelnen Apple/Gemma-4B-Zelle und werden nicht in
GEMM-Labels umgedeutet.

`ExperimentSpec` (`ironmule.experiment.v1`) bindet den Run, native Backend-Aktionen,
Input-/Modell-/Codehashes, Partition, FP32-Semantik, Warmup, Paarzahl und Zeitgrenzen.
Ein `TrialRecord` hält echte Geräte-/Frameworkidentität, Rohpaare, exakte Outputs,
separate veränderliche Speicherwerte, Kosten und terminale Fehlerklassen fest.
Der öffentliche Import prüft Metadaten und Hashes und ist idempotent. Trainingslabels
benötigen zusätzlich den exakt passenden lokalen Ausführungsbeleg aus der
kettengeprüften Steuerungshistorie. Fremde selbstdeklarierte Berichte bleiben Diagnose.

Der Within-Backend-Split betrachtet Hardware/Modell als Bedingungen; gemeinsame
Tensorherkunft, einzelne Gewichtstensoren, Prompt-/Shape-Familien und Sessions
bleiben unteilbar. Die Partition steht vor den Ergebnissen fest. Holdout-Ausgaben
werden aus der öffentlichen Dataset-Projektion entfernt; nur aggregierte Coverage
ist sichtbar. Die Auswertung versiegelt jeden Holdout einmalig. Eine geänderte
Policy oder ergänzte Trainingsdaten erlauben keine erneute Verwendung desselben
Holdouts.

Der native A/A-Arm misst vor anderen Kandidaten, ob die vorab festgelegte
5-%-Effektauflösung mit höchstens 2,5 % Unsicherheit erreichbar ist. Bei zu hohem
Rauschen endet der Lauf; eine Paarzahlempfehlung startet keinen neuen Versuch.
Korrektheit bleibt pro Backend byteexakt. Smoke-Daten erzeugen keine Timinglabels.

Ridge und ein kleines GBDT werden nur nach ausreichender unabhängiger Coverage
fitten dürfen. Die Wahl benutzt ausschließlich Validation. Der verborgene Test
vergleicht feste Regeln, Random/Grid und echte GP-basierte BO; gemessene CPU-
Auswahl-/Trainingskosten und vollständige Versuchskosten werden ausgewiesen.
Gates gelten für jede Hardware-/Modellzelle, nicht nur für einen gepoolten Mittelwert.
OOD und unzureichende Coverage führen zu `no_recommendation`.

Ein bestandenes Replay-Gate ist noch kein gemessener Produktvorteil:
`performance_claim` bleibt false, bis eine unabhängige native End-to-End-Prüfung
der Suchpolicy das vorrangige Leistungsziel gegen Stock und den bisherigen
Beststand bestätigt. Das frühere 20-%-Kostenziel ist nach dem Nutzerentscheid
sekundär. Der native LLM-Servingpfad wird nicht aktiviert oder verändert.

## Tatsächlich geprüfter Stand

- Neuer Capture aus dem lokalen eingefrorenen Gemma-1B-Snapshot: `[36,1152,1024]`,
  9,001 Sekunden interne Capture-Wallzeit, 0,089744 Sekunden konservativ erfasste
  Gerätearbeit, 4,010 Sekunden Pflichtpause, 10 Elternprozess-Ressourcenbeobachtungen,
  maximal 0 B Swapdelta, Exitcode 0, Quellhash vor/nachher identisch.
- Erste Vorbereitung stoppte vor Hardwarestart wegen zu offenen Rechten neu
  erstellter Elternverzeichnisse. Erstellung korrigiert; Fehlversuch historisiert.
- Erster Matmul-Smoke wartete vergeblich auf ausreichend niedrige Last und wurde
  vor GPU-Start zurückgestellt. Neue Versuch-ID erst nach real bestätigter Readiness.
- Zweiter Metal-Smoke `bf371897d29b4c34b9aea337580d7355`: echte native Ausführung,
  verifizierter Import, keine Performancebehauptung und keine Timing-Trainingslabels.
- Zwei echte A/A-Läufe mit je fünf Warmup-Paaren und zwölf Messpaaren:
  `4dba2c2d6c1b4dee9c96ab14d3064c55` (`36×1152×1024`) und
  `c77f0649b3644a39af44d86a92782ebd` (`274×1152×6912`). Gemessene Unsicherheit
  6,6802 % beziehungsweise 16,1426 % gegen die vorher fixierten 2,5 %; beide
  `censored`, kein Kandidatenvergleich und keine Trainingslabels. Nicht für
  günstigere Zufallswerte wiederholen. In beiden Läufen maximal 0 B Swapdelta.
- Zweiter echter Capture: `274×1152×6912`, 16,348 Sekunden interne Wallzeit,
  0,095753 Sekunden erfasste Gerätearbeit, 12,019 Sekunden Pflichtpausen, Exit 0.
- Eine parallele Aufgabe meldete später zwei kurze CPU-Kontrollsuiten; deren
  genaue Überlappung ist unbekannt. Das ist kein Ursachenbeweis für das Rauschen
  und erlaubt keine Behauptung sicherer Fremdlastfreiheit. Die negativen Gates
  bleiben unverändert.
- Kaggle-Konto authentifiziert; Telefonverifizierung abgeschlossen. Browser und
  CLI zeigen 30 h GPU und 20 h TPU. Erster code-only-Versuch lief wegen der damals
  fehlenden Telefonfreigabe ohne Accelerator und endete mit `cuda_unavailable`;
  keine Modell-/Tensordaten hochgeladen.
- Korrigierter privater T4-Smoke `7c88ef1c0ddf42a3b442839fcec81fe9`
  terminal bestanden: echte Tesla T4, Compute Capability 7.5, 15.636.037.632 B,
  PyTorch 2.10.0+cu128, CUDA 12.8, TF32 aus und FP32-Matmul gegen CPU-Referenz
  korrekt. Browser bestätigt `GPU T4 x2`, Erfolg und 0 aktive Events. Provider-
  Laufzeitdiagnose 11,822 s; keine Performanceaussage. Das Quotenjournal berechnet
  beide Versuche konservativ mit je 300 s; die Provideranzeige bleibt auf 0,00 h.
- Kaggle-MCP-Werkzeugliste ist erreichbar. Private Notebook-Abfragen antworten mit
  fehlender `kernels.get`-Berechtigung beziehungsweise `Unauthenticated`; Status,
  Output und Hardware wurden deshalb zusätzlich über CLI und Browser verifiziert.
- TPU-Live-Nachweis, zwölf unabhängig erfasste Pilotfälle und ein qualifizierter
  Lernkorpus bleiben offen. Fehlende Kontobelege werden nicht durch Annahmen ersetzt.
- 57 gezielte Kontroll-/Regressionstests bestanden (56 DATA1, ein bestehender
  CLI-Hilfetest). Vorherige Evidenz-/Dataset-/CLI-Baseline: 63 Tests und sieben
  Untertests bestanden. Installiertes Wheel in einer separaten Umgebung ohne MLX:
  `data status` und `data train` funktionieren mit korrektem `no_learning_claim`.

Die lokalen Artefakte liegen in `.friday-data/portable`. Die Ansicht ist unter
`http://127.0.0.1:8789` verfügbar, wenn der Dashboard-Befehl läuft. Sie liest nur;
das Öffnen startet weder Hardware, Cloud-Calls noch Training.

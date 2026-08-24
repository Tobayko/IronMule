# Begrenzter Prefill-Head-Skip-Runtime-Prototyp

Status: **durch Nutzer am 24.08.2026 freigegeben; vor der ersten Runtime-Messung
eingefroren**

Runtime-ID: `head-skip-runtime-20260824-01`

## 1. Freigabe und Zweck

Der Nutzer erteilte am 24.08.2026 ausdrücklich die zuvor angeforderte
Architekturfreigabe für den kleinen, rückrollbaren Runtime-Prototyp. Die Freigabe
deckt Implementierung, Offline-Prüfung und genau die in
`experiments/head_skip_runtime/PREREGISTRATION.md` beschriebene einmalige lokale
Qualifikation ab. Sie deckt keine Installation, keinen Download, keine
Systemänderung, keine allgemeine Produktfreischaltung und keine Erweiterung auf
ungeprüfte Modelle oder Eingaben ab.

Der Prototyp überführt den formal bestätigten Prefill-Head-Skip in einen getrennten
lokalen Ausführungspfad. Er soll zeigen, dass die gemessene Verbesserung mit enger
Evidenz- und Eingabeprüfung sicher auswählbar ist und bei jeder Unsicherheit der
bisherige Pfad verwendet wird.

Zulässig ist höchstens die Aussage:

> Auf dem versiegelten Apple M1 Max, mit dem versiegelten lokalen Gemma-3-4B-
> Snapshot und für den exakt registrierten greedy Festlängen-Workload kann die
> getrennte Runtime den bestätigten Prefill-Head-Skip auswählen, ohne die erzeugten
> Token zu verändern oder den bestätigten Mindestgewinn aufzuheben.

Die Runtime-Qualifikation ist Engineering-Evidenz und erweitert den formalen
Einzelworkload-Claim nicht.

## 2. Unveränderliche formale Grundlage

- Studie: `head-skip-prefill-v1-20260824`;
- Kandidat: `prefill-head-skip-20260824-02`;
- terminale DB: `.friday-data/head-skip-v1.sqlite3`;
- DB-SHA-256:
  `15ee462bbad5a8f757373f093fdf2ccfb8bdd0048c03447c1cb635acd38ec8d9`;
- Kettenkopf:
  `8a568e61f0e087794b1997f273e580c72e7f5abaa1eb8bad7954b303dd38a2d4`;
- genau 16 Records und genau ein `formal_claim=true`;
- Preregistration-Payload:
  `175a7238520d2a01a5c1c24898ff34773eb1b7a1cbbd6324b988d11fe8bc9cc6`;
- Provenienz:
  `66a62e506c16294ab0034efed17b51b37f602eaedaf8ddde8f6b7b473f2a2453`;
- Confirmation-Seal:
  `2571670a87fc5bd536d4ccee40d4c889afa30c37e65110e18f70607fd6caf11e`;
- Decision-SHA-256:
  `99820747b874dfdfa72a2d65abbb1d9644a20cca3bd816d9058f4374aeb7428a`;
- Status `head_skip_gain_confirmed`, Claim
  `prefill_head_skip_is_faster_beyond_mde`, formale Aktion
  `permit_bounded_architecture_review`;
- alle Charakterisierungs-, Validierungs-, Gesamt- und Tokenidentitätsgates
  bestanden;
- gemessenes Verhältnis `0,8463845562`, Gesamt-95-%-KI
  `[0,8431470041; 0,8512844842]`.

Die Runtime prüft diese Datei bytegenau und replayt die gesamte Hashkette read-only.
Eine ähnlich aussehende oder nur teilweise passende Evidenz autorisiert nichts.

## 3. Geschlossener schneller Fall

Der schnellere Pfad ist nur erlaubt, wenn sämtliche tatsächlichen Werte passen:

| Feld | Erlaubter Wert |
| :--- | :--- |
| Modell | `mlx-community/gemma-3-4b-it-4bit` |
| Snapshot-Revision | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |
| Promptinhalt-SHA | `73675a7043bd40e61586757d8252cf1fb69bfb53b8747ff47f1c08d5fb8f69e5` |
| gerenderte Prompt-Token | `897` |
| Prefill-Blockgröße | `256` |
| Batch | `1` |
| Auswahl | greedy, Temperatur `0` |
| Prompt-Logprobs | aus |
| Ausgabe | feste Länge `32` Token, kein vorzeitiger Stopp |

Der Controller leitet Prompt-Hash und Tokenzahl aus dem tatsächlichen Prompt und
Tokenizergebnis ab. Freie Caller-Labels können den schnellen Pfad nicht
autorisieren. Andere Prompts, Längen, Modelle, Samplingarten oder Logprob-Anfragen
laufen über die unveränderte Baseline.

## 4. Ausführung und sicherer Rückfall

Der Referenzpfad berechnet den LM-Head für jeden Prefill-Block. Der schnelle Pfad
führt für frühere Blöcke nur den Modellkörper aus und berechnet den LM-Head genau
einmal für die letzte Position des letzten Blocks. Der Decode nach dem ersten Token
ist in beiden Pfaden identisch.

Die Runtime lädt und prüft ihre formale Grundlage einmal je Controller. Danach gilt:

1. fehlende, veränderte oder unlesbare Evidenz → Baseline;
2. abweichende aktuelle Modell-, Software- oder Hardwareidentität → Baseline;
3. nicht exakt passender Request → Baseline;
4. unbekannter Plan → sichtbarer Fehler;
5. Fehler im schnellen Pfad → aktueller Aufruf wird nicht still wiederholt,
   Circuit Breaker wird verriegelt, spätere Aufrufe verwenden die Baseline.

Der Rückfall ist immer im Ergebnisobjekt sichtbar. Der bestehende `mlx_lm`-Code und
die versiegelte Studie werden nicht verändert.

## 5. Schnittstelle

Das getrennte Paket `friday_head_skip_runtime/` stellt bereit:

- eine evidenzgebundene Policy;
- einen Controller mit gecachter Entscheidung und Circuit Breaker;
- einen MLX-Adapter, der ausschließlich den lokal gebundenen Snapshot lädt;
- einen Referenz- und einen Head-Skip-Prefillpfad;
- das Repository-Werkzeug `tools/run_head_skip_runtime.py` für Policyprüfung,
  kontrollierte Qualifikation, Historien-Snapshot und lokale UI.

Der Prototyp wird nicht in `mlx_lm` gepatcht und überschreibt keinen globalen
Generator. Entfernung des getrennten Aufrufpunkts stellt den vorherigen Zustand
vollständig wieder her.

## 6. Qualifikation

Vor jeder Live-Qualifikation müssen Code, dieser Vertrag, die Mini-Vorregistrierung
und Offline-Tests auf einem sauberen lokalen Commit liegen. Danach ist autorisiert:

1. read-only Evidenzload und CPU-Policy-Overhead;
2. nur bei bestandenem CPU-Gate genau ein MLX/GPU-Qualifikationslauf;
3. kein Retry eines fehlgeschlagenen Hardwarelaufs;
4. Netzbetrieb und `BudgetGuard` mit Duty-Faktor `0,15`;
5. Tokenidentität vor der Timingauswertung;
6. Laufzeitstopp immer vor `charge()`;
7. keine Ausreißerentfernung und keine nachträgliche Schwellenänderung.

Der genaue Mess- und Entscheidungsvertrag steht in der Mini-Vorregistrierung.

## 7. Historie und lokale Oberfläche

Engineering-Ergebnisse werden getrennt in
`.friday-data/head-skip-runtime.sqlite3` gespeichert: privater Modus `0600`, eigenes
SQLite-v1-Schema, append-only Trigger, kanonisches JSON und vollständige Hashkette.
Jeder Record trägt `formal_claim=false` und bindet die formale Decision-SHA.

Die UI bindet ausschließlich an `127.0.0.1:8775`, öffnet die Datenbank read-only und
zeigt Status, Kennzahlen, Historie und Revision. Sie besitzt keinen
Mutationsendpunkt.

## 8. Nicht freigegeben

- andere Prompts oder Promptlängen im schnellen Pfad;
- Prompt-Logprobs, Perplexität oder Bewertung;
- zufälliges Sampling oder variable Stoppregeln;
- andere Modelle, Quantisierungen, Geräte oder Snapshot-Revisionen;
- Patchen installierter Pakete;
- automatische produktive Aktivierung;
- ein neuer Optimierungskandidat oder eine Erweiterung des formalen Claims.

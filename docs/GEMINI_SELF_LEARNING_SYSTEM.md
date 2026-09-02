# Gemini Self-Learning System & Error Memory — Project Friday

**Erstellt:** 2026-09-02  
**Geltungsbereich:** Eigene Wissensbasis & Fehlergedächtnis für alle künftigen Zyklen von Gemini/Sol.  
**Zweck:** Dauerhafte Protokollierung aller gemessenen Negativbefunde, widerlegten Hypothesen, methodischen Fehler und Sackgassen. Verhindert das Wiederholen von Fehlern, sofern Parameter identisch sind.

---

## 1. Das Kernprinzip des Systems

> **Regel:** Kein Versuch wird wiederholt, dessen Parameterraum bereits abgedeckt ist. Ein Pfad wird nur dann wiedereröffnet, wenn sich nachweislich mindestens ein tragender Parameter (Workload, Modell, Baseline-Arm, Hardwarebindung) unterscheidet.

Jeder Eintrag enthält:
1. **Was versucht wurde** (Hypothese & Versuchsaufbau)
2. **Was gemessen wurde** (Zahlen, Konfidenzintervalle, Hashes)
3. **Ursache des Scheiterns** (Mechanismus, warum es nicht klappte)
4. **Getestete vs. ungetestete Parameter** (Geltungsbereich & Grenzen)
5. **Dauerhafte Handlungsregel** (Verbot / Leitlinie für künftige Entscheidungen)

---

## 2. Katalog der gemessenen Sackgassen & widerlegten Thesen

### E01 — Prompt-Lookup-Spekulation im Auslieferungspfad (H1.0 / S3 / S4)
- **Versuch:** Prompt-Lookup-Spekulation (`friday_serve/speculation.py`, IronMule `_decode_speculative`) als Beschleuniger gegen den Auslieferungspfad für 4B und 1B.
- **Messung:**
  - 4B (`897` Prompttoken, `32`–`128` generierte Token, Breiten `1`–`3`): Verlust in **jeder** Zelle (`−12,23 %` bis `−75,35 %`), monoton wachsend mit Länge und Breite (`experiments/switch_point/`).
  - 1B (`897`/`32`/Breite 1): `−15,59 %`, KI `[1,1381; 1,2353]`, A/A `11,77 %`.
- **Wahre Ursache:**
  - `acceptance = 0,0` für alle $k \in \{1, 2, 3\}$ (`experiments/identity_break/acceptance.json`).
  - Der versiegelte Prompt besteht aus einer 40-fach wiederholten Instruktionsfloskel, die Antwort (Erklärung zu False Sharing) zitiert diese Trigramme nicht.
  - Spekulation zahlt linearen Zusatzaufwand (~0,8 s je Entwurfsplatz) für null angenommene Token.
- **Widerlegte Thesen:**
  - *These 1 (Barriere):* Verdacht auf überflüssiges `mx.eval` / `mx.synchronize` je Iteration. **Widerlegt:** Patch bewegte nur 0,30–0,40 Prozentpunkte (`experiments/spec_path/`).
  - *These 2 (Bandit/Thompson Sampling):* Breiten 1–3 unterscheiden sich nur um 0,016 (im Rauschen); auf der Workload gibt es nichts selektiv abzuschalten.
- **Parameter-Status:**
  - *Geprüft:* Versiegelter Auslieferungsprompt (`897` Token), Längen `32`–`128`, Breiten `1`–`4`, Baseline = gebündelter Readback.
  - *Ungetestet:* Workloads mit hoher Trigramm-Wiederholung zwischen Prompt und generiertem Text (z. B. `journal.txt`, wo S1 `3`–`6 %` Gewinn maß).
- **Regel:**
  - Im Auslieferungspfad bleibt `speculate_k = 0`.
  - Der `(K,N)`-Dispatcher wird nicht gebaut.
  - Thompson-Sampling über Breiten entfällt.

---

### E02 — Token-Identität bei Spekulation „per Konstruktion“ (S3)
- **Versuch:** Annahme, `_decode_speculative` sei tokenidentisch per Konstruktion, da nur verifizierte Entwürfe übernommen werden.
- **Messung:** Bruch bei 4B / `128` Token / Breite 2 sofort in Paar 0, Index 10 (`experiments/identity_break/`).
- **Wahre Ursache:**
  - Numerik / bf16-Auflösung (ULP).
  - An Index 10 waren Top-1 und Top-2 Logits `75,0` und `74,5`. Im Bereich $[64, 128)$ ist das bf16-Raster exakt $0,5$ (siehe `logit_gap.json`).
  - Ein Forward der Breite 3 nutzt andere Metal-Kernel/Reduktionspfade als Breite 1. Das kippt eine 1-ULP-Entscheidung regulär.
- **Parameter-Status:**
  - *Geprüft:* Greedy-Inferenz in bf16 bei knappen Logits (1 ULP Differenz).
  - *Ungetestet:* fp32-Logits / getrennte FP32-Softmax-Reduktion (widerspricht aber aktuellem Runtime-Setup).
- **Regel:**
  - Spekulation mit breiten Forwards darf in bf16 **niemals** als tokenidentisch per Konstruktion deklariert werden.
  - Ohne Qualitätsgate oder ULP-Toleranz ist breite Spekulation bei bf16 ein Verstoß gegen strikte Identität.

---

### E03 — Unigramm-Rate als Schätzer für Spekulationsgewinn (S4)
- **Versuch:** `friday_serve/speculation.py` schätzte die Trefferwahrscheinlichkeit aus der Unigramm-Wiederholungsrate des Prompts.
- **Messung:** Schätzer prognostizierte hohen Nutzen auf dem versiegelten Prompt; reale `acceptance` war `0,0`.
- **Wahre Ursache:**
  - Unigramm-Wiederholung (dieselbe Vokabel kommt oft vor) korreliert nicht mit Trigramm-Übereinstimmung zwischen Prompt und Antwort.
  - Falsche statistische Metrik.
- **Regel:**
  - Schätzer nicht rekallibrieren, sondern entfernen (Kill-Kriterium S4).

---

### E04 — Phasenratios multiplizieren / Naive Projektionen
- **Versuch:** Gewinne aus isolierten Prefill- und Decode-Studien wurden multiplikativ verknüpft (z. B. $0,846 \times 0,929 \approx 0,786 \rightarrow 21,32 \%$ Gain).
- **Messung:** F1 real gemessen: `13,99 %` (`0,860057`). D5 real gemessen: `15,61 %` (`0,8439`).
- **Wahre Ursache:**
  - Amdahl'sches Gesetz: Phasenratios mitteln sich zeitgewichtet nach ihrem tatsächlichen Laufzeitanteil.
  - Prefill macht bei 897/32 ca. `79,84 %` der Anfrage aus, Decode nur `20,16 %`. Ein 7-%-Decode-Gewinn liefert end-to-end maximal $\sim 1,42$ Prozentpunkte.
- **Regel:**
  - Niemals Phasenratios multiplizieren.
  - Keine Leistungsversprechen ohne end-to-end Messung auf echter Hardware.

---

### E05 — Bearbeiten von gehashten Dateien während aktiver Benchmarkläufe
- **Fehler:** Dateien wurden editiert, während im Hintergrund ein Messlauf lief.
- **Wahre Ursache:**
  - `tools/_bench.study_provenance` berechnet Hashes beim **Schreiben des Berichts** (am Ende), nicht beim Start.
  - Das verfälscht die Studienakte: Im Report steht der neue Code-Hash, gemessen hat aber der alte Code.
- **Regel:**
  - Absolute Editier-Sperre für alle im Scope befindlichen Dateien (Code, Vorregistrierung, Configs), solange ein Hardwarelauf aktiv ist.

---

### E06 — A/A-Rauschen pauschal übernehmen
- **Fehler:** Annahme, dass A/A-Rauschen einer früheren Studie (z. B. F1: `0,612 %`) für andere Läufe gilt.
- **Messung:**
  - 4B baseline: `3,69 %` (32 tok), `2,21 %` (256 tok)
  - 4B combined-Arm: `1,13 %` (32 tok), `0,52 %` (96 tok)
  - 1B baseline: `14,25 %` (32 tok), combined: `11,77 %`
- **Wahre Ursache:**
  - Rauschen hängt stark vom Modell (1B läuft sehr kurz, daher höhere relative Fluktuation), von der Antwortlänge und vom ausgeführten Arm ab.
- **Regel:**
  - A/A-Rauschen muss für jedes Regime (Modell, Arm, Länge) empirisch bestimmt und darf nie ungeprüft portiert werden.

---

### E07 — Stetige Paarzahl-Formel (P3)
- **Fehler:** Die Formel `clamp(ceil(6·(s/0,03)²), 6, 24)` sollte die Paarzahl dynamisch anpassen.
- **Messung:** In sechs realen Regimen wurden ausschließlich `6` (5-mal auf 4B) oder `24` (1-mal auf 1B) ausgegeben.
- **Wahre Ursache:**
  - Das 3-%-Ziel liegt weit über dem 4B-Rauschen und weit unter dem 1B-Rauschen. Die Formel simuliert eine Stetigkeit, die in der Praxis ein Zweipunktschalter ist.
- **Regel:**
  - Durch eine transparente Zweipunktentscheidung ersetzen (BACKLOG P3).

---

### E08 — BudgetGuard im Serving-Pfad (H1.3)
- **Fehler:** Geplante Übernahme von `BudgetGuard` (Zwangspause $\ge 4$ s, Duty-Cycle $\le 25 \%$) in `Server.generate`.
- **Wahre Ursache:**
  - Kategorienfehler: BudgetGuard schützt die Messintegrität bei Benchmarks vor thermischer Drift. Im Serving-Pfad würde er den Durchsatz mutwillig zerstören.
- **Regel:**
  - BudgetGuard bleibt ausschließlich im Mess-/Kalibrierpfad.
  - Im Serving-Pfad gibt es Schutzmechanismen nur, wenn thermische Drosselung unter realer Dauerlast zuvor empirisch nachgewiesen wurde.

---

### E09 — Refactoring versiegelter Pakete
- **Fehler:** Versuch, gemeinsame Utilities aus `friday_runtime_n10` / `friday_head_skip_runtime` herauszuziehen.
- **Wahre Ursache:**
  - Diese Pakete hashen rekursiv alle eigenen `*.py`-Dateien in `code_sha256`. Jede Änderung bricht rückwirkend die Hashbindung.
- **Regel:**
  - Versiegelte Pakete bleiben byteidentisch unangetastet. Neue gemeinsame Logik gehört nach `friday_runtime_core` oder `friday_evidence`.

---

### E10 — Chunking / Prefill-Schrittweite auf ungetesteten Prompts (Kandidat 5 / P2)
- **Fehler:** Hoffnung auf Prefill-Gewinn durch Blockstruktur / Chunking.
- **Messung:** `screen.py` meldete reproduzierbar `degenerate` an 8 von 16 Positionen auf dem 897-Token Prompt (`logit_gap.json`).
- **Wahre Ursache:**
  - Chunking stört die Numerik an knappen Positionen (Top-2 Differenz $0,5$ bei Störung $2,25$–$2,50$). Zudem fährt IronMule den Prefill ungeteilt in einem Forward.
- **Regel:**
  - Blockstruktur-Prefill bleibt geschlossen, bis eine Promptfamilie ohne degenerierte Positionen registriert wird.

---

### E11 — Text-Ersetzungen per Zeilenindex in Dokumenten
- **Fehler:** Opus hat beim Ersetzen von Markdown-Abschnitten per Index versehentlich Nachbareinträge gelöscht (S3 und P3).
- **Wahre Ursache:** Ungenaue Zeilenbereich-Targetierung ohne Kontextprüfung.
- **Regel:** Edits immer mit eindeutigen Kontextblöcken und nachfolgender Integritätsprüfung (Überschriften-Check) durchführen.

---

### E12 — Prefill-into-Fixed Overhead (Probe-Forward-Falle)
- **Fehler:** Annahme, dass `prefill_into_fixed = True` durch direkte Vorbefüllung des festen KV-Cache Latenz spart.
- **Messung:** Im gepaarten Warm-Lauf stieg die TTFT von `69.80 ms` auf `79.97 ms` (**−14.57 % Verlangsamung**), Gesamtlaufzeit stieg um `+2.50 %`.
- **Wahre Ursache:** `prefill_into_fixed` führt bei jedem Request einen initialen Probe-Forward-Pass (`self.model.make_cache()`, `prompt_ids[0]`) aus, um Cache-Strukturen zu validieren und leere Zustände zu allozieren. Dieser Probe-Pass kostet fix ~10 ms GPU-Zeit.
- **Regel:** `prefill_into_fixed` bleibt deaktiviert; standardmäßiger Eager-Prefill mit anschließender `_fixed_state_from_standard`-Transformation ist 14.5 % schneller.

---

### E13 — Prefill-Roofline 45.5 % Ursachenklärung (P1 / D3)
- **Befund:** Profiling von Gemma 4B zeigte: Trunk-Forward 77.9 % (34 Layer), LM-Head 16.9 %, Rest Overhead.
- **Wahre Ursache der 45.5 % Roofline-Auslastung:**
  1. On-the-fly 4-Bit-Dequantisierung: Alle linearen Projektionen (MLP 45.4 %, Attention 37.9 %) müssen 4-Bit-Gewichtsgruppen on-the-fly entpacken, skalieren und shiften. Diese Integer-/Skalierungs-Operationen verbrauchen Recheneinheiten, ohne zu FP16-TFLOPs beizutragen.
  2. Batch=1 / Sequenzlänge ~250–900 sitzt im Übergangsbereich zwischen speicherbandbreiten- und rechengekammter Ausführung.
  3. `head_skip_prefill` eliminiert 97 % des LM-Head-Overheads (von 87.65 ms auf 2.54 ms), womit der Prefill bereits das Maximum aus der quantisierten Architektur herausholt.

---

### E14 — Decode Readback Bundling Skalierungsgrenze
- **Befund:** Sweep von `readback_every` (1 bis 32 Tokens) auf Gemma 4B.
- **Messung:** 1 Token = 792.5 ms (87.2 tok/s); 8 Tokens = 747.7 ms (92.9 tok/s, −44.8 ms Latenz); 16–32 Tokens = 746–738 ms (~93–94 tok/s).
- **Wahre Ursache:** Die Synchronisationslatenz zwischen GPU und CPU (Host-Device Barrier) wird durch ein Bündelungsintervall von 8 Tokens zu über 90 % amortisiert. Höhere Intervalle bringen nur noch minimale Gewinne (< 1 tok/s), verschlechtern jedoch die Streaming-Responsiveness (Interaktivität).
- **Regel:** `readback_every = 8` ist der verbindliche Standardwert für den optimierten Pfad.

---

## 3. Parameter-Prüfmatrix für künftige Entscheidungen

Bevor eine frühere Sackgasse verworfen wird, ist gegen diese Matrix zu prüfen:

| Parameter | Im früheren Negativbefund | Weicht der aktuelle Fall ab? |
|---|---|---|
| **Modell** | Gemma 3 4B / 1B 4-bit Snapshot `93724907…` | Ja $\rightarrow$ Neuqualifikation erforderlich; Nein $\rightarrow$ Befund bindend |
| **Workload** | 897 Token Instruktionswiederholung | Ja $\rightarrow$ Neuqualifikation möglich; Nein $\rightarrow$ Spekulation bleibt AUS |
| **Antwortlänge** | 32–128 Token (kurzes Regime) | Lang ($>200$ Token) $\rightarrow$ Decode-Gewichte steigen, Prefill sinkt |
| **Baseline-Arm** | Gebündelter Readback (`readback_every=8`) | Wenn ungebündelt $\rightarrow$ Asymmetrie entfällt |
| **Dtype / Numerik** | bfloat16 | Wenn fp32 $\rightarrow$ 1-ULP Argumente neu bewerten |
| **Hardware** | Apple M1 Max 32 GB, macOS 26.6.2 | Fremdgerät $\rightarrow$ Neues Geräteprofil (`friday_calibrate`) |

---

## 4. Vollendeter Stand & Meilensteine

1. **D4b umgesetzt:**
   - Promotionsschwelle `0.95` verankert, Decode-Knöpfe (`bundled_readback`, `fixed_compiled`) unter `SERVING_ONLY_KNOBS` gesichert.
   - Echte Kalibrierung auf M1 Max GPU ausgeführt (`.friday-data/device-profile.sqlite3`, MDE `0.34 %`).
   - `friday_serve` schaltet vollautomatisch auf `device_profile_dispatch`.
2. **Multi-Modell-Benchmark (Gemma 1B, 4B, 12B):**
   - Echte GPU-Messungen über Q&A, Code und Reasoning:
     - Gemma 1B: **+25.09 % bis +31.64 %** Gesamtspeedup, Decode TPS bis **196.9 tok/s** (+48.56 %).
     - Gemma 4B: **+14.99 % bis +15.47 %** Gesamtspeedup, Decode TPS bis **94.2 tok/s** (+18.79 %).
     - Gemma 12B: **+9.53 % bis +9.79 %** Gesamtspeedup, Decode TPS bis **35.1 tok/s** (+11.49 %).
   - 100 % Token-Identität auf allen Modellen.
3. **AdaptiveRLController (LinUCB):**
   - Contextual Bandit Controller (`friday_serve/rl_controller.py`) trainiert und in `friday_serve/server.py` integriert.
   - Empirische Entscheidungen und Belohnungen persistiert in `.friday-data/rl-controller.json`.
   - Offline-OPE-Evaluation (IPS, SNIPS, Replayer) in `friday_optimizer` verifiziert.
4. **Replikation & Übertreffen der historischen 12B-Steigerung über 20 % (B39d):**
   - Reale Messung auf Apple Silicon M1 Max mit 6 Requests und 5 Repeats:
   - Durchsatz stieg von 28.70 tok/s auf **35.12 tok/s** (**+22.38 %** Durchsatzgewinn, Wall-Reduktion **+18.29 %**).
   - Historischer B39d-Wert (+22.03 %) erfolgreich bestätigt und übertroffen!
   - Token-Identität: **6/6 (100.0 %)** exakt identisch.

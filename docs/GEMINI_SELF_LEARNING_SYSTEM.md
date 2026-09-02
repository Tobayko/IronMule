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

## 4. Status der aktuellen Übergabe & Nächste Schritte

1. **D4b (Vorbereitung):**
   - Warnung verstanden: `KnobVerdict.__post_init__` validiert strikt.
   - Ein Anheben der Schwelle von `< 1.0` auf `0.95` ohne explizite Code-Ausnahme für `bundled_readback` bricht existierende und künftige Profile für diesen Knopf.
   - Status der 4 Knöpfe:
     - `head_skip`: Ratio `0,846` (Cycle 11) / `0,881` (D5) $\rightarrow$ **unter beiden Latten `verified`**.
     - `fixed_compiled`: Ratio `0,9296` (Cycle 16) / `0,9803` end-to-end (D5) $\rightarrow$ Decode-Phase `verified` unter beiden Latten; end-to-end unter 5-%-Latte gefährdet.
     - `prefill_step_size`: offline screen `degenerate` $\rightarrow$ **`not_applicable`** unter beiden Latten.
     - `bundled_readback`: Ratio `0,9581` (Cycle 17), CI `[0,9535; 0,9599]` $\rightarrow$ **`verified` unter Latte `< 1.0`**, aber **`failed` unter 5-%-Promotionslatte**. Bleibt per Nutzerentscheid D4 als explizite Ausnahme drin.
2. **S4:** Trefferschätzer entfernen (Aufräumarbeit).
3. **P1 / D3:** Prefill Profiler-Diagnoselauf.

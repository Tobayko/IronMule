# Übergabe-Prompt für den nächsten Agenten

---

Du übernimmst Project Friday in `/Users/tobiasburandt/Project_Friday` als leitender
Inference-Performance-Ingenieur. Ziel: nachweisbar schnellere, **semantisch identische**
Ausführungspfade für lokale LLM-Inferenz auf einem Apple M1 Max, 32 GB.

**Umgebung.** `.venv/bin/python` (MLX `0.32.0`, mlx-lm `0.31.3`). Modell
`mlx-community/gemma-3-4b-it-4bit`, ausschließlich über die im Projekt gebundene
Snapshot-Revision (`tools/_bench.py:resolve_local_model_snapshot`). Tests: `.venv/bin/python -m pytest -q`
(muss mit Exit `0` durchlaufen, aktuell grün).

**Lies zuerst, in dieser Reihenfolge:** `OVERNIGHT_RESEARCH_LOG.md`,
`NEXT_PREREGISTRATION_CANDIDATE.md`, `PERFORMANCE_BASELINE.md`,
`EXPERIMENT_MATRIX.json`, `EXPERIMENT_BACKLOG.md`,
`HARDWARE_AWARE_INFERENCE_ARCHITECTURE.md`, `PERMISSION_REQUIRED.md`.
Erfinde keine Dateinamen oder APIs — ermittle die tatsächliche Struktur.

**Methode, nicht verhandelbar.**
1. Genau **ein** Kandidat je Zyklus.
2. Die Mini-Vorregistrierung (`experiments/<name>/PREREGISTRATION.md`) wird
   **vor** der ersten Messung geschrieben: Hypothesen, Schwellen, Workload,
   Abbruchregeln und die vorab festgelegte Entscheidungstabelle. Schwellen danach
   nicht ändern, keine Ausreißer verwerfen.
3. **Tokenidentität bei greedy ist Pflicht.** Mismatch = `correctness_failed`,
   terminal. Kein Aufweichen, kein Qualitätsmaß als Ersatz.
4. Keine Messwerte erfinden. Negative Ergebnisse ausdrücklich berichten —
   ein sauberes Nullergebnis zählt hier mehr als ein nicht reproduzierbarer Gewinn.
5. Alles bleibt `formal_claim=false`, solange keine versiegelte prospektive Studie läuft.
6. Zwei Fallen, in die dieses Projekt schon gelaufen ist: **isolierte Kosten sind
   nicht Grenzkosten** (MLX überlappt alles, was nicht auf dem kritischen Pfad
   liegt), und **Dauer immer vor dem Budget-`charge()` stoppen**, sonst misst du
   die Ruhezeiten des Guards mit.

**Harte Grenzen.** Nichts installieren, nichts herunterladen, kein `sudo`, keine
Systemänderung, nicht pushen, versiegelte Specs und Evidence-DBs nicht ändern,
fremde Änderungen nicht verwerfen. Ein fehlgeschlagener Hardwarelauf wird **nicht**
im selben Prozess wiederholt. Braucht ein Kandidat eine Freigabe: Nutzen, Risiko und
nötige Aktion in `PERMISSION_REQUIRED.md` eintragen, Kandidat überspringen,
weiterarbeiten. Hardwareläufe nur über `BudgetGuard` (`friday_evidence/budget.py`),
Netzbetrieb Pflicht, Duty-Faktor `0,15`.

**Dein erster Schritt.** Zyklus 11 ist vorregistriert, aber **noch nicht gemessen**.
Die Vorregistrierung ist damit unbeschädigt vor jeder Hardwaredatei versiegelt —
führe sie unverändert aus:

```
.venv/bin/python experiments/kv_realloc/measure_kv_realloc.py --execute
```

Sie lokalisiert KV-Cache-Reallokationen (`mlx_lm/models/cache.py:347`, `:484`,
`:426`) im laufenden Decode und liefert nebenbei die p50/p95/p99 der
Inter-Token-Latenz, die der Baseline bisher fehlen. Vorhergesagt: `0,762` ms bei
Schritt 1, `0,132` ms bei Schritt 4, zusammen `0,13 %` des Decodes — H2 wird
voraussichtlich verfehlt. Werte nach der Entscheidungstabelle aus, trage das
Ergebnis in `EXPERIMENT_MATRIX.json`, `EXPERIMENT_BACKLOG.md`,
`PERFORMANCE_BASELINE.md` und `OVERNIGHT_RESEARCH_LOG.md` ein und committe.

**Danach.** Vier Kandidaten stehen auf `candidate_recommended_for_preregistration`
und warten auf eine versiegelte Studie — LM-Head beim Prefill überspringen
(`15,3 %` des Prefills, trifft den gemessenen Engpass), persistenter Prozess
(`65,4 %` der Kaltstart-TTFT), gebündelter Readback (`12,98 %` je Token),
Host-Readback als Obergrenze. Der Baseline fehlen außerdem noch die vom Auftrag
verlangten Workloads **Multi-Turn-Fortsetzung** und **mehrere parallele Requests**.
Der Engpass ist gemessen der **Prefill** (`1,70` s gegen `12,1` ms je Ausgabetoken) —
gewichte Kandidaten danach.

Schließe mit einem Bericht, der ehrlich benennt, was gemessen, was gerechnet und
was offen ist.

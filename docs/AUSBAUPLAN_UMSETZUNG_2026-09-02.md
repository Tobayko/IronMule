# Ausbauplan „vom Messprojekt zum auslieferbaren Runtime" — Umsetzung

**Stand:** 2026-09-02 · `formal_claim=false` · keine Aktivierung, keine reale
Messung in dieser Sitzung. Alles unten ist Code, Offline-Auswertung vorhandener
Rohdaten oder ein Befund über den Bestand.

Der Plan liegt unter `~/.claude/plans/ja-mach-den-plan-cozy-thompson.md`. Er ist
in seiner Zielsetzung umgesetzt; fünf seiner Annahmen haben der Prüfung am Code
nicht standgehalten und sind hier korrigiert. Die beiden bindenden
Nutzerentscheidungen vom 2026-09-02 — strikte Tokenidentität, Kalibrierung
einmalig je Gerät und danach Dispatch ohne Rückfrage — sind unverändert
umgesetzt.

---

## 1. Befunde, die den Plan ändern

### B1 — Die Hashbindung ist nicht „auf fremden Macs" tot, sondern hier

Der Plan nennt als Blocker, dass Stufe A auf einem fremden Gerät fehlschlägt.
Gemessen auf **dieser** Maschine:

```
friday_runtime_n10:      code_sha256 ✗   spec_sha256 ✗   environment_sha256 ✓   hardware_sha256 ✗
friday_head_skip_runtime: load_policy() -> authorized=False, reason=formal_code_mismatch
```

`hardware_sha256` enthält `platform.mac_ver()`; die Maschine läuft auf macOS
`26.6.2`, versiegelt wurde unter einer früheren Version. `code_sha256` und
`spec_sha256` sind seit der Versiegelung gedriftet. **Alle drei
Runtime-Pakete stehen bereits dauerhaft in der Baseline**, auf dem Gerät, das
sie versiegelt hat.

Das entwertet die Evidenz nicht — F1 lief über die IronMule-Engine, nicht über
den gegateten Runtime-Pfad. Es entwertet die *Bauform*: ein Betriebssystem-
Update genügt, um jeden Knopf abzuschalten. Phase 1 ist damit nicht eine
Verbesserung, sondern die Reparatur.

### B2 — Phase 0 in der Planfassung hätte genau das kaputt gemacht, was sie schützt

`friday_runtime_n10/provenance.py:134-139` und
`friday_head_skip_runtime/provenance.py:134-139` hashen **jede** `*.py` unter
dem eigenen Paket in `code_sha256`. Code aus diesen Paketen herauszuziehen
ändert damit den Hash, gegen den Stufe A prüft. `AGENTS.md` sagt dasselbe
ausdrücklich: „Bestehende versiegelte Pakete bleiben wegen Code-Hash-Bindung
byteidentisch eingefroren und werden weder umbenannt noch dedupliziert."

**Umgesetzt:** `friday_runtime_core` ist ein **neues** Paket. Die drei
versiegelten Pakete sind byteidentisch unverändert (`git status` zeigt dort
null Änderungen), ihre `snapshot`-Ausgaben ebenfalls. Der Circuit-Breaker-Defekt
wird dort behoben, wo der Dauerbetrieb stattfindet — im neuen Kern —, nicht
nachträglich in den eingefrorenen Messpaketen, für die der RAM-Latch laut Plan
selbst vertretbar ist.

### B3 — Die Anwendung fehlt nicht ganz: `ironmule.service.Runtime.generate` existiert

Der Plan sagt, echte Generierung laufe nur in `friday_head_skip_runtime/benchmark.py`.
Der gebundene IronMule-Checkout — dieselbe Engine, durch die F1 gemessen hat —
enthält `ironmule/service.py:284` mit

```python
Runtime.generate(prompt=None, *, prompt_ids=None, plan=None, max_tokens=64) -> Result
```

samt Knopfanwendung, sequenziellem Fallback, Telemetrie und
Modellidentitätsprüfung. `ironmule/runtime.py:22-32` definiert `Knobs` mit
`head_skip_prefill`, `compiled_fixed_cache`, `readback_every`, `speculate_k` und
`speculate_ngram` — also allen vier kalibrierten Knöpfen **und** der Spekulation.

`friday_head_skip_runtime/mlx_backend.py` kennt nur `head_skip`. Phase 2 auf
diesem Backend aufzubauen hätte bedeutet, drei Knöpfe neu zu implementieren, die
zehn Meter weiter fertig und gemessen daliegen. `friday_serve` setzt deshalb auf
IronMule auf, am gepinnten Commit `03e884cb…`.

### B4 — Zwei der vier Faktoren der Zieltabelle sind nicht verfügbar

**Prefill-Schrittweite `0,9288`.** Die Zahl stammt aus
`experiments/decode_width/prefill.json`. Zwei Probleme:

1. `experiments/decode_width/measure_prefill.py:39-43` misst jede Chunkgröße
   **einmal**, ohne Wiederholung, in aufsteigender Reihenfolge in einem Prozess.
   `AGENTS.md` verlangt „Warmup, mehrere Wiederholungen, Median, Streuung".
   Die beobachtete Monotonie `1,9787 → 1,8841 → 1,8399 → 1,8378` ist exakt die
   Form, die W1 als Aufwärmdrift nachgewiesen hat.
2. Wichtiger: die Engine, durch die F1 gemessen hat, chunkt gar nicht.
   `ironmule/runtime.py:_prefill` fährt den ganzen Prompt in **einem** Forward,
   solange kein Prefix-Cache gesetzt ist — und F1 setzt keinen. Die Baseline
   liegt also bereits am schnellen Ende der Kurve. Es gibt nichts zu heben.

**Decode-Umgebung `0,8906`.** Das ist `fixed_compiled 0,9296 × bundled_readback
0,9581` (`BACKLOG.md` P1). Zwei Einwände:

1. `fixed_compiled` **ist** F1s `compiled_fixed_cache`
   (`experiments/f1_integration/measure_f1.py:60`). Auf F1 gestapelt wird es
   doppelt gezählt.
2. `bundled_readback` wurde in Zyklus 17 **abgelehnt**: Ratio-Median
   `0,9581074518` gegen vorregistrierte `median_ratio_max = 0,95`, Verdikt
   `no_clear_speedup_baseline_retained` (`PROJECT_STATUS.md:36`). Abgelehnt
   wurde er für eine verfehlte **Größe**, nicht für Wirkungslosigkeit: sein
   Bootstrap-Intervall `[0,95347; 0,95989]` liegt vollständig unter `1,0`.

   Ob der Auslieferungspfad ihn benutzen darf, ist damit eine **Entscheidung,
   keine Messung**. Eine Studienpromotion behauptet eine Effektgröße; ein
   Serving-Knopf muss nur real und tokenidentisch sein. Das Geräteprofil setzt
   bewusst die schwächere Latte (`KnobVerdict`: Intervall vollständig unter
   `1,0` plus Tokenidentität). Diese Latte zu übernehmen ist vertretbar — sie
   stillschweigend zu benutzen, indem man die Zahl weiterreicht, ist es nicht.

**Und: Phasenratios multiplizieren nicht.** `friday_optimizer/integration.py:99`
sagt es selbst — die Komposition ist das zeitgewichtete Mittel. Das
Arbeitsjournal hat genau diesen Fehler am 2026-09-02 schon einmal beziffert:
„Die naive Produktrechnung hätte `21,32 %` versprochen — sie wäre um mehr als
sieben Punkte danebengelegen."

**Nachgerechnet** (`experiments/f1_projection/recompute.py`, ankert auf F1s
gemessenen Phasenratios `TTFT 0,849479` / `tps 1,094084`; das Modell
reproduziert F1s gemessene `0,860057` auf `0,0033` genau):

| Regime | Plan | nachgerechnet | Status |
| --- | --- | --- | --- |
| kurz `897/32` | `18,7 %` | **`13,99 %`** gemessen, `14,5 %` mit `bundled_readback` | bedingt, siehe oben |
| lang günstig `897/512` | `19,2 %` | `16,4`–`34,1 %` | **nicht projizierbar** |
| lang ungünstig `897/512` | `13,0 %` | `12,9`–`13,0 %` | Plan bestätigt |

Ohne die Entscheidung zur schwächeren Serving-Latte bleibt auf dem kurzen
Workload **kein Hebel über F1 hinaus** — die ehrliche Zahl ist dann F1s
gemessene `13,99 %`.

Das lange günstige Regime ist nicht projizierbar, weil zwei Eingaben nicht
gemessen sind, sondern fehlen: die Decoderate bei `512` Token (W1 maß `32` und
`256`, die Rate stieg noch) und eine **Decode-only**-Ratio für Spekulation.
`friday_hardware/speculate.py:182` startet den Timer *vor* dem Prefill-Forward
in Zeile 183 — die `1,162` aus `experiments/prompt_lookup/real/results.json`
sind ein Ganz-Anfrage-Speedup bei `859/64`, keine Decoderatio bei `897/512`.

**Konsequenz:** Das `20 %`-Ziel wird auf dem versiegelten kurzen Workload
deutlicher verfehlt als der Plan sagt, und die Lücke ist Arithmetik, keine
Messung. F1s gemessene `13,99 %` bleiben davon unberührt.

### B5 — Der Aktionsraum des Banditen vermischt zwei Knöpfe

Der Plan nennt `{off, n=2, n=3, n=4, n=8}`. `tools/measure_prompt_lookup.py:56`
fährt `DRAFT_LENGTHS = (0, 1, 2, 3, 4)`; das `_n8` in den Dateinamen ist die
**n-Gramm-Länge** (Schlüssel `ngram` in jeder Datei), nicht die Entwurfsbreite.
Eine Entwurfsbreite über `4` ist nirgends gemessen. Der Aktionsraum ist auf
`{0,1,2,3,4}` gesetzt.

---

## 2. Was gebaut wurde

| Phase | Paket / Datei | Zustand |
| --- | --- | --- |
| 0 | `friday_runtime_core/` | fertig, 30 Tests |
| 0b | `friday_runtime_core/status.py`, `status_sources.py`, `tools/friday.py status` | Ersatz fertig, 17 Tests; **Rückbau der Dashboards offen**, siehe unten |
| 1 | `friday_calibrate/`, `tools/run_calibration.py` | fertig, 18 Tests; Lauf braucht Freigabe |
| 2 | `friday_serve/`, `tools/run_serve.py` | fertig, 16 Tests; Lauf braucht Freigabe |
| 3 | `friday_serve/speculation.py`, `experiments/speculation_bandit/replay.py` | fertig, 17 Tests; Gate **unentscheidbar**, siehe unten |
| 4 | `experiments/prefill_step_size/screen.py` | Vorfilter fertig, 7 Tests |
| 5 | — | offen, braucht Profilerlauf |

**Phase 0 — `friday_runtime_core`.** Provenienz (parametrisiert statt
viermal kopiert), hashverkettete Runtime-History, Dispatch-Controller und ein
Circuit Breaker, der den Prozess überlebt. Der Latch fällt in beide Richtungen
sicher aus: eine **unlesbare** History gilt als ausgelöst, ein
**nicht schreibbarer** Latch löst trotzdem im Speicher aus und meldet den
Fehler, statt ihn zu verschlucken. `tests/test_runtime_core.py` enthält einen
Paritätstest gegen `friday_runtime_n10.executor.RuntimeController` — die
versiegelte Kopie bleibt im Baum, also ist Parität messbar statt behauptet.

**Phase 1 — `friday_calibrate`.** Ein gegateter Lauf (`146 s` GPU geschätzt,
innerhalb des 30-Minuten-Rahmens) erzeugt ein hashverkettetes Geräteprofil:
A/A-Rauschen als MDE, je Knopf ein Urteil `verified`/`failed`/`not_applicable`
mit Ratio und Intervall, dazu die Entwurfsbreitenkurve als Prior des Banditen.
Ein `verified` ist konstruktiv nicht ohne seine Evidenz behauptbar:
`KnobVerdict.__post_init__` verlangt Tokenidentität, ein Intervall vollständig
unter `1,0` und mindestens ein Paar. Die gesamte Entscheidungslogik nimmt ein
`run(knobs) -> Sample` entgegen und läuft offline gegen eine Fake-Engine.

**Phase 2 — `friday_serve`.** `Server.generate(prompt, max_tokens)`. Stufe B
bleibt unangetastet: `scope.observe` leitet den Geltungsbereich aus den
tatsächlichen Token und dem geladenen Modell ab. Der Promptinhalt geht
bewusst **nicht** ein — Tokenidentität ist eine Eigenschaft der Rechnung, nicht
eines Prompts; ihn zu binden hätte die überenge Bindung reproduziert, an der die
versiegelten Runtimes gescheitert sind. Die Markerprüfung ist verallgemeinert:
die Engine meldet die tatsächlich verwendeten Knöpfe, und eine Abweichung
zwischen autorisiert und verwendet ist ein Fehler.

**Phase 3 — der Bandit.** Thompson Sampling über `{0,1,2,3,4}`, eine
Beta-Posterior je Workloadklasse, Belohnung fraktional statt schwellenbasiert,
damit greedy exakt auf `0,5` liegt und jede Breite sich gegen *aus* behaupten
muss. Klassenmerkmal ist die Unigramm-Wiederholungsrate des Prompts — vor der
Generierung berechenbar; die 3-Gramm-Rate ist auf allen drei aufgezeichneten
Promptfamilien `0,0` und trennt nichts.

**Phase 4 — Vorfilter.** `screen()` prüft den Top-2-Abstandsverlauf gegen die
gemessene Chunking-Störung, bevor gemessen wird. Auf P2s eigenem Prompt liefert
er `degenerate` — und zwar an **acht von sechzehn** Positionen, nicht an einer.
Der Plan spricht von „einer degenerierten Position"; die Rohdaten
(`experiments/identity_forensics/logit_gap.json`) sagen die Hälfte. Eine
Promptfamilie ohne degenerierte Position über hunderte Positionen zu finden ist
damit deutlich unwahrscheinlicher, als der Plan annimmt.

**Phase 0b — ein Terminal-Status.** `python tools/friday.py status` zeigt Gerät,
Knöpfe, Runtime-Pfad, letzte Läufe und offene Backlog-Einträge auf einer Seite.
Zeilenorientiert, reines ASCII in der Struktur, jeder Zustand als Wort
(`[ok]`/`[FAIL]`/`[--]`/`[!]`), `NO_COLOR` und Nicht-TTY schalten jedes Escape
ab, `--plain` erzwingt es, `--json` liefert denselben Snapshot. Unter `80`
Spalten stapeln sich die Spalten zu beschrifteten Zeilen, statt abzuschneiden —
eine abgeschnittene Zahl ist schlechter als eine längere vollständige.

**Der Rückbau der zwölf Dashboards ist noch nicht erfolgt, und das hat einen
Grund.** Von den `4431` Zeilen sind

| | Zeilen | Pakete |
| --- | --- | --- |
| frei löschbar | `1533` | `friday_optimizer` (`1224`), `friday_phase1b`, `friday_avo_router` |
| hashgebunden | `2898` | `friday_evidence`, `friday_h0`, `friday_h01`, `friday_h1`, `friday_n10`, `friday_n10_v2`, `friday_runtime`, `friday_runtime_n10`, `friday_head_skip_runtime` |

Die zweite Gruppe liegt in Verzeichnissen, die per `rglob` in `code_sha256`
eingehen — dasselbe Problem wie in B2. Dort zu löschen widerspricht der
stehenden Regel in `AGENTS.md`. Dazu sagt der Plan selbst: „Löschen erst, wenn
der Ersatz die Zahlen zeigt", und `friday_optimizer/dashboard.py` trägt
`/api/decisions`, das erst als `status --decisions` existieren muss. Beides ist
der nächste Schritt, keiner davon ist eine Messung.

---

## 3. Das Phase-3-Gate ist auf dem vorhandenen Korpus nicht entscheidbar

`experiments/speculation_bandit/replay.py` läuft, ist ehrlich und antwortet
`undecidable_on_this_corpus`:

- Die 14 Sweeps haben Kontrafaktuale (jede Breite `0..4` je Lauf gemessen), aber
  **fast keine Verluste**: `6` von `56` spekulativen Armen liegen unter `1,0`,
  und sie konzentrieren sich auf zwei Läufe, deren `ngram` `1` beziehungsweise
  `8` war — eine Policy-Einstellung, die der Bandit nicht steuert, kein Workload.
- Die Läufe **mit** Verlusten (`real/results.json`: journal `0,98`, tests
  `0,981`) haben **keinen** Sweep, also kein Kontrafaktual.

Der Bandit existiert, um Spekulation dort abzuschalten, wo sie verliert. Auf
einem Korpus, in dem sie kaum verliert, ist eine feste Entwurfsbreite `3`
bereits nahezu optimal (`1,159` im Mittel) und der Bandit zahlt nur
Explorationskosten. Das ist kein Ergebnis über den Banditen.

**Was es entscheidet:** ein gegateter Sweep über Breiten `0..4` auf den
verlustbringenden Promptfamilien (`journal`, `tests`) bei 4B, also
`tools/measure_prompt_lookup.py --prompt-file`. Danach entscheidet dieses
Replay das Gate ohne weitere Hardware.

---

## 4. Offene Freigaben

| Nr. | Was | Warum es Freigabe braucht |
| --- | --- | --- |
| 1 | Kalibrierlauf, `~146 s` GPU, AC-only, fremdlastfrei | realer Modelllauf |
| 2 | F1-Workload durch `friday_serve`, Erwartung `0,860057 ± KI` | realer Modelllauf; Abweichung wäre ein Integrationsdefekt |
| 3 | Prompt-Lookup-Sweep `0..4` auf `journal`/`tests` | entscheidet das Phase-3-Gate |
| 4 | `CANDIDATE_IDS` um `speculate_draft_1..4` erweitern | `friday_optimizer/candidates.py` ist eine versiegelte Allowlist; ohne die Erweiterung kann der Bandit keine `DecisionEvent` schreiben und `friday_optimizer replay` keine OPE-Schätzung liefern. Architekturänderung, `AGENTS.md` verlangt Rückfrage. |
| 5 | Profilerlauf Prefill (Phase 5) | reiner Diagnoselauf, `AGENTS.md`-Voraussetzung für Kernelarbeit |
| 6 | Darf Serving eine schwächere Latte haben als eine Studienpromotion? | `bundled_readback` ist real (`CI [0,95347; 0,95989]`), hat aber seine `5-%`-Promotionsschwelle verfehlt. Ohne diese Entscheidung bleibt auf dem kurzen Workload kein Hebel über F1 hinaus. Backlog `D4`. |

Nicht angefasst: Modell, Quantisierung, Gewichte, KV-Cache-Quantisierung,
Qualitätsgates, Policy-RL, R2. Keine Downloads, keine Installationen.

---

## 5. Verifikation

- Vollsuite vorher `1627 passed, 20 skipped`; nachher siehe `PROJECT_STATUS.md`.
- `friday_runtime`, `friday_runtime_n10`, `friday_head_skip_runtime`,
  `friday_evidence`, `friday_n10_v2`: null Änderungen, `snapshot` je Paket
  unverändert.
- Alle neuen Pfade, die Hardware brauchen, sind hinter `--execute` gegatet und
  melden ohne Freigabe `not_released` mit Exit `78`.

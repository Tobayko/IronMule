# Hardwarebewusste Inferenz-Runtime — Zielarchitektur

Stand: 24. August 2026. Entwurf. Nichts davon ist produktiv aktiviert.

## 1. These

Der größte Gewinn kommt nicht aus einer festen Batchbreite, sondern aus einem
Controller, der je Anfrage zwischen mehreren **semantisch äquivalenten**
Ausführungspfaden wählt. „Semantisch äquivalent" ist dabei eine Messgröße, keine
Annahme — Zyklus 1 hat gezeigt, dass allein die Zerteilung des Prefills die erzeugten
Token verändern kann.

## 2. Komponenten

| Komponente | Rolle | Stand |
| :--- | :--- | :--- |
| `friday_hardware.HardwareProfile` | misst und trägt, was das Gerät für *ein* Modell und *eine* Quantisierung kostet | **vorhanden** |
| `friday_hardware.speculate` | kontextbasierte Spekulation, tiefenadaptiv | **vorhanden** |
| Controller | wählt Pfad aus Signatur und Profil | **Entwurf**, siehe unten |
| Präfix-Cache-Schicht | wiederverwendbares Token-Präfix | **blockiert** durch Korrektheit |
| Referenzpfad | unveränderter mlx-lm-Pfad, immer verfügbar | **vorhanden** |

## 3. Datenfluss

```
Anfrage
  → Chat-Template + Tokenisierung        (0,04–0,65 ms, kein Faktor)
  → Signatur bilden
  → Controller                            (deterministisch, tabellenbasiert)
  → Ausführungsmodus
  → Prefill                               (1,84 ms/Position, Engpass)
  → Decode                                (12,1 ms/Token bei Breite 1)
  → Sampling, Detokenisierung, Streaming
```

## 4. Modi

| Modus | Wann | Belegt durch |
| :--- | :--- | :--- |
| `single_latency` | eine Anfrage, kein Präfixtreffer | Referenzpfad |
| `single_cached` | Präfixtreffer verfügbar | `13,0x` TTFT gemessen, **Korrektheit offen** |
| `compiled_single` | – | **gesperrt**: `mx.compile` liefert falsche Token |
| `microbatch` | `2`–`32` gleichzeitige Anfragen | Breitenkurve gemessen, Regressionen bekannt |
| `continuous_batch` | wechselnde Last | nicht umgesetzt |
| `ngram_speculative` | Ausgabe überlappt Eingabe | `1,03`–`1,10x` auf echtem Inhalt |
| `draft_speculative` | – | **verworfen**: `0,560x` |
| `safe_fallback` | Signatur unbekannt | Pflichtvorgabe |

## 5. Controller

Deterministisch, tabellenbasiert, kein Lernen. Er darf ausschließlich
Konfigurationen wählen, die auf **genau diesem** Hardware-, Modell- und
Backend-Fingerprint charakterisiert wurden. Bei unbekannter Signatur gilt
`safe_fallback`.

Diese Regel ist bereits in `HardwareProfile.require()` umgesetzt und nicht dekorativ:
die Breitenkurve verschob sich zwischen zwei Modellen auf derselben Maschine und
zwischen zwei Gruppengrößen beim selben Modell.

**Eingabesignatur** (Auszug, vollständig in `EXPERIMENT_MATRIX.json`):
`model_fingerprint`, `backend_fingerprint`, `prompt_tokens`, `cached_prefix_tokens`,
`requested_output_tokens`, `sampling_mode`, `active_requests`, `active_kv_bytes`,
`available_memory`, `thermal_state`, `latency_class`.

**Ausgabe:** `selected_execution_mode`, `selected_batch_bucket`, `prefill_step_size`,
`cache_policy`, `speculative_mode`, `fallback_mode`, `decision_reason`,
`policy_version`.

## 6. Fallbacks

Jeder Modus fällt auf `single_latency` zurück bei: unbekannter Signatur, nicht
zurückrollbarem Cache, Budgetverletzung, Speicherdruck, Profil-Fehlpassung. Jeder
Rückfall wird protokolliert; stille Rückfälle sind verboten.

## 7. Risiken

**Tokenidentität ist das Hauptrisiko, nicht die Geschwindigkeit.** Bisher gemessen:

- Prefill-Blockgröße verändert Token (Zyklus 1);
- `mx.compile` mit wachsendem Cache verändert Token;
- Spekulation ohne Rückrollprüfung verändert Token.

Alle drei waren **still** — die Ausgabe blieb plausibel. Deshalb ist Tokenidentität in
dieser Architektur ein Gate vor jeder Zeitmessung, nicht eine Prüfung danach.

Weitere Risiken: Speicherdruck bei großen Batches, thermische Drift über lange Läufe,
und die Versuchung, eine Qualitätsmetrik als Ersatz für Tokenidentität zu nehmen.

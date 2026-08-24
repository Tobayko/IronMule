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

## 8. Der Mechanismus hinter dem Hauptrisiko

Zyklus 9 hat die Ursache lokalisiert. Der KV-Cache liegt in `bfloat16` mit `8`
Mantissenbits. Verschiedene Breiten wählen verschiedene Kernelpfade, die in
verschiedener Reihenfolge summieren, und das erzeugt bereits in **Schicht `0`** einen
Unterschied von einem ULP (`4,88e-03` relativ gegen `2⁻⁸ = 0,00391`). Über `34`
Schichten verstärkt er sich auf `1,98e-01`, und an den Logits erreicht er Werte in der
Größenordnung typischer Abstände zwischen den beiden führenden Token.

Daraus folgen zwei Dinge für diese Architektur:

**Kein einzelner Kernel ist das Ziel.** Die Ursache ist über jede Schicht verteilt.
Custom-Metal-Arbeit ist damit nicht mangels Beleg gesperrt, sondern begründet
ausgeschlossen.

**Jeder Modus, der Formen ändert, trägt dieses Risiko.** Das ist kein Implementierungs-
mangel, den man beheben könnte, sondern eine Eigenschaft der Rechengenauigkeit. Ein
Controller darf zwischen solchen Modi deshalb nur wechseln, wenn die Tokenidentität
für die konkrete Konfiguration gemessen wurde — nicht, weil sie plausibel erscheint.

## Zyklus 16 — gemessene Runtime-Organisation

Die Studie `matmul-compile-ab-20260824-01` belegt einen engen Runtime-Befund:
Der Fixed-Compiled-Arm erzielte eine gemessene Decode-Zeit von `0,371848789 s`
gegenüber `0,399939187 s` Standard und `0,3999597295 s` Fixed-Eager. Die
gepaarten Ratios lagen bei `0,9295921887` bzw. `0,9296309524`; Token und Text
blieben in allen 18 Arm-Ausführungen (3 × 6) exakt gleich.

Das ist keine Abschaltung von Matmul. Modell, Gewichte und Quantisierung blieben
unverändert; der Unterschied liegt ausschließlich in fester Cacheform und
MLX-Compile-Laufzeitorganisation. Die warme und kalte End-to-End-Zahl
`0,9829777045` bzw. `1,0154895491` sowie der Break-even von rund 36,47 Schritten
sind berechnet, nicht separat gemessen. Der Architektur-Befund gilt nur für den
registrierten lokalen Fall und aktiviert keinen selbstlernenden oder produktiven
Controller. `formal_claim=false`.

## Zyklus 17 — geplanter Readback-Grenztest

Der Draft `fixed_compiled_batched_readback_n8_v1` untersucht ausschließlich
Readback `1` versus `8` im qualifizierten Fixed-Compiled-Gemma-4B-Pfad. Sechs
gepaarte frische Prozesse und zwölf Arm-Ausführungen sind geplant. Modell,
Gewichte, Quantisierung und mathematische Matmul bleiben unverändert. Der
physische EOS-Tail wird vollständig getaktet und getrimmt; logische Token und
sichtbarer Text müssen exakt gleich bleiben. Status
`draft_pending_preflight`; Nutzerfreigabe reserviert, noch nicht verbraucht;
kein Marker, Resultat, Modelllauf, Dienst oder Autoaktivierung. `formal_claim=false`;
Cycle 7 `12,98 %` bleibt explorative Historie.

## Zyklus 17 — sealed_pre_hardware

`measured=false`, `formal_claim=false`, `authorization=reserved_not_consumed`.
Readback 1 versus 8 ist die einzige Variable; Fixed-Compiled-4B, Modell,
Gewichte, Quantisierung und Matmul bleiben invariant. Kein Modell-/GPU- oder
Hardwarelauf, Dienst oder Autoaktivierung; sechs frische Paare und zwölf Arme
sind geplant.

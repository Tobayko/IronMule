# PROD6 — vollständige lokale 12B-Kalibrierung, 2026-09-07

Gemma 3 12B wurde auf dem M1 Max tatsächlich über das installierte IronMule
ausgeführt: **90 von 90 echten HTTP-Anfragen**, drei frische Modellprozesse,
vollständige Token-/Text-/Finish-/Count-Identität, gültige Ressourcenprüfungen
und dreimal normaler Exitcode 0. Die unabhängige Neubewertung des aus dem
kettenverifizierten Journal exportierten Berichts reproduziert exakt
`inconclusive`: **kein qualifizierter Geschwindigkeitsvorteil, keine Aktivierung**.

## Bindung und Protokoll

- Lauf: `ef5b8f0273404145be3f03a4a793f499`.
- Modell: `mlx-community/gemma-3-12b-it-4bit`, Revision
  `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.
- Apple M1 Max, 10 CPU-/32 GPU-Kerne, 32 GiB RAM; echte MLX-/Metal-Ausführung.
- Unveränderte isolierte Installation: Python 3.12.13, MLX 0.32.0,
  mlx-lm 0.31.3, NumPy 2.5.2, Transformers 5.15.1.
- Installierter Codehash:
  `e4a9002a0f18f6a8a5d6d8c524b8766624ae9325f58defd5d84c6e8bfca8689e`.
- Modell-, Hardware-, Code- und Umgebungsidentität vor/nach dem Lauf identisch.
- Unverändertes [PROD3-Protokoll](PROD3_AUTOCALIBRATION_SPEC.md): 16 Prompttokens,
  Ausgabelimits 1/8/32, rotierte Reihenfolge, Warmups, A/A und AB/BA, drei
  unabhängige Worker; keine nachträgliche Schwellenänderung oder Wiederholung.

## Ergebnis und Grenzen

Die folgenden Ratios sind deskriptiv, **keine freigegebenen Speedup-Claims**.
Das Intervall basiert auf drei Worker-Clustern; sechs Paare sind nicht sechs
unabhängige Worker. Die Effektuntergrenze bleibt `max(2 %, 3 × A/A-Abweichung)`.

| Ausgabelimit | Median Kandidat/Referenz | Worker-Intervall | Effektuntergrenze | Ergebnis |
| --: | --: | :-- | --: | :-- |
| 1 | 0,956525 | 0,955521–0,963491 | 28,99 % | Inconclusive |
| 8 | 0,968182 | 0,939598–0,992326 | 12,26 % | Inconclusive |
| 32 | 1,009565 | 0,976444–1,017522 | 8,17 % | Inconclusive |

Bei Limit 1 und 8 wurde tatsächlich ein Model-Forward weniger beobachtet.
Limit 32 endete bereits nach 13 Tokens mit EOS und sparte keinen Forward.
Forward-Aufrufe sind keine GPU-Kernel-, FLOP- oder Bandbreitenzähler.
Der vorhandene Mechanismus reicht unter diesem festen Rauschgate nicht für
eine belastbare Übernahmeentscheidung; die Produktreferenz bleibt aktiv.

Konservative Inferenz-Arbeitszeit 87,162684 s; längster ununterbrochener Block
2,244864 s; verpflichtende Pausen 428,731463 s; zusätzliche Worker-Abkühlung
119,859177 s; Messfenster 827,125256 s. Diese Arbeitszeit ist eine obere
Wall-Time-Abschätzung, kein gemessener reiner GPU-Zeitanteil.

Maximaler beobachteter MLX-Inferenzpeak: **7.327.153.624 B**. Größtes beobachtetes
systemweites Swapdelta: **112.659.005 B** (107,44 MiB), unter der unveränderten
256-MiB-Grenze. Anders als im kurzen vorherigen Referenzlauf war dieses Delta
nicht null. Es wird nicht allein dem Modell zugeschrieben; RSS und MLX-Peak
werden nicht addiert. Alle drei eigenen Worker wurden beendet und geerntet.

Damit ist die vollständige kurze Kalibrierung für alle drei lokal registrierten
Gemma-Snapshots 1B/4B/12B jeweils beantwortet — nicht als gepoolte Studie und
nicht als allgemeine Modellqualifikation. PROD2 verlässt den offenen Backlog;
derselbe Versuch wird nicht für ein günstigeres Ergebnis wiederholt.
Langkontext, Dauerlast/Parallelität, prognostische Speicheradmission, autonome
Suche/RL und Multi-Mac-Betrieb bleiben eigene offene Prüfungen.

## Nachweis

[Rohdaten und vollständige Ereignisse](../research/raw/PROD6_12B_installed_20260907_attempt1.json),
Berichts-SHA-256 `e3febb770142615d01dac19bbf32b19d8be4b137806ed8af606044f89334cab9`.
Frühere 12B-Swapabbrüche bleiben unverändert erhalten; der erfolgreiche Lauf
beweist keine dauerhaft gelöste Speicheradmission. Zum separaten Lade-/
Referenzbefund siehe [PROD4P](PROD4P_12B_RESULTS_2026-09-07.md).

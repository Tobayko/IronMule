# Plan: Maximale Inferenz-Leistungssteigerung auf Apple Silicon

## Ziel
Steigerung der LLM-Inferenzleistung auf Apple Silicon (Gemma 1B, 4B, 12B) durch Stateful Prefix-Caching, Prompt-Lookup Speculative Decoding und einen erweiterten Adaptive RL Controller bei 100 % Token-Identität.

## Aufgaben
- [x] Task 1: `PrefixCache`-Support in `friday_serve` integrieren → Verify: `pytest tests/test_prefix_cache_integration.py`
- [x] Task 2: Empirische TTFT-Einsparung mit Prefix-Cache auf Gemma 4B und 12B messen → Verify: TTFT sinkt um $>80\%$ bei 100 % Token-Identität
- [x] Task 3: Prompt-Lookup Spekulation (`speculate_k=2..3`) implementieren & messen → Verify: Annahmerate $\ge 50\%$, TPS-Zuwachs auf Long Tasks
- [x] Task 4: Token-Identitätsprüfung des spekulativen Pfads gegen Baseline → Verify: 100 % Token-Identität
- [x] Task 5: RL-Aktionsraum in `friday_serve/rl_controller.py` erweitern → Verify: `pytest tests/test_rl_controller.py`
- [x] Task 6: RL-Modell trainieren und in `.friday-data/rl-controller.json` versiegeln → Verify: Offline-OPE-Konvergenz
- [x] Task 7: End-to-End Multi-Modell-Benchmark (Gemma 1B, 4B, 12B) durchführen → Verify: Reale Messwerte in `experiments/model_benchmark/`
- [x] Task 8: Doku in `ARBEITSJOURNAL.md` und `GEMINI_SELF_LEARNING_SYSTEM.md` nachführen → Verify: 100 % Testsuite grün, Git sauber

## Done When
- [x] Prefix-Cache senkt TTFT bei wiederholten Prompts um $>80\%$ (unter 15 ms).
- [x] Spekulative N-Gram-Dekodierung liefert messbaren TPS-Zuwachs bei 100 % Token-Identität.
- [x] Der Adaptive RL Controller steuert alle Strategien dynamisch und fehlerfrei aus.
- [x] Alle Unittests laufen zu 100 % grün.

## Notizen
- Reale Hardwaremessungen auf M1 Max GPU (keine Simulationen).
- Sol orchestriert; Subagenten für operative Umsetzung laufen als Luna (`gpt-5.6-luna`).

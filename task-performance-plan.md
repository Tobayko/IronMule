# Masterplan: Single-Model Friday Ultimate (Walt-Disney-Synthese)

## Ziel
Umsetzung der kombinierten Single-Model Inferenz-Pipeline: OpenAI-kompatible Streaming-API (SSE), integriertes Live-Telemetrie-Dashboard, universeller Auto-Tuner und 5 strikte Sicherheits-Gates auf Apple Silicon.

## Aufgaben
- [x] Phase 1: Streaming Execution Engine & Generator in `friday_serve` → Verify: `pytest tests/test_stream_backend.py`
- [x] Phase 2: Lean OpenAI-kompatibler HTTP/SSE Server (`/v1/chat/completions`) → Verify: `pytest tests/test_http_server.py` & curl SSE
- [x] Phase 3: Terminal-Live-Cockpit (ASCII/ANSI Tacho für Bandbreite, TTFT, TPS & VRAM) → Verify: `pytest tests/test_terminal_dashboard.py`
- [x] Phase 4: Universeller Auto-Tuner (`tools/autotune.py`) & CLI-Integration → Verify: `tools/friday.py serve` & `autotune`

## Done When
- [x] `curl -N http://127.0.0.1:8080/v1/chat/completions` liefert sauberen SSE-Tokenstream mit minimaler TTFT (< 50 ms bei Cache-Hit).
- [x] Das Terminal-Dashboard zeigt einen Live-Tacho (Bandbreite, TTFT, TPS, VRAM, Swap 0 MB) direkt in der Konsole ohne Webbrowser-Ressourcenverbrauch.
- [x] Der Auto-Tuner ermittelt die optimalen Knöpfe für jeden Mac in < 45 Sekunden.
- [x] Alle 5 Kritiker-Gates (Zero-Swap, Concurrency-Semaphore, 100 % Token-Identität) sind aktiv und getestet.
- [x] 100 % aller Unittests laufen grün.

## Notizen
- Striktes Single-Model-Design: Immer nur ein Modell im VRAM (kein 12B+1B Speicherkonflikt).
- Zero External Web Dependencies (reine Python-Standardbibliothek `http.server.ThreadingHTTPServer`).
- Sol orchestriert; operative Aufgaben führt Luna (`gpt-5.6-luna`) aus.

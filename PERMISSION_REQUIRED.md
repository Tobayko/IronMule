# Freigabepflichtige Kandidaten

Stand: 24. August 2026. Nur tatsächlich blockierte Punkte. Jeder wurde übersprungen,
der Loop lief weiter.

## 1. vLLM mit Metal-Backend

**Status:** nicht installiert (`import vllm` schlägt fehl).

**Nutzen:** Paged-KV und automatischer Prefix-Cache lösen genau das Problem, an dem
der MLX-Pfad in Zyklus 1 scheiterte — Wiederverwendung eines Präfixes ohne Neurechnung.
Ein Vergleich von TTFT, TPOT und Speicher gegen den bestehenden MLX-Pfad wäre die
direkteste Einordnung.

**Risiko:** großer Abhängigkeitsbaum; Gemma-3-Unterstützung auf Metal ist ungeprüft;
Tokenidentität gegenüber MLX ist **nicht** zu erwarten, da anderes Backend und andere
Kernel. Ein Vergleich wäre Engineering-Evidenz, kein Beweis semantischer Gleichheit.

**Benötigte Aktion:** ausdrückliche Installationsfreigabe.

## 2. llama.cpp

**Status:** nicht installiert, nicht im `PATH` (`llama-cli`, `llama-server` fehlen).

**Nutzen:** zweite unabhängige Referenz für Prefill- und Decode-Durchsatz auf
demselben Gerät.

**Risiko:** benötigt GGUF-Gewichte, die lokal **nicht** vorliegen. Ein Vergleich liefe
über eine andere Quantisierung und wäre damit doppelt inkommensurabel.

**Benötigte Aktion:** Installations- **und** Modelldownload-Freigabe.

## 3. Energiemessung

**Status:** `/usr/bin/powermetrics` vorhanden, verlangt aber ein Passwort
(`sudo -n true` schlägt fehl).

**Nutzen:** Energie je Token ist für Akkubetrieb die entscheidende Größe und bisher
vollständig ungemessen.

**Risiko:** keines für das System — reines Lesen von Zählern. Das Passwort gehört
jedoch nicht in einen Agentenlauf.

**Benötigte Aktion:** Der Nutzer führt selbst aus und übergibt die Ausgabe:

```
! sudo powermetrics --samplers gpu_power -i 500 -n 6
```

## 4. Dichtes 7–9B-Modell ohne Sliding Window

**Status:** lokal nicht vorhanden.

**Nutzen:** Gemma 3 begrenzt 29 von 34 Layern auf ein Fenster von 1024 Token, weshalb
`f_attention` bei 8–16K Kontext nur 12–17 % beträgt und jede Attention-Optimierung bei
`1,20x` deckelt. Bei voller Attention über 32K läge der Anteil plausibel bei 40–60 %.

**Risiko:** mehrere GB Download.

**Benötigte Aktion:** ausdrückliche Download-Freigabe.

# Forschungsprotokoll — Hardware-Aware Inference Runtime

Append-only. Jeder Zyklus, auch die negativen.

---

## Zyklus 1 — 24.08.2026

**Ausgangszustand.** Repo sauber auf `8dd934d`. Python `3.12.13` arm64,
macOS `26.5.2`, MLX `0.32.0`, mlx-lm `0.31.3`, Metal verfügbar. `torch`,
`transformers`, `sentencepiece` vorhanden; **vLLM und llama.cpp nicht** → nach
`PERMISSION_REQUIRED.md`, Loop fortgesetzt.

**Vorhandene Vorarbeit wurde nicht verworfen.** `friday_hardware` (Profil,
Breiten-Policy, Segmentierung, kontextbasierte Spekulation), zwei gemessene Profile,
zehn Messwerkzeuge, 627 Tests. Diese Arbeit deckt **Decode** ab. Zyklus 1 richtet
sich deshalb auf das, was sie nicht abdeckt: TTFT, Prefill, Präfix-Wiederverwendung.

### Schritt 2 — Pfadanalyse

| Komponente | Messwert | Fundstelle |
| :--- | ---: | :--- |
| Chat-Template + Tokenisierung | `0,044`–`0,649` ms | vernachlässigbar |
| Modellladen | `1,47`–`1,76` s | einmalig je Prozess |
| Detokenizer | je Token | `mlx_lm/generate.py:724` |
| Prefill-Schrittweite | `2048` bzw. `512` | `generate.py:316` / `:483` |
| `save_prompt_cache` / `load_prompt_cache` | vorhanden | `models/cache.py:43` / `:62` |

Tokenisierung liegt unter `0,1 %` der TTFT und ist **kein** Engpass. Die
Prefill-Schrittweite ist zwischen den beiden Codepfaden inkonsistent.

### Schritt 3/4 — TTFT nach Klassen, Engpass

Erster Versuch war methodisch falsch: eine reine System-Message als Präfix. **Gemma 3
hat keine eigene System-Rolle** — der Inhalt wird verworfen (`<bos>` allein) und bei
vorhandenem User-Turn in diesen hineingemischt. Das gemessene „Präfix" betrug `1`
Token. Korrigiert auf das gemeinsame **Token-Präfix** der gerenderten Prompts.

| Klasse | Wert |
| :--- | ---: |
| Modellladen (`cold_process`-Anteil) | `1,487` s |
| `warm_uncached` TTFT, `898`-Token-Prompt | `1702,86` ms |
| `warm_prefix_hit`, `886` Token wiederverwendet | `131,02` ms |
| **Verhältnis** | **`13,0x`** |

Engpass damit eindeutig: **Prompt Prefill**, nicht Decode, nicht Tokenisierung,
nicht Kernel.

### Schritt 5 — Kandidat: exakte Präfix- und KV-Cache-Wiederverwendung

Priorität 4 der Liste. Höchste erwartete Wirkung, Maschinerie vorhanden.

### Korrektheitsgate — gescheitert

Tokenidentität gegen den frischen Pfad, vier Präfixlängen:

| Präfix | Prompt | identisch | erste Abweichung |
| ---: | ---: | :--- | ---: |
| `666` | `677` | **nein** | 10 |
| `1326` | `1337` | **nein** | 10 |
| `2646` | `2657` | ja | – |
| `4406` | `4417` | **nein** | 20 |

Kein Muster des rotierenden Fensters (`1024`): `666` liegt darunter und weicht bereits
ab, `2646` liegt darüber und ist identisch.

**Ursache isoliert, ohne Präfix-Cache** — dieselben Token, nur anders gestückelt:

| Zerteilung | Blöcke | identisch | erste Abweichung |
| :--- | ---: | :--- | ---: |
| ein Block (`677`) | 1 | Referenz | – |
| `512`+`165` | 2 | **nein** | 10 |
| `666`+`11` | 2 | ja | – |
| `256`er | 3 | ja | – |
| `128`er | 6 | **nein** | 10 |

**Allein die Blockgröße des Prefills verändert die erzeugten Token.** Nicht die Anzahl
der Blöcke — drei Blöcke sind identisch, zwei nicht. Es sind bestimmte Breiten: `512`
und `128` weichen ab, `256` und `665` nicht. Das deckt sich mit dem früheren Befund,
dass MLX den quantisierten Matmul nach Breite auswählt; verschiedene Kernel summieren
in verschiedener Reihenfolge.

**Entscheid:** `candidate_correctness_failed` für die naive Umsetzung. Der Kandidat
wird **nicht** wiederholt und **nicht** durch Lockerung des Kriteriums gerettet. Die
`13,0x` bleiben als charakterisierende Beobachtung stehen und sind **kein** Gewinn,
solange die Ausgabe eine andere ist.

**Abgeleiteter Folgekandidat** in `EXPERIMENT_BACKLOG.md`: eine Blockgrößen-Policy,
die Tokenidentität erhält. Ob eine solche Größe längenunabhängig existiert, ist offen.

**Alles aus diesem Zyklus:** `formal_claim=false`.

---

## Zyklus 2 — 24.08.2026

**Kandidat:** `chunk-identity-20260824-01`. Vorregistrierung vor der Messung
geschrieben: `experiments/chunk_identity/PREREGISTRATION.md`.

**Frage.** Existiert eine Prefill-Blockgröße, unter der ein zerteiltes Prefill
tokenidentisch zum Einzelblock bleibt? Vier Kandidaten der Liste hängen daran.

### Phase A — Matrix

Vier Promptlängen × fünf Blockgrößen, `16` Ausgabetoken, greedy:

| Prompt | 64 | 128 | 256 | 512 | 1024 |
| ---: | :--- | :--- | :--- | :--- | :--- |
| `303` | ✓ | ✓ | ✓ | – | – |
| `677` | ✓ | **✗** | ✓ | **✗** | – |
| `1205` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `1997` | **✗** | ✓ | ✓ | **✗** | ✓ |

Die Fehlschläge sind **sporadisch, nicht systematisch**: `64` scheitert nur bei
`1997`, `128` nur bei `677`. Kein Muster nach Blockanzahl, Zweierpotenz oder
Fenstergrenze.

`256` hielt 4/4 — aber bei einer Ausfallrate von rund `29 %` je Zelle hat eine von
fünf Blockgrößen mit Wahrscheinlichkeit `0,71⁴ ≈ 25 %` zufällig vier Treffer. Das
allein rechtfertigt keine Regel.

### Phase B — Bestätigung

Sechs weitere Längen für `256` und `1024`. Der Lauf endete vorzeitig: eine
**Einzelblock-Referenz über rund `2600` Token überschreitet das `6`-s-Continuous-Limit**.
Das ist eine echte Messgrenze der Policy, kein Fehler. Teilergebnis persistiert, kein
Retry im selben Prozess.

| Blockgröße | identisch | Längen |
| ---: | ---: | :--- |
| **`256`** | **`7/8`** | `303`–`2503`, **Fehlschlag bei `1513`** |
| `1024` | `4/4` | `1205`–`2503` |

**`256` scheitert bei `1513` Token.** Der 4/4-Befund aus Phase A war Glück, wie
vorab vermutet.

### Entscheid

Über beide Phasen: **`6` Fehlschläge auf `23` Zellen, rund `26 %`**.

`candidate_correctness_failed`. Das ist der in der Vorregistrierung als Ausgang 2
festgelegte Fall: **keine Blockgröße erhält die Tokenidentität zuverlässig.** Die
Deutung wird nicht nachträglich geändert.

### Folgen

Vier Kandidaten bleiben blockiert, weil alle die Blockstruktur verändern:
Präfix-Wiederverwendung, Prefill-Step-Size-Sweep, Microbatching, Continuous Batching.

Der Korrektheitsvertrag des Auftrags — identische Token-IDs — ist damit für jede
Optimierung, die die Prefill-Zerteilung ändert, auf dieser Plattform **nicht
erfüllbar**. Das ist eine Entscheidung, die dem Nutzer gehört, nicht eine, die ich
durch Aufweichen des Kriteriums treffe.

Übrig bleiben nur Kandidaten, die die Numerik gar nicht berühren: persistenter
Modellprozess und deterministischer Warm-up.

**`formal_claim=false`.**

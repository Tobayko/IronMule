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

---

## Zyklus 3 — 24.08.2026

**Kandidat:** `persistent-process-20260824-01`. Vorregistrierung vorab:
`experiments/cold_start/PREREGISTRATION.md`. Dieser Zyklus **misst nur** und ändert
den Ausführungspfad nicht.

### Zerlegung des Kaltstarts

Drei frische Prozesse, Mediane, `897`-Token-Prompt:

| Anteil | s | Anteil an `cold_process`-TTFT |
| :--- | ---: | ---: |
| Interpreterstart | `0,025` | `0,5 %` |
| stdlib-Importe | `0,002` | `0,0 %` |
| `import mlx.core` | `0,005` | `0,1 %` |
| **`import mlx_lm`** | **`1,881`** | **`36,9 %`** |
| Snapshot-Auflösung | `0,009` | `0,2 %` |
| **Modellladen** | **`1,410`** | `27,7 %` |
| Warm-up, erster Forward | `0,050` | `1,0 %` |
| **Prefill bis erstes Token** | **`1,712`** | `33,6 %` |
| **Summe** | **`5,094`** | |

Zweiter Forward: `0,016` s. Der Warm-up-Aufwand beträgt also rund `0,034` s —
klein, aber real. RSS nach dem Laden: `3,77` GB.

### Hypothese widerlegt

Vorhergesagt war, das **Modellladen** dominiere den Kaltstart. Tatsächlich kostet
`import mlx_lm` mit `1,881` s **mehr** als das Laden des Modells mit `1,410` s.

Ursache über `python -X importtime` eindeutig:

```
mlx_lm 1,89 s
 └ mlx_lm.convert
    └ mlx_lm.utils
       └ mlx_lm.tokenizer_utils
          └ transformers
             └ torch  (0,52 s, davon torch.distributed.fsdp 0,27 s)
```

`mlx_lm/__init__.py` importiert `convert` unbedingt, und über diese Kette wird die
vollständige PyTorch-Distributed-Trainingsmaschinerie geladen — in einem Projekt, das
Torch nie benutzt.

### Versuchte Abhilfe — negativ

`transformers` wertet `USE_TORCH` aus. Gemessen, je drei Läufe:

| Konfiguration | Importzeit |
| :--- | :--- |
| normal | `1,887` / `1,896` / `1,873` s |
| `USE_TORCH=0` | `1,943` / `1,969` / `1,875` s |

**Kein Gewinn.** Die Variable verhindert den Torch-Import auf diesem Pfad nicht. Die
Importzeit ist von Anwendercode aus **nicht** behebbar; sie sitzt im
`__init__` des Pakets.

### Mein vorab festgelegtes Kriterium war zu eng gefasst

Die Vorregistrierung nannte als Schwelle: **Modellladen plus Warm-up ≥ `30 %`** der
`cold_process`-TTFT. Gemessen sind es `1,460` s von `5,094` s = **`28,7 %`** — knapp
**verfehlt**.

Das Kriterium bildet den Kandidaten allerdings falsch ab. Ein persistenter Prozess
entfernt **auch die Importe**, nicht nur Laden und Warm-up. Tatsächlich entfällt
alles außer dem Prefill: `3,382` s von `5,094` s = **`66,4 %`**.

Ich berichte beide Zahlen. Die Schwelle nachträglich auf die günstigere Größe
umzudeuten wäre genau das, was der Auftrag untersagt; sie war schlecht formuliert, und
das ist ein Mangel der Vorregistrierung, kein Ergebnis.

### Entscheid

`candidate_characterized`, gemäß der vorab festgelegten Abbruchbedingung: Importe
dominieren statt des Modellladens.

Der Kandidat bleibt trotzdem der aussichtsreichste verbliebene — nicht weil das Laden
teuer wäre, sondern weil die **nicht behebbare** Importzeit nur durch einen
persistenten Prozess vermeidbar ist. Eine belastbare Empfehlung verlangt eine neue
Vorregistrierung mit korrekt gefasster Schwelle.

**`formal_claim=false`.**

---

## Zyklus 4 — 24.08.2026

**Kandidat:** `divergence-impact-20260824-01`. Vorregistrierung vorab:
`experiments/divergence/PREREGISTRATION.md`.

**Warum.** Zyklus 2 legte dem Nutzer eine Vertragsentscheidung vor — Tokenidentität
halten und vier Kandidaten sperren, oder den Vertrag präzisieren. Diese Entscheidung
wurde **ohne Daten** vorgelegt: die einzige beobachtete Abweichung war ein Satzzeichen.
Dieser Zyklus beschafft die fehlende Zahl. Er schlägt **keine** Lockerung vor.

### Aufbau

`10` maschinell erzeugte Rechenaufgaben mit eindeutiger ganzzahliger Lösung, langer
identischer Vorspann auf `~1200` Prompt-Token, `160` Ausgabetoken, greedy.
Einzelblock gegen Blockgröße `512` — letztere wich in Zyklus 2 bei zwei von drei
Längen ab, erzeugt also zuverlässig Fälle.

### Ergebnis

| Größe | Wert |
| :--- | ---: |
| Token-Divergenzrate | **`0,70`** (`7` von `10`) |
| **Antwort gleich unter den Abweichenden** | **`0,286`** (`2` von `7`) |
| abweichende Token, Maximum | `153` von `160` |
| erste Abweichung, Median | Position `5` |

Einzelfall zur Illustration: Wahrheit `54`, Referenzarm antwortet `54`, der zerteilte
Arm antwortet `32`.

**Die Abweichungen sind nicht oberflächlich.** Die Hypothese dieses Zyklus ist
widerlegt. Es geht nicht um Formulierung, sondern um die Antwort.

Die höhere Divergenzrate gegenüber Zyklus 2 (`0,70` statt `0,26`) erklärt sich aus
längeren Prompts und zehnmal längerer Ausgabe: eine frühe Abweichung wirkt sich mit
zunehmender Länge weiter aus.

### Nicht zu lesen als Qualitätsaussage

Die Trefferquoten der beiden Arme lauteten `0,3` und `0,5`. Das ist **kein** Befund:
`n=10`, und die Trefferquote war nicht der Endpunkt. Der zerteilte Arm ist nicht
„besser"; er ist **anders**, und genau das ist das Problem.

### Entscheid

`candidate_characterized`. Die vorab festgelegte Deutung greift ohne Auslegung:

> Antwortzahl weicht häufig ab → **Vertrag halten; jede Lockerung ist ausgeschlossen.**

Damit ist die in Zyklus 2 offene Frage **beantwortet, und zwar mit Daten**. Die vier
blockierten Kandidaten — Präfix-Wiederverwendung (`13,0x` TTFT), Prefill-Step-Sweep,
Microbatching, Continuous Batching — bleiben auf dieser Plattform unter diesem Vertrag
dauerhaft gesperrt.

Das ist ein korrektes Nullergebnis und wiegt schwerer als ein nicht reproduzierbarer
Geschwindigkeitsgewinn.

### Folge für BW1

`docs/BW1_VORREGISTRIERUNG.md` bleibt unversiegelt. Sein Korrektheitsgate ist nun
dreifach belastet: formabhängige Numerik verändert die Ausgabe beim Prefill
(Zyklus 2), bei `mx.compile` (frühere Runde), und die Abweichungen ändern die
**Antwort**, nicht nur die Formulierung (dieser Zyklus). Ein Start der Studie ist nur
sinnvoll, wenn der wahrscheinliche Ausgang `bw1_correctness_failed` als Ergebnis
akzeptiert wird.

**`formal_claim=false`.**

---

## Zyklus 5 — 24.08.2026

**Kandidat:** `persistent-process-20260824-02`. Vorregistrierung vorab:
`experiments/persistent/PREREGISTRATION.md`. Ersetzt den Zyklus-3-Kandidaten mit
korrigierter Schwelle und einem Korrektheitsgate.

### H1 — Korrektheit: hält

Arm A: frischer Prozess, Prompt `P`. Arm B: ein Prozess, Reihenfolge `P Q R` mit
`P` dreimal. Frischer KV-Cache je Anfrage; Präfix-Wiederverwendung ausdrücklich aus.

| # | Anfrage | TTFT s | RSS GB | tokenidentisch zu kalt |
| ---: | :--- | ---: | ---: | :--- |
| 0 | `P` | `1,747` | `3,77` | **ja** |
| 1 | `Q` | `1,752` | `3,77` | – |
| 2 | `P` | `1,771` | `3,77` | **ja** |
| 3 | `R` | `1,747` | `3,77` | – |
| 4 | `P` | `1,755` | `3,77` | **ja** |

Drei von drei. Keine Zustandsverschleppung durch die eingeschobenen `Q` und `R`, RSS
über alle fünf Anfragen unverändert.

Das ist bemerkenswert vor dem Hintergrund der Zyklen 1, 2 und 4: dreimal veränderte
eine formabhängige Numerik die Ausgabe still. Hier ändert sich die Form nicht — nur,
wie oft der Prozess startet — und die Ausgabe bleibt bitgleich.

### H2 — Wirkung: hält

| Anteil | s |
| :--- | ---: |
| Importe (Zyklus 3, Median dreier frischer Prozesse) | `1,881` |
| Snapshot-Auflösung und Modellladen | `1,426` |
| Prefill bis erstes Token | `1,747` |
| **`cold_process` TTFT** | **`5,053`** |
| **warm TTFT** | **`1,747`** |
| **entfernt** | **`65,4 %`** |

Schwelle war `50 %`.

**Einschränkung, offen benannt:** Das Messskript hat die eigene Vorregistrierung
unvollständig umgesetzt — der Zeitzähler startete **nach** den Importen, obwohl die
Spezifikation sie ausdrücklich verlangt. Die fehlende Größe wurde aus Zyklus 3
übernommen, gemessen mit identischer Methode auf derselben Maschine. Der Gegencheck
stützt die Ersetzung: `5,053` s gegen dort unabhängig gemessene `5,094` s, `0,8 %`
Abweichung. Ein erneuter Lauf nach Sicht eines knappen Ergebnisses wäre genau das,
wovor diese Messreihe wiederholt gewarnt hat, und unterblieb deshalb.

### Entscheid

Nach der vorab festgelegten Tabelle (`H1` hält, `H2` hält):
**`candidate_recommended_for_preregistration`**.

Der erste Kandidat in fünf Zyklen, der ein Korrektheitsgate besteht. Er ist es genau
deshalb, weil er als einziger **nichts an der Numerik ändert** — keine Blockgröße,
keine Batchbreite, keine Cache-Struktur.

### Was das nicht ist

Keine Empfehlung zur Produktivaktivierung und kein `formal_claim`. Eine belastbare
Bestätigung verlangt eine eigene versiegelte Studie mit A/A-Gate und eingefrorener
MDE, wie N8, N10 und Phase 1B sie hatten.

Und es verschiebt den Engpass nur: nach Entfernen der `3,31` s bleibt der Prefill mit
`1,747` s stehen, und dessen wirksamste Optimierung ist die Präfix-Wiederverwendung —
in Zyklus 1 am Korrektheitsgate gescheitert und in Zyklus 4 endgültig gesperrt.

**`formal_claim=false`.**

---

## Zyklus 6 — 24.08.2026

**Kandidat:** `host-sync-20260824-01`. Vorregistrierung vorab:
`experiments/sync/PREREGISTRATION.md`.

**Warum.** Der Auftrag nennt „unnötige CPU-GPU-Synchronisationen" und „vollständige
Logit-Readbacks" unter Schritt 2. Beides war nach fünf Zyklen ungeprüft. Der heutige
Pfad liest jedes Token zum Host (`mlx_lm/generate.py:466`, `y.item()`); der eigene
Spekulationspfad ebenso (`friday_hardware/speculate.py:223`, `.tolist()`).

Der Kandidat war attraktiv, weil er als einziger verbliebener **die Numerik nicht
berührt** — gerechnet wird dasselbe, verschoben wird nur der Zeitpunkt des Lesens.

### Ergebnis

`128` Decode-Schritte nach `~900`-Token-Prefill, drei Wiederholungen, Median,
Arme abwechselnd:

| Arm | ms je Token | Verhältnis | tokenidentisch |
| :--- | ---: | ---: | :--- |
| `readback` (heutiger Pfad) | `14,3671` | `1,000` | ✓ |
| **`deferred`** | **`12,1683`** | **`0,847`** | ✓ |
| `eos_check` | `14,4429` | `1,0053` | ✓ |

**H1 hält:** alle drei Arme erzeugen dieselben `129` Token.
**H2 hält:** `15,3 %` Ersparnis, Schwelle war `3 %`.

Der `eos_check`-Arm liegt bei `1,0053` — innerhalb der Streuung von `readback`. Die
**Prüfung** des Stop-Tokens ist damit gratis; teuer ist allein das **Lesen** zum Host.

### Einschränkung, die den Befund begrenzt

Der `deferred`-Arm **kann nicht anhalten**. Er läuft eine feste Schrittzahl, weil er
nie erfährt, was erzeugt wurde. Ein echter Generator braucht die Stop-Token-Prüfung,
und die verlangt den Readback.

Die `15,3 %` sind also **nicht direkt abrufbar**. Sie sind die Obergrenze dessen, was
eine Bündelung der Lesevorgänge erreichen könnte: prüft man alle `N` Schritte statt
jeden, fällt der Readback nur noch `1/N`-mal an, um den Preis eines Überlaufs von
höchstens `N−1` Token über das Stop-Token hinaus. Dieselbe Abwägung wie beim
segmentierten Decode-Loop früherer Runden.

Erwartung bei `N=8`: rund `13,4 %`. **Nicht gemessen** und deshalb nicht behauptet.

### Entscheid

Nach der vorab festgelegten Tabelle (`H1` hält, `H2` hält):
**`candidate_recommended_for_preregistration`**.

Zweiter Kandidat in sechs Zyklen, der ein Korrektheitsgate besteht — und wieder einer,
der die Numerik unberührt lässt. Das Muster ist inzwischen deutlich: **jeder Kandidat,
der Formen verändert, scheitert; jeder, der nur Zeitpunkte verschiebt, besteht.**

### Folgekandidat

Gebündelter Readback mit Prüfintervall `N`, gemessen gegen `N=1`. Er ist die
tatsächlich abrufbare Variante dieses Befunds und der natürliche Kandidat für
Zyklus 7.

**`formal_claim=false`.**

---

## Zyklus 7 — 24.08.2026

**Kandidat:** `batched-readback-20260824-01`. Vorregistrierung vorab:
`experiments/sync/PREREGISTRATION_BATCHED.md`. Abrufbare Form des Zyklus-6-Befunds.

### Ergebnis

`128` Decode-Schritte, feste Schrittzahl, zwei Wiederholungen, Median:

| `N` | ms je Token | Ersparnis | tokenidentisch | Break-even Länge |
| ---: | ---: | ---: | :--- | ---: |
| 1 | `14,3132` | – | ✓ | – |
| 2 | `13,3041` | `7,05 %` | ✓ | `6,5` |
| 4 | `12,6777` | `11,43 %` | ✓ | `13,0` |
| **8** | `12,4554` | **`12,98 %`** | ✓ | `26,0` |
| 16 | `12,2222` | `14,61 %` | ✓ | `52,1` |
| 32 | `12,1334` | `15,23 %` | ✓ | `104,2` |

**H1 hält:** alle sechs Intervalle erzeugen identische Token.
**H2 hält:** `N=4` überschreitet die Schwelle von `8 %` bereits.

`N=32` erreicht `15,23 %` und damit praktisch die in Zyklus 6 gemessene Obergrenze
von `15,3 %`. Die Sättigung bestätigt das Modell: mehr als der Readback selbst ist
nicht zu holen.

### Die Break-even-Längen sind abgeleitet, nicht gemessen

Der Lauf hatte **feste** Schrittzahl ohne vorzeitiges Anhalten. Der Überlauf über das
Stop-Token fiel also nie an, und die Ersparnisspalte zeigt den Gewinn **ohne** seinen
Preis.

Die Break-even-Spalte korrigiert das rechnerisch: erwarteter Überlauf `(N−1)/2` Token
zur vollen Schrittzeit gegen `(1 − 1/N)` der gesparten Readback-Kosten. Sie ist als
Ableitung gekennzeichnet und wird nicht als Messwert geführt.

Praktisch heißt das: `N=8` lohnt ab rund `26` Ausgabetoken, `N=32` erst ab `104`.
Für eine typische Antwort von `100`–`300` Token ist `N=16` bis `32` richtig, für eine
Einzeilenantwort keines.

### Umgesetzt

`HardwareProfile.readback_interval(expected_tokens)` wählt das größte Intervall, das
sich bei der erwarteten Länge noch rechnet, und begründet die Wahl. Die beiden
Kostengrößen (`readback_ms_per_step`, `step_ms`) stehen im Profil, damit die Wahl eine
Rechnung ist und keine Konvention — sie sind gerätespezifisch wie Breite, Fenster und
Trefferschwellen zuvor. Fünf neue Tests.

### Entscheid

Nach der vorab festgelegten Tabelle: **`candidate_recommended_for_preregistration`**.

Dritter Kandidat in sieben Zyklen, und der erste, dessen Gewinn **direkt abrufbar**
ist — Zyklus 5 verlangt eine Architekturänderung, Zyklus 6 konnte nicht anhalten,
dieser hier läuft.

**`formal_claim=false`.**

---

## Zyklus 8 — 24.08.2026

**Kandidat:** `prefill-head-skip-20260824-01`. Vorregistrierung vorab:
`experiments/head_skip/PREREGISTRATION.md`.

**Wie er gefunden wurde.** Beim erneuten Durchgehen der Kandidatenliste des Auftrags
fiel auf, dass Punkt 6 — „Entfernung unnötiger Kopien" — nach sieben Zyklen ungeprüft
war. `gemma3_text.Model.__call__` wendet `lm_head` auf **alle** Positionen an:

```python
out = self.model(inputs, cache, input_embeddings)
out = self.lm_head(out)
```

Beim Prefill wird davon genau eine Zeile gelesen. Bei einem `256`-Token-Block und
`262208` Vokabular sind das rund `172` GFLOP, von denen `1/256` verwendet wird.

### Ergebnis

`~900`-Token-Prompt, zwei Wiederholungen, Median:

| Blockgröße | Head auf allen | Head nur letzte | Anteil am Prefill | tokenidentisch |
| ---: | ---: | ---: | ---: | :--- |
| 128 | `1,7624` s | `1,5018` s | `14,79 %` | ✓ |
| 256 | `1,7614` s | `1,4788` s | **`16,05 %`** | ✓ |
| 512 | `1,7050` s | `1,4450` s | `15,25 %` | ✓ |

**H1 hält:** identische Token bei allen drei Blockgrößen.
**H2 hält:** `14,8`–`16,1 %`, Schwelle war `10 %`.

Der Anteil ist über die Blockgrößen stabil, was zum Mechanismus passt: der Head
skaliert linear mit der Positionszahl, und die ist unabhängig von der Zerteilung.

### Warum die Korrektheit hier hält, anders als in den Zyklen 1, 2 und 4

Übersprungen werden Logits, die **nie gelesen** werden. Es ändert sich keine Form
einer Rechnung, deren Ergebnis verwendet wird — im Unterschied zu Blockgröße,
Batchbreite und Graphform, an denen dreimal die Ausgabe still kippte. Geprüft wurde
es trotzdem, und zwar vorab als Gate.

### Bedeutung

Erster Kandidat dieser Reihe, der die **TTFT** verbessert — und die Zyklen 1 bis 4
haben gezeigt, dass genau dort der Engpass sitzt. Der Prefill sinkt von `1,76` s auf
`1,48` s, also rund `1,18x`.

Zum Vergleich: die Präfix-Wiederverwendung hätte `13,0x` gebracht und ist an der
Korrektheit gescheitert. Dieser Gewinn ist um eine Größenordnung kleiner, aber
**abrufbar**.

### Grenze, vorab benannt und bestätigt

Zulässig nur, solange niemand die übersprungenen Logits braucht — also greedy
Decoding ohne Logprob-Ausgabe je Prompt-Token. Für Perplexität, Bewertung oder
Logprob-Rückgabe ist der Kandidat **nicht** anwendbar. Diese Grenze gehört in jede
Umsetzung.

### Entscheid

**`candidate_recommended_for_preregistration`.** Vierter in acht Zyklen.

Das Muster hält auch hier: verändert wird kein Ergebnis, das gelesen wird — nur
Arbeit, die ohnehin verworfen wurde, entfällt.

**`formal_claim=false`.**

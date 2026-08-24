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

---

## Zyklus 9 — 24.08.2026

**Kandidat:** `divergence-source-20260824-01`. Vorregistrierung vorab:
`experiments/divergence_source/PREREGISTRATION.md`. Diagnose, keine Optimierung.

**Warum.** Zyklus 2 zeigte die Zerteilungs-Divergenz, Zyklus 4 ihre Wirkung auf die
Antwort. Keiner nannte die Ursache; „sporadisch" ist eine Beschreibung. Der Auftrag
verlangt für Kerneloptimierung den vorherigen Nachweis eines Kernelengpasses — diese
Diagnose liefert ihn oder schließt ihn aus.

### Ergebnis

`677` Token, ein Block gegen `512`+`165`, KV-Caches schichtweise verglichen:

| Schicht | keys relativ | values relativ | keys absolut |
| ---: | ---: | ---: | ---: |
| 0 | `4,88e-03` | `7,04e-03` | `0,2500` |
| 5 | `1,08e-02` | `4,46e-03` | `0,3750` |
| 11 | `9,16e-02` | `8,14e-04` | `2,7812` |
| **17** | **`1,98e-01`** | `1,61e-02` | `11,0625` |
| 23 | `1,60e-01` | `1,13e-02` | `8,9297` |
| 33 | `7,36e-02` | `1,00e-01` | `2,5195` |

**Erste abweichende Schicht: `0`.** Nicht eine bestimmte Operation — von Anfang an.

Der KV-Cache liegt in **`bfloat16`**: 8 Mantissenbits, relative Auflösung
`2⁻⁸ = 0,00391`. Die Abweichung bei Schicht `0` beträgt `4,88e-03`, also **ein bis
zwei ULP**. Über die Schichten verstärkt sie sich auf `1,98e-01` — rund das
Vierzigfache.

### Die Kette, geschlossen

1. Verschiedene Breiten wählen verschiedene Kernelpfade, die in verschiedener
   Reihenfolge summieren.
2. In `bfloat16` erzeugt das sofort einen Unterschied von einem ULP.
3. Über `34` Schichten verstärkt er sich um rund das Vierzigfache.
4. An den Logits der letzten Position beträgt er `1,1875`.
5. Der Abstand zwischen Top-1 und Top-2 lag hier bei `1,75`, in Zyklus 1 bei `0,344`.
   Ist der Abstand kleiner als das Rauschen, kippt das Token.

Damit ist das „Sporadische" erklärt: es ist ein Wettlauf zwischen der Verteilung der
Logit-Abstände und dem akkumulierten Rauschen. Ob er kippt, hängt vom Inhalt ab, nicht
von einer Fehlfunktion.

Im gemessenen Fall kippte er **nicht** (`1,1875 < 1,75`) — die Prüfung
`difference_can_flip_choice` steht auf `False`. Bei Zyklus 1 mit einem Abstand von
`0,344` kippte er.

### Was daraus folgt

Nach der vorab festgelegten Deutung: **Unterschied ab Schicht `0` überall →
Eingangsverarbeitung, kein einzelner Kernel als Ziel.**

Damit ist Punkt 15 der Kandidatenliste — **Custom Metal Kernel** — nicht nur mangels
Beleg gesperrt, sondern **begründet ausgeschlossen**. Es gibt keinen Hotspot, der die
Divergenz verursacht; die Ursache ist Präzision, verteilt über jede Schicht.

Und es schließt eine ganze Klasse: **keine reihenfolgetolerante Implementierung kann
das in `bfloat16` beheben.** Abhilfe verlangte höhere Akkumulationspräzision, also
eine Framework- oder Modelländerung — außerhalb des Auftrags.

### Entscheid

`candidate_characterized`. Eine Diagnose ist kein Kandidat; sie **begründet** oder
**schließt** welche. Hier schließt sie.

**`formal_claim=false`.**

---

## Zyklus 10 — 24.08.2026

**Kandidat:** `logsumexp-skip-20260824-01`. Vorregistrierung vorab:
`experiments/logsumexp/PREREGISTRATION.md`.

**Beobachtung.** `mlx_lm/generate.py`, `generate_step._step` rechnet unbedingt
`logprobs = logits - mx.logsumexp(logits, keepdims=True)`, und `make_sampler(temp=0)`
ist danach ein reines `mx.argmax`. Argmax ist gegenüber dem Abzug einer Konstante
invariant — bei greedy ohne Logprob-Ausgabe ist die Normalisierung wirkungslos.
Gleiche Form wie Zyklus 8.

### Ein Messfehler zuerst

Der erste Lauf meldete `107,57` und `76,70` ms je Token, also das Achtfache des
Erwarteten, und daraus `28,7 %` Ersparnis. Ursache: die Laufzeit wurde **nach**
`charge()` festgehalten, und `charge()` schläft die Guard-Pausen. Gemessen wurde damit
die Pausenverteilung.

Der Fehler wurde behoben und der Lauf wiederholt. Das ist zulässig und nicht das, wovor
diese Reihe warnt: das fehlerhafte Ergebnis war **günstiger** als das korrekte, die
Korrektur also gegen das eigene Interesse.

### Ergebnis

`128` Decode-Schritte, drei Wiederholungen, Median:

| Arm | ms je Token |
| :--- | ---: |
| mit Normalisierung | `14,3003` |
| ohne Normalisierung | `14,5226` |
| **Ersparnis** | **`−1,56 %`** |

| Größe | Wert |
| :--- | ---: |
| logsumexp **isoliert** gemessen | `0,4086` ms |
| davon als Anteil eines Schritts | `2,86 %` |
| Tokenidentität | ✓ |

**H1 hält. H2 verfehlt.**

### Der eigentliche Befund

Die Operation kostet **isoliert** `0,41` ms und im Loop **nichts**. MLX überlappt sie
mit dem Modell-Forward, der GPU-gebunden ist; sie füllt Leerlauf, den es ohnehin gibt.
Die `−1,56 %` liegen innerhalb der Streuung, die frühere Zyklen mit `0,4`–`1,8 %`
gemessen haben.

Das ist dieselbe Falle wie bei einem früheren Matmul-Mikrobenchmark: **isolierte
Kosten sind nicht Grenzkosten**, sobald die Pipeline Spielraum hat. Wer nur isoliert
misst, findet Optimierungen, die es nicht gibt.

Die Vorregistrierung hatte genau diesen Ausgang als wahrscheinlich benannt
(`unter 0,1 %` überschlägig). Die isolierte Messung fiel mit `2,86 %` höher aus als
der Überschlag — und blieb im Loop trotzdem wirkungslos.

### Entscheid

Nach der vorab festgelegten Tabelle (`H1` hält, `H2` verfehlt):
**`candidate_characterized`** — korrekt, aber unter der Messschwelle.

Ein korrektes Nullergebnis. Der Auftrag wertet es höher als einen nicht
reproduzierbaren Gewinn, und hier ist es zusätzlich nützlich: es benennt eine
Messfalle, in die spätere Zyklen sonst laufen würden.

**`formal_claim=false`.**

---

## Zyklus 11 — 24.08.2026

**Kandidat:** `kv-cache-realloc-20260824-01`. Vorregistrierung vorab und vor jeder
Hardwaredatei versiegelt: `experiments/kv_realloc/PREREGISTRATION.md`. Der Lauf
erfolgte einmalig und unverändert über `BudgetGuard` bei Netzbetrieb und Duty-Faktor
`0,15`; Exit `0`. Acht Wiederholungen nach einem verworfenen Aufwärmlauf, keine
Ausreißer verworfen.

### Lokalisierung

`765` Prompt-Token, Prefill-Blöcke zu `256`, `48` Decodeschritte. Die Cacheformen
änderten sich in allen acht Wiederholungen an exakt den vorab erwarteten Schritten:

| Schritt | Cacheklasse | Layer | Überschuss gegen Median ohne Reallokation | vorab gerechnet |
| ---: | :--- | ---: | ---: | ---: |
| `1` | `RotatingKVCache` | `29` | **`31,5853` ms** | `0,7616` ms |
| `4` | `KVCache` | `5` | `0,2968` ms | `0,1317` ms |

Median der Schritte ohne Reallokation: `14,2671` ms. Der mittlere Überschuss der
beiden beobachteten Reallokationsschritte beträgt abgeleitet `15,9411` ms; ihre
Summe `31,8821` ms beziehungsweise **`4,4263 %`** der Decodezeit.

Die Vorhersage von `0,13 %` wurde klar widerlegt. Sie war aus Cachebreite und
effektiver Bandbreite gerechnet, nicht gemessen. Insbesondere der erste
Decodeschritt enthält deutlich mehr als die reine, so berechnete Kopierzeit.

### Inter-Token-Latenz

| Quantil | ms |
| :--- | ---: |
| p50 | `14,2670` |
| p95 | `15,1385` |
| p99 | `46,7879` |
| min / max | `13,8230` / `49,4430` |

Der p99 wird durch das Ereignis am ersten Decodeschritt getragen. Das mediane
Rauschband je Schritt betrug `0,5566` ms, das maximale `30,5072` ms. Diese Werte
bleiben vollständig im Ergebnis; kein Lauf und kein Schritt wurde entfernt.

### Hypothesen und Entscheid

**H3 hält:** alle acht Wiederholungen erzeugten identische Token-IDs.

**H1 hält auf dem vorregistrierten Gruppenendpunkt:** Reallokationsschritte liegen
im Mittel `15,9411` ms über dem Median der übrigen Schritte; das vorab als Grundlage
der `0,30-ms`-Schwelle benannte große Ereignis liegt `31,5853` ms darüber. Das kleine
Ereignis allein liegt mit `0,2968` ms um `0,0032` ms unter der Schwelle. Diese
Randlage wird berichtet, ohne die Schwelle zu ändern.

**H2 hält:** `4,4263 %` sind größer als die vorregistrierten `1 %`.

Nach der vorab festgelegten Tabellenzeile (`H3` hält, `H1` hält, `H2` hält):
**`candidate_recommended_for_preregistration`**.

### Grenze des Befunds

Schritt `1` ist zugleich der erste Decodeschritt. Die Beobachtungsstudie kann dessen
sonstige einmalige Grenzkosten nicht von der `RotatingKVCache`-Reallokation trennen.
Die `4,4263 %` sind deshalb **kein behaupteter Optimierungsgewinn**. Eine versiegelte
A/B-Studie müsste die Cacheallokation ändern, Tokenidentität erneut gaten und den
kausalen Gewinn messen. Das wäre ein Framework-/Architektureingriff und ist hier
nicht erfolgt.

Guard-Bilanz: `21,086457` s GPU-Arbeit, maximal `1,270024` s kontinuierlich,
`116,119931` s Pflichtpausen, `139,595394` s Wall-Zeit.

**`formal_claim=false`.**

---

## Zyklus 12 — 24.08.2026

**Kandidat:** `prefill-head-skip-20260824-02`; Studie
`head-skip-prefill-v1-20260824`. Genau ein Kandidat wurde geprüft: Beim greedy
Prefill ohne Prompt-Logprobs projiziert der Kandidatenarm nur die tatsächlich
gelesene letzte Promptposition durch den LM-Head.

### Prospektive Versiegelung

Präregistrierung und Harness wurden vor der ersten Hardwaredatei auf dem sauberen
Commit `9466bb9f9f01813bcbd86b6d16837e90ad2523da` versiegelt. Gebunden waren:

- lokaler Modell-Snapshot `mlx-community/gemma-3-4b-it-4bit` auf Revision
  `93724907d4ed1745d2fe50baadf3b0b01a65abf2`;
- Dokument-SHA `8f7a9a854639824d337aa9ff3ef97ae2255c804291577c5021af2e93abbbeec6`,
  Script-SHA `b39bd6be0768173d293647d45cc7f0d3b1c469fd234375c8f0d46ce3c227dc14`,
  versiegelter Präregistrierungs-Payload
  `175a7238520d2a01a5c1c24898ff34773eb1b7a1cbbd6324b988d11fe8bc9cc6`;
- `897` Prompt-Token, Prefill-Chunk `256`, Batch `1`, `32` greedy
  Correctness-Token;
- sechs A/A- und danach sechs A/B-Sessionprozesse, je zwei Warmup- und vier
  Messpaare, balancierte Reihenfolge;
- getrennte Charakterisierung C und Validierung V, jeweils vorab festgelegter
  hierarchischer Bootstrap mit `10.000` Ziehungen;
- MDE `max(5 %, 2 × Session-SD × sqrt(2/3))`, gedeckelt bei `15 %`, sowie die
  unveränderliche Entscheidungstabelle.

Das Harness sperrte Ausführung ohne `--execute`, verwendete eine eigene
append-only SQLite-Hashkette und führte jeden Hardwarelauf in einem frischen Prozess
über `BudgetGuard` bei Netzbetrieb, Duty-Policy `0,15` und Pacing-Ziel `0,14` aus.
Es gab keinen Retry eines Hardwareprozesses.

### A/A-Kalibrierung

Die sechs gemessenen Sessionquotienten waren:

`[0,998498; 1,004692; 0,994007; 1,005769; 1,004463; 1,001198]`.

Die vorregistrierte Auswertung ergab `R=1,002829`, 95-%-KI
`[0,994931; 1,005964]` und Session-SD `0,004526`. Die daraus gerechnete rohe MDE
war `0,7391 %`; gemäß Vertrag wurde der konservative Boden `5 %` für die
Bestätigungsphase eingefroren. Alle A/A-, Correctness-, Budget- und
Provenienzgates bestanden. Der anschließende Confirmation-Seal band den Payload
`2571670a87fc5bd536d4ccee40d4c889afa30c37e65110e18f70607fd6caf11e`.

### A/B-Bestätigung

Alle sechs einmalig ausgeführten Sessions waren erfolgreich:

| Session | Quotient Kandidat/Baseline |
| :--- | ---: |
| C0 | `0,845257` |
| V0 | `0,846173` |
| C1 | `0,843401` |
| V1 | `0,847653` |
| C2 | `0,846596` |
| V2 | `0,852478` |

Die versiegelte Entscheidungsrechnung ergab:

| Split | Quotient | 95-%-KI |
| :--- | ---: | :--- |
| Charakterisierung | `0,845257` | `[0,840544; 0,848452]` |
| Validierung | `0,847653` | `[0,842683; 0,854941]` |
| Gesamt | **`0,846385`** | **`[0,843147; 0,851284]`** |

Alle drei oberen Intervallgrenzen liegen unter dem vorab eingefrorenen Gain-Gate
`0,95`. Der daraus gerechnete Effekt ist **`−15,3615 %`**. Alle zwölf Sessiongates
meldeten identische greedy Token-IDs; der gemeinsame Token-SHA ist
`666dcfb103d263a12b29ed9a1c1ec496c6922f96c3a6e7cec083eab47fb5127c`.
Tokenmismatches, verworfene Ausreißer und Schwellenänderungen gab es nicht.

Aus den gemessenen A/B-Sessionmedianen abgeleitet lagen die Armmediane bei
`1995,444239` ms und `1688,116333` ms. Beide Arme meldeten jeweils
`3.213.903.666` Byte MLX-Peak; das Prozess-RSS lag zwischen `3.768.795.136` und
`3.769.696.256` Byte. Über alle zwölf Sessions wurden `332,277940` s GPU-Arbeit,
`3.077,978881` s Guard-Pausen und `3.430,234516` s Session-Wall-Zeit summiert.
Diese Summen sind Rechnungen aus den gespeicherten Sessionrecords, keine neuen
Messläufe.

### Terminaler Entscheid und Evidenz

Nach der unveränderten Tabellenzeile lautet der Status
**`head_skip_gain_confirmed`**, Aktion
**`permit_bounded_architecture_review`**, `formal_claim=true`. Der formale Claim
gilt ausschließlich für **ein Gerät, einen Modell-Snapshot, einen Prompt, einen
Prefill-Plan und greedy ohne Prompt-Logprobs**. Er aktiviert keinen Produktpfad und
belegt weder allgemeine TTFT-Wirkung noch andere Promptlängen, Modelle oder Geräte.

Der Decision-Payload hat SHA-256
`99820747b874dfdfa72a2d65abbb1d9644a20cca3bd816d9058f4374aeb7428a`.
`.friday-data/head-skip-v1.sqlite3` enthält `16` hashverkettete Records, genau einen
formalen Claim, Modus `0600`, Größe `77.824` Byte, Datei-SHA
`15ee462bbad5a8f757373f093fdf2ccfb8bdd0048c03447c1cb635acd38ec8d9` und
Kettenkopf `8a568e61f0e087794b1997f273e580c72e7f5abaa1eb8bad7954b303dd38a2d4`.
Read-only Replay und reale GET-Abfrage der Historien-UI bestanden; der DB-Hash blieb
unverändert.

### Fehler, Ursachen und Lösungen

- Vor Versiegelung und vor Hardware schlug der erste Offline-Selbsttest fehl, weil
  `canonical_json` aus `friday_h1.canonical` statt aus
  `friday_evidence.canonical` importiert wurde. Der Import wurde korrigiert, alle
  Offline-Tests wurden erneut bestanden und erst danach sauber committed und
  versiegelt. Es entstand weder DB- noch Hardwareevidenz aus dem Fehler.
- Ein UI-Probeversuch verwendete `HEAD` und erhielt `501`, weil der neue read-only
  Server nur `GET` implementiert. Die korrekte GET-Abfrage lieferte `200` und den
  realen Verlauf. Dies war ein Diagnosefehler, kein Studien- oder DB-Fehler.
- Das manuelle `Ctrl-C` beendete den UI-Prozess mit sichtbarem
  `KeyboardInterrupt`/Exit `1`. Die UI war bereits read-only verifiziert; der
  versiegelte Code wurde nach dem terminalen Studienentscheid nicht verändert.

Es wurde nichts installiert, nichts heruntergeladen, keine versiegelte Spezifikation
oder Evidenz-DB verändert, kein Kandidat automatisch integriert und nichts gepusht.

### Abschlussverifikation

Nach der Dokumentation erreichte `.venv/bin/python -m pytest -q` `100 %` und Exit
`0` (äußere Wall-Zeit `41,86` s). `xcodebuild -checkFirstLaunchStatus` endete mit
Exit `0`. MLX `0.32.0` und mlx-lm `0.31.3` meldeten auf `arm64` das Standardgerät
`Device(gpu, 0)` und `Apple M1 Max`. ProjectAtlas wurde inkrementell aktualisiert
(`7` geänderte Symbolquellen, kein Timeout) und bestätigte Runtime `0.4.5-rc1`;
die projektlokale MCP-JSON-Datei war gültig. Der abschließende read-only Replay
bestätigte erneut `16` Records, genau einen formalen Claim und denselben DB-Hash vor
und nach der Prüfung. Dokument- und Script-SHA der versiegelten Studie blieben
bytegleich; beide Ergebnis-JSON-Dateien und `git diff --check` waren fehlerfrei.

---

## Freigegebener Runtime-Einbau — 24.08.2026, vor Live-Qualifikation

Der Nutzer hat die begrenzte Integration des formal bestätigten Head-Skips
freigegeben. Dies ist kein neuer Kandidatenzyklus: Der formale Einzelworkload-Claim
bleibt unverändert, und die neue Qualifikation trägt durchgehend
`formal_claim=false`.

Vor jeder Runtime-Messung wurden ein enger Architekturvertrag und eine
Mini-Vorregistrierung eingefroren. Der schnelle Pfad darf nur beim exakt
bestätigten lokalen Modell-Snapshot, Prompt, `897` Prompt-Token, Chunk `256`, Batch
`1`, greedy ohne Prompt-Logprobs und fester Ausgabe von `32` Token laufen. Alle
anderen oder unklaren Fälle wählen den Referenzpfad. Der getrennte Controller,
MLX-Adapter, Circuit Breaker, die private hashverkettete Historie und die read-only
Loopback-UI sind offline implementiert.

Gemessen wurde bislang nur die unveränderte Testbaseline vor dem Einbau:
`4,44 s` außen, `50.380.800 B` maximales RSS, keine Swaps. Nach dem Einbau bestanden
`20` fokussierte Offline-Tests; das ist ein Korrektheits- und Schutzbefund, kein
Geschwindigkeitsnachweis. Ein read-only Policy-Aufruf im absichtlich schmutzigen
Arbeitsstand fiel korrekt auf `worktree_dirty` zurück. Es gab noch keinen neuen
CPU-Overhead-Messrecord, keinen GPU-Lauf und keine Runtime-Evidenzdatei.

Eine zusätzliche Prüfung vor dem Live-Lauf schloss drei Lücken: Der schnelle Pfad
wird erst nach den bestandenen CPU- und GPU-Gates freigegeben, eine vor dem
Modellladen gespeicherte Startmarke verhindert einen zweiten Hardwareversuch, und
unbekannter Swap-Verbrauch zählt nicht mehr als bestanden. Die vollständige
Projekttestsammlung bestand mit Exit `0` in `38,50 s` ohne Swaps.

Nächste vorregistrierte Reihenfolge: vollständige Sicherheits- und Testsuite,
sauberer lokaler Commit, genau ein CPU-Policy-Lauf und nur bei bestandenem Gate
genau ein kontrollierter MLX/GPU-Prozess. Ein fehlgeschlagener Hardwareprozess wird
nicht wiederholt.

Die lokale Quellprüfung wurde anschließend vollständig versiegelt: `12/12`
Quellzeilen mit vollständigem Lesebeleg, Abdeckung `complete`, keine offene Arbeit
und `0` berichtspflichtige Sicherheitsbefunde. Der versiegelte Bericht liegt unter
`/private/tmp/codex-security-scans/Project_Friday/c7db74f_20260824T084223Z`.

---

## Head-Skip-Runtime-Qualifikation abgeschlossen — 24.08.2026

Die vorab festgelegte CPU-Prüfung wurde genau einmal ausgeführt und bestand. Die
zusätzliche Entscheidung kostete im Median `839,47295 ns` pro Aufruf; die
vorregistrierte Obergrenze wurde damit eingehalten. Netzbetrieb, BudgetGuard und
Duty-Faktor `0,15` wurden auch für diesen Lauf protokolliert.

Danach lief genau ein GPU-Qualifikationsprozess; er wurde nicht wiederholt. Der
Median der vier gepaarten Verhältnisse betrug `0,8458362744682114`. Der
Referenzpfad benötigte im Median `1.806.461.854,5 ns`, der begrenzte schnelle Pfad
`1.528.206.979,0 ns`. Daraus ergibt sich für den gemessenen Prefill eine
Verbesserung um `15,416372553178858 %`. Alle `32` greedy erzeugten Token waren in
jedem Korrektheits- und Messpaar exakt identisch. Der Pfadnachweis bestätigte vier
Transformer-Blöcke bei beiden Varianten und nur einen statt vier LM-Head-Aufrufen
im Kandidaten.

Die festgelegten Ressourcenregeln wurden eingehalten: kein Swap-Anstieg,
`25,346709 s` GPU-Arbeit, längster zusammenhängender GPU-Abschnitt
`2,283342 s`, `192,383843 s` Guard-Pausen und Duty-Faktor `0,15`. Alle fünf
vorregistrierten Hürden bestanden. Die Entscheidung lautet deshalb
`engineering_go_exact_scope`; `formal_claim` bleibt ausdrücklich `false`.

Die betriebliche Freigabe gilt nur für den exakt qualifizierten lokalen
Modell-Snapshot, Prompt, `897` Prompt-Token, Chunk `256`, Batch `1`, greedy ohne
Prompt-Logprobs und genau `32` Ausgabetoken. Jede Abweichung bleibt auf dem
Referenzpfad. Die private Runtime-Historie enthält exakt drei verkettete Records;
ihre Datei-SHA ist
`6dcf6e4cb942b842dca6e9b0b071df8e7c6cb81ba28fdc5e0fdb05c414d20567`.
Die read-only Loopback-UI lieferte die drei Records mit HTTP `200`, bestätigte die
Hashkette und veränderte die Datenbank nicht.

Offen bleiben die geforderten Workloads Multi-Turn-Fortsetzung und mehrere
parallele Requests sowie jede Verallgemeinerung auf andere Prompts oder
Einstellungen. Gemessen sind die Laufzeiten und die Tokenidentität; der
Prozentwert ist aus den vorregistrierten gepaarten Laufzeiten berechnet.

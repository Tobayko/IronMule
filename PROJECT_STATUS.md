# Projektstatus

**Stand:** 21. August 2026
**Zielgerät:** Apple M1 Max, 32 GB Unified Memory, 10-Core CPU, 32-Core GPU

## Auditierter aktueller Stand

| Bereich | Verifizierbarer Stand | Zulässige Aussage |
| --- | --- | --- |
| Root-Provenienz | Root-Git-Repository; formaler H1-v2-Code auf sauberem Commit `1fbe73c`; `ProjectAtlas/` als gepinntes, unverändertes Gitlink | formale und native Läufe sind an konkrete Root-Revisionen gebunden |
| H0 | `.friday-data/h0.sqlite3` mit `28` Runs, darunter `9` `aa_gpu`-Runs | H0-Rohhistorie vorhanden; **kein** formal geschlossenes A/A-Gate |
| H0.1 | `3` Legacy-Beobachtungen, `6` Paced-Sessions, `1` Study mit `h01_complete_unresolved` | replizierte Stationarität nicht unterstützt; gültiger Negativbefund |
| H1/H2 historisch | zehn rekonstruierbare Zusammenfassungen, keine Rohblöcke und keine vollständige historische Provenienz | ausschließlich `legacy_summary`; formale H1/H2-Claims `false` |
| H1/H2 künftig | SQLite-v1-Evidenz, saubere Git-/Code-/Spec-/Environment-Bindung, gemeinsame Budgets und read-only Historien-UI implementiert; vier native Ereignisse vorhanden | prospektive Exploration möglich; formale Claims bleiben in v1 ausdrücklich `false` |
| H1-v2 formal | terminale 16-Record-Historie: versiegelte Präregistrierung, sechs bestandene A/A-Sessions, MDE `5 %`, sechs frische A/B-Sessions und Split-Entscheid `h1_gain_confirmed` | für genau ein Gerät, FP16-`2048²`, acht Matmuls und den Batch-Dispatch-Plan ist der Gain jenseits der MDE formal bestätigt; kein Modell-/Cross-Device-Claim |
| Begrenzte Runtime | exakte H1-Bindung, tensorbasierte Scope-Prüfung, serieller Fallback, Circuit Breaker, Hash-Ketten-Historie und read-only UI; CPU- und MLX/GPU-Gates auf sauberem Commit bestanden | Batch ist nur für den exakt registrierten Workload freigegeben; Policy-/Runtime-Befund ist Engineering-Validierung, kein neuer formaler oder Modell-Claim |
| H2 Gemma-Minimallauf | eine offline erzwungene Gemma-4B-Runde schlug `N=3,10,16` vor; Harness bestätigte explorativ `N=10` mit frischen drei Replikaten | nützliche Modellselektion beobachtet, aber Schema v1 bleibt `formal_claim=false`; keine Runtime-Erweiterung und keine zweite Runde |

Die produktive Research-DB enthält `10` verifizierte `legacy_summary`-Zeilen und
`4` native Ereignisse: drei gültige Berichte mit Rohmessungen sowie einen
sanitisierten Guard-Abbruch. Datei: `118.784 B`, Modus `0600`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision `c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Ein zweiter Import war idempotent (`0` neu, `10` bereits vorhanden) und ließ den
damaligen Dateihash unverändert. Der vollständige Offline-Testlauf nach
Implementierung des Runtime-Prototyps bestand mit `468` Tests und `2.463`
Subtests in `34,58 s` (Wall des umgebenden Prozesses `34,87 s`, User `135,36 s`,
System `3,38 s`, Exit `0`). Der letzte H1-v2-Implementierungsstand davor lag bei
`455` Tests und `2.447` Subtests in `33,04 s`; die H1-v2-Baseline bei `439`
Tests in `32,01 s`.

Das Evidenzaudit korrigiert die frühere Statussprache: Der dokumentierte formale
A/A-Loader verlangt global genau sechs kompatible Prozesse, die append-only H0-DB
enthält aber neun `aa_gpu`-Runs aus mehreren Generationen. Mindestens ein relevanter
Prozess war `warmup_unstable`; zudem fehlte allen historischen Runs eine Root-Git-
Revision. Unmittelbar vor dem ersten A/B-Lauf waren hierarchischer Bootstrap,
formales A/A-Gate und MDE noch nicht geschlossen. Spätere gepaarte, replizierte
H1/H2-Zahlen bleiben technisch wertvoll, können aber nicht rückwirkend
vorregistriert werden.

Aktueller Entscheid: Der **begrenzte Runtime-Prototyp** hat seine Gates bestanden;
die genau eine freigegebene H2-Gemma-Runde ist ebenfalls abgeschlossen. `N=10`
ist ein vielversprechender, aber explorativ selektierter Kandidat. Er wird nicht
in die auf `N=8` versiegelte Runtime übernommen. Nächster möglicher Schritt ist
eine neue prospektive Ein-Kandidaten-Studie mit frischen Daten und eigener
Architekturfreigabe. Phase 1B/Custom Metal bleibt **NO-GO**, Cross-Device
**NO-CLAIM**, weitere Modellrunden und breiterer Live-Suchraum **NO-GO**.
Details: [`docs/FORSCHUNGSENTSCHEID_2026-08-21.md`](docs/FORSCHUNGSENTSCHEID_2026-08-21.md),
Persistenzvertrag: [`docs/H1H2_EVIDENZ_ARCHITEKTUR.md`](docs/H1H2_EVIDENZ_ARCHITEKTUR.md).
Der initiale Auditlauf installierte nichts, lud nichts herunter und führte keinen
GPU- oder Modelllauf aus. Nach späterer ausdrücklicher Rechenfreigabe wurden die
unten dokumentierten lokalen Läufe ausgeführt; auch dabei gab es weder Download
noch Installation.

## Formales H1-v2-Ergebnis und Runtime-Pre-Live-Stand

Die formale Studie lief vollständig auf dem sauberen Commit
`1fbe73c69cedeb69284a264c5e3f45e3e393b822`. Die Präregistrierung bindet Code,
Spezifikation, Python/MLX-Umgebung und Apple-M1-Max-Hardware; alle zwölf Sessions
liefen am Netzteil in getrennten Prozessen mit realem Inter-Session-Cooldown.

Die sechs A/A-Sessions ergaben ein aggregiertes Verhältnis `1,000109` mit
95%-Intervall `[0,999193; 1,000540]`. Die rohe kalibrierte MDE war rund
`0,0752 %`; prospektiv blieb deshalb der konservative Floor von `5 %` maßgeblich.
Alle vier Kalibrierungsgates bestanden. Die sechs anschließenden A/B-Sessions
waren byte-identisch und ergaben insgesamt `R=0,879718`, 95%-Intervall
`[0,877045; 0,880403]`, Effekt `−12,028 %`. Charakterisierung
(`R=0,879415`) und Validierung (`R=0,880044`) bestanden das Gain-Gate getrennt.
Der terminale Record
`f508fc9e2b1f44a1b60084bdbeca581024f1f3599535b3dd662a9305c99a9357`
trägt als einziger `formal_claim=true` und erlaubt nur
`permit_bounded_runtime_prototype`.

Die formale Datei `.friday-data/h1-v2.sqlite3` enthält `16` vollständig
replaybare Records, Modus `0600`, Größe `163.840 B`, SHA-256
`141f010bf4946ec39f5f87d2c8fbc50daf57305fa3d4772a7b962b101e78a4c4`.
Ein erneuter read-only Runtime-Preflight ließ diesen Hash unverändert.

Der getrennte Prototyp ist in `friday_runtime/` und
[`docs/RUNTIME_PROTOTYPE_SPEC.md`](docs/RUNTIME_PROTOTYPE_SPEC.md) definiert.
Er autorisiert Batching nur bei exakt derselben terminalen H1-Entscheidung,
unverändertem H1-Code/Spec-Fingerprint, derselben Umgebung/Hardware, sauberem
Worktree und aus tatsächlichen Tensoren abgeleitetem Workload. Alle anderen
Fälle wählen seriell. Ein Batch-Fehler wird nicht im selben Aufruf wiederholt,
sondern verriegelt alle Folgeaufrufe seriell. Im absichtlich schmutzigen
Entwicklungsstand verifizierte der reale Preflight alle `16` H1-Records und fiel
korrekt mit `worktree_dirty` auf seriell zurück. Eine Live-Messung erfolgte vor
dem sauberen Runtime-Commit bewusst noch nicht.

Auf dem anschließend sauberen Commit
`0b0a893f58e9c757a0aa7b49565a8b1c1eb2a561` autorisierte derselbe Preflight den
exakten Scope. Das CPU-Gate (5 Warmups, 21 balancierte Blöcke, je 20.000 Aufrufe)
ergab Policy-Median `11.045 ns`, p95 `11.078 ns` und gepaarten zusätzlichen
Median `11.017 ns`; Record
`a9c08e2b4d79590e1cfa1d5270c53a80a69b1ff1f39507f003fcd6d8d2be1815`.
Alle Grenzen von `25/50/20 µs` bestanden.

Die anschließende MLX/GPU-Validierung (2 Warmup-Paare, 12 balancierte Blöcke)
ergab seriell `20,360 ms`, Runtime-Batch `17,643 ms`, gepaartes
`R=0,879209` und Effekt `−12,079 %`. Die acht Outputs waren byte-identisch,
maximaler absoluter Fehler `0,0`; Circuit Breaker blieb offen. GPU-Arbeit
`0,667252 s`, Wall im Guard `1,059850 s`, maximale kontinuierliche Last
`0,667252 s`, MLX-Peak `209.715.200 B`, RSS-Peak `440.401.920 B`. Record:
`643af8606c83cbcd0a591ba63bebb8745ddf5d4a346971c1d733c8d2b566c2dc`.
Der Policy-Median entspricht rund `0,063 %` der Kandidatenlaufzeit.

`.friday-data/runtime.sqlite3` enthält damit zwei vollständig replaybare,
hashverkettete Records, Modus `0600`, `45.056 B`, SHA-256
`ad4f0ef703d1426c85853eb00a5f50ea8b1bd73a25fb121b13570d9676473d82`;
Snapshot-Revision
`a53e6b31c8266b1881ebebfc4dca8c28e9a4177d7648496863fc2b6d4cd6eb3f`.
Read-only UI-Snapshot und H1-Readback änderten keine Datei.

## Geschlossener H2-Gemma-Minimallauf

Auf dem sauberen Dokumentationscommit
`99267d3422f5a8573cad0f53e7009a4cf8f52198` lief genau eine Runde des bereits
implementierten `model-loop`. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` und
der projektlokale Resolver schlossen einen Netzwerkfallback aus; geladen wurde
ausschließlich der vorhandene Snapshot
`mlx-community/gemma-3-4b-it-4bit` Revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, eine MLX-Gewichtsdatei mit
`3.400.569.562 B`. Es gab keinen Download und keine Installation.

Die Modellantwort war ausschließlich `[3, 10, 16]`; Parser und Allowlist ließen
genau diese drei Integer-Kandidaten zu. Explorative 20-Block-Messungen:

| Batchgröße | B/A-Ratio | 95%-Intervall |
| ---: | ---: | ---: |
| `3` | `0,849019` | `[0,797567; 0,903789]` |
| `10` | `0,784921` | `[0,741686; 0,830676]` |
| `16` | `0,889566` | `[0,881424; 0,897784]` |

Der Harness wählte `N=10` und bestätigte es separat mit drei Replikaten
`0,6649/0,6716/0,7014`: hierarchisch `R=0,671573`, 95%-Intervall
`[0,648895; 0,731190]`, explorativer Effekt `−32,84 %`. Korrektheitsgates und
5%-Schwelle bestanden. Der Bericht bleibt explizit `formal_claim=false`, weil
das Modell drei Kandidaten aus vorhandener Evidenz auswählte und diese Studie
nicht prospektiv als formale N=10-Bestätigung registriert war. Die produktive
Runtime bleibt deshalb auf `N=8`; `N=10` fällt dort seriell zurück.

Der Guard verbuchte `9,908610 s` GPU-Arbeit, maximal `1,730481 s`
kontinuierlich, `180,024674 s` Kandidaten-Cooldown, `16,022076 s` Pflichtpausen
und `212,268826 s` Wall. Evidenz-ID:
`5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a`.
Die Research-DB enthält nun `14` verifizierte Zeilen, davon `4` native und eine
native `model-loop`-Zeile mit Rohdaten; Modus `0600`, `118.784 B`, SHA-256
`70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91`,
Snapshot-Revision
`c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b`.
Der read-only Replay ließ den Hash unverändert. Es wurde keine zweite Runde
gestartet.

## Neue native v1-Exploration nach Rechenfreigabe

Alle folgenden Läufe waren an einen sauberen Root-Commit gebunden, liefen am
Netzteil unter dem gemeinsamen Guard und wurden vor stdout append-only
persistiert. Schema v1 erzwingt weiterhin `formal_claim=false`.

**Einzeloperation vor dem Modelltest:** FP16-Matmul `2048²`, acht Operationen,
drei Replikate mit je 25 gepaarten Blöcken. Batched Dispatch war byte-identisch
und erreichte `R=0,780054`, hierarchisches 95%-Intervall
`[0,765530; 0,877456]`, Effekt `−21,995 %`. GPU-Arbeit `2,803 s`, Wall
`11,468 s`; Evidenz-ID `b866022a…a92eb6`. Das überschreitet explorativ die
fixierte 5%-Schwelle, ist aber noch kein formaler H1-v2-Nachweis.

**Lokaler Gemma-Roofline-Lauf:** Die Werkzeuge lösen ausschließlich validierte
Snapshots im Projektcache auf; Repository-ID, Revision und Gewichtsumfang stehen
im Bericht. Fünf Messwiederholungen je Modell, 369 Prompt-Token:

| native v1 | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Snapshot | `2d44e83d…` | `93724907…` |
| Folge-Token | `5,012 ms` / `199,5 Token/s` | `10,949 ms` / `91,3 Token/s` |
| Prefill | `0,3271 s` / `1.128,0 Token/s` | `0,8735 s` / `422,4 Token/s` |
| geschätzte Bandbreitennutzung | `36,53 %` | `58,47 %` |
| geschätzte FP16-Rechennutzung | `2,78 %` | `4,45 %` |
| exploratives Roofline-Urteil | speicherbegrenzt | speicherbegrenzt |

Gesamtbudget: `10,360 s` GPU-Arbeit, maximal `1,129 s` kontinuierlich,
`52,123 s` verifizierte Pausen und `68,111 s` Wall. Evidenz-ID
`31c20b1e…647c36`, Commit `faa4f88`. Ein erster Versuch auf Commit `29a2b74`
wurde korrekt vor 4B abgebrochen, weil Warmups/Wiederholungen ohne Zwischenpause
die `6-s`-Kontinuierlichkeitsgrenze überschritten; Fehler-ID
`ffe98ffa…1a0ac4`. `pace_generation` und drei Regressionstests schließen diese
Lücke.

Die neue Rohmessung reproduziert die Richtung der historischen Roofline-
Zusammenfassung, wertet sie aber nicht formal auf. Phase 1B/Custom Metal bleibt
**NO-GO**. Das zuvor fehlende Schema/Protokoll v2 wurde anschließend unter
`friday_h1/` implementiert, offline verifiziert und auf dem sauberen Commit
`1fbe73c` formal ausgeführt; das Ergebnis steht im vorherigen Abschnitt.

## Historisches Arbeitsprotokoll

Die folgenden Abschnitte erhalten die damaligen Messungen und Entscheidungen. Wo
sie „bestätigt“ sagen, ist das die **historische explorative Klassifikation**, nicht
der aktuelle formale Evidenzgrad.

## Historischer H0-Implementierungsstand

- H0 ist eine einzelne FP16-`2048²`-Matmul-Workload, kein Modelltest und kein
  Self-Optimization- oder Hardware-Generalisation-Nachweis.
- Der Offline-Unterbau umfasst SQLite v1 unter `.friday-data/h0.sqlite3`, festen Worker
  Option A und ein read-only Dashboard auf `127.0.0.1`. Die lokale Datenbank enthält
  `22` Runs einschließlich Run22; Run22 ist eine abgeschlossene H0-eager-baseline
  reference. Das ist kein Modell-, A/A- oder Self-Optimization-Nachweis.
- `run_mlx` und der statische Schalter `mlx-run --execute` sind implementiert. Ohne
  `--execute` endet der Befehl vor Runner-/Worker-/Benchmark-/MLX-Import weiterhin mit
  `EXIT_MLX_LOCKED=78`/`state=not_released`.
- W1v3 wurde in Benchmark, Worker, Runner und Aggregation umgesetzt und offline geprüft:
  äußere Warmup-Blöcke von mindestens `50 ms`, maximal `4096` Evals mit
  `repetition_window_unreachable` bei Nichterreichen, Gate `round(block_ns/evals)`,
  `8..16` Blöcke, ±`5 %` für die letzten fünf Gate-Werte, geschlossene bounded
  Block-Summaries sowie Warmup-Fehlerdiagnose Schema v2 mit v1-Readback. Das tote
  `_Timed.output`-Feld ist entfernt.
- Run22 lief genau einmal. Der Common-Wrapper ist `completed/measurement_complete/
  baseline_fallback` mit `error=null`; gemäß Worker-/Runner-Vertrag sind die verschachtelte
  `benchmark_classification=baseline_reference`, `benchmark_action=not_run` und
  `aggregation_required=false` die erfolgreiche eager-baseline-reference. Die anfängliche
  Operator-Deutung von `baseline_fallback` als Fehler wurde anhand dieses Vertrags korrigiert;
  es liegt kein Produktfehler vor.
- H0-Baseline läuft. Weder `aa_gpu` noch Optimierung oder Self-Optimization sind bewiesen.

## Historischer H0.1-Stand — separate Stationaritätsforschung

- H0.1 ist strikt von H0 getrennt. Es besitzt einen vorregistrierten Paced-Trajectory-
  Vertrag mit exakt sechs vorgesehenen Sessions `C0,V0,C1,V1,C2,V2`, einen
  stdlib-only Analyse-/Study-Core, eine eigene append-only SQLite-v1-Datenbank und ein
  read-only Dashboard. Am 21.08.2026 wurden `6` Paced-Sessions und `1` Paced-Study
  auf dem Zielgerät ausgeführt; der H0.1-Stationaritätsentscheid liegt damit vor und
  lautet `h01_complete_unresolved`.
- Vier historische H0-Generationen wurden vollständig über den öffentlichen H0-
  Bundle-Verifier inventarisiert und durch exakte provenance-/strukturgebundene Adapter
  geschlossen behandelt. A (`runtime_unavailable`) ist eine erkannte Exklusion. B, C
  und D wurden am 20.08.2026 in genau einem atomaren Produktions-Execute als rein
  deskriptive `legacy_h0_warmup_observation`-Bundles importiert. Es gab keine
  H0-Reklassifikation und keine Leistungs- oder Stabilitätsaussage.
- `.friday-data/h01.sqlite3`: `3` verifizierte Legacy-Bundles, Größe `53,248 B`,
  Mode `0600`, SHA-256
  `fd2c6e56d5f108d6670745a338930d6050c38b03eac8cc050170a466818d9d57`.
  Execute-Report-SHA-256:
  `4e73ab2d7b0aa0bf0cb7e559550de254ddadfa41c0f31ee86b92b9203bef788f`;
  H0 blieb bytegleich bei SHA-256
  `4478c1b47d92ea64ccb14a06056cb0062b2efd8f7804513defc56831a0fe5c51`.
- Das read-only H0.1-Dashboard zeigt Total `3`, Kind
  `legacy_h0_warmup_observation=3`, Status `legacy_observation=3` und Revision
  `d9bc6e5ab430b68e16c9b9dfa62463896c9ad9d64ef003a4a862460378b2af3f`.
  Der finale socketfreie Snapshot lief in `0.04000666690990329 s` bei Peak-RSS
  `29,261,824 B`. Die reale read-only HTML-/API-Grenze wurde danach auf
  `http://127.0.0.1:8766/` mit Exit `0` geprüft; der Server läuft in Session `40690`.
- Finale H0.1-Verifikation: `57/57` Tests und `2,244/2,244` Subtests,
  `0` Failures/Errors/Skips, Wall `25.714773458894342 s`, Self-User/System
  `22.572449/0.502031 s`, Peak-RSS `43,368,448 B`; NumPy-/MLX-Importe und
  Socketkonstruktionen jeweils `0`.
- Forschungsgrenze: Der Legacy-Import ist **GO** als Evidenzmigration. Der
  Sechs-Session-Paced-Vertrag wurde am 21.08.2026 auf dem Zielgerät ausgeführt und
  vollständig replayt; siehe den folgenden Abschnitt. Das war kein Modelltest; es
  wurde kein Modell geladen oder installiert.

## Störprozess aufgeklärt (21.08.2026)

- Der Untergrund, der ungepaarte Messung wertlos macht, ist charakterisiert:
  **unimodal mit langem rechtem Schwanz**, **zufällig verteilt** (Runs-Test über
  sechs Sessions: beobachtet ≈ erwartet), **blockweit** (`22` von `150` Blöcken
  treffen beide Arme, erwartet wären `4,1`) und mit einer **Zeitskala von rund
  `340 ms`** (Autokorrelation `+0,576` bei `68 ms`, `0,000` bei `408 ms`).
- Ein langsam variierender, gerätweiter Prozess — plausibel OS-Scheduling und
  fremde Last. Nicht eliminierbar, aber erklärend: Beide Arme eines Blocks liegen
  in derselben Störungsepisode, weshalb sich die Störung im Quotienten herauskürzt.
- Erklärt vier bisher getrennte Beobachtungen als ein Phänomen: das ungelöste
  H0.1, das zu breite A/A-Bootstrap-Intervall, die nicht funktionierende
  Cutoff-Metrik und die Wertlosigkeit ungepaarter Messung.
- **Neue harte Messregel:** Vergleichsarme innerhalb von rund `340 ms` messen.

## Roofline — die Inferenz ist speicherbegrenzt (21.08.2026)

| | Gemma 3 1B | Gemma 3 4B |
| --- | ---: | ---: |
| Bandbreite genutzt | `31,9 %` | `51,2 %` |
| Rechenwerke genutzt | `2,4 %` | `3,9 %` |
| Prefill je Token schneller | `7,3x` | `5,4x` |

- **Faktor `13`** zwischen beiden Auslastungen, in beiden Modellen `memory_bound`.
  Zwei unabhängige Wege (Auslastungsrechnung und Prefill-Vergleich) sagen dasselbe.
- **Konsequenz:** Code „näher an der Maschinensprache" optimiert den Anteil, der
  mit `2,4`–`3,9 %` ohnehin leerläuft. Wirksam sind nur weniger Bytes
  (Quantisierung — bei 4-bit-Modellen bereits eingelöst) und weniger Durchgänge
  (Kernel-Fusion).
- **Obergrenze:** Bei `51,2 %` Bandbreitenauslastung bringt selbst eine perfekte
  Optimierung ohne Gewichtsverkleinerung höchstens rund `2x`.
- Spitzenwerte `400 GB/s` und `21 TFLOPS` sind Herstellerangaben, nicht gemessen.

## Fusions-Layer — geprüft und verworfen (21.08.2026)

- Ein `mx.compile`-Wrapper über den Forward-Pass zeigte `−12,4 %` (1B) und
  `−15,0 %` (4B) bei bytegleichen Logits. **Praktisch wertlos:** die
  End-to-End-Messung an der echten Generierungsschleife ergibt `−0,5 %` und
  `−0,1 %`.
- Ursache belegt: Die Generierung übergibt bei jedem Aufruf einen KV-Cache
  (`18` Aufrufe mit Cache, `0` ohne), `mx.compile` kann `RotatingKVCache` nicht
  entgegennehmen, und **`mlx-lm` fusioniert bereits selbst**
  (`@partial(mx.compile, shapeless=True)` in `gemma3_text.py` und
  `activations.py`).
- Eigener Messfehler auf dem Weg: `model.__call__` auf der **Instanz** gesetzt —
  Python löst `obj()` über `type(obj).__call__` auf und ignoriert das
  Instanzattribut, der Patch war wirkungslos.
- Wert des Ergebnisses: Ein ganzer Lösungsweg ist **mit Begründung**
  ausgeschlossen. Eine wirksame Layer müsste unterhalb ansetzen — Cache-Layout,
  Speicherverwaltung oder eigene fusionierte Kernel.
- Werkzeuge `tools/measure_roofline.py` und `tools/measure_fusion_layer.py`;
  letzteres misst ausdrücklich den cache-freien Forward-Pass, **nicht** einen
  Generierungsgewinn.

## Historische explorative H2-Codegen-Beobachtung (21.08.2026)

- Nutzerfreigabe für Ausführung modellgenerierten Codes und erhöhtes GPU-Budget
  erteilt. Kein zweites Gerät, Cross-Device bleibt offen.
- **Bestätigt: `R = 0,8838`, `−11,62 %`, `95%-KI [0,8676, 0,8975]`**, drei
  Replikate. Fünf Pläne geschrieben, fünf gemessen, drei über der Schwelle.
- Drei Schutzschichten: heute semantisch begrenzte AST-Plansprache (ein
  Iterationslevel, höchstens `32` statisch gewichtete Matmuls, keine freien
  Allokationsprimitive), Prozessisolation mit Timeout/CPU-Grenze/bereinigter
  Umgebung sowie ein Correctness-Gate. Die MLX-Speichereinstellung ist nur eine
  Richtlinie, kein hartes OS-Limit.
- Zwei Anläufe scheiterten an eigenen Fehlern: Der Prompt zeigte die Baseline zu
  prominent (Modell kopierte sie viermal), und der Validator blockierte
  `out.append(x)` — genau die gesuchte Optimierung. Beides korrigiert und
  getestet.
- Werkzeuge `tools/plan_sandbox.py` und `tools/codegen_loop.py`, in der CLI als
  `codegen`. Gesamtsuite `387` Tests / `2.377` Subtests grün.
- Grenze: Der Plan bleibt eine Umsortierung derselben festen Rechnung. Keine
  Kernel, keine Algorithmenwahl, keine Numerikänderung — die Allowlist lässt das
  nicht zu.

## Vollständiger Testlauf und Selbstoptimierung (21.08.2026)

- **Testsuite `90 s` → `31 s`** (Faktor `2,9`) über `pytest-xdist`, in `pytest.ini`
  festgelegt. Untere Schranke ist ein `17,6 s`-Test mit `16` vollen Bootstraps;
  mehr Worker bringen nichts (Amdahl). Das Bootstrap zu beschleunigen wurde
  **verworfen**, weil `friday_h0/aggregation.py` in der geschlossenen Code-Liste
  steht und eine Änderung die Provenienz aller H0-Läufe brechen würde.
- **Sicherheitsfund: `aa` besaß kein `--execute`-Gate.** Der Prüfaufruf startete
  real einen A/A-Lauf; nur der Resume-Mechanismus verhinderte eine Aufzeichnung.
  Gate nachgerüstet, alle vier messenden Werkzeuge gesperrt. `ReleaseGateTest`
  prüft jetzt, dass jedes registrierte Werkzeug einer Gruppe zugeordnet ist —
  ein neues Werkzeug ohne Einordnung lässt die Suite fehlschlagen.
- **Entdoppelung:** `require_ac_power` (vier Kopien) und das Release-Gate (drei
  Kopien) liegen jetzt in `tools/_bench.py`. Dateiübergreifender Duplikatscan
  findet nichts mehr.
- `aa` hat jetzt auch `--self-check`; das README-Versprechen gilt damit für alle
  vier Werkzeuge.
- **`337` Tests / `2.322` Subtests grün in `31,3 s`**, Guard `pass`, Provenienz
  beider Phasen unverändert, End-to-End nach dem Refactoring bestätigt.

## Einsatzreife für Dritte (21.08.2026)

- **Einziger Einstieg `tools/friday.py`:** `list`, `doctor` und Durchreichen an die
  fünf Werkzeuge. `doctor` prüft Python, MLX/Metal, NumPy, `mlx-lm`, Netzbetrieb
  und Plattenplatz.
- **`docs/ERGEBNISSE.md`** fasst alle Befunde, sieben Nullbefunde, Grenzen und
  sechs Messregeln auf einer Seite zusammen; jeder Befund nennt sein
  Reproduktionskommando. Ersetzt für Einsteiger die `2.540` Journalzeilen.
- **README neu:** Kernbefund im ersten Absatz (ungepaart messen ist hier nahezu
  wertlos, Beispiel `mx.compile`), Schnellstart, Werkzeuge, Messregeln, Budgets,
  Grenzen.
- Keine absoluten Pfade im Code; `requirements-apple-silicon.txt` führt `mlx-lm`
  als optionale Zeile mit Dry-Run-Hinweis.
- `tests/test_friday_cli.py` (`12` Tests, `34` Subtests) prüft Werkzeugregistrierung,
  Release-Gates, Self-Checks und dass jeder Dokumentationslink auflöst.
- Gesamtsuite `326` Tests / `2.312` Subtests grün, H0.1-Guard `pass`.

## Historische explorative Self-Optimization-Loop-Beobachtung (21.08.2026)

- **`3` von `3` Läufen: `optimization_confirmed`.** Gewählt `N=8` (`−13,60 %`),
  `N=6` (`−11,13 %`), `N=6` (`−14,11 %`). Der Loop konvergiert auf `N=6`–`8`.
- **Autonomer Fund:** `N=6` und `N=7` kamen in der manuellen Suche nicht vor
  (dort nur `2,4,8,16`). Der Loop schlug sie selbst vor, maß sie und bestätigte
  einen davon unabhängig.
- Drei Runden: `explore` über eine feste Kandidatenmenge, `refine` mit selbst
  vorgeschlagenen Nachbarn, `confirm` mit `3` Replikaten und hierarchischem
  Bootstrap. Correctness-Gate vor jeder Zeitmessung: bytegleich oder verworfen.
- **Erst nicht reproduzierbar (`1/3`), Ursache Winner's Curse.** Rangfolge nach
  dem Punktschätzer wählte den glücklichsten Ausreißer (`0,750`, `0,741`), der bei
  unabhängiger Nachmessung auf `0,87`–`0,96` regressierte. Korrigiert auf
  Rangfolge nach Konfidenzobergrenze; die Schwelle `MDE = 5 %` blieb unverändert.
- Werkzeug `tools/optimization_loop.py` (`--execute`-Gate, Netzbetrieb, GPU- und
  Wall-Budget, `--self-check`), Tests `tests/test_optimization_loop.py` (`15`).
- Grenze: fester, von Hand definierter Suchraum. Kein Codegenerieren, keine
  Kernel — das bleibt H2 mit eigener Sicherheitsfreigabe.

## Historische explorative H2-Vorstufe — Modelltests Gemma 3 (21.08.2026)

- Nutzerfreigabe für Download und Installation erteilt, Auflage Projektordner
  eingehalten: `HF_HOME` auf `.friday-data/models`, Pakete im lokalen `.venv`.
  Belegt sind `3,9 GB`; `16 GB` bleiben frei.
- **Provenienz ungebrochen.** `uv pip install mlx-lm` zog `24` Pakete, ließ aber
  `mlx 0.32.0` und `numpy 2.5.2` unverändert. `environment_sha256` bleibt
  `74ca2dac…`, `code_sha256` H0 `101cdadf…` und H0.1 `f66e4b5a…` ebenfalls.
  Alle früheren Läufe bleiben vergleichbar.
- **Stufe 1, `gemma-3-1b-it-4bit`:** `737 MB`, TTFT `205 ms` ohne Pause,
  Folge-Token `5,0 ms` (rund `200 Token/s`).
- **Stufe 2, `gemma-3-4b-it-4bit`:** `3,40 GB` auf Disk, davon werden nur
  `2.560,8 MB` geladen. Folge-Token `11`–`12,8 ms` (rund `85 Token/s`),
  TTFT `304,8 ms` ohne Pause.
- **Widerlegte Annahme: es gibt keinen Vision-Tower-Offset im Speicher.** Der
  SigLIP-Tower (`833,7 MB`) und der Projektor (`5,9 MB`) liegen im Repo, werden von
  `mlx_lm.load` aber **nicht geladen**. Die früher geplante Bestimmung über die
  Peak-RSS-Differenz beider Stufen ist damit gegenstandslos; der Anteil wurde direkt
  aus dem safetensors-Index quantifiziert.
- **Cooldown-Effekt bei 4B bestätigt:** `R = 1,414` (`+41 %`),
  `95%-KI [1,209, 1,653]`, `10` Paare im direkten Wechsel. Betrifft nur die TTFT.
- **Der Effekt skaliert nicht mit der Modellgröße:** `1B` `1,37x`, `4B` `1,414x`.
  Trotz vierfacher Parameterzahl praktisch gleich — er ist eine Eigenschaft des
  Geräts, nicht der Arbeitslast, konsistent mit dem Matmul-Befund ohne jedes Modell.
- Zwei eigene Zwischenzahlen wurden durch sorgfältigere Messung nach unten
  korrigiert (`503,9 ms` bei 1B, `4,16x` bei 4B). Abgeleitete Regel: kein Befund
  aus weniger als zehn Wiederholungen, und Behandlungsarme im direkten Wechsel statt
  frei randomisiert, wenn die Behandlung eine Zeitkomponente hat.

## Cooldown-Effekt — isoliert und erklärt (21.08.2026)

- **Dosis-Wirkungs-Beziehung nachgewiesen.** Das erste Sample nach einer Pause ist
  verlangsamt, monoton mit der Pausenlänge: `0,94x` bei `0 s`, `1,89x` bei
  `0,25 s`, `3,67x` bei `2 s`, `4,12x` bei `20 s`. Sättigung bei rund `4x` ab
  etwa `2 s`. Ohne Pause ist der Exzess exakt `0,00`.
- **Ursache überwiegend GPU-Taktung.** Eine Idle-Pause von `5 s` ergibt `R = 4,02`;
  dieselbe Pause mit periodischer Mini-Matmul nur `R = 2,53` (Verhältnis `0,487`,
  `95%-KI [0,311, 0,762]`). Der MLX-Allocator scheidet aus: der Cache bleibt über
  die Pause konstant bei `8,6 MB`.
- **Keep-Alive ist keine brauchbare Optimierung.** Sieben Dosierungen gemessen; die
  beste senkt das erste Sample von `10,05 ms` auf `5,11 ms`, kostet aber `14,44 ms`
  eigene GPU-Zeit. Netto `−9,50 ms`. Alle Varianten netto negativ.
- **Der Effekt erklärt H0.1 nicht.** Post-hoc deskriptiv: ohne die ersten sechs
  Main-Samples bestünde `trend` `1/6`, `changepoint` `0/6`, `tail` `0/6` — identisch
  zum realen Ergebnis. Die H0.1-Instabilität stammt von über die Session verteilten
  Ausreißern. Zwei unabhängige Phänomene. Study unverändert
  `h01_complete_unresolved`.
- Praktische Konsequenz für künftige Messungen: Nach einer Pause gehen bis zu
  `5,12` Sample-Äquivalente verloren. Wer nach einer Pause misst, ohne den Anlauf
  zu verwerfen, verzerrt ein 80-Sample-Mittel um bis zu `5,8 %` — mehr als die
  H1-Nachweisschwelle von `5 %`.
- Werkzeuge: `tools/measure_cooldown_effect.py` (`--execute`-Gate, Netzbetrieb,
  GPU-/Wall-Budget, `--self-check`) und `tests/test_cooldown_effect.py` (`15` Tests).

## Historische explorative H1-Beobachtung — Dispatch-Plan (21.08.2026)

- **Ergebnis: `−14,7 %` bei `N = 8`, Optimum `−17,4 %` bei `N = 4`.** Gepaart
  gemessen, über fünf Replikate repliziert, `95%-KI [0,8263, 0,8777]`, Correctness
  `byte_identical`. Verdikt `effect_confirmed` gegen die vorab eingefrorene
  Schwelle `MDE = 5 %`.
- Der Kandidat ist eine **Ausführungsplanänderung**, keine Kernel-Optimierung:
  `N` Matmuls mit einer einzigen Synchronisation statt `N` einzelner
  Synchronisationen. Identische Arithmetik, bytegleiche Ergebnisse.
- `serial 2,572 ms/Matmul` gegen `batched 2,212 ms/Matmul`; Ersparnis `0,360 ms`
  je Matmul. GPU-Arbeit `5,8 s` gegen Budget `120 s`.
- **Ohne die A/A-Vorarbeit wäre das Ergebnis falsch gewesen.** Ungepaart erschien
  `mx.compile` mit `−27,6 %`; gepaart ergibt derselbe Kandidat `R = 1,0019`,
  `KI [0,9990, 1,0047]` — kein Effekt. Der scheinbare Gewinn war reines Rauschen.
- Ausgeschlossene Fehlerquellen: Deduplizierung identischer Teilausdrücke (alle
  Messungen mit paarweise verschiedenen Operanden), Ergebnis-Caching, veränderte
  Arithmetik.
- Geprüfte Nullbefunde: prätransponiertes `B` `+3,1 %`, `mx.einsum` `+0,6 %`,
  eigener GPU-Stream `±0 %`, echter 3D-Batch-Matmul `−3,9 %`/`−1,8 %` mit
  `1,0` im Konfidenzintervall. Die Optimierung ist damit ausgereizt.
- Reichweite: gilt für **unabhängige** Operationen. Keine Aussage über Modelle,
  Transformer-Inferenz oder andere Geräte. H0 und H0.1 bleiben unverändert.
- Werkzeuge: `tools/measure_dispatch_plan.py` (`--execute`-Gate, Netzbetriebspflicht,
  GPU-Budget, Correctness-Gate, `--self-check`) und `tests/test_dispatch_plan.py`
  (`12` Tests, offline).

## H0.1 — ausgeführte Sechs-Session-Paced-Study (21.08.2026)

- Nutzerfreigabe: ausdrückliche Freigabe für den GPU-Lauf. Keine Installation, kein
  Download, kein Modell. H0.1 misst die feste `2048²`-FP16-Matmul, nicht ein Modell.
- Zuvor fehlte der Ausführungspfad vollständig: `build_trace` nahm bereits
  aufgezeichnete `durations_ns` entgegen, aber niemand erzeugte sie. Neu sind
  `friday_h01/provenance.py`, `friday_h01/runner.py`, `friday_h01/cli.py` und
  `tests/test_h01_runner.py`. Der GPU-Pfad liegt hinter demselben
  `--execute`-Release-Gate wie H0 (`state=not_released`, Exit `78`).
- Preflight: `preflight_ok`, Parent Run22, `fixture_seed=4051312678`, Netzbetrieb
  `ac_power`, `thermal_state=api_unavailable`, Proben `24.60/4.80/3.86 ms`.
  Eingefrorener `code_sha256=f66e4b5a2444643fb375a098398bbd3829d717a7b956e62f46a6a54617986e94`.
- Lauf: sechs getrennte Prozesse `C0,V0,C1,V1,C2,V2`, alle `h01_session_complete`,
  Wall je Session `66.00`–`66.37 s`.
- Gate-Bilanz: `changepoint` 6/6 fail, `tail` 6/6 fail, `trend` 5/6 fail,
  `pacing` 5/6 fail, `ess` 1/6 fail, `acf` 0/6 fail; Summe `23`.
- Study: `h01-study-1812a894…c39ca`, `status=h01_complete_unresolved`,
  `conclusion=replicated_stationarity_not_supported`, `session_count=6`,
  `failed_gate_count=23`, `action=no_h0_conclusion`, `h0_reclassification=false`,
  `promotion_applicable=false`.
- **Ergebnis: replizierte Stationarität ist nicht unterstützt.** Das ist ein gültiges
  negatives Ergebnis, kein fehlgeschlagener Lauf. Der Envelope wird deutlich und nicht
  knapp verfehlt. Dominierend ist die Tail-Ratio `2,53`–`3,13` bei Grenze `1,20`;
  `acf` besteht überall und das Trendvorzeichen wechselt zwischen Sessions, die Daten
  sind also von sporadischen Ausreißern geprägt und nicht von gerichtetem Drift.
- Beobachtung am gleichen Messpunkt, ohne Ursachenbehauptung: Run22 misst rund
  `2,07 ms` je Matmul innerhalb eines dichten 32er-Batches, H0.1 misst dieselbe
  Operation einzeln mit `50 ms`/`750 ms` Pacing bei einem Median um `6,97 ms`. Die
  Ursache ist nicht gemessen. Festhaltbar ist nur, dass die H0-Baseline einen dicht
  gepackten Batch-Zustand charakterisiert und nicht die isoliert gepacte Einzeloperation.
- Verifikation: alle sechs Sessions und die Study unabhängig neu berechnet und
  bytegleich; Gesamtsuite `266` Tests / `2.265` Subtests grün; Dashboard-Snapshot
  Total `10` (`paced_session=6`, `paced_study=1`, `legacy_h0_warmup_observation=3`),
  Revision `a2d1b2469e21f01de04e03b747ac897bb602059d5ec5ceeb098dcba5b03b4e1b`.
- Betriebsrisiko: Eine aufgezeichnete Session ist nicht wiederholbar. Die `run_id` ist
  deterministisch aus Provenienz abgeleitet, die Messdaten sind es nicht; der
  append-only Store antwortet mit `StorageConflict`. Ein Abbruch mitten in der Study
  macht die bereits aufgezeichneten Sessions dieser Provenienz unbrauchbar. Ein
  Wiederholungslauf ist eine neue Study, kein Patch.
- Grenze: H0 bleibt unverändert. Keine Reklassifikation, keine Promotion, keine
  Performanceaussage, kein Nachweis von Self-Optimization oder Generalisation.

## Run22 — abgeschlossener eager-baseline-Reference-Lauf (20.08.2026)

- Nutzerfreigabe und Umfang: begrenzte W1v3-/Output-Fix-Umsetzung plus genau ein
  `eager_baseline`-Canary; kein `aa_gpu`, keine Installation und kein Retry. Der einzige
  Live-Befehl lief mit dem freigegebenen `eager_baseline`-Pfad. Run-ID:
  `h0-eager_baseline-characterization-0-14d435dcc2170feec70d8baaa712860e59a6148ca3f211aad98eff1c9d7cf0ff`.
- Ergebnis: äußerer `real=3.79 s`, Exit `10`, DB danach `22` Runs. Der Common-Wrapper ist
  `completed/measurement_complete/baseline_fallback`, `error=null`. Der verschachtelte
  Vertrag meldet `benchmark_classification=baseline_reference`,
  `benchmark_action=not_run`, `aggregation_required=false`; damit ist der eager-baseline
  Reference-Lauf erfolgreich abgeschlossen. Die anfängliche Operator-Deutung von
  `baseline_fallback` als Fail wurde anhand des Worker-/Runner-Vertrags korrigiert und ist
  kein Produktfehler.
- Warmup und Baseline: `8` stabile Warmups; Gate-Werte
  `[2566556,2179783,2188775,2143891,2155069,2174895,2195533,2192185]`, Median der
  letzten fünf `2174895 ns`; `30` Messblöcke mit jeweils `32` Reps; Calibration
  `68155792 ns`; Baseline-Median `2138574.859375 ns`, MAD `17041.671875 ns`, IQR
  `35343.0859375 ns`, Minimum `2105915.34375 ns`, Maximum `2210087.25 ns`.
- Correctness: Gate bestanden; `9/9` Cases und `86/86` Metrics. `abs_max=
  0.0310508173`, `normalized_l2=0.0002074681`, `abs_q99=0.0110023008`,
  `rel_q99_abs_oracle_ge_1=0.0004333980`.
- Speicher: active `16777216 B`, peak `25165824 B`, cache `8422698 B`, RSS-Peak
  `369655808 B`. Der Memory-Gate ist `not_evaluable_missing_required_metric`,
  `hard_limit=false`. Retention-Nachprobe nach dem Fix: `67108864 B` erzeugt,
  `0` Payloads/`0 B` retained; `_Timed` enthält nur `duration_ns`, `evaluation_ns`
  und `synchronize_ns`.
- Freeze und Artefakte: Code
  `101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc`, Spec
  `b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; kein Git-Root.
  Manifest `73058165244fe505035182f0044dc5ab8bd16ef523ebfc44b44d5b6f616e239e`, Result
  `bda3d23d56e49c2d26bf7c3e73d52b61c3ea022c3fb61ab0719bfedef58a6d09`, Evidence
  `edaf6cae5a98185f183fd368189a8be3a56c194540e4f64300903cff42d1a6a0`, Projection
  `a51aa4b3cadf00dc5338eee199206b2b8f876c4fd3aeaaa2d5261364254ed790`, Bundle
  `a566c912032efab919dddf5ca7f67b986f29464a655abf15617733aeb6947c49`.
- Read-only Dashboard-Snapshot: `snapshot_id=325afcc9a45311ba716f64a51e7395cd7f2cf1c872c9a3f349c6daf9361398de`,
  `source_revision=7cdad7edcb6099894d588bb9927de322bd4f7ce02d256673768647db54131c73`,
  `run_count=22`, `completed`; der Dashboard-Socket war frei.
- Nach Abschluss wurde der read-only Dashboard-Server erneut auf
  `http://127.0.0.1:8765/` gestartet: `state=serving`, Session `4414`. Die integrierte
  Browseransicht traf vor dem Serverstart zunächst `connection refused` und blockierte
  danach den lokalen Reload per Browser-URL-Policy. Es gab keine Umgehung und keine
  Datenänderung; der socketfreie Snapshot bleibt die verifizierte UI-Evidenz, der direkte
  lokale Link ist verfügbar.
- Umgebung und Verifikation: voller einmaliger Pytest-Lauf Exit `0`, Wall `66.837 s`,
  `228 passed`, `2211` Subtests in `66.24 s`. Ein engerer Unittest-Lauf wurde nach
  `30.018 s` mit Exit `124` absichtlich gestoppt, nachdem `103` Marker grün und keine
  Fehler/Fehlschläge sichtbar waren; er wird nicht als Fehler verschwiegen. CLI-Lock:
  Exit `78`; Usage-Fehler: Exit `64`; `xcodebuild -checkFirstLaunchStatus`: Exit `0`.
  ProjectAtlas `0.4.5-rc1`; Python `3.12.13`, NumPy `2.5.2`, MLX `0.32.0`, macOS
  `26.5.2 arm64`. Der sandboxed Import meldete kein Metal-Gerät; es lief dabei keine
  MLX-Operation.
- Vorher/Nachher: vorher `206 passed + 47 subtests`, DB `21`, Retention `64` Payloads /
  `67108864 B` lebend, Code `aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c`,
  Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`; nachher die
  Freeze- und Testwerte dieses Run22-Abschnitts.

Dieser eine Baseline-Lauf ist keine vergleichende Performanceaussage. H0-Baseline ist
damit ausführbar und referenziert; A/A, Optimierung und Self-Optimization sind nicht
bewiesen. Weitere Ausführung benötigt eine neue ausdrückliche Freigabe.

## Historischer Live-Pfad-/Preflight-Stand vor Run22 — 20.08.2026

Die folgenden Abschnitte dokumentieren frühere Preflight-, Run20- und Run21-Stände.
Sie sind historisch und nicht der aktuelle Projektstatus.

- Live-Pfad `45/45 OK`: Wall `4.022908 s`, User/System `3.149974/0.200018 s`, Peak-RSS
  `42,139,648 B`; keine Self-/Child-Aufteilung belegt. Nach dem Fix auf die reale
  `get_cache_memory`-API: `16/16 OK`, Wall `0.086906 s`, User/System
  `0.140900/0.054489 s`, Peak-RSS `49,938,432 B`, ebenfalls ohne belegte Aufteilung.
- Aktuelle Nicht-Live-Suite: `133/133`, Wall `23.720160 s`, User/System
  `22.722187/0.559409 s`, Self-/Child-Peak-RSS `71,368,704/23,642,112 B`;
  unabhängiger Replay `133/133`, Wall `23.588426 s`, User/System
  `22.769535/0.504137 s`, Self-/Child-Peak-RSS `60,342,272/23,707,648 B`. Ein echter
  Importguard belegte, dass dabei keine MLX-Matmul-/GPU-Workload lief.
- Socketfreies Dashboard: `4/4` plus `3` Setup-Subtests, Wall `0.001793 s`, User/System
  `0.001437/0.000137 s`, Self-/Child-Peak-RSS `31,457,280/0 B`, null Socketaufrufe.
  Historisch bestand eine autorisierte HTTP-Prüfung `13/13`; spätere Sandbox-Bindefehler
  und der nicht wiederholte `16`-er Scope sind weder Produktfehler noch finaler Grünnachweis.
- Der Sandbox-Preflight hatte kein Metal. Der danach autorisierte Zielgeräte-Smoke
  bestätigte MLX `0.32.0` und eine 1-Element-Operation in Tool-Wall `1.741108708 s`;
  das war keine Matmul und kein H0-Ergebnis. Eine spätere reine API-Prüfung bestätigte
  `get_cache_memory`, ohne die API oder eine GPU-Workload aufzurufen.
- Canary: äußerer Wall `0.166578416 s`; Child User/System `0.106607/0.040468 s`, Child-
  Peak-RSS `28,442,624 B`, gespeicherter Worker-RSS `23,150,592 B`. Äußere Self-User/
  System/RSS wurden nicht separat gemessen. Ergebnis
  `invalid/runtime_unavailable/baseline_fallback`, Fehler
  `NumPy import unavailable: ModuleNotFoundError`: `0` Rohsamples, `0` Correctness-
  Zeilen, `3` Supervisor-Scalars und `1` Projection-Artifact; Performance, Ratio/KI,
  Warmup/Repetitions und MLX active/peak/cache fehlen. Kein `aa_gpu`, keine Promotion.
- Canary-Hashes: Code `246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b`,
  Spec `a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac`, Environment
  `74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`, Manifest
  `11ac87fb704169e58ac506eda5d0549a91ad19e8ff52b43c5bb7f28e61d982c1`, Result
  `cb97e223fd26c87aa1f1e3a87e56b4c61c76c5b69e7d0420721392727e31aa02`, Evidence
  `406c42b4a99f72703b9623fd8ba5e5c0e68c46495f5a7bd0db1cef1674e0499d`, Projection
  `d9071855d3b1dc6318aa8c832c66c368314ef9ce4ff790911dd4a96939fdaf24`, Bundle
  `1de0c11763c38462420bb74277d8018b2db1517f9eb17e234938b27681a8c41b`.
- Ursache: `Path(sys.executable).resolve()` kollabiert den lexikalischen Launcher
  `/Users/tobiasburandt/Project_Friday/.venv/bin/python` zum Basisinterpreter
  `/opt/homebrew/Cellar/python@3.12/3.12.13_2/Frameworks/Python.framework/Versions/3.12/bin/python3.12`;
  dadurch fehlt in der bereinigten Worker-Umgebung die venv-Paketsuche.
- Minimaler Vorschlag: den fest erwarteten absoluten, aber lexikalischen venv-Launcher an
  `Popen` übergeben und Launcher, Parent und Ziel vor/nach Spawn eng über Owner, Modus,
  Typ, Device und Inode prüfen. Restgrenze: pfadbasiertes `Popen` ist nicht fd-gebunden;
  vollständige TOCTOU-Schließung erfordert Helper/`fexecve` und eine neue Architektur.
  Status: **AWAITING USER APPROVAL**.
- DB: `16` Runs (`15` unveränderte Offline-Controls + Canary). Zwei stabile read-only
  Dashboard-Snapshots: `snapshot_id=aaddbae85cd0e0b94d740eb5e4298532c7bc9d4538dc3a4dfd44f46f66bd019b`,
  `source_revision=f5e2d3286114a238278f08eeec9d95bce1865f759755e0639e97c73385d0ee58`,
  `run_count=16`, `returned_count=16`, `truncated=false`, `query_only=1`. Die UI zeigt
  die Historie ohne Schreibzugriff; nur der Beobachtungszeitpunkt variiert.

## Historischer Offline-Pre-Live-Nachweis — anderer Scope

- Hauptsuite ohne Dashboard: `177` bestanden, `3` Windows-Skips und `12` Subtests;
  Wall `26.034290 s`, Total User/System `23.373336/1.227233 s`, Self-Peak-RSS
  `15,499,264 B`, Child-Peak-RSS `74,186,752 B`.
- Socketfreie Dashboard-Prüfung: `4/4` plus `3` Setup-Subbranches; Wall `0.002041 s`,
  Peak-RSS `31,260,672 B`. Eine frühere autorisierte HTTP-Prüfung bestand `13/13`;
  der spätere `16`-er HTTP-Scope wurde nach der letzten Härtung wegen Sandbox-/Usage-
  Limit nicht final wiederholt. Es wird kein grüner finaler `16`-er HTTP-Lauf behauptet.
- Der CLI-Lock wurde bestätigt: `mlx-run` endet mit Exit `78`, ohne Runner oder Worker zu
  importieren.
- Dritte Offline-Control-Generation:

  | Lauf | Wall | Exit | Ergebnis |
  |---|---:|---:|---|
  | slow | `0.191745 s` | `10` | `regression` |
  | known | `0.156192 s` | `0` | nur synthetisch |
  | wrong | `0.157021 s` | `10` | `correctness` |
  | missing | `0.157268 s` | `10` | `missing` |
  | exit70 | `0.145334 s` | `10` | `worker_exit` |
  | replay | `0.159542 s` | — | `idempotent` |

  Sequenz: `0.967681 s`, Self-Peak-RSS `16,334,848 B`, Child-Peak-RSS
  `28,819,456 B`; Provenance `5745e93f…39d57`, Replay-Bundle
  `6ae4a453…b7335`.
- Finale DB-Evidenz: `15` Runs (`3 × 5`), jeder mit genau einem verifizierten
  `common_result`; die älteren `10` Runs blieben unverändert. Snapshot:
  `source_revision=3b70324f…ab658d`, `id=512934c9…b5b52`, `run_count=15`, nicht
  abgeschnitten. DB-Größe `229,376 B`, Datei `0600`, Verzeichnis `0700`,
  `query_only=1`. Das sind Offline-Controls, keine H0-Hardwarewerte.

Der `177`-er Scope ist historisch und anders enumeriert; er ist mit dem aktuellen
`133`-er Scope nicht als Regression oder Zuwachs vergleichbar.

## Verifiziert

- macOS 26.5.2 (Build 25F84)
- Xcode 26.6 (Build 17F113) unter `/Applications/Xcode.app`
- `xcode-select` zeigt `/Applications/Xcode.app/Contents/Developer`
- `xcodebuild -checkFirstLaunchStatus` erfolgreich
- Python 3.12.13 und uv 0.11.19
- MLX 0.32.0 im bestehenden `.venv`; die aktuelle Luna-Read-only-Introspection bestätigte
  `mx.matmul`, `mx.eval`, `mx.synchronize`, `mx.compile` sowie
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`
- ProjectAtlas-Runtime 0.4.5-rc1 unter `/Users/tobiasburandt/.local/bin/projectatlas`
- ProjectAtlas-Codex-Plugin 0.4.5-rc1 installiert; offizieller Marketplace auf `v0.4.5-rc1`
- Codex-MCP-Server `projectatlas` aktiviert und auf die Project-Friday-Datenbank versioniert
- Historischer Setup-Snapshot (nicht aktuell): `projectatlas init` und anschließender
  `watch --once` indizierten 543 Dateien und 257 Ordner; der damalige lokale Index meldete
  281 offene Purpose-Hinweise.
- `scripts/verify_environment.sh` erfolgreich: Xcode, ProjectAtlas, MLX Metal, PyTorch MPS und
  alle drei MCP-JSON-Dateien geprüft

## Heutige Read-only-Audits

- Erster Sandbox-Lauf: Exit 1 wegen `RuntimeError: No Metal device available`; Ursache ist der
  fehlende GPU-Zugriff innerhalb der Sandbox.
- Genehmigter Lauf außerhalb der Sandbox: Exit 0; Tool-Walltime 1.741108708 s.
- Der aktuelle Luna-Read-only-Introspektionslauf im bestehenden `.venv` bestätigte MLX 0.32.0
  und die APIs `mx.matmul`, `mx.eval`, `mx.synchronize`, `mx.compile` sowie
  `mx.metal.get_active_memory/get_peak_memory/get_cache_memory/reset_peak_memory/
  set_memory_limit/clear_cache`. Die Sandbox hatte kein Metal; daher fand in diesem Lauf
  kein GPU-Lauf statt.
- Es wurden keine lokalen KI-, Modell- oder Software-Installationen und keine Downloads ausgeführt.

## Projektintegration

- Repository: `/Users/tobiasburandt/Project_Friday/ProjectAtlas`
- Projektlokale ProjectAtlas-Daten: `/Users/tobiasburandt/Project_Friday/.projectatlas/`
- generierte MCP-Dateien: `projectatlas.mcp.json`, `projectatlas.claude.mcp.json`,
  `projectatlas.opencode.json`
- vollständiges Konzept kopiert nach `docs/TECHNISCHES_KONZEPT.md`
- Phase-1A/H0-Vorregistrierung ergänzt: `docs/PHASE1_MATMUL_SPEC.md` (Matmul-
  Messsystem-Preflight;
  kanonische FP16-`2048²`-Performance-Workload, separate Correctness-Matrix, Correctness-/
  Memory-/Safety-Gates, Prozess-/Bootstrap-Regeln und Fallback).

## Atlas- und Indexstände

- `543 Dateien / 257 Ordner / 281 Purpose-Hinweise` ist ein historischer,
  definitionsgebundener Stand aus dem früheren `init`-/`watch --once`-Audit; die ursprüngliche
  Verifizierungs-Bullet oben bezeichnet ebenfalls nur diesen damaligen Snapshot. Er wird nicht
  mit späteren Atlas-Zählungen zusammengeführt.
- Der aktuelle Post-Edit-Atlas-Snapshot meldet Generation `22`, `549 Dateien` und
  `257 Ordner`.
- Die aktuelle Atlas-Overview meldet `280` fehlende Purpose-Angaben; das ist eine separate
  Overview-/Coverage-Metrik.
- Der aktuelle Session-Brief meldet `805` Blocker. Dieser Wert ist eine separate
  Session-/Health-Metrik und nicht identisch mit Datei-, Ordner- oder Overview-Zahlen.

## Offen nach Run22

- optional: ProjectAtlas-Purpose-Queue gezielt für die wenigen Projektdateien kuratieren; die große
  Upstream-Codebasis muss nicht manuell mit erfundenen Zwecken versehen werden;
- Der Offline-Unterbau für SQLite v1, das read-only `127.0.0.1`-Dashboard und Worker
  Option A einschließlich Pre-Live-Adapter ist offline implementiert und final geprüft.
  Die feste DB enthält `22` Runs; Run22 ist ein einzelner Baseline-Reference-Lauf, kein
  Hardware-Optimization-Loop.
- Der separate H0.1-Unterbau und die historische Evidenzmigration sind implementiert
  und verifiziert. Nächster wissenschaftlicher Schritt ist nicht ein weiterer H0-Retry,
  sondern die einmalige Durchführung des vorregistrierten Sechs-Session-Paced-Protokolls
  und dessen geschlossener Study-Replay. Bis dahin bleibt H0.1 `unresolved`.
- Die kritische Neubewertung reklassifiziert Phase 1A zu H0: Sie darf nur Messsystem-,
  Correctness-, Kontrollarm- und Fallbackverhalten belegen, keine Self-Optimization oder
  Hardware-Generalisation. Ein Forschungspivot auf H1 deterministic template-constrained
  tuning ist freigegeben, aber erst nach H0-Go/No-Go und A/A-3+3-Aggregation
  wissenschaftlich zu planen.
- `docs/PHASE1A_ARCHITEKTURFREIGABE.md` dokumentiert SQLite v1, read-only Loopback-
  Historien-Dashboard und Worker Option A als `approved/implemented-offline`. Nach der
  späteren allgemeinen Nutzerfreigabe wurde ausschließlich der dokumentierte H0.1-
  Legacy-Import ausgeführt; `aa_gpu`, Custom Metal, Modelle und weitere Optimierung
  wurden nicht ausgeführt.

## Phase-1A-Readiness

- Vorregistrierte Operation: `Y = mx.matmul(A, B)`, FP16 C-contiguous `2048²`, exakt
  17.179869184 GFLOP und 25,165,824 Bytes A+B+Y-Nutzdaten.
- Correctness-only-Matrix separat vorregistriert: sichtbare Seeds `0xC0DE0001` bis
  `0xC0DE0005` sowie Holdout-Seeds `0xC0DE1001` und `0xC0DE1002`; sie ist nicht Teil der
  Performanceaggregation. Zero-RHS muss exakt null sein; die `64²`-Sign-Invariante wird
  innerhalb des eingefrorenen Fehler-Envelopes geprüft.
- Timing-Contract: pro Output `mx.eval(out)` und vor Zeitfensterende `mx.synchronize()`;
  `time.perf_counter_ns`, `mx.eval` und `mx.synchronize` werden im Manifest benannt.
- MLX 0.32.0 und die aufgeführten Matmul-, Eval-, Compile- und Memory-APIs wurden in
  einem früheren read-only Introspektionslauf im bestehenden `.venv` bestätigt; die
  Sandbox hatte kein Metal, daher gab es keinen GPU-Lauf.
- `mx.compile` ist als sichere Framework-Vergleichsvariante eingeordnet; A/A ist der echte
  H0-Nullpfad. Es ist kein Custom-MLX-Metal-Kandidat.
- Keine Tests, Downloads, Installationen, GPU-Läufe oder Modelltests in diesem
  Dokumentationsschritt; die früheren fokussierten Offline-Testmetriken sind im
  append-only Arbeitsjournal vermerkt. Es wurden keine Modelle ausgewählt oder
  festgeschrieben.

## Offen: Phase 1B und vollständiger Phase-1-DoD

Der vollständige DoD aus `IMPLEMENTIERUNGSPLAN.md` ist mit Phase 1A/H0 nicht erfüllt. Offen
bleiben eine separate Phase 1B mit begrenztem Custom-MLX-Metal-Kandidaten sowie dessen
Ausführung in einem isolierten Worker-Prozess mit Timeout, Ressourcenlimits, Correctness-
Test und Rollback. Phase 1B ist nicht implementiert und benötigt vorab eine separate
Sicherheits- und Architekturfreigabe. Die H1-Workload-/Shape-Familienaufteilung und eine
cluster-level Powerplanung nach A/A-Pilot sind ebenfalls offen.

## Grenzen dieses Status

Die Smoke-Tests beweisen nur Erreichbarkeit und einfache Korrektheit. Sie beweisen keine stabile
Performanceverbesserung, keine optimale Kernelkonfiguration, keine Neural-Engine-Kontrolle und keine
Übertragbarkeit auf iOS, Android, NVIDIA oder Rechenzentren.

## Historischer finaler Vertragsabschluss und Run21-Canary — 20.08.2026

Dieser Abschnitt ist der historische Run21-Stand; der vorstehende Run22-Abschnitt ist
maßgeblich.

Der finale Contract-Stand ist dokumentiert und offline geprüft: Core `175/0`, Dashboard
`4/4`, `0` offline MLX-Imports. Die zugehörige Provenienz ist
`575286d8b9a44e47ec355feef3def66ab7cf51ee55a63253ce0667ad054cc203`; Code-Hash
`aae3245e…` (im übergebenen Evidenzsatz nur als Präfix vorhanden), Spec
`a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac` und Environment
`74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782`.

Run21 wurde genau einmal ausgeführt und fail-closed beendet: Exit `10`, Wall `1.14 s`,
User `0.98 s`, System `0.16 s`, Peak-RSS `369,573,888 B`. Der gespeicherte Befund lautet
exakt `invalid/invalid/baseline_fallback` mit `warmup_unstable` nach `16` Warmups. Die
Agent-Statistik zeigt für `all` Median `2,391,354.5 ns`, MAD `287,125 ns`, IQR
`582,260.25 ns`; für `last5` Median `2,155,792 ns`, MAD `87,876 ns`, IQR `396,043 ns`,
Minimum `2,067,916 ns`, Maximum `2,677,583 ns`, Stabilität `false`. Persistiert wurden
`0` Rohsamples, `0` Correctness-Zeilen, `3` Scalars und `1` Artifact. Es gab kein `aa_gpu`
und daraus folgt weder eine Performance- noch eine Correctness-Aussage.

Die DB-Evidenz vor Run20 trägt den übergebenen Hash `c9a521…`; die Run21-DB den Hash
`420b7c…`. Bundle `027908…`, Result `ac4a82…`, Payload `cd409d…` und Evidence
`837841…` sind im Evidenzsatz nur verkürzt übergeben; die Ellipsen werden nicht durch
erfundene Vollhashes ersetzt. Die lokale UI liest die SQLite-Historie automatisch
read-only; die statische Prüfung von `friday_h0/dashboard.py` bestätigt Run-Auflistung und
Statusübernahme einschließlich `invalid`. In diesem Nachweis wurden weder Server noch
Socket gestartet.

Wissenschaftliche Entscheidung: Der eingefrorene Vertrag `8 → maximal 16` Warmups mit
Stabilität der letzten fünf Werte innerhalb `±5 %` entspricht dem Code. Es liegt kein
Implementierungsdefekt vor. Die Ursache der instabilen Messung bleibt als OS-/Thermik-/MLX-
Unsicherheit offen; Schwelle und Daten wurden nicht nachträglich geändert und Run21 wurde
nicht wiederholt. Der `python`-Aliasfehler und der Dashboard-`self.path`-Fehler sind separat
als Harnessfehler klassifiziert, nicht als Projektfehler. Konvergenzregel: Harnessbefunde
werden nur nach reproduzierbarer Wiederholung und unabhängigem Readback bewertet; sie ändern
keine wissenschaftliche Schwelle und ersetzen keinen Canary.

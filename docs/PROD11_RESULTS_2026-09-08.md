# PROD11 — Präfixcache auf Gemma 1B, 4B und 12B nativ korrekt

Die neue, noch deaktivierte Wiederverwendung hat auf dem lokalen M1 Max
(32 GiB) für alle drei vorhandenen Gemma-3-4bit-Snapshots die native
Korrektheitsprüfung bestanden. **Das ist noch kein Geschwindigkeitsgewinn und
keine Freigabe für automatische Aktivierung oder beliebige Chatverläufe.**

## Ergebnis und geprüfter Umfang

| Modell | Vollständige Generierungen | Cache-Gates | Gespeicherter N−1-Zustand (`cache.nbytes`) | Eigener Worker |
| --- | ---: | ---: | ---: | --- |
| Gemma 3 1B 4bit | 12 | 11/11 | 29.483.008 B | PID74147, Exit0 |
| Gemma 3 4B 4bit | 12 | 11/11 | 154.025.984 B | PID74385, Exit0 |
| Gemma 3 12B 4bit | 12 | 11/11 | 436.469.760 B | PID74536, Exit0 |

Je Modell: vier Stock-Generierungen (erster Durchlauf Warmup), eine kalte
Kandidatenantwort plus drei Cachetreffer, zwei Generierungen aus unabhängig
restaurierten Klonen und zwei Recovery-Antworten nach einem echten Abbruch.
Das ergibt zusammen 36 vollständige Generierungen. Zusätzlich wurden je Modell
zwei Prefill-Checkpoints ohne vollständige Antwort und ein nach dem ersten
generierten Token abgebrochener Aufruf ausgeführt; sie zählen nicht als
vollständige Antworten. Die native Kontrolle des Abbruchs gehört zum gebundenen
Testprogramm und seinen Checks; ein eigenständiger vollständiger Cancel-Trace
wurde in dieser ersten Fassung nicht gespeichert.

Der öffentliche Testprompt wird zu genau 1.077 Tokens gerendert, die Ausgabe
ist greedy und auf acht Tokens begrenzt. Ein echter Treffer überspringt die
erneute Verarbeitung der ersten 1.076 Tokens. Stock-/Kandidatenantworten und
Logprobs stimmen exakt überein. Der ursprüngliche kanonische Cache stimmt nach
den Treffern und nach Nutzung zweier Klone bytegenau mit einem unabhängig
erzeugten N−1-Checkpoint überein, einschließlich Positions-/Rotationsmetadaten.

Die elf Checks behandeln falsche Modell-/Umgebungsbindung, falschen privaten
Scope, abweichenden Token-Key, veraltete Commits nach `clear`/`close`,
Klonisolation, Cancel/Recovery, Eviction durch Entrylimit und durch Bytebudget,
Oversize-Skip und unveränderten kanonischen Zustand. Die native Verarbeitung
bleibt seriell; nebenläufige Cache-Restores sind kein Beleg für parallele
MLX-Modellgenerierung.

## Provenienz und Grenzen

Alle fünf Vorher-/Nachherbindungen (Quellen, installierte Module, Provider,
Modellsnapshot und Runtime-Identität) stimmen pro Lauf überein. Die tatsächlich
installierte Kandidatendatei ist bytegleich zur gebundenen Quelldatei. Alle drei
Worker sind nachweislich beendet, jeweils mit Exit0. Die Ressourcenbeobachtung
enthält 10/21/57 Messpunkte für 1B/4B/12B und keine beobachteten Fehler.

Installierter Code:
`a3d1b0e28b8c5e22496005f99d79e2befb13f495709ca88333386adbb28803f8`.
Die Bibliotheksumgebung bleibt
`6e32542c2cd4d2950ec828640d196c7e91be2b4790439041770c93cb7f60ee95`
(Python3.12.13, MLX0.32.0, MLX-LM0.31.3). Gegenüber dem bisherigen Produktwheel
ist ausschließlich `ironmule_product/prefix_reuse.py` hinzugefügt; sämtliche
vorhandenen Pythondateien sind bytegleich. Alte Ergebnisse bleiben an ihren
alten Code gebunden und werden nicht auf den Kandidaten übertragen.

`cache.nbytes` beschreibt den gespeicherten Cachezustand, nicht den gesamten
zusätzlichen RAM-Verbrauch einschließlich temporärer Klone, Modell, Python und
Diagnoseinstrumentierung. Die Hashmaterialisierung beeinflusst Laufzeit und
Speicher; daraus wird kein normaler Serving-Performancewert abgeleitet.

Der Kandidat ist auf **identische vollständige kanonische Promptfolgen** in
einem privaten Scope beschränkt. Unterschiedliche Folgefragen mit nur teilweise
gemeinsamem Präfix, andere Kontextlängen, Sampling, HTTP-Integration, Mehrnutzer-
oder Multi-Mac-Betrieb sind hier nicht qualifiziert. Als Nächstes müssen der
vollständige Nutzen einschließlich Cacheaufbau/-verwaltung und der beste
kompatible Produktvergleich gepaart gemessen werden; anschließend folgt die
geprüfte Integration und autonome Wahl. Noch kein RL- oder Kernelgewinn.

## Nachprüfbare Quellen

- [Native Spezifikation](PROD11_NATIVE_CORRECTNESS_SPEC.md)
- [Deaktivierter Entwurf](PROD11_PREFIX_REUSE_DESIGN.md)
- [1B-Rohbericht](../research/raw/PROD11_1B_native_correctness_20260908_attempt1.json), Run `cea97b6e012d43489e3033bf3e5ef9a8`
- [4B-Rohbericht](../research/raw/PROD11_4B_native_correctness_20260908_attempt1.json), Run `94b6a5293b8b44fb878ef1b9be6e6a4a`
- [12B-Rohbericht](../research/raw/PROD11_12B_native_correctness_20260908_attempt1.json), Run `5674042c94d5448fbc6b9b4e9385f68f`
- [Read-only-Auditor](../tools/product_prefix_audit.py)
- [Unabhängiger Journalabgleich, korrigierter Export](../research/raw/PROD11_native_correctness_audit_20260908_v2.json)

Analyse-QA: mit den genannten Grenzen teilbar. Der unabhängige Abgleich
bestätigt 26/37/73 Journalereignisse, terminale Berichtsdigests, sämtliche
Ressourcenzeilen, den Kindbericht und die Cache-/Ausgabeidentität. Der erste
abgeleitete Auditexport verwechselte den nach `close` leeren Recovery-Cache
mit den gespeicherten Checkpointbytes (fälschlich 0 B). Export v2 trennt
beide Größen; die nativen Rohberichte sind unverändert.

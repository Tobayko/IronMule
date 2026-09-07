# PROD4 — Ladeüberwachung und sauberer Worker-Shutdown

IronMule überwacht die Modellladung während der Kalibrierung jetzt mit realen
RSS-/Swap-Probes und anschließend mit tatsächlichen MLX-/Prozess-Peaks. Ein
nachgewiesener Python-Shutdown-Abort ist repariert. Die Standard-Generierung
wurde nicht optimiert; keine neue Performancefreigabe, kein RL-Abschluss.

## Installierter Hardwarestand

M1 Max, 32 GiB RAM; MLX 0.32.0, MLX-LM 0.31.3, NumPy 2.5.2,
Transformers 5.15.1. Getestet wird das gebaute Wheel in einer unabhängigen
temporären Installation von außerhalb des Entwicklerverzeichnisses.
Keine Abhängigkeit der Projektumgebung wurde verändert.

Nach dem Shutdown-Fix hat der installierte Code den SHA-256
`0f0502605e9a25aa34ac2cbb895825b78f2887b88e93b1496dc313358f4d3532`.
Die Vorregistrierung und nachträgliche, ausdrücklich markierte Exit-Gate-Korrektur
stehen in [PROD4_LOAD_MEMORY_SPEC.md](PROD4_LOAD_MEMORY_SPEC.md).

## Load-only-Ergebnisse

Ein frischer Worker pro Durchgang, drei Readiness-Beobachtungen im Abstand von
mindestens 5 s vor der Ladung und erneut nach Hashing, überwachte 4-s-Pause und
Readiness nach Ladung. Mindestens 60 s
zwischen geschlossenen Modellworkern. Keine Generierung in diesem Versuch.
Bytes sind exakt; einzelne Ladezeiten sind deskriptiv, kein Leistungsvergleich.

| Modell / Versuch | MLX-Peak (B) | Prozess-Peak (B) | größtes beobachtetes Swapdelta (B) | Exit | Bewertung |
|---|---:|---:|---:|---:|---|
| 1B / 1, vor Fix | 735.852.808 | 1.697.595.392 | 0 | -6 | Shutdown fehlerhaft |
| 1B / 2, nach Fix | 735.852.808 | 1.701.330.944 | 0 | 0 | Load-only bestanden |
| 4B / 1, nach Fix | 2.560.801.800 | 2.735.194.112 | 0 | 0 | Load-only bestanden |
| 12B / 1, nach Fix | nicht erreicht | nicht erreicht | 468.587.643 | -15 | vor `ready` kontrolliert abgebrochen |

Die bestandenen 1B-/4B-Läufe haben unveränderte Modell-/Umgebungs-/Codeidentität
vorher/nachher sowie übereinstimmende Quellmanifeste. RSS und MLX werden nicht
addiert; diese Messgrößen sind keine unabhängigen Speicherreservierungen.

Der 12B-Versuch `f61b8b980aa841d3bfc4c7a5ccb9bdbd` zeichnet 14 Samples auf.
Vorlade-Swap: 4.021.614.018 B. Die letzten zwei Beobachtungen steigen von
+51.579.454 B auf +468.587.643 B, bei rund 288 ms Abstand zwischen Sampleenden.
Die zweite verletzt das unveränderte 268.435.456-B-Gate. Der eigene PID 12025
wurde mit SIGTERM beendet, `exit_class=forced_abort`; kein `ready`, keine
Generierung. Start bis festgestellte Verletzung: etwa 3,60 s; bis reaped: 4,06 s.
Der letzte Prozess-RSS beträgt nur 792.756.224 B. Das ist gerade **kein** Beleg
für ausreichend freien Modell-RAM: RSS ist kein vollständiger MLX-Footprint,
und der Swapwert ist systemweit. Die genaue Lade-Allokationsursache bleibt offen.

Die Quellen vor/nach dem abgebrochenen Versuch sind identisch; eine vollständige
Modell-/Umgebungs-Nachqualifikation wird für diesen Fehlerlauf nicht behauptet.
Der Lauf belegt den früheren Abbruchpfad, keine harte Speichergarantie und keinen
gepaart nachgewiesenen Rückgang des Speicherverbrauchs gegenüber PROD3.

## Der tatsächliche Defekt

Der erste 1B-Lauf speichert den realen Exitcode -6, obwohl der frühere Treiber
allein wegen erfüllter Speichergrenzen und beendeter PID `passed` meldet. Diese
Rohdatei bleibt unverändert; sie wird hier ausdrücklich nicht als sauberer
Produkttest gewertet. Eine echte Pipe-/Prozessreproduktion auf dem Reader-Pfad
liefert `_enter_buffered_busy`: ein Daemon-Thread wartet beim Interpreter-Exit
noch auf gepuffertem stdin. Es wird kein passender Crashlog für die Modell-PID
behauptet; der Fehlertext stammt aus der kontrollierten Reproduktion.

Der Reader beendet sich jetzt nach Shutdown/EOF, ein terminaler Queueeintrag
kann nicht an einer vollen Queue hängen bleiben, und der Hauptthread joint den
Reader begrenzt. Normale Kalibrierungsabschlüsse verlangen Exitcode 0. Auch der
unabhängige Evaluator prüft diesen Nachweis bei der neuen Load-Monitor-Version;
erzwungene Fehlerabbrüche bleiben separat mit ihrem ursprünglichen Grund sichtbar.

Die Gesamtsuite nach Reparatur: **897 passed, 16 deselected**, 51,74 s.
Zwei reale Pipe-Lifecycle-Tests bestehen zusätzlich zehn Wiederholungen (20/20).
Ein zuvor erkanntes Rennen im Full-Queue-Testaufbau und sein fehlgeschlagener
Suitenlauf sind im Arbeitsjournal festgehalten. Ruff-F und Wheelbau bestehen.

Ein separates Read-only-Review verifiziert die lokale Journal-Kette über alle
269 Ereignisse und gleicht alle vier Rohberichte mit ihren tatsächlichen
Samples/Identitäten/Worker-Enden ab. Die Worker-Abstände betragen 520,818 s,
167,831 s und 131,967 s. Keine privaten Pfade, Nutzernachrichten oder Zugangsdaten
wurden in den geprüften Rohberichten/Journal-Strings gefunden.

## Grenzen

- Polling erkennt eine beobachtete Verletzung, reserviert aber keinen System-RAM
  und verhindert keine Auslagerung zwischen zwei Messpunkten.
- Systemweiter Swap ist kein allein dem Modell zurechenbarer Verbrauch.
- Gewichtsdateigröße, MLX-Active und Prozess-RSS sind verschiedene Größen.
  Load-only ist keine Messung des vollständigen Inferenz-/KV-Speicherbedarfs.
- Der ältere installierte 12B-Swapfehler bleibt gültige historische Evidenz;
  andere heutige Lastbedingungen wären kein Beleg, dass der Guard Speicher spart.
- Aktivierung, Generierungsqualifikation, breitere Admission, automatisches
  Neuplanen, RL und Multi-Mac-Betrieb sind durch diesen Schnitt nicht erledigt.

Rohdaten: `research/raw/PROD4_1B_load_20260907_attempt1.json`,
`research/raw/PROD4_1B_load_20260907_attempt2.json`,
`research/raw/PROD4_4B_load_20260907_attempt1.json`.
Hinzu kommt `research/raw/PROD4_12B_load_20260907_attempt1.json`.

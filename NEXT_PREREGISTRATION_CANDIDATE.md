# Nächster prospektiver Kandidat

Stand: 24. August 2026, nach Zyklus 2. Genau einer. Noch **kein** `formal_claim=true`.

## Vorbemerkung: die Liste ist kürzer geworden

Zyklus 2 hat den Kandidaten `chunk_identity_policy` terminal beendet. Über `23`
geprüfte Zellen veränderte die Prefill-Zerteilung in `26 %` der Fälle die erzeugten
Token, und **keine** Blockgröße hielt zuverlässig — `256` hielt sieben von acht
Längen und fiel bei `1513`.

Damit bleiben vier Kandidaten blockiert, weil alle die Blockstruktur verändern:
Präfix-Wiederverwendung (`13,0x` TTFT), Prefill-Step-Size-Sweep, Microbatching,
Continuous Batching.

## Empfehlung: persistenter Modellprozess

Der einzige verbliebene Kandidat mit messbarer Wirkung, der die Numerik **gar nicht
berührt**.

**Mechanismus.** Das Modell wird einmal geladen und über viele Anfragen gehalten.
Gemessen kostet das Laden `1,47`–`1,76` s. Bei einer Anfrage mit `898` Token und
`1,70` s Prefill verdoppelt ein Kaltstart die Zeit bis zum ersten Token annähernd.

**Warum die Korrektheit hier nicht in Gefahr ist.** Es ändert sich weder Blockgröße
noch Batchbreite noch Cache-Struktur — nur, wie oft der Prozess startet. Die
Tokenidentität ist trivialerweise gegeben, und genau das macht ihn nach zwei
gescheiterten Korrektheitsgates zum richtigen nächsten Schritt.

**Erwarteter Endpunkt.** `cold_process`-TTFT gegen `warm_uncached`-TTFT. Erwartung
aus vorhandenen Messungen: rund `1,5` s Unterschied, also grob eine Halbierung der
TTFT bei kurzen bis mittleren Prompts. Die Zahl ist **abzuleiten, nicht zu
behaupten** — Prozessstart, Import und Speicherlage sind bisher nicht getrennt
gemessen.

**Aufwand.** Gering. Der Messteil existiert; es fehlt eine saubere Trennung von
Prozessstart, Import, Modellladen und Warm-up.

**Abbruchbedingung.** Wenn die Trennung zeigt, dass der Import und nicht das
Modellladen dominiert, ist der Kandidat für die Runtime uninteressant und endet als
`candidate_characterized`.

## Die Entscheidung, die dem Nutzer gehört

Der Korrektheitsvertrag verlangt identische Token-IDs. Zyklus 2 zeigt, dass dieser
Vertrag für jede Optimierung, die die Prefill-Zerteilung verändert, auf dieser
Plattform **nicht erfüllbar** ist.

Drei Wege, und keiner davon ist meiner:

1. **Vertrag halten.** Dann bleiben Präfix-Cache, Microbatching und Continuous
   Batching dauerhaft gesperrt, und die `13,0x` TTFT sind nicht abrufbar.
2. **Vertrag präzisieren.** Etwa: Identität nur innerhalb einer festen
   Ausführungskonfiguration verlangen, nicht zwischen verschiedenen. Dann wären die
   Kandidaten zugänglich, aber zwei Läufe derselben Anfrage in verschiedenen Modi
   könnten verschiedene Texte liefern.
3. **Vertrag durch eine Verteilungsaussage ersetzen.** Etwa: gleiche Verteilung statt
   gleicher Token. Das ist bei greedy Sampling schwer zu prüfen und öffnet genau die
   Tür, die der Auftrag ausdrücklich schließen wollte.

Ich empfehle keine dieser Optionen. Ich habe das Problem vermessen und lege es vor.

## Verhältnis zu BW1

`docs/BW1_VORREGISTRIERUNG.md` bleibt unversiegelt und gültig. Zyklus 2 verschärft
seine Lage: dass formabhängige Numerik die Ausgabe verändert, ist jetzt zweifach
belegt — beim Prefill (Zyklus 2) und bei `mx.compile` (frühere Runde). Das
Korrektheitsgate von BW1 ist damit der wahrscheinliche Ausgang, nicht das Risiko.

# Nächster prospektiver Kandidat

Stand: 24. August 2026. Genau einer. Noch **kein** `formal_claim=true`.

## Empfehlung: Blockgrößen-Policy für Tokenidentität

**Nicht** der Präfix-Cache, obwohl er mit `13,0x` die größte gemessene Wirkung dieser
Sitzung hat. Er ist in Zyklus 1 am Korrektheitsgate gescheitert, und die Ursache liegt
nicht in ihm, sondern unter ihm.

## Warum dieser zuerst

Gemessen, ohne jeden Präfix-Cache: dieselben `677` Token, nur anders gestückelt,
erzeugen unterschiedliche Ausgaben.

| Zerteilung | identisch zum Einzelblock |
| :--- | :--- |
| `512` + `165` | **nein**, ab Position 10 |
| `666` + `11` | ja |
| `256`er | ja |
| `128`er | **nein**, ab Position 10 |

Vier Kandidaten der Liste hängen daran, weil alle die Blockstruktur verändern:
Präfix-Wiederverwendung, Prefill-Step-Size-Sweep, Microbatching, Continuous Batching.
Solange offen ist, welche Zerteilungen die Ausgabe erhalten, kann keiner von ihnen den
Korrektheitsvertrag erfüllen.

## Zu klärende Frage

Existiert eine Blockgröße — oder eine von der Gesamtlänge abhängige Regel —, unter der
das Prefill in mehreren Blöcken tokenidentisch zum Einzelblock bleibt?

Drei mögliche Ausgänge, alle verwertbar:

1. **Es gibt eine.** Dann bekommt die Architektur eine harte Regel, und die vier
   blockierten Kandidaten werden wieder zugänglich.
2. **Es gibt keine.** Dann ist Tokenidentität über verschiedene Prefill-Zerteilungen
   auf dieser Plattform grundsätzlich nicht herstellbar, und der Korrektheitsvertrag
   des Auftrags braucht eine bewusste Entscheidung des Nutzers — nicht eine stille
   Aufweichung durch mich.
3. **Sie hängt vom Modell ab.** Dann gehört sie ins Profil, wie Breite, Sample-Zahl
   und Nachschlagfenster zuvor.

Ausgang 3 ist nach dem bisherigen Muster der wahrscheinlichste: viermal in dieser
Sitzung erwies sich ein für konstant gehaltener Wert als modellabhängig.

## Warum noch keine versiegelte Vorregistrierung

Die bisherige Evidenz ist ein einziger Prompt bei einer Länge, fünf Zerteilungen, ein
Modell. Das reicht, um ein Problem zu belegen, nicht um eine Regel zu registrieren.
Vor einer Versiegelung fehlen mindestens: mehrere Promptlängen, beide lokalen Modelle,
und eine Prüfung, ob die abweichenden Breiten mit den bereits bekannten
Kernel-Regressionen (`6`–`9`, `48`) zusammenfallen.

## Verhältnis zu BW1

Der Entwurf `docs/BW1_VORREGISTRIERUNG.md` (Batch-Decode) bleibt gültig und
unversiegelt. Zyklus 1 stärkt allerdings ausdrücklich seinen Abschnitt 3: Bündelung
verändert Kernelformen, und dass genau das die Ausgabe verändern kann, ist jetzt
unabhängig vom Batching belegt. Das Korrektheitsgate von BW1 ist damit nicht eine
Formalie, sondern der wahrscheinlichste Ausgang.

# Mini-Vorregistrierung — Zerlegung des Kaltstarts

**Kandidaten-ID:** `persistent-process-20260824-01`
**Zyklus:** 3 · **Status:** vor der Messung geschrieben · `formal_claim=false`

## Hypothese

Die `cold_process`-TTFT wird vom **Modellladen** dominiert, nicht vom Interpreterstart
oder den Imports. Ein persistenter Prozess entfernt damit einen erheblichen und
konstanten Anteil der wahrgenommenen Latenz.

## Warum dieser Kandidat

Nach zwei terminalen Korrektheitsfehlschlägen (Präfix-Cache in Zyklus 1,
Blockgrößen-Policy in Zyklus 2) ist er der einzige verbliebene Kandidat mit messbarer
Wirkung, der **weder Blockgröße noch Batchbreite noch Cache-Struktur** verändert.
Tokenidentität ist trivialerweise gegeben.

## Genau geänderte Variable

Keine. Dieser Zyklus **misst nur** und ändert am Ausführungspfad nichts. Gemessen
wird die Zerlegung eines Kaltstarts.

## Zu trennende Anteile

1. Interpreterstart bis erste ausführbare Zeile
2. `import mlx.core`
3. `import mlx_lm`
4. Snapshot-Auflösung
5. Modellladen
6. erster Forward (Warm-up)
7. zweiter Forward (eingeschwungen)
8. Prefill des Testprompts bis erstes Token

Die Anteile werden **nicht** in einem gemeinsamen Schätzer vermischt. Jeder wird
einzeln berichtet.

## Workload

Prompt mit `~900` Token, greedy, ein Token erzeugt. Drei frische Prozesse; jeder
Prozess misst genau einen Kaltstart und beendet sich.

## Primärer Endpunkt

Anteil des Modellladens an der Summe der Schritte 1–6, als Median über drei Prozesse.

## Sekundäre Endpunkte

Absolutwerte je Anteil; Differenz zwischen erstem und zweitem Forward als Maß des
Warm-up-Aufwands; Prozess-RSS nach dem Laden.

## Mindestwirkung

Der Kandidat gilt als **lohnend**, wenn Modellladen plus Warm-up zusammen mindestens
`30 %` der `cold_process`-TTFT ausmachen. Darunter ist ein persistenter Prozess für
die Runtime uninteressant.

## Abbruchbedingung, vorab festgelegt

Dominieren Interpreterstart und Imports statt des Modellladens, endet der Kandidat als
`candidate_characterized` und **nicht** als Empfehlung. Ein persistenter Prozess
würde dann etwas entfernen, das ohnehin klein ist.

## Ressourcengrenzen

`BudgetGuard` je Teilprozess. Modellladen ist überwiegend Datenträger- und
Speicherarbeit und wird **nicht** als GPU-Last verbucht; nur Forward-Läufe werden
verbucht.

## Auswertung

Tabelle der Anteile, Median über drei Prozesse, plus die Streuung. Keine Aggregation
über die Klassen hinweg.

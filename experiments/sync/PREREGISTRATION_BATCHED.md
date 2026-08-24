# Mini-Vorregistrierung — gebündelter Host-Readback mit Prüfintervall

**Kandidaten-ID:** `batched-readback-20260824-01`
**Zyklus:** 7 · **Status:** vor der Messung geschrieben · `formal_claim=false`
**Vorgänger:** `host-sync-20260824-01` (Zyklus 6) zeigte `15,3 %` Ersparnis bei
vollständig entfallendem Readback — ein Arm, der nicht anhalten kann. Dieser Kandidat
misst die tatsächlich abrufbare Form.

## Zwei gegenläufige Effekte

Prüft man alle `N` Schritte statt jeden:

- **Gewinn:** der Readback fällt nur noch `1/N`-mal an;
- **Verlust:** trat das Stop-Token mitten im Intervall auf, wurden bis zu `N−1`
  Token umsonst erzeugt.

Eine Messung, die nur den Gewinn erfasst, überschätzt den Kandidaten. Beide Größen
gehören in die Auswertung.

## Hypothesen

**H1 (Korrektheit).** Bei fester Schrittzahl erzeugt jedes Intervall `N` dieselben
Token wie `N=1`.

**H2 (Wirkung).** Es existiert ein `N > 1`, das die Schrittzeit um mindestens `8 %`
gegenüber `N=1` senkt.

Die Schwelle liegt bewusst unter den `15,3 %` aus Zyklus 6 — jene sind die Obergrenze
bei `N → ∞`, und ein brauchbares `N` erreicht sie nicht.

## Genau geänderte Variable

Das Prüfintervall `N`. Sonst nichts.

## Workload

`128` Decode-Schritte nach `~900`-Token-Prefill, greedy, Batch `1`.
`N ∈ {1, 2, 4, 8, 16, 32}`. Zwei Wiederholungen je Arm, Arme abwechselnd, Median.
Aufwärmlauf vorab, verworfen.

Feste Schrittzahl ohne vorzeitiges Anhalten, damit der Gewinn **rein** gemessen wird.
Der Verlust wird nicht mitgemessen, sondern aus den gemessenen Größen **abgeleitet**
und als solcher ausgewiesen.

## Primärer Endpunkt

Sekunden je Token bei Intervall `N`, relativ zu `N=1`.

## Sekundärer Endpunkt, abgeleitet

Die Generierungslänge, ab der ein Intervall netto lohnt. Erwarteter Überlauf ist
`(N−1)/2` Token; er kostet die volle Schrittzeit. Der Gewinn beträgt
`(1 − 1/N)` mal der Readback-Kosten je Schritt mal der Länge.

Break-even-Länge: `L > ((N−1)/2 × t_Schritt) / ((1 − 1/N) × t_Readback)`

Diese Größe wird **berechnet, nicht gemessen**, und im Bericht als Ableitung
gekennzeichnet.

## Abbruchregeln

Budgetverletzung beendet den Lauf, Teilergebnis wird persistiert. Kein Retry im
selben Prozess. Keine nachträgliche Änderung der Schwelle.

## Vorab festgelegte Deutung

| H1 | H2 | Entscheid |
| :--- | :--- | :--- |
| hält | hält | `candidate_recommended_for_preregistration` |
| hält | verfehlt | `candidate_characterized` — Bündelung lohnt nicht |
| scheitert | egal | `candidate_correctness_failed`, terminal |

Ein Scheitern von H1 wäre schwerwiegend: das Intervall ändert nur, wann gelesen wird,
nicht was gerechnet wird. Träte dennoch eine Abweichung auf, wäre auf dieser Plattform
kein alternativer Ausführungspfad mehr konstruierbar, der den Vertrag erfüllt.

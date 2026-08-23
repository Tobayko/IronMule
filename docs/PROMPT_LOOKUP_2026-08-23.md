# Verlustfrei schneller: spekulieren ohne Draft-Modell

**Stand:** 23. August 2026 · **Status:** explorativ, `formal_claim=false`.
Werkzeug: `tools/measure_prompt_lookup.py`, Berichte unter
`experiments/prompt_lookup/`.

## 1. Warum das nach dem gescheiterten Versuch noch einmal geprüft wurde

Spekulatives Decoding mit dem 1B als Entwurfsmodell wurde hier mit `0,560x` gemessen
und verworfen. Der Grund war nicht das Verfahren, sondern der Preis des Entwurfs: das
1B kostet `0,46` eines 4B-Schritts und traf in `39 %` der Fälle.

Ein Entwurf muss aber nicht aus einem Modell kommen. Vieles von dem, was ein Modell
schreibt, hat es vorher gelesen — ein Bezeichner, ein Pfad, eine zitierte Zeile, eine
wiederholte Struktur. Die letzten `n` Token lassen sich im Kontext nachschlagen und
das, was dort folgte, als Fortsetzung vorschlagen. Der Entwurf ist eine Textsuche auf
der CPU und kostet nichts Messbares.

**Und die Ausgabe bleibt bitgleich.** Ein vorgeschlagenes Token wird nur behalten,
wo es dem entspricht, was das Modell ohnehin erzeugt hätte. Am Modell wird nichts
verändert, nichts quantisiert, nichts genähert. Es verliert kein Wissen.

## 2. Break-even aus der gemessenen Kurve

Die Verifikation von `k` Entwurfstoken ist ein Pass der Breite `k+1`. Was der kostet,
steht in der Breitenkurve — und die ist je Modell verschieden:

| `k` | Verify-Breite | 4B-Kosten | 4B-Break-even | 1B-Kosten | 1B-Break-even |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | `1,49x` | `0,47` | `1,24x` | **`0,24`** |
| 2 | 3 | `1,91x` | `0,65` | `1,48x` | **`0,35`** |
| 3 | 4 | `2,51x` | `0,72` | `1,80x` | `0,47` |
| 4 | 5 | `2,90x` | `0,75` | `2,08x` | `0,54` |

Mit dem 1B-Entwurfsmodell lag die zusätzliche Hürde bei `+0,46` je Entwurfstoken. Mit
einem kostenlosen Entwurf verschwindet sie, und aus einer unerreichbaren Schwelle wird
eine gewöhnliche.

## 3. Gemessen

`96` Token, greedy, `3`-Gramm-Nachschlag, alle Arme gegen den greedy-Lauf verglichen.

**Code umschreiben** (Funktion mit umbenanntem Parameter zurückgeben):

| `k` | 4B Sek. | 4B Tok/Schritt | Akzeptanz | 4B Speedup | 1B Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1,589` | `1,01` | – | `1,000` | `1,000` |
| 1 | `1,364` | `1,75` | `93,2 %` | `1,165` | `1,330` |
| **2** | **`1,308`** | **`2,34`** | **`93,3 %`** | **`1,215`** | **`1,695`** |
| 3 | `1,315` | `2,74` | `87,5 %` | `1,208` | `1,669` |
| 4 | `1,352` | `3,00` | `77,4 %` | `1,175` | `1,654` |

**Alle Arme lieferten Token für Token dieselbe Ausgabe wie greedy.**

**Freier Fließtext** (Absatz ohne Vorlage), als Gegenprobe:

| `k` | Sek. | Tok/Schritt | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | `1,386` | `1,01` | `1,000` |
| 2 | `1,329` | `1,08` | `1,042` |
| 4 | `1,378` | `1,08` | `1,006` |

`1,08` Token je Schritt heißt: fast kein Schritt fand überhaupt einen Treffer. Der
Nachschlag lief ins Leere, der Schritt war ein gewöhnlicher, und es kostete nichts.

## 4. Die Eigenschaft, auf die es ankommt

Der schlechteste in dieser Messreihe beobachtete Wert ist `1,006x`. **Nie langsamer,
manchmal deutlich schneller, immer bitgleich.** Findet die Suche nichts, ist der
Schritt ein normaler; das ist ein `if`, keine Wette.

Das unterscheidet es grundlegend vom Entwurfsmodell, das bei jedem Schritt bezahlt
wurde, ob es traf oder nicht — und deshalb `0,44` verlor.

## 5. Vorhersage gegen Messung

Der Wrapper (`HardwareProfile.speculation_speedup`) rechnet aus Breitenkurve und
Akzeptanz:

| Modell | `k` | vorhergesagt | gemessen |
| :--- | ---: | ---: | ---: |
| 4B | 2 | `1,357` | `1,215` |
| 4B | 3 | `1,277` | `1,208` |
| 1B | 2 | `1,897` | `1,695` |
| 1B | 3 | `1,835` | `1,669` |

Die Vorhersage ist durchgehend um rund `10`–`15 %` zu optimistisch, und der Grund ist
bekannt: sie unterstellt, dass **jeder** Schritt einen Entwurf findet. Real fand nicht
jeder einen, und ein Schritt ohne Entwurf liefert genau ein Token. Als obere Schranke
ist die Zahl brauchbar, als Prognose ist sie zu hoch.

## 6. Warum das auf einem Telefon mehr bringen sollte

Der Gewinn hängt daran, wie flach die Breitenkurve ist — und die wird flacher, je
stärker ein Gerät bandbreitengebunden ist. Bei Breite `2` werden die Gewichte **einmal**
gelesen, genau wie bei Breite `1`; nur die Rechenarbeit verdoppelt sich, und die ist
dort ohnehin nicht der Engpass.

Das 1B ist auf diesem Laptop zu `73 %` dispatch-gebunden und erreicht `1,695x`. Auf
einem Gerät, wo dasselbe Modell zu `72 %` bandbreitengebunden ist, müsste die Kurve
noch flacher liegen und der Break-even weiter fallen.

Das ist eine **Vorhersage aus dem Gerätemodell, keine Messung**. Sie ist auf keinem
zweiten Gerät geprüft.

## 7. Grenzen

Zwei Prompts, ein Gerät, zwei Modelle, greedy Sampling. Die Akzeptanzrate ist eine
Eigenschaft der Aufgabe, nicht des Verfahrens: `93 %` beim Umschreiben von Code,
praktisch keine Treffer bei freiem Text. Für einen Agenten mit Werkzeugausgaben,
Dateipfaden und wiederholten Strukturen sollte sie eher hoch liegen — gemessen ist das
hier nicht.

Der `3`-Gramm-Nachschlag ist die einfachste denkbare Variante. Größere Fenster,
mehrere Kandidaten gleichzeitig oder ein Baum statt einer Kette sind ungeprüft.

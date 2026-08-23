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

---

# Nachtrag: das Nachschlagfenster, und wo „nie langsamer" nicht gilt

**Ergänzt:** 23. August 2026. **Dieser Abschnitt schränkt Abschnitt 4 ein.**

## 8. Agentenkontext

Der bisherige Code-Fall lässt eine Funktion nahezu unverändert zurückgeben — die
freundlichste denkbare Aufgabe für einen Nachschlag. Ein realistischerer Kontext
(Verzeichnislisting, Quelltextauszug, Testausgabe, dann eine Änderung an einer
Dataclass) fällt schwächer aus, aber positiv:

| Modell | Fenster | `k` | Token/Schritt | Akzeptanz | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 4B | 3 | 4 | `1,746` | `76,9 %` | `1,098` |
| 4B | 8 | 4 | `1,600` | `97,2 %` | `1,133` |

Die Akzeptanz ist hoch, die Trefferhäufigkeit niedriger als beim reinen Umschreiben:
weniger von der Ausgabe stand schon im Kontext.

## 9. Das Fenster ist der wichtigste Parameter — und `n=1` schadet

Sweep über die Fensterlänge, Agentenkontext, 4B, jeweils bestes `k`:

| Fenster | `k` | Akzeptanz | Speedup |
| ---: | ---: | ---: | ---: |
| **1** | 1 | `45,5 %` | **`0,987`** |
| 2 | 3 | `59,1 %` | `1,076` |
| 3 | 4 | `76,9 %` | `1,092` |
| 4 | 2 | `88,9 %` | `1,105` |
| 5 | 3 | `89,7 %` | `1,123` |
| 6 | 4 | `90,0 %` | `1,125` |
| **8** | 4 | **`97,2 %`** | **`1,133`** |

**Bei `n=1` ist das Verfahren langsamer als gar nicht zu spekulieren.** Ein einzelnes
Token trifft überall und sagt fast nichts; die Akzeptanz von `45,5 %` liegt unter der
Break-even-Schwelle von `0,47`, die die gemessene Breitenkurve für `k=1` setzt.

Damit ist die Aussage aus Abschnitt 4 einzuschränken: **"nie langsamer" gilt nicht
unbedingt, sondern nur bei ausreichend langem Fenster.** Ein zu kurzes Fenster kauft
Entwürfe, die man verwirft, und bezahlt sie trotzdem.

## 10. Das beste Fenster hängt vom Modell ab

| Fall | `n=3` | `n=8` |
| :--- | ---: | ---: |
| Code umschreiben, 4B | `1,215` | **`1,274`** |
| Code umschreiben, 1B | **`1,695`** | `1,538` |

Das längere Fenster hilft dem 4B und **schadet** dem 1B. Der Grund steht in der
Breitenkurve: beim 1B kostet Tiefe wenig (Breite `2` nur `1,24x`), also zahlt sich
dort Trefferhäufigkeit mehr aus als Treffergenauigkeit. Beim steileren 4B ist es
umgekehrt.

Das Fenster gehört damit in dieselbe Kategorie wie die Breiten-Policy: **je Modell zu
messen, nicht zu setzen.**

## 11. Kein Treffer, keine Kosten — sauber gezeigt

Das 1B fand im Agentenkontext bei Fenster `8` über `96` Token **keinen einzigen**
Treffer:

| `k` | gedraftet | Speedup |
| ---: | ---: | ---: |
| 0 | `0` | `1,000` |
| 1 | `0` | `0,979` |
| 2 | `0` | `1,020` |
| 3 | `0` | `1,004` |
| 4 | `0` | `0,999` |

Null Entwürfe, und die Zeiten streuen um `1,0` ohne Richtung. Die Streuung von
`±2 %` ist Messrauschen. Der Fall, in dem der Nachschlag nichts findet, kostet
nichts — was bei `n=1` gerade **nicht** gilt, weil dort etwas gefunden und verworfen
wird.

## 12. Warum kein Kandidatenbaum gebaut wurde

Naheliegend wäre, mehrere Fortsetzungen gleichzeitig zu prüfen: das Breitenplateau
macht `32` Positionen kaum teurer als `8`. Gerechnet lohnt es hier trotzdem nicht.

Der Gewinn eines Baums ist am größten, wenn die Akzeptanz **niedrig** ist — er hedgt
gegen falsche Entwürfe. Ein längeres Fenster hebt die Akzeptanz im Agentenfall aber
bereits von `76,9 %` auf `97,2 %`, und gegen `97 %` hat ein zweiter Kandidat kaum noch
etwas zu gewinnen. Gleichzeitig kostet ein Baum von `32` Knoten auf dem 4B `5,97x`
eines einfachen Schritts und müsste rund sechs Token je Schritt liefern; gemessen sind
`2,34`.

Das längere Fenster ist damit der billigere Hebel für dasselbe Ziel, und er ist
gemessen. Ein Baum bleibt **ungeprüft**, nicht widerlegt — die Rechnung spricht nur
nicht dafür, ihn als Nächstes zu bauen.

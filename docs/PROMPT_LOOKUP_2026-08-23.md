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

---

# Korrektur: echter Projektinhalt, saubere Methodik, nüchternere Zahlen

**Ergänzt:** 23. August 2026. **Dieser Abschnitt hebt die Zahlen aus Abschnitt 3
und 10 auf.** Sie bleiben stehen, damit die Korrektur nachvollziehbar ist.

## 13. Zwei Fehler, beide meine

**Ein Korrektheitsfehler.** `trim_prompt_cache` meldet mit einem Rückgabewert von
`0`, dass es nicht zurückrollen konnte — es wirft nicht. Dieser Rückgabewert wurde
ignoriert. Verworfene Entwurfstoken blieben damit im Cache stehen und jeder folgende
Token wurde gegen einen Kontext gerechnet, den es nie gab.

Sichtbar wurde er erst an echtem Inhalt. Gemma 3 hält die meisten Layer in einem
rotierenden Cache; `RotatingKVCache.is_trimmable` ist `offset < max_size`, und das
Fenster misst **`512`** beim 1B und **`1024`** beim 4B. Die synthetischen Prompts
lagen bei `150`–`284` Token und damit innerhalb beider Fenster. Die echten liegen bei
`749` und `859` — beim 1B außerhalb. Genau dieses Muster zeigten die Läufe: das 1B wich
bei den beiden langen Prompts ab und stimmte beim `159`-Token-Prompt.

Es war **kein numerischer Gleichstand**. An der ersten abweichenden Stelle führte das
gewinnende Logit um `0,344`, und zwei greedy-Läufe desselben Prompts stimmten
miteinander überein.

Der Generator prüft jetzt vor jedem Entwurf, ob der Cache zurückrollbar ist, und macht
sonst einen gewöhnlichen Schritt. Ein zu kurzer Rückrollvorgang nach bestandener
Prüfung wirft, statt weiterzurechnen.

**Ein Messfehler.** Die erste Messung auf echtem Inhalt hatte **keinen Aufwärmlauf**.
Der erste Lauf nach dem Modellladen zahlt Allokation und Kernelaufbau, und das kam als
`1,539x` Speedup heraus — in einem Lauf, in dem wegen des rotierenden Caches
**überhaupt nicht spekuliert wurde**. Ein Speedup ohne Spekulation ist ein Messfehler
und nichts sonst.

## 14. Echter Projektinhalt, korrigiert

Aufwärmlauf, zwei Wiederholungen, Median, Arme abwechselnd. Prompts aus dem Repository
selbst: eine Quelldatei mit Änderungswunsch, echte Testausgabe mit Rückfrage, ein
Abschnitt des Arbeitsjournals mit Extraktionsauftrag.

| Modell | Prompt | Kontext | Akzeptanz | fest | adaptiv | zurückgefallen |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 4B | Quelldatei | `859` | `1,000` | **`1,096`** | `1,092` | `0` |
| 4B | Journal | `749` | `0,455` | `0,992` | `0,995` | `0` |
| 4B | Testausgabe | `159` | `0,333` | `0,974` | `0,986` | `0` |
| 1B | Quelldatei | `859` | – | `1,008` | `1,009` | **`63`** |
| 1B | Journal | `749` | – | `1,006` | `1,000` | **`63`** |
| 1B | Testausgabe | `159` | `0,700` | **`1,079`** | `1,070` | `0` |

Streuung zwischen Wiederholungen: `0,12`–`1,83 %`.

Drei Dinge stehen darin, und keines davon ist so gut wie Abschnitt 3 nahelegte.

**Der Gewinn ist kleiner.** Auf echtem Inhalt `1,096x` beim 4B statt der `1,215x`
aus dem konstruierten Beispiel. Jener Prompt bat darum, eine Funktion nahezu
unverändert zurückzugeben — die günstigste denkbare Aufgabe.

**Das 1B kann jenseits von `512` Kontext-Token nicht spekulieren.** Bei zwei der drei
echten Prompts fiel es in **jedem** Schritt zurück. Das gemessene `1,695x` aus
Abschnitt 3 stammt von einem `150`-Token-Prompt und gilt nicht für Agentenkontexte,
die typischerweise länger sind. Immerhin kostet der Rückfall nichts: `1,006`–`1,008x`.

**"Nie langsamer" ist endgültig widerlegt.** Bei Akzeptanz `0,333` misst der feste
Entwurf `0,974x`. Die Laufzeitanpassung holt davon einen Teil zurück (`0,986x`), aber
nicht alles.

## 15. Laufzeitanpassung der Entwurfstiefe

`speculative_generate(..., adapt=True)` führt einen exponentiell gewichteten
Akzeptanzschätzer und wählt je Schritt die tiefste Entwurfslänge, die bei diesem Wert
laut gemessener Breitenkurve noch bezahlt. Fällt der Schätzer unter jede
Break-even-Schwelle, wird gar nicht entworfen.

Der Horizont ist kurz gewählt (`memory=0,7`). Bei `0,9` brauchte der Schätzer rund
dreißig Entwurfsschritte, um von seinem optimistischen Start zu fallen — bei einer
kurzen Antwort ist das der ganze Lauf. Mit dem kürzeren Horizont lehnte er im
Journalfall `33` von rund `85` Schritten ab.

Der Gewinn daraus ist klein: `0,974` auf `0,986`, `0,992` auf `0,995`. Er ist dort
positiv, wo er gebraucht wird, und kostet dort nichts, wo Akzeptanz hoch ist
(`1,096` gegen `1,092`, innerhalb der Streuung).

## 16. Wofür es sich jetzt noch lohnt

Nach diesen Zahlen: für **Aufgaben, die Bestehendes umschreiben**, und dort mit rund
`10 %`. Für alles andere ist es ungefähr neutral, mit einem Verlust von bis zu `2,6 %`
im ungünstigsten gemessenen Fall.

Das ist erheblich weniger, als Abschnitt 3 nahelegte, und es ist die Zahl, die zählt —
die synthetischen Prompts waren von mir so gebaut, dass sie dem Verfahren
entgegenkamen.

---

# Nachtrag: die Länge der Übereinstimmung sagt die Akzeptanz vorher

**Ergänzt:** 23. August 2026.
`experiments/prompt_lookup/real/measure_match_length_signal.py`.

## 17. Das Signal

Bisher entschied ein festes Fenster beides: **ob** ein Entwurf gemacht wird und **wie
tief**. Das sind zwei verschiedene Fragen. Ein kurzes Fenster findet mehr Treffer; wie
weit die Übereinstimmung sich rückwärts fortsetzt, sagt, wie sehr man ihr trauen darf.

Gemessen auf den drei echten Projekt-Prompts, 4B, Suchfenster `3`, Entwurfstiefe `4`:

| Länge der Übereinstimmung | gedraftet | akzeptiert | Akzeptanz |
| :--- | ---: | ---: | ---: |
| 3–4 Token | `84` | `45` | `0,536` |
| 5–8 Token | `28` | `16` | `0,571` |
| 9–15 Token | `4` | `4` | **`1,000`** |
| 16+ Token | `44` | `44` | **`1,000`** |

**Übereinstimmungen ab neun Token wurden 48 von 48 Mal akzeptiert.** Kurze knapp zur
Hälfte. Das ist kein schwacher Zusammenhang, den man statistisch herausarbeiten müsste.

## 18. Die Regel, die daraus folgt

Kurzes Fenster suchen, Tiefe aus der gemessenen Trefferlänge. Bei den Break-even-Werten
des 4B (`0,47` für einen Entwurfstoken, `0,715` für drei) bedeutet das:

| Übereinstimmung | erwartete Akzeptanz | Tiefe |
| :--- | ---: | ---: |
| keine | – | `0` |
| 3–8 Token | `0,55` | `1` |
| ab 9 Token | `0,98` | `4` |

Eine Suche, zwei Entscheidungen.

## 19. Gemessen gegen das feste Fenster

Echte Projekt-Prompts, 4B, Median aus zwei Wiederholungen nach Aufwärmlauf:

| Prompt | festes Fenster `8` | **Trefferlänge** | Akzeptanz fest | Akzeptanz neu |
| :--- | ---: | ---: | ---: | ---: |
| Quelldatei | `1,099` | `1,097` | `1,000` | `0,970` |
| Journal | `0,997` | **`1,029`** | `0,417` | **`0,682`** |
| Testausgabe | `0,976` | **`0,994`** | `0,333` | `0,375` |

Die beiden Verlustfälle werden zu einem Gewinn und einem Fast-Nullsummenspiel, ohne
dass der gute Fall etwas abgibt (`1,099` gegen `1,097` liegt in der Streuung).

Der Journalfall zeigt den Mechanismus am deutlichsten: die Akzeptanz steigt von
`0,417` auf `0,682`, weil kurze Treffer jetzt einen Entwurfstoken bekommen statt drei.
Es werden nicht weniger Treffer benutzt — es wird weniger auf sie gesetzt.

Damit ist `by_match_length` die Voreinstellung, und die Profile suchen mit Fenster `3`
statt `8`. Der frühere Befund "Fenster `8` ist besser als `3`" galt nur, solange das
Fenster auch die Tiefe bestimmte.

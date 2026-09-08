# PROD10G — Vollcapture verworfen, GPU-Zeitdiagnose weiter offen

Der vollständige Metal-Capture liefert **keinen fertigen Profiling- oder
Geschwindigkeitsnachweis**. Zwei getrennte Versuche bleiben dokumentiert:

- Versuch1, Run `b1d144c453b94f0d846922063e75f1c0`: Der isolierte Pythonstart
  scheitert vor Modellladung an einem nicht auflösbaren Hilfsdateiimport.
  Worker70675 endet mit1, keine Antwort und keine Aufzeichnung. Der Import
  wurde explizit manifestgebunden repariert und der echte `python -I --help`-
  Start als Regression geprüft.
- Versuch2, Run `a47e6bb093584d3c8ac8f7479decb945`: Drei echte 12B-Stock-
  Warmups sind ausgabeidentisch zur gebundenen Referenz. Der anschließende
  vollständige Capture wurde nach über14Minuten instrumentierter Arbeit
  bewusst abgebrochen, um zur Originalzeitmessung per Metal System Trace zu
  wechseln. Controller endet mit1/KeyboardInterrupt, Worker71104 mit−15 und
  bestätigtem Cleanup. 918 Ressourcenbeobachtungen ohne Fehler und sämtliche
  fünf Vorher-/Nachherbindungen unverändert. Kein Modellfehler daraus ableiten.

Die private unvollständige Aufzeichnung belegt laut `du` rund6,7GiB. Das
historische Feld `capture.size_bytes=4477959520256` in Versuch2 ist **keine
gültige Speicherplatzangabe**: die damalige Summierung folgte631.458 Symlink-
Aliasen und zählte deren Dateiziele mehrfach. Metadatenprüfung bestätigt diesen
Mechanismus. Die neue Auswertungsfunktion folgt keinen Symlinks, dedupliziert
Hardlinks und trennt reguläre Payloadbytes von belegten Dateisystemblöcken;
echte temporäre Dateisystemtests sichern die Korrektur. Alte Rohdaten bleiben
unverändert. Die Aufzeichnung selbst wird wegen möglicher Modelldaten nicht
veröffentlicht.

Der methodische Unterschied bleibt entscheidend: `.gputrace` erlaubt Xcode-
Replayanalyse, nicht das nachträgliche Erfinden ursprünglicher GPU-Dauern aus
Hostzeiten. Der nächste eigenständige Ansatz muss echte Originalzeitstempel
des vollständigen Workloads erfassen. Weder der lange Capture noch sein
Abbruch begründet eine neue Kerneloptimierung oder eine neue Hardwaregrenze.

Quellen: [Spezifikation](PROD10_GPU_CAPTURE_SPEC.md),
[Versuch1](../research/raw/PROD10G_12B_capture_20260908_attempt1.json),
[Versuch2](../research/raw/PROD10G_12B_capture_20260908_attempt2.json).

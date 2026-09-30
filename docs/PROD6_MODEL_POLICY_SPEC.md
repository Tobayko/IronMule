# PROD6 — keine ausführbaren Modelldateien aus Snapshots

Vorregistrierung der nativen Prüfung, 2026-09-07. Sicherheits-/Kompatibilitäts-
reparatur innerhalb der bestehenden isolierten Produkt-Worker-Grenze, keine
neue Performanceoptimierung und keine Freigabe fremden Python-Codes.

## Nachgewiesener Pfad und Korrektur

Der installierte MLX-LM-0.31.3-Loader liest `config.json` und führt bei jedem
nicht-`None`-Wert von `model_file` die angegebene Python-Datei aus. Die vorhandene
Tokenizer-Option `trust_remote_code=False` sperrt diesen separaten Pfad nicht.
Die Produktgrenze wird deshalb an zwei Stellen durchgesetzt:

1. Vor MLX-Import und Modellladung wird die Konfiguration begrenzt gelesen und
   geprüft. Nicht-Objekte, doppelte Schlüssel, nicht-endliche Zahlen, Tiefe über
   64, übergroße oder instabile Dateien sowie Root-entkommende Links werden
   verworfen. HF-Links auf reguläre Blob-Dateien innerhalb desselben Modell-Repos
   bleiben zulässig. Jeder nicht-`None`-Wert von `model_file` wird zurückgewiesen.
2. Der öffentliche `mlx_lm.load`-Aufruf erhält zusätzlich ausdrücklich
   `model_config={"model_file": None}`. Im geprüften Loader überschreibt dies
   die erneut gelesene Konfiguration vor der Codeauswahl. Ein bloßer früher
   Dateicheck wäre wegen einer möglichen Änderung zwischen zwei Reads ungenügend.

Die Prüfung verändert keine Bibliotheksdateien, Gewichte, Tokenizer, Sampling-
oder Inferenzoperationen. Sie ist kein Schutz gegen bösartige bereits installierte
Bibliotheken oder einen Angreifer, der den Produktcode selbst ersetzen kann.
Andere Bibliotheksversionen erben keine hier gemessene Hardwarequalifikation.

## Prüfplan und Grenzen

- Echte Dateien/Pipes und ein Kindprozess ohne Site-Packages prüfen die Grenze
  vor MLX. Eine explizite Markerdatei darf durch ein zurückgewiesenes
  `custom.py` nicht entstehen. Diese Metadaten-Fixtures sind keine Modelltests.
- Alle drei vorhandenen unveränderten Gemma-Konfigurationen (1B, 4B, 12B) müssen
  den Metadatencheck bestehen. Das alleine ist keine Inferenzqualifikation.
- Neues Wheel bauen und nur in der unabhängigen Testinstallation ersetzen.
  Environment-/Codebindung vor/nachher dokumentieren; bestehende Modelle und
  versiegelte Studien bleiben unverändert.
- Für 1B folgt ein neuer installierter Durchgang des unveränderten, bereits
  registrierten PROD3-90-Anfragen-Protokolls. Damit wird gleichzeitig die noch
  offene vollständige 1B-Bestätigung bearbeitet. Exakte Ausgaben, reale Forward-
  Zähler, drei frische Worker, normale Exitcodes und alle Speicher-/Zeit-/Last-
  Gates bleiben zwingend. Keine neue Auswertungsschwelle und kein versteckter
  Retry nach ungültigem Lauf.
- Für 4B prüft ein eigener installierter Referenz-Regressionslauf Ladepfad und
  Ausgabe: ein Warmup plus drei greedy Acht-Token-Anfragen zum unveränderten
  Apples-Prompt, vollständiger Token-/Text-/Finish-/Zählervergleich gegen die
  tatsächlich gemessene Stock-Referenz aus PROD1-Screen Versuch 3. Die Revision
  muss exakt passen. Es werden nur Hashes der Ausgaben neu gespeichert.
  Lade-/Swap-/Readiness-Gates, begrenzte Anfragezeit, vier Sekunden Pause und
  sauberer Exit bleiben aktiv. Diese vor seiner Ausführung präzisierte Prüfung
  ersetzt den vorgesehenen Load-only-Schritt; das beantwortete 4B-Performance-
  protokoll wird nicht für einen günstigeren Zufallswert wiederholt.
- Der bekannte 12B-Swapfehler wird nicht für einen günstigeren Zufallswert erneut
  geladen. Seine nächste Hardwareprüfung braucht zusätzliche Ladephasen-
  Evidenz aus PROD4. PROD6 behauptet für 12B nur geprüfte Metadatenkompatibilität.

Kill: Snapshot-Python wird ausgeführt, eine Race umgeht den erzwungenen Override,
ein gültiger builtin-Gemma-Pfad divergiert, ein Worker bleibt bestehen oder ein
Ressourcenfehler wird verschwiegen. Kein Speedup wird dem Modellcode-Schutz
zugeschrieben; eine getrennte Kalibrierungsbewertung bleibt workloadgebunden
und aktiviert keinen Kandidaten.

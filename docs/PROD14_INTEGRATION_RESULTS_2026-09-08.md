# PROD14 — neue Worker-/HTTP-Einbindung auf allen lokalen Gemma-Modellen korrekt

Die beiden neuen Workerpfade bestehen den gezielten Integrationsscreen auf
Gemma3 1B, 4B und 12B: **45 neue vollständige Generierungen und drei echte
partielle Abbrüche**, keine wiederholte Stock-Inferenz. Alle sechs eigenen
Modellworker endeten mit Exit0.

## Was tatsächlich neu geprüft wurde

| Modell | Prefix-Worker | Bestehende Engine im Worker | Worker-PIDs, jeweils Exit0 |
| --- | --- | --- | --- |
| 1B | 8 neue Antworten +1 Cancel | 7 neue Antworten | 91405 /91674 |
| 4B | 8 neue Antworten +1 Cancel | 7 neue Antworten | 91910 /92625 |
| 12B | 8 neue Antworten +1 Cancel | 7 neue Antworten | 92889 /93017 |

Prefix jeModell: vier direkte Aufrufe (Aufbau plus drei Treffer), Cache löschen,
nach dem ersten tatsächlich gelieferten Token abbrechen, keinen Cache übernehmen,
auf demselben Worker korrekt antworten und erneut löschen; anschließend drei
echte JSON-HTTP-Aufrufe (Aufbau plus zwei Treffer). Engine jeModell: vier direkte
und drei echte JSON-HTTP-Aufrufe. Prompt:1.077 kanonische Tokens, Ausgabe bis8;
der separate Cancel nutzt Limit128 und zählt nicht als vollständige Antwort.

Die Ausgaben werden gegen vorhandene PROD10-Stockdaten geprüft. Pro Lauf sind
vier gespeicherte Referenzrecords eingebunden, insgesamt24 wiederverwendete
Records, **null neue Stock-Läufe**. Modell-, Umgebungs-, Hardware- und Provider-
Bindung sowie Dateihash müssen vor Wiederverwendung passen. Alte Laufzeiten
werden nicht übernommen. Jeder neue Workeraufruf meldet den Hash seiner
tatsächlich gerenderten Eingabetokens; er stimmt mit der alten Referenz überein.

Direkte Ausgaben sind in Token-/Text-/Output-Hash, Usage und Finish identisch.
HTTP prüft Text, Usage und Finish; die Schnittstelle exponiert keine Token-IDs.
Der Prefix-Commit erfolgt erst an der mit Cancel synchronisierten normalen
Modellabschlussentscheidung. Abbruch/Recovery blieben auf demselben Worker.
Die Engine liefert vollständige Ergebnisse gepuffert; das ist kein Nachweis
echten Tokenstreamings oder einer verbesserten Time to First Token.

## Wichtige Abgrenzung der Engine-Konfiguration

Alle sechs Einbindungen sind opt-in. In den drei Engine-Screens wurde
`current_profile` ohne kompatibles Profil im betreffenden Engine-Profilstore
verwendet, also dessen **BASELINE-Knobs**. Keine Runtime-Fallbacks traten auf.
Das behauptet weder, dass ältere gemessene Konfigurationen fehlen, noch dass
BASELINE der stärkste vorhandene Vergleich ist.

Die tatsächlichen B39d-A/B/C/D-Konfigurationen sind separat im Bridge vorbereitet:
Core = compiled_fixed_cache + head_skip_prefill; Gruppierung =Throughput mit
Breite4. **Diese Core-/Gruppierungsvarianten und ihr durchgängiger Batchtransport
sind durch diesen Screen noch nicht nativ qualifiziert.** Die automatische
Auswahl existiert bisher als geprüftes Daten-/Lernregelwerk, nicht als belegte
automatische Entscheidung des laufenden Servers.

## Provenienz und Ressourcen

Installierter Code:
`7b527cda282b35ee001be56ef2cf4536d3d0ef9506b50e321ffd6ecba21f54b4`.
Die Codebindung umfasst jetzt auch die tatsächlich ausführbare bestehende
`ironmule`-Engine, ohne sie im Controller zu importieren. Environmenthash bleibt
`6e32542c2cd4d2950ec828640d196c7e91be2b4790439041770c93cb7f60ee95`.
Keine MLX-/MLX-LM-/NumPy-/Transformers-Aktualisierung. Alte Studien behalten
ihre eigenen Codebindungen.

Alle fünf Vorher-/Nachherbindungen und vollständigen Abschlussdigests stimmen.
Der unabhängige Auditor prüft auch die tatsächlichen Oracle-Dateien, jeden
neuen Prompt-Hash, Cachetransaktionen, die Reihenfolge und die aus dem Journal
neu berechneten Ressourcenzähler. Die Ressourcenbeobachtungen sind fehlerfrei,
aber **Swap war nicht null**: Letztes systemweites Delta der Prefixläufe
1B/4B/12B rund0,59/1,23/4,49GB. Diese Werte sind weder zusätzliche Modellbytes
noch ein isolierter Modellfehler. Gleichzeitige Desktoplast und erhebliche
Kompression wurden read-only beobachtet; keine fremden Prozesse beendet.

**Noch kein Speedup- oder Aktivierungsnachweis:** Die Messanordnung dient
Korrektheit und Lebenszyklus. Für Geschwindigkeitsvergleiche braucht es aktuelle
gepaarte End-to-End-Messungen gegen passende bestehende Konfigurationen, inklusive
Cacheaufbau und Verwaltung, mit getrennter Aussage zu Workerstart/Modellbindung.
Gruppierungs- und automatische Auswahlprüfung bleiben offen.

## Quellen

- [Screen-Spezifikation](PROD14_NATIVE_SCREEN_SPEC.md)
- [Unabhängiger Audit aller sechs Läufe](../research/raw/PROD14_integration_audit_20260908.json)
- [Read-only-Auditor](../tools/product_variant_audit.py)
- [1B Prefix](../research/raw/PROD14_1B_prefix_worker_20260908_attempt1.json), [1B Engine](../research/raw/PROD14_1B_engine_worker_20260908_attempt1.json)
- [4B Prefix](../research/raw/PROD14_4B_prefix_worker_20260908_attempt1.json), [4B Engine](../research/raw/PROD14_4B_engine_worker_20260908_attempt1.json)
- [12B Prefix](../research/raw/PROD14_12B_prefix_worker_20260908_attempt1.json), [12B Engine](../research/raw/PROD14_12B_engine_worker_20260908_attempt1.json)

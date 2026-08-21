# H0.1 Storage- und Dashboard-Spezifikation (v1)

Status: vor jeder H0.1-Persistenz und vor jedem H0.1-Live-Lauf registriert.

## 1. Geltungsbereich und Forschungsgrenze

Diese Spezifikation ergänzt die deterministische H0.1-Paced-Trajectory-Studie um eine
separate, append-only Evidenzablage und eine ausschließlich lesende Loopback-Ansicht.
Sie ändert weder H0 noch dessen SQLite-v1-Vertrag, Messungen, Status oder Schlussfolgerungen.

Die H0.1-Datenbank ist ausschließlich `.friday-data/h01.sqlite3`. Sie wird nicht mit
`h0.sqlite3` zusammengeführt. Das Dashboard erzeugt keinen atomaren Snapshot über beide
Datenbanken und behauptet keinen solchen. In dieser Implementierungsphase wird keine
Produktivdatenbank erzeugt oder importiert.

## 2. Entitäten und erlaubte Aussagen

Schema v1 kennt exakt drei `entity_kind`-Werte:

- `paced_session`: eine vollständige oder ungültige H0.1-Einzelsitzung; Status exakt
  `h01_session_complete` oder `h01_invalid`.
- `paced_study`: die vorregistrierte geordnete Sechs-Sitzungs-Studie; Status exakt
  `h01_stationarity_supported`, `h01_complete_unresolved` oder `h01_invalid`.
- `legacy_h0_warmup_observation`: eine getrennt ausgewiesene historische
  H0-Warmup-Beobachtung; Status exakt `legacy_observation`.

Eine Legacy-Beobachtung ist nie eine `paced_session` oder `paced_study`, erfüllt keine
H0.1-Gates und darf weder `h01_stationarity_supported` noch eine andere H0.1-Aussage
beanspruchen. Für jede Entität ist die gespeicherte Aktion exakt `no_h0_conclusion`.
Es gibt durch diese Ablage keine H0-Reklassifikation, keine Promotion und keinen
Performance-Claim.

Der nachträglich vor Daten registrierte Importvertrag
`docs/H01_LEGACY_IMPORT_SPEC.md` schließt auch die Legacy-Nutzlast: eine geordnete
Warmup-Nanosekundenfolge mit exakt rational replayten Median-/MAD-/IQR-Werten, ein
registrierter Completed- oder `warmup_unstable`-Adapter und eine um Evidence-, Code-,
Spec-, Environment- und Quelldatenbank-Hash erweiterte H0-Elternlinie. Resultate tragen
zusätzlich exakt `interpretation=descriptive_only`,
`stationarity_supported=false` und `paced_gate_applicable=false`. Diese Erweiterung
ändert weder die SQLite-v1-DDL noch die beiden Paced-Entity-Verträge.

## 3. Kanonisches Evidenz-Bundle

Jede Zeile enthält genau ein kanonisches Bundle mit:

- `schema_version = 1`, stabiler `entity_id`, `entity_kind`, `status`,
  `action = no_h0_conclusion` und nichtnegative signed-int64-Erstellungszeit;
- geschlossenen JSON-Objekten `manifest`, `trace`, `result` und `lineage`;
- SHA-256 über jedes dieser vier Objekte;
- SHA-256 über das Bundle ohne sein eigenes Hash-Feld sowie die kanonischen Bundle-Bytes.

JSON ist UTF-8, ohne NaN/Infinity, mit sortierten Schlüsseln und kompakten Separatoren.
Boolesche Werte sind an jeder Integer-Grenze ungültig. Integer müssen signed-int64 sein.
Jedes JSON-Objekt und das gesamte Bundle sind auf 1 MiB kanonische Bytes begrenzt.
Beim Lesen werden Kanonizität, Spalten/Bundle-Spiegelung und alle fünf Hashes erneut
berechnet; ein Byte-, JSON- oder Hashfehler macht die Evidenz ungültig.

## 4. SQLite-v1-Vertrag

Die Migration `friday_h01/migrations/0001_initial.sql` ist die einzige v1-DDL-Quelle.
Sie setzt `application_id = 0x48303131` (ASCII `H011`) und `user_version = 1`, legt
eine kompakte Tabelle `bundles`, zwei explizite Abfrageindizes sowie exakte
`BEFORE INSERT`-, `BEFORE UPDATE`- und `BEFORE DELETE`-Abbruchtrigger an. Der
Insert-Trigger prüft per `EXISTS` sowohl `entity_id` als auch `bundle_sha256`; damit
ist `INSERT OR REPLACE` unabhängig von rekursiver Delete-Triggerausführung verboten.
Jede Verbindung setzt und prüft zusätzlich `recursive_triggers=1`.

Vor jeder Nutzung werden exakt geprüft:

- `application_id`, `user_version` und `PRAGMA integrity_check = ok`;
- alle `sqlite_master`-Objekte einschließlich Autoindizes und normalisierter DDL;
- `PRAGMA table_xinfo` für jede Tabelle;
- `PRAGMA index_list` und `PRAGMA index_xinfo` für jeden Index;
- Existenz und exakter Inhalt aller drei append-only Trigger.

Zusätzliche, fehlende oder veränderte Tabellen, Spalten, Indizes oder Trigger sind
Schema-Drift und werden fail-closed abgelehnt. Eine neue Datenbank entsteht nur durch
einen expliziten schreibbaren Aufruf mit einem konkreten Pfad. Die neu von SQLite
angelegte reguläre Datei wird vor Migration über einen `O_NOFOLLOW`-Dateideskriptor
auf exakt `0600` gesetzt und erneut geprüft. Bestehende Dateien mit Gruppen- oder
Sonstigen-Rechten werden read-only wie schreibbar fail-closed abgelehnt; read-only
erfordert Owner-Read, schreibbar zusätzlich Owner-Write. Ein Persistenzaufruf
umfasst genau eine `BEGIN IMMEDIATE`-Transaktion und höchstens 200 vorab vollständig
replayte Bundles. Schema, Dateiidentität und alle vorhandenen Evidenzzeilen werden erst
innerhalb dieser Transaktion geprüft. Danach wird jede vorhandene ID mit
byteidentischem Bundle idempotent beantwortet, dieselbe ID mit anderen Bytes abgelehnt
oder jede neue Zeile eingefügt und vor Commit replayt. Doppelte IDs innerhalb eines
Batch sind ungültig. Ein Fehler rollt alle neuen Zeilen dieses Batch zurück; die
Einzelbundle-API delegiert auf denselben Batchpfad. Update, Delete und Replace sind
verboten.

## 5. Read-only-Grenze

Ein Read-only-Aufruf akzeptiert ausschließlich einen lokalen Dateisystempfad, niemals
eine vom Aufrufer gelieferte SQLite-URI. Der finale Pfad und sein direktes Elternobjekt
dürfen keine Symlinks sein. Vor und nach Connect sowie nach Transaktionsbeginn werden
Device, Inode, UID und Mode von Elternobjekt und Datenbank verglichen. Die aus
`PRAGMA database_list` aufgelöste Main-Datei muss dieselbe Dateiidentität besitzen.
Danach erzeugt der Aufruf intern eine percent-encodierte URI mit `mode=ro`, setzt
`query_only=1` und prüft Schema/Integrität innerhalb der expliziten Lesetransaktion.
Diese Invarianten erkennen die getesteten Pfad-/Inode-Swaps; sie sind ausdrücklich kein
unbelegter Anspruch, jedes mögliche Dateisystem-TOCTOU auf allen Plattformen auszuschließen.
Verifikation, Snapshot und Detailansicht dürfen Dateiinhalt und Datei-Hash nicht verändern.

## 6. Dashboard

`DashboardService` öffnet H0.1 für jeden Snapshot oder Detailabruf neu in
`mode=ro/query_only=1`. Nach `BEGIN` werden in derselben Lesetransaktion zuerst
Dateibindung, Schema und Integrität und danach jede ausgewählte Zeile durch vollständigen
kanonischen und wissenschaftlichen Replay geprüft. Erst aus diesen verifizierten Zeilen
entstehen Gesamtzahl, Status-/Kind-Zählungen und maximal 200 jüngste Einträge; eine
beschädigte Zeile verwirft den gesamten Snapshot. Die Revision ist SHA-256 über den
Schema-Fingerprint und die nach Row-ID geordneten vollständigen Content-Identitäten
(`rowid`, ID, Bundle-Hash, Erstellungszeit), nicht über bloße Zähler/Maxima.

Eine Detailansicht enthält maximal 200 Records und 200 Trace-Punkte. Die H0-Elternlinie
steht ausschließlich separat unter `parent_h0_lineage`; explizite View-Feldlisten
entfernen `source` aus Manifest-/Trace-/Result-Projektionen, ohne die gespeicherten und
gehashten Originalbytes zu verändern.

Der optionale HTTP-Adapter bindet ausschließlich an `127.0.0.1`, akzeptiert nur
`GET` und `HEAD`, begrenzt Pfad und Query strikt und antwortet mit CSP sowie weiteren
Security-Headern. Jede Methode außer `GET`/`HEAD`, auch eine unbekannte `do_*`-Methode,
liefert `405`. Jede fertig serialisierte HTML- oder JSON-Antwort ist auf 1 MiB begrenzt.
Der Adapter besitzt keinen Schreibpfad und keine Datenbankmutation. Die HTML-Ansicht
zeigt eine kleine lokale Historie; JSON-Endpunkte liefern Snapshot und Detail.

## 7. Abbruch- und Akzeptanzkriterien

Jede Schemaabweichung, nichtkanonisches oder zu großes JSON, ungültige Integer-/Bool-
Grenze, Hashabweichung, ID-Kollision, verbotener Legacy-Status, Schreibversuch im
Read-only-Modus oder ungebundene Netzwerkadresse wird abgelehnt. Ein negativer Befund
bleibt sichtbar und wird nicht gelöscht oder umgedeutet.

Akzeptiert wird der Slice erst nach PyCompile und der vollständigen H0.1-Testsuite unter
hartem NumPy-/MLX-/Socket-Konstruktionsguard sowie dokumentierten Test-, Subtest-, Zeit-
und Speicherwerten.

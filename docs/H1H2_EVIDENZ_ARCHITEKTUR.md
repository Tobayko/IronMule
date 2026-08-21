# H1/H2-Evidenzarchitektur — exploratives SQLite v1 und formales H1-v2

**Status:** SQLite v1 ist implementiert und produktiv initialisiert. Der getrennte
formale H1-v2-Pfad ist implementiert und durch den Nutzer für die eine vorregistrierte
Dispatch-Studie freigegeben. Custom Metal, Cross-Device-Aussagen und freie
Modell-/Codesuche bleiben nicht freigegeben.

## 1. Forschungsgrenze

H0 und H0.1 behalten ihre bestehenden, getrennten Datenbanken und Verträge. Die neue
Datei `.friday-data/research.sqlite3` ist eine dritte, unabhängige Evidenzdomäne für die
Werkzeuge `dispatch`, `cooldown`, `loop`, `model-loop`, `codegen`, `roofline` und
`fusion`. Es gibt absichtlich keine Cross-DB-Transaktion und keine nachträgliche
Umschreibung alter H0/H0.1-Befunde.

Die formale Studie verwendet zusätzlich `.friday-data/h1-v2.sqlite3` als vierte,
eigenständige Domäne. Sie migriert oder ergänzt v1 ausdrücklich nicht: v1 bleibt
explorative Historie, während H1-v2 eine neue Study-ID, eine neue Protokollversion und
frische Messdaten verlangt.

Die Architektur unterscheidet zwei Evidenzklassen strikt:

| Klasse | Bedeutung | Rohmessungen | Provenienz |
| --- | --- | ---: | --- |
| `native` | neuer Bericht, nach Git-/Code-/Spec-/Umgebungs-Preflight gespeichert | verpflichtend für erfolgreiche Messberichte | vollständig und vor/nach dem Lauf identisch |
| `legacy_summary` | aus dem Arbeitsjournal übernommene historische Zusammenfassung | **nein** | fehlende historische Angaben explizit als nicht verfügbar markiert |

Legacy-Zeilen dürfen weder Rohdaten noch einen formalen H1/H2-Nachweis behaupten. Die
Quelldatei `experiments/legacy_h1h2_summaries_v1.json` kennzeichnet deshalb alle
historischen H1/H2-Zahlen als explorative Zusammenfassungen. Es werden keine
Zeitreihen, Hashes oder Zeitstempel erfunden.

Auch `native` bedeutet in Schema v1 nur „neuer, rohdatengespeicherter und vollständig
provenienzgebundener Bericht“, nicht „formal vorregistriert“. Jeder v1-Bericht muss
maschinenlesbar `formal_claim=false` tragen. Eine künftige formale H1-v2-Studie
benötigt eine neue Schema-/Protokollversion, welche Study-ID, versiegelte MDE und
Familien-/Splitvertrag selbst validiert; v1 darf nicht still umgedeutet werden.

## 2. Lebenszyklus eines neuen Messlaufs

```text
--execute
   │
   ├─ Netzbetrieb prüfen
   ├─ SQLite-Schema und Integrität prüfen/initialisieren
   ├─ sauberen Git-Stand + Code/Spec/Environment/Hardware hashen
   │      (vor jedem MLX-/Modellimport)
   ├─ Messung unter gemeinsamem BudgetGuard ausführen
   ├─ Rohmessungen + Zusammenfassung bilden
   ├─ Provenienz erneut erfassen und Identität vergleichen
   ├─ Bericht append-only persistieren
   └─ erst danach Ergebnis samt record_id auf stdout ausgeben
```

Scheitert eine bereits gestartete Messung, wird ein sanitisiertes
`measurement_failed`-Ereignis ohne interne Fehlermeldung gespeichert. Scheitern
Speicherung oder die zweite Provenienzprüfung, wird kein gültiges Messergebnis
ausgegeben. Ein geänderter Checkout während eines Laufs macht den Lauf ungültig.

## 3. Provenienzvertrag

Ein nativer Datensatz bindet mindestens:

- vollständige Git-Revision und einen leeren Arbeitsbaum;
- SHA-256 jedes Python-/SQL-Files in `friday_evidence/`, `friday_h0/`,
  `friday_h01/` und `tools/`;
- SHA-256 der relevanten H1/H2-Spezifikationen, des Requirements-Locks und der
  Pytest-Konfiguration;
- Python-Implementierung, Interpreterpfad und installierte Versionen der
  registrierten Laufzeit-/Testpakete;
- nicht-sensitive Hardwareangaben aus öffentlichen Systemabfragen;
- Tool-ID und geschlossenen Workload-Key.

Der Preflight verlangt einen sauberen Root-Checkout. `ProjectAtlas/` ist als
separat gepinntes Gitlink aus der Dirty-Prüfung ausgenommen, weil es kein
Messcode ist; seine Commit-ID ist bereits im Root-Commit gebunden. Lokale Modelle,
Datenbanken und Atlas-Indizes liegen in ignorierten Pfaden und gehen nicht in den
Repository-Scan ein.

Modellgestützte Werkzeuge dürfen eine Repository-ID nicht direkt an einen
Downloader/Resolver übergeben. `resolve_local_model_snapshot` löst ausschließlich
den projektlokalen Hugging-Face-Cache auf, validiert Ref und 40-stellige
Snapshot-Revision, Config, Tokenizer sowie die vom installierten nichtverteilten
MLX-LM-Loader tatsächlich gelesenen `model*.safetensors`-Dateien. Bericht und
Evidenz enthalten Repository-ID, Snapshot-Revision, Gewichtsdateinamen und
Gewichtsumfang, aber keinen absoluten lokalen Pfad. Ein fehlender oder
inkonsistenter Snapshot bricht fail-closed ab; es gibt keinen Netzwerk-Fallback.

## 4. SQLite-v1-Vertrag

Die Migration `friday_evidence/migrations/0001_initial.sql` definiert eine
geschlossene Tool-/Evidenzklassenmenge, kanonisches JSON, SHA-256-Projektionen und
append-only Trigger. Beim Öffnen werden geprüft:

- `application_id`, `user_version` und Migration-Hash;
- vollständige Tabellen-, Spalten-, Index- und Trigger-Menge;
- `PRAGMA integrity_check(1)`;
- kanonische JSON-Bytes, Report-/Provenienz-Hashes und daraus abgeleitete
  `record_id` jeder Zeile.

Die DB wird atomar als reguläre, nutzereigene Datei mit Modus `0600` angelegt;
unsichere Eigentümer-/Verzeichnis-/Dateimodi und Symlinks werden abgelehnt.
Read-only-Aufrufe verwenden SQLite `mode=ro` als eigentliche Schreibsperre und
zusätzlich `query_only=ON`. Jeder Handle muss `DEFENSIVE=ON`,
`TRUSTED_SCHEMA=OFF`, `DQS_DDL/DML=OFF` und deaktivierbare Extension-Loads setzen;
fehlen diese Python-/SQLite-Schalter, wird nicht geöffnet. Updates und Deletes
werden zusätzlich durch Trigger abgelehnt. Ein identischer Import ist idempotent;
dieselbe Quell-ID mit anderen Bytes ist ein Konflikt und wird verworfen.

### 4.1 Abgleich mit Primärquellen

- SQLite dokumentiert [`mode=ro`](https://www.sqlite.org/uri.html) als
  read-only Öffnungsmodus. [`PRAGMA query_only`](https://www.sqlite.org/pragma.html#pragma_query_only)
  verhindert normale SQL-Schreibbefehle, macht eine Verbindung laut Dokumentation
  aber **nicht** allein wirklich read-only; deshalb werden beide Mechanismen
  kombiniert.
- SQLite empfiehlt [`trusted_schema=OFF`](https://www.sqlite.org/pragma.html#pragma_trusted_schema)
  und beschreibt den zusätzlichen
  [`SQLITE_DBCONFIG_DEFENSIVE`](https://www.sqlite.org/c3ref/c_dbconfig_defensive.html)-Schutz.
  Python stellt diese Schalter über
  [`Connection.setconfig`](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.setconfig)
  bereit, sofern die eingebundene SQLite-Version sie unterstützt; hier ist
  Nichtverfügbarkeit ein Fehler, kein stilles Downgrade.
- [`integrity_check`](https://www.sqlite.org/pragma.html#pragma_integrity_check)
  wird statt `quick_check` verwendet, weil SQLite dokumentiert, dass
  `quick_check` unter anderem UNIQUE- und Indexkonsistenz nicht vollständig prüft.

Append-only ist ein Anwendungs- und Verifikationsvertrag, kein Schutz vor einem
böswilligen Dateieigentümer: Wer die lokale Datei und den Code kontrolliert, kann
beides ersetzen. Für externe Unveränderbarkeit wäre zusätzlich ein außerhalb des
Arbeitsplatzes verankerter Signatur-/Archivdienst nötig; er ist nicht Teil von v1.

## 5. Formale H1-v2-Studienebene

`friday_h1/` implementiert ausschließlich die Studie
`h1v2-dispatch-n8-20260821-01`. Der maschinenlesbare Studienvertrag und
`docs/H1_VORREGISTRIERUNG_V2.md` frieren Workload, Kandidatenzahl, Seeds,
Armreihenfolge, Session-Splits, Statistik, MDE-Ableitung, Budgets und
Entscheidungsregeln vor der ersten Messung ein.

Der Lebenszyklus ist geschlossen:

```text
Präregistrierung
  -> C0,V0,C1,V1,C2,V2 A/A
  -> Kalibrierungs-Replay und MDE
  -> separates Bestätigungssiegel
  -> C0,V0,C1,V1,C2,V2 A/B
  -> terminaler Studienentscheid
```

Jede Session läuft in einem frischen Prozess. A/B kann ohne bestandene A/A-
Kalibrierung und ohne daraus erzeugtes Siegel nicht beginnen. Ein fehlgeschlagener
Live-Versuch ist terminal und darf nicht wiederholt werden. Nur der terminale
Studienentscheid trägt `formal_claim=true`; Sessionwerte und Zwischenberichte sind
keine eigenständigen Claims.

SQLite v2 sperrt Update, Delete und Ersetzen per Trigger. Jede Zeile bindet
kanonische Payload- und Provenienzbytes sowie daraus abgeleitete Record-IDs. Beim
Lesen werden Schema-Snapshot, Metadaten, Zeilenprojektionen, Hashes und die gesamte
Zustandsmaschine erneut berechnet; dadurch werden unter anderem fehlende,
vertauschte oder nachträglich veränderte Sessions abgelehnt. Die Datei wird mit
Modus `0600` angelegt, Symlinks werden abgelehnt und die UI nutzt `mode=ro` plus
`query_only=ON`.

Diese Studienebene ist keine allgemeine H1/H2-Plattform. Ein weiterer Kandidat,
eine andere Operation, ein anderes Gerät oder ein Modellloop benötigt eine neue
Study-ID und eine neue prospektive Spezifikation.

## 6. Gemeinsame Hardwareschutzbudgets

Alle sieben H1/H2-Messwerkzeuge importieren exakt denselben `BudgetGuard`:

| Budget | Grenze | Verhalten |
| --- | ---: | --- |
| GPU-Arbeit je Lauf | `≤ 120 s` | kumuliert, Überschreitung bricht ab |
| ununterbrochene GPU-Arbeit | `≤ 6 s` | kumuliert bis zu realer Pause |
| Pflichtpause | `≥ 4 s` | setzt nur nach realer Wartezeit zurück |
| Duty-Cycle je gleitendem `60-s`-Fenster | `≤ 25 %` | Intervallüberlappung wird angerechnet |
| Wall-Clock | `≤ 20 min` | fail-closed |
| Cooldown zwischen Suchkandidaten | `≥ 60 s` | für `loop`, `model-loop`, `codegen` |
| Netzbetrieb | verpflichtend | Prüfung vor Evidenz-/MLX-Pfad |

`dispatch`-Replikate und Bestätigungsreplikate erhalten mindestens die
Pflichtpause. Die Cooldown-Charakterisierung ist eine einzelne Studie, keine
Kandidatensuche; ihre vorgegebenen Pausen werden dennoch durch dieselbe
GPU-/Duty-/Wall-Buchführung erfasst.
`roofline` trennt jede Modellgenerierung nach der ersten durch eine verifizierte
Pflichtpause; Warmups und Messwiederholungen dürfen nicht zu einem einzigen
kontinuierlichen GPU-Intervall zusammenfallen.

Der Codegen-Worker ist zusätzlich begrenzt: AST-Allowlist, erneute Validierung an
der Ausführungsgrenze, nur `matmul`/`eval`/`synchronize`, kleine Literale, ein
Iterationslevel und höchstens `32` statisch gewichtete Matmuls bei maximal `16`
Operanden, frischer Prozess, `30 s` Wall-Timeout,
`25 s` Kernel-CPU-Limit, eine `8 GiB` MLX-Speicherrichtlinie und `6 s`
kontinuierliche, synchronisierte GPU-Arbeit. Freie Allokationsprimitive sind nicht
zugelassen. Kann CPU-Limit oder MLX-Speicherrichtlinie nicht gesetzt werden, endet
der Worker **vor** dem generierten Plan. Timeout, unlesbare Ausgabe und
Prozessfehler beenden den gesamten Lauf; sie werden nicht als gewöhnlich
verworfener Kandidat weitergeführt.

Wichtig: MLX beschreibt
[`set_memory_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
ausdrücklich als *guideline*. Sie löst erst dann einen Fehler aus, wenn das Limit
überschritten ist und kein RAM einschließlich Swap mehr verfügbar ist. Sie ist
daher **keine harte Speicherisolation**. Die eigentliche Risikoreduktion in v1
kommt aus der geschlossenen, größenbegrenzten Plansprache; ein echter OS-harter
RAM-Container fehlt auf diesem Pfad. Das ist ein zusätzlicher Grund für den
aktuellen Phase-1B-NO-GO-Entscheid. MLX nutzt auf Apple Silicon
[`Unified Memory`](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html),
sodass GPU- und CPU-Speicher nicht als unabhängige Budgets behauptet werden.

## 7. Lokale UI und Betrieb

Historie verifizieren bzw. lesen:

```bash
.venv/bin/python tools/friday.py evidence verify
.venv/bin/python tools/friday.py evidence snapshot
.venv/bin/python tools/friday.py evidence detail --id <record_id>
```

Read-only UI auf Loopback starten:

```bash
.venv/bin/python tools/friday.py evidence serve --port 8767
```

Der Server bindet ausschließlich `127.0.0.1`, akzeptiert nur `GET`/`HEAD`, setzt
No-Store-, CSP-, Frame-, Referrer-, MIME- und Cross-Origin-Schutzheader und öffnet
die DB pro Abfrage read-only. Die Seite zeigt native und Legacy-Einträge, Status,
Tool, Rohdatenverfügbarkeit und eine revisionsgebundene Historie.

Initialisierung und der einmalige Legacy-Import sind explizite Schreiboperationen:

```bash
.venv/bin/python tools/friday.py evidence init --apply
.venv/bin/python tools/friday.py evidence import-legacy --apply
```

Produktiver Auditstand vom 21.08.2026: `10` `legacy_summary`, `0` `native`,
`0` mit Rohmessungen; Modus `0600`, SHA-256
`4489e6114229f386a74f2066833846fa58a211789dc25e7ad8ded20939ecd74a`.
Der Replay-Import meldete `10` bereits vorhandene Zeilen und änderte den Dateihash
nicht. Ein read-only Snapshot änderte die Datei ebenfalls nicht.

Der formale H1-v2-Store wird getrennt bedient:

```bash
.venv/bin/python tools/run_h1_v2.py self-check
.venv/bin/python tools/run_h1_v2.py snapshot
.venv/bin/python tools/run_h1_v2.py dashboard --port 8768
```

Auch diese UI bindet nur Loopback, akzeptiert nur Lesezugriffe und replayt die
vollständige formale Historie vor der Ausgabe.

## 8. Grenzen und Wiederherstellung

- Die ignorierte produktive SQLite-Datei ist lokale Evidenz und kein Backup. Für
  Archivierung muss sie bei gestopptem Writer als Ganzes kopiert und anschließend
  erneut verifiziert werden.
- Die UI vereinigt H0, H0.1 und Research nicht atomar. Die jeweiligen UIs bleiben
  getrennt und benennen ihre Datenquelle.
- Ein Legacy-Eintrag wird durch Import nicht zu nativer Evidenz.
- Die Architektur beweist keine Performance, Übertragbarkeit oder formale
  Vorregistrierung; sie macht künftige Aussagen lediglich überprüfbar.

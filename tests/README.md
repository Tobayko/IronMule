# Tests

Die Tests sind in drei Ebenen zu halten:

1. **Correctness:** gleiche Eingaben, Referenzrechnung, toleranzbasierter Vergleich.
2. **Safety:** Timeout, Prozessabbruch, Compilerfehler, Ressourcenlimit und Rollback.
3. **Performance:** Warmup, mehrere Wiederholungen, robuste Statistik und Baseline-Gate.

Ein Performance-Test darf nicht den Correctness-Test ersetzen. Hardware- und Framework-Versionen
gehören in jede Testausgabe.

## Aktueller Offline-Nachweis

`requirements-apple-silicon.txt` pinnt `pytest-xdist==3.8.0`; `pytest.ini` führt
die Suite standardmäßig parallel aus. Auditlauf vom 21.08.2026:

```text
429 passed, 2443 subtests passed in 31.86s
```

Die neuen Evidenztests decken kanonische JSON-Bytes, Root-Provenienz, private
SQLite-Dateien, exaktes Schema und Integrität, append-only/idempotente Persistenz,
Legacy-Herabstufung, reale Budgetpausen, Failure-Lifecycle sowie die read-only
Loopback-HTTP-Grenze ab. Sie importieren weder MLX noch Modelle und führen keine
GPU-Messung aus.

# Tests

Die Tests sind in drei Ebenen zu halten:

1. **Correctness:** gleiche Eingaben, Referenzrechnung, toleranzbasierter Vergleich.
2. **Safety:** Timeout, Prozessabbruch, Compilerfehler, Ressourcenlimit und Rollback.
3. **Performance:** Warmup, mehrere Wiederholungen, robuste Statistik und Baseline-Gate.

Ein Performance-Test darf nicht den Correctness-Test ersetzen. Hardware- und Framework-Versionen
gehören in jede Testausgabe.

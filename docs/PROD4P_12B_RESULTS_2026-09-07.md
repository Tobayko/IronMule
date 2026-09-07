# Gemma 12B: lokale Ladung und Referenzgenerierung bestanden

Auf dem lokalen M1 Max wurden der fest registrierte Gemma-3-12B-4bit-Snapshot
und die installierte IronMule-Version tatsächlich über MLX/Metal ausgeführt.
Nach der erfolgreichen Ladediagnose bestanden ein Warmup und drei greedy
Acht-Token-Anfragen den vollständigen Vergleich von Tokens, Text, Stopgrund
und Tokenzählern mit der bereits tatsächlich gemessenen Stock-Referenz.

## Nachweise

- Ladediagnose: `ffab58946c0a47fd9db22524bb9ff263`, PID 32156, Exit 0.
  18 Kontrollbeobachtungen ohne Modellworker und 67 Worker-Beobachtungen.
  In beiden Phasen größtes beobachtetes Swapdelta 0 B.
  MLX-Ladepeak 7.188.274.696 B; von Darwin gemeldeter maximaler Prozess-
  Footprint 8.133.171.488 B. Der Prozess-RSS-Peak war nur 2.369.748.992 B:
  RSS allein beschreibt den benötigten Speicher hier nicht ausreichend.
- Generierungsregression: `c633cec2b3e84410b10adb3050f5451b`, PID 32717,
  Exit 0. Alle vier vollständigen Ausgabehashes stimmen mit der eingefrorenen
  12B-Stock-Referenz überein. Höchster gemeldeter MLX-Peak während Generierung:
  7.327.146.968 B. Größtes beobachtetes Swapdelta 0 B.
- Erfasste vollständige Anfragezeiten einschließlich Kontrollaufwand zusammen
  4,539806 s, längste 2,040807 s; keine Behauptung reiner GPU-Zeit.
  20,185301 s Pflichtpausen. Alle festgelegten Speicher-/Readiness-/Zeitgates
  waren eingehalten. Der Test begründet keine neue Geschwindigkeitsfreigabe.
- Beide Läufe haben identische Vorher-/Nachher-Bindungen für Modell, Software,
  Hardware und Code sowie identische Quellmanifeste innerhalb des jeweiligen
  Versuchs. Die Modellrevision ist
  `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.

Installiert: MLX 0.32.0, MLX-LM 0.31.3, NumPy 2.5.2, Transformers 5.15.1.
Codehash `e4a9002a0f18f6a8a5d6d8c524b8766624ae9325f58defd5d84c6e8bfca8689e`.
Die Projektumgebung blieb unverändert; nur das IronMule-Wheel in der separaten
Testumgebung wurde um die Diagnose ergänzt. Kein Cloud-LLM und keine simulierte
Generierung wurden verwendet.

## Was das nicht beweist

Der frühere Swapabbruch bleibt dokumentiert und wird nicht nachträglich als
Erfolg umgewertet. Die heutige Systemlage ist anders; der Read-only-Sensor
verändert weder Loader noch Speichergrenzen. Ein dauerhafter Speicher-Fix oder
ein kausaler Rückgang des Speicherbedarfs ist damit nicht belegt.

Es sind kurze, feste Prompts und Acht-Token-Ausgaben geprüft, kein Langkontext-,
Dauerlast-, Multiuser- oder vollständiger 12B-Performancevergleich. Prozess-
Footprint und MLX-Peak stammen teils aus verschiedenen Phasen und Läufen; sie
dürfen nicht addiert oder als austauschbare Messgrößen behandelt werden.

Rohprotokolle:

- `research/raw/PROD4P_12B_footprint_20260907_attempt1.json`
- `research/raw/PROD4P_12B_reference_20260907_attempt1.json`

Vorgehen und vorab festgelegte Grenzen:
[PROD4_PROCESS_MEMORY_SPEC.md](PROD4_PROCESS_MEMORY_SPEC.md).

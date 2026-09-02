# Spezifikation: Geräteprofil (Device Profile)

**Status:** Vorregistriert für `friday_calibrate` und `friday_serve`  
**Runtime-ID:** `friday-device-profile-v1`  
**Schema-Version:** 1  
**Ziel-Hardware:** Apple Silicon (M1 Max, Unified Memory, GPU)  
**Ziel-Modell:** `mlx-community/gemma-3-4b-it-4bit` (Snapshot `93724907…`)  

---

## 1. Zweck und Motivation

Die bisherigen versiegelten Runtimes (`friday_runtime_n10`, `friday_head_skip_runtime`) banden ihre Gültigkeit an statische Konstanten (`N10_HARDWARE_SHA256`), die unter anderem `platform.mac_ver()` enthielten. Ein reguläres macOS-Update auf `26.6.2` führte dazu, dass diese Bindungen auf dem Ursprungsgerät selbst rissen und alle optimierten Pfade dauerhaft in die Baseline zurückfielen.

Das Geräteprofil (`DeviceProfile`) ersetzt statische, ortsgebundene Konstanten durch ein auf dem Zielgerät empirisch ermitteltes, hashverkettetes Qualifikationsprofil:
- Jede Optimierung wird auf dem Zielgerät gegen ein A/A-Rauschgate und ein Tokenidentitätsgate gemessen.
- Nur Knöpfe mit dem Urteil `verified` werden im Serving-Pfad (`friday_serve`) aktiv geschaltet.
- Das Profil wird in einer append-only SQLite-Datenbank (`.friday-data/device-profile.sqlite3`) mit Hash-Ketten-Integrität abgelegt.

---

## 2. Kalibrierte Knöpfe und Phasen

Die erlaubte Menge an Knöpfen ist in `friday_calibrate.profile.CALIBRATED_KNOBS` geschlossen definiert:

| Knopf | Phase | Engine-Parameter | Anforderungskriterium |
|---|---|---|---|
| `head_skip` | Prefill | `head_skip_prefill=True` | Tokenidentität, $CI_{high} < 1,0$ (bzw. Promotionslatte $\le 0,95$) |
| `fixed_compiled` | Decode | `compiled_fixed_cache=True` | Tokenidentität, $CI_{high} < 1,0$ (bzw. Promotionslatte $\le 0,95$) |
| `prefill_step_size` | Prefill | — | Aktuell `not_applicable`, da Baseline ungechunkt in einem Forward läuft |
| `bundled_readback` | Decode | `readback_every=8` | Tokenidentität, $CI_{high} < 1,0$ (Nutzerentscheid D4 vom 2026-09-02) |

---

## 3. Messprotokoll und Gates

1. **A/A-Noise (`aa_noise`):**
   - 6 balancierte Paare des unoptimierten Baseline-Arms gegen sich selbst.
   - Bestimmt die empirische Rauschschwelle (MDE) dieses Geräts.
   - Tokenidentität zwingend; Abbruch bei Divergenz.

2. **Knopf-Prüfung (`knob:<name>`):**
   - 6 balancierte Paare (alternierende AB/BA-Reihenfolge zur Eliminierung von Aufwärmdrift).
   - Exakte Tokenidentität auf jedem Paar.
   - Ratio und Bootstrap-Konfidenzintervall (95 %).
   - Urteil `verified` nur, wenn das Intervall vollständig unter der jeweiligen Latte liegt.

3. **Breitenkurve (`width_curve`):**
   - Entwurfsbreiten $\{0, 2, 3, 4, 8\}$ zur Bestimmung des Verhaltens.

4. **Roofline (`roofline`):**
   - Prefill-Rechenauslastung und Decode-Bandbreite.

---

## 4. Integrität und Persistenz

- Speicherung über `RuntimeHistory` mit `RecordKind.SYSTEM`.
- Provenienz bindet Git-Revision, Diff, Code-Hashes und Umgebungsdaten.
- `KnobVerdict.__post_init__` validiert konstruktiv jedes Urteil; unzulässige Urteile oder fehlende Evidenz führen zu `ProfileError`.

# Process Map — HETS Trip (High Exhaust Temperature Spread)

## Purpose
Guide post-trip investigation when the unit trips (or is force-reduced) on **High Exhaust Temperature Spread (HETS)** / exhaust TC spread protection.

## Typical plant signatures
- Rapid rise of `EGT_spread_C` toward protection limit
- One or more zones hot, opposing zones relatively cold (true circumferential imbalance)
- `trip_active` / first-out HETS (or equivalent) in SOE
- Load and fuel collapse at trip; IGV may run closed
- Possible concurrent vibration or dynamics spike

## Preconditions
- Trip first-out list available
- Exhaust protection setpoints and voting logic known
- Unit state after trip: coast-down, fired restart, or held offline

## Decision flow
1. **Confirm first-out and data quality**
   - Is first-out truly HETS / exhaust spread — not flameout, overspeed, or vibration?
   - Any single TC step-change that would fool spread logic?
2. **Timeline reconstruction**
   - Load, fuel, IGV, CDP/CDT for 2–6 hours pre-trip
   - Zone EGT pattern at peak spread
   - Concurrent alarms (fuel, flame, dynamics, oil)
3. **Root-cause branches**
   - **Instrumentation** — bad TC, harness, transmitter; verify offline
   - **Fuel / nozzle** — clogged or failed nozzle, manifold imbalance, staging fault
   - **Combustion hardware** — liner, transition, burner damage (outage scope)
   - **Process** — fuel transfer, load reject, IGV runaway coupled to spread
4. **Restart criteria**
   - Do not restart if metal-temp / fire / fuel-leak indicators present
   - After sensor proof: monitored restart with tighter spread watch
   - After hardware suspicion: engineering hold until inspection plan agreed
5. **Capture for learning**
   - Save case in GT Diagnostic Harness with trip narrative and corrections

## Data to capture for GT Diagnostic Harness
- Historian: ≥10 min preferred 1 min for pre-trip hour; include `trip_active` if available
- SOE / first-out print, alarm summary
- Pre-trip load schedule and any fuel-gas quality notes

## Sample CSVs
`samples/hets_trip/hets_trip_01.csv` … `hets_trip_05.csv` (≈2 months @ 10 min, trip embedded)

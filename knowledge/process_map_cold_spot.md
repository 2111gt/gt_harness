# Process Map — Combustion Cold Spot

## Purpose
Guide operators and reliability engineers when a circumferential **cold sector** appears in exhaust gas temperature (EGT) pattern — often one or more thermocouple zones running cold while neighbors are normal/hot — elevating **EGT spread** without necessarily a full HETS trip.

## Typical plant signatures
- One or a cluster of exhaust thermocouples **TC1–TC27** depressed vs ring mean
- Rising `EGT_spread_C` (max–min of the TC ring) over hours to days
- Adjacent TCs often co-cool (sector), not a single random probe
- Possible mild load / heat-rate degradation
- Fuel flow may tick up slightly if control compensates
- Vibration usually secondary unless mechanical insult coexists

## Preconditions
- Unit fired / online
- Exhaust TC map and zone numbering known
- Recent fuel nozzle / combustor work history available
- No active gas leak / fire protection alarms

## Decision flow
1. **Validate instrumentation**
   - Flatline, open TC, swapped leads, or recent harness work on the cold zone?
   - Compare redundant TCs in the same sector if installed
2. **Characterize the cold sector**
   - Which zones are cold vs hot? Stable location or rotating?
   - Onset: gradual (nozzle / staging) vs step (sensor / valve)
   - Correlation with load, IGV, fuel transfer, ambient
3. **Combustion / fuel path**
   - Fuel nozzle restriction or imbalance in that sector
   - Pilot/main staging, manifold ΔP, valve positions
   - Dynamics (`comb_dyn_psi`) — cold spots can coexist with dynamics
4. **Performance cross-check**
   - Power vs expected, CDP/CDT, emissions (CO/NOx) if available
5. **Action branches**
   - **Sensor-only** → TC verification; continue if protection margins OK
   - **True cold sector / nozzle imbalance** → site combustion procedure; consider load limit
   - **Spreading or trip approach** → treat as HETS precursor; escalate

## Data to capture for GT Diagnostic Harness
- Historian export ≥10 min (prefer 1 min): load, CDP, CDT, EGT avg/spread, **TC1–TC27**, fuel, IGV, vib, comb_dyn
- Window: days before onset through present
- Operator notes: cold sector / TC numbers, load band, recent combustor/fuel work
- Alarm/SOE list

## Sample CSVs
`samples/cold_spot/cold_spot_01.csv` … `cold_spot_10.csv` (≈2 months @ 10 min, **TC1–TC27**)

# Process Map — Combustion Dynamics (Pulsation)

## Purpose
Guide response when **combustion dynamics** (pressure pulsation / humming) increase — elevated `comb_dyn_psi` (or site dynamics probes) with possible EGT flutter and bearing vibration coupling.

## Typical plant signatures
- Elevated, often oscillating `comb_dyn_psi` over minutes to hours
- EGT zone flutter → intermittent `EGT_spread_C` growth
- Mild-to-moderate `vib_DE` / `vib_NDE` rise tracking dynamics peaks
- Fuel flow / IGV interaction during load changes
- Operator reports of audible tone or cabin vibration

## Preconditions
- Dynamics probe health known (not iced, not failed high)
- Load band and fuel schedule documented
- Site limits for dynamics (psi or kPa) available

## Decision flow
1. **Validate the dynamics signal**
   - Probe saturation, cable noise, or single-channel failure?
   - Compare redundant dynamics sensors if installed
2. **Characterize the event**
   - Amplitude vs site limit; continuous vs bursting
   - Dominant period / tone (if spectrum available)
   - Load, IGV, fuel staging at onset
3. **Combustion / control response**
   - Staging, pilot ratio, fuel temperature, Wobbe swings
   - Avoid aggressive load slews while dynamics high
4. **Hardware risk**
   - Prolonged high dynamics → liner / transition / basket risk
   - If dynamics + HETS or cold-spot pattern → escalate jointly
5. **Action branches**
   - **Below limit, stable** → monitor, log, trend
   - **Above limit or rising** → load reduction per procedure; notify combustion SME
   - **Trip or hardware suspicion** → hold restart pending inspection criteria

## Data to capture for GT Diagnostic Harness
- Prefer **1-minute** (or faster) historian: `comb_dyn_psi`, EGT zones, load, fuel, IGV, vib
- Window: at least several hours before onset through decay
- Operator narrative: tone, load band, weather/fuel notes

## Sample CSVs
`samples/combustion_dynamics/comb_dyn_01.csv` … `comb_dyn_10.csv` (10 days @ 1 min)

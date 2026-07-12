# Process Map — Exhaust Temperature Spread Excursion

## Purpose
Guide operators and reliability engineers when exhaust thermocouple spread increases beyond site limits during fired operation.

## Preconditions
- Unit online, fuel system normal
- No active fire / gas hazard alarms
- Thermocouple maintenance status known

## Decision flow
1. **Confirm data quality**
   - Any TC reading flatline, open circuit, or stuck?
   - Recent TC replacement or harness work?
2. **Characterize the spread**
   - Peak-to-peak vs historical baseload band
   - Which circumferential sectors are hot/cold?
   - Correlation with load, fuel transfer, ambient?
3. **Cross-check combustion**
   - Fuel stroke / valve positions
   - Dynamics (pulsation) sensors if installed
   - Flame scanners
4. **Cross-check performance**
   - Power vs expected
   - CDP/CDT
   - Emissions (NOx/CO) if available
5. **Action branches**
   - **Sensor-only pattern** → schedule TC verification, continue monitored operation if protection allows
   - **True combustion imbalance** → follow site combustion trouble procedure; consider load reduction
   - **Hardware risk indicators** (metal temperature, oil, vibration concurrent) → escalate, prepare outage scope

## Data to capture for GT Diagnostic Harness
- 1 Hz (or faster) historian export: TTX_1..N, load_MW, CDP, CDT, fuel_flow, IGV, vib_bearing_*
- Operator narrative and alarm list
- Last offline borescope / fuel nozzle service date

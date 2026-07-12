# Gas Turbine Diagnostic Background

## Plant overview
Industrial gas turbines convert fuel chemical energy into mechanical shaft power and often drive generators (simple cycle) or feed HRSGs (combined cycle). Core sections:

1. **Inlet & filtration** — air quality, differential pressure, icing risk
2. **Compressor** — axial stages, IGVs, bleed valves, surge margin
3. **Combustor** — fuel nozzles, DLN/DLE systems, flame scanners
4. **Turbine** — nozzles, buckets, cooling flows, exhaust
5. **Bearings & shaft train** — vibration, lube oil, alignment
6. **Controls** — speed/load, temperature limits, protection trips

## Alerts focus
- Steady baseload or known part-load point with alarm / advisory review
- Compare key KPIs to baseline: CDP, CDT, exhaust temps, spreads, power, heat rate proxy, vibration overalls
- Look for slow drifts (fouling, sensor bias, nozzle wear) vs noise; validate alert validity

## Trips/Event focus
- Trip, runback, load rejection, combustion dynamics alarm, high vibration, fire eye loss
- Align timestamps of alarms with sensor spikes
- Preserve sequence-of-events (SOE) and first-out trips

## Common failure / degradation modes
| Symptom | Possible causes |
|---------|-----------------|
| High exhaust temperature spread | Fuel nozzle imbalance, clogged nozzle, thermocouple fault, combustion hardware damage |
| Rising compressor discharge temperature | Fouling, IGV miscalibration, intercooler issues (if any), sensor drift |
| Step change in vibration 1x | Unbalance, coupling, blade loss (severe), rotor bow |
| High lube oil temperature | Cooler fouling, low flow, bearing distress |
| Flame scanner dropout | Scanner FO fouling, true flame instability, power supply |

## Safe response hierarchy
1. Protect people and machine (trips already designed into controls)
2. Validate instrumentation before invasive maintenance
3. Reduce load / change mode only per procedure
4. Capture data windows for OEM / specialist review

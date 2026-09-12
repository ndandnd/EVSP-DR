# **Current research figures**

## **Importing columns dominates warm-run time**

Full predecessor pools were replayed in these completed runs. The new overnight treatment caps import and validation at 512 sequences and 15 minutes. Its results are pending.

![][image1]

Source: completed full-pool warm runs, chains 3 and 5\. “Other CG work” includes optimization and remaining overhead; final MIP time is excluded.

## **Charging savings with matched return energy**

Five buses, 62 trips, 240 kWh batteries, 350 kW charging. Both optimized cases return at least 280.7833 kWh in total. Shared charger capacity is not imposed.

![][image2]

| Price peak | GIRO repriced ($) | Fixed duties ($) | Joint ($) |
| ----- | ----- | ----- | ----- |
| 08:00 | 490.29–490.98 | 275.87 | 257.69 |
| 12:00 | 549.59–550.60 | 328.55 | 328.36 |
| 18:00 | 483.45–483.72 | 230.78 | 215.76 |

Fixed duties: preserve trips on each bus and use DP to optimize charging. Joint: also change trip assignment using the saved route pool. Costs include electricity and 5 units per charging start, measured by physical replay.

The original range reflects unknown power within charging windows. At noon the discrete objectives tie; the small replay difference is not an improvement in that objective. Optimality claims apply only to the saved pool.

## **Geography suggests groups, but they still interact**

The full GIRO input has 40 duties and 948 trips. This map shows 838 trips as straight lines between known endpoints, not road paths. PARX and 110 trips with unmapped endpoints are omitted. Coordinates: OpenStreetMap.

![][image3]

Route 21 has 14 duties / 203 trips; local services have 26 duties / 745 trips. An optimistic timetable screen finds 31.6% of connections cross the groups. It ignores battery and charging feasibility, so geography does not establish independence.

The current decomposition pilot uses a 32-duty, 750-trip subset split into four groups of eight. Shared capacity and exchanges between groups must be checked when recombining solutions.

Earlier figures and discussions: [research archive](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [slide archive](https://docs.google.com/presentation/d/1RAzaiZSh7DRf_By32mQXPCcwT1PvPDsk0S2xzxMOnDQ/edit)
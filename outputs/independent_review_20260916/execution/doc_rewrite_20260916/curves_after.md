# **CG curves and bus schedules**

The first six convergence plots are earlier fresh-start runs with set partitioning and 240 kWh / 240 kW, presented on 10 September. They explain how CG progresses; they are not the new warm covering runs. Later captions identify the other experiments. Use Week of 14 September for the latest verified results.

The complete earlier document, including the original figures and discussion, remains in the [research archive](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit).

## **Chain 3 at target 5**

The fractional bus count reaches five quickly, but CG continues to reduce charging cost. The right plot shows how much higher the LP cost was than its final value. It is not a constraint violation or a negative reduced cost.

![][image1]

Earlier trace presented on 10 September. The horizontal axis is elapsed CG time; the labels i=144 and i=552 are iteration numbers. The final marker is where pricing met its stopping tolerance. Network preparation and the final integer solve are excluded.

## **Chain 3 at target 10**

The left plot shows how far the fractional route count is above the timetable lower bound. The right shows LP cost above the final value. A log scale makes the small late improvements visible. Reaching zero on the left does not mean the charging objective is finished.

![][image2]

## **Chain 3 at target 4**

The fractional route count reaches its timetable floor before the charging objective finishes improving. The right plot is cost above the final LP value, measured afterwards. Small positive plotting values stand in for exact zeros on the log axis.

![][image3]

## **Chain 3 at target 8**

Most improvement happens early, followed by smaller improvements in charging cost. The last marker shows when pricing met the stopping tolerance. This is an earlier CG trace, not a result from the new limited-route warm starts.

![][image4]

## **Chain 1 at target 5**

This input takes a different path to convergence. Read the horizontal axis as elapsed CG minutes and the labels as iteration numbers. The gap above the eventual LP cost is different from a reduced cost.

![][image5]

## **Chain 5 at target 10**

The fractional fleet reaches the timetable floor while LP cost continues to fall. This is one earlier input and run. It does not establish how every chain or initialization method behaves.

![][image6]

## **Time spent inside earlier CG runs**

Each bar is one run. Blue is the search for new routes; orange is LP solving; purple builds the trip-by-route coefficients; grey is other recorded work. These earlier fresh-start traces exclude network preparation and the final integer solve. They are different experiments from the new warm starts and the charger-capacity pilot.

![][image7]

## **Earlier charging-cost comparison**

This is the original three-way figure for price peaks at 08:00, 12:00 and 18:00. It preserves the five-bus comparison before return energy was matched. The optimized buses finish with less energy, so the savings include using more initial battery energy. Use Figures with explanations for the newer comparison with matched return energy.

![][image8]

## **Historical charging schedules with GIRO initialization**

May RND002, 193 trips, with 300 kWh batteries and 300 kW charging. Starting from GIRO duties produced ten buses at each tariff peak. This is the historical figure you recalled, not the April 175-trip experiment or a controlled comparison with today’s model.

![][image9]

## **Historical charging schedules without GIRO initialization**

Same historical cohort as the previous page. The time-limited runs found 12, 12 and 11 buses at the three tariff peaks. NO\_CHEAT used artificial feasibility variables with no real routes initially; it was not a greedy route initializer.

![][image10]

## **Saved routes improved the integer solution but cost time**

Chain 3, set covering, 240 kWh batteries and 240 kW charging. At k=8, the fresh pool requires nine buses while the inherited pool needs eight. Checking every saved route made the warm runs slower overall. F means the fleet minimum is proved within the pool; OPT also proves the charging stage; \~ means fleet optimality was not proved. At k=10 the augmented pool also includes validated routes from a fresh solver solution.

![][image11]

## **A recent fresh-start schedule at target 10**

Chain 3, 205 trips, fresh set covering, 240 kWh batteries and 240 kW charging, flat electricity price. This saved solution uses 11 buses; it is not the later ten-bus warm solution. Blue blocks are trips and other colors are charging sites. Hatching marks trips assigned more than once. Individual routes passed replay; removing duplicate trips and checking shared charger capacity remain separate tasks.

![][image12]

## **How charging moves when the price peak changes**

Earlier five-bus, 62-trip comparison at 240 kWh and 350 kW. Orange keeps GIRO trip sequences and optimizes charging; green also changes route choices. The original GIRO power within each charging window was not recorded, so it is shown as a range. These earlier schedules have unequal ending battery energy: use them to inspect timing, not as the matched-return-energy cost comparison.

![][image13]

## **Charging windows when electricity peaks at noon**

Top: original GIRO charging windows. Middle: the same bus trip sequences, with charging optimized by dynamic programming. Bottom: joint route and charging optimization. Colors identify charging sites. Driving and waiting are omitted. This is the same earlier five-bus experiment with unequal ending battery energy; the newer matched-energy costs are in Figures with explanations.

![][image14]
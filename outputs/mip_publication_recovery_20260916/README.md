# Recovering a completed MIP result after wrapper timeout

Chain2 target31 job228599 ended FAILED124 because its wrapper waited6300seconds for the Python subprocess. The subprocess had already written a complete result and printed its final status after3974.6seconds. The precise reason it did not exit is unresolved. This is not preemption and not a claim that optimization failed to produce a result.

The recovery checked finished scheduler state, the original watchdog record, source/input/code hashes, selected-route hashes,32selectedroutes covering725trips, and the original solver's individual-physical-replay and two-stage validation flags. The coverage includes122duplicatedtrips. This is an integrity/publication check, not independent physical replay or duplicate removal.

The canonical result is an exact byte copy of the original result.json; new recovery provenance is separate. The failed job state and execution record are preserved. Result:32buses,bound31,open. The prepared longer search can now use this completed source. No original rerun or changed scientific values. recover.py is the reviewed fail-closed recovery procedure; verification.json, execution.json, result.json and gurobi.log preserve evidence.

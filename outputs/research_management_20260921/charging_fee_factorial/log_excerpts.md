# Gurobi endpoint evidence

The complete logs remain at the absolute paths below. Excerpts are the solver endpoint lines, not a replacement for physical validation or the restricted-model scope.

## original_peak08_fee0

[original.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/original_peak08_fee0/668988_r0_1790021162195498255/original.gurobi.log)

```text
Explored 1981 nodes (10900 simplex iterations) in 2.05 seconds (0.98 work units)
Solution count 4: 174.524 174.524 174.532 176.587 
Optimal solution found (tolerance 1.00e-04)
Best objective 1.745244585144e+02, best bound 1.745160518827e+02, gap 0.0048%
```

## original_peak08_fee5

[original.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/original_peak08_fee5/668989_r0_1790021161641214996/original.gurobi.log)

```text
Explored 81052 nodes (4252659 simplex iterations) in 239.26 seconds (224.47 work units)
Solution count 10: 369.636 369.636 369.636 ... 371.43
Optimal solution found (tolerance 1.00e-04)
Best objective 3.696362480391e+02, best bound 3.696362480391e+02, gap 0.0000%
```

The raw incumbent failed physical replay by 0.00002816 kWh. A separately validated 6 ms charging extension supplies the reported feasible upper bound; the log above is the original search, whose bound is unchanged.

## original_peak12_fee0

[original.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/original_peak12_fee0/668990_r0_1790021162571868926/original.gurobi.log)

```text
Explored 2064 nodes (25103 simplex iterations) in 3.82 seconds (2.51 work units)
Solution count 5: 243.829 243.829 259.997 ... 260.46
Optimal solution found (tolerance 1.00e-04)
Best objective 2.438285205102e+02, best bound 2.438263742551e+02, gap 0.0009%
```

## original_peak12_fee5

[original.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/original_peak12_fee5/668991_r0_1790021162390877124/original.gurobi.log)

```text
Explored 68696 nodes (3126710 simplex iterations) in 204.39 seconds (156.12 work units)
Solution count 10: 438.192 438.192 438.192 ... 442.314
Optimal solution found (tolerance 1.00e-04)
Best objective 4.381915012574e+02, best bound 4.381915012574e+02, gap 0.0000%
```

## original_peak18_fee0

[original.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/original_peak18_fee0/668992_r0_1790021162278106885/original.gurobi.log)

```text
Explored 16325 nodes (275293 simplex iterations) in 25.31 seconds (19.90 work units)
Solution count 9: 159.378 159.378 159.378 ... 160.728
Optimal solution found (tolerance 1.00e-04)
Best objective 1.593782060117e+02, best bound 1.593774759173e+02, gap 0.0005%
```

## original_peak18_fee5

[original.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/original_peak18_fee5/668993_r0_1790021165401940842/original.gurobi.log)

```text
Explored 229446 nodes (7460050 simplex iterations) in 600.00 seconds (490.10 work units)
Solution count 10: 349.575 349.604 349.795 ... 351.189
Time limit reached
Best objective 3.495746221378e+02, best bound 3.431076141434e+02, gap 1.8500%
```

## saved_joint_fee0_peak08_fee0

[saved_joint_fee0.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee0_peak08_fee0/668994_r0_1790021162352549148/saved_joint_fee0.gurobi.log)

```text
Explored 1755 nodes (22161 simplex iterations) in 2.85 seconds (2.12 work units)
Solution count 1: 158.706 
Optimal solution found (tolerance 1.00e-04)
Best objective 1.587055950856e+02, best bound 1.587055950856e+02, gap 0.0000%
```

## saved_joint_fee0_peak08_fee5

[saved_joint_fee0.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee0_peak08_fee5/668995_r0_1790021162355469361/saved_joint_fee0.gurobi.log)

```text
Explored 166097 nodes (9791330 simplex iterations) in 600.00 seconds (543.93 work units)
Solution count 10: 344.488 344.488 344.488 ... 345.236
Time limit reached
Best objective 3.444875220459e+02, best bound 3.418965857808e+02, gap 0.7521%
```

## saved_joint_fee0_peak12_fee0

[saved_joint_fee0.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee0_peak12_fee0/668996_r0_1790021162352409089/saved_joint_fee0.gurobi.log)

```text
Explored 2616 nodes (29529 simplex iterations) in 3.44 seconds (2.51 work units)
Solution count 10: 234.295 234.295 234.295 ... 235.213
Optimal solution found (tolerance 1.00e-04)
Best objective 2.342946062547e+02, best bound 2.342946062547e+02, gap 0.0000%
```

## saved_joint_fee0_peak12_fee5

[saved_joint_fee0.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee0_peak12_fee5/668997_r0_1790021162352439017/saved_joint_fee0.gurobi.log)

```text
Explored 155127 nodes (5704587 simplex iterations) in 600.00 seconds (521.22 work units)
Solution count 10: 430.642 430.642 430.642 ... 431.577
Time limit reached
Best objective 4.306424023590e+02, best bound 4.178612074000e+02, gap 2.9679%
```

## saved_joint_fee0_peak18_fee0

[saved_joint_fee0.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee0_peak18_fee0/668998_r0_1790021162352625521/saved_joint_fee0.gurobi.log)

```text
Explored 18988 nodes (283713 simplex iterations) in 50.20 seconds (24.04 work units)
Solution count 10: 163.149 163.149 163.41 ... 164.678
Optimal solution found (tolerance 1.00e-04)
Best objective 1.631488990843e+02, best bound 1.631488990843e+02, gap 0.0000%
```

## saved_joint_fee0_peak18_fee5

[saved_joint_fee0.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee0_peak18_fee5/668999_r0_1790021161685358178/saved_joint_fee0.gurobi.log)

```text
Explored 182725 nodes (7178732 simplex iterations) in 600.00 seconds (503.32 work units)
Solution count 10: 359.573 359.573 359.573 ... 364.743
Time limit reached
Best objective 3.595731965874e+02, best bound 3.544747566376e+02, gap 1.4179%
```

## saved_joint_fee5_peak08_fee0

[saved_joint_fee5.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee5_peak08_fee0/669000_r0_1790021165226740314/saved_joint_fee5.gurobi.log)

```text
Explored 2275 nodes (11584 simplex iterations) in 1.79 seconds (1.24 work units)
Solution count 2: 160.962 160.982 
Optimal solution found (tolerance 1.00e-04)
Best objective 1.609622191994e+02, best bound 1.609493508938e+02, gap 0.0080%
```

## saved_joint_fee5_peak08_fee5

[saved_joint_fee5.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee5_peak08_fee5/669001_r0_1790021165285732750/saved_joint_fee5.gurobi.log)

```text
Explored 172147 nodes (7844846 simplex iterations) in 600.00 seconds (518.34 work units)
Solution count 10: 333.715 333.715 333.716 ... 336.908
Time limit reached
Best objective 3.337148209025e+02, best bound 3.269862223768e+02, gap 2.0163%
```

## saved_joint_fee5_peak12_fee0

[saved_joint_fee5.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee5_peak12_fee0/669002_r0_1790021162001160033/saved_joint_fee5.gurobi.log)

```text
Explored 1964 nodes (23550 simplex iterations) in 4.72 seconds (2.97 work units)
Solution count 2: 233.839 233.84 
Optimal solution found (tolerance 1.00e-04)
Best objective 2.338387208947e+02, best bound 2.338177486673e+02, gap 0.0090%
```

## saved_joint_fee5_peak12_fee5

[saved_joint_fee5.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee5_peak12_fee5/669003_r0_1790021165222338316/saved_joint_fee5.gurobi.log)

```text
Explored 14653 nodes (733653 simplex iterations) in 43.73 seconds (37.83 work units)
Solution count 10: 397.469 397.469 397.737 ... 420.677
Optimal solution found (tolerance 1.00e-04)
Best objective 3.974688869275e+02, best bound 3.974566150646e+02, gap 0.0031%
```

## saved_joint_fee5_peak18_fee0

[saved_joint_fee5.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee5_peak18_fee0/669004_r0_1790021165222302191/saved_joint_fee5.gurobi.log)

```text
Explored 34649 nodes (821924 simplex iterations) in 64.81 seconds (54.52 work units)
Solution count 10: 146.33 146.33 146.622 ... 157.713
Optimal solution found (tolerance 1.00e-04)
Best objective 1.463303536517e+02, best bound 1.463303536517e+02, gap 0.0000%
```

## saved_joint_fee5_peak18_fee5

[saved_joint_fee5.gurobi.log](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/charging_fee_factorial/native/results/saved_joint_fee5_peak18_fee5/668984_r0_1790021162125816523/saved_joint_fee5.gurobi.log)

```text
Explored 193804 nodes (6596788 simplex iterations) in 600.00 seconds (490.50 work units)
Solution count 10: 313.062 313.062 313.077 ... 316.812
Time limit reached
Best objective 3.130620694532e+02, best bound 3.010668762468e+02, gap 3.8316%
```

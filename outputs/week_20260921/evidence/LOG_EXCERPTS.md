# Exact Gurobi log excerpts

The complete logs are adjacent. Line numbers below refer to the unchanged complete source log.

## c1_followup/628441_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/c1_followup/628441_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/research_execution_20260921/c1_followup/628441_r0/gurobi.log`

SHA-256: `040a81d9e5dbf06adbd2196bcd9453c0d04a7e2d99cdcf96db006221d28369cf`

**Fleet stage**

```text
66: Explored 1 nodes (7452 simplex iterations) in 11.43 seconds (3.82 work units)
67: Thread count was 8 (of 64 available processors)
68: 
69: Solution count 1: 8 
70: 
71: Optimal solution found (tolerance 1.00e-04)
72: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
308: Explored 25489 nodes (3265630 simplex iterations) in 1577.25 seconds (1305.80 work units)
309: Thread count was 8 (of 64 available processors)
310: 
311: Solution count 1: 447.44 
312: 
313: Optimal solution found (tolerance 1.00e-04)
314: Best objective 4.474400000000e+02, best bound 4.474400000000e+02, gap 0.0000%
```

## k15_12h/c1_k15__heur

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c1_k15__heur/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c1_k15__heur/gurobi.log`

SHA-256: `13a01b653a5b0648bc2d51402a2261e605b72fc251f2924983d2543a76651d1a`

**Fleet stage**

```text
1021: Explored 613278 nodes (49322008 simplex iterations) in 43200.15 seconds (130244.48 work units)
1022: Thread count was 8 (of 56 available processors)
1023: 
1024: Solution count 10: 18 19 20 ... 31
1025: 
1026: Time limit reached
1027: Best objective 1.800000000000e+01, best bound 1.500000000002e+01, gap 16.6667%
```

**Charging stage**

```text
1111: Explored 0 nodes (0 simplex iterations) in 1797.67 seconds (2613.48 work units)
1112: Thread count was 8 (of 56 available processors)
1113: 
1114: Solution count 10: 1110.32 1110.32 1110.32 ... 1137.72
1115: 
1116: Time limit reached
1117: Best objective 1.110319961502e+03, best bound 6.707500367140e+02, gap 39.5895%
```

## k15_12h/c1_k15__plain

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c1_k15__plain/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c1_k15__plain/gurobi.log`

SHA-256: `2b8d1b01f99b88d29362de4971b3e03a1569c5fbfc41ba96ac6b2144a24163d8`

**Fleet stage**

```text
332: Explored 144481 nodes (871073778 simplex iterations) in 43200.11 seconds (117311.10 work units)
333: Thread count was 8 (of 56 available processors)
334: 
335: Solution count 10: 18 19 20 ... 31
336: 
337: Time limit reached
338: Best objective 1.800000000000e+01, best bound 1.500000000000e+01, gap 16.6667%
```

**Charging stage**

```text
552: Explored 8241 nodes (974570 simplex iterations) in 1798.02 seconds (4963.16 work units)
553: Thread count was 8 (of 56 available processors)
554: 
555: Solution count 10: 1163.62 1175.57 1187.02 ... 1265.13
556: 
557: Time limit reached
558: Best objective 1.163623999996e+03, best bound 6.744843030487e+02, gap 42.0359%
```

## k15_12h/c2_k15__heur

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c2_k15__heur/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c2_k15__heur/gurobi.log`

SHA-256: `fdfe880a249c62dff2b6c2e1e1fbe414d8fbe9d2b1b5b19b3cb16c3dc5471541`

**Fleet stage**

```text
1131: Explored 679987 nodes (102080494 simplex iterations) in 43200.14 seconds (157832.14 work units)
1132: Thread count was 8 (of 56 available processors)
1133: 
1134: Solution count 10: 17 18 19 ... 88
1135: 
1136: Time limit reached
1137: Best objective 1.700000000000e+01, best bound 1.500000000000e+01, gap 11.7647%
```

**Charging stage**

```text
1225: Explored 0 nodes (0 simplex iterations) in 1798.46 seconds (2876.73 work units)
1226: Thread count was 8 (of 56 available processors)
1227: 
1228: Solution count 10: 1072.17 1072.17 1072.17 ... 1117.46
1229: 
1230: Time limit reached
1231: Best objective 1.072167805059e+03, best bound 5.730146233221e+02, gap 46.5555%
```

## k15_12h/c2_k15__plain

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c2_k15__plain/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c2_k15__plain/gurobi.log`

SHA-256: `3235cfd29292a4bff0bfa261f7df705e497354c2adb67aaa326b0747091984f6`

**Fleet stage**

```text
572: Explored 352135 nodes (114370685 simplex iterations) in 43200.14 seconds (132927.99 work units)
573: Thread count was 8 (of 56 available processors)
574: 
575: Solution count 10: 17 18 19 ... 26
576: 
577: Time limit reached
578: Best objective 1.700000000000e+01, best bound 1.500000000000e+01, gap 11.7647%
```

**Charging stage**

```text
788: Explored 8306 nodes (1451606 simplex iterations) in 1798.33 seconds (5040.58 work units)
789: Thread count was 8 (of 56 available processors)
790: 
791: Solution count 6: 1074.94 1076.18 1085.36 ... 1150.9
792: 
793: Time limit reached
794: Best objective 1.074943999998e+03, best bound 5.764651830951e+02, gap 46.3725%
```

## k15_12h/c3_k15__heur

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c3_k15__heur/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c3_k15__heur/gurobi.log`

SHA-256: `6f8b1bb888bb6718d1816606e6e448e6a9971cda8cf27352f71b2fe314bd04b6`

**Fleet stage**

```text
773: Explored 445603 nodes (1156973432 simplex iterations) in 43200.15 seconds (132683.56 work units)
774: Thread count was 8 (of 56 available processors)
775: 
776: Solution count 10: 16 17 18 ... 100
777: 
778: Time limit reached
779: Best objective 1.600000000000e+01, best bound 1.500000000000e+01, gap 6.2500%
```

**Charging stage**

```text
841: Explored 0 nodes (0 simplex iterations) in 1799.41 seconds (3009.12 work units)
842: Thread count was 8 (of 56 available processors)
843: 
844: Solution count 4: 957.312 960.04 961.776 1000.09 
845: 
846: Time limit reached
847: Best objective 9.573119999974e+02, best bound 4.726145475821e+02, gap 50.6311%
```

## k15_12h/c3_k15__plain

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c3_k15__plain/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c3_k15__plain/gurobi.log`

SHA-256: `710a23ea86e7af2043b55b596bc7d1933285d1c1520aa5e1f91bf054148046bf`

**Fleet stage**

```text
2892: Explored 2087152 nodes (247389346 simplex iterations) in 43200.07 seconds (115280.36 work units)
2893: Thread count was 8 (of 56 available processors)
2894: 
2895: Solution count 10: 17 18 19 ... 26
2896: 
2897: Time limit reached
2898: Best objective 1.700000000000e+01, best bound 1.500000000000e+01, gap 11.7647%
```

**Charging stage**

```text
3119: Explored 14761 nodes (4910701 simplex iterations) in 1799.26 seconds (6333.96 work units)
3120: Thread count was 8 (of 56 available processors)
3121: 
3122: Solution count 7: 705.6 708.864 709.112 ... 744.616
3123: 
3124: Time limit reached
3125: Best objective 7.055999999999e+02, best bound 4.592354162284e+02, gap 34.9156%
```

## k15_12h/c4_k15__heur

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c4_k15__heur/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c4_k15__heur/gurobi.log`

SHA-256: `24c1d718b5f02b219a40c0c5a35301ad885e6dc12084f2499032dffa33c4744a`

**Fleet stage**

```text
1142: Explored 706891 nodes (65642902 simplex iterations) in 43200.16 seconds (127805.16 work units)
1143: Thread count was 8 (of 56 available processors)
1144: 
1145: Solution count 10: 17 18 19 ... 33
1146: 
1147: Time limit reached
1148: Best objective 1.700000000000e+01, best bound 1.500000000000e+01, gap 11.7647%
```

**Charging stage**

```text
1248: Explored 0 nodes (0 simplex iterations) in 1798.08 seconds (2761.48 work units)
1249: Thread count was 8 (of 56 available processors)
1250: 
1251: Solution count 10: 1033.69 1034.18 1038.44 ... 1127.34
1252: 
1253: Time limit reached
1254: Best objective 1.033688000000e+03, best bound 5.930931351101e+02, gap 42.6236%
```

## k15_12h/c4_k15__plain

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c4_k15__plain/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c4_k15__plain/gurobi.log`

SHA-256: `298ed4cd80f51fdaa9c661449d607b62b15789e506619d3fb339c5aacc2b6cc1`

**Fleet stage**

```text
332: Explored 124810 nodes (1408239780 simplex iterations) in 43200.10 seconds (141037.53 work units)
333: Thread count was 8 (of 56 available processors)
334: 
335: Solution count 10: 19 20 22 ... 29
336: 
337: Time limit reached
338: Best objective 1.900000000000e+01, best bound 1.500000000000e+01, gap 21.0526%
```

**Charging stage**

```text
563: Explored 8951 nodes (1822266 simplex iterations) in 1798.49 seconds (6284.32 work units)
564: Thread count was 8 (of 56 available processors)
565: 
566: Solution count 10: 784.424 786.408 786.656 ... 849.768
567: 
568: Time limit reached
569: Best objective 7.844239999999e+02, best bound 5.807268935479e+02, gap 25.9677%
```

## k15_12h/c5_k15__heur

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c5_k15__heur/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c5_k15__heur/gurobi.log`

SHA-256: `83a4225edb2f3d655fecab356552ff8d6ede0ce11bd95f5f050d43f7e6c96626`

**Fleet stage**

```text
1231: Explored 1067811 nodes (468694823 simplex iterations) in 43200.76 seconds (130657.10 work units)
1232: Thread count was 8 (of 56 available processors)
1233: 
1234: Solution count 9: 16 17 18 ... 76
1235: 
1236: Time limit reached
1237: Best objective 1.600000000000e+01, best bound 1.500000000000e+01, gap 6.2500%
```

**Charging stage**

```text
1327: Explored 0 nodes (0 simplex iterations) in 1798.30 seconds (2547.86 work units)
1328: Thread count was 8 (of 56 available processors)
1329: 
1330: Solution count 10: 923.56 923.56 923.56 ... 992.408
1331: 
1332: Time limit reached
1333: Best objective 9.235597822851e+02, best bound 6.394724815635e+02, gap 30.7600%
```

## k15_12h/c5_k15__plain

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c5_k15__plain/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c5_k15__plain/gurobi.log`

SHA-256: `5a1c8316b812c9346492d4147c27b84409a9456d49f4cdd6e88c56232c9c1fe6`

**Fleet stage**

```text
1266: Explored 1462169 nodes (572238639 simplex iterations) in 43200.11 seconds (147419.98 work units)
1267: Thread count was 8 (of 56 available processors)
1268: 
1269: Solution count 10: 16 17 18 ... 26
1270: 
1271: Time limit reached
1272: Best objective 1.600000000000e+01, best bound 1.500000000000e+01, gap 6.2500%
```

**Charging stage**

```text
1479: Explored 7993 nodes (1036592 simplex iterations) in 1801.80 seconds (4485.52 work units)
1480: Thread count was 8 (of 56 available processors)
1481: 
1482: Solution count 7: 937.488 960.056 979.312 ... 1063.58
1483: 
1484: Time limit reached
1485: Best objective 9.374879999966e+02, best bound 6.418386424695e+02, gap 31.5363%
```

## k15_12h/c6_k15__heur

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c6_k15__heur/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c6_k15__heur/gurobi.log`

SHA-256: `f99fb8e1433256f523e820878745a6da4bc57f0786b47a490534b4b2ccb90216`

**Fleet stage**

```text
1126: Explored 641682 nodes (93367201 simplex iterations) in 43200.10 seconds (164479.46 work units)
1127: Thread count was 8 (of 56 available processors)
1128: 
1129: Solution count 10: 17 18 19 ... 32
1130: 
1131: Time limit reached
1132: Best objective 1.700000000000e+01, best bound 1.500000000000e+01, gap 11.7647%
```

**Charging stage**

```text
1219: Explored 0 nodes (0 simplex iterations) in 1801.51 seconds (3383.46 work units)
1220: Thread count was 8 (of 56 available processors)
1221: 
1222: Solution count 10: 1103.03 1103.03 1103.03 ... 1167.3
1223: 
1224: Time limit reached
1225: Best objective 1.103031896427e+03, best bound 5.705356522585e+02, gap 48.2757%
```

## k15_12h/c6_k15__plain

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k15_12h/c6_k15__plain/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_fresh_k15_longmip_20260917/results/c6_k15__plain/gurobi.log`

SHA-256: `9589ff608c362794690d1ceedca7f99ca2befd19ce6395b289685a8bfe608faf`

**Fleet stage**

```text
408: Explored 179639 nodes (154200345 simplex iterations) in 43200.13 seconds (154555.36 work units)
409: Thread count was 8 (of 56 available processors)
410: 
411: Solution count 10: 18 19 20 ... 27
412: 
413: Time limit reached
414: Best objective 1.800000000000e+01, best bound 1.500000000000e+01, gap 16.6667%
```

**Charging stage**

```text
623: Explored 7895 nodes (1185921 simplex iterations) in 1798.00 seconds (4947.00 work units)
624: Thread count was 8 (of 56 available processors)
625: 
626: Solution count 9: 1087.14 1087.39 1087.64 ... 1153.73
627: 
628: Time limit reached
629: Best objective 1.087144000000e+03, best bound 5.664583968788e+02, gap 47.8948%
```

## k8_witness/c1_k08/augmented/391805_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c1_k08/augmented/391805_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c1_k08/augmented/391805_r0/gurobi.log`

SHA-256: `0609b7fdad528fc6f9fd149a3f89380782063b473134ab74892e75a7482c92f7`

**Fleet stage**

```text
106: Explored 675 nodes (436484 simplex iterations) in 92.38 seconds (178.97 work units)
107: Thread count was 8 (of 56 available processors)
108: 
109: Solution count 10: 8 9 10 ... 57
110: 
111: Optimal solution found (tolerance 1.00e-04)
112: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
287: Explored 34620 nodes (4794335 simplex iterations) in 392.34 seconds (1254.62 work units)
288: Thread count was 8 (of 56 available processors)
289: 
290: Solution count 1: 479.96 
291: 
292: Optimal solution found (tolerance 1.00e-04)
293: Best objective 4.799600000000e+02, best bound 4.799600000000e+02, gap 0.0000%
```

## k8_witness/c1_k08/control/391804_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c1_k08/control/391804_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c1_k08/control/391804_r0/gurobi.log`

SHA-256: `6c229b25b045023bde4e28a603b0142149e7f15f5b703860a71d3abbd4f088c6`

**Fleet stage**

```text
150: Explored 28694 nodes (10867636 simplex iterations) in 822.12 seconds (2773.96 work units)
151: Thread count was 8 (of 56 available processors)
152: 
153: Solution count 8: 9 10 11 ... 63
154: 
155: Optimal solution found (tolerance 1.00e-04)
156: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
529: Explored 129401 nodes (12798388 simplex iterations) in 2777.19 seconds (10014.39 work units)
530: Thread count was 8 (of 56 available processors)
531: 
532: Solution count 10: 422.36 423.848 426.824 ... 491.584
533: 
534: Time limit reached
535: Best objective 4.223600000000e+02, best bound 3.765025839712e+02, gap 10.8574%
```

## k8_witness/c2_k08/augmented/391811_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c2_k08/augmented/391811_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c2_k08/augmented/391811_r0/gurobi.log`

SHA-256: `232c96578cb1063dd93eeea78c02380eb200866c4311b84b8a57b4e52a4386a7`

**Fleet stage**

```text
72: Explored 242 nodes (311785 simplex iterations) in 56.03 seconds (100.84 work units)
73: Thread count was 8 (of 56 available processors)
74: 
75: Solution count 10: 8 9 10 ... 46
76: 
77: Optimal solution found (tolerance 1.00e-04)
78: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
879: Explored 1244021 nodes (77876377 simplex iterations) in 3543.68 seconds (9998.37 work units)
880: Thread count was 8 (of 56 available processors)
881: 
882: Solution count 9: 361.72 365.688 381.104 ... 418.096
883: 
884: Time limit reached
885: Best objective 3.617200000000e+02, best bound 2.511980454672e+02, gap 30.5546%
```

## k8_witness/c2_k08/control/391810_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c2_k08/control/391810_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c2_k08/control/391810_r0/gurobi.log`

SHA-256: `03efe3360fbca99fa79aadd52333615ce70a93a71cd8e1af1f6dbfc49911c8e4`

**Fleet stage**

```text
339: Explored 317647 nodes (108073782 simplex iterations) in 1800.03 seconds (5538.00 work units)
340: Thread count was 8 (of 56 available processors)
341: 
342: Solution count 9: 9 10 11 ... 46
343: 
344: Time limit reached
345: Best objective 9.000000000000e+00, best bound 8.000000000000e+00, gap 11.1111%
```

**Charging stage**

```text
634: Explored 83429 nodes (12857642 simplex iterations) in 1799.68 seconds (6656.57 work units)
635: Thread count was 8 (of 56 available processors)
636: 
637: Solution count 10: 294.648 295.144 295.392 ... 307.296
638: 
639: Time limit reached
640: Best objective 2.946480000000e+02, best bound 2.189803120503e+02, gap 25.6807%
```

## k8_witness/c3_k08/augmented/391813_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c3_k08/augmented/391813_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c3_k08/augmented/391813_r0/gurobi.log`

SHA-256: `64f237ed7f24af00aec1b9173c43f52f05c6c1a67da83d8b68f0e83d00ba6369`

**Fleet stage**

```text
59: Explored 1 nodes (20909 simplex iterations) in 5.17 seconds (7.44 work units)
60: Thread count was 8 (of 56 available processors)
61: 
62: Solution count 10: 8 9 10 ... 49
63: 
64: Optimal solution found (tolerance 1.00e-04)
65: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
168: Explored 9278 nodes (795833 simplex iterations) in 64.51 seconds (162.90 work units)
169: Thread count was 8 (of 56 available processors)
170: 
171: Solution count 1: 227.688 
172: 
173: Optimal solution found (tolerance 1.00e-04)
174: Best objective 2.276879999996e+02, best bound 2.276879999996e+02, gap 0.0000%
```

## k8_witness/c3_k08/control/391812_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c3_k08/control/391812_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c3_k08/control/391812_r0/gurobi.log`

SHA-256: `3a26820d24b8bee01cae3d4fcba3454a8bb547d6d426f88626de6b232fa244dd`

**Fleet stage**

```text
363: Explored 699193 nodes (108000573 simplex iterations) in 1757.78 seconds (4774.76 work units)
364: Thread count was 8 (of 56 available processors)
365: 
366: Solution count 8: 9 10 11 ... 51
367: 
368: Optimal solution found (tolerance 1.00e-04)
369: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
693: Explored 131801 nodes (11554159 simplex iterations) in 1842.04 seconds (4751.43 work units)
694: Thread count was 8 (of 56 available processors)
695: 
696: Solution count 10: 273.568 275.056 288.944 ... 345.384
697: 
698: Time limit reached
699: Best objective 2.735680000000e+02, best bound 1.944476934677e+02, gap 28.9216%
```

## k8_witness/c4_k08/augmented/391807_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c4_k08/augmented/391807_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c4_k08/augmented/391807_r0/gurobi.log`

SHA-256: `633b56584e75953427f2520d9adc4541f0fe3716326dab04e476550dd24b1732`

**Fleet stage**

```text
84: Explored 1 nodes (132002 simplex iterations) in 41.37 seconds (81.01 work units)
85: Thread count was 8 (of 56 available processors)
86: 
87: Solution count 6: 8 9 10 ... 59
88: 
89: Optimal solution found (tolerance 1.00e-04)
90: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
219: Explored 14833 nodes (2385332 simplex iterations) in 224.80 seconds (662.82 work units)
220: Thread count was 8 (of 56 available processors)
221: 
222: Solution count 1: 428.256 
223: 
224: Optimal solution found (tolerance 1.00e-04)
225: Best objective 4.282559999979e+02, best bound 4.282559999979e+02, gap 0.0000%
```

## k8_witness/c4_k08/control/391806_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c4_k08/control/391806_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c4_k08/control/391806_r0/gurobi.log`

SHA-256: `2c2ed84af2d8665e354ed339e43249176b2ac2c3d8c08afcae677ec0b60e8bdf`

**Fleet stage**

```text
135: Explored 16624 nodes (3654901 simplex iterations) in 506.37 seconds (1611.48 work units)
136: Thread count was 8 (of 56 available processors)
137: 
138: Solution count 7: 9 10 11 ... 59
139: 
140: Optimal solution found (tolerance 1.00e-04)
141: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
465: Explored 83521 nodes (9563453 simplex iterations) in 3092.94 seconds (9374.73 work units)
466: Thread count was 8 (of 56 available processors)
467: 
468: Solution count 5: 411.24 411.736 411.984 ... 457.736
469: 
470: Time limit reached
471: Best objective 4.112400000000e+02, best bound 3.505816374143e+02, gap 14.7501%
```

## k8_witness/c5_k08/augmented/391809_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c5_k08/augmented/391809_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c5_k08/augmented/391809_r0/gurobi.log`

SHA-256: `9886ea8d5420d55d1e9b3e4c79a1cadde8f6fbb03dbcbf00a2c1e9cc0b2b3668`

**Fleet stage**

```text
83: Explored 130 nodes (320119 simplex iterations) in 52.86 seconds (112.37 work units)
84: Thread count was 8 (of 56 available processors)
85: 
86: Solution count 9: 8 9 10 ... 42
87: 
88: Optimal solution found (tolerance 1.00e-04)
89: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
212: Explored 20999 nodes (2180186 simplex iterations) in 145.30 seconds (424.94 work units)
213: Thread count was 8 (of 56 available processors)
214: 
215: Solution count 1: 505.176 
216: 
217: Optimal solution found (tolerance 1.00e-04)
218: Best objective 5.051760000000e+02, best bound 5.051760000000e+02, gap 0.0000%
```

## k8_witness/c5_k08/control/391808_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c5_k08/control/391808_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/review_witness_columns_20260917/results/c5_k08/control/391808_r0/gurobi.log`

SHA-256: `8f7ed75f26ffe80cc21f3db01f47b9d545315173d4d02cae02c061b17c56b25f`

**Fleet stage**

```text
104: Explored 14224 nodes (2521738 simplex iterations) in 151.16 seconds (395.66 work units)
105: Thread count was 8 (of 56 available processors)
106: 
107: Solution count 8: 9 10 11 ... 42
108: 
109: Optimal solution found (tolerance 1.00e-04)
110: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
743: Explored 408426 nodes (32747012 simplex iterations) in 3448.47 seconds (9142.33 work units)
744: Thread count was 8 (of 56 available processors)
745: 
746: Solution count 10: 449.472 452.656 452.904 ... 571.144
747: 
748: Time limit reached
749: Best objective 4.494720000000e+02, best bound 4.350908064572e+02, gap 3.1996%
```

## pilot/c1_k08/control_arm_a/586634_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c1_k08/control_arm_a/586634_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c1_k08/control_arm_a/586634_r0/gurobi.log`

SHA-256: `dc916a2ac99262b03a923072ff155e23366b2266fd9cbaae218e17d8e53981a0`

**Fleet stage**

```text
169: Explored 14942 nodes (5967171 simplex iterations) in 1800.07 seconds (1273.29 work units)
170: Thread count was 8 (of 64 available processors)
171: 
172: Solution count 10: 9 10 11 ... 63
173: 
174: Time limit reached
175: Best objective 9.000000000000e+00, best bound 8.000000000000e+00, gap 11.1111%
```

**Charging stage**

```text
392: Explored 8054 nodes (576059 simplex iterations) in 1798.12 seconds (1247.82 work units)
393: Thread count was 8 (of 64 available processors)
394: 
395: Solution count 10: 501.752 506.256 508.24 ... 579.984
396: 
397: Time limit reached
398: Best objective 5.017519999994e+02, best bound 3.650537614360e+02, gap 27.2442%
```

## pilot/c1_k08/control_arm_b/586635_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c1_k08/control_arm_b/586635_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c1_k08/control_arm_b/586635_r0/gurobi.log`

SHA-256: `8528f29abc73d097f3301fdf196b128c788d8afe035d3447241cc2ef680c7ee6`

**Fleet stage**

```text
178: Explored 34072 nodes (11121616 simplex iterations) in 1244.73 seconds (2520.80 work units)
179: Thread count was 8 (of 80 available processors)
180: 
181: Solution count 10: 9 10 11 ... 63
182: 
183: Optimal solution found (tolerance 1.00e-04)
184: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
722: Explored 213489 nodes (24274245 simplex iterations) in 9633.26 seconds (23352.45 work units)
723: Thread count was 8 (of 80 available processors)
724: 
725: Solution count 10: 474.184 501.752 506.256 ... 575.768
726: 
727: Time limit reached
728: Best objective 4.741839999994e+02, best bound 3.754888434602e+02, gap 20.8137%
```

## pilot/c1_k08/treatment/586633_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c1_k08/treatment/586633_r0/mip_gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c1_k08/treatment/586633_r0/mip_gurobi.log`

SHA-256: `dffce1a7e0f674a897ef53a75444045e052c1657ab91fa4f24c48d76f96800ce`

**Fleet stage**

```text
125: Explored 13614 nodes (10598471 simplex iterations) in 771.03 seconds (1485.10 work units)
126: Thread count was 8 (of 40 available processors)
127: 
128: Solution count 9: 9 10 11 ... 63
129: 
130: Time limit reached
131: Best objective 9.000000000000e+00, best bound 8.000000000000e+00, gap 11.1111%
```

**Charging stage**

```text
247: Explored 1057 nodes (78390 simplex iterations) in 768.91 seconds (752.43 work units)
248: Thread count was 8 (of 40 available processors)
249: 
250: Solution count 10: 454.184 459.392 459.392 ... 600.976
251: 
252: Time limit reached
253: Best objective 4.541840000000e+02, best bound 3.629956427650e+02, gap 20.0774%
```

## pilot/c3_k08/control_arm_a/586637_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c3_k08/control_arm_a/586637_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c3_k08/control_arm_a/586637_r0/gurobi.log`

SHA-256: `9c9d67a0f41337f3a69dd9ad9b0ef381824d976a139d422e3db858d3c2c3cbbf`

**Fleet stage**

```text
380: Explored 567851 nodes (72145848 simplex iterations) in 1800.04 seconds (3423.80 work units)
381: Thread count was 8 (of 40 available processors)
382: 
383: Solution count 10: 9 10 11 ... 51
384: 
385: Time limit reached
386: Best objective 9.000000000000e+00, best bound 8.000000000000e+00, gap 11.1111%
```

**Charging stage**

```text
674: Explored 71497 nodes (8809309 simplex iterations) in 1799.56 seconds (3837.05 work units)
675: Thread count was 8 (of 40 available processors)
676: 
677: Solution count 10: 242.776 320.632 322.616 ... 413.04
678: 
679: Time limit reached
680: Best objective 2.427760000000e+02, best bound 1.943814452375e+02, gap 19.9338%
```

## pilot/c3_k08/control_arm_b/586638_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c3_k08/control_arm_b/586638_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c3_k08/control_arm_b/586638_r0/gurobi.log`

SHA-256: `982e9a789b4b3deca819869dc2904c9b337a70adf346daeeee56c6c44645dfcd`

**Fleet stage**

```text
446: Explored 814334 nodes (114842323 simplex iterations) in 2614.90 seconds (5134.22 work units)
447: Thread count was 8 (of 40 available processors)
448: 
449: Solution count 10: 9 10 11 ... 51
450: 
451: Optimal solution found (tolerance 1.00e-04)
452: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
1133: Explored 417388 nodes (31438404 simplex iterations) in 4279.76 seconds (8960.90 work units)
1134: Thread count was 8 (of 40 available processors)
1135: 
1136: Solution count 10: 242.776 320.632 322.616 ... 413.04
1137: 
1138: Time limit reached
1139: Best objective 2.427760000000e+02, best bound 1.989839761458e+02, gap 18.0380%
```

## pilot/c3_k08/treatment/586636_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c3_k08/treatment/586636_r0/mip_gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c3_k08/treatment/586636_r0/mip_gurobi.log`

SHA-256: `8c6697812db2912e30a95276f496ec1499ef3d9a3cf6afcfb9c8085851e1e3ff`

**Fleet stage**

```text
104: Explored 4332 nodes (2365579 simplex iterations) in 200.19 seconds (217.54 work units)
105: Thread count was 8 (of 40 available processors)
106: 
107: Solution count 8: 8 9 10 ... 51
108: 
109: Optimal solution found (tolerance 1.00e-04)
110: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
280: Explored 21317 nodes (2213483 simplex iterations) in 352.89 seconds (642.18 work units)
281: Thread count was 8 (of 40 available processors)
282: 
283: Solution count 8: 218.184 228.6 356.176 ... 395.192
284: 
285: Optimal solution found (tolerance 1.00e-04)
286: Best objective 2.181840000000e+02, best bound 2.181840000000e+02, gap 0.0000%
```

## pilot/c4_k08/control_arm_a/586640_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c4_k08/control_arm_a/586640_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c4_k08/control_arm_a/586640_r0/gurobi.log`

SHA-256: `49a6da3222e6da61776baa1fa6444a91e6cdc59e592850764e9c3c4d95203e77`

**Fleet stage**

```text
123: Explored 5014 nodes (1427492 simplex iterations) in 305.99 seconds (533.00 work units)
124: Thread count was 8 (of 40 available processors)
125: 
126: Solution count 8: 9 10 11 ... 59
127: 
128: Optimal solution found (tolerance 1.00e-04)
129: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
419: Explored 31163 nodes (5280999 simplex iterations) in 3292.92 seconds (7306.44 work units)
420: Thread count was 8 (of 40 available processors)
421: 
422: Solution count 7: 447.4 448.144 448.392 ... 497.944
423: 
424: Time limit reached
425: Best objective 4.474000000000e+02, best bound 3.449153022044e+02, gap 22.9067%
```

## pilot/c4_k08/control_arm_b/586641_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c4_k08/control_arm_b/586641_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c4_k08/control_arm_b/586641_r0/gurobi.log`

SHA-256: `492e38ef819d3793f11074ac7a72268cda2c74502d736644effc1731af567fce`

**Fleet stage**

```text
123: Explored 5014 nodes (1427492 simplex iterations) in 300.65 seconds (533.00 work units)
124: Thread count was 8 (of 40 available processors)
125: 
126: Solution count 8: 9 10 11 ... 59
127: 
128: Optimal solution found (tolerance 1.00e-04)
129: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
573: Explored 141344 nodes (18689289 simplex iterations) in 7468.38 seconds (18219.63 work units)
574: Thread count was 8 (of 40 available processors)
575: 
576: Solution count 7: 447.4 448.144 448.392 ... 497.944
577: 
578: Time limit reached
579: Best objective 4.474000000000e+02, best bound 3.517703561977e+02, gap 21.3745%
```

## pilot/c4_k08/treatment/586639_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c4_k08/treatment/586639_r0/mip_gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c4_k08/treatment/586639_r0/mip_gurobi.log`

SHA-256: `f93f542f05b7967117348730ee5e05f8e1f3d35fd2785369f13eefb0b6609a6c`

**Fleet stage**

```text
85: Explored 52 nodes (292500 simplex iterations) in 114.46 seconds (203.42 work units)
86: Thread count was 8 (of 56 available processors)
87: 
88: Solution count 9: 8 9 10 ... 59
89: 
90: Optimal solution found (tolerance 1.00e-04)
91: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
294: Explored 18644 nodes (2213120 simplex iterations) in 313.34 seconds (677.54 work units)
295: Thread count was 8 (of 56 available processors)
296: 
297: Solution count 3: 397.72 411.152 449.672 
298: 
299: Optimal solution found (tolerance 1.00e-04)
300: Best objective 3.977200000000e+02, best bound 3.977200000000e+02, gap 0.0000%
```

## pilot/c5_k08/control_arm_a/586643_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c5_k08/control_arm_a/586643_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c5_k08/control_arm_a/586643_r0/gurobi.log`

SHA-256: `f10f611ec33140bed83f19f2accafa2669e1b4fcd6e0c691c401a6869442887b`

**Fleet stage**

```text
155: Explored 22901 nodes (6531540 simplex iterations) in 688.17 seconds (1490.47 work units)
156: Thread count was 8 (of 40 available processors)
157: 
158: Solution count 8: 9 10 11 ... 42
159: 
160: Optimal solution found (tolerance 1.00e-04)
161: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
669: Explored 262279 nodes (19152641 simplex iterations) in 2911.18 seconds (5818.53 work units)
670: Thread count was 8 (of 40 available processors)
671: 
672: Solution count 10: 450.632 453.36 463.072 ... 496.92
673: 
674: Time limit reached
675: Best objective 4.506320000000e+02, best bound 4.316492504293e+02, gap 4.2125%
```

## pilot/c5_k08/control_arm_b/586644_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c5_k08/control_arm_b/586644_r0/gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c5_k08/control_arm_b/586644_r0/gurobi.log`

SHA-256: `5110152841a9f87cbef5e193a71b69776c1eda29554d7d07c6271f94c0ec95c6`

**Fleet stage**

```text
155: Explored 22901 nodes (6531540 simplex iterations) in 675.26 seconds (1490.47 work units)
156: Thread count was 8 (of 40 available processors)
157: 
158: Solution count 8: 9 10 11 ... 42
159: 
160: Optimal solution found (tolerance 1.00e-04)
161: Best objective 9.000000000000e+00, best bound 9.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
931: Explored 478586 nodes (45226199 simplex iterations) in 5444.21 seconds (9920.38 work units)
932: Thread count was 8 (of 40 available processors)
933: 
934: Solution count 10: 450.632 453.36 463.072 ... 496.92
935: 
936: Time limit reached
937: Best objective 4.506320000000e+02, best bound 4.395401115982e+02, gap 2.4614%
```

## pilot/c5_k08/treatment/586642_r0

Local: `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/pilot/c5_k08/treatment/586642_r0/mip_gurobi.log`

Unicorn: `/home/nc437/ladder-lite/diving_pricing_20260919/results/c5_k08/treatment/586642_r0/mip_gurobi.log`

SHA-256: `76d4e6fe8ec46d476276cbb644f6311cd10ee8627e0882683f30d5aff73462cc`

**Fleet stage**

```text
93: Explored 1 nodes (162459 simplex iterations) in 51.84 seconds (70.47 work units)
94: Thread count was 8 (of 56 available processors)
95: 
96: Solution count 8: 8 9 10 ... 42
97: 
98: Optimal solution found (tolerance 1.00e-04)
99: Best objective 8.000000000000e+00, best bound 8.000000000000e+00, gap 0.0000%
```

**Charging stage**

```text
205: Explored 4766 nodes (517991 simplex iterations) in 110.13 seconds (231.83 work units)
206: Thread count was 8 (of 56 available processors)
207: 
208: Solution count 5: 447.28 450.504 451.992 ... 461.208
209: 
210: Optimal solution found (tolerance 1.00e-04)
211: Best objective 4.472800000000e+02, best bound 4.472800000000e+02, gap 0.0000%
```

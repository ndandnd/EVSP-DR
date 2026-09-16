# Chain 4, target 31: longer search of unchanged columns

The original MIP found40 buses with bound30; target31 remains unresolved. Job290897 performs the first longer search of the same238,817-column pool with the same greedy initializer policy and frozen native871 worker. No GIRO seed or new columns. Hardware and search timing are not controlled away.

12600total/10800fleet seconds,8CPUs24GB4.5h/default partition,requeue/private attempts,scaglione-compute-01 excluded. Manifest/source/pool and submission validation passed. Source CG was already in the09:42snapshot; the completed MIP result was read and hashed directly after that snapshot. The current10:43collection began before submission; next collection includes this campaign.

Preflight initially considered chain2target31 but stopped before any manifest or submission because its published validated result did not exist yet. Only chain4 is submitted. Keep chain2's validation dependency; do not use its unfinished attempt output. Held537227 and EVSPV2G untouched.

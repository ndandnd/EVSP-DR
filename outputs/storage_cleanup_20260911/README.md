# Unicorn storage cleanup audit — 2026-09-11

This record covers the bounded read-only audit and approved cleanup on Unicorn for `nc437`. The remote host was reachable through the shared SSH control socket. The Scaglione policy at `/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md` was read before any cluster action. No Git remotes or secrets are recorded here.

## Scope and protections

- Initial `/home/nc437` usage was 960G of 130T. The initial top-level scan was one `du -x -d 1` pass.
- `ladder-lite` was 913G. The largest roots were `threshold_9_15_event_20260904_9bdbb17` 279G, `nested_probability_k2_15_fresh84_20260908_21fbecb` 157G, `cg_acceleration_20260903` 108G, `finetime` 41G, and `factfill` 36G.
- Active and pending Slurm jobs included capacity retry 872397/872403–872406, warm-chain jobs 779035/779072/779073, multichain jobs 810293/810332/810344 and 810952–810995, and held job 537227. The held job was left untouched. CPU jobs retain the policy exclusion of `scaglione-compute-01`.
- No active, pending, or held job command path or command script referenced `finetime` or `factfill` at the archive check. The selected sources were dated August 20–21, 2026.
- No research content was discarded. Three cold column streams were removed from their original locations only after their verified compressed copies were fsynced and hash checked; result JSON, journals, inputs, logs, checkpoints, active stochastic directories, extracted environments and Git objects were untouched.

## Approved cleanup completed

Only the Conda repodata cache was removed. No package archives (`*.conda` or `*.tar.bz2`) existed, and no Conda/mamba install or update process was running. `/home/nc437/.conda/pkgs/cache` went from 36 files and 221,643,886 bytes to zero files and zero bytes. The extracted package directories and both environments (`myenv`, `virtenv`) were preserved.

The 65M Gurobi installer archive seen in the first top-level inventory had disappeared before a deletion precheck, so it was not deleted or hashed by this audit. The installed runtime remained present at `/home/nc437/gurobi1102/linux64/bin/gurobi_cl` with `libgurobi110.so` symlinked in the installed library directory; `/home/nc437/temp` was zero bytes after the independent disappearance.

## Lossless archive pilot

Three cold column streams were archived under `/home/nc437/ladder-lite/storage_cleanup_20260911_archive/`. For each source, the worker recorded a SHA-256 and `(size, mtime, inode, device)` before compression; streamed gzip level 1 to a collision-checked temporary file; fsynced it; checked gzip CRC; streamed decompression into SHA-256 and matched the source; rechecked source metadata; fsynced the archive directory; and only then removed the source. The source `.json`, `.iters.csv`, logs and all other files remain untouched. The exact records are in `archive_pilot.tsv`.

The pilot saved 7,038,926,342 source bytes as 591,004,154 compressed bytes (11.91x aggregate compression), freeing 6,447,922,188 bytes while preserving the verified compressed copies. The archive paths must be restored before any old script that names the original source paths is run; restore with `gzip -cd ARCHIVE.gz > ORIGINAL`, then verify the recorded source SHA-256 before execution.

The remaining `finetime` and `factfill` streams are listed exactly in `remaining_manifest.tsv` (124 files, 74,843,933,989 bytes). The first worker submission, Slurm job 884356, was canceled while still pending before it started after review found that its status record was written after source unlink. The second submission, 884918, was also canceled while pending after review found two resume edge cases. Their superseded scripts are preserved remotely for audit. The immutable v3 worker uses a filesystem lock, writes and fsyncs a `prepared` record containing both hashes before unlinking, reconciles prepared records after interruption with a current source SHA check, ignores later failure rows when selecting durable prepared state, and writes `success` only after source removal. The remote fixture test passed the interruption and same-size mutation cases. Corrected job 884924 is submitted only after the script, manifest, restore utility and readme hashes matched their local copies.

## Completion — 11 September, 12:53 EDT

Job884924 completed successfully in44m31s. All124 manifest files have success records; no failed events. The final audit matched archive paths and sizes against the manifest, confirmed all archives exist and their uncompressed sources are removed, and retained the worker’s per-file source/decompressed and archive hashes. The74,843,933,989source bytes occupy6,731,489,738compressed bytes, reclaiming68,112,444,251bytes. Including the pilot and Conda index cache, total reclaimed space is74,782,010,325bytes (74.78decimalGB). This is measured file-byte reduction, not a claim about current total home usage while research jobs continue writing.

See completion_884924.json and ARCHIVED_POOLS_STATUS.tsv. ManifestSHA2561ccefec18d68388711ff57779fcf08dd1066112a412cfc17a227658c86823386; statusSHA256e09441538164676920faf469c45ab46abb72524688e94161d3d9e2e4fe098649. No expansion beyond the authorized historical directories. The bounded archival task is complete; no further archival polling is needed unless a restoration or validation issue arises.

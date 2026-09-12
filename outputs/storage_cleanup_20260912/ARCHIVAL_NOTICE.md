# Cold data archival — 12 September 2026 UTC

The user authorized lossless archival of the old `phys240kw` column journals and `cg_acceleration_20260903` column journals/graph pickles to reclaim space. These files are research evidence, not discarded results. Logs, result JSON, iteration CSVs and graph-cache metadata remain unchanged.

Archive root: `/home/nc437/ladder-lite/storage_cleanup_20260912_archive`.

Check `status_0.tsv` through `status_3.tsv` there. Only a `success` row establishes that the original was replaced by its verified compressed archive. Each row retains original path, SHA-256 and size, plus archive path, SHA-256 and size. `manifest.json` records the full selected set and pre-archival metadata. The final reconciliation confirmed all 156 files successfully archived at 05:09:38 UTC on 12 September; restore the needed originals before any legacy rerun.

Before running any legacy command that expects an archived original:

```bash
bash /home/nc437/ladder-lite/storage_cleanup_20260912_archive/restore_archive.sh \
  ARCHIVE_PATH.gz ORIGINAL_PATH SOURCE_SHA256
```

Use the exact values from its successful status row. Restore rejects existing destinations and verifies the original content hash. A `.pkl.manifest.json` without its `.pkl` may indicate this intentional archival; restore the verified pickle instead of automatically rebuilding the graph. Do not rerun optimization merely because a journal is archived.

Code, audit and restoration manifests: https://github.com/ndandnd/EVSP-DR/tree/codex/storage-cleanup-20260912/outputs/storage_cleanup_20260912

The initial maintenance array950484 stopped safely before any source removal because NFS device numbers differed across nodes. Corrected array 950555 completed all four workers with exit 0:0; preserved failed events from the first attempt are not solver failures. The verified completion report is `completion_950555.json`; original-file restoration was also tested. New savings total 133,737,953,364 bytes, with 15,583,969,540 compressed bytes retained.

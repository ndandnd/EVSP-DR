# Initial archival attempt — safely stopped

Array950484 stopped all four tasks before any archive was created or source removed. The worker logged156 identity-check rejections. All original files remain; no bytes saved by this attempt. Scheduler states are FAILED1:0 after16–18seconds. These are maintenance failures, not optimization outcomes.

The manifest was captured on the login node. Read-only compute-node probe950543 checks which fields differ across NFS clients. Original logs, runtime dependency audits, and failed status rows are preserved. Any correction and retry must be committed and recorded separately. No research job is changed.

Probe950543 ran on snavely-cpu-02 and found all156 source files identical in size, nanosecond mtime, inode, mode and link count; only st_dev differed (login255, compute146). NFS device numbers are client-local. V2 excludes that field only from the cross-client manifest comparison and still enforces device identity within the worker's before/after checks. It does not relax inode, full mtime, size, regular-file, link-count or content-hash checks. A new fixture explicitly checks cross-client device numbers.

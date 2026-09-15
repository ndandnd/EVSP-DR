# Unicorn access lost — 15 September

The20:31UTC collection failed with SSH exit255 and produced an empty snapshot. Its process reported failure21:55UTC after5034seconds elapsed. A subsequent22:21UTC direct check confirmed authentication rejection: Permission denied(publickey,password). The expected SSH control socket no longer exists. The exact disconnection time and cause are unknown.

The user was notified promptly after the direct check. The last successful operational snapshot remains191846, completed19:28UTC(15:28EDT). Last normalized scientific data remain181736. Do not run normalization or preemption refresh on the empty failed snapshot. No current queue/result claims can be made; no jobs were changed.

On the next heartbeat, first perform a short BatchMode connection probe. If still unavailable, retain lost status and do not repeat the notification. Once restored, notify once, collect fresh results and inspect all accumulated scheduler transitions before any repair. The current Doc/workbook remain unchanged. Cluster mirror verification is pending restored access.

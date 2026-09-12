import faulthandler,json,sys,time
sample=sys.argv[1]=='timer'
if sample:faulthandler.dump_traceback_later(.002,repeat=True,file=sys.stderr)
a={'kind':'charge','from_trip':1,'next_trip':2,'station':'2190L_0','arrival_min':123.25,'start_min':125.0,'end_min':155.0,'energy_kwh':120.0,'cost':11.904}
start=time.monotonic();n=0
while time.monotonic()-start<3:
    json.dumps(a,sort_keys=True);n+=1
if sample:faulthandler.cancel_dump_traceback_later()
print(json.dumps({'calls':n,'seconds':time.monotonic()-start,'timer':sample,'python':sys.version}),flush=True)

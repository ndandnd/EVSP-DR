import json,multiprocessing,signal,sys,time
mode=sys.argv[1]
def inherited_handler(signum,frame):pass
def work(value):
 print(json.dumps({'worker_started':True,'handler_inherited':signal.getsignal(signal.SIGTERM)==inherited_handler}),flush=True)
 time.sleep(30)
 return value
if __name__=='__main__':
 signal.signal(signal.SIGTERM,inherited_handler if mode=='inherited_handler' else signal.SIG_DFL)
 pool=multiprocessing.get_context('fork').Pool(1)
 result=pool.apply_async(work,(1,))
 try:result.get(timeout=.2)
 except multiprocessing.TimeoutError:pass
 print(json.dumps({'phase':'terminate_begin'}),flush=True)
 start=time.monotonic();pool.terminate();pool.join()
 print(json.dumps({'phase':'terminate_end','seconds':time.monotonic()-start}),flush=True)

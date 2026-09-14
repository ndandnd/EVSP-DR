import unittest
from unittest.mock import patch
import gate
class Clock:
 def __init__(self):self.t=0
 def now(self):return self.t
 def sleep(self,n):self.t+=n
class AccountingLag(unittest.TestCase):
 def test_missing_and_stale_row_then_terminal(self):
  clock=Clock();rows=iter([None,{'state':'RUNNING'},{'state':'TIMEOUT','job_id':'1_0','raw_job_id':'2','exit_code':'0:15'}]);seen=[]
  def reader(job,timeout):seen.append((job,timeout));return next(rows)
  r=gate.wait_scheduler_terminal('1_0',reader,clock=clock.now,pause=clock.sleep)
  self.assertEqual(r['accounting_polls'],3);self.assertEqual(r['accounting_wait_s'],10);self.assertTrue(all(j=='1_0' for j,_ in seen))
 def test_real_active_parent_never_bypassed(self):
  clock=Clock()
  with self.assertRaisesRegex(ValueError,'within120seconds'):gate.wait_scheduler_terminal('1_0',lambda *a:{'state':'RUNNING'},clock=clock.now,pause=clock.sleep)
  self.assertEqual(clock.t,120)
 def test_missing_parent_bounded(self):
  clock=Clock()
  with self.assertRaises(ValueError):gate.wait_scheduler_terminal('1_0',lambda *a:None,clock=clock.now,pause=clock.sleep)
  self.assertEqual(clock.t,120)
 def test_ambiguous_rows_fail_immediately(self):
  clock=Clock()
  def read(*a):raise ValueError('Ambiguous')
  with self.assertRaisesRegex(ValueError,'Ambiguous'):gate.wait_scheduler_terminal('1_0',read,clock=clock.now,pause=clock.sleep)
  self.assertEqual(clock.t,0)
 def test_no_retry_crosses_deadline_during_query(self):
  clock=Clock()
  def read(job,timeout):clock.t+=timeout;raise gate.subprocess.TimeoutExpired('sacct',timeout)
  with self.assertRaises(ValueError):gate.wait_scheduler_terminal('1_0',read,clock=clock.now,pause=clock.sleep)
  self.assertEqual(clock.t,120)
if __name__=='__main__':unittest.main()

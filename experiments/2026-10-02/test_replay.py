import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from replay import load_trajectory, interpolate, require_near_start, approach_duration, approach_target

class ReplayTests(unittest.TestCase):
    def test_approach(self):
        a,b=[0.]*6,[.1934]*6
        duration=approach_duration(a,b)
        self.assertEqual(approach_target(a,b,0,duration),a)
        self.assertEqual(approach_target(a,b,duration,duration),b)
        points=[approach_target(a,b,i*duration/1000,duration)[0] for i in range(1001)]
        self.assertLessEqual(max(abs(y-x)/(duration/1000) for x,y in zip(points,points[1:])),.05)
        with self.assertRaises(ValueError):approach_duration(a,[.36]*6)

    def test_interpolation(self):
        self.assertEqual(interpolate([0,2],[[0]*6,[1]*6],1),[.5]*6)
        self.assertEqual(interpolate([0,2],[[0]*6,[1]*6],3),[1]*6)
        self.assertEqual(interpolate([0,2],[[0]*6,[1]*6],-1),[0]*6)
    def test_start_gate(self):
        require_near_start([.01]*6,[0]*6)
        for q in [[.06]*6,[float('nan')]*6,[0]*5]:
            with self.assertRaises(ValueError):require_near_start(q,[0]*6)
    def test_validation_and_default_no_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'e.json'
            data={'metadata':{'status':'motion_complete','finalization_complete':True,'units':{'q':'rad'}},'samples':[{'t_monotonic_s':10.,'q':[0]*6},{'t_monotonic_s':10.02,'q':[.01]*6}]}
            path.write_text(json.dumps(data))
            self.assertAlmostEqual(load_trajectory(path)[0][-1],.08)
            p=subprocess.run([sys.executable,str(Path(__file__).with_name('replay.py')),str(path)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads(p.stdout)['will_connect_hardware'])
            for speed in [0,1,-1,float('nan')]:
                with self.assertRaises(ValueError):load_trajectory(path,speed)
            for t in [10.,9.,10.2,float('nan')]:
                data['samples'][1]['t_monotonic_s']=t;path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):load_trajectory(path)
            data['samples'][1]['t_monotonic_s']=10.02
            data['metadata']['status']='failed';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_trajectory(path)

if __name__=='__main__':unittest.main()

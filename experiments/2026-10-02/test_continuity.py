"""Hardware-free trajectory contract tests; no claim of physical stability."""
import unittest
import numpy as np
from robot_proc import smooth_step

class TrajectoryTests(unittest.TestCase):
    def test_reproduced_close_target_does_not_erase_momentum(self):
        q,v=smooth_step(np.array([0.]),np.array([.3]),np.array([.001]),.5,3.,.01)
        self.assertAlmostEqual(v[0],.27)
        self.assertAlmostEqual(q[0],.00285)
        self.assertGreater(q[0],.001)  # target is inside unavoidable braking distance

    def test_fixed_targets_arrive_at_rest_without_crossing_from_rest(self):
        for goal in [-2.,-.01,-1e-7,0.,1e-7,.01,2.]:
            q=np.zeros(1);v=np.zeros(1)
            for _ in range(600):
                before=q.copy();pv=v.copy();q,v=smooth_step(q,v,np.array([goal]),.5,3.,.01)
                self.assertLessEqual(abs(v[0]-pv[0]),.03000000001)
                self.assertLessEqual(abs(v[0]),.50000000001)
                self.assertGreaterEqual(q[0],min(0,goal)-1e-10)
                self.assertLessEqual(q[0],max(0,goal)+1e-10)
                self.assertLessEqual(abs(q[0]-before[0]),.00500000001)
            np.testing.assert_allclose(q,[goal],atol=1e-10)
            np.testing.assert_allclose(v,[0],atol=1e-10)

    def test_reversal_and_moving_targets_respect_bounds(self):
        rng=np.random.default_rng(4701);q=np.zeros(6);v=np.zeros(6);goal=q.copy()
        for k in range(6000):
            if k%7==0:goal=rng.uniform(-2,2,6)
            dt=float(rng.uniform(.003,.05));oldq=q.copy();oldv=v.copy()
            q,v=smooth_step(q,v,goal,.5,3.,dt)
            self.assertTrue(np.all(np.isfinite(q)))
            self.assertLessEqual(np.max(np.abs(v-oldv)),3*dt+1e-9)
            self.assertLessEqual(np.max(np.abs(v)),.5+1e-9)
            self.assertLessEqual(np.max(np.abs(q-oldq)),.5*dt+1e-9)
            # Integral of velocity cannot depart from v0*dt by more than a*dt²/2.
            self.assertLessEqual(np.max(np.abs(q-oldq-oldv*dt)),1.5*dt*dt+1e-9)
        for _ in range(1200):q,v=smooth_step(q,v,goal,.5,3.,.01)
        np.testing.assert_allclose(q,goal,atol=1e-9)
        np.testing.assert_allclose(v,0,atol=1e-9)

    def test_lowered_speed_limit_brakes_instead_of_clipping_velocity(self):
        q,v=smooth_step(np.array([0.]),np.array([.5]),np.array([1.]),.1,3.,.01)
        self.assertAlmostEqual(v[0],.47)
        self.assertAlmostEqual(q[0],.00485)

    def test_phase_boundaries_are_independent_of_sample_partition(self):
        for goal in [-.1,.001,.1,1.]:
            q=np.array([0.]);v=np.array([.3]);g=np.array([goal])
            a,b=smooth_step(q,v,g,.5,3.,.4)
            for _ in range(40):q,v=smooth_step(q,v,g,.5,3.,.01)
            np.testing.assert_allclose(q,a,atol=1e-9)
            np.testing.assert_allclose(v,b,atol=1e-9)

    def test_invalid_values_rejected(self):
        for args in [(0,3,.01),(.5,0,.01),(.5,3,-1),(.5,float('nan'),.01)]:
            with self.assertRaises(ValueError):smooth_step(np.zeros(1),np.zeros(1),np.ones(1),*args)

if __name__=='__main__':unittest.main()

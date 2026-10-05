import tempfile,time,unittest
from pathlib import Path
from usb_watch import UsbWatcher,list_devices


def add(root,name,vendor,product='0001'):
    d=root/name;d.mkdir();(d/'idVendor').write_text(vendor+'\n');(d/'idProduct').write_text(product+'\n')
    return d


def wait_for(cond,timeout=1.0):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if cond():return True
        time.sleep(0.01)
    return False


class UsbWatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        add(self.root,'usb1','1d6b');add(self.root,'1-2','8086');(self.root/'1-2:1.0').mkdir()
    def tearDown(self):self.tmp.cleanup()
    def test_interfaces_are_not_devices(self):
        self.assertEqual(sorted(list_devices(self.root)),['1-2','usb1'])
    def test_baseline_mode_detects_insert_and_remove(self):
        w=UsbWatcher(targets=[],period_s=0.01,root=self.root).start()
        self.assertFalse(w.connected)
        t0=time.monotonic();phone=add(self.root,'1-3.4','05ac','12a8')
        self.assertTrue(wait_for(lambda:w.connected))
        (phone/'idVendor').unlink();(phone/'idProduct').unlink();phone.rmdir()
        self.assertTrue(wait_for(lambda:not w.connected));w.stop()
        self.assertEqual([e['connected'] for e in w.events_between(t0,time.monotonic())],[True,False])
    def test_target_mode_ignores_other_devices(self):
        w=UsbWatcher(targets=['05ac'],period_s=0.01,root=self.root).start()
        add(self.root,'1-3.1','046d')
        time.sleep(0.05);self.assertFalse(w.connected)
        add(self.root,'1-3.4','05ac','12a8')
        self.assertTrue(wait_for(lambda:w.connected));w.stop()
    def test_target_already_present_counts_as_connected(self):
        add(self.root,'1-3.4','05ac')
        self.assertTrue(UsbWatcher(targets=['05ac:'],root=self.root).connected)
    def test_port_target_matches_any_device_behind_that_hub(self):
        add(self.root,'1-3','05e3')  # the hub itself is not a target
        w=UsbWatcher(targets=['port:1-3'],period_s=0.01,root=self.root).start()
        add(self.root,'1-4.1','05ac');time.sleep(0.05);self.assertFalse(w.connected)
        add(self.root,'1-3.2','2717','ff08')
        self.assertTrue(wait_for(lambda:w.connected));w.stop()


if __name__=='__main__':unittest.main()

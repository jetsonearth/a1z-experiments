"""Offline fixtures only: optional LeRobot playback must not replace raw evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import server


class OptionalMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.patch=patch.object(server,'EPISODES',self.root)
        self.patch.start()
        self.name='fixture-teaching.json'
        self.raw={'metadata':{},'samples':[{'q':[0]*6,'t_monotonic_s':100+i/49} for i in range(5)]}
        self.samples=[{'q':[i]*6,'t_monotonic_s':100+i/30} for i in range(3)]
        self.dataset={'dataset_dir':'/home/dimos/a1z-experiments/2026-10-02/datasets/fixture-unified',
                      'success':True,'validated':True,'frames':3,
                      'video_checks':[{'file':'videos/wrist/file-000.mp4'}]}

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def fetch(self,remote,dest,progress=None):
        if remote.endswith('playback.json'): value={'samples':self.samples}
        elif remote.endswith('timestamps.json'): value={'frames':[{'host_monotonic_s':s['t_monotonic_s']} for s in self.samples]}
        else: value={}
        dest.write_bytes(b'fixture-video' if remote.endswith('.mp4') else json.dumps(value).encode())
        return {'bytes':dest.stat().st_size,'sha256':'fixture'}

    def test_requires_explicit_exporter_success_and_validation(self):
        for flags in [{},{'success':True},{'validated':True},{'success':1,'validated':True},{'success':True,'validated':False}]:
            value={'dataset_dir':self.dataset['dataset_dir'],**flags}
            self.assertIsNone(server.successful_dataset({'lerobot_dataset':value}))
        self.assertEqual(server.successful_dataset({'lerobot_dataset':self.dataset}),self.dataset)

    def test_canonical_reads_derivative_preserves_raw_bytes(self):
        self.raw['metadata']['lerobot_dataset']=self.dataset
        original=json.dumps(self.raw).encode();(self.root/self.name).write_bytes(original)
        with patch.object(server,'fetch_remote_file',side_effect=self.fetch):
            server.sync_media(self.name,self.raw['metadata'])
        result=server.episode_data(self.name)
        self.assertEqual(len(result['samples']),3)
        self.assertEqual(result['raw_sample_count'],5)
        self.assertEqual(result['playback_basis'],'LeRobot 30 Hz 对齐观测')
        self.assertEqual(result['local_media']['source'],'lerobot')
        self.assertEqual((self.root/self.name).read_bytes(),original)

    def test_failed_export_keeps_raw_video_and_error(self):
        metadata={'lerobot_dataset':{'success':False,'validated':False,'error':'readback failed'},
                  'camera':{'dataset_dir':'/home/dimos/a1z-experiments/2026-10-02/datasets/fixture-raw',
                            'video_checks':[{'file':'videos/wrist/file-000.mp4'}]}}
        with patch.object(server,'fetch_remote_file',side_effect=self.fetch):server.sync_media(self.name,metadata)
        media=server.local_media(self.name)
        self.assertEqual(media['source'],'raw')
        self.assertEqual(media['export_error'],'readback failed')

    def test_legacy_without_export_is_not_failure(self):
        (self.root/self.name).write_text(json.dumps(self.raw))
        server.sync_media(self.name,self.raw['metadata'])
        result=server.episode_data(self.name)
        self.assertEqual(result['samples'],self.raw['samples'])
        self.assertIsNone(result['local_media'])


if __name__=='__main__':unittest.main()

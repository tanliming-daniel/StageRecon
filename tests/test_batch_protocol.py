"""Prevent failed stages from being reported as completed datasets."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from tools.batch_protocol import stage_result, portable_bundle_is_valid
from tools.clip_render_job import write_render_job


class BatchProtocolTests(unittest.TestCase):
    def test_dry_run_uses_gpu4_and_registered_anchors_without_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'clips/S08E04/reference_clips/clean/general_f00000001-f00000002.mp4'
            video.parent.mkdir(parents=True);video.write_bytes(b'')
            out=root/'outputs';script=Path(__file__).resolve().parents[1]/'tools/run_clip_render_data_batch.py'
            run=subprocess.run([sys.executable,str(script),'--clips-root',str(root/'clips'),
                                '--output-root',str(out),'--dry-run'],check=True,capture_output=True,text=True)
            plan=json.loads(run.stdout)
            self.assertEqual(plan['stage1_gpu'],'4')
            self.assertEqual(plan['stage2_gpu'],'4')
            self.assertEqual(plan['anchor_source'],'registered-first')
            self.assertEqual(len(plan['jobs']),1)
            self.assertIsNotNone(plan['actor_world_up'])
            self.assertFalse((out/'batch_manifest.json').exists())

    def test_stage2_failure_overrides_stage1_success(self):
        result = stage_result({'type': 'stage1_result', 'status': 'ok', 'clip_key': 'sample'},
                              stage=2, status='failed', error='Depth scale failed')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['type'], 'stage2_result')
        self.assertEqual(result['error'], 'Depth scale failed')
        self.assertEqual(result['clip_key'], 'sample')

    def test_success_does_not_inherit_old_error(self):
        result = stage_result({'status': 'failed', 'error': 'old failure', 'traceback': 'old'},
                              stage=2, status='ok')
        self.assertNotIn('error', result)
        self.assertNotIn('traceback', result)

    def test_four_filenames_alone_are_not_a_valid_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['camera_track.json', 'actor_tracks.json', 'frames.json', 'render_job.json']:
                (root/name).write_text('{}')
            self.assertFalse(portable_bundle_is_valid(root))

    def test_valid_dataset_then_checksum_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = root/'camera_track.json';actor=root/'actor_tracks.json';frames=root/'frames.json'
            camera.write_text(json.dumps({'frames': [{'frame_index': 0}], 'resolution': {'width': 320, 'height': 240}}))
            actor.write_text(json.dumps({'tracks': []}))
            frames.write_text(json.dumps({'frame_count': 1, 'fps': 24.0}))
            write_render_job(output=root/'render_job.json', camera_track_path=camera,
                             actor_tracks_path=actor, frames_path=frames, base_blend=root/'base.blend')
            self.assertTrue(portable_bundle_is_valid(root))
            actor.write_text(json.dumps({'tracks': [], 'changed': True}))
            self.assertFalse(portable_bundle_is_valid(root))

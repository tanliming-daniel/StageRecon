import json,tempfile,unittest
from pathlib import Path
from tools.actor_proxy_config import resolve_actor_world_up

class ActorUpConfigTests(unittest.TestCase):
    def test_explicit_up_is_normalized(self):
        self.assertEqual(resolve_actor_world_up('/scene','base.blend',[0,0,2],'/missing'),[0,0,1])
    def test_scene_configuration_requires_matching_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'scene.json').write_text(json.dumps({'base_blend_filename':'base.blend','raw_world_up':[0,-1,0]}))
            self.assertEqual(resolve_actor_world_up('/scene','base.blend',None,tmp),[0,-1,0])
            self.assertIsNone(resolve_actor_world_up('/scene','another.blend',None,tmp))
    def test_zero_vector_is_rejected(self):
        with self.assertRaises(ValueError):resolve_actor_world_up('/scene','base.blend',[0,0,0],'/missing')

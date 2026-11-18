"""Resolve one fixed actor up vector for the scene; never use camera tilt as floor."""
import json
import math
from pathlib import Path


def resolve_actor_world_up(environment, base_blend, explicit, config_dir):
    values = explicit
    if values is None:
        path = Path(config_dir) / (Path(environment).name + '.json')
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        if data['base_blend_filename'] != Path(base_blend).name:
            return None
        values = data['raw_world_up']
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Actor world up must contain three finite values')
    length = math.sqrt(sum(v*v for v in values))
    if length < 1e-8:
        raise ValueError('Actor world up must be nonzero')
    return [float(v/length) for v in values]

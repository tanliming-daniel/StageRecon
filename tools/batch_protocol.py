"""Authoritative worker outcomes and validation of completed render bundles."""
import hashlib
import json
from pathlib import Path


def stage_result(job, *, stage, status, **details):
    if stage not in (1, 2) or status not in ('ok', 'failed'):
        raise ValueError('Invalid stage result')
    inherited = {k: v for k, v in job.items()
                 if k not in ('type', 'status', 'error', 'traceback', 'failed_stage')}
    return {**inherited, **details, 'type': f'stage{stage}_result', 'status': status}


def portable_bundle_is_valid(directory):
    directory = Path(directory)
    names = {'camera_track.json', 'actor_tracks.json', 'frames.json', 'render_job.json'}
    try:
        if {p.name for p in directory.iterdir()} != names:
            return False
        data = {name: json.loads((directory/name).read_text()) for name in names}
        job = data['render_job.json']
        if job['kind'] != 'stagerecon_clip_render_job':
            return False
        frames = data['camera_track.json']['frames']
        count = len(frames)
        if count == 0 or [int(f['frame_index']) for f in frames] != list(range(count)):
            return False
        if int(data['frames.json']['frame_count']) != count:
            return False
        if int(job['timeline']['frame_count']) != count:
            return False
        for track in data['actor_tracks.json']['tracks']:
            if any(not 0 <= int(f['frame_index']) < count for f in track['frames']):
                return False
        for label, name in [('camera_track','camera_track.json'), ('actor_tracks','actor_tracks.json'), ('frames','frames.json')]:
            if job['data'][label] != name:
                return False
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != job['data']['sha256'][label]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False

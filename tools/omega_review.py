"""Publish and reuse editorially accepted Omega clip bundles."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from batch_protocol import portable_bundle_is_valid

FILES = ('camera_track.json','frames.json','actor_tracks.json','render_job.json')


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def scene_signature(cfg):
    keys=('source_blend_sha256','pointcloud_sha256','predictions_sha256','omega_to_blender','omega_units_per_meter','world_up',
          'render_objects','hide_collections','depth_mesh_objects','reviewed_image_regions')
    content={k:cfg.get(k) for k in keys}
    if cfg.get('contacts'):
        content['contacts_sha256']=sha256(Path(__file__).resolve().parents[1]/cfg['contacts'])
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def copy_bundle(src,dst,overwrite=False):
    if not portable_bundle_is_valid(src):
        raise ValueError(f'Invalid reviewed bundle: {src}')
    if dst.exists() and portable_bundle_is_valid(dst) and sha256(dst/FILES[-1])==sha256(src/FILES[-1]):
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as temp:
        stage=Path(temp)/'bundle'
        shutil.copytree(src,stage)
        if not portable_bundle_is_valid(stage):
            raise ValueError(f'Incomplete bundle copy: {src}')
        if not dst.exists():
            stage.rename(dst)
            return
        if not overwrite or not portable_bundle_is_valid(dst):
            raise ValueError(f'Output already exists: {dst}; use a fresh output root')
        for name in FILES:  # Publish the checksum-bearing job last.
            os.replace(stage/name,dst/name)


def exclude_reviewed_regions(actors, measurements, job, scene):
    rule=scene.get('reviewed_image_regions',{}).get(Path(job.get('clip_key','')).as_posix())
    if not rule or sha256(Path(job['video']))!=rule['video_sha256']:return
    by_track={}
    for r in measurements['rows']:
        by_track.setdefault(r['track'],[]).append(r)
    rejected=set()
    for tid,rows in by_track.items():
        for x0,y0,x1,y1 in rule['rectangles_xyxy_01']:
            def inside(r):
                h,w=r['image'];x,y,bw,bh=r['bbox']
                return x>=x0*w and y>=y0*h and x+bw<=x1*w and y+bh<=y1*h
            if rows and all(inside(r) for r in rows):
                rejected.add(tid)
    actors['reviewed_exclusions']={str(tid):'inside reviewed picture region throughout observations' for tid in sorted(rejected)}
    actors['tracks']=[t for t in actors['tracks'] if t['track_id'] not in rejected]


def publish(args):
    cfg=read(args.scene)
    entries={}
    for case in read(args.cases):
        key=case['clip']
        rel=Path(key)
        if rel.is_absolute() or '..' in rel.parts or key in entries:
            raise ValueError(f'Invalid or duplicate clip path: {key}')
        video=args.clips_root/rel
        src=args.bundles/rel.parts[0]/rel.stem
        if not portable_bundle_is_valid(src):raise ValueError(src)
        cam=read(src/'camera_track.json')['scene_frame']
        job=read(src/'render_job.json')
        if cam['source_blend_sha256']!=cfg['source_blend_sha256'] or cam['pointcloud_sha256']!=cfg['pointcloud_sha256'] or job['source_scene']['sha256']!=cfg['source_blend_sha256']:
            raise ValueError(f'Review belongs to another scene: {key}')
        for key_name in ('render_objects','hide_collections'):
            if key_name in cfg and cam.get(key_name)!=cfg[key_name]:
                raise ValueError(f'Review lacks approved {key_name} settings: {key}')
        matrix_ok = all(math.isclose(cam['world_to_blender'][i][j],cfg['omega_to_blender'][i][j],abs_tol=1e-5)
                        for i in range(4) for j in range(4))
        if not matrix_ok or not math.isclose(cam['units_per_meter'],cfg['omega_units_per_meter'],rel_tol=1e-5):
            raise ValueError(f'Review alignment differs from the scene profile: {key}')
        path=Path('bundles')/rel.parts[0]/rel.stem
        copy_bundle(src,args.release/path)
        entries[rel.as_posix()]=dict(video_sha256=sha256(video),bundle=str(path),job_sha256=sha256(src/'render_job.json'))
    index=dict(profile_sha256=scene_signature(cfg),scene_sha256=cfg['source_blend_sha256'],pointcloud_sha256=cfg['pointcloud_sha256'],clips=entries)
    path=args.release/'index.json'
    if path.exists() and read(path)!=index:raise ValueError('Use a new release directory for changed reviews')
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(index,f,indent=2);tmp=Path(f.name)
    tmp.replace(path)
    print(f'Published {len(entries)} reviewed bundles to {path}')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for flag in ('scene','cases','clips-root','bundles','release'):
        ap.add_argument('--'+flag,type=Path,required=True)
    publish(ap.parse_args())

"""Prepare Omega scene metadata and measured contact surfaces inside Blender."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    from tempfile import NamedTemporaryFile
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, allow_nan=False)
        tmp = Path(f.name)
    tmp.replace(path)


def surface(obj, step):
    # Use evaluated mesh, including modifiers, parent transform and scale.
    obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if obj.type != 'MESH':
        raise ValueError(f'Contact surface must be a mesh: {obj.name}')
    m = obj.matrix_world
    inv = m.inverted()
    corners = np.array([list(m @ Vector(v)) for v in obj.bound_box])
    lo,hi = corners.min(axis=0), corners.max(axis=0)
    top=[]
    for ix in range(math.ceil((hi[0]-lo[0])/step)):
        for iy in range(math.ceil((hi[1]-lo[1])/step)):
            origin=Vector((lo[0]+(ix+.5)*step,lo[1]+(iy+.5)*step,hi[2]+1))
            hit,point,_,_=obj.ray_cast(inv@origin,inv.to_3x3()@Vector((0,0,-1)))
            if hit:
                top.append([ix,iy,round((m@point).z,4)])
    if not top:raise ValueError(f'Empty surface: {obj.name}')
    return dict(object=obj.name,origin=lo[:2].tolist(),step=step,top=top)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-images',type=Path)
    ap.add_argument('--layout',required=True,type=Path)
    ap.add_argument('--reconstruction',required=True,type=Path)
    ap.add_argument('--pointcloud',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    layout=json.loads(args.layout.read_text())
    base=Path(bpy.data.filepath).resolve()
    objects=bpy.context.scene.objects
    mat=np.asarray(objects['pointcloud'].matrix_world)
    basis=mat[:3,:3]
    scale=np.linalg.norm(basis[:,0])
    if scale<=0 or np.linalg.det(basis)<=0 or not np.allclose(basis.T@basis,np.eye(3)*scale**2,rtol=1e-5,atol=1e-6):
        raise ValueError('pointcloud requires a positive uniform similarity transform')
    if not (args.reconstruction/'predictions.npz').is_file():raise FileNotFoundError('predictions.npz')
    step=float(layout.get('step',.045))
    if not np.isfinite(step) or step<=0:raise ValueError('invalid grid step')
    seats=layout.get('seats',{})
    for name,s in seats.items():
        obj=objects[s['object']]
        l,r=np.asarray(s['left']),np.asarray(s['right'])
        front=np.asarray(s['front_direction'])
        if obj.type!='MESH' or l.shape!=(3,) or r.shape!=(3,) or front.shape!=(2,) or not np.all(np.isfinite([*l,*r,*front,s['height']])) or np.linalg.norm(l[:2]-r[:2])<1e-6 or np.linalg.norm(front)<1e-6:
            raise ValueError(f'Invalid seat: {name}')
    for name in [*layout.get('render_objects',[]),*layout.get('depth_mesh_objects',[])]:objects[name]
    floors=[]
    for name in layout.get('floors',[]):
        grid=surface(objects[name],step)
        top=grid.pop('top')
        rows={}
        for x,y,z in top:
            row=rows.setdefault(str(x),[y,y]);row[1]=max(y,row[1])
        grid.update(rows=rows,z=float(np.median([v[2] for v in top])))
        floors.append(grid)
    if layout.get('backrest_seat') and layout['backrest_seat'] not in seats:
        raise ValueError('backrest_seat must identify a configured seat')
    scene_sha=digest(base)
    contacts=dict(scene_sha256=scene_sha,seats=seats,floors=floors,clips={},
                  occluders=[surface(objects[n],step) for n in layout.get('occluders',[])])
    if 'backrest_seat' in layout:contacts['backrest_seat']=layout['backrest_seat']
    contact_path=args.output.resolve().with_name('contacts_'+scene_sha[:12]+'_'+hashlib.sha256(json.dumps(contacts,sort_keys=True).encode()).hexdigest()[:10]+'.json')
    write_json(contact_path,contacts)
    up=np.linalg.solve(basis,[0,0,1]);up/=np.linalg.norm(up)
    result=dict(coordinate_system='omega_ba_world',blend=str(base),source_blend_sha256=scene_sha,
                predictions_sha256=digest(args.reconstruction/'predictions.npz'),
                alignment_object='pointcloud',pointcloud=str(args.pointcloud.resolve()),pointcloud_sha256=digest(args.pointcloud),
                omega_to_blender=mat.tolist(),omega_units_per_meter=float(1/scale),world_up=up.tolist(),
                contacts=str(contact_path),render_objects=layout.get('render_objects',[]),
                hide_collections=layout.get('hide_collections',[]),depth_mesh_objects=layout.get('depth_mesh_objects',[]),
                blender_executable=bpy.app.binary_path)
    if args.reference_images:
        if not args.reference_images.is_dir():raise FileNotFoundError(args.reference_images)
        result['reference_images']=str(args.reference_images.resolve())
    write_json(args.output,result)
    print('Prepared',args.output,'contacts',contact_path,flush=True)


if __name__=='__main__':main()

"""Blender background ray queries in camera-depth metres for selected static meshes."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector

source,destination=map(Path,sys.argv[sys.argv.index('--')+1:])
data=json.loads(source.read_text());camera=data['camera'];sf=camera['scene_frame']
objects=bpy.context.scene.objects
M=Matrix(sf['world_to_blender'])
if np.max(np.abs(np.asarray(objects['pointcloud'].matrix_world)-np.asarray(M)))>1e-5:
    raise ValueError('Camera and Blender alignment disagree')
for name in data['render_objects']:objects[name].hide_render=False
deps=bpy.context.evaluated_depsgraph_get()
mesh=[]
for name in data['objects']:
    o=objects[name].evaluated_get(deps)
    if not o.hide_render and o.type=='MESH':mesh.append((o,o.matrix_world.inverted()))
caps={}
for q in data['queries']:
    c=camera['frames'][q['frame']]
    origin=M@Vector(c['camera_center'])
    unproject=M.to_3x3()@Matrix(c['world_to_camera_rotation']).transposed()@Matrix(c['intrinsics']).inverted()
    nearest=[]
    for uv in q['uv']:
        ray=unproject@Vector((*uv,1))
        depths=[]
        for o,inv in mesh:
            hit,pt,_,_=o.ray_cast(inv@origin,inv.to_3x3()@ray)
            if hit:
                z=(o.matrix_world@pt-origin).dot(ray)/ray.length_squared/sf['units_per_meter']
                if z>0:depths.append(z)
        if depths:nearest.append(min(depths))
    if nearest:caps.setdefault(q['track'],[]).append(float(np.quantile(nearest,.1)))
destination.write_text(json.dumps(caps))

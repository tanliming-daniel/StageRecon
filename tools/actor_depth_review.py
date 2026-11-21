"""Recheck implausibly large proxies against measured torso depth and scene occlusion."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from estimate_actor_boxes import fixed_proxy_box


def extreme_tracks(actors, camera, measurements):
    rows = {(r['track'],r['frame']):r for r in measurements['rows']}
    result={}
    unit=camera['scene_frame']['units_per_meter']
    for track in actors['tracks']:
        reliable={}
        ratios=[]; widths=[]
        for f in track['frames']:
            i=f['frame_index']
            if f.get('pelvis_update')!='observed' or not f.get('shoulder_observed'):continue
            row=next((rows[key] for tid in track.get('source_track_ids',[track['track_id']])
                      if (key:=(tid,i)) in rows and 'xyz' in rows[key]),None)
            if row is None or min(row['support'][5:7])<1:continue
            xyz=np.asarray(row['xyz']);uv=np.asarray(row['uv'])
            cam=camera['frames'][i];R=np.asarray(cam['world_to_camera_rotation'])
            z=(R@f['shoulder_world']+cam['world_to_camera_translation'])[2]
            model_z=xyz[[5,6],2].mean()*unit
            if z<=0 or model_z<=0:continue
            focal=cam['intrinsics'][0][0]
            observed=max(1.,np.linalg.norm(uv[5]-uv[6]))
            widths.append(2*track['proxy_radius_world']*focal/(z*observed))
            ratios.append(model_z/z);reliable[i]=row
        # Require independent agreement of metric depth and projected shoulder width.
        if len(ratios)>=max(4,.65*len(track['frames'])) and np.quantile(ratios,.2)>2.5 and np.quantile(widths,.2)>1.8:
            result[track['track_id']]=reliable
    return result


def correct_large_proxies(actors,camera,measurements,base,config,tmp):
    targets=extreme_tracks(actors,camera,measurements)
    if not targets:return
    unit=camera['scene_frame']['units_per_meter'];conv=actors['proxy_convention']
    up=np.asarray(conv['up_world']);height=conv['height_world']
    requests=[]
    for tid,rows in targets.items():
        for i,r in rows.items():
            uv=np.asarray(r['uv']);a,b=uv[5:7];w,h=r['image'][::-1]
            probes=[a+t*(b-a)+[0,q*np.linalg.norm(a-b)] for t in (.2,.4,.6,.8) for q in (.2,.38,.55)]
            requests.append(dict(track=tid,frame=i,uv=[v.tolist() for v in probes if 0<=v[0]<w and 0<=v[1]<h-10]))
    limits={}
    if config.get('depth_mesh_objects'):
        payload=Path(tmp)/'depth_queries.json';output=Path(tmp)/'depth_limits.json'
        payload.write_text(json.dumps(dict(camera=camera,queries=requests,objects=config['depth_mesh_objects'],
                                          render_objects=config.get('render_objects',[]))))
        blender=os.environ.get('STAGERECON_BLENDER') or config.get('blender_executable') or shutil.which('blender')
        if not blender:raise FileNotFoundError('Set STAGERECON_BLENDER for scene-depth queries')
        subprocess.run([str(blender),'-b',str(base),'-t','4','--python-exit-code','2','-P',
                        str(Path(__file__).with_name('query_actor_occlusion.py')),'--',str(payload),str(output)],check=True,stdout=sys.stderr)
        limits=json.loads(output.read_text())
    for track in actors['tracks']:
        tid=track['track_id']
        if tid not in targets:continue
        rows=targets[tid];ids=np.array(sorted(rows));caps=limits.get(str(tid),[])
        cap=float(np.median(caps)-.16) if caps else np.inf
        if cap<=0:continue
        zs=[];probes=[];widths=[];ear_spans=[];offsets=[]
        for i in ids:
            r=rows[i];v=np.array(r['xyz']);uv=np.array(r['uv'])
            zs.append(min(cap,v[[5,6],2].mean()));xy=uv[[5,6]].mean(axis=0)
            contour=r.get('outer_shoulder_y')
            if contour is None:contour=r.get('shoulder_contour_y')
            if contour is not None:xy[1]=contour
            probes.append(xy);widths.append(np.linalg.norm(v[5]-v[6]));ear_spans.append(np.linalg.norm(v[3]-v[4]))
            R=np.asarray(camera['frames'][i]['world_to_camera_rotation']);offsets.append(R.T@(v[[7,8]].mean(0)-v[[5,6]].mean(0))*unit)
        radius=float(np.median(widths)*.56*unit);head=float(np.median(ear_spans)*.65*unit)
        world=[]
        for j,i in enumerate(ids):
            c=camera['frames'][i];z=np.median(np.asarray(zs)[abs(ids-i)<=4])*unit
            xy=probes[j].copy();xy[1]=np.median(np.asarray(probes)[abs(ids-i)<=2,1])
            world.append(np.asarray(c['world_to_camera_rotation']).T@(np.linalg.solve(c['intrinsics'],[*xy,1])*z-c['world_to_camera_translation']))
        dense=np.column_stack([np.interp(np.arange(len(track['frames'])),ids,v) for v in np.asarray(world).T])
        track.update(proxy_radius_world=radius,proxy_head_radius_world=head,depth_source='sam3d_shoulders_with_scene_cap',
                     support_segments=[dict(first=0,last=len(dense)-1,support='depth')])
        track.pop('seat_contact',None)
        for f,sh in zip(track['frames'],dense):
            top=sh+2*head*up;center=top-.5*height*up
            for k in ('seat_object','rear_head_clearance'):f.pop(k,None)
            f.update(shoulder_world=sh.tolist(),head_anchor_world=top.tolist(),proxy_center_world=center.tolist(),
                     position_world=center.tolist(),support='depth',support_confidence=0.,pelvis_world=(sh+np.median(offsets,axis=0)).tolist(),
                     body_height_world=height,position_source='sam3d_shoulder_depth',animation_keyframe=True)
            f['proxy_bbox_3d']=fixed_proxy_box(center,f['facing_world'],up,height,radius)

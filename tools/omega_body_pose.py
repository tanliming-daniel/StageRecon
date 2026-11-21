"""Fuse independently supported SAM 3D Body observations in Omega coordinates."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__:
    from .omega_actor_keyframes import stabilize, project, fixed_proxy_box
else:
    from omega_actor_keyframes import stabilize, project, fixed_proxy_box


def evidence(row):
    if 'support' not in row:
        return False, False
    d = np.asarray(row['support'])
    uv = np.asarray(row['uv'])
    # A nose and eye/ear cluster must lie in this person's foreground.
    face = (d[0] >= 0 and max(d[1:3]) >= 1 and (d[:5] >= 1).sum() >= 3
            and np.linalg.norm(uv[3]-uv[4]) >= 5)
    # For cropped medium shots a pelvis estimate can be inferred from a broad
    # visible upper torso. A lone shoulder or forearm is never sufficient.
    chest = min(d[5],d[6],d[9],d[10]) >= 1 and d[12] >= 1
    hips = (min(d[7],d[8],d[11],d[12]) >= 1 and max(d[5],d[6]) >= 1)
    upper = (len(d)>=16 and min(d[5],d[6],d[9],d[10])>=1
             and d[15]>=2 and max(d[13],d[14])>=2)
    return bool(face), bool(chest or hips or upper)


def unit(v):
    return v / max(1.e-9,np.linalg.norm(v))


def head_track(rows, cameras, count):
    ids = sorted(i for i,r in rows.items() if evidence(r)[0])
    if not ids:
        return {}, []
    # Determine the anatomical rest axes once per track from the model's 3-D face.
    # MHR70 uses left ear 3 and right ear 4. The Blender band faces local +X.
    axes = []
    for i in ids:
        r = rows[i]
        face = np.asarray(r['xyz'])
        left = unit(face[3]-face[4])
        fwd = face[0]-(face[3]+face[4])/2
        fwd = unit(fwd-left*(left@fwd))
        axes.append(np.asarray(r['head_rotation']).T @ np.column_stack((fwd,left,np.cross(fwd,left))))
    rest = Rotation.from_matrix(axes).mean().as_matrix()
    orientations = []
    for i in ids:
        w2c = np.asarray(cameras[i]['world_to_camera_rotation'])
        orientations.append(w2c.T @ rows[i]['head_rotation'] @ rest)
    rotations = Rotation.from_matrix(orientations)
    # Reject isolated discordant estimates with a local angular medoid.
    robust = []
    for j,i in enumerate(ids):
        nearby = np.flatnonzero(abs(np.asarray(ids)-i)<=2)
        candidates = rotations[nearby]
        cost = [sum((q.inv()*candidates).magnitude()) for q in candidates]
        robust.append(candidates[np.argmin(cost)].as_matrix())
    knots = Rotation.from_matrix(robust)
    dense = (Slerp(ids,knots)(np.clip(np.arange(count),ids[0],ids[-1])).as_matrix()
             if len(ids)>1 else np.repeat(knots.as_matrix(),count,axis=0))
    return dense, ids


def status(i, ids):
    if not ids:
        return 'no_measurement'
    if i in ids:
        return 'observed'
    return 'hold' if i < ids[0] or i > ids[-1] else 'interpolated'


def stabilize_body(actors, camera, contacts, clip, fps, measurements):
    cams = camera['frames']
    units = camera['scene_frame']['units_per_meter']
    indexed = {}
    for row in measurements['rows']:
        indexed.setdefault(row['track'],{})[row['frame']] = row
    rejected = []
    used = {}
    for track in actors['tracks']:
        tid = track['track_id']
        rows = indexed.get(tid,{})
        supported = [f for f in track['frames'] if any(evidence(rows.get(f['frame_index'],{})))]
        ids = [f['frame_index'] for f in supported]
        if not any(0<b-a<=3 for a,b in zip(ids,ids[1:])) and not (len(cams)==1 and ids):
            rejected.append(tid)
            continue
        if not track.get('proxy_quality',{}).get('eligible',True):
            rejected.append(tid)
            continue
        track['original_mask_range'] = [track['frames'][0]['frame_index'],track['frames'][-1]['frame_index']]
        trusted = [f for f in supported if evidence(rows[f['frame_index']])[1]]
        # Face-only characters receive one scene anchor; no pelvis velocity is inferred.
        if not trusted:
            track['body_depth_only'] = True
            trusted = [max(supported,key=lambda f: rows[f['frame_index']]['area'])]
        else:
            observed = {}
            for f in trusted:
                i = f['frame_index']
                hips = np.mean(np.asarray(rows[i]['xyz'])[[7,8]],axis=0)
                c = cams[i]
                observed[i] = (hips*units-c['world_to_camera_translation']) @ np.asarray(c['world_to_camera_rotation'])
            sample = np.stack(list(observed.values()))
            # A single offset per identity anchors monocular translation to scene depth.
            offset = np.median([np.asarray(f['head_anchor_world'])-observed[f['frame_index']] for f in trusted],axis=0)
            stride = max(1,round(fps/2))
            times = np.array(sorted(observed))
            knots = np.unique(np.r_[times[0],times[np.searchsorted(times,np.arange(times[0],times[-1],stride))],times[-1]])
            values = [np.median(sample[np.abs(times-k)<=max(1,stride//2)],axis=0) for k in knots]
            # Restrict smooth root movement when the camera lacks PnP support.
            low_camera = camera.get('absolute_anchor_policy')=='estimated'
            for j in range(1,len(knots)):
                delta = values[j]-values[j-1]
                cap = (1.0 if low_camera else 2.5)*units*(knots[j]-knots[j-1])/fps
                values[j] = values[j-1]+delta*min(1,cap/max(1.e-9,np.linalg.norm(delta)))
            pelvis = np.column_stack([np.interp(times,knots,col) for col in np.asarray(values).T])
            for f,point in zip(trusted,pelvis):
                f['head_anchor_world'] = (point+offset).tolist()
                f['body_pelvis_measured_world'] = observed[f['frame_index']].tolist()
            track['body_alignment_offset_world'] = offset.tolist()
        track['frames'] = trusted
        used[tid] = rows
    actors['tracks'] = [t for t in actors['tracks'] if t['track_id'] in used]
    stabilize(actors,camera,contacts,clip,fps)
    if __package__:
        from .actor_shoulders import align_shoulders
    else:
        from actor_shoulders import align_shoulders
    for t in actors['tracks']:
        rows = {i:r for tid in t['source_track_ids'] for i,r in used[tid].items()}
        conv = actors['proxy_convention']
        catalog = contacts if contacts.get('scene_sha256')==camera['scene_frame']['source_blend_sha256'] else {}
        align_shoulders(t,rows,camera,conv,catalog,fps)
        t['support_segments'] = []
        for f in t['frames']:
            f['proxy_bbox_3d'] = fixed_proxy_box(f['proxy_center_world'],f['facing_world'],
                np.asarray(conv['up_world']),conv['height_world'],t['proxy_radius_world'])
            if t['support_segments'] and t['support_segments'][-1]['support']==f['support']:
                t['support_segments'][-1]['last']=f['frame_index']
            else:
                t['support_segments'].append(dict(first=f['frame_index'],last=f['frame_index'],support=f['support']))
        dense, heads = head_track(rows,cams,len(cams))
        if not len(dense):
            up = unit(np.asarray(conv['up_world']))
            front = np.asarray(t['frames'][0]['facing_world'])
            front = unit(front-up*(front@up))
            if np.linalg.norm(front)<.5:
                front = unit(np.cross(up,np.eye(3)[np.argmin(abs(up))]))
            basis = np.column_stack((front,np.cross(up,front),up))
            dense = np.repeat(basis[None],len(cams),axis=0)
            t['head_fallback'] = 'held_initial_body_orientation'
        pelvis_ids = sorted(i for i,r in rows.items() if evidence(r)[1])
        for f in t['frames']:
            i = f['frame_index']
            if len(dense):
                f['head_rotation_world'] = dense[i].tolist()
            f['head_update'] = status(i,heads)
            f['pelvis_update'] = status(i,pelvis_ids)
            if f.get('observation_status','').endswith('extrapolation'):
                f['pelvis_update'] = f['observation_status']
            if 'pelvis_world' not in f and 'body_alignment_offset_world' in t:
                f['pelvis_world'] = (np.asarray(f['head_anchor_world'])-t['body_alignment_offset_world']).tolist()
        t['body_evidence'] = dict(head_frames=len(heads),pelvis_frames=len(pelvis_ids))
    actors['animation_method'] = 'omega_sam3d_shoulders_v2'
    actors['fragment_track_ids'] = rejected

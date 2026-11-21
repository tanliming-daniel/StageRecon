"""Anchor the cylinder top at the observed shoulder line and clear the couch back."""
import numpy as np


def shoulder_contour(mask, uv):
    # Use the garment silhouette above each outer shoulder, with bounded reach
    # to avoid selecting the head or a raised hand.
    a,b = uv[5:7]
    span = abs(a[0]-b[0])
    if span<10:return None
    edges=[]
    for side,other in ((a,b),(b,a)):
        samples=[]
        for t in np.linspace(.01,.19,7):
            x,y = np.rint(side+t*(other-side)).astype(int)
            lo=max(0,int(y-.29*span))
            if 0<=x<mask.shape[1] and lo<min(mask.shape[0],y+1):
                col=mask[lo:y+1,x]
                hits=np.flatnonzero(col)
                if len(hits) and hits[0]>0:samples.append(lo+hits[0])
        if len(samples)>=3:edges.append(np.median(samples))
    return float(np.mean(edges)) if len(edges)==2 else None


def outer_shoulder_height(mask, uv):
    ears=np.sort(uv[[3,4],0]);width=ears[1]-ears[0]
    if width<5:return None
    sides=[]
    for xs in (np.linspace(ears[0]-.85*width,ears[0]-.40*width,11),
               np.linspace(ears[1]+.40*width,ears[1]+.85*width,11)):
        heights=[]
        for x in xs.astype(int):
            if 0<=x<mask.shape[1]:
                col=np.flatnonzero(mask[:,x])
                if len(col) and uv[0,1]-.12*width<col[0]<uv[[5,6],1].mean()+.45*width:
                    heights.append(col[0])
        if heights:sides.append(np.median(heights))
    return float(np.mean(sides)) if sides else None


def align_shoulders(track, rows, camera, convention, catalog, fps):
    frames = track['frames']
    cams = camera['frames']
    matrix = np.asarray(camera['scene_frame']['world_to_blender'])
    inverse = np.linalg.inv(matrix)
    scale = camera['scene_frame']['units_per_meter']
    height = convention['height_world']
    up = np.asarray(convention['up_world'])
    radius = convention['radius_world']
    valid = {}
    spans = []
    for i, row in rows.items():
        if 'support' not in row or min(row['support'][5:7]) < 1 or row['support'][10] < 1:
            continue
        valid[i] = np.mean(np.asarray(row['uv'])[[5,6]],axis=0)
        xyz = np.asarray(row['xyz'])
        c=cams[i]
        z=(np.asarray(c['world_to_camera_rotation'])@frames[i]['head_anchor_world']+c['world_to_camera_translation'])[2]
        spans.append(np.linalg.norm(xyz[5]-xyz[6])*z/max(scale*.1,scale*xyz[[5,6],2].mean()))
        if row.get('shoulder_contour_y') is not None:
            valid[i][1] = row['shoulder_contour_y']
    if spans:
        radius = scale*np.clip(np.median(spans)*.56,.17,.32)
    track['proxy_radius_world'] = float(radius)
    head_radius = float(np.clip(.55*radius,.105*scale,.165*scale))
    track["proxy_head_radius_world"] = head_radius
    tops = np.array([f['head_anchor_world'] for f in frames])
    shoulder = tops-2*head_radius*up
    locations = shoulder@matrix[:3,:3].T+matrix[:3,3]
    sofa_label = catalog.get('backrest_seat','sofa')
    sofa = catalog.get('seats',{}).get(sofa_label)
    sofa_mask = np.array([f.get('support')==sofa_label for f in frames])
    if sofa:
        left,right = (np.asarray(sofa[k][:2]) for k in ('left','right'))
        axis = right-left
        length = np.linalg.norm(axis)
        axis /= length
        front = np.asarray(sofa['front_direction'])
        front /= np.linalg.norm(front)
        side = (locations[:,:2]-left)@axis
        depth = (locations[:,:2]-left)@front
        # A roughly stationary actor in the cushion footprint with a seated
        # shoulder height can be identified even with just a head visible.
        near = ((side>=-.28)&(side<=length+.28)&(depth>=-.70)&(depth<=.26)
                &(locations[:,2]>sofa['height']+.25)&(locations[:,2]<sofa['height']+.86))
        if np.mean(near)>.70 and np.ptp(side)<.48 and np.ptp(depth)<.45:
            sofa_mask |= np.array([f.get('support') in (sofa_label,'depth') for f in frames])
        # Sample the high, rear portion of the measured sofa surface.
        surface = next((o for o in catalog['occluders'] if o['object']==sofa['object']),None)
        back = -.20
        if surface:
            points = np.asarray(surface['top'])
            xy = (points[:,:2]+.5)*surface['step']+surface['origin']
            s = (xy-left)@axis
            d = (xy-left)@front
            ridge = d[(s>.15)&(s<length-.15)&(points[:,2]>sofa['height']+.28)&(d<0)]
            if len(ridge):
                back = float(np.quantile(ridge,.98)+surface['step']/2)
        clearance = max(.015,back+radius/scale+.025)
        target_depth = np.clip(depth,clearance,max(clearance,.14))
        locations[sofa_mask,:2] += np.outer((target_depth-depth)[sofa_mask],front)
    shoulder = locations@inverse[:3,:3].T+inverse[:3,3]
    ids = sorted(valid)
    correction = np.zeros_like(shoulder)
    if ids:
        shifts = []
        for i in ids:
            c = cams[i]
            r = np.asarray(c['world_to_camera_rotation'])
            k = np.asarray(c['intrinsics'])
            cam = r@shoulder[i]+c['world_to_camera_translation']
            target = np.linalg.solve(k,[*valid[i],1])*cam[2]
            shifts.append(r.T@(target-cam))
        times = np.asarray(ids)
        shifts = np.asarray(shifts)
        smoothed = np.array([np.median(shifts[abs(times-i)<=max(1,round(fps*.09))],axis=0) for i in ids])
        correction = np.column_stack([np.interp(np.arange(len(frames)),ids,col) for col in smoothed.T])
    shoulder += correction
    points = shoulder@matrix[:3,:3].T+matrix[:3,3]
    # In a rear view the shoulder line may itself be hidden. Constrain a
    # confirmed head against the measured back crest, at its actual seat depth.
    crest = None
    if sofa and surface and len(ridge):
        high = np.asarray(surface['top'])
        crest = float(np.quantile(high[:,2],.98))
    seat_top = {}
    if crest is not None:
        for i in np.flatnonzero(sofa_mask):
            r=rows.get(i,{})
            if 'uv' not in r or sum(v>=1 for v in r['support'][:5])<3:continue
            cam=cams[i]
            origin=matrix[:3,:3]@np.asarray(cam['camera_center'])+matrix[:3,3]
            if (origin[:2]-left)@front>=0:continue
            # Use the visible crown in this mask only when it belongs to the
            # predicted head. A raised hand cannot pull the head upwards.
            uv=np.asarray(r['uv']); nose=uv[0]; earspan=np.linalg.norm(uv[3]-uv[4])
            crown=r['bbox'][1]
            if not 0<=nose[1]-crown<=max(22,2.4*earspan):continue
            ray=matrix[:3,:3]@np.asarray(cam['world_to_camera_rotation']).T@np.linalg.solve(cam['intrinsics'],[nose[0],crown,1])
            along=ray[:2]@front
            if along<=1e-6:continue
            plane=np.clip((points[i,:2]-left)@front,clearance,max(clearance,.14))
            t=(plane-(origin[:2]-left)@front)/along
            q=origin+t*ray
            diameter=2*head_radius/scale
            seat_top[i]=(np.clip((q[:2]-left)@axis,-.10,length+.10),
                         np.clip(max(q[2],crest+diameter+.065),crest+diameter+.065,sofa['height']+1.15))
    times=np.array(sorted(seat_top))
    if len(times):
        values=np.array([seat_top[j] for j in times])
        smooth=np.array([np.median(values[abs(times-j)<=max(1,round(fps*.12))],axis=0) for j in times])
        side_fit,top_fit=[np.interp(np.arange(len(frames)),times,v) for v in smooth.T]
    for i,f in enumerate(frames):
        if sofa_mask[i]:
            xy = points[i,:2]
            side = np.clip((xy-left)@axis,-.10,length+.10)
            depth = np.clip((xy-left)@front,clearance,max(clearance,.14))
            cam_origin=matrix[:3,:3]@np.asarray(cams[i]['camera_center'])+matrix[:3,3]
            if len(times) and (cam_origin[:2]-left)@front<0:
                side=side_fit[i]
                points[i,2]=max(points[i,2],top_fit[i]-2*head_radius/scale)
                f['rear_head_clearance']=True
            xy[:] = left+side*axis+depth*front
            shoulder[i] = inverse[:3,:3]@points[i]+inverse[:3,3]
            pelvis = inverse[:3,:3]@np.r_[xy,sofa['height']+.09]+inverse[:3,3]
            f.update(support=sofa_label,posture='sitting',seat_object=sofa['object'],pelvis_world=pelvis.tolist())
            f['body_height_world'] = max(.32*scale,(shoulder[i]-pelvis)@up+2*head_radius+.16*scale)
        top = shoulder[i]+2*head_radius*up
        shift = top-tops[i]
        if 'pelvis_world' in f and not sofa_mask[i]:
            f['pelvis_world'] = (np.asarray(f['pelvis_world'])+shift).tolist()
        f['head_anchor_world'] = top.tolist()
        center = top-height*.5*up
        f['proxy_center_world'] = center.tolist()
        f['position_world'] = f['pelvis_world'] if sofa_mask[i] else center.tolist()
        f['shoulder_world'] = shoulder[i].tolist()
        f['shoulder_observed'] = i in valid
        f['animation_keyframe'] = True
    track['shoulder_frames'] = len(ids)

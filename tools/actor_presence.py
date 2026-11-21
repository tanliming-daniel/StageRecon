"""Resolve fragmented IDs and complete actor motion across an entire shot."""
from copy import deepcopy
import numpy as np


def stitch_tracks(tracks, fps, units, cameras):
    # Join only mutually unambiguous, non-overlapping track fragments.
    from operator import itemgetter
    live = [t for t in tracks if t["frames"] and t.get("proxy_quality", {}).get("eligible", True)]
    other = [t for t in tracks if not t["frames"] or not t.get("proxy_quality", {}).get("eligible", True)]
    for t in live:
        t["frames"].sort(key=itemgetter("frame_index"))
        t["source_track_ids"] = [t["track_id"]]
    def match(a,b):
        if ("original_mask_range" in a and "original_mask_range" in b
                and a["original_mask_range"][1] >= b["original_mask_range"][0]):
            return None
        x,y = a["frames"][-1], b["frames"][0]
        dt=y["frame_index"]-x["frame_index"]
        if not 0<dt<=fps*1.5: return None
        old=np.asarray(x["head_anchor_world"])
        j=max(0,len(a["frames"])-5)
        prev=a["frames"][j]
        v=(old-prev["head_anchor_world"])/max(1,x["frame_index"]-prev["frame_index"])
        pred=old+v*min(dt,fps*.3)
        d=np.linalg.norm(pred-y["head_anchor_world"])/units
        if d>min(.65,.18+.035*dt):return None
        cam=cameras[y["frame_index"]]
        pc=np.asarray(cam["world_to_camera_rotation"])@pred+cam["world_to_camera_translation"]
        k=np.asarray(cam["intrinsics"])
        if pc[2]<=0:return None
        uv=(k@pc)[:2]/pc[2]
        box=y.get("bbox_xyxy",[0,0,100,200])
        u=np.linalg.norm(uv-y["body_anchor_xy"])/max(50,box[3]-box[1])
        if u>.34:return None
        return d+.6*u+.003*dt
    while True:
        edges=[]
        for a in live:
            for b in live:
                if a is b:continue
                score=match(a,b)
                if score is not None:edges.append((score,a["track_id"],b["track_id"]))
        pairs=[]
        for cost,i,j in sorted(edges):
            if any((a==i or b==j) and (a,b)!=(i,j) and c<cost+.13 for c,a,b in edges):continue
            pairs.append((i,j))
        if not pairs:break
        i,j=pairs[0]
        a=next(t for t in live if t["track_id"]==i); b=next(t for t in live if t["track_id"]==j)
        a["frames"].extend(b["frames"]);a["source_track_ids"].extend(b["source_track_ids"])
        if "original_mask_range" in a and "original_mask_range" in b:
            a["original_mask_range"][1] = b["original_mask_range"][1]
        live.remove(b)
    return live+other


def complete(track, cameras, fps, units, convention):
    observed={f["frame_index"]:f for f in track["frames"]}
    if not observed:return
    ids=sorted(observed); count=len(cameras)
    track["observed_ranges"]=[]
    for i in ids:
        if not track["observed_ranges"] or track["observed_ranges"][-1][1]+1!=i:
            track["observed_ranges"].append([i,i])
        else:track["observed_ranges"][-1][1]=i
    up=np.asarray(convention["up_world"]); height=convention["height_world"]
    def endpoint(index, sign, distance):
        item=observed[index]; head=np.array(item["head_anchor_world"])
        subset=[i for i in ids if 0 < (i-index)*sign <= fps*.35]
        if not subset or item.get("support") not in ("ground","depth"):
            return head,"hold"
        j=max(subset,key=lambda x:abs(x-index))
        velocity=(head-np.asarray(observed[j]["head_anchor_world"]))/abs(j-index)
        velocity-=up*velocity.dot(up)
        speed=np.linalg.norm(velocity)
        if speed*fps < .18*units:return head,"hold"
        velocity*=min(1,2.3*units/(speed*fps))
        # Use relative world motion; pure camera movement gives zero displacement.
        horizon=max(1,round(fps*.55)); n=min(distance,horizon)
        predicted=head+velocity*(n-n*n/(2*horizon))
        if item.get("touches_image_boundary"):
            cam=cameras[index]
            r=np.asarray(cam["world_to_camera_rotation"]);k=np.asarray(cam["intrinsics"])
            p=r@head+cam["world_to_camera_translation"]
            dv=r@velocity
            if p[2]>0:
                # Continue to the nearest horizontal edge only with outward evidence.
                px=(k@p)[0]/p[2]; dx=k[0,0]*(dv[0]*p[2]-p[0]*dv[2])/p[2]**2
                width=2*k[0,2]+1;box=item.get("bbox_xyxy",[0,0,0,0])
                half=max(5,(box[2]-box[0])/2)
                edge=-half if dx<0 else width+half
                if abs(dx)>.3 and ((dx<0 and box[0]<=2) or (dx>0 and box[2]>=width-3)):
                    # Solve pinhole projection for displacement along the measured
                    # world velocity, including change of camera-space depth.
                    denom=k[0,0]*dv[0]+(k[0,2]-edge)*dv[2]
                    travel=(edge-px)*p[2]/denom if abs(denom)>1e-8 else -1
                    if 0<travel<=2*fps and p[2]+travel*dv[2]>0:
                        span=max(1,2*travel); n=min(distance,span)
                        return head+velocity*(n-n*n/(2*span)),"offscreen_extrapolation" if n<span else "hold"
        return predicted,"motion_extrapolation" if distance<horizon else "hold"
    dense=[]
    for i in range(count):
        if i in observed:
            f=observed[i];f["observation_status"]="observed"
        else:
            k=int(np.searchsorted(ids,i))
            lo=ids[k-1] if k else None;hi=ids[k] if k<len(ids) else None
            src=observed[lo if lo is not None else hi]
            f={key:deepcopy(src[key]) for key in ("support","support_confidence","facing_world","proxy_up_world","pelvis_world","seat_object","body_height_world") if key in src}
            if lo is None or hi is None:
                index=hi if lo is None else lo
                h,status=endpoint(index,1 if lo is None else -1,abs(i-index))
            else:
                t=(i-lo)/(hi-lo); a,b=observed[lo],observed[hi]
                front=(1-t)*np.array(a["facing_world"])+t*np.array(b["facing_world"])
                if np.linalg.norm(front)>1e-6:f["facing_world"]=(front/np.linalg.norm(front)).tolist()
                h=(1-t)*np.array(a["head_anchor_world"])+t*np.array(b["head_anchor_world"])
                status="interpolated"
                f["body_height_world"]=(1-t)*a.get("body_height_world",height)+t*b.get("body_height_world",height)
                if "pelvis_world" in a and "pelvis_world" in b:
                    f["pelvis_world"]=((1-t)*np.array(a["pelvis_world"])+t*np.array(b["pelvis_world"])).tolist()
                if a.get("support")!=b.get("support"):f["support"]="transition"
            f.update(frame_index=i,frame_name=cameras[i].get("frame_name",f"{i:06}.jpg"),
                     head_anchor_world=h.tolist(),observation_status=status,position_source=status)
            center=(h-up*height/2).tolist()
            f["position_world"]=f.get("pelvis_world",center);f["proxy_center_world"]=center
            f["animation_keyframe"]=status.startswith(("offscreen","motion")) or i in (0,count-1) or (i>0 and dense[-1]["observation_status"]!=status)
        if "body_height_world" not in f:
            f["body_height_world"]=height
        if "proxy_up_world" not in f:
            f["proxy_up_world"]=up.tolist()
        dense.append(f)
    # The endpoints of each observed range bound every interpolation.
    for a,b in zip(dense,dense[1:]):
        if a["observation_status"]!=b["observation_status"]:
            a["animation_keyframe"]=b["animation_keyframe"]=True
    for start,end in track["observed_ranges"]:
        dense[start]["animation_keyframe"]=dense[end]["animation_keyframe"]=True
    track.update(first_frame=0,last_frame=count-1,frame_count=count,frames=dense,
                 animation_segments=[{"first":0,"last":count-1,"keys":[f["frame_index"] for f in dense if f.get("animation_keyframe")]}])

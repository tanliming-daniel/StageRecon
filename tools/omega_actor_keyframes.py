"""Fit sparse actor animation in Omega coordinates, with optional measured seats."""
import numpy as np

if __package__:
    from .estimate_actor_boxes import fixed_proxy_box
else:
    from estimate_actor_boxes import fixed_proxy_box


def affine(matrix, xyz):
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def project(camera, points):
    p = np.asarray(points) @ np.asarray(camera["world_to_camera_rotation"]).T + camera["world_to_camera_translation"]
    k = np.asarray(camera["intrinsics"])
    return p[..., :2] / np.maximum(p[..., 2:], 1e-8) * [k[0, 0], k[1, 1]] + k[:2, 2]


def fit_seat(seat, heads, origins, surface, units, matrix, reliable):
    left, right = np.array(seat["left"]), np.array(seat["right"])
    axis = right[:2] - left[:2]
    rays = (heads - origins) / surface[:, None]
    offsets = np.linspace(-1.2, 0.65, 186) * units
    candidates = affine(matrix, heads[reliable, None] + offsets[None, :, None] * rays[reliable, None])
    t = np.clip(((candidates[..., :2]-left[:2]) @ axis) / (axis @ axis), 0, 1)
    feet = left[:2] + t[..., None] * axis
    side_error = np.linalg.norm(candidates[..., :2] - feet, axis=-1)
    height = candidates[..., 2] - seat["height"]
    # Contact is approximate; allow the torso to lean over a seated pelvis.
    cost = side_error + 0.5*np.maximum(0, 0.65-height) + 0.5*np.maximum(0,height-1.1)
    score = np.median(cost, axis=0) + 0.025 * abs(offsets/units - 0.15)
    best = int(np.argmin(score))
    seat_xy = np.median(feet[:, best], axis=0)
    pelvis = affine(np.linalg.inv(matrix), np.r_[seat_xy, seat["height"] + 0.09])
    front = np.linalg.solve(matrix[:3,:3], np.r_[seat["front_direction"], 0])
    return heads + rays * offsets[best], pelvis, front/np.linalg.norm(front), float(offsets[best])


def fit_segment(items, camera, track, seat, stride, units, matrix, convention):
    n = len(items)
    ids = np.array([f["frame_index"] for f in items])
    cams = [camera[i] for i in ids]
    head = np.array([f["head_anchor_world"] for f in items])
    origin = np.array([c["camera_center"] for c in cams])
    surface = np.array([f["depth"]["visible_surface_world"] for f in items])
    confidence = np.array([f["depth"]["confidence"] for f in items])
    good = np.flatnonzero(confidence >= max(0.05, np.quantile(confidence, .2)))
    if len(good) == 0:
        good = np.arange(n)
    up = np.array(convention["up_world"])
    pelvis = front = None
    if seat:
        center_z = np.array([f["depth"]["median_world"] for f in items])
        surface_heads = origin + (head-origin) * (surface/center_z)[:,None]
        head, pelvis, front, offset = fit_seat(seat, surface_heads, origin, surface, units, matrix, good)
        track["seat_contact"] = {"object": seat["object"], "pelvis_world": pelvis.tolist(),
                                  "surface_offset_m": offset/units, "facing_world": front.tolist()}
    # Median over nearby reliable observations suppresses isolated bad head samples.
    observed = head.copy()
    for j in range(n):
        neighbors = good[abs(ids[good]-ids[j]) <= 2]
        if len(neighbors):
            head[j] = np.median(observed[neighbors],axis=0)
    keys = {0, n-1}
    for target in range(ids[0]+stride, ids[-1], stride):
        nearby = good[abs(ids[good]-target) <= max(1,stride//4)]
        if len(nearby):
            score = confidence[nearby] - .04 * abs(ids[nearby]-target)
            keys.add(int(nearby[np.argmax(score)]))
    def interpolate(values):
        ix = sorted(keys)
        return np.column_stack([np.interp(ids,ids[ix],col[ix]) for col in values.T])
    def image_errors(positions):
        return np.array([np.linalg.norm(project(c,h) - f["body_anchor_xy"]) for c,h,f in zip(cams,positions,items)])
    # Add a motion key only when it improves at least three consecutive observations.
    while n >= 3:
        linear = interpolate(head)
        gain = image_errors(linear) - image_errors(head)
        eligible = [j for j in good if 0 < j < n-1 and min(abs(j-k) for k in keys)>=2
                    and np.median(gain[j-1:j+2]) > 0.008 * cams[j]["intrinsics"][0][2]*2]
        if not eligible:
            break
        keys.add(max(eligible,key=lambda j: gain[j]))
    if pelvis is not None:
        # Keep the whole vertical proxy over the usable cushion depth. Preserve
        # lateral head motion within ten centimetres of the assigned seat.
        side = np.cross(up, front)
        displacement = head - pelvis
        head = (pelvis + np.outer(displacement @ up, up)
                + np.outer(np.clip(displacement @ front, -.06*units, .06*units), front)
                + np.outer(np.clip(displacement @ side, -.10*units, .10*units), side))
    stable = interpolate(head)
    forward = np.array([f["facing_world"] for f in items])
    if front is not None:
        forward[:] = front
    else:
        # Use a circular median neighborhood before interpolating unwrapped yaw.
        e1 = np.eye(3)[np.argmin(abs(up))]
        e1 -= e1.dot(up)*up
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up,e1)
        yaw = np.unwrap(np.arctan2(forward@e2, forward@e1))
        angles = np.array([np.median(yaw[max(0,i-2):i+3]) for i in range(n)])
        yaw = interpolate(angles[:,None])[:,0]
        forward = np.cos(yaw[:,None])*e1 + np.sin(yaw[:,None])*e2
    for i, f in enumerate(items):
        h = stable[i]
        center = h - .5*convention["height_world"]*up
        f["head_anchor_world"] = h.tolist()
        f["proxy_center_world"] = f["position_world"] = center.tolist()
        f["position_source"] = "omega_world_keyframe_interpolation"
        f["facing_world"] = forward[i].tolist()
        f["animation_keyframe"] = i in keys
        f["head_reprojection_error_px"] = float(np.linalg.norm(project(cams[i],h)-f["body_anchor_xy"]))
        if pelvis is not None:
            f["pelvis_world"] = pelvis.tolist()
            f["posture"] = "sitting"
            f["position_world"] = pelvis.tolist()
            f["position_source"] = "measured_seat_pelvis"
            f["posture_source"] = "scene_geometry_and_camera"
        f["proxy_bbox_3d"] = fixed_proxy_box(center,forward[i],up,convention["height_world"],convention["radius_world"])
    return {"first": int(ids[0]), "last": int(ids[-1]), "keys": [int(ids[k]) for k in sorted(keys)]}


def stabilize(actors, cameras, contacts, clip, fps):
    if __package__:
        from .actor_presence import complete, stitch_tracks
        from .actor_support import SceneSupport, asset_point
    else:
        from actor_presence import complete, stitch_tracks
        from actor_support import SceneSupport, asset_point

    scene = cameras["scene_frame"]
    units = scene["units_per_meter"]
    stride = max(1,round(fps/2))
    catalog = contacts if contacts["scene_sha256"] == scene["source_blend_sha256"] else {}
    support = SceneSupport(catalog,cameras)
    cams = cameras["frames"]
    convention = actors["proxy_convention"]
    tracks = stitch_tracks(actors["tracks"],fps,units,cams)
    for track in sorted(tracks,key=lambda t: -len(t["frames"])):
        items = track["frames"]
        if not items or not track.get("proxy_quality",{}).get("eligible",True):
            continue
        if track.get("body_depth_only"):
            for item in items:
                item.update(support="depth",support_confidence=0.)
            runs = [("depth",items)]
        else:
            runs = support.classify(items,stride)
        for label, span in runs:
            seat = catalog.get("seats",{}).get(label)
            if label == "ground":
                for item in span:
                    support.ground(item)
            fit_segment(span,cams,track,seat,stride,units,support.matrix,convention)
            if seat:
                xy = asset_point(support.matrix,track["seat_contact"]["pelvis_world"])[:2]
                support.claims.append((span[0]["frame_index"],span[-1]["frame_index"],label,xy))
            for f in span:
                f["body_height_world"] = convention["height_world"]
                if seat:
                    f["seat_object"] = seat["object"]
                    delta = np.subtract(f["head_anchor_world"],f["pelvis_world"])
                    f["body_height_world"] = max(.32*units,delta @ convention["up_world"]+.16*units)
            # Important direction changes in a walk need their own animation key.
            for j in range(2,len(span)-2):
                if np.dot(span[j-2]["facing_world"],span[j+2]["facing_world"]) < .82:
                    span[j]["animation_keyframe"] = True
        # Blend posture changes over the surrounding observations in world space.
        for (_,left),(_,right) in zip(runs,runs[1:]):
            window=left[-max(1,stride//3):]+right[:max(1,stride//3)]
            a,b=window[0],window[-1]
            for f in window[1:-1]:
                t=(f["frame_index"]-a["frame_index"])/(b["frame_index"]-a["frame_index"])
                f["head_anchor_world"]=((1-t)*np.array(a["head_anchor_world"])+t*np.array(b["head_anchor_world"])).tolist()
                f["body_height_world"]=(1-t)*a["body_height_world"]+t*b["body_height_world"]
                f["support"]="transition"
                f["animation_keyframe"]=True
        complete(track,cams,fps,units,convention)
        track["support_segments"]=[]
        up=np.asarray(convention["up_world"])
        for f in track["frames"]:
            center=np.asarray(f["head_anchor_world"])-up*convention["height_world"]/2
            f["proxy_center_world"]=center.tolist()
            f["position_world"]=f.get("pelvis_world",center.tolist()) if f.get("support") in catalog.get("seats",{}) else center.tolist()
            f["proxy_bbox_3d"]=fixed_proxy_box(center,f["facing_world"],up,convention["height_world"],convention["radius_world"])
            if track["support_segments"] and track["support_segments"][-1]["support"]==f["support"]:
                track["support_segments"][-1]["last"]=f["frame_index"]
            else:
                track["support_segments"].append({"first":f["frame_index"],"last":f["frame_index"],"support":f["support"]})

    actors["tracks"]=tracks
    actors["animation_method"]="omega_persistent_actors_v3"
    actors["keyframe_stride"]=stride
    actors["presence_policy"]="confirmed_actor_exists_for_whole_clip"

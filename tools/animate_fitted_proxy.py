"""Keyframe a visibility-aware head and torso in the raw camera-track coordinate system."""

from mathutils import Matrix, Vector


def basis_from_forward(forward, up):
    x = Vector(forward)
    x -= up * up.dot(x)
    if x.length < 1e-6:
        x = up.orthogonal()
    x.normalize()
    return Matrix((x, up.cross(x), up)).transposed()


def animate_proxy(
    root,
    track,
    cameras,
    convention,
    alignment,
    torso,
    head,
    band,
    radius,
    height,
    head_radius,
):
    # Reuse the native importer meshes; normalize their coordinates so animated
    # scales are expressed directly in scene units.
    torso.data.transform(
        Matrix.Diagonal((1 / radius, 1 / radius, 2 / (height - 2 * head_radius), 1))
    )
    head.data.transform(Matrix.Scale(1 / head_radius, 4))
    head_offset = Vector((0, 0, height / 2 - head_radius))
    band.data.transform(
        Matrix.Scale(1 / head_radius, 4) @ Matrix.Translation(-head_offset)
    )
    band.parent = head
    band.matrix_parent_inverse = Matrix.Identity(4)
    for mesh in (torso, head, band):
        mesh.rotation_mode = "QUATERNION"
    head.color = torso.color
    root.matrix_world = alignment
    frames = {f["frame_index"]: f for f in track["frames"]}
    previous = {}

    def rotate(obj, orientation, frame):
        q = orientation.to_quaternion()
        if obj.name in previous:
            q.make_compatible(previous[obj.name])
        previous[obj.name] = q.copy()
        obj.rotation_quaternion = q
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    visible = None
    for i, cam in sorted(cameras.items()):
        frame = i + 1
        if (i in frames) != visible:
            visible = i in frames
            for obj in (torso, head, band):
                for key in ("hide_render", "hide_viewport"):
                    setattr(obj, key, not visible)
                    obj.keyframe_insert(data_path=key, frame=frame)
        if not visible:
            continue
        f = frames[i]
        up = Vector(
            convention.get("up_world")
            or [-x for x in cam["world_to_camera_rotation"][1]]
        ).normalized()
        facing = f["proxy_orientation"]["body_facing_world"]
        base = basis_from_forward(facing, up)
        shape = f.get("fitted_proxy")
        if shape:
            center = Vector(shape["center_world"])
            w, h = shape["head_wh_world"]
            pelvis = Vector(shape["pelvis_world"])
            tw = shape["torso_width_world"]
        else:
            h = w = 0.24
            tw = 0.36
            if "head_anchor_world" in f:
                center = Vector(f["head_anchor_world"]) - up * h / 2
            else:
                center = Vector(f["proxy_center_world"]) + up * (
                    convention.get("height_world", 1.70) / 2 - h / 2
                )
            pelvis = center - up * 0.75
        axis = (center - pelvis).normalized()
        if axis.length < 1e-6:
            axis = up
        end = center - axis * h * 0.36
        torso.location = (pelvis + end) * 0.5
        torso.scale = (tw * 0.35, tw * 0.5, max(0.001, (end - pelvis).length / 2))
        head.location = center
        head.scale = (w / 2, w / 2, h / 2)
        rotate(torso, basis_from_forward(facing, axis), frame)
        rotate(head, base, frame)
        applied = f.get("proxy_orientation", {}).get("head_basis_world")
        rotate(
            band,
            base.transposed() @ Matrix(applied) if applied else Matrix.Identity(3),
            frame,
        )
        for mesh in (torso, head):
            mesh.keyframe_insert(data_path="location", frame=frame)
            mesh.keyframe_insert(data_path="scale", frame=frame)

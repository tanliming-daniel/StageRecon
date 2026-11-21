"""Run in Blender background on a base scene. Audit all frames and render previews."""
import argparse
import json
import runpy
import sys
from pathlib import Path

import bpy
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundles",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--percent",type=int,default=25,choices=range(1,101),metavar="1..100")
    ap.add_argument("--only",default="")
    ap.add_argument("--skip-render",action="store_true")
    args=ap.parse_args(sys.argv[sys.argv.index("--")+1:])
    importer=Path(__file__).with_name("import_clip_animation.py").resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    for path in sorted(args.bundles.glob("S*/*/render_job.json")):
        if args.only and args.only not in path.parent.name:continue
        d=path.parent; out=args.output/d.name
        if (out/"audit.json").is_file():continue
        out.mkdir(parents=True,exist_ok=True)
        job=json.loads(path.read_text());actors=json.loads((d/"actor_tracks.json").read_text())
        sys.argv=[str(importer),"--","--camera-track",str(d/"camera_track.json"),"--actor-tracks",str(d/"actor_tracks.json"),
                  "--output-blend",str(out/"preview.blend"),"--no-save-blend","--metadata",str(out/"import.json"),
                  "--fps",str(job["timeline"]["fps"]),*[v for name in job["scene_import"]["hide_collections"] for v in ("--hide-collection",name)]]
        runpy.run_path(str(importer),run_name="__main__")
        meta=json.loads((out/"import.json").read_text());scene=bpy.context.scene
        scene.render.resolution_percentage=args.percent
        scene.render.engine="BLENDER_WORKBENCH"
        scene.display.shading.light="STUDIO";scene.display.shading.color_type="TEXTURE"
        scene.display.shading.show_shadows=False;scene.display.shading.show_cavity=False
        scene.display.render_aa="OFF"
        if hasattr(scene.render.image_settings,"media_type"):
            scene.render.image_settings.media_type="IMAGE"
        scene.render.image_settings.file_format="PNG";scene.render.image_settings.color_mode="RGB"
        scene.render.image_settings.compression=12
        scene.render.filepath=str((out/"frame_").resolve())
        w,h=[round(v*args.percent/100) for v in (scene.render.resolution_x,scene.render.resolution_y)]
        expected={f"ClipActor_{t['track_id']:04d}" for t in actors["tracks"] if t.get("proxy_quality",{}).get("eligible",True)}
        assert expected==set(meta["actor_proxies"])
        nodes={n:bpy.data.objects[n] for n in expected}
        colors={n:tuple(bpy.data.objects[n+'_Body'].data.materials[0].diffuse_color) for n in nodes}
        assert len(set(colors.values()))==len(colors), 'Character colors collide'
        points={}
        for n,root in nodes.items():
            children={o.name:o for o in root.children}
            assert set(children)=={n+"_Body",n+"_Head",n+"_Facing"}
            for label,o in children.items():
                if label.endswith("_Facing"):continue
                coords=np.empty(len(o.data.vertices)*3,np.float32);o.data.vertices.foreach_get("co",coords)
                points[label]=np.column_stack((coords.reshape(-1,3)[::2],np.ones(len(coords.reshape(-1,3)[::2]))))
        outlines=[]; identities={name:obj.as_pointer() for name,obj in nodes.items()}
        scene_objects=set(scene.objects)
        for i in range(scene.frame_start,scene.frame_end+1):
            scene.frame_set(i);deps=bpy.context.evaluated_depsgraph_get();cam=scene.camera.evaluated_get(deps)
            vp=np.asarray(cam.calc_matrix_camera(deps,x=w,y=h,scale_x=scene.render.pixel_aspect_x,scale_y=scene.render.pixel_aspect_y))@np.asarray(cam.matrix_world.inverted())
            record={}
            for name,root in nodes.items():
                assert root.as_pointer()==identities[name] and root in scene_objects
                assert bpy.context.view_layer.objects.get(name)==root
                for o in [root,*root.children]:
                    assert not o.hide_render and not o.hide_get() and o.visible_get()
                    matrix=np.asarray(o.evaluated_get(deps).matrix_world)
                    assert np.all(np.isfinite(matrix)) and np.min(np.linalg.norm(matrix[:3,:3],axis=0))>1e-6
                    assert np.linalg.det(matrix[:3,:3])>1e-10
                    if o.name not in points:continue
                    projected=points[o.name]@(vp@matrix).T
                    projected=projected[projected[:,3]>1e-6]
                    uv=projected[:,:2]/projected[:,3:]
                    uv=(uv*[.5,-.5]+.5)*[w,h]
                    record[o.name]=np.rint(np.clip(uv,-8192,8192)).astype(int).tolist()
            outlines.append(record)
        (out/"projected.json").write_text(json.dumps(outlines,separators=(",",":")))
        if not args.skip_render:
            bpy.ops.render.render(animation=True)
            assert len(list(out.glob("frame_*.png")))==scene.frame_end-scene.frame_start+1
        (out/"audit.json").write_text(json.dumps(dict(frames=len(outlines),actors=len(nodes),
            actor_frames=len(outlines)*len(nodes),colors=colors,stable_ids=sorted(identities),resolution=[w,h],rendered=not args.skip_render)))
        print("PRESENCE VERIFIED",d.name,len(outlines),len(nodes),flush=True)


if __name__=="__main__":
    main()

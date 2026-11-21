"""Render reviewed clip bundles and encode a frame-matched GT/Blender comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from batch_protocol import portable_bundle_is_valid
from omega_review import sha256


def main():
    ap=argparse.ArgumentParser()
    for name in ('clips-root','bundles','case-list','base-blend','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--blender',default='blender')
    ap.add_argument('--jobs',type=int,default=2)
    args=ap.parse_args()
    cases=json.loads(args.case_list.read_text())
    cases=[Path(c if isinstance(c,str) else c['clip']) for c in cases]
    if not cases:raise ValueError('Empty case list')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scene_sha=sha256(args.base_blend)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        work=Path(temp).resolve()
        items=[]
        for i,rel in enumerate(cases):
            src=args.bundles/rel.parts[0]/rel.stem
            if not portable_bundle_is_valid(src):raise ValueError(src)
            job=json.loads((src/'render_job.json').read_text())
            if job['source_scene'].get('sha256')!=scene_sha:raise ValueError(f'Scene mismatch: {src}')
            original=(args.clips_root/rel).resolve()
            if not original.is_file():raise FileNotFoundError(original)
            root=work/str(i);link=root/'input'/rel.parts[0]/rel.stem
            link.parent.mkdir(parents=True);link.symlink_to(src.resolve(),target_is_directory=True)
            fps=Fraction(str(job['timeline']['fps'])).limit_denominator(100000)
            items.append((root,rel,original,job['timeline']['frame_count'],fps))
        rate=items[0][-1]
        if any(item[-1]!=rate for item in items):raise ValueError('Comparison requires a common frame rate')
        def render(item):
            root,rel,*_=item
            with (root/'blender.log').open('w') as log:
                result=subprocess.run([args.blender,'-b',str(args.base_blend.resolve()),'-t','4','--python-exit-code','2','-P',
                    str(Path(__file__).with_name('verify_clip_presence_blender.py').resolve()),'--','--bundles',str(root/'input'),
                    '--output',str(root/'render'),'--percent','40'],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{rel}: '+(root/'blender.log').read_text()[-3000:])
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(render,items))
        target=work/'comparison.mp4'
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','960x300',
            '-framerate',str(rate),'-i','-','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        try:
            for i,(root,rel,original,count,_) in enumerate(items,1):
                decode=subprocess.Popen(['ffmpeg','-v','error','-i',str(original),'-an','-vf','scale=480:270',
                    '-vsync','0','-pix_fmt','bgr24','-f','rawvideo','-'],stdout=subprocess.PIPE)
                try:
                    for j in range(1,count+1):
                        raw=decode.stdout.read(480*270*3)
                        if len(raw)!=480*270*3:raise ValueError(f'Missing GT frame {j}: {rel}')
                        panel=np.zeros((300,960,3),np.uint8);panel[30:,:480]=np.frombuffer(raw,np.uint8).reshape(270,480,3)
                        frame=cv2.imread(str(root/'render'/rel.stem/f'frame_{j:04d}.png'))
                        panel[30:,480:]=cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
                        for x,label in [(8,f'{i:02d}/{len(items)}  {rel.parts[0]}  {rel.stem.split("_")[1] if "_" in rel.stem else rel.stem}  |  GT'),(488,f'Blender  |  {j}/{count}')]:
                            cv2.putText(panel,label,(x,21),0,.54,(238,238,238),1,cv2.LINE_AA)
                        encoder.stdin.write(panel.tobytes())
                    if decode.stdout.read(1) or decode.wait():raise ValueError(f'GT frame count mismatch: {rel}')
                finally:
                    decode.stdout.close();decode.kill();decode.wait()
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('Video encoding failed')
        finally:
            if encoder.poll() is None:encoder.kill();encoder.wait()
        expected=sum(item[3] for item in items)
        actual=int(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames','-of','csv=p=0',str(target)]))
        if actual!=expected:raise ValueError(f'Encoded {actual} frames; expected {expected}')
        os.replace(target,args.output)
        print(f'{args.output}: {len(items)} clips, {expected} frames')


if __name__=='__main__':main()

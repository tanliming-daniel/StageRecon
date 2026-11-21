"""Measure visible facial and torso landmarks with the body-only MHR model."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def run(input_dir, camera_path, output, checkpoint, flow):
    sys.path.insert(0, str(flow / 'third_party/sam-3d-body'))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    skeleton = checkpoint.parent / 'assets/mhr_model.pt'
    joint_names = list(torch.jit.load(str(skeleton), map_location='cpu').joint_names_list)
    head_index = joint_names.index('c_head')
    original = torch.hub.load
    def load_backbone(repo, *args, **kwargs):
        if repo == 'facebookresearch/dinov3':
            repo = str(flow / '.cache/torch/hub/facebookresearch_dinov3_main')
            kwargs['source'] = 'local'
        return original(repo, *args, **kwargs)
    torch.hub.load = load_backbone
    try:
        model, cfg = load_sam_3d_body(str(checkpoint), device='cuda', mhr_path=str(skeleton))
    finally:
        torch.hub.load = original
    estimator = SAM3DBodyEstimator(model, cfg)
    cameras = json.loads(camera_path.read_text())['frames']
    tracks = json.loads((input_dir/'masks/person_tracks.json').read_text())['tracks']
    by_frame = [[] for _ in cameras]
    for track in tracks:
        for item in track['frames']:
            by_frame[item['frame_index']].append((track['track_id'], item))
    rows = []
    for i, entries in enumerate(by_frame):
        candidates = []
        for tid, item in entries:
            path = Path(item['mask'])
            if not path.is_absolute() and not path.exists():
                path = input_dir / path
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(path)
            mask = (mask > 0).astype('uint8')
            x,y,w,h = cv2.boundingRect(mask)
            area = cv2.countNonZero(mask)
            row = dict(frame=i, track=tid, area=area, bbox=[x,y,w,h], image=list(mask.shape))
            rows.append(row)
            # Normalize to image size so a small, complete distant figure can pass.
            if area < max(48, mask.size*.00065) or min(w,h) < min(mask.shape)*.014:
                continue
            candidates.append((row,item,mask))
        if not candidates:
            continue
        boxes = np.array([item['bbox_xyxy'] for _,item,_ in candidates], np.float32)
        masks = np.stack([m for _,_,m in candidates])
        frame = input_dir/'frames'/candidates[0][1]['frame_name']
        intrinsics = torch.tensor(cameras[i]['intrinsics'], dtype=torch.float32)[None]
        predictions = estimator.process_one_image(str(frame), bboxes=boxes, masks=masks,
                                                   cam_int=intrinsics, inference_type='body')
        if len(predictions) != len(candidates):
            raise ValueError(f'Frame {i}: unmatched mask and prediction counts')
        for (row,item,mask), out in zip(candidates,predictions):
            indices = [0,1,2,3,4,5,6,9,10,69]
            uv = np.asarray(out['pred_keypoints_2d'])[indices]
            xyz = np.asarray(out['pred_keypoints_3d'])[indices] + np.asarray(out['pred_cam_t']).reshape(3)
            rot = np.diag([1.,-1.,-1.]) @ np.asarray(out['pred_global_rots'])[head_index]
            if not all(np.isfinite(a).all() for a in (uv,xyz,rot)) or xyz[:,2].min() <= 0:
                continue
            # Signed distance in pixels tests actual foreground and image truncation.
            inside = cv2.distanceTransform(mask,cv2.DIST_L2,3)
            outside = cv2.distanceTransform(1-mask,cv2.DIST_L2,3)
            def support(p):
                x,y = np.rint(p).astype(int)
                if 0<=x<mask.shape[1] and 0<=y<mask.shape[0]:
                    return float(inside[y,x]-outside[y,x])
                return -999.
            probes = list(uv) + [np.mean(uv[[5,6]],axis=0),np.mean(uv[[7,8]],axis=0),
                                  np.mean(uv[[5,6,7,8]],axis=0)]
            # The abdomen and hips are often below frame in a medium shot.
            upper_chest = .75*uv[[5,6]] + .25*uv[[7,8]]
            probes.extend([upper_chest[0],upper_chest[1],upper_chest.mean(axis=0)])
            from actor_shoulders import shoulder_contour, outer_shoulder_height
            row['outer_shoulder_y'] = outer_shoulder_height(mask,uv)
            row['shoulder_contour_y'] = shoulder_contour(mask,uv)
            row.update(uv=uv.tolist(), xyz=xyz.tolist(), head_rotation=rot.tolist(),
                       support=[support(p) for p in probes])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(joint=joint_names[head_index], rows=rows),allow_nan=False))
    print(f'{output}: {len(rows)} mask observations',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('input-dir','camera','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('/root/userdata/300workspace/ckpt/sam-3d-body-dinov3/model.ckpt'))
    p.add_argument('--flow',type=Path,default=Path('/root/userdata/300workspace/flowhmr'))
    a=p.parse_args()
    run(a.input_dir,a.camera,a.output,a.checkpoint,a.flow)

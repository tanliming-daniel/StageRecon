"""Whole-shot background evidence for a stationary camera at 768 px width."""
import cv2
import numpy as np


def background_motion(paths, mask_dir):
    if len(paths) < 3 or mask_dir is None:
        return None

    def load(path):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir / path.with_suffix(".png").name), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            return None, None
        size = (768, round(gray.shape[0]*768/gray.shape[1]))
        return cv2.resize(gray,size,interpolation=cv2.INTER_AREA), cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)

    first, mask = load(paths[0])
    if first is None:
        return None
    points = cv2.goodFeaturesToTrack(first,600,.008,12,mask=(mask==0).astype(np.uint8))
    if points is None or len(points)<35:
        return None
    points = points.reshape(-1,2)
    h,w = first.shape
    worst = 0.
    for path in paths[1:]:
        img, mask = load(path)
        if img is None:
            return None
        xy, ok, _ = cv2.calcOpticalFlowPyrLK(first,img,points,None)
        back, rev, _ = cv2.calcOpticalFlowPyrLK(img,first,xy,None)
        x,y = np.rint(xy).astype(int).T
        keep = (ok.ravel().astype(bool) & rev.ravel().astype(bool)
                & (np.linalg.norm(back-points,axis=1)<.7) & (x>=0)&(x<w)&(y>=0)&(y<h)
                & (mask[y.clip(0,h-1),x.clip(0,w-1)]==0))
        if keep.sum()<35 or cv2.contourArea(cv2.convexHull(points[keep])) < .12*w*h:
            return None
        worst=max(worst,float(np.quantile(np.linalg.norm(xy[keep]-points[keep],axis=1),.8)))
        if worst>3.5:
            return None
    return dict(max_background_p80_pixels_at_768=worst,frames_checked=len(paths))

"""Score support surfaces using final clip cameras and scene-space depth."""
import numpy as np


def asset_point(matrix, p):
    return np.asarray(p) @ matrix[:3,:3].T + matrix[:3,3]


def floor_at(scene, xy):
    for obj in scene.get("floors", []):
        x,y=np.floor((xy-obj["origin"])/obj["step"]).astype(int)
        bounds=obj["rows"].get(str(x))
        if bounds and bounds[0]<=y<=bounds[1]:return obj["z"]
    return None


class SceneSupport:
    def __init__(self, scene, camera):
        self.scene=scene
        self.cam=camera["frames"]
        self.matrix=np.asarray(camera["scene_frame"]["world_to_blender"])
        self.inverse=np.linalg.inv(self.matrix)
        self.units=camera["scene_frame"]["units_per_meter"]
        self.points=[]
        for obj in scene.get("occluders",[]):
            p=np.array(obj["top"],float)
            if len(p):
                p[:,:2]=(p[:,:2]+.5)*obj["step"]+obj["origin"]
                self.points.extend(asset_point(self.inverse,p))
        self.points=np.asarray(self.points)
        self.depth_cache={}
        self.claims=[]
        self.strength={}
        for k in camera.get("da3",{}).get("actor_depth_alignment",{}).get("keyframes",[]):
            self.strength[k["frame_index"]]=abs(k.get("near_person_static_median_log_residual") or 0)

    def static_depth(self, i, uv):
        if not self.points.size:return np.inf
        cam=self.cam[i];k=np.asarray(cam["intrinsics"])
        if i not in self.depth_cache:
            p=self.points@np.asarray(cam["world_to_camera_rotation"]).T+cam["world_to_camera_translation"]
            p=p[p[:,2]>1e-4];pixels=(p@k.T)[:,:2]/p[:,2:]
            size=np.maximum(1,np.ceil((k[:2,2]*2+1)/10).astype(int))
            xy=np.floor(pixels/10).astype(int)
            ok=np.all((xy>=0)&(xy<size),axis=1)
            buf=np.full(tuple(size[::-1]),np.inf)
            np.minimum.at(buf,(xy[ok,1],xy[ok,0]),p[ok,2])
            self.depth_cache[i]=buf
        buf=self.depth_cache[i]
        x,y=np.floor(np.asarray(uv)/10).astype(int)
        return buf[y,x] if 0<=y<buf.shape[0] and 0<=x<buf.shape[1] else np.inf

    def evidence(self,f):
        i=f["frame_index"];cam=self.cam[i];k=np.array(cam["intrinsics"]);r=np.array(cam["world_to_camera_rotation"])
        origin=asset_point(self.matrix,cam["camera_center"])
        def ray(uv):
            return self.matrix[:3,:3] @ r.T @ np.linalg.solve(k,[*uv,1.])
        direction=ray(f["body_anchor_xy"])
        surf=f["depth"]["visible_surface_world"]
        height=1.7
        box=f.get("bbox_xyxy")
        floor=None; foot=None; ground_cost=2.
        if box is not None and box[3]<2*k[1,2]-3:
            footray=ray([(box[0]+box[2])/2,box[3]])
            for level in {s["z"] for s in self.scene.get("floors",[])}:
                if abs(footray[2])<1e-6:continue
                z=(level-origin[2])/footray[2]
                pos=origin+z*footray
                if z<=0 or floor_at(self.scene,pos[:2])!=level:continue
                h=origin+z*direction
                cost=abs(h[2]-level-height)+.4*abs((z-surf)/self.units-.15)
                if cost<ground_cost:floor,foot,ground_cost=level,pos,cost
        # Compare projected depth at the visible head to static furniture.
        zstatic=self.static_depth(i,f["body_anchor_xy"])
        offsets=np.linspace(-1.15,.65,37)
        z=surf+self.units*offsets
        heads=origin+z[:,None]*direction
        weight=.12
        if self.strength:
            residual=self.strength[min(self.strength,key=lambda j:abs(j-i))]
            weight/=1+5*residual
        occlusion=np.maximum(0,(z-zstatic)/self.units-.1)
        raw=asset_point(self.matrix,f["head_anchor_world"])
        zfloor=floor_at(self.scene,raw[:2])
        raw_height=raw[2]-zfloor if zfloor is not None else None
        if raw_height is not None and 1.34 <= raw_height <= 2.06:
            # A cropped lower body can still provide a compatible standing head ray.
            ground_cost = min(ground_cost, .16+.75*abs(raw_height-height))
        costs={"ground":ground_cost,"depth":.39}
        for name,seat in self.scene.get("seats",{}).items():
            a,b=np.asarray(seat["left"][:2]),np.asarray(seat["right"][:2]);axis=b-a
            t=np.clip(((heads[:,:2]-a)@axis)/max(axis@axis,1e-8),0,1)
            xy=a+t[:,None]*axis
            dh=heads[:,2]-seat["height"]
            cost=np.linalg.norm(heads[:,:2]-xy,axis=1)+weight*abs(offsets-.12)+np.maximum(0,.46-dh)*2+np.maximum(0,dh-1.12)*2+.025*abs(dh-.8)+.7*occlusion
            for start,end,s,where in self.claims:
                if start<=i<=end and s==name:
                    cost+=2*np.maximum(0,.31-np.linalg.norm(xy-where,axis=1))
            if raw_height is not None and raw_height>1.45 and ground_cost<.34:
                cost+=.28
            costs[name]=float(np.min(np.where(z>0,cost,10)))
        return costs,foot

    def classify(self, items, stride):
        chunks=[]
        for f in items:
            if not chunks or f["frame_index"]-chunks[-1][0]["frame_index"]>=stride:
                chunks.append([])
            chunks[-1].append(f)
        labels=["depth","ground",*self.scene.get("seats",{})]
        evidence=[]
        for chunk in chunks:
            samples=chunk[::max(1,len(chunk)//4)]
            vals=[self.evidence(f)[0] for f in samples]
            e=np.array([np.median([v[n] for v in vals]) for n in labels])
            if len(samples)>1:
                h=np.array([asset_point(self.matrix,f["head_anchor_world"]) for f in samples])
                e[2:]+=max(0,np.linalg.norm(h[-1,:2]-h[0,:2])-.16)
            evidence.append(e)
        previous=evidence[0];history=[]
        penalty=.19*(1-np.eye(len(labels)))
        for e in evidence[1:]:
            costs=previous[:,None]+penalty
            history.append(np.argmin(costs,axis=0))
            previous=np.min(costs,axis=0)+e
        states=[int(np.argmin(previous))]
        for prev in reversed(history):states.append(int(prev[states[-1]]))
        runs=[]
        for chunk,state,scores in zip(chunks,reversed(states),evidence):
            name=labels[state]; alternatives=np.delete(scores,state)
            conf=float(np.clip(.5+(min(alternatives)-scores[state]),0,1))
            for f in chunk:f["support"]=name;f["support_confidence"]=conf
            if runs and runs[-1][0]==name:runs[-1][1].extend(chunk)
            else:runs.append((name,list(chunk)))
        return runs

    def ground(self,f):
        _,foot=self.evidence(f)
        head=asset_point(self.matrix,f["head_anchor_world"])
        if foot is None:
            level=floor_at(self.scene,head[:2])
            if level is None:return
            cam=self.cam[f["frame_index"]]
            origin=asset_point(self.matrix,cam["camera_center"])
            ray=head-origin
            if abs(ray[2])>.04:
                target=origin+ray*((level+1.7-origin[2])/ray[2])
                if np.linalg.norm(target-head)<=.45:
                    f["head_anchor_world"]=asset_point(self.inverse,target).tolist()
            return
        proposed=foot+np.array([0,0,1.7])
        if np.linalg.norm(proposed-head)<1.0:
            f["head_anchor_world"]=asset_point(self.inverse,proposed).tolist()

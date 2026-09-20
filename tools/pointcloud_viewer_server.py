#!/usr/bin/env python3
"""Local WebGL viewer for semantic point clouds and proposal highlights."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from plyfile import PlyData


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
DEFAULT_PROPOSAL_DIR = REPO_ROOT / "outputs/Friends-monica_apartment/semantic_proposals_instances"
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))
from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402

_CACHE: dict[str, dict] = {}
_COLMAP_IMAGE_CACHE: dict[str, dict] = {}

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>StageRecon Point Cloud Viewer</title>
<style>
:root { color-scheme: dark; --bg:#111318; --panel:#1d222a; --line:#333b47; --text:#edf2f7; --muted:#9aa4b2; --accent:#52b788; --hot:#ffde59; }
* { box-sizing:border-box; }
body { margin:0; height:100vh; overflow:hidden; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.app { display:grid; grid-template-columns:320px 1fr; height:100vh; }
aside { background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; min-width:0; }
.top { padding:12px; border-bottom:1px solid var(--line); display:grid; gap:10px; }
label { font-size:12px; color:var(--muted); display:block; margin-bottom:5px; }
select, input, button { width:100%; border:1px solid var(--line); border-radius:6px; background:#11151b; color:var(--text); padding:8px; font:inherit; }
button { cursor:pointer; background:#2a3039; }
button:hover { border-color:#5d6979; }
button.primary { background:var(--accent); color:#07130d; border-color:var(--accent); font-weight:700; }
.stats { font-size:12px; color:var(--muted); line-height:1.5; }
.legend { padding:12px; border-top:1px solid var(--line); font-size:12px; color:var(--muted); line-height:1.55; }
canvas { width:100%; height:100%; display:block; background:#090b0f; }
main { position:relative; min-width:0; }
.hud { position:absolute; left:12px; top:12px; padding:8px 10px; background:rgba(10,12,16,.72); border:1px solid rgba(255,255,255,.08); border-radius:6px; font-size:12px; color:#cbd5e1; pointer-events:none; }
</style>
</head>
<body>
<div class="app">
<aside>
  <div class="top">
    <div><label>Proposal</label><select id="proposal"></select></div>
    <div><label>Background sample</label><input id="sample" type="number" min="10000" max="217331" step="10000" value="120000" /></div>
    <button class="primary" id="loadBtn">Reload Cloud</button>
    <button id="resetBtn">Reset Camera</button>
    <div class="stats" id="stats">Loading metadata...</div>
  </div>
  <div class="legend">
    Mouse drag rotates. Wheel zooms. Shift+drag pans.<br><br>
    Full point cloud is rendered as small colored points. A selected proposal or image bbox ∩ segmentation region is drawn as larger yellow points with a pulse.
  </div>
</aside>
<main><canvas id="gl"></canvas><div class="hud" id="hud">not loaded</div></main>
</div>
<script>
const el = id => document.getElementById(id);
const canvas = el('gl');
let gl, program, buffers = {}, meta = null;
let camera = { rx: -0.55, ry: 0.35, tx: 0, ty: 0, dist: 12, center:[0,0,0], radius: 8 };
let drag = null;
let cloudLoaded = false;
const selectionQuery = new URLSearchParams(location.search);
const imageSelection = {
  image: selectionQuery.get('image'),
  bbox: selectionQuery.get('bbox'),
  category: selectionQuery.get('category')
};
imageSelection.active = Boolean(imageSelection.image && imageSelection.bbox && imageSelection.category);

const vs = `
attribute vec3 aPosition;
attribute vec3 aColor;
attribute float aHighlight;
uniform mat4 uMvp;
uniform float uTime;
uniform float uPointSize;
varying vec3 vColor;
varying float vHighlight;
void main(){
  gl_Position = uMvp * vec4(aPosition, 1.0);
  vHighlight = aHighlight;
  float pulse = 1.0 + 0.65 * (0.5 + 0.5 * sin(uTime * 6.0));
  gl_PointSize = mix(uPointSize, uPointSize * 4.5 * pulse, aHighlight);
  vColor = mix(aColor, vec3(1.0, 0.86, 0.16), aHighlight);
}`;
const fs = `
precision mediump float;
varying vec3 vColor;
varying float vHighlight;
void main(){
  vec2 p = gl_PointCoord - vec2(0.5);
  float d = dot(p,p);
  if(d > 0.25) discard;
  float alpha = mix(0.72, 1.0, vHighlight);
  gl_FragColor = vec4(vColor, alpha);
}`;

function mat4Identity(){ return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
function mat4Mul(a,b){ const o=new Array(16).fill(0); for(let r=0;r<4;r++) for(let c=0;c<4;c++) for(let k=0;k<4;k++) o[c*4+r]+=a[k*4+r]*b[c*4+k]; return o; }
function perspective(fovy, aspect, near, far){ const f=1/Math.tan(fovy/2), nf=1/(near-far); return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]; }
function translate(x,y,z){ return [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]; }
function rotateX(a){ const c=Math.cos(a),s=Math.sin(a); return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]; }
function rotateY(a){ const c=Math.cos(a),s=Math.sin(a); return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]; }
function compile(type, src){ const sh=gl.createShader(type); gl.shaderSource(sh,src); gl.compileShader(sh); if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; }
function initGl(){ gl=canvas.getContext('webgl',{antialias:true}); if(!gl) throw new Error('WebGL unavailable'); program=gl.createProgram(); gl.attachShader(program,compile(gl.VERTEX_SHADER,vs)); gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fs)); gl.linkProgram(program); if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program)); gl.useProgram(program); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.clearColor(0.035,0.043,0.06,1); }
function resize(){ const dpr=window.devicePixelRatio||1; const w=Math.floor(canvas.clientWidth*dpr), h=Math.floor(canvas.clientHeight*dpr); if(canvas.width!==w||canvas.height!==h){ canvas.width=w; canvas.height=h; gl.viewport(0,0,w,h); } }
function setBuffer(name, data, size, usage=gl.STATIC_DRAW){ let b=buffers[name]; if(!b){ b=gl.createBuffer(); buffers[name]=b; } gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,data,usage); const loc=gl.getAttribLocation(program,name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0); }
async function loadMeta(){ meta=await (await fetch('/api/state')).json(); el('proposal').innerHTML=meta.proposals.map(p=>`<option value="${p.id}">${p.id} · ${p.category} · ${p.point_count} pts</option>`).join(''); el('stats').textContent=`${meta.point_count} points, ${meta.proposals.length} proposals`; }
async function readBinaryJson(url){ const res=await fetch(url); const headerLen=Number(res.headers.get('X-Header-Length')); const buf=await res.arrayBuffer(); const header=JSON.parse(new TextDecoder().decode(buf.slice(0,headerLen))); return {header, data:buf.slice(headerLen)}; }
async function loadCloud(){ const sample=Number(el('sample').value||120000); el('hud').textContent='loading point cloud...'; const {header,data}=await readBinaryJson(`/api/cloud?sample=${sample}`); const n=header.count; const floats=new Float32Array(data); const pos=floats.slice(0,n*3); const col=floats.slice(n*3,n*6); setBuffer('aPosition',pos,3); setBuffer('aColor',col,3); window.currentCount=n; camera.center=header.center; camera.radius=header.radius; camera.dist=header.radius*2.2; camera.tx=0; camera.ty=0; cloudLoaded=true; await loadHighlight(); }
async function loadSelection(){
  const sample=Number(el('sample').value||120000);
  const query=new URLSearchParams({image:imageSelection.image,bbox:imageSelection.bbox,category:imageSelection.category,sample:String(sample)});
  const {header,data}=await readBinaryJson(`/api/selection?${query.toString()}`);
  const hi=new Float32Array(data);
  setBuffer('aHighlight',hi,1,gl.DYNAMIC_DRAW);
  const centroid=header.selection_centroid ? header.selection_centroid.map(v=>Number(v).toFixed(2)).join(', ') : 'n/a';
  el('hud').textContent=`${header.category} · ${header.selection_point_count} selected 3D pts · ${header.highlight_count} drawn · center (${centroid})`;
}
async function loadHighlight(){
  if(!cloudLoaded) return loadCloud();
  if(imageSelection.active) return loadSelection();
  const proposal=el('proposal').value;
  const sample=Number(el('sample').value||120000);
  const {header,data}=await readBinaryJson(`/api/highlight?proposal=${encodeURIComponent(proposal)}&sample=${sample}`);
  const hi=new Float32Array(data);
  setBuffer('aHighlight',hi,1,gl.DYNAMIC_DRAW);
  el('hud').textContent=`${proposal}: ${header.highlight_count} highlighted / ${window.currentCount} drawn`;
}
function mvp(){ const aspect=canvas.width/Math.max(1,canvas.height); let view=mat4Identity(); view=mat4Mul(view,translate(camera.tx,camera.ty,-camera.dist)); view=mat4Mul(view,rotateX(camera.rx)); view=mat4Mul(view,rotateY(camera.ry)); view=mat4Mul(view,translate(-camera.center[0],-camera.center[1],-camera.center[2])); return mat4Mul(perspective(Math.PI/4,aspect,0.01,1000),view); }
function render(t=0){ resize(); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.useProgram(program); gl.uniformMatrix4fv(gl.getUniformLocation(program,'uMvp'),false,new Float32Array(mvp())); gl.uniform1f(gl.getUniformLocation(program,'uTime'),t/1000); gl.uniform1f(gl.getUniformLocation(program,'uPointSize'),Math.max(1.1, window.devicePixelRatio||1)); if(window.currentCount) gl.drawArrays(gl.POINTS,0,window.currentCount); requestAnimationFrame(render); }
canvas.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,shift:e.shiftKey,rx:camera.rx,ry:camera.ry,tx:camera.tx,ty:camera.ty};});
window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{ if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; if(drag.shift){ camera.tx=drag.tx+dx/canvas.clientWidth*camera.radius*2; camera.ty=drag.ty-dy/canvas.clientHeight*camera.radius*2; } else { camera.ry=drag.ry+dx*0.006; camera.rx=drag.rx+dy*0.006; }});
canvas.addEventListener('wheel',e=>{ e.preventDefault(); camera.dist*=Math.exp(e.deltaY*0.001); camera.dist=Math.max(0.1,camera.dist); },{passive:false});
el('loadBtn').onclick=loadCloud; el('proposal').onchange=loadHighlight; el('resetBtn').onclick=()=>{camera.rx=-0.55;camera.ry=0.35;camera.tx=0;camera.ty=0;camera.dist=camera.radius*2.2;};
initGl(); loadMeta().then(loadCloud); requestAnimationFrame(render);
</script>
</body>
</html>
"""


def load_cloud(proposal_dir: Path) -> dict:
    key = str(proposal_dir.resolve())
    if key in _CACHE:
        return _CACHE[key]
    ply = PlyData.read(str(proposal_dir / "semantic_pointcloud.ply"))
    v = ply["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32) / 255.0
    point_ids = np.asarray(v["point3D_id"], dtype=np.int64)
    manifest = json.loads((proposal_dir / "proposals.json").read_text())
    proposals = manifest.get("proposals", [])
    center = xyz.mean(axis=0).astype(np.float32)
    radius = float(np.linalg.norm(xyz - center, axis=1).max())
    data = {"xyz": xyz, "rgb": rgb, "point_ids": point_ids, "proposals": proposals, "category_labels": manifest.get("category_labels", {}), "center": center, "radius": radius}
    _CACHE[key] = data
    return data


def proposal_ids(proposal_dir: Path, proposal: dict) -> set[int]:
    rel = Path(proposal["point3d_ids"])
    path = rel if rel.is_absolute() else proposal_dir / rel
    return set(int(v) for v in np.load(path).tolist())


def all_proposal_ids(proposal_dir: Path, proposals: list[dict]) -> set[int]:
    ids: set[int] = set()
    for proposal in proposals:
        ids.update(proposal_ids(proposal_dir, proposal))
    return ids


def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def colmap_images(environment_dir: Path):
    key = str(environment_dir.resolve())
    if key not in _COLMAP_IMAGE_CACHE:
        _, images, _ = read_model(str(environment_dir / "colmap"), ext="")
        _COLMAP_IMAGE_CACHE[key] = images
    return _COLMAP_IMAGE_CACHE[key]


def selection_point_ids(environment_dir: Path, cloud: dict, image_name: str, bbox: list[int], category: str) -> tuple[set[int], dict]:
    category_norm = normalize_name(category)
    labels = cloud["category_labels"].get(category_norm)
    if not labels:
        available = ", ".join(sorted(cloud["category_labels"]))
        raise ValueError(f"Unknown segmentation category '{category}'. Available: {available}")
    image = next((item for item in colmap_images(environment_dir).values() if item.name == image_name), None)
    if image is None:
        raise ValueError(f"Image not registered in COLMAP: {image_name}")

    stem = Path(image_name).stem
    stuff_path = environment_dir / "segmentations" / "stuff" / f"{stem}.png"
    thing_path = environment_dir / "segmentations" / "thing" / f"{stem}.png"
    if not stuff_path.exists() or not thing_path.exists():
        raise FileNotFoundError(f"Missing segmentation masks for {image_name}")
    stuff = np.asarray(Image.open(stuff_path))
    thing = np.asarray(Image.open(thing_path))
    if stuff.shape != thing.shape:
        raise ValueError(f"stuff/thing mask shape mismatch for {image_name}")
    semantic = np.zeros(stuff.shape, dtype=bool)
    for item in labels:
        source = str(item["source"])
        label_id = int(item["label_id"])
        if source == "thing":
            semantic |= thing == label_id
        elif source == "stuff":
            semantic |= (thing == 0) & (stuff == label_id)

    height, width = semantic.shape
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width, x + max(0, w)))
    y2 = max(0, min(height, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}")

    xy = np.asarray(image.xys, dtype=np.float64)
    point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
    px = np.rint(xy[:, 0]).astype(np.int64)
    py = np.rint(xy[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    in_bbox = in_image & (point_ids >= 0) & (px >= x1) & (px < x2) & (py >= y1) & (py < y2)
    semantic_track = np.zeros(len(point_ids), dtype=bool)
    semantic_track[in_image] = semantic[py[in_image], px[in_image]]
    selected = in_bbox & semantic_track
    ids = {int(item) for item in point_ids[selected].tolist()}
    return ids, {
        "source_image": image_name,
        "category": category_norm,
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "raw_bbox_track_count": int(np.count_nonzero(in_bbox)),
        "semantic_track_count": int(np.count_nonzero(selected)),
        "semantic_pixel_area_in_bbox": int(np.count_nonzero(semantic[y1:y2, x1:x2])),
    }


def draw_indices(
    proposal_dir: Path,
    cloud: dict,
    sample: int,
    keep_ids: set[int] | None = None,
) -> np.ndarray:
    proposal_union = all_proposal_ids(proposal_dir, cloud["proposals"])
    protected_ids = proposal_union | (keep_ids or set())
    protected = np.isin(cloud["point_ids"], np.fromiter(protected_ids, dtype=np.int64))
    protected_idx = np.flatnonzero(protected)
    bg_idx = np.flatnonzero(~protected)
    if len(bg_idx) > sample:
        rng = np.random.default_rng(7)
        bg_idx = rng.choice(bg_idx, size=sample, replace=False)
    return np.concatenate([bg_idx, protected_idx]).astype(np.int64)


def binary_response_arrays(header: dict, arrays: list[np.ndarray]) -> tuple[bytes, int]:
    header_bytes = json.dumps(header).encode()
    body = b"".join(np.asarray(array, dtype=np.float32).tobytes() for array in arrays)
    return header_bytes + body, len(header_bytes)


class ViewerHandler(BaseHTTPRequestHandler):
    proposal_dir: Path
    environment_dir: Path

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[pointcloud-viewer] " + fmt % args + "\n")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(data, indent=2).encode(), "application/json", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                return
            cloud = load_cloud(self.proposal_dir)
            if parsed.path == "/api/state":
                self.send_json({
                    "point_count": int(len(cloud["xyz"])),
                    "proposals": [
                        {"id": p["id"], "category": p["category"], "point_count": p["point_count"]}
                        for p in cloud["proposals"]
                    ],
                })
                return
            if parsed.path == "/api/cloud":
                q = parse_qs(parsed.query)
                sample = int(q.get("sample", ["120000"])[0])
                idx = draw_indices(self.proposal_dir, cloud, sample)
                xyz = cloud["xyz"][idx]
                rgb = cloud["rgb"][idx]
                payload, header_len = binary_response_arrays({
                    "count": int(len(idx)),
                    "center": cloud["center"].tolist(),
                    "radius": cloud["radius"],
                }, [xyz.reshape(-1), rgb.reshape(-1)])
                self.send_bytes(payload, "application/octet-stream", headers={"X-Header-Length": str(header_len)})
                return
            if parsed.path == "/api/selection":
                q = parse_qs(parsed.query)
                image_name = q.get("image", [""])[0]
                category = q.get("category", [""])[0]
                bbox_text = q.get("bbox", [""])[0]
                try:
                    bbox = [int(float(value)) for value in bbox_text.split(",")]
                except ValueError as exc:
                    raise ValueError(f"Invalid bbox query: {bbox_text}") from exc
                if len(bbox) != 4:
                    raise ValueError(f"Expected bbox=x,y,w,h, got: {bbox_text}")
                ids, selection = selection_point_ids(
                    self.environment_dir, cloud, image_name, bbox, category
                )
                sample = int(q.get("sample", ["120000"])[0])
                idx = draw_indices(self.proposal_dir, cloud, sample, keep_ids=ids)
                id_array = np.fromiter(ids, dtype=np.int64) if ids else np.empty(0, dtype=np.int64)
                selected_drawn = np.isin(cloud["point_ids"][idx], id_array)
                all_selected = np.isin(cloud["point_ids"], id_array)
                selected_xyz = cloud["xyz"][all_selected]
                if len(selected_xyz):
                    selection["selection_centroid"] = selected_xyz.mean(axis=0).astype(float).tolist()
                    selection["selection_bbox_min"] = selected_xyz.min(axis=0).astype(float).tolist()
                    selection["selection_bbox_max"] = selected_xyz.max(axis=0).astype(float).tolist()
                else:
                    selection["selection_centroid"] = None
                    selection["selection_bbox_min"] = None
                    selection["selection_bbox_max"] = None
                selection["selection_point_count"] = len(ids)
                selection["highlight_count"] = int(np.count_nonzero(selected_drawn))
                selection["drawn_count"] = int(len(idx))
                payload, header_len = binary_response_arrays(selection, [selected_drawn.astype(np.float32)])
                self.send_bytes(payload, "application/octet-stream", headers={"X-Header-Length": str(header_len)})
                return
            if parsed.path == "/api/highlight":
                q = parse_qs(parsed.query)
                proposal_id = q.get("proposal", [cloud["proposals"][0]["id"]])[0]
                sample = int(q.get("sample", ["120000"])[0])
                proposal = next((p for p in cloud["proposals"] if p["id"] == proposal_id), None)
                if proposal is None:
                    self.send_json({"error": f"unknown proposal: {proposal_id}"}, 404)
                    return
                idx = draw_indices(self.proposal_dir, cloud, sample)
                ids = proposal_ids(self.proposal_dir, proposal)
                hi = np.isin(cloud["point_ids"][idx], np.fromiter(ids, dtype=np.int64)).astype(np.float32)
                payload, header_len = binary_response_arrays({
                    "count": int(len(idx)),
                    "highlight_count": int(np.count_nonzero(hi)),
                    "proposal": proposal_id,
                }, [hi])
                self.send_bytes(payload, "application/octet-stream", headers={"X-Header-Length": str(header_len)})
                return
            self.send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, 500)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a WebGL point cloud proposal viewer.")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8017)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    ViewerHandler.environment_dir = args.environment_dir.resolve()
    ViewerHandler.proposal_dir = args.proposal_dir.resolve()
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Point cloud viewer: http://{args.host}:{args.port}")
    print(f"Proposal dir: {ViewerHandler.proposal_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()

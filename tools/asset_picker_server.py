#!/usr/bin/env python3
"""Local web UI for manually selecting scene-asset reference images."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = REPO_ROOT / "sitcoms3D/data/sparse_reconstruction_and_nerf_data/Friends-monica_apartment"
DEFAULT_PROPOSALS = REPO_ROOT / "outputs/Friends-monica_apartment/semantic_proposals_instances"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/Friends-monica_apartment/manual_asset_marks"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SITCOMS3D_ROOT = REPO_ROOT / "sitcoms3D"
if str(SITCOMS3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SITCOMS3D_ROOT))
from sitcoms3D.utils.colmap_utils import read_model  # noqa: E402

_COLMAP_IMAGE_CACHE = {}
_PROPOSAL_ID_CACHE = {}

def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())

def colmap_images(environment_dir: Path):
    key = str(environment_dir.resolve())
    if key not in _COLMAP_IMAGE_CACHE:
        _, images, _ = read_model(str(environment_dir / "colmap"), ext="")
        _COLMAP_IMAGE_CACHE[key] = images
    return _COLMAP_IMAGE_CACHE[key]

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>StageRecon Asset Picker</title>
<style>
:root { color-scheme: dark; --bg:#15171b; --panel:#20242b; --muted:#9aa3b2; --line:#343a45; --text:#eef2f7; --accent:#52b788; --warn:#f2b84b; --danger:#ee6c6c; }
* { box-sizing:border-box; }
body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); height:100vh; overflow:hidden; }
button, input, select, textarea { font:inherit; color:var(--text); }
button { border:1px solid var(--line); background:#2b3039; padding:7px 10px; border-radius:6px; cursor:pointer; }
button:hover { border-color:#566173; }
button.primary { background:var(--accent); border-color:var(--accent); color:#07130d; font-weight:700; }
button.danger { color:#ffd6d6; border-color:#7f3e45; background:#3a2429; }
input, select, textarea { width:100%; border:1px solid var(--line); background:#121419; border-radius:6px; padding:7px 8px; }
textarea { resize:vertical; min-height:64px; }
label { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }
.app { display:grid; grid-template-columns:300px minmax(520px,1fr) 370px; height:100vh; }
.sidebar, .inspector { background:var(--panel); border-right:1px solid var(--line); min-width:0; overflow:hidden; display:flex; flex-direction:column; }
.inspector { border-right:0; border-left:1px solid var(--line); }
.top { padding:12px; border-bottom:1px solid var(--line); display:grid; gap:8px; }
.stats { font-size:12px; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; }
.thumbGrid { overflow:auto; padding:8px; display:grid; grid-template-columns:1fr 1fr; gap:8px; align-content:start; }
.thumb { border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#111; cursor:pointer; min-height:118px; display:grid; grid-template-rows:94px auto; }
.thumb.active { outline:2px solid var(--accent); border-color:var(--accent); }
.thumb img { width:100%; height:94px; object-fit:cover; display:block; }
.thumb span { padding:5px 6px; font-size:11px; color:#d7dde8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.stage { display:flex; flex-direction:column; min-width:0; }
.toolbar { height:52px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:8px; padding:0 12px; background:#181b21; }
.toolbar .name { font-size:13px; color:#c6ceda; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.canvasWrap { flex:1; min-height:0; overflow:auto; display:grid; place-items:center; padding:16px; }
.canvasBox { position:relative; border:1px solid var(--line); background:#0b0d10; box-shadow:0 10px 35px rgba(0,0,0,.35); }
canvas { display:block; max-width:100%; height:auto; cursor:crosshair; }
.form { padding:12px; display:grid; gap:10px; overflow:auto; }
.row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.assetList { border-top:1px solid var(--line); padding:10px 12px; overflow:auto; min-height:180px; }
.assetItem { border:1px solid var(--line); border-radius:6px; padding:8px; margin-bottom:8px; cursor:pointer; background:#191d23; }
.assetItem.active { border-color:var(--accent); }
.assetItem strong { display:block; font-size:13px; margin-bottom:3px; }
.assetItem small { color:var(--muted); display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.hint { color:var(--muted); font-size:12px; line-height:1.45; }
.badge { display:inline-flex; align-items:center; height:20px; padding:0 7px; border-radius:10px; background:#2a3038; color:#cbd3df; font-size:11px; }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="top">
      <input id="search" placeholder="Search frames" />
      <div class="stats"><span id="imageCount"></span><span id="saveState">not saved</span></div>
    </div>
    <div id="thumbGrid" class="thumbGrid"></div>
  </aside>
  <main class="stage">
    <div class="toolbar">
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <button id="fitBtn">Fit</button>
      <button id="clearBoxBtn">Clear Box</button>
      <span class="badge" id="bboxBadge">no bbox</span>
      <span class="name" id="currentName"></span>
    </div>
    <div class="canvasWrap" id="canvasWrap"><div class="canvasBox"><canvas id="canvas"></canvas></div></div>
  </main>
  <aside class="inspector">
    <div class="form">
      <div class="row">
        <div><label>Asset ID</label><input id="assetId" placeholder="couch_monica_main" /></div>
        <div><label>Segmentation category</label><select id="category"></select></div>
      </div>
      <div><label>Bind 3D Proposal</label><select id="proposal"></select></div>
      <div class="row">
        <div><label>Occlusion</label><select id="occlusion"><option value="low">low</option><option value="medium">medium</option><option value="high">high</option></select></div>
        <div><label>Priority</label><select id="priority"><option value="use">use</option><option value="maybe">maybe</option><option value="reject">reject</option></select></div>
      </div>
      <div><label>Image-to-3D Prompt</label><textarea id="prompt" placeholder="A clean isolated 3D asset of the selected couch..."></textarea></div>
      <div><label>Notes</label><textarea id="notes" placeholder="manual notes, visible side, occluders, desired cleanup"></textarea></div>
      <div class="row">
        <button class="primary" id="saveAssetBtn">Save Asset</button>
        <button id="downloadCropBtn">Download Crop</button>
      </div>
      <div class="row">
        <button id="match3dBtn">Match 3D</button>
        <button id="show3dBtn">Show 3D Selection</button>
      </div>
      <button id="show3dBboxBtn">Show 3D BBox</button>
      <div class="hint" id="matchResult">No 3D match yet.</div>
      <div class="row">
        <button id="newAssetBtn">New</button>
        <button class="danger" id="deleteAssetBtn">Delete</button>
      </div>
      <div class="hint">Choose the semantic segmentation category, draw a rectangle around the asset, and save. The builder later keeps only COLMAP tracks inside bbox ∩ segmentation.</div>
    </div>
    <div id="assetList" class="assetList"></div>
  </aside>
</div>
<script>
const state = { images: [], proposals: [], categoryLabels: {}, annotations: { assets: [] }, index: 0, image: null, bbox: null, scale: 1, drawing: null, selectedAssetUid: null };
const el = id => document.getElementById(id);
const canvas = el('canvas');
const ctx = canvas.getContext('2d');

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
function imageUrl(name) { return `/image/${encodeURIComponent(name)}`; }
function thumbUrl(name) { return `/thumb/${encodeURIComponent(name)}`; }
function currentImage() { return state.images[state.index]; }

async function init() {
  const data = await api('/api/state');
  state.images = data.images;
  state.proposals = data.proposals;
  state.categoryLabels = data.category_labels || {};
  state.annotations = data.annotations || { assets: [] };
  renderCategories(); renderProposals(); renderThumbs(); renderAssets();
  el('imageCount').textContent = `${state.images.length} images`;
  if (state.images.length) await loadImage(0);
}
function renderCategories() {
  const select = el('category');
  const categories = Object.keys(state.categoryLabels).sort();
  select.innerHTML = '<option value="">choose segmentation</option>' + categories.map(category => {
    const labels = state.categoryLabels[category].map(item => `${item.source}:${item.label_id}`).join(', ');
    return `<option value="${category}">${category} · ${labels}</option>`;
  }).join('');
}
function renderProposals() {
  const select = el('proposal');
  select.innerHTML = '<option value="">none</option>' + state.proposals.map(p => `<option value="${p.id}">${p.id} · ${p.category} · ${p.point_count} pts</option>`).join('');
}
function renderThumbs() {
  const q = el('search').value.toLowerCase();
  const grid = el('thumbGrid');
  grid.innerHTML = '';
  state.images.forEach((img, i) => {
    if (q && !img.name.toLowerCase().includes(q)) return;
    const div = document.createElement('div');
    div.className = 'thumb' + (i === state.index ? ' active' : '');
    div.innerHTML = `<img src="${thumbUrl(img.name)}" loading="lazy"><span>${img.name}</span>`;
    div.onclick = () => loadImage(i);
    grid.appendChild(div);
  });
}
async function loadImage(i) {
  state.index = Math.max(0, Math.min(state.images.length - 1, i));
  const imgInfo = currentImage();
  el('currentName').textContent = imgInfo.name;
  const img = new Image();
  img.onload = () => { state.image = img; fitCanvas(); loadAssetForImage(); renderThumbs(); };
  img.src = imageUrl(imgInfo.name);
}
function fitCanvas() {
  if (!state.image) return;
  const wrap = el('canvasWrap');
  const maxW = Math.max(360, wrap.clientWidth - 40);
  const maxH = Math.max(260, wrap.clientHeight - 40);
  state.scale = Math.min(maxW / state.image.naturalWidth, maxH / state.image.naturalHeight, 1);
  canvas.width = Math.round(state.image.naturalWidth * state.scale);
  canvas.height = Math.round(state.image.naturalHeight * state.scale);
  draw();
}
function draw() {
  if (!state.image) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(state.image,0,0,canvas.width,canvas.height);
  if (state.bbox) {
    const [x,y,w,h] = state.bbox.map(v => v * state.scale);
    ctx.save(); ctx.strokeStyle = '#52b788'; ctx.lineWidth = 3; ctx.strokeRect(x,y,w,h);
    ctx.fillStyle = 'rgba(82,183,136,.18)'; ctx.fillRect(x,y,w,h); ctx.restore();
    el('bboxBadge').textContent = `${Math.round(state.bbox[0])},${Math.round(state.bbox[1])} ${Math.round(state.bbox[2])}x${Math.round(state.bbox[3])}`;
  } else { el('bboxBadge').textContent = 'no bbox'; }
}
function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  // Use the actual CSS display size. The canvas can be scaled again by max-width,
  // so state.scale alone is not a reliable inverse transform.
  const sx = state.image.naturalWidth / r.width;
  const sy = state.image.naturalHeight / r.height;
  return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
}
canvas.addEventListener('mousedown', e => { const [x,y] = canvasPoint(e); state.drawing = { x0:x, y0:y }; });
canvas.addEventListener('mousemove', e => {
  if (!state.drawing) return;
  const [x,y] = canvasPoint(e); const x0 = state.drawing.x0, y0 = state.drawing.y0;
  state.bbox = [Math.min(x0,x), Math.min(y0,y), Math.abs(x-x0), Math.abs(y-y0)]; draw();
});
canvas.addEventListener('mouseup', () => { state.drawing = null; });

function formAsset() {
  const img = currentImage();
  return {
    uid: state.selectedAssetUid || `mark_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    id: el('assetId').value.trim() || `asset_${Date.now()}`,
    category: el('category').value.trim(), proposal_id: el('proposal').value,
    source_image: img.name, image_path: img.path, bbox_xywh: state.bbox ? state.bbox.map(v => Math.round(v)) : null,
    occlusion: el('occlusion').value, priority: el('priority').value,
    prompt: el('prompt').value, notes: el('notes').value, matched_candidates: state.lastMatch?.candidates || [], bbox_point3d_count: state.lastMatch?.bbox_point3d_count || 0,
    updated_at: new Date().toISOString()
  };
}
function fillForm(asset) {
  state.selectedAssetUid = asset ? (asset.uid || asset.id) : null;
  el('assetId').value = asset?.id || '';
  el('category').value = asset?.category || '';
  el('proposal').value = asset?.proposal_id || '';
  el('occlusion').value = asset?.occlusion || 'low';
  el('priority').value = asset?.priority || 'use';
  el('prompt').value = asset?.prompt || '';
  el('notes').value = asset?.notes || '';
  state.bbox = asset?.bbox_xywh || null;
  state.lastMatch = asset?.matched_candidates ? { candidates: asset.matched_candidates, bbox_point3d_count: asset.bbox_point3d_count || 0 } : null;
  renderMatch(); draw(); renderAssets();
}
function loadAssetForImage() {
  const img = currentImage();
  const asset = state.annotations.assets.find(a => a.source_image === img.name && a.priority !== 'reject');
  fillForm(asset || null);
}
async function saveAsset() {
  const asset = formAsset();
  if (!asset.source_image || !asset.bbox_xywh) { alert('Draw a bbox first.'); return; }
  const idx = state.annotations.assets.findIndex(a => (a.uid || a.id) === asset.uid);
  if (idx >= 0) state.annotations.assets[idx] = asset; else state.annotations.assets.push(asset);
  await saveAnnotations(); fillForm(asset);
}
async function saveAnnotations() {
  await api('/api/annotations', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(state.annotations) });
  el('saveState').textContent = `saved ${new Date().toLocaleTimeString()}`;
}
function renderAssets() {
  const list = el('assetList');
  list.innerHTML = '';
  [...state.annotations.assets].reverse().forEach(asset => {
    const div = document.createElement('div');
    div.className = 'assetItem' + ((asset.uid || asset.id) === state.selectedAssetUid ? ' active' : '');
    div.innerHTML = `<strong>${asset.id}</strong><small>${asset.category || 'uncategorized'} · ${asset.proposal_id || 'no proposal'} · ${asset.source_image}</small>`;
    div.onclick = async () => { const idx = state.images.findIndex(i => i.name === asset.source_image); if (idx >= 0) await loadImage(idx); fillForm(asset); };
    list.appendChild(div);
  });
}
function renderMatch() {
  const box = el('matchResult');
  if (!state.lastMatch) { box.textContent = 'No 3D match yet.'; return; }
  const rows = state.lastMatch.candidates.slice(0, 5).map(c => `${c.proposal_id} ${c.category} score=${c.score.toFixed(3)} hits=${c.shared_point_count}`).join(' | ');
  box.textContent = `bbox points=${state.lastMatch.bbox_point3d_count}; ${rows || 'no candidate'}`;
}
async function match3d() {
  if (!state.bbox) { alert('Draw a bbox first.'); return; }
  const payload = { source_image: currentImage().name, bbox_xywh: state.bbox.map(v => Math.round(v)), category: el('category').value.trim() };
  const result = await api('/api/match', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  state.lastMatch = result;
  if (result.candidates.length) { el('proposal').value = result.candidates[0].proposal_id; }
  renderMatch();
}
function show3dSelection(useSegmentation = true) {
  if (!state.bbox) { alert('Draw a bbox first.'); return; }
  const category = el('category').value.trim();
  if (useSegmentation && !category) { alert('Choose a segmentation category first.'); return; }
  const params = new URLSearchParams({
    image: currentImage().name,
    bbox: state.bbox.map(v => Math.round(v)).join(','),
    mode: useSegmentation ? 'segmentation' : 'bbox'
  });
  if (useSegmentation) params.set('category', category);
  window.open(`${location.protocol}//${location.hostname}:8017/?${params.toString()}`, '_blank');
}
function downloadCrop() {
  if (!state.image || !state.bbox) return;
  const [x,y,w,h] = state.bbox.map(Math.round);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  c.getContext('2d').drawImage(state.image, x, y, w, h, 0, 0, w, h);
  const a = document.createElement('a');
  a.href = c.toDataURL('image/png'); a.download = `${el('assetId').value || 'asset'}_${currentImage().name.replace(/\.[^.]+$/, '')}.png`; a.click();
}

el('search').oninput = renderThumbs;
el('prevBtn').onclick = () => loadImage(state.index - 1);
el('nextBtn').onclick = () => loadImage(state.index + 1);
el('fitBtn').onclick = fitCanvas;
el('clearBoxBtn').onclick = () => { state.bbox = null; draw(); };
el('saveAssetBtn').onclick = saveAsset;
el('downloadCropBtn').onclick = downloadCrop;
el('match3dBtn').onclick = match3d;
el('show3dBtn').onclick = () => show3dSelection(true);
el('show3dBboxBtn').onclick = () => show3dSelection(false);
el('newAssetBtn').onclick = () => fillForm(null);
el('deleteAssetBtn').onclick = async () => { const uid = state.selectedAssetUid; if (uid) state.annotations.assets = state.annotations.assets.filter(a => (a.uid || a.id) !== uid); await saveAnnotations(); fillForm(null); };
window.addEventListener('resize', fitCanvas);
init().catch(err => { console.error(err); alert(err.message); });
</script>
</body>
</html>
"""


def build_state(environment_dir: Path, proposal_dir: Path, output_dir: Path) -> dict:
    image_dir = environment_dir / "images"
    images = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append({"name": path.name, "path": str(path.resolve())})
    proposals = []
    category_labels = {}
    manifest_path = proposal_dir / "proposals.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        category_labels = manifest.get("category_labels", {})
        proposals = [
            {"id": item["id"], "category": item["category"], "point_count": item.get("point_count", 0)}
            for item in manifest.get("proposals", [])
        ]
    annotations_path = output_dir / "annotations.json"
    annotations = {"assets": []}
    if annotations_path.exists():
        annotations = json.loads(annotations_path.read_text())
    return {
        "images": images,
        "proposals": proposals,
        "category_labels": category_labels,
        "annotations": annotations,
    }


def proposal_point_ids(proposal_dir: Path, proposal: dict) -> set[int]:
    rel = Path(str(proposal["point3d_ids"]))
    path = rel if rel.is_absolute() else proposal_dir / rel
    key = str(path.resolve())
    if key not in _PROPOSAL_ID_CACHE:
        import numpy as np
        _PROPOSAL_ID_CACHE[key] = set(int(item) for item in np.load(path).tolist())
    return _PROPOSAL_ID_CACHE[key]


def match_bbox_to_proposals(environment_dir: Path, proposal_dir: Path, image_name: str, bbox_xywh: list[int], category: str | None = None) -> dict:
    manifest_path = proposal_dir / "proposals.json"
    manifest = json.loads(manifest_path.read_text())
    category_norm = normalize_name(category or "") if category else None
    if not category_norm or category_norm not in manifest.get("category_labels", {}):
        available = ", ".join(manifest.get("category_labels", {}).keys())
        raise ValueError(f"Set category before matching. Available: {available}")

    images = colmap_images(environment_dir)
    image = next((item for item in images.values() if item.name == image_name), None)
    if image is None:
        raise ValueError(f"Image not registered in COLMAP: {image_name}")

    import numpy as np
    stem = Path(image_name).stem
    stuff_path = environment_dir / "segmentations" / "stuff" / f"{stem}.png"
    thing_path = environment_dir / "segmentations" / "thing" / f"{stem}.png"
    if not stuff_path.exists() or not thing_path.exists():
        raise FileNotFoundError(f"Missing segmentation masks for {image_name}")
    stuff = np.asarray(Image.open(stuff_path))
    thing = np.asarray(Image.open(thing_path))
    if stuff.shape != thing.shape:
        raise ValueError(f"stuff/thing mask shape mismatch for {image_name}")
    height, width = stuff.shape

    semantic = np.zeros((height, width), dtype=bool)
    for item in manifest["category_labels"][category_norm]:
        source = str(item["source"])
        label_id = int(item["label_id"])
        if source == "thing":
            semantic |= thing == label_id
        elif source == "stuff":
            semantic |= (thing == 0) & (stuff == label_id)

    x, y, w, h = [int(v) for v in bbox_xywh]
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width, x + max(0, w)))
    y2 = max(0, min(height, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox_xywh}")

    xy = np.asarray(image.xys, dtype=np.float64)
    point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
    px = np.rint(xy[:, 0]).astype(np.int64)
    py = np.rint(xy[:, 1]).astype(np.int64)
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    in_bbox = in_image & (point_ids >= 0) & (px >= x1) & (px < x2) & (py >= y1) & (py < y2)
    semantic_track = np.zeros(len(point_ids), dtype=bool)
    semantic_track[in_image] = semantic[py[in_image], px[in_image]]
    selected = in_bbox & semantic_track
    raw_bbox_count = int(np.count_nonzero(in_bbox))
    bbox_ids = set(int(item) for item in point_ids[selected].tolist())

    candidates = []
    for proposal in manifest.get("proposals", []):
        if normalize_name(proposal.get("category", "")) != category_norm:
            continue
        ids = proposal_point_ids(proposal_dir, proposal)
        shared = bbox_ids & ids
        if not shared:
            continue
        shared_count = len(shared)
        bbox_fraction = shared_count / max(1, len(bbox_ids))
        proposal_fraction = shared_count / max(1, len(ids))
        score = (2.0 * bbox_fraction * proposal_fraction) / max(1e-6, bbox_fraction + proposal_fraction)
        candidates.append({
            "proposal_id": proposal["id"],
            "category": proposal["category"],
            "shared_point_count": shared_count,
            "bbox_point_fraction": bbox_fraction,
            "proposal_point_fraction": proposal_fraction,
            "score": score,
            "proposal_point_count": len(ids),
        })
    candidates.sort(key=lambda item: (item["score"], item["shared_point_count"]), reverse=True)
    return {
        "source_image": image_name,
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "category": category_norm,
        "raw_bbox_point3d_count": raw_bbox_count,
        "bbox_point3d_count": len(bbox_ids),
        "semantic_pixel_area_in_bbox": int(np.count_nonzero(semantic[y1:y2, x1:x2])),
        "candidates": candidates[:12],
    }


class AssetPickerHandler(BaseHTTPRequestHandler):
    environment_dir: Path
    proposal_dir: Path
    output_dir: Path

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[asset-picker] " + fmt % args + "\n")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: object, status: int = 200) -> None:
        self.send_bytes(json.dumps(data, indent=2).encode("utf-8"), "application/json", status)

    def image_path(self, name: str) -> Path:
        safe = Path(unquote(name)).name
        path = self.environment_dir / "images" / safe
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.exists():
            raise FileNotFoundError(safe)
        return path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self.send_json(build_state(self.environment_dir, self.proposal_dir, self.output_dir))
            elif parsed.path.startswith("/image/"):
                path = self.image_path(parsed.path.removeprefix("/image/"))
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_bytes(path.read_bytes(), content_type)
            elif parsed.path.startswith("/thumb/"):
                path = self.image_path(parsed.path.removeprefix("/thumb/"))
                with Image.open(path) as image:
                    image.thumbnail((360, 220))
                    from io import BytesIO
                    buffer = BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=82)
                    self.send_bytes(buffer.getvalue(), "image/jpeg")
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if parsed.path == "/api/match":
            result = match_bbox_to_proposals(
                self.environment_dir,
                self.proposal_dir,
                str(data["source_image"]),
                list(data["bbox_xywh"]),
                str(data.get("category") or "") or None,
            )
            self.send_json(result)
            return
        if parsed.path != "/api/annotations":
            self.send_json({"error": "not found"}, 404)
            return
        if not isinstance(data, dict) or not isinstance(data.get("assets"), list):
            self.send_json({"error": "expected assets list"}, 400)
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "annotations.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
        self.send_json({"ok": True, "path": str(path.resolve()), "asset_count": len(data["assets"])})


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the StageRecon manual asset picker UI.")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8015)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    AssetPickerHandler.environment_dir = args.environment_dir.resolve()
    AssetPickerHandler.proposal_dir = args.proposal_dir.resolve()
    AssetPickerHandler.output_dir = args.output_dir.resolve()
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), AssetPickerHandler)
    print(f"Asset picker: http://{args.host}:{args.port}")
    print(f"Images: {AssetPickerHandler.environment_dir / 'images'}")
    print(f"Annotations: {AssetPickerHandler.output_dir / 'annotations.json'}")
    server.serve_forever()


if __name__ == "__main__":
    main()

"""Generate a self-contained-enough local review page for one caption sample."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def build_review_html(document: dict[str, Any], validation: dict[str, Any]) -> str:
    assets = document["assets"]
    bindings = document["extensions"]["mtss_3d"]["entity_bindings"]
    cards = []
    for binding in bindings:
        source_track = binding.get("source_track")
        track_label = (
            str(source_track["sam3_track_id"])
            if isinstance(source_track, dict) else "not used"
        )
        candidates = ", ".join(
            f"{item['canonical_identity']}={item['score']:.3f}"
            for item in binding.get("candidate_matches", [])
        )
        cards.append(
            "<article>"
            f"<img src=\"{html.escape(binding['appearance_image'])}\" alt=\"{html.escape(binding['ref_id'])}\">"
            f"<h3>{html.escape(binding['ref_id'])}</h3>"
            f"<p>identity: {html.escape(str(binding.get('canonical_identity')))}</p>"
            f"<p>status: {html.escape(binding['identity_status'])}</p>"
            f"<p>track: {track_label}</p>"
            f"<p>candidates: {html.escape(candidates or 'none')}</p>"
            "</article>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MTSS-3D review: {html.escape(document['sample_id'])}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #f6f7f9; color: #202124; }}
video {{ width: min(720px, 100%); background: #111; margin-right: 1rem; }}
.videos, .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
article {{ background: white; padding: 1rem; border-radius: 8px; max-width: 260px; }}
article img {{ max-width: 240px; max-height: 280px; object-fit: contain; background: #ddd; }}
pre {{ white-space: pre-wrap; background: white; padding: 1rem; overflow-wrap: anywhere; }}
.failed {{ color: #b3261e; }} .passed {{ color: #137333; }}
</style>
</head>
<body>
<h1>{html.escape(document['sample_id'])}</h1>
<p class="{validation['status']}">validation: {validation['status']} · status: {document['status']}</p>
<section class="videos">
<div><h2>Source/Target</h2><video controls src="{html.escape(assets['source_video']['path'])}"></video></div>
<div><h2>Video 1 Previs</h2><video controls src="{html.escape(assets['previs_video']['path'])}"></video></div>
</section>
<h2>Reference Bank</h2><section class="cards">{''.join(cards)}</section>
<h2>MTSS</h2><pre>{_json(document['mtss'])}</pre>
<h2>Validation</h2><pre>{_json(validation)}</pre>
<h2>Human patch</h2>
<p>审核修改请写入同目录 human_patch.json；不要覆盖 raw response。</p>
</body></html>
"""


def write_review(path: Path, document: dict[str, Any], validation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_review_html(document, validation), encoding="utf-8")

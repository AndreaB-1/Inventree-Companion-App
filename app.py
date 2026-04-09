# app.py
# FastAPI backend for InvenTree photo upload + QR "scan by photo" + AI name/desc
# pip install: fastapi uvicorn pillow inventree python-multipart opencv-python-headless numpy pyzbar openai

import os
import io
import re
import json
import tempfile
from io import BytesIO
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from PIL import Image, ImageOps

# InvenTree
from inventree.api import InvenTreeAPI
from inventree.part import Part

# QR decode (photo)
import numpy as np
import cv2
from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

# AI
from openai import OpenAI

import mimetypes
import requests
from urllib.parse import urljoin
from fastapi.responses import StreamingResponse
import base64
import math




# ------------------------------- ENV -------------------------------
INVENTREE_URL = os.getenv("INVENTREE_URL")  # e.g. http://192.168.1.110/api/
INVENTREE_TOKEN = os.getenv("INVENTREE_TOKEN")
INVENTREE_LABEL_PLUGIN_KEY = os.getenv("INVENTREE_LABEL_PLUGIN_KEY", "inventreelabelmachine")
INVENTREE_LABEL_MACHINE_ID = os.getenv("INVENTREE_LABEL_MACHINE_ID")  # UUID of your QL printer

if not INVENTREE_URL or not INVENTREE_TOKEN:
    raise RuntimeError("Set INVENTREE_URL and INVENTREE_TOKEN in the environment (.env / compose).")
# -------------------------------------------------------------------

def inv_api():
    return InvenTreeAPI(host=INVENTREE_URL, token=INVENTREE_TOKEN)

def root_url_from_api(api_url: str) -> str:
    # Convert http://host/api/  ->  http://host/
    if api_url.endswith("/api/"):
        return api_url[:-5] + "/"
    if api_url.endswith("/api"):
        return api_url[:-4] + "/"
    return api_url if api_url.endswith("/") else api_url + "/"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # LAN-friendly; tighten later if you like
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

def _resolve_default_label_plugin_id(api: InvenTreeAPI) -> int | None:
    """
    Try to find the 'InvenTreeLabelMachine' plugin id.
    Falls back to any active plugin whose name contains 'LabelMachine'.
    Returns None if none found (server will use its own default).
    """
    try:
        res = api.get("plugin/", params={"active": True})
        rows = res["results"] if isinstance(res, dict) and "results" in res else (res or [])
        # exact name match
        for p in rows:
            if (p.get("name") or "").strip() == "InvenTreeLabelMachine":
                return p.get("pk")
        # fuzzy fallback
        for p in rows:
            name = (p.get("name") or "")
            if "labelmachine" in name.lower():
                return p.get("pk")
    except Exception:
        pass
    return None

def _resolve_label_plugin_key(api: InvenTreeAPI) -> Optional[str]:
    """
    Return the key (string) for InvenTreeLabelMachine, or any plugin whose
    name contains 'LabelMachine'. Returns None if not found.
    """
    try:
        res = api.get("plugin/", params={"active": True})
        rows = res["results"] if isinstance(res, dict) and "results" in res else (res or [])
        # exact name
        for p in rows:
            if (p.get("name") or "").strip() == "InvenTreeLabelMachine":
                return p.get("key")
        # fuzzy
        for p in rows:
            if "labelmachine" in (p.get("name") or "").lower():
                return p.get("key")
    except Exception:
        pass
    return None


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/manifest.json")
def manifest():
    return FileResponse("static/manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse("static/sw.js", media_type="application/javascript")


# ----------------- Parts & Locations -----------------

@app.get("/api/part/by-ipn")
def part_by_ipn(ipn: str):
    try:
        api = inv_api()
        res = api.get("part/", params={"IPN": ipn})
        rows = res["results"] if isinstance(res, dict) and "results" in res else res
        if not rows:
            raise HTTPException(status_code=404, detail="No part found")
        p = rows[0]
        return {"pk": p.get("pk"), "name": p.get("name"), "ipn": p.get("IPN")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/part/by-id")
def part_by_id(pk: str):
    try:
        api = inv_api()
        p = api.get(f"part/{pk}/")
        if not p or not p.get("pk"):
            raise HTTPException(status_code=404, detail="No part found")
        return {"pk": p.get("pk"), "name": p.get("name"), "ipn": p.get("IPN")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/parts")
def list_parts(q: Optional[str] = None, limit: int = 20, offset: int = 0):
    """Search parts by name/IPN (DRF 'search')."""
    try:
        api = inv_api()
        params = {"limit": limit, "offset": offset}
        if q:
            params["search"] = q
        res = api.get("part/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            out.append({
                "pk": r.get("pk"),
                "name": r.get("name"),
                "ipn": r.get("IPN"),
                "description": r.get("description"),
            })
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/locations")
def list_locations(q: Optional[str] = None, limit: int = 20, offset: int = 0):
    """List stock locations (with optional text search)."""
    try:
        api = inv_api()
        params = {"limit": limit, "offset": offset}
        if q:
            params["search"] = q
        res = api.get("stock/location/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        return [
            {"pk": r.get("pk"), "name": r.get("name"), "path": r.get("pathstring") or r.get("name")}
            for r in rows if isinstance(r, dict)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/part/create")
def create_part(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    ipn: Optional[str] = Form(None),
    initial_qty: Optional[float] = Form(None),
    location_id: Optional[int] = Form(None),
):
    try:
        api = inv_api()
        data = {"name": name}
        if description:
            data["description"] = description
        if ipn:
            data["IPN"] = ipn

        created = api.post("part/", data)
        if not isinstance(created, dict) or not created.get("pk"):
            raise HTTPException(status_code=400, detail=f"Unexpected response: {created}")

        part_pk = created["pk"]
        stock_msg = None

        if initial_qty and float(initial_qty) > 0:
            if not location_id:
                raise HTTPException(status_code=400, detail="location_id required when initial_qty is provided.")
            try:
                stock_payload = {"part": part_pk, "quantity": float(initial_qty), "location": int(location_id)}
                api.post("stock/", stock_payload)
                stock_msg = f"Stock item created: {initial_qty} @ location {location_id}"
            except Exception as se:
                stock_msg = f"Stock not created: {se}"

        return {"ok": True, "pk": part_pk, "name": created.get("name"), "message": stock_msg or "Part created."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Part image helper -----------------

@app.get("/api/part/image-url")
def part_image_url(pk: str):
    """
    Return absolute image + thumbnail URLs for a Part.
    InvenTree returns paths like '/media/part_images/...'; we convert to absolute.
    """
    try:
        api = inv_api()
        p = api.get(f"part/{pk}/")
        if not p or not p.get("pk"):
            raise HTTPException(status_code=404, detail="Part not found")

        root = root_url_from_api(INVENTREE_URL)
        img = p.get("image") or ""
        th = p.get("thumbnail") or ""

        def abs_url(v: str) -> Optional[str]:
            if not v:
                return None
            return v if v.startswith("http") else (root.rstrip("/") + v)

        return {"image": abs_url(img), "thumbnail": abs_url(th)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Image Search / Auto-Download -----------------

class ImageSearchReq(BaseModel):
    query: str

class ImageDownloadReq(BaseModel):
    part_id: str
    image_url: str


# ----------------- Part Metadata Update -----------------

class PartUpdateReq(BaseModel):
    pk: int
    name: Optional[str] = None
    description: Optional[str] = None
    ipn: Optional[str] = None

@app.post("/api/part/update-meta")
def update_part_meta(req: PartUpdateReq):
    data = {}
    if req.name is not None:
        data["name"] = req.name
    if req.description is not None:
        data["description"] = req.description
    if req.ipn is not None:
        data["IPN"] = req.ipn
    if not data:
        return {"ok": True, "message": "Nothing to update"}
    try:
        inv_url = INVENTREE_URL.rstrip("/")
        resp = requests.patch(
            f"{inv_url}/part/{req.pk}/",
            json=data,
            headers={"Authorization": f"Token {INVENTREE_TOKEN}"},
            timeout=15,
        )
        if not resp.ok:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/part/image-search")
def part_image_search(req: ImageSearchReq):
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.images(req.query, max_results=6):
                url = r.get("image") or r.get("url")
                if url:
                    results.append(url)
                if len(results) >= 6:
                    break
        return {"images": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/part/image-download")
def part_image_download(req: ImageDownloadReq):
    tmp_path = None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(req.image_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Image host returned HTTP {resp.status_code}")
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((800, 800), Image.LANCZOS)
        fd, tmp_path = tempfile.mkstemp(prefix="inventree_imgdl_", suffix=".jpg")
        os.close(fd)
        img.save(tmp_path, "JPEG", quality=92)
        api = inv_api()
        Part(api, pk=str(req.part_id)).uploadImage(tmp_path)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ----------------- Image Upload -----------------

@app.post("/api/upload")
async def upload(part_id: str = Form(...), image: UploadFile = File(...)):
    tmp_path = None
    try:
        raw = await image.read()
        img = Image.open(BytesIO(raw)).convert("RGB")

        # center-crop square + resize
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((800, 800), Image.LANCZOS)

        fd, tmp_path = tempfile.mkstemp(prefix="inventree_upload_", suffix=".jpg")
        os.close(fd)
        img.save(tmp_path, "JPEG", quality=92)

        print(f"[UPLOAD] temp path = {tmp_path}")

        api = inv_api()
        part = Part(api, pk=str(part_id))
        part.uploadImage(tmp_path)

        return {"ok": True, "message": "Image uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ----------------- QR "Scan by Photo" (HTTP-friendly) -----------------

@app.post("/api/decode_qr")
async def decode_qr(image: UploadFile = File(...)):
    """
    Robust QR decode from a phone photo:
    1) Fix EXIF rotation
    2) Try ZBar (pyzbar) across multiple preprocess + scales + rotations
    3) Fallback to OpenCV QRCodeDetector variants
    Returns: { text, payload? }
    """
    try:
        raw = await image.read()

        # Load with PIL first to respect EXIF orientation
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil)  # fix sideways/upside-down
        pil = pil.convert("RGB")

        # Convert to OpenCV BGR
        base = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Normalize size (help small photos)
        def normalize_size(img):
            h, w = img.shape[:2]
            max_side = max(h, w)
            if max_side < 900:
                scale = 900 / max_side
                return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            if max_side > 2000:
                scale = 2000 / max_side
                return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            return img

        base = normalize_size(base)

        def parse_hits(hits: List[str]):
            if not hits:
                return None
            # Prefer JSON {"part": ...}
            for t in hits:
                try:
                    obj = json.loads(t)
                    if isinstance(obj, dict) and "part" in obj:
                        return t, obj
                except Exception:
                    pass
            # Prefer URL ?part=/id=/ipn=
            for t in hits:
                m = re.search(r'(?:\?|#|&)(?:part|id|ipn)=([^&]+)', t, re.I)
                if m:
                    val = m.group(1)
                    try:
                        ival = int(val)
                        return t, {"part": ival}
                    except Exception:
                        return t, {"query": val}
            return hits[0], None

        def try_pyzbar(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_clahe = clahe.apply(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 10)

            variants = [img, gray, gray_clahe, thr]
            rotations = [0, 90, 180, 270]
            h, w = img.shape[:2]
            scales = [1.0, 1.5, 2.0] if max(h, w) < 1200 else [1.0]

            results: List[str] = []
            for v in variants:
                for ang in rotations:
                    rot = v if ang == 0 else cv2.rotate(
                        v,
                        cv2.ROTATE_90_CLOCKWISE if ang == 90 else
                        cv2.ROTATE_180 if ang == 180 else
                        cv2.ROTATE_90_COUNTERCLOCKWISE
                    )
                    for s in scales:
                        rot_s = rot if s == 1.0 else cv2.resize(
                            rot, (int(rot.shape[1]*s), int(rot.shape[0]*s)), interpolation=cv2.INTER_CUBIC
                        )
                        dec = zbar_decode(rot_s, symbols=[ZBarSymbol.QRCODE])
                        if dec:
                            for d in dec:
                                try:
                                    txt = d.data.decode("utf-8", errors="ignore").strip()
                                except Exception:
                                    txt = str(d.data)
                                if txt:
                                    results.append(txt)
                            if results:
                                best = parse_hits(results)
                                if best:
                                    return best
            return None

        def try_opencv(img):
            det = cv2.QRCodeDetector()
            def decode_any(cv_img):
                try:
                    texts, _, _ = det.detectAndDecodeMulti(cv_img)
                    hits = [t for t in texts if t] if texts else []
                    if hits:
                        return hits
                except Exception:
                    pass
                try:
                    text, _, _ = det.detectAndDecode(cv_img)
                    if text:
                        return [text]
                except Exception:
                    pass
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_clahe = clahe.apply(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 10)

            variants = [img, gray, gray_clahe, thr]
            rotations = [0, 90, 180, 270]
            scales = [1.0, 1.5, 2.0] if max(img.shape[:2]) < 1200 else [1.0]

            for v in variants:
                for ang in rotations:
                    rot = v if ang == 0 else cv2.rotate(
                        v,
                        cv2.ROTATE_90_CLOCKWISE if ang == 90 else
                        cv2.ROTATE_180 if ang == 180 else
                        cv2.ROTATE_90_COUNTERCLOCKWISE
                    )
                    for s in scales:
                        rot_s = rot if s == 1.0 else cv2.resize(
                            rot, (int(rot.shape[1]*s), int(rot.shape[0]*s)), interpolation=cv2.INTER_CUBIC
                        )
                        hits = decode_any(rot_s)
                        if hits:
                            return parse_hits(hits)
            return None

        out = try_pyzbar(base)
        if out:
            text, payload = out
            return {"text": text, "payload": payload}

        out = try_opencv(base)
        if out:
            text, payload = out
            return {"text": text, "payload": payload}

        raise HTTPException(status_code=400, detail="No QR detected")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Stock (for "kiosk" page) -----------------

@app.get("/api/stock/by-part")
def stock_by_part(part_id: int, limit: int = 250, offset: int = 0):
    """
    Return stock items for a given part, with location info.
    """
    try:
        api = inv_api()
        params = {"part": part_id, "limit": limit, "offset": offset}
        res = api.get("stock/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []

        # Build a small cache of locations to reduce calls
        loc_cache: Dict[int, Dict[str, Any]] = {}

        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            sid = r.get("pk")
            qty = r.get("quantity")
            loc_id = r.get("location")
            loc_name = None
            loc_path = None
            if isinstance(loc_id, int):
                if loc_id not in loc_cache:
                    try:
                        ld = api.get(f"stock/location/{loc_id}/")
                        loc_cache[loc_id] = {
                            "name": ld.get("name"),
                            "path": ld.get("pathstring") or ld.get("name")
                        }
                    except Exception:
                        loc_cache[loc_id] = {"name": None, "path": None}
                loc_name = loc_cache[loc_id]["name"]
                loc_path = loc_cache[loc_id]["path"]

            out.append({
                "pk": sid,
                "quantity": qty,
                "location_id": loc_id,
                "location_name": loc_name,
                "location_path": loc_path,
                "batch": r.get("batch"),
                "status": r.get("status"),
                "serial": r.get("serial")
            })
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StockAdjustReq(BaseModel):
    stock_id: int
    quantity: float
    notes: Optional[str] = None

@app.post("/api/stock/add_qty")
def stock_add_qty(req: StockAdjustReq):
    """
    Add quantity to an existing StockItem via /api/stock/add/
    """
    try:
        api = inv_api()
        payload = {
            "items": [{"pk": int(req.stock_id), "quantity": str(req.quantity)}],
            "notes": req.notes or "Adjusted via companion app",
        }
        api.post("stock/add/", payload)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stock/remove_qty")
def stock_remove_qty(req: StockAdjustReq):
    """
    Remove quantity from an existing StockItem via /api/stock/remove/
    """
    try:
        api = inv_api()
        payload = {
            "items": [{"pk": int(req.stock_id), "quantity": str(req.quantity)}],
            "notes": req.notes or "Adjusted via companion app",
        }
        api.post("stock/remove/", payload)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- AI suggest endpoint (unified fastener + electronics) -----------------

class AISuggestReq(BaseModel):
    prompt: str
    language: Optional[str] = None      # e.g. "en" or "it"
    name_only: Optional[bool] = False
    strict: Optional[bool] = True       # default ON

def _fallback_desc_from_name(name: str) -> str:
    return name

@app.post("/api/ai/suggest-part")
async def ai_suggest_part(req: AISuggestReq):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    strict_flag = True if req.strict is None else bool(req.strict)

    schema = {
        "name": "inventory_unified",
        "strict": strict_flag,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "domain": {"type": "string", "enum": ["fastener", "electronics"]},
                "render_name": {"type": "string"},
                "render_description": {"type": "string"},
                "fastener": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "size_kind": {"type": "string", "enum": ["metric_thread","sheet_metal_thread","washer","nut","other"]},
                        "diameter_mm": {"type": "number"},
                        "length_mm": {"type": "number", "nullable": True},
                        "pitch_mm": {"type": "number", "nullable": True},
                        "standard": {"type": "string"},
                        "type_name": {"type": "string"},
                        "head": {"type": "string", "nullable": True},
                        "drive": {"type": "string", "nullable": True},
                        "material_short": {"type": "string"},
                        "material_long": {"type": "string"},
                        "property_class": {"type": "string", "nullable": True},
                        "finish": {"type": "string", "nullable": True},
                        "af_mm": {"type": "number", "nullable": True},
                        "non_standard": {"type": "boolean"},
                        "alt_standard": {"type": "string", "nullable": True},
                        "notes": {"type": "string", "nullable": True}
                    },
                    "required": [
                        "size_kind","diameter_mm","length_mm","pitch_mm","standard","type_name","head","drive",
                        "material_short","material_long","property_class","finish","af_mm","non_standard","alt_standard","notes"
                    ]
                },
                "electronics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {"type": "string"},
                        "type_name": {"type": "string"},
                        "part_number": {"type": "string", "nullable": True},
                        "series": {"type": "string", "nullable": True},
                        "brand": {"type": "string", "nullable": True},
                        "pitch_mm": {"type": "number", "nullable": True},
                        "pins": {"type": "integer", "nullable": True},
                        "gender": {"type": "string", "nullable": True, "enum": ["Male","Female","Male-Female","None"]},
                        "color": {"type": "string", "nullable": True},
                        "size_inch": {"type": "string", "nullable": True},
                        "dimensions": {"type": "string", "nullable": True},
                        "protocol": {"type": "string", "nullable": True},
                        "voltage_v": {"type": "string", "nullable": True},
                        "frequency_hz": {"type": "string", "nullable": True},
                        "capacity": {"type": "string", "nullable": True},
                        "extras": {"type": "string", "nullable": True},
                        "non_standard": {"type": "boolean", "default": False},
                        "alt_standard": {"type": "string", "nullable": True}
                    },
                    "required": [
                        "category","type_name","part_number","series","brand","pitch_mm","pins","gender","color",
                        "size_inch","dimensions","protocol","voltage_v","frequency_hz","capacity","extras",
                        "non_standard","alt_standard"
                    ]
                }
            },
            "required": ["domain","render_name","render_description"]
        }
    }

    lang = (req.language or "en").strip()

    system_rules = f"""
You create inventory-friendly names/descriptions for fasteners AND electronics/tools.

GENERAL:
- Output must be valid JSON that matches the provided JSON schema.
- ONE clean result. No duplicates. No code fences in your output.
- Keep Names SHORT and readable. Description compact (single line).
- Use US terminology.
- In strict mode, include every key in the selected sub-object (fill non-applicable fields with null / empty string).

[... rules + examples trimmed for brevity, same as previous long version ...]
"""

    examples = [
        {"role": "user", "content": "M3x22 A2 DIN 912 socket head cap screw"},
        {"role": "assistant", "content": json.dumps({
            "domain":"fastener",
            "render_name":"M3 × 22 – Socket Head Cap Screw A2",
            "render_description":"M3 × 22 – Socket Head Cap Screw – DIN 912 – A2 Stainless Steel",
            "fastener":{
                "size_kind":"metric_thread","diameter_mm":3,"length_mm":22,"pitch_mm":None,
                "standard":"DIN 912","type_name":"Socket Head Cap Screw","head":"Socket","drive":"Hex",
                "material_short":"A2","material_long":"A2 Stainless Steel","property_class":None,"finish":None,
                "af_mm":None,"non_standard":False,"alt_standard":None,"notes":""
            }
        })},
        {"role": "user", "content": "BC548 transistor"},
        {"role": "assistant", "content": json.dumps({
            "domain":"electronics",
            "render_name":"Transistor NPN Bipolar - BC548",
            "render_description":"Transistor NPN Bipolar - BC548",
            "electronics":{
                "category":"Transistor","type_name":"Transistor NPN Bipolar","part_number":"BC548",
                "series": None,"brand": None,"pitch_mm": None,"pins": None,"gender": "None","color": None,
                "size_inch": None,"dimensions": None,"protocol": None,"voltage_v": None,"frequency_hz": None,
                "capacity": None,"extras": None,"non_standard": False,"alt_standard": None
            }
        })}
    ]

    user_msg = "Prompt:\n" + req.prompt.strip() + "\n\nReturn ONLY JSON matching the schema."

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":system_rules},
                *examples,
                {"role":"user","content":user_msg}
            ],
            response_format={"type":"json_schema","json_schema":schema},
            temperature=0.2,
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)

        name = (data.get("render_name") or "").strip()
        desc = (data.get("render_description") or "").strip()
        if not name:
            raise HTTPException(status_code=500, detail="AI returned empty name")
        if not desc:
            desc = _fallback_desc_from_name(name)
        if req.name_only:
            desc = _fallback_desc_from_name(name)

        return {"name": name, "description": desc, "structured": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")

@app.get("/api/part/image-url")
def part_image_url(pk: str):
    """
    Return absolute URLs for image and thumbnail from InvenTree for a Part.
    """
    try:
        api = inv_api()
        p = api.get(f"part/{pk}/")
        if not p or not p.get("pk"):
            raise HTTPException(status_code=404, detail="Part not found")

        image_path = p.get("image") or ""
        thumb_path = p.get("thumbnail") or ""

        # Build absolute base (strip trailing /api[/] from INVENTREE_URL)
        base = INVENTREE_URL.rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        if base.endswith("/api/"):
            base = base[:-5]

        def abs_url(path: str) -> str:
            if not path:
                return ""
            if path.startswith("http://") or path.startswith("https://"):
                return path
            # InvenTree returns "/media/..." -> join with base
            return base.rstrip("/") + path

        return {
            "image": abs_url(image_path),
            "thumbnail": abs_url(thumb_path),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/part/image-proxy")
def part_image_proxy(pk: str, thumb: bool = False):
    """
    Stream the part image (or thumbnail) inline with a correct Content-Type so the <img> renders.
    """
    try:
        api = inv_api()
        p = api.get(f"part/{pk}/")
        if not p or not p.get("pk"):
            raise HTTPException(status_code=404, detail="Part not found")

        path = (p.get("thumbnail") if thumb else p.get("image")) or p.get("thumbnail")
        if not path:
            raise HTTPException(status_code=404, detail="No image available for this part")

        # Compute absolute URL as above
        base = INVENTREE_URL.rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        if base.endswith("/api/"):
            base = base[:-5]
        url = path if path.startswith("http") else (base.rstrip("/") + path)

        # Some setups need auth for media; include token just in case
        headers = {}
        if INVENTREE_TOKEN:
            headers["Authorization"] = f"Token {INVENTREE_TOKEN}"

        r = requests.get(url, headers=headers, stream=True, timeout=20)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"InvenTree returned {r.status_code}")

        ctype = r.headers.get("Content-Type", "")
        if not ctype or "octet-stream" in ctype:
            ctype = mimetypes.guess_type(url)[0] or "image/jpeg"

        # Force inline display
        return StreamingResponse(
            r.raw,
            media_type=ctype,
            headers={"Content-Disposition": "inline"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------- Label Templates / Plugins / Print (minimal) ---------------

@app.get("/api/labels/templates")
def label_templates(
    model_type: str,
    enabled: bool = True,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """Proxy InvenTree label templates (model_type: part|stockitem|stocklocation)."""
    try:
        api = inv_api()
        params: Dict[str, Any] = {"model_type": model_type, "limit": limit, "offset": offset}
        if enabled is not None:
            params["enabled"] = enabled
        if search:
            params["search"] = search
        res = api.get("label/template/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        return [
            {
                "pk": r.get("pk"),
                "name": r.get("name"),
                "model_type": r.get("model_type"),
                "description": r.get("description"),
                "enabled": r.get("enabled"),
            }
            for r in rows if isinstance(r, dict)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plugins/label")
def plugins_with_label_mixin(active: Optional[bool] = True):
    """List label-capable plugins (LabelPrintingMixin) with their keys."""
    try:
        api = inv_api()
        params: Dict[str, Any] = {"mixin": "label"}
        if active is not None:
            params["active"] = active
        # Path is '/plugin/' (singular)
        res = api.get("plugin/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        return [
            {"pk": r.get("pk"), "key": r.get("key"), "name": r.get("name"), "active": r.get("active")}
            for r in rows if isinstance(r, dict)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/labels/print")
def print_labels(payload: dict):
    """
    Body from UI can be minimal:
      {
        "template_id": int,     # required
        "items": [int, ...],    # required (part ids when model=part)
        "copies": 1,            # optional
        "plugin_key": "inventreelabelmachine",  # optional; backend will default
        "machine_id": "UUID"    # optional; backend will default if env set
      }
    We then POST to InvenTree as it expects:
      { "template": id, "items": [...], "plugin": key, "machine": uuid, "copies": n }
    """
    try:
        template_id = int(payload.get("template_id") or 0)
        items = payload.get("items") or []
        if not template_id or not isinstance(items, list) or not items:
            raise HTTPException(status_code=400, detail="template_id and non-empty items are required")

        copies = payload.get("copies")
        user_plugin_key = payload.get("plugin_key")
        user_machine_id = payload.get("machine_id")

        api = inv_api()

        # Resolve plugin key: prefer payload -> env -> auto-detect -> None
        plugin_key = (
            (user_plugin_key or "").strip()
            or (INVENTREE_LABEL_PLUGIN_KEY or "").strip()
            or _resolve_label_plugin_key(api)
        )

        # Resolve machine id: prefer payload -> env (LabelMachine expects a UUID)
        machine_id = (user_machine_id or "").strip() or (INVENTREE_LABEL_MACHINE_ID or "").strip()

        body = {
            "template": template_id,
            "items": items,
        }
        if plugin_key:
            body["plugin"] = plugin_key  # <- key STRING, e.g. "inventreelabelmachine"
        if machine_id:
            body["machine"] = machine_id  # <- UUID from your LabelMachine config
        if copies:
            body["copies"] = int(copies)

        res = api.post("label/print/", body)
        return {"ok": True, "result": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HARDWARE LABEL GENERATION ====================

LABEL_HEIGHT_PX = 106   # 12mm DK-2210 printable width on QL-710W at 300 dpi
LABEL_DPI = 300


def _hw_font(size: int = 10):
    """Return a PIL font, falling back to the built-in bitmap font."""
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    ]:
        if os.path.exists(path):
            try:
                from PIL import ImageFont as _IF
                return _IF.truetype(path, size)
            except Exception:
                pass
    from PIL import ImageFont as _IF
    try:
        return _IF.load_default(size=size)
    except TypeError:
        return _IF.load_default()


def _hex_pts(cx, cy, r):
    return [(cx + r * math.cos(math.radians(30 + 60 * i)),
             cy + r * math.sin(math.radians(30 + 60 * i))) for i in range(6)]


# ── top-view renderers ─────────────────────────────────────────────

def _tv_screw(d, specs, x, y, w, h):
    head  = specs.get('head_type', 'Socket Cap')
    drive = specs.get('drive', 'Hex (Allen)')
    cx, cy = x + w // 2, y + h // 2
    r  = max(4, min(w, h) // 2 - 2)
    if head == 'Hex':
        d.polygon(_hex_pts(cx, cy, r), outline=0)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
    dr = max(2, int(r * 0.5))
    if drive == 'Hex (Allen)':
        d.polygon(_hex_pts(cx, cy, dr), fill=0)
    elif drive == 'Slotted':
        t = max(1, dr // 3)
        d.rectangle([cx - dr, cy - t, cx + dr, cy + t], fill=0)
    elif drive in ('Phillips', 'Pozidriv'):
        t = max(1, dr // 3)
        d.rectangle([cx - dr, cy - t, cx + dr, cy + t], fill=0)
        d.rectangle([cx - t, cy - dr, cx + t, cy + dr], fill=0)
    elif drive == 'Torx':
        for ang in [0, 60, 120]:
            dx = dr * math.cos(math.radians(ang))
            dy = dr * math.sin(math.radians(ang))
            d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=0, width=max(2, dr // 3))
    elif drive == 'Square':
        sq = max(2, dr // 2)
        d.rectangle([cx - sq, cy - sq, cx + sq, cy + sq], fill=0)


def _tv_nut(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    r = max(4, min(w, h) // 2 - 2)
    d.polygon(_hex_pts(cx, cy, r), outline=0)
    hr = max(2, int(r * 0.45))
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill='white', outline=0, width=1)


def _tv_washer(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    ro = max(5, min(w, h) // 2 - 2)
    ri = max(2, int(ro * 0.4))
    d.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=0, width=1)
    d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], outline=0, width=1)


def _tv_standoff(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    r = max(4, min(w, h) // 2 - 2)
    if specs.get('body_shape', 'Hex') == 'Hex':
        d.polygon(_hex_pts(cx, cy, r), outline=0)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
    hr = max(2, int(r * 0.35))
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill='white', outline=0, width=1)


def _tv_rivet(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    r = max(4, min(w, h) // 2 - 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
    if specs.get('rivet_type', '') == 'Blind/Pop':
        mr = max(1, r // 3)
        d.ellipse([cx - mr, cy - mr, cx + mr, cy + mr], fill=0)


def _tv_pin(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    r = max(3, min(w, h) // 2 - 3)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
    if specs.get('pin_type', '') in ('Roll Pin (Spring)', 'Slotted'):
        d.line([(cx - r, cy), (cx + r, cy)], fill=0, width=1)


def _tv_circlip(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    ro = max(5, min(w, h) // 2 - 2)
    ri = max(3, int(ro * 0.65))
    gap = 55
    start, end = gap // 2, 360 - gap // 2
    for a in range(start, end, 5):
        a1 = math.radians(a)
        a2 = math.radians(min(a + 5, end))
        for ri_ in [ri, ro]:
            d.line([(cx + ri_ * math.cos(a1), cy + ri_ * math.sin(a1)),
                    (cx + ri_ * math.cos(a2), cy + ri_ * math.sin(a2))], fill=0, width=1)
    rm = (ri + ro) // 2
    for a in range(start, end, 3):
        a1 = math.radians(a + 1.5)
        d.point((cx + rm * math.cos(a1), cy + rm * math.sin(a1)), fill=0)
    for ta in [math.radians(start - 5), math.radians(end + 5)]:
        tx_, ty_ = cx + ro * math.cos(ta), cy + ro * math.sin(ta)
        d.ellipse([tx_ - 2, ty_ - 2, tx_ + 2, ty_ + 2], fill=0)


def _tv_insert(d, specs, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    r = max(4, min(w, h) // 2 - 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
    hr = max(2, int(r * 0.35))
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill='white', outline=0, width=1)
    for i in range(3):
        a = math.radians(i * 60)
        lx  = cx + (r - 1) * math.cos(a)
        ly_ = cy + (r - 1) * math.sin(a)
        lx2 = cx + (r + 3) * math.cos(a + math.radians(15))
        ly2 = cy + (r + 3) * math.sin(a + math.radians(15))
        d.line([(lx, ly_), (lx2, ly2)], fill=0, width=1)


# ── side-view renderers ────────────────────────────────────────────

def _sv_screw(d, specs, x, y, w, h):
    head = specs.get('head_type', 'Socket Cap')
    cx = x + w // 2
    hh = max(5, int(h * 0.26))
    sw = max(4, int(w * 0.30))
    hw = min(int(sw * 2.1), w - 4)
    sh = h - hh - 2
    sx1, sx2 = cx - sw // 2, cx + sw // 2
    hx1, hx2 = cx - hw // 2, cx + hw // 2
    if head == 'Countersunk':
        d.polygon([(hx1, y), (hx2, y), (sx2, y + hh), (sx1, y + hh)], outline=0)
    elif head == 'Set Screw':
        hh = 0; sh = h - 2
    elif head == 'Button':
        dome = hh * 2
        d.pieslice([hx1, y - dome // 2, hx2, y + dome // 2], 180, 360, fill=0, outline=0)
    elif head == 'Flange Hex':
        d.rectangle([hx1, y, hx2, y + hh], outline=0, width=1)
        d.rectangle([hx1 - 2, y + hh, hx2 + 2, y + hh + 2], fill=0)
    else:
        d.rectangle([hx1, y, hx2, y + hh], outline=0, width=1)
    d.rectangle([sx1, y + hh, sx2, y + hh + sh], outline=0, width=1)
    nt = max(3, sh // 5)
    step = sh / nt
    for i in range(nt):
        ty = y + hh + i * step
        d.line([(sx1, ty), (sx2, ty + step * 0.7)], fill=0, width=1)


def _sv_nut(d, specs, x, y, w, h):
    cx = x + w // 2
    nut_h = max(6, int(h * 0.60))
    ny = y + (h - nut_h) // 2
    af = min(w - 4, int(min(w, h) // 2 * 1.8))
    nx = cx - af // 2
    c = max(1, nut_h // 5)
    hr = max(2, int(af * 0.22))
    d.polygon([
        (nx + c, ny), (nx + af - c, ny), (nx + af, ny + c),
        (nx + af, ny + nut_h - c), (nx + af - c, ny + nut_h),
        (nx + c, ny + nut_h), (nx, ny + nut_h - c), (nx, ny + c),
    ], outline=0)
    d.rectangle([cx - hr, ny, cx + hr, ny + nut_h], fill='white')
    d.line([(cx - hr, ny), (cx - hr, ny + nut_h)], fill=0, width=1)
    d.line([(cx + hr, ny), (cx + hr, ny + nut_h)], fill=0, width=1)


def _sv_washer(d, specs, x, y, w, h):
    cx = x + w // 2
    ro = max(5, min(w, h) // 2 - 2)
    ri = max(2, int(ro * 0.4))
    t = max(2, int(h * 0.35))
    sy = y + (h - t) // 2
    lx = cx - ro
    gap = ro - ri
    d.rectangle([lx, sy, lx + gap, sy + t], outline=0, width=1)
    d.rectangle([lx + gap + ri * 2, sy, lx + ro * 2, sy + t], outline=0, width=1)


def _sv_standoff(d, specs, x, y, w, h):
    cx = x + w // 2
    r = max(4, min(w, h) // 2 - 2)
    bh = int(h * 0.75)
    by = y + (h - bh) // 2
    hr = max(2, int(r * 0.35))
    st = specs.get('standoff_type', 'Hex F-F')
    d.rectangle([cx - r, by, cx + r, by + bh], outline=0, width=1)
    d.line([(cx - hr, by), (cx - hr, by + bh)], fill=0, width=1)
    d.line([(cx + hr, by), (cx + hr, by + bh)], fill=0, width=1)
    tl = max(4, bh // 5)
    sw2 = max(3, hr * 2)
    if 'M-F' in st:
        d.rectangle([cx - sw2 // 2, by - tl, cx + sw2 // 2, by], outline=0, width=1)
        d.line([(cx - sw2 // 2, by - tl + 1), (cx + sw2 // 2, by - 1)], fill=0, width=1)
    if 'M-M' in st:
        d.rectangle([cx - sw2 // 2, by + bh, cx + sw2 // 2, by + bh + tl], outline=0, width=1)
        d.line([(cx - sw2 // 2, by + bh + 1), (cx + sw2 // 2, by + bh + tl - 1)], fill=0, width=1)


def _sv_rivet(d, specs, x, y, w, h):
    cx = x + w // 2
    rt = specs.get('rivet_type', 'Blind/Pop')
    ht = specs.get('head_type', 'Dome')
    hh = max(5, int(h * 0.28))
    rw = max(4, min(w, h) - 8)
    hw = min(int(rw * 1.8), w - 4)
    sh = h - hh - 2
    hx1, hx2 = cx - hw // 2, cx + hw // 2
    sx1, sx2 = cx - rw // 2, cx + rw // 2
    if ht == 'Dome':
        d.pieslice([hx1, y - hh, hx2, y + hh], 180, 360, fill=0, outline=0)
    elif ht in ('Countersunk', 'Flat'):
        d.polygon([(hx1, y + hh), (hx2, y + hh), (sx2, y), (sx1, y)], outline=0)
    else:
        d.rectangle([hx1, y, hx2, y + hh], outline=0, width=1)
    d.rectangle([sx1, y + hh, sx2, y + hh + sh], outline=0, width=1)
    if rt == 'Blind/Pop':
        mr = max(1, rw // 4)
        d.rectangle([cx - mr, y, cx + mr, y + hh + sh + 3], fill=0)


def _sv_pin(d, specs, x, y, w, h):
    cx = x + w // 2
    r = max(3, min(w, h) // 2 - 3)
    d.rectangle([cx - r, y + 2, cx + r, y + h - 2], outline=0, width=1)
    if specs.get('pin_type', '') in ('Roll Pin (Spring)', 'Slotted'):
        d.line([(cx, y + 2), (cx, y + h - 2)], fill=0, width=1)


def _sv_circlip(d, specs, x, y, w, h):
    cx = x + w // 2
    ro = max(5, min(w, h) // 2 - 2)
    t = max(2, int(h * 0.30))
    sy = y + (h - t) // 2
    gp = max(3, int(ro * 0.25))
    gx = cx - gp // 2
    d.rectangle([cx - ro, sy, cx + ro, sy + t], outline=0, width=1)
    d.rectangle([gx, sy - 1, gx + gp, sy + t + 1], fill='white')
    d.line([(gx, sy), (gx, sy + t)], fill=0, width=1)
    d.line([(gx + gp, sy), (gx + gp, sy + t)], fill=0, width=1)


def _sv_insert(d, specs, x, y, w, h):
    cx = x + w // 2
    r = max(4, min(w, h) // 2 - 2)
    bh = int(h * 0.70)
    by = y + (h - bh) // 2
    hr = max(2, int(r * 0.35))
    d.rectangle([cx - r, by, cx + r, by + bh], outline=0, width=1)
    d.line([(cx - hr, by), (cx - hr, by + bh)], fill=0, width=1)
    d.line([(cx + hr, by), (cx + hr, by + bh)], fill=0, width=1)
    for i in range(1, 4):
        ky = by + bh * i // 4
        d.line([(cx - r, ky), (cx - r - 2, ky + 2)], fill=0, width=1)
        d.line([(cx + r, ky), (cx + r + 2, ky + 2)], fill=0, width=1)


_HW_TV = {
    'screw': _tv_screw, 'nut': _tv_nut, 'washer': _tv_washer,
    'standoff': _tv_standoff, 'rivet': _tv_rivet, 'pin': _tv_pin,
    'circlip': _tv_circlip, 'insert': _tv_insert,
}
_HW_SV = {
    'screw': _sv_screw, 'nut': _sv_nut, 'washer': _sv_washer,
    'standoff': _sv_standoff, 'rivet': _sv_rivet, 'pin': _sv_pin,
    'circlip': _sv_circlip, 'insert': _sv_insert,
}


def _draw_hw_diagram(d, hw_type, specs, x, y, w, h, tv=True, sv=True):
    tfn = _HW_TV.get(hw_type)
    sfn = _HW_SV.get(hw_type)
    if tv and sv:
        half = h // 2
        if tfn: tfn(d, specs, x, y + 1, w, half - 2)
        if sfn: sfn(d, specs, x, y + half + 1, w, half - 2)
        d.line([(x + 2, y + half), (x + w - 2, y + half)], fill=180, width=1)
        try:
            f7 = _hw_font(7)
            d.text((x + 1, y + 1), 'T', fill=160, font=f7)
            d.text((x + 1, y + half + 1), 'S', fill=160, font=f7)
        except Exception:
            pass
    elif tv and tfn:
        tfn(d, specs, x, y, w, h)
    elif sv and sfn:
        sfn(d, specs, x, y, w, h)


def _mat_ab(m: str) -> str:
    return {
        'A2-304 Stainless': 'A2', 'A4-316 Stainless': 'A4',
        'Carbon Steel': 'CS', 'Alloy Steel': 'AS',
        'Zinc Plated': 'ZnP', 'Black Oxide': 'BLK',
        'Brass': 'Brass', 'Nylon': 'Nylon', 'Titanium': 'Ti',
    }.get(m, m[:4] if m else '')


def _label_lines(hw_type: str, specs: dict, opts: dict) -> list:
    sm = opts.get('show_material', True)
    ss = opts.get('show_standard', True)
    sg = opts.get('show_grade', True)
    lines: list = []

    def mat_line():
        m = specs.get('material', '') or specs.get('body_material', '')
        g = specs.get('grade', '')
        if sm and m:
            lines.append(_mat_ab(m) + (' ' + g if sg and g else ''))

    if hw_type == 'screw':
        sz, ln = specs.get('thread_size', ''), specs.get('length_mm', '')
        head, drive = specs.get('head_type', ''), specs.get('drive', '')
        ha = {'Socket Cap': 'SHC', 'Button': 'BHC', 'Pan': 'PH', 'Countersunk': 'CSK',
              'Hex': 'HEX', 'Flange Hex': 'FHEX', 'Truss': 'TRU', 'Set Screw': 'SS',
              }.get(head, head[:3].upper() if head else '')
        lines.append(f"{sz}{'×' + str(ln) if ln else ''}")
        ts = ha + (' ' + drive[:3].upper() if drive and drive != 'None' else '') if ha else ''
        if ts: lines.append(ts)
        mat_line()
        if ss and specs.get('standard'): lines.append(specs['standard'])

    elif hw_type == 'nut':
        na = {'Hex': 'HEX', 'Hex Thin/Jam': 'JAM', 'Nyloc': 'NYL', 'Flanged': 'FLG',
              'Flanged Nyloc': 'FNL', 'Castle': 'CAS', 'Wing': 'WNG',
              'Cap/Dome': 'CAP', 'T-nut': 'T-NUT'}.get(specs.get('nut_type', ''))
        lines.append(specs.get('thread_size', ''))
        if na: lines.append(f"{na} NUT")
        mat_line()
        if ss and specs.get('standard'): lines.append(specs['standard'])

    elif hw_type == 'washer':
        wa = {'Plain (Form A)': 'FLAT', 'Plain (Form B/Large)': 'FLAT-L',
              'Split Lock': 'LOCK', 'Star Lock (Int)': 'STR-I',
              'Star Lock (Ext)': 'STR-E', 'Fender': 'FEND',
              'Spring': 'SPR', 'Belleville': 'BELL', 'Nord-Lock': 'NL',
              }.get(specs.get('washer_type', ''))
        lines.append(specs.get('fits_size', ''))
        if wa: lines.append(f"{wa} WSH")
        mat_line()
        if ss and specs.get('standard'): lines.append(specs['standard'])

    elif hw_type == 'standoff':
        sz, ln = specs.get('thread_size', ''), specs.get('length_mm', '')
        sa = {'Hex F-F': 'HFF', 'Hex M-F': 'HMF', 'Hex M-M': 'HMM',
              'Round F-F': 'RFF', 'PCB': 'PCB'}.get(specs.get('standoff_type', ''))
        lines.append(f"{sz}{' ' + str(ln) + 'mm' if ln else ''}")
        if sa: lines.append(f"{sa} STNDF")
        mat_line()

    elif hw_type == 'rivet':
        dia = specs.get('diameter_mm', '')
        ra = {'Blind/Pop': 'POP', 'Solid': 'SOLID', 'Drive': 'DRV',
              'Tubular': 'TUB'}.get(specs.get('rivet_type', ''))
        gn, gx_ = specs.get('grip_min', ''), specs.get('grip_max', '')
        lines.append(f"\u00d8{dia}mm" if dia else '')
        if ra: lines.append(f"{ra} RIVET")
        if gn or gx_: lines.append(f"grip {gn}-{gx_}mm")
        mat_line()

    elif hw_type == 'pin':
        dia, ln = specs.get('diameter_mm', ''), specs.get('length_mm', '')
        pa = {'Solid Cylindrical': 'DWLP', 'Roll Pin (Spring)': 'ROLL',
              'Slotted': 'SLOT'}.get(specs.get('pin_type', ''))
        tol = specs.get('tolerance', '')
        lines.append(f"\u00d8{dia}\u00d7{ln}mm" if dia and ln else f"\u00d8{dia}mm" if dia else '')
        if pa: lines.append(f"{pa} PIN")
        if tol: lines.append(tol.split(' ')[0])
        mat_line()

    elif hw_type == 'circlip':
        dia = specs.get('shaft_diameter_mm', '')
        ca = {'External': 'EXT', 'Internal': 'INT'}.get(specs.get('circlip_type', ''))
        lines.append(f"\u00d8{dia}mm" if dia else '')
        if ca: lines.append(f"{ca} CIRCLIP")
        mat_line()

    elif hw_type == 'insert':
        sz, ln = specs.get('thread_size', ''), specs.get('length_mm', '')
        ia = {'Heat-Set (Brass)': 'HEAT', 'Threaded': 'THRD',
              'Self-Tapping': 'ST'}.get(specs.get('install_method', ''))
        lines.append(f"{sz}{' ' + str(ln) + 'mm' if ln else ''}")
        if ia: lines.append(f"{ia} INSERT")
        mat_line()

    else:
        lines.append(hw_type.upper())

    return [l.strip() for l in lines if l and l.strip()]


def _make_hw_label(hw_type: str, specs: dict, length_mm: float, opts: dict) -> Image.Image:
    from PIL import ImageDraw as _ID
    w = max(LABEL_HEIGHT_PX, round(length_mm * LABEL_DPI / 25.4))
    h = LABEL_HEIGHT_PX
    img = Image.new('RGB', (w, h), 'white')
    d = _ID.Draw(img)

    tv   = opts.get('show_topview', True)
    sv_  = opts.get('show_sideview', True)
    diag = opts.get('show_diagram', True) and (tv or sv_)

    dw = 0
    if diag:
        dw = min(90, max(52, w // 4))
        _draw_hw_diagram(d, hw_type, specs, 2, 2, dw - 4, h - 4, tv, sv_)
        d.line([(dw, 2), (dw, h - 2)], fill=0, width=1)

    tx, tw = dw + 4, w - dw - 8
    lines = _label_lines(hw_type, specs, opts)
    if lines and tw > 10:
        yp = 5
        f_big = _hw_font(22)
        for sz in [22, 18, 14, 11]:
            f = _hw_font(sz)
            try:
                bb = d.textbbox((tx, yp), lines[0], font=f)
                if bb[2] - bb[0] <= tw:
                    f_big = f
                    break
            except Exception:
                f_big = f
                break
        d.text((tx, yp), lines[0], fill=0, font=f_big)
        try:
            yp = d.textbbox((tx, yp), lines[0], font=f_big)[3] + 3
        except Exception:
            yp += 25
        fs = _hw_font(11)
        for line in lines[1:]:
            if yp + 12 > h - 2:
                break
            d.text((tx, yp), line, fill=0, font=fs)
            try:
                yp = d.textbbox((tx, yp), line, font=fs)[3] + 2
            except Exception:
                yp += 13
    return img


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _make_view_img(hw_type: str, specs: dict, size: int = 200, which: str = 'top') -> Image.Image:
    from PIL import ImageDraw as _ID
    img = Image.new('RGB', (size, size), 'white')
    d = _ID.Draw(img)
    fn = (_HW_TV if which == 'top' else _HW_SV).get(hw_type)
    if fn:
        fn(d, specs, 8, 8, size - 16, size - 16)
    return img


# ── Hardware label endpoints ───────────────────────────────────────

@app.get("/api/labels/hardware/printer-test")
def hw_printer_test(printer: str):
    """Verify that brother_ql is installed and the printer address parses correctly."""
    try:
        from brother_ql.raster import BrotherQLRaster  # noqa: F401
        addr = printer if printer.startswith(('tcp://', 'usb://')) else f'tcp://{printer}'
        return {"ok": True, "printer": addr}
    except ImportError:
        raise HTTPException(status_code=500, detail="brother_ql not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/labels/hardware/generate")
def hw_label_generate(payload: dict):
    """Return a hardware label PNG (base64) plus 200×200 top/side view images."""
    try:
        hw_type = payload.get("type", "screw")
        specs   = payload.get("specs", {})
        length  = float(payload.get("label_length_mm", 36))
        opts    = payload.get("options", {})
        img    = _make_hw_label(hw_type, specs, length, opts)
        tv_img = _make_view_img(hw_type, specs, 200, 'top')
        sv_img = _make_view_img(hw_type, specs, 200, 'side')
        return {
            "image":    _img_to_b64(img),
            "topview":  _img_to_b64(tv_img),
            "sideview": _img_to_b64(sv_img),
            "width":    img.width,
            "height":   img.height,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/labels/hardware/print")
def hw_label_print(payload: dict):
    """Generate a hardware label and send it to a Brother QL-710W printer."""
    try:
        from brother_ql.raster import BrotherQLRaster
        from brother_ql.conversion import convert
        from brother_ql.backends.helpers import send
    except ImportError:
        raise HTTPException(status_code=500, detail="brother_ql not installed")
    try:
        hw_type = payload.get("type", "screw")
        specs   = payload.get("specs", {})
        length  = float(payload.get("label_length_mm", 36))
        opts    = payload.get("options", {})
        printer = payload.get("printer", "")
        copies  = max(1, int(payload.get("copies", 1)))
        if not printer:
            raise HTTPException(status_code=400, detail="printer address required")
        if not printer.startswith(('tcp://', 'usb://')):
            printer = f'tcp://{printer}'
        backend = 'network' if printer.startswith('tcp://') else 'pyusb'
        img = _make_hw_label(hw_type, specs, length, opts)
        qlr = BrotherQLRaster('QL-710W')
        qlr.exception_on_warning = True
        instr = convert(
            qlr=qlr, images=[img] * copies, label='12',
            rotate='0', threshold=70, dither=False,
            compress=False, red=False, dpi_600=False, hq=True, cut=True,
        )
        send(instructions=instr, printer_identifier=printer,
             backend_identifier=backend, blocking=True)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

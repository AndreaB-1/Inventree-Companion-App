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




# ------------------------------- ENV -------------------------------
INVENTREE_URL = os.getenv("INVENTREE_URL")  # e.g. http://192.168.1.110/api/
INVENTREE_TOKEN = os.getenv("INVENTREE_TOKEN")
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
        res = api.get("/plugin/", params={"active": True})
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


@app.get("/")
def index():
    return FileResponse("static/index.html")


# ----------------- Parts & Locations -----------------

@app.get("/api/part/by-ipn")
def part_by_ipn(ipn: str):
    try:
        api = inv_api()
        res = api.get("/part/", params={"IPN": ipn})
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
        p = api.get(f"/part/{pk}/")
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
        res = api.get("/part/", params=params)
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
        res = api.get("/stock/location/", params=params)
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

        created = api.post("/part/", data)
        if not isinstance(created, dict) or not created.get("pk"):
            raise HTTPException(status_code=400, detail=f"Unexpected response: {created}")

        part_pk = created["pk"]
        stock_msg = None

        if initial_qty and float(initial_qty) > 0:
            if not location_id:
                raise HTTPException(status_code=400, detail="location_id required when initial_qty is provided.")
            try:
                stock_payload = {"part": part_pk, "quantity": float(initial_qty), "location": int(location_id)}
                api.post("/stock/", stock_payload)
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
        p = api.get(f"/part/{pk}/")
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
        res = api.get("/stock/", params=params)
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
                        ld = api.get(f"/stock/location/{loc_id}/")
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
        api.post("/stock/add/", payload)
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
        api.post("/stock/remove/", payload)
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
        p = api.get(f"/part/{pk}/")
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
        p = api.get(f"/part/{pk}/")
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
        res = api.get("/label/template/", params=params)
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
    """List label-capable plugins (LabelPrintingMixin)."""
    try:
        api = inv_api()
        params: Dict[str, Any] = {"mixin": "label"}
        if active is not None:
            params["active"] = active
        res = api.get("/plugins/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        return [
            {
                "pk": r.get("pk"),
                "key": r.get("key"),
                "name": r.get("name"),
                "active": r.get("active"),
            }
            for r in rows if isinstance(r, dict)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PrintJob(BaseModel):
    template_id: int
    items: List[int]
    plugin_id: Optional[int] = None  # optional if you have a default


@app.post("/api/labels/print")
def print_labels(payload: dict):
    """
    Body: {
      "template_id": int,
      "items": [int, ...],
      "plugin_id": int | null   # optional; if missing we default to InvenTreeLabelMachine when available
    }
    """
    try:
        template_id = int(payload.get("template_id"))
        items = payload.get("items") or []
        plugin_id = payload.get("plugin_id")

        if not template_id or not isinstance(items, list) or not items:
            raise HTTPException(status_code=400, detail="template_id and non-empty items are required")

        api = inv_api()

        # 👇 default plugin to InvenTreeLabelMachine if client didn't send one
        if not plugin_id:
            plugin_id = _resolve_default_label_plugin_id(api)

        body = {
            "template": template_id,
            "items": items,
        }
        if plugin_id:
            body["plugin"] = int(plugin_id)  # InvenTree expects 'plugin' field

        res = api.post("/label/print/", body)
        # res may be a job dict; just mirror ok
        return {"ok": True, "result": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


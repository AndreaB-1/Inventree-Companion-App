# app.py
# FastAPI backend for InvenTree Companion (parts + kiosk + label printing)
# pip install: fastapi uvicorn pillow inventree python-multipart opencv-python-headless numpy openai

import os
import io
import re
import json
import tempfile
from io import BytesIO
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
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

# (Optional) AI suggest
from openai import OpenAI


# ------------------------------- ENV -------------------------------
INVENTREE_URL = os.getenv("INVENTREE_URL")
INVENTREE_TOKEN = os.getenv("INVENTREE_TOKEN")

if not INVENTREE_URL or not INVENTREE_TOKEN:
    raise RuntimeError("Set INVENTREE_URL and INVENTREE_TOKEN env vars.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# -------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # LAN-friendly; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


def inv_api():
    """Authenticated InvenTree API client."""
    return InvenTreeAPI(host=INVENTREE_URL, token=INVENTREE_TOKEN)


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
    """Search parts by name/IPN using DRF 'search'."""
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
                "thumbnail": r.get("thumbnail") or r.get("image"),
                "image": r.get("image"),
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


@app.get("/api/stock/by-part")
def stock_by_part(part: int, limit: int = 200, offset: int = 0):
    """List stock items for a part (for kiosk page)."""
    try:
        api = inv_api()
        params = {"part": part, "limit": limit, "offset": offset}
        res = api.get("/stock/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            out.append({
                "pk": r.get("pk"),
                "quantity": r.get("quantity"),
                "serial": r.get("serial"),
                "batch": r.get("batch"),
                "status": r.get("status"),
                "location": (r.get("location_detail") or {}).get("name") if r.get("location_detail") else None,
                "location_id": r.get("location"),
            })
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stock/adjust")
def stock_adjust(
    stock_id: int = Form(...),
    delta: float = Form(...),
    notes: Optional[str] = Form(None),
):
    """Adjust quantity of a stock item (positive add / negative remove)."""
    try:
        api = inv_api()
        payload = {"items": [{"pk": int(stock_id), "quantity": float(delta)}]}
        if notes:
            payload["notes"] = notes
        res = api.post("/stock/adjust/", payload)
        return {"ok": True, "response": res}
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


# ----------------- QR "Scan by Photo" -----------------

@app.post("/api/decode_qr")
async def decode_qr(image: UploadFile = File(...)):
    """
    Robust QR decode from a phone photo:
    1) Fix EXIF rotation
    2) Try ZBar (pyzbar) across multiple preprocess + scales + rotations
    3) Fallback to OpenCV QRCodeDetector variants
    """
    try:
        raw = await image.read()

        # Load with PIL first to respect EXIF orientation
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")

        # Convert to OpenCV BGR
        base = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Normalize size
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

        def parse_hits(hits):
            if not hits:
                return None
            for t in hits:
                try:
                    obj = json.loads(t)
                    if isinstance(obj, dict) and "part" in obj:
                        return t, obj
                except Exception:
                    pass
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
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 10)
            variants = [img, gray, clahe, thr]
            rotations = [0, 90, 180, 270]
            scales = [1.0, 1.5, 2.0] if max(img.shape[:2]) < 1200 else [1.0]
            results = []
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
                                txt = d.data.decode("utf-8", errors="ignore").strip()
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
                    texts, pts, _ = det.detectAndDecodeMulti(cv_img)
                    if texts:
                        hits = [t for t in texts if t]
                        if hits:
                            return hits
                except Exception:
                    pass
                try:
                    text, pts, _ = det.detectAndDecode(cv_img)
                    if text:
                        return [text]
                except Exception:
                    pass
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 10)

            variants = [img, gray, clahe, thr]
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

        out = try_pyzbar(base) or try_opencv(base)
        if out:
            text, payload = out
            return {"text": text, "payload": payload}
        raise HTTPException(status_code=400, detail="No QR detected")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- AI suggest (unified) -----------------

class AISuggestReq(BaseModel):
    prompt: str
    language: Optional[str] = None
    name_only: Optional[bool] = False
    strict: Optional[bool] = True


def _fallback_desc_from_name(name: str) -> str:
    return name


@app.post("/api/ai/suggest-part")
async def ai_suggest_part(req: AISuggestReq):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    client = OpenAI(api_key=api_key)
    model = OPENAI_MODEL
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
                        "size_kind": {"type": "string", "enum": ["metric_thread", "sheet_metal_thread", "washer", "nut", "other"]},
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
                        "gender": {"type": "string", "nullable": True, "enum": ["Male", "Female", "Male-Female", "None"]},
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

    system_rules = """
You create inventory-friendly names/descriptions for fasteners AND electronics/tools.
- Output valid JSON matching the provided schema.
- SHORT Name; compact single-line Description.
- US terminology.
- In strict mode, fill all fields of the chosen sub-object (null if N/A).

FASTENERS:
Name: "[SIZE] – [type_name] [material_short]"
Examples:
- "M3 × 22 – Socket Head Cap Screw A2"
- "ST 3.9 × 9.5 – Pan Head Tapping Screw Steel"
- "M4 – Split Spring Lock Washer Steel"
Description: "[SIZE] – [type_name] – [STANDARD or 'Non-Standard'] – [material_long]"

ELECTRONICS / TOOLS:
Follow examples like:
- "Transistor NPN Bipolar - BC548"
- "Diode LED White 5mm"
- "Connector JST XH 2.54mm 3P Male"
- "Display 7-Segment 0.4\" 4-Digit Red CA"
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
                "series": None, "brand": None, "pitch_mm": None, "pins": None, "gender":"None",
                "color": None, "size_inch": None, "dimensions": None, "protocol": None,
                "voltage_v": None, "frequency_hz": None, "capacity": None, "extras": None,
                "non_standard": False, "alt_standard": None
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
        desc = (data.get("render_description") or "").strip() or _fallback_desc_from_name(name)

        if not name:
            raise HTTPException(status_code=500, detail="AI returned empty name")

        if req.name_only:
            desc = _fallback_desc_from_name(name)

        return {"name": name, "description": desc, "structured": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")


# ----------------- Label Templates / Plugins / Print -----------------

@app.get("/api/labels/templates")
def label_templates(
    model_type: str = Query(..., regex="^(part|stockitem|stocklocation)$"),
    enabled: bool = True,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    Proxy to InvenTree label templates list.
    model_type: 'part' | 'stockitem' | 'stocklocation'
    """
    try:
        api = inv_api()
        params: Dict[str, Any] = {"model_type": model_type, "limit": limit, "offset": offset}
        if enabled is not None:
            params["enabled"] = enabled
        if search:
            params["search"] = search

        res = api.get("/label/template/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        out = []
        for r in rows:
            out.append({
                "pk": r.get("pk"),
                "name": r.get("name"),
                "description": r.get("description"),
                "model_type": r.get("model_type"),
                "enabled": r.get("enabled"),
            })
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plugins/label")
def plugins_with_label_mixin(active: Optional[bool] = True):
    """
    List label-capable plugins (LabelPrintingMixin), so user can select printer plugin.
    """
    try:
        api = inv_api()
        params: Dict[str, Any] = {"mixin": "label"}
        if active is not None:
            params["active"] = active
        res = api.get("/plugins/", params=params)
        rows = res["results"] if isinstance(res, dict) and "results" in res else res or []
        out = []
        for r in rows:
            out.append({
                "pk": r.get("pk"),
                "key": r.get("key"),
                "name": r.get("name"),
                "active": r.get("active"),
                "mixins": r.get("mixins"),
                "is_builtin": r.get("is_builtin"),
            })
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PrintJob(BaseModel):
    template_id: int
    items: List[int]
    plugin_id: Optional[int] = None


@app.post("/api/labels/print")
def labels_print(job: PrintJob):
    """
    Send a label print job through InvenTree (server routes it to the selected plugin).
    For Brother (via plugin), select the appropriate plugin_id from /api/plugins/label.
    """
    try:
        api = inv_api()
        payload = {
            "template": job.template_id,
            "items": job.items,
        }
        if job.plugin_id is not None:
            payload["plugin"] = job.plugin_id

        resp = api.post("/label/print/", payload)
        # The response is a "LabelPrint" object; InvenTree tracks jobs at /api/label/output/
        return {"ok": True, "response": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

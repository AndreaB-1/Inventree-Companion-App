# app.py
# FastAPI backend for InvenTree photo upload + QR "scan by photo" (no HTTPS needed)
# pip install: fastapi uvicorn pillow inventree python-multipart opencv-python-headless numpy

import os
import json
import tempfile
from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

# InvenTree
from inventree.api import InvenTreeAPI
from inventree.part import Part

# QR decode (photo)
import numpy as np
import cv2
import io
import json
import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile, File
import re
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

from pydantic import BaseModel
from openai import OpenAI


# -------------------------------
INVENTREE_URL = os.getenv("INVENTREE_URL")
INVENTREE_TOKEN = os.getenv("INVENTREE_TOKEN")
# -------------------------------

if not INVENTREE_TOKEN:
    raise RuntimeError("Set INVENTREE_TOKEN (hardcoded for dev or via env).")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep simple in LAN; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


def inv_api():
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
    """
    Search parts by name/IPN (DRF 'search').
    """
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
    """
    List stock locations (with optional text search).
    """
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

        # Keep reasonable size (upscale small inputs to help the decoder)
        def normalize_size(img):
            h, w = img.shape[:2]
            max_side = max(h, w)
            # Upscale small images; cap very large ones for speed
            if max_side < 900:
                scale = 900 / max_side
                return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            if max_side > 2000:
                scale = 2000 / max_side
                return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            return img

        base = normalize_size(base)

        # ---------- helpers ----------
        def parse_hits(hits):
            """Return (best_text, payload_dict_or_None) from a list of strings."""
            if not hits:
                return None
            # Prefer JSON with {"part": ...}
            for t in hits:
                try:
                    obj = json.loads(t)
                    if isinstance(obj, dict) and "part" in obj:
                        return t, obj
                except Exception:
                    pass
            # Next, prefer URL with ?part=/id=/ipn=
            for t in hits:
                m = re.search(r'(?:\?|#|&)(?:part|id|ipn)=([^&]+)', t, re.I)
                if m:
                    val = m.group(1)
                    try:
                        ival = int(val)
                        return t, {"part": ival}
                    except Exception:
                        return t, {"query": val}
            # Otherwise just return first non-empty text
            return hits[0], None

        def try_pyzbar(img):
            """Try pyzbar under multiple preprocess/scale/rotation variants."""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_clahe = clahe.apply(gray)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 10)

            variants = [
                img,                # original
                gray,               # grayscale
                gray_clahe,         # contrast enhanced
                thr,                # thresholded
            ]
            rotations = [0, 90, 180, 270]
            # If still small, try a couple of upscales
            h, w = img.shape[:2]
            scales = [1.0, 1.5, 2.0] if max(h, w) < 1200 else [1.0]

            results = []
            for v in variants:
                for ang in rotations:
                    if ang == 0:
                        rot = v
                    else:
                        rot = cv2.rotate(
                            v,
                            cv2.ROTATE_90_CLOCKWISE if ang == 90 else
                            cv2.ROTATE_180 if ang == 180 else
                            cv2.ROTATE_90_COUNTERCLOCKWISE
                        )
                    for s in scales:
                        if s != 1.0:
                            hh, ww = rot.shape[:2]
                            rot_s = cv2.resize(rot, (int(ww*s), int(hh*s)), interpolation=cv2.INTER_CUBIC)
                        else:
                            rot_s = rot

                        # pyzbar handles both gray and color inputs; restrict to QR
                        dec = zbar_decode(rot_s, symbols=[ZBarSymbol.QRCODE])
                        if dec:
                            # collect unique strings (decode bytes safely)
                            for d in dec:
                                try:
                                    txt = d.data.decode("utf-8", errors="ignore").strip()
                                except Exception:
                                    txt = str(d.data)
                                if txt:
                                    results.append(txt)
                            if results:
                                # short-circuit on first success
                                best = parse_hits(results)
                                if best:
                                    return best
            return None

        def try_opencv(img):
            """Fallback: OpenCV QRCodeDetector (single + multi)."""
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
                        if s != 1.0:
                            hh, ww = rot.shape[:2]
                            rot_s = cv2.resize(rot, (int(ww*s), int(hh*s)), interpolation=cv2.INTER_CUBIC)
                        else:
                            rot_s = rot
                        hits = decode_any(rot_s)
                        if hits:
                            return parse_hits(hits)
            return None

        # 1) Prefer pyzbar (ZBar)
        out = try_pyzbar(base)
        if out:
            text, payload = out
            return {"text": text, "payload": payload}

        # 2) Fallback: OpenCV detector
        out = try_opencv(base)
        if out:
            text, payload = out
            return {"text": text, "payload": payload}

        # 3) (Optional) WeChat detector can be added if you want — ask me to wire it.
        raise HTTPException(status_code=400, detail="No QR detected")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- AI suggest endpoint (unified fastener + electronics) ---

class AISuggestReq(BaseModel):
    prompt: str
    language: Optional[str] = None      # e.g. "en" or "it"
    name_only: Optional[bool] = False   # frontend can request only the Name

def _fallback_desc_from_name(name: str) -> str:
    # If we get a short electronics name, mirror a compact description.
    # (You can expand this later.)
    return name

@app.post("/api/ai/suggest-part")
async def ai_suggest_part(req: AISuggestReq):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # One top-level schema: model MUST return render_name/description,
    # and can include a structured sub-object for either domain.
    schema = {
        "name": "inventory_unified",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "domain": {"type": "string", "enum": ["fastener", "electronics"]},
                "render_name": { "type": "string", "description": "Final Name string to use" },
                "render_description": { "type": "string", "description": "Final Description string to use" },

                "fastener": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "size_kind": { "type": "string", "enum": ["metric_thread", "sheet_metal_thread", "washer", "nut", "other"] },
                        "diameter_mm": { "type": "number" },
                        "length_mm": { "type": "number", "nullable": True },
                        "pitch_mm": { "type": "number", "nullable": True },
                        "standard": { "type": "string" },
                        "type_name": { "type": "string" },
                        "head": { "type": "string", "nullable": True },
                        "drive": { "type": "string", "nullable": True },
                        "material_short": { "type": "string" },
                        "material_long": { "type": "string" },
                        "property_class": { "type": "string", "nullable": True },
                        "finish": { "type": "string", "nullable": True },
                        "af_mm": { "type": "number", "nullable": True },
                        "non_standard": { "type": "boolean" },
                        "alt_standard": { "type": "string", "nullable": True },
                        "notes": { "type": "string", "nullable": True }
                    }
                },

                "electronics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": { "type": "string" },     # e.g., Transistor, MOSFET, IC, Connector, Display, Module, Tool, Bearing, Sensor, Adapter, Cable
                        "type_name": { "type": "string" },    # e.g., "Transistor NPN Bipolar", "RJ45 CAT6 Connector"
                        "part_number": { "type": "string", "nullable": True },
                        "series": { "type": "string", "nullable": True },   # e.g., JST XH, JST SH
                        "brand": { "type": "string", "nullable": True },
                        "pitch_mm": { "type": "number", "nullable": True },
                        "pins": { "type": "integer", "nullable": True },
                        "gender": { "type": "string", "nullable": True, "enum": ["Male", "Female", "Male-Female", "None"] },
                        "color": { "type": "string", "nullable": True },
                        "size_inch": { "type": "string", "nullable": True },    # e.g., 0.36", 4"
                        "dimensions": { "type": "string", "nullable": True },   # e.g., 8x22x7mm, 32x32mm
                        "protocol": { "type": "string", "nullable": True },     # Zigbee, Wi-Fi, RF 433MHz
                        "voltage_v": { "type": "string", "nullable": True },
                        "frequency_hz": { "type": "string", "nullable": True },
                        "capacity": { "type": "string", "nullable": True },
                        "extras": { "type": "string", "nullable": True },       # "Pass-Through", "Tool-Less", "w/Button", "Shielded", etc.
                        "non_standard": { "type": "boolean", "default": False },
                        "alt_standard": { "type": "string", "nullable": True }
                    }
                }
            },
            "required": ["domain", "render_name", "render_description"]
        }
    }

    lang = (req.language or "en").strip()

    # --- SYSTEM PROMPT (rules distilled from both of your chats) ---
    system_rules = f"""
You create inventory-friendly names/descriptions for fasteners AND electronics/tools.

GENERAL:
- Output must be valid JSON that matches the provided JSON schema.
- ONE clean result. No duplicates. No code fences in your output.
- Keep Names SHORT and readable. Description compact (single line), not marketing copy.
- Use US terminology (e.g., "Adjustable Wrench" not "Spanner").

FASTENERS — NAME & DESCRIPTION STYLE:
- Name: "[SIZE] – [type_name] [material_short]"
  • metric_thread: "M{{d}} × {{L}}" (include pitch only if explicitly non-coarse and helpful)
  • sheet_metal_thread: "ST {{d}} × {{L}}"
  • washer/nut: "M{{d}}"
  • Examples:
    - "M3 × 22 – Socket Head Cap Screw A2"
    - "ST 3.9 × 9.5 – Pan Head Tapping Screw Steel"
    - "M4 – Split Spring Lock Washer Steel"
- Description: "[SIZE] – [type_name] – [STANDARD or 'Non-Standard'] – [material_long]"
  • Include finish in material_long when known (e.g., "Zinc Plated Steel").
- Classify standards consistently:
  • DIN 912 → Socket Head Cap Screw
  • DIN 933 (ISO 4017) → Hexagon Head Bolt (full thread)
  • DIN 931 (ISO 4014) → Hexagon Head Bolt (partial thread)
  • DIN 985 → Nylon-Insert Lock Nut
  • ISO 7089 (DIN 125-A) → Plain Washer
  • DIN 7980 → Split Spring Lock Washer
  • DIN 7997 → Countersunk Wood Screw
  • DIN 571 → Hex Head Wood Screw (coach screw). If AF known, you may note it.
  • DIN 7981 / ISO 7049 → Pan Head Tapping Screw (ST thread)
  • Truss head machine screws → Non-Standard (alt_standard: "ASME B18.6.3 Truss Head")
  • Hex head masonry screws (e.g., TechFast) → Non-Standard (alt_standard: brand or "Masonry Screw")

ELECTRONICS/TOOLS — NAME STYLE (examples from user’s corpus):
- Discrete semis: "Transistor NPN Bipolar - BC548", "Transistor PNP Bipolar - 2N3906", "MOSFET N-Channel Enhancement Mode - 2N7000"
- Logic/ICs: "IC Quad AND Gate CD74HC08E", "Comparator LM311", "ADC 24-Bit Delta-Sigma LTC2442CG", "Shift Register 8-Bit SN74HC595N"
- Regulators/Refs: "Voltage Regulator AMS1117-3.3", "Voltage Reference 3.3V NCP51460 SOT-23-3", "Voltage Reference TL431A Adjustable"
- Opto: "Optocoupler PC817"
- LEDs/Diodes: Put "Diode" first → "Diode LED White 5mm"
- Connectors:
  • JST: "Connector JST XH 2.54mm 3P Male" (or Female / Male-Female). Keep pitch and pin count.
  • Dupont: "Connector Dupont 2.54mm 1P Plastic Shell", "Terminal Crimp JST XH 2.54mm"
  • RJ45: "RJ45 CAT6 Shielded Pass-Through Connector", "Keystone Jack RJ45 CAT6 Tool-Less White (6pcs)"
  • Coax: "Coaxial Connector F Male RG6/U Twist-On"
- Displays:
  • 7-segment: "Display 7-Segment 0.4\" 4-Digit Red CA" (CA/CC abbreviation)
  • Dot matrix: "Display Dot Matrix 8x8 3.0mm Red 32x32mm"
  • Bar graph: "Display LED Bar Graph 10-Segment Red BAS10251A"
- Modules/Boards:
  • "WiFi Module ESP8266MOD AI-Thinker", "WiFi Board ESP32-S2 Mini 4MB 2MB PSRAM"
  • "RTC Module DS3231", "Adapter Module I2C LCD PCF8574"
  • "Universal IR Blaster Wi-Fi RM4C Mini (Bestcon)"
  • "mmWave Presence Sensor ZY-M100-24G Wall-Mount (Zigbee)"
  • "Smart Power Switch Wi-Fi SONOFF POWR316"
- Networking/Powerline: "Powerline Adapter TP-Link AV1000"
- Tools: "Adjustable Wrench 4\"", "Utility Knife - Trojan"
- Bearings:
  • "Bearing 608ZZ 8x22x7mm", "Bearing V-Groove 3x12x4mm"
- Rivets: "Pop Rivet Aluminum Dome 3.15mm x 12mm Grip"
- Short commodity sets: "Springs Assorted", "3D Printer Nozzles Assorted"

FORMAT RULES FOR ELECTRONICS:
- Keep names as short as the examples. If a part number exists, place it at the end (with a dash or within parentheses) depending on the example style.
- For LEDs, put 'Diode' first → "Diode LED White 5mm".
- For connectors: include pitch (mm), pins (P) and gender (Male/Female) when known.
- For displays: include size (inches), digits, color, and CA/CC if given.
- Use US terms (e.g., Wrench). Avoid duplicates. Single line only.

OUTPUT:
- Choose domain = "fastener" or "electronics".
- ALWAYS fill "render_name" and "render_description".
  • If you cannot craft an expanded Description, mirror the Name in "render_description".
- Also fill the relevant sub-object ("fastener" or "electronics") with useful fields.
"""

    # A couple of minimal few-shot anchors (we keep it short; rules above do the heavy lifting)
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
                "non_standard":False,"alt_standard":None
            }
        })}
    ]

    user_msg = (
        "Prompt:\n" + req.prompt.strip() + "\n\n"
        "Return ONLY JSON matching the schema."
    )

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

        # Safety: ensure we always have both fields
        name = (data.get("render_name") or "").strip()
        desc = (data.get("render_description") or "").strip()
        if not name:
            raise HTTPException(status_code=500, detail="AI returned empty name")

        if not desc:
            desc = _fallback_desc_from_name(name)

        # Frontend requested a name only?
        if req.name_only:
            desc = _fallback_desc_from_name(name)

        return {
            "name": name,
            "description": desc,
            "structured": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")



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
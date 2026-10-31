"""
Minimal in-memory mock of the InvenTree REST API — just enough for the
companion app to be exercised end-to-end without a real InvenTree server.

Run:  uvicorn tests.mock_inventree:app --port 8001
Then point the app at it:  INVENTREE_URL=http://localhost:8001/api/  INVENTREE_TOKEN=test
"""
import io
import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

app = FastAPI(title="Mock InvenTree")
_lock = threading.Lock()

LOCATIONS: Dict[int, Dict[str, Any]] = {
    1: {"pk": 1, "name": "Workshop", "pathstring": "Workshop", "parent": None},
    2: {"pk": 2, "name": "Shelf A", "pathstring": "Workshop/Shelf A", "parent": 1},
    3: {"pk": 3, "name": "Bin A1", "pathstring": "Workshop/Shelf A/Bin A1", "parent": 2},
    4: {"pk": 4, "name": "Bin A2", "pathstring": "Workshop/Shelf A/Bin A2", "parent": 2},
    5: {"pk": 5, "name": "Electronics Drawer", "pathstring": "Office/Electronics Drawer", "parent": None},
}

_SEED = [
    ("M3 × 12 – Socket Head Cap Screw A2", "M3 × 12 – Socket Head Cap Screw – DIN 912 – A2 Stainless Steel", "SCR-M3-12"),
    ("M3 × 22 – Socket Head Cap Screw A2", "M3 × 22 – Socket Head Cap Screw – DIN 912 – A2 Stainless Steel", "SCR-M3-22"),
    ("M4 Hex Nut A2", "M4 – Hex Nut – ISO 4032 – A2 Stainless Steel", "NUT-M4"),
    ("M4 Washer A2", "M4 – Plain Washer – ISO 7089 – A2 Stainless Steel", "WSH-M4"),
    ("m5 button head screw stainless", "button head m5x10", ""),
    ("Transistor NPN Bipolar - BC548", "Transistor NPN Bipolar - BC548", "TR-BC548"),
    ("Resistor 10k 1/4W", "Resistor 10kΩ 1/4W 1% metal film", "RES-10K"),
    ("RJ45 CAT6 Shielded Pass-Through", "RJ45 Connector CAT6 Shielded Pass-Through", ""),
]
PARTS: Dict[int, Dict[str, Any]] = {}
STOCK: Dict[int, Dict[str, Any]] = {}
IMAGES: Dict[int, bytes] = {}
PRINTS: List[Dict[str, Any]] = []
_next = {"part": 1, "stock": 1}


def _new_part(name, description="", ipn=""):
    pk = _next["part"]; _next["part"] += 1
    PARTS[pk] = {"pk": pk, "name": name, "description": description, "IPN": ipn or None,
                 "units": "", "active": True, "image": None, "thumbnail": None,
                 "category_detail": {"pk": 1, "name": "Fasteners", "pathstring": "Hardware/Fasteners"}}
    return PARTS[pk]


def _new_stock(part, qty, loc, batch=None):
    pk = _next["stock"]; _next["stock"] += 1
    STOCK[pk] = {"pk": pk, "part": part, "quantity": float(qty), "location": loc, "batch": batch, "status": 10, "serial": None}
    return STOCK[pk]


def _seed():
    for n, d, i in _SEED:
        _new_part(n, d, i)
    for i in range(20):  # filler parts to exercise paging
        _new_part(f"Spacer Nylon M3 × {i + 3}", f"Nylon spacer {i + 3}mm", f"SPC-M3-{i + 3}")
    _new_stock(1, 120, 3); _new_stock(1, 40, 4, "B-2024")
    _new_stock(2, 75, 3); _new_stock(3, 300, 4); _new_stock(6, 25, 5)


_seed()


def _in_stock(pk):
    return sum(s["quantity"] for s in STOCK.values() if s["part"] == pk)


def _part_out(p):
    return {**p, "in_stock": _in_stock(p["pk"])}


def _page(rows, request: Request):
    q = request.query_params
    if "limit" in q:
        lim = int(q.get("limit") or 100); off = int(q.get("offset") or 0)
        return {"count": len(rows), "next": None, "previous": None, "results": rows[off:off + lim]}
    return rows


async def _body(request: Request) -> Dict[str, Any]:
    ct = request.headers.get("content-type", "")
    if "multipart" in ct or "form" in ct:
        form = await request.form()
        return {k: v for k, v in form.items()}
    try:
        return await request.json()
    except Exception:
        return {}


@app.get("/api/")
def root():
    return {"server": "InvenTree", "version": "0.16.0", "apiVersion": 300, "worker_running": True}


@app.get("/api/user/me/")
def me():
    return {"pk": 1, "username": "demo"}


# ── Parts ──
@app.get("/api/part/")
def parts(request: Request, search: Optional[str] = None, IPN: Optional[str] = None):
    rows = list(PARTS.values())
    if search:
        s = search.lower()
        rows = [p for p in rows if s in (p["name"] or "").lower() or s in (p["IPN"] or "").lower() or s in (p["description"] or "").lower()]
    if IPN is not None:
        rows = [p for p in rows if (p["IPN"] or "") == IPN]
    return _page([_part_out(p) for p in rows], request)


@app.get("/api/part/{pk:int}/")
def part(pk: int):
    if pk not in PARTS:
        raise HTTPException(404, "Not found")
    return _part_out(PARTS[pk])


@app.post("/api/part/", status_code=201)
async def part_create(request: Request):
    d = await _body(request)
    if not d.get("name"):
        raise HTTPException(400, {"name": ["This field is required."]})
    with _lock:
        p = _new_part(d["name"], d.get("description", ""), d.get("IPN", ""))
    return _part_out(p)


@app.patch("/api/part/{pk:int}/")
async def part_patch(pk: int, request: Request):
    if pk not in PARTS:
        raise HTTPException(404, "Not found")
    d = await _body(request)
    p = PARTS[pk]
    img = d.pop("image", None)
    if img is not None and hasattr(img, "read"):
        IMAGES[pk] = await img.read()
        p["image"] = f"/media/part_images/part_{pk}.jpg"
        p["thumbnail"] = f"/media/part_images/part_{pk}.thumbnail.jpg"
    for k in ("name", "description", "IPN"):
        if k in d:
            p[k] = d[k]
    return _part_out(p)


@app.get("/media/part_images/{fname}")
def media(fname: str):
    try:
        pk = int(fname.split("_")[1].split(".")[0])
    except Exception:
        raise HTTPException(404)
    if pk not in IMAGES:
        raise HTTPException(404)
    return Response(IMAGES[pk], media_type="image/jpeg")


# ── Stock ──
@app.get("/api/stock/")
def stock(request: Request, part: Optional[int] = None):
    rows = [s for s in STOCK.values() if part is None or s["part"] == part]
    return _page(rows, request)


@app.get("/api/stock/{pk:int}/")
def stock_one(pk: int):
    if pk not in STOCK:
        raise HTTPException(404, "Not found")
    return STOCK[pk]


@app.post("/api/stock/", status_code=201)
async def stock_create(request: Request):
    d = await _body(request)
    with _lock:
        s = _new_stock(int(d["part"]), float(d["quantity"]), int(d["location"]) if d.get("location") else None)
    return s


def _items(d):
    out = []
    for it in d.get("items", []):
        pk = int(it["pk"])
        if pk not in STOCK:
            raise HTTPException(400, f"Stock item {pk} not found")
        out.append((STOCK[pk], float(it["quantity"]), it))
    return out


@app.post("/api/stock/add/", status_code=201)
async def stock_add(request: Request):
    for s, q, _ in _items(await _body(request)):
        s["quantity"] += q
    return {"success": True}


@app.post("/api/stock/remove/", status_code=201)
async def stock_remove(request: Request):
    for s, q, _ in _items(await _body(request)):
        s["quantity"] = max(0.0, s["quantity"] - q)
    return {"success": True}


@app.post("/api/stock/transfer/", status_code=201)
async def stock_transfer(request: Request):
    d = await _body(request)
    for s, q, it in _items(d):
        dest = int(it.get("location") or d.get("location"))
        if dest not in LOCATIONS:
            raise HTTPException(400, "Invalid location")
        if q > s["quantity"]:
            raise HTTPException(400, "Quantity exceeds stock")
        if q >= s["quantity"]:
            s["location"] = dest
        else:
            s["quantity"] -= q
            with _lock:
                _new_stock(s["part"], q, dest, s.get("batch"))
    return {"success": True}


@app.get("/api/stock/location/")
def locations(request: Request, search: Optional[str] = None):
    rows = list(LOCATIONS.values())
    if search:
        rows = [l for l in rows if search.lower() in l["pathstring"].lower()]
    return _page(rows, request)


@app.get("/api/stock/location/{pk}/")
def location(pk: int):
    if pk not in LOCATIONS:
        raise HTTPException(404)
    return LOCATIONS[pk]


# ── Labels / plugins ──
TEMPLATES = [
    {"pk": 11, "name": "Part QR 12mm", "model_type": "part", "description": "", "enabled": True},
    {"pk": 12, "name": "Part big", "model_type": "part", "description": "", "enabled": True},
    {"pk": 21, "name": "Stock item QR", "model_type": "stockitem", "description": "", "enabled": True},
    {"pk": 31, "name": "Location", "model_type": "stocklocation", "description": "", "enabled": True},
]


@app.get("/api/label/template/")
def templates(request: Request, model_type: Optional[str] = None):
    rows = [t for t in TEMPLATES if not model_type or t["model_type"] == model_type]
    return _page(rows, request)


@app.get("/api/plugin/")
def plugins(request: Request):
    rows = [{"pk": 1, "key": "inventreelabelmachine", "name": "InvenTreeLabelMachine", "active": True},
            {"pk": 2, "key": "inventreelabel", "name": "InvenTreeLabel", "active": True}]
    return _page(rows, request)


@app.post("/api/label/print/", status_code=201)
async def label_print(request: Request):
    d = await _body(request)
    PRINTS.append(d)
    return {"pk": len(PRINTS), "complete": True, "output": None}


@app.get("/_mock/prints")
def prints():
    return PRINTS

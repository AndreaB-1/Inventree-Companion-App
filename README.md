# InvenTree Companion App

A phone-first companion web app (PWA) for [InvenTree](https://inventree.org): scan a label, adjust or move stock,
create parts with AI-generated names, and print Brother QL hardware labels.

## What it does

| Area | Features |
|---|---|
| **Home** | Search, quick actions, recently viewed parts |
| **Scan** | Photo of a QR label → opens the part (`{"part": id}`, InvenTree `stockitem` codes and part/stock URLs). Live camera scanning where the browser allows (HTTPS). Manual lookup by #ID / IPN / name |
| **Search** | All-words search (any order), grid or list view, stock totals, infinite scroll |
| **Part** | Image (tap to zoom, camera upload, web image search), edit name/IPN/description (with AI suggestion), stock items with +/− adjust, move (full or split), new stock item, InvenTree label printing (part / stock items / locations, template + plugin choice) |
| **New part** | AI suggest (name-only / strict), IPN, description, initial quantity + location, photo, then optional label print and web image search |
| **Label maker** | 9 hardware types with spec forms, metric auto-fill (pitch, AF, washer dims), standard → head/drive auto-fill, live 12 mm tape preview, top/side views, QR link to a part (or create one from the specs), sideways IPN, cut margin, copies, PNG download, Brother QL-710W network print + connection test |
| **Tidy names** | Batch AI review of part names/descriptions with accept / skip / accept-all |
| **Settings** | Theme, post-create behaviour, default part label template, printer address & cut margin, editable thread sizes / standards / materials |

## Configuration

| Variable | Required | Description |
|---|---|---|
| `INVENTREE_URL` | yes | e.g. `http://192.168.1.110/api/` |
| `INVENTREE_TOKEN` | yes | InvenTree API token |
| `INVENTREE_LABEL_PLUGIN_KEY` | no | Label plugin key (default `inventreelabelmachine`, auto-detected otherwise) |
| `INVENTREE_LABEL_MACHINE_ID` | no | UUID of the label machine |
| `OPENAI_API_KEY` | no | Enables AI naming |
| `OPENAI_MODEL` | no | Default `gpt-4o-mini` |
| `DATA_DIR` | no | Where `hw_config.json` is stored — mount a volume here so the hardware lists survive updates |

## Run

```bash
cp .env.example .env   # fill in your values
docker compose up -d
```

## Test a local build before publishing

`docker-compose.dev.yml` builds the image from this checkout (port **8570**, so it can run next to the published one).

Against your real InvenTree (reads `.env`):

```bash
docker compose -f docker-compose.dev.yml up --build web
```

Fully offline with a fake InvenTree (no `.env` needed):

```bash
docker compose -f docker-compose.dev.yml --profile mock up --build
```

Browser smoke test of every page (needs Node + Playwright, run against the mock stack):

```bash
node tests/e2e_smoke.js http://localhost:8570
```

## Layout

```
app.py                 FastAPI backend (InvenTree proxy, QR decode, AI, label rendering/printing)
static/index.html      App shell
static/css/app.css     Styles (light/dark)
static/js/app.js       Navigation + hash router
static/js/core.js      API client, UI helpers (sheets, toasts, pickers, image search)
static/js/hardware.js  Hardware type definitions for the label maker
static/js/pages/*.js   One module per page
tests/                 Mock InvenTree + e2e smoke test
```

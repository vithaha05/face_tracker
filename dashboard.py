"""
dashboard.py — Read-only Flask web dashboard for the Face Tracker system.

Run:  python3 dashboard.py
Open: http://localhost:5050
"""

import os
import sqlite3
import json
from datetime import datetime, timezone, date
from flask import Flask, Response, send_from_directory, request, abort

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def _load_cfg():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

CFG      = _load_cfg()
DB_PATH  = os.path.join(BASE_DIR, CFG.get("db_path",  "faces_db/faces.db"))
LOGS_DIR = os.path.join(BASE_DIR, CFG.get("log_dir",  "logs"))

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# DB helpers  (read-only — never touch existing modules)
# ---------------------------------------------------------------------------
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_query(sql, params=()):
    try:
        with _conn() as c:
            return c.execute(sql, params).fetchall()
    except Exception:
        return []

def db_scalar(sql, params=(), default=0):
    try:
        with _conn() as c:
            row = c.execute(sql, params).fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception:
        return default

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _img_url(image_path):
    """Convert absolute DB image_path → Flask-servable /logs/... URL."""
    if not image_path:
        return None
    image_path = os.path.normpath(image_path)
    logs_norm  = os.path.normpath(LOGS_DIR)
    if image_path.startswith(logs_norm):
        rel = image_path[len(logs_norm):].lstrip(os.sep)
    else:
        marker = "logs" + os.sep
        idx = image_path.find(marker)
        if idx == -1:
            return None
        rel = image_path[idx + len(marker):]
    if not os.path.exists(image_path):
        return None
    return "/logs/" + rel.replace(os.sep, "/")

def _first_img_for_face(face_id):
    row = db_query(
        "SELECT image_path FROM events "
        "WHERE face_id=? AND event_type='entry' "
        "AND image_path IS NOT NULL AND image_path != '' "
        "ORDER BY timestamp ASC LIMIT 1",
        (face_id,)
    )
    return _img_url(row[0]["image_path"]) if row else None

# ---------------------------------------------------------------------------
# HTML helpers  (plain string concat — no Jinja2, avoids CSS-brace conflicts)
# ---------------------------------------------------------------------------
CSS = """
<style>
  :root {
    --accent: #e94560;
    --card-bg: #16213e;
    --dark-bg: #0a0a1a;
    --muted: #8892b0;
  }
  body { background: var(--dark-bg); color: #ccd6f6; min-height: 100vh; }
  .stat-card {
    background: var(--card-bg);
    border: 1px solid rgba(233,69,96,.25);
    border-radius: 16px;
    text-align: center;
    padding: 2rem 1.5rem;
    transition: transform .2s, box-shadow .2s;
  }
  .stat-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(233,69,96,.3); }
  .stat-number {
    font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(135deg,#e94560,#a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .stat-label { color: var(--muted); font-size: .9rem; letter-spacing: 1px; text-transform: uppercase; }
  table { color: #ccd6f6; }
  .tbl { background: var(--card-bg); border-radius: 12px; overflow: hidden; }
  .tbl th {
    background: rgba(233,69,96,.15); color: var(--accent);
    font-size: .8rem; letter-spacing: 1.5px; text-transform: uppercase; border: none !important;
  }
  .tbl td { border-color: rgba(255,255,255,.05) !important; vertical-align: middle; }
  .tbl tr:hover td { background: rgba(255,255,255,.03); }
  .badge-entry {
    background: linear-gradient(135deg,#00b09b,#96c93d);
    color: #fff; border-radius: 8px; padding: .35em .75em; font-size: .75rem; letter-spacing: .5px;
  }
  .badge-exit {
    background: linear-gradient(135deg,#e94560,#c0392b);
    color: #fff; border-radius: 8px; padding: .35em .75em; font-size: .75rem; letter-spacing: .5px;
  }
  .thumb { width:52px; height:52px; object-fit:cover; border-radius:8px; border:2px solid rgba(233,69,96,.4); }
  .no-thumb {
    width:52px; height:52px; border-radius:8px;
    background:rgba(255,255,255,.06);
    display:inline-flex; align-items:center; justify-content:center;
    font-size:1.4rem; color:rgba(255,255,255,.2);
  }
  .face-card {
    background: var(--card-bg);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 16px; overflow: hidden;
    transition: transform .2s, box-shadow .2s;
  }
  .face-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(233,69,96,.25); }
  .face-card img { width:100%; height:160px; object-fit:cover; }
  .face-card .no-img {
    width:100%; height:160px;
    background: rgba(255,255,255,.04);
    display:flex; align-items:center; justify-content:center;
    font-size:3rem; color:rgba(255,255,255,.15);
  }
  .face-card .card-body { padding: 1rem; }
  .face-id { font-family:monospace; font-size:.8rem; color:var(--accent); word-break:break-all; }
  .sh { font-size:1.6rem; font-weight:700;
    background:linear-gradient(90deg,#e94560,#a855f7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-bottom:1.5rem; }
  .ubadge { background:rgba(255,255,255,.07); border-radius:20px; padding:.3rem .9rem;
    font-size:.8rem; color:var(--muted); }
  .filter-btn.af { background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important; }
  a { color:#ccd6f6; text-decoration:none; }
  a:hover { color:var(--accent); }
</style>
"""

def _nav(active):
    def _cls(key):
        return "active" if active == key else ""
    return (
        '<nav class="navbar navbar-expand-lg navbar-dark" '
        'style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);'
        'box-shadow:0 2px 20px rgba(0,0,0,.5);">'
        '<div class="container-fluid">'
        '<a class="navbar-brand fw-bold" href="/" style="letter-spacing:1px;">'
        '<span style="color:#e94560;">&#9679;</span>&nbsp;FaceTracker</a>'
        '<button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nb">'
        '<span class="navbar-toggler-icon"></span></button>'
        '<div class="collapse navbar-collapse" id="nb">'
        '<ul class="navbar-nav ms-auto">'
        '<li class="nav-item">'
        '<a class="nav-link ' + _cls("dash")   + '" href="/">&#128202; Dashboard</a></li>'
        '<li class="nav-item">'
        '<a class="nav-link ' + _cls("faces")  + '" href="/faces">&#128100; Faces</a></li>'
        '<li class="nav-item">'
        '<a class="nav-link ' + _cls("events") + '" href="/events">&#128203; Events</a></li>'
        '</ul></div></div></nav>'
    )

def _page(title, body, active, refresh=False):
    refresh_tag = '<meta http-equiv="refresh" content="5">' if refresh else ""
    html = (
        "<!DOCTYPE html>"
        '<html lang="en"><head>'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        + refresh_tag
        + "<title>" + title + " \u2014 FaceTracker</title>"
        + '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">'
        + CSS
        + "</head><body>"
        + _nav(active)
        + '<div class="container-fluid px-4 py-4">'
        + body
        + "</div>"
        + '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>'
        + "</body></html>"
    )
    return Response(html, mimetype="text/html")

def _badge(event_type):
    if event_type == "entry":
        return '<span class="badge-entry">&#x25B2; ENTRY</span>'
    return '<span class="badge-exit">&#x25BC; EXIT</span>'

def _thumb(img_url):
    if img_url:
        return '<img src="' + img_url + '" class="thumb" alt="face">'
    return '<span class="no-thumb">&#128100;</span>'

# ---------------------------------------------------------------------------
# Route: Log images
# ---------------------------------------------------------------------------
@app.route("/logs/<path:filename>")
def serve_log_image(filename):
    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.isfile(filepath):
        abort(404)
    return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))

# ---------------------------------------------------------------------------
# Route: Dashboard  /
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    today   = date.today().isoformat()

    unique  = db_scalar("SELECT COUNT(DISTINCT face_id) FROM events")
    entries = db_scalar(
        "SELECT COUNT(*) FROM events WHERE event_type='entry' AND timestamp LIKE ?",
        (today + "%",))
    exits = db_scalar(
        "SELECT COUNT(*) FROM events WHERE event_type='exit' AND timestamp LIKE ?",
        (today + "%",))

    recent = db_query(
        "SELECT face_id, event_type, timestamp, image_path "
        "FROM events ORDER BY id DESC LIMIT 20")

    if recent:
        rows = ""
        for ev in recent:
            url = _img_url(ev["image_path"])
            rows += (
                "<tr>"
                "<td>" + _thumb(url) + "</td>"
                '<td><code style="color:#64ffda;font-size:.82rem;">' + (ev["face_id"] or "unknown") + "</code></td>"
                "<td>" + _badge(ev["event_type"]) + "</td>"
                '<td style="font-size:.85rem;color:#8892b0;">' + (ev["timestamp"] or "—") + "</td>"
                "</tr>"
            )
        tbody = rows
    else:
        tbody = '<tr><td colspan="4" class="text-center py-5" style="color:#8892b0;">No events recorded yet.</td></tr>'

    body = (
        '<div class="d-flex justify-content-between align-items-center mb-4">'
        '<h1 class="sh mb-0">&#128202; Live Dashboard</h1>'
        '<span class="ubadge">&#128336; Last updated: ' + now_utc + '</span>'
        '</div>'

        '<div class="row g-4 mb-5">'

        '<div class="col-12 col-md-4"><div class="stat-card">'
        '<div class="stat-number">' + str(unique) + '</div>'
        '<div class="stat-label">Unique Visitors</div>'
        '</div></div>'

        '<div class="col-12 col-md-4"><div class="stat-card">'
        '<div class="stat-number" style="background:linear-gradient(135deg,#00b09b,#96c93d);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">'
        + str(entries) +
        '</div><div class="stat-label">Entries Today</div>'
        '</div></div>'

        '<div class="col-12 col-md-4"><div class="stat-card">'
        '<div class="stat-number" style="background:linear-gradient(135deg,#e94560,#c0392b);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">'
        + str(exits) +
        '</div><div class="stat-label">Exits Today</div>'
        '</div></div>'

        '</div>'  # /row

        '<h2 class="sh" style="font-size:1.2rem;">Recent Events (Last 20)</h2>'
        '<div class="tbl table-responsive mb-4">'
        '<table class="table table-borderless mb-0">'
        '<thead><tr><th>Image</th><th>Face ID</th><th>Event</th><th>Timestamp (UTC)</th></tr></thead>'
        '<tbody>' + tbody + '</tbody>'
        '</table></div>'
    )
    return _page("Dashboard", body, "dash", refresh=True)

# ---------------------------------------------------------------------------
# Route: Face Gallery  /faces
# ---------------------------------------------------------------------------
@app.route("/faces")
def faces():
    face_rows = db_query("SELECT id, first_seen FROM faces ORDER BY first_seen DESC")

    if not face_rows:
        body = (
            '<h1 class="sh">&#128100; Registered Faces</h1>'
            '<div class="text-center py-5" style="color:#8892b0;">'
            '<div style="font-size:4rem;">&#128100;</div>'
            '<p class="mt-3">No faces registered yet. Run the tracker to populate data.</p>'
            '<a href="/" class="btn btn-outline-danger mt-2">&#8592; Back to Dashboard</a>'
            '</div>'
        )
        return _page("Faces", body, "faces")

    cards = ""
    for face in face_rows:
        fid   = face["id"]
        fs    = (face["first_seen"] or "Unknown")[:19]
        count = db_scalar(
            "SELECT COUNT(*) FROM events WHERE face_id=? AND event_type='entry'", (fid,))
        url   = _first_img_for_face(fid)
        img   = ('<img src="' + url + '" alt="' + fid + '" loading="lazy">'
                 if url else '<div class="no-img">&#128100;</div>')
        entry_word = "entry" if count == 1 else "entries"
        cards += (
            '<div class="col-6 col-md-4 col-lg-3 col-xl-2">'
            '<div class="face-card">'
            + img +
            '<div class="card-body">'
            '<div class="face-id">' + fid + '</div>'
            '<div style="font-size:.78rem;color:#8892b0;margin-top:.4rem;">&#128337; ' + fs + '</div>'
            '<div style="font-size:.78rem;margin-top:.3rem;">'
            '<span style="color:#00b09b;">&#9650;</span> '
            '<strong>' + str(count) + '</strong> ' + entry_word +
            '</div>'
            '</div></div></div>'
        )

    total = len(face_rows)
    noun  = "face" if total == 1 else "faces"
    body  = (
        '<div class="d-flex justify-content-between align-items-center mb-4">'
        '<h1 class="sh mb-0">&#128100; Registered Faces</h1>'
        '<span class="ubadge">' + str(total) + ' ' + noun + ' registered</span>'
        '</div>'
        '<div class="row g-3">' + cards + '</div>'
        '<div class="mt-4"><a href="/" class="btn btn-sm btn-outline-secondary">&#8592; Dashboard</a></div>'
    )
    return _page("Faces", body, "faces")

# ---------------------------------------------------------------------------
# Route: Events Log  /events  (paginated, filterable)
# ---------------------------------------------------------------------------
PAGE_SIZE = 20

@app.route("/events")
def events():
    ftype = request.args.get("filter", "all")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    if ftype == "entry":
        total_rows = db_scalar("SELECT COUNT(*) FROM events WHERE event_type='entry'")
        where      = "WHERE event_type='entry'"
    elif ftype == "exit":
        total_rows = db_scalar("SELECT COUNT(*) FROM events WHERE event_type='exit'")
        where      = "WHERE event_type='exit'"
    else:
        total_rows = db_scalar("SELECT COUNT(*) FROM events")
        where      = ""

    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    page        = min(page, total_pages)
    offset      = (page - 1) * PAGE_SIZE

    rows = db_query(
        "SELECT id, face_id, event_type, timestamp, image_path "
        "FROM events " + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
        (PAGE_SIZE, offset)
    )

    def _fbtn(label, fval, icon):
        af = " af" if ftype == fval else ""
        return (
            '<a href="/events?filter=' + fval + '&amp;page=1" '
            'class="btn btn-sm btn-outline-secondary filter-btn' + af + ' me-2">'
            + icon + ' ' + label + '</a>'
        )

    filter_html = (
        '<div class="mb-3">'
        + _fbtn("All",   "all",   "&#127919;")
        + _fbtn("Entry", "entry", "&#x25B2;")
        + _fbtn("Exit",  "exit",  "&#x25BC;")
        + '</div>'
    )

    if rows:
        tbody = ""
        for ev in rows:
            url = _img_url(ev["image_path"])
            tbody += (
                "<tr>"
                '<td style="font-size:.8rem;color:#8892b0;">#' + str(ev["id"]) + "</td>"
                "<td>" + _thumb(url) + "</td>"
                '<td><code style="color:#64ffda;font-size:.82rem;">' + (ev["face_id"] or "unknown") + "</code></td>"
                "<td>" + _badge(ev["event_type"]) + "</td>"
                '<td style="font-size:.85rem;color:#8892b0;">' + (ev["timestamp"] or "—") + "</td>"
                "</tr>"
            )
    else:
        tbody = '<tr><td colspan="5" class="text-center py-5" style="color:#8892b0;">No events match this filter.</td></tr>'

    def _purl(p):
        return "/events?filter=" + ftype + "&amp;page=" + str(p)

    prev_btn = (
        '<a href="' + _purl(page - 1) + '" class="btn btn-sm btn-outline-secondary me-2">&#8592; Prev</a>'
        if page > 1 else
        '<button class="btn btn-sm btn-outline-secondary me-2" disabled>&#8592; Prev</button>'
    )
    next_btn = (
        '<a href="' + _purl(page + 1) + '" class="btn btn-sm btn-outline-secondary">Next &#8594;</a>'
        if page < total_pages else
        '<button class="btn btn-sm btn-outline-secondary" disabled>Next &#8594;</button>'
    )
    event_noun = "event" if total_rows == 1 else "events"

    body = (
        '<div class="d-flex justify-content-between align-items-center mb-4">'
        '<h1 class="sh mb-0">&#128203; Events Log</h1>'
        '<a href="/" class="btn btn-sm btn-outline-secondary">&#8592; Dashboard</a>'
        '</div>'
        + filter_html
        + '<div class="tbl table-responsive">'
        '<table class="table table-borderless mb-0">'
        '<thead><tr><th>#</th><th>Image</th><th>Face ID</th><th>Event</th><th>Timestamp (UTC)</th></tr></thead>'
        '<tbody>' + tbody + '</tbody>'
        '</table></div>'
        '<div class="d-flex justify-content-between align-items-center mt-3">'
        '<div style="font-size:.85rem;color:#8892b0;">'
        'Page <strong>' + str(page) + '</strong> of <strong>' + str(total_pages) + '</strong>'
        '&nbsp;|&nbsp; ' + str(total_rows) + ' total ' + event_noun +
        '</div>'
        '<div>' + prev_btn + next_btn + '</div>'
        '</div>'
    )
    return _page("Events Log", body, "events")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  FaceTracker Web Dashboard")
    print("  http://localhost:5050")
    print("  Press Ctrl+C to stop")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False)

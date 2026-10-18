"""STEP 6. Build the planner map.

    python scripts/05_forecast.py --replay-year 2025     # first, make the weekly forecasts
    python scripts/06_make_map.py --serve                # build the map, serve it from your computer, open the browser

--serve matters: OpenStreetMap's free map tiles refuse pages opened directly from a file, but accept pages
served from localhost. Press Ctrl+C in this window to stop. Without --serve the page is just written to
data/map.html (it then falls back to another key-free basemap, or to zones on a plain background).
"""
import argparse
import functools
import http.server
import json
import socketserver
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import config, mapview


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def make_server(port: int) -> socketserver.TCPServer:
    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(QuietHandler, directory=str(config.DATA_DIR))
    return socketserver.TCPServer(("127.0.0.1", port), handler)


def build() -> Path:
    if not config.REPLAY_JSON.exists():
        sys.exit("Run: python scripts/05_forecast.py --replay-year 2025   (use a year you have data for)")
    replay = json.loads(config.REPLAY_JSON.read_text())
    metrics = json.loads(config.METRICS_JSON.read_text()) if config.METRICS_JSON.exists() else None
    out = config.DATA_DIR / "map.html"
    out.write_text(mapview.build_map_html(replay, metrics), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="serve the map from localhost and open it in your browser")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    out = build()
    print(f"Wrote {out}.")
    if not args.serve:
        print("Open it in a browser, or run again with --serve for the best map.")
        sys.exit(0)
    try:
        with make_server(args.port) as httpd:
            url = f"http://localhost:{args.port}/map.html"
            print(f"Serving at {url} (press Ctrl+C to stop)")
            webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as e:
        sys.exit(f"Could not start the local server on port {args.port} ({e}). Try --port 8766.")

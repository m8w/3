#!/usr/bin/env python3
"""
texture-review.py — visually review a folder of textures and remove the ones
you don't want (e.g. pot leaves, photos of people).

It opens a local web page showing every image as a thumbnail. Click an image to
mark it (red = remove). "Suggest" pre-marks likely cannabis/people images by
filename as a starting point — you still eyeball them. "Remove marked" MOVES the
marked files into a `_removed/` subfolder (recoverable — delete it yourself when
you're happy). Nothing is hard-deleted.

USAGE:
    python3 scripts/texture-review.py "/Users/wvn/Music/3/Sources/ButterchurnVisualizer/Resources/Textures big"

Then open the URL it prints (http://localhost:8765). Press Ctrl+C to stop.
"""
import http.server, socketserver, json, os, sys, shutil, urllib.parse

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
PORT = 8765
EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tga", ".tif", ".tiff", ".webp")
REMOVED = os.path.join(ROOT, "_removed")

# Filename hints (just a starting suggestion — you confirm visually).
HINTS = [
    # cannabis
    "weed", "pot", "cannabis", "marijuana", "ganja", "420", "hemp", "kush",
    "blunt", "joint", "doob", "dank", "bud", "leaf", "stoner", "smoke",
    # people
    "face", "person", "people", "portrait", "selfie", "model", "nude", "naked",
    "lady", "girl", "boy", "woman", "man", "guy", "human", "body", "celeb",
    "podcast", "hand", "eye", "skin", "hair",
    # animals
    "cat", "dog", "animal", "bird", "horse", "lion", "tiger", "bear", "wolf",
    "deer", "snake", "spider", "insect", "butterfly", "fish", "frog", "pet",
    # recognizable / likely-copyright
    "logo", "brand", "poster", "movie", "film", "album", "cover", "cartoon",
    "anime", "meme", "photo", "picture", "ground", "landscape", "city",
]

def list_images():
    out = []
    for dirpath, _, names in os.walk(ROOT):
        if os.path.basename(dirpath) == "_removed":
            continue
        for n in names:
            if n.lower().endswith(EXTS):
                rel = os.path.relpath(os.path.join(dirpath, n), ROOT)
                out.append(rel)
    out.sort()
    return out

PAGE = """<!DOCTYPE html><html><head><meta charset=utf-8><title>Texture review</title>
<style>
 body{margin:0;background:#111;color:#eee;font:13px system-ui}
 #bar{position:sticky;top:0;background:#1c1c1c;padding:10px 14px;display:flex;gap:10px;align-items:center;z-index:9;border-bottom:1px solid #333}
 input{background:#222;border:1px solid #444;color:#eee;border-radius:6px;padding:6px 9px}
 button{background:#2b6cff;border:0;color:#fff;border-radius:6px;padding:7px 12px;cursor:pointer;font-weight:600}
 button.alt{background:#444}
 button.danger{background:#d23b3b}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px;padding:10px}
 .cell{position:relative;cursor:pointer;border:3px solid transparent;border-radius:6px;overflow:hidden;background:#000}
 .cell img{width:100%;height:140px;object-fit:cover;display:block}
 .cell.mark{border-color:#ff3b3b}
 .cell.mark:after{content:"✕ REMOVE";position:absolute;inset:0;background:rgba(210,40,40,.45);color:#fff;
   display:flex;align-items:center;justify-content:center;font-weight:800;letter-spacing:1px}
 .name{font-size:10px;color:#aaa;padding:3px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 #count{margin-left:auto;color:#bbb}
</style></head><body>
<div id=bar>
  <input id=filter placeholder="filter by name…" oninput=render()>
  <button class=alt onclick=suggest()>Suggest pot/people</button>
  <button class=alt onclick=clearMarks()>Clear marks</button>
  <button class=danger onclick=removeMarked()>Remove marked</button>
  <span id=count></span>
</div>
<div id=grid></div>
<script>
let files=[], marked=new Set();
async function load(){ files=await (await fetch('/list')).json(); render(); }
function render(){
  const q=document.getElementById('filter').value.toLowerCase();
  const g=document.getElementById('grid'); g.innerHTML='';
  let shown=0;
  for(const f of files){
    if(q && !f.toLowerCase().includes(q)) continue;
    shown++;
    const c=document.createElement('div'); c.className='cell'+(marked.has(f)?' mark':'');
    c.onclick=()=>{ marked.has(f)?marked.delete(f):marked.add(f); c.classList.toggle('mark'); updateCount(); };
    const img=document.createElement('img'); img.loading='lazy'; img.src='/img/'+encodeURI(f);
    const nm=document.createElement('div'); nm.className='name'; nm.textContent=f.split('/').pop();
    c.appendChild(img); c.appendChild(nm); g.appendChild(c);
  }
  updateCount(shown);
}
function updateCount(shown){ document.getElementById('count').textContent =
  (shown!==undefined?shown+' shown · ':'') + files.length+' total · '+marked.size+' marked'; }
const HINTS=__HINTS__;
function suggest(){ for(const f of files){ const l=f.toLowerCase(); if(HINTS.some(h=>l.includes(h))) marked.add(f);} render(); }
function clearMarks(){ marked.clear(); render(); }
async function removeMarked(){
  if(!marked.size) return alert('Nothing marked.');
  if(!confirm('Move '+marked.size+' file(s) into _removed/ ? (recoverable)')) return;
  const r=await fetch('/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({paths:[...marked]})});
  const j=await r.json();
  alert('Moved '+j.moved+' file(s) to _removed/.');
  marked.clear(); load();
}
load();
</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?")[0])
        if path == "/":
            self._send(200, PAGE.replace("__HINTS__", json.dumps(HINTS)).encode(), "text/html; charset=utf-8")
        elif path == "/list":
            self._send(200, json.dumps(list_images()).encode(), "application/json")
        elif path.startswith("/img/"):
            rel = path[len("/img/"):]
            full = os.path.normpath(os.path.join(ROOT, rel))
            if not full.startswith(ROOT) or not os.path.isfile(full):
                self._send(404, b"nope"); return
            with open(full, "rb") as f: data = f.read()
            self._send(200, data, "image/*")
        else:
            self._send(404, b"nope")
    def do_POST(self):
        if self.path != "/delete": self._send(404, b"nope"); return
        n = int(self.headers.get("Content-Length", 0))
        paths = json.loads(self.rfile.read(n) or b"{}").get("paths", [])
        moved = 0
        for rel in paths:
            full = os.path.normpath(os.path.join(ROOT, rel))
            if not full.startswith(ROOT) or not os.path.isfile(full): continue
            dest = os.path.join(REMOVED, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try: shutil.move(full, dest); moved += 1
            except Exception: pass
        self._send(200, json.dumps({"moved": moved}).encode(), "application/json")

if __name__ == "__main__":
    if not os.path.isdir(ROOT):
        print(f"Folder not found: {ROOT}"); sys.exit(1)
    print(f"▶ Reviewing: {ROOT}")
    print(f"▶ Open:      http://localhost:{PORT}   (Ctrl+C to stop)")
    print(f"  Removed files go to: {REMOVED}  (delete it yourself when satisfied)")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nstopped.")

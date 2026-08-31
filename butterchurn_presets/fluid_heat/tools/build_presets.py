#!/usr/bin/env python3
"""Build Butterchurn JSON presets from the .milk sources.

Pipeline:
    .milk source (readable, commented)
      -> strip shader comments, fully parenthesise every expression
      -> normalised .milk in a temp dir
      -> milkdrop-preset-converter (node)  [real HLSL->GLSL + EEL->JS]
      -> Butterchurn JSON
      -> GATE: reject any output containing && / || / bool( in its shaders

The gate is the point. These shaders contain no boolean logic whatsoever, so
a boolean operator in the translated GLSL can only have come from the
hlslparser-js operand-chain bug (see reparen.py). Without the gate the
presets convert "successfully" and render arithmetic-as-boolean nonsense.

    python3 tools/build_presets.py [--out DIR]

Requires node with milkdrop-preset-converter installed (see --npm-dir).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reparen import reparen_expression  # noqa: E402

HERE = Path(__file__).resolve().parent
PRESET_DIR = HERE.parent / "presets"

ASSIGN_RE = re.compile(r"(\+=|-=|\*=|/=|=)")
SHADER_LINE_RE = re.compile(r"^(warp|comp)_(\d+)=`?(.*)$")


def split_top_level_assign(stmt: str) -> tuple[str, str, str] | None:
    """Split `lhs op rhs` on the first assignment operator at paren depth 0."""
    depth = 0
    i = 0
    while i < len(stmt):
        ch = stmt[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0:
            if stmt.startswith(("+=", "-=", "*=", "/="), i):
                return stmt[:i], stmt[i:i + 2], stmt[i + 2:]
            if ch == "=" and not stmt.startswith("==", i) and \
               (i == 0 or stmt[i - 1] not in "=!<>"):
                return stmt[:i], "=", stmt[i + 1:]
        i += 1
    return None


def process_shader(body: str) -> tuple[str, int]:
    """Strip comments, re-parenthesise every assignment RHS."""
    # drop // comments (they have served their purpose in the .milk source)
    body = re.sub(r"//[^\n]*", "", body)

    # peel the shader_body { ... } wrapper so we only touch statements
    m = re.search(r"shader_body\s*\{(.*)\}\s*$", body, re.S)
    if not m:
        raise ValueError("no shader_body { } wrapper found")
    inner = m.group(1)

    out_stmts: list[str] = []
    changed = 0
    for raw in inner.split(";"):
        stmt = " ".join(raw.split())          # collapse whitespace/newlines
        if not stmt:
            continue
        parts = split_top_level_assign(stmt)
        if parts is None:
            out_stmts.append(stmt)
            continue
        lhs, op, rhs = parts
        lhs, rhs = lhs.strip(), rhs.strip()
        try:
            new_rhs = reparen_expression(rhs)
        except ValueError as e:
            raise ValueError(f"cannot parse RHS {rhs!r}: {e}") from e
        if new_rhs != rhs:
            changed += 1
        out_stmts.append(f"{lhs} {op} {new_rhs}")

    return "shader_body {\n" + "".join(f"    {s};\n" for s in out_stmts) + "}", changed


def normalise_milk(src: str) -> tuple[str, int]:
    """Return .milk text with both shaders re-parenthesised."""
    lines = src.splitlines()
    shaders: dict[str, dict[int, str]] = {"warp": {}, "comp": {}}
    other: list[str] = []
    for l in lines:
        m = SHADER_LINE_RE.match(l)
        if m:
            shaders[m.group(1)][int(m.group(2))] = m.group(3)
        else:
            other.append(l)

    total_changed = 0
    rebuilt: list[str] = []
    for name in ("warp", "comp"):
        entries = shaders[name]
        if not entries:
            continue
        body = "\n".join(entries[k] for k in sorted(entries))
        new_body, changed = process_shader(body)
        total_changed += changed
        for i, bl in enumerate(new_body.splitlines(), start=1):
            rebuilt.append(f"{name}_{i}=`{bl}")

    return "\n".join(other + rebuilt) + "\n", total_changed


CONVERT_JS = r"""
const pkg = require('milkdrop-preset-converter');
const fs = require('fs'), path = require('path');
const [,, inDir, outDir] = process.argv;
fs.mkdirSync(outDir, {recursive:true});
(async () => {
  const files = fs.readdirSync(inDir).filter(f=>f.endsWith('.milk')).sort();
  const report = [];
  for (const f of files) {
    const src = fs.readFileSync(path.join(inDir,f),'utf8');
    try {
      const p = await pkg.convertPreset(src);
      fs.writeFileSync(path.join(outDir, f.replace(/\.milk$/,'.json')),
                       JSON.stringify(p, null, 1));
      report.push({file:f, ok:true,
                   warp:(p.warp||'').length, comp:(p.comp||'').length});
    } catch(e) {
      report.push({file:f, ok:false, error:String(e.message||e).split('\n')[0]});
    }
  }
  fs.writeFileSync(path.join(outDir,'_report.json'), JSON.stringify(report,null,1));
})();
"""

# my shaders use no boolean logic; these can only come from the parser bug
MANGLE_RE = re.compile(r"&&|\|\||\bbool\s*\(|\bbvec\d\s*\(")


def gate(json_path: Path) -> list[str]:
    d = json.loads(json_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for key in ("warp", "comp"):
        shader = d.get(key, "")
        if not shader:
            problems.append(f"{key}: EMPTY")
            continue
        # only inspect the translated body, not hlslparser's fixed preamble
        idx = shader.find("main_shader_sentinel")
        body = shader[idx:] if idx >= 0 else shader
        hits = MANGLE_RE.findall(body)
        if hits:
            problems.append(f"{key}: {len(hits)} boolean artifact(s) "
                            f"{sorted(set(hits))} - operand-chain bug")
    for key in ("frame_eqs_str", "pixel_eqs_str", "baseVals"):
        if key not in d:
            problems.append(f"missing key {key}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "butterchurn_json"))
    ap.add_argument("--npm-dir", default="/tmp/conv",
                    help="directory with milkdrop-preset-converter installed")
    args = ap.parse_args()

    npm_dir = Path(args.npm_dir)
    if not (npm_dir / "node_modules" / "milkdrop-preset-converter").is_dir():
        print(f"✗ milkdrop-preset-converter not found under {npm_dir}\n"
              f"  npm i milkdrop-preset-converter  (in that directory)")
        return 2

    srcs = sorted(PRESET_DIR.glob("*.milk"))
    if not srcs:
        print("no .milk sources found")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="fh_milk_"))
    print(f"normalising {len(srcs)} preset(s)…")
    for s in srcs:
        norm, changed = normalise_milk(s.read_text(encoding="utf-8"))
        (tmp / s.name).write_text(norm, encoding="utf-8")
        print(f"  {s.name}: {changed} expression(s) re-parenthesised")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # node resolves require() from the script's own directory, so the helper
    # has to live beside node_modules, not in the temp dir
    js = npm_dir / "_fh_convert.cjs"
    js.write_text(CONVERT_JS, encoding="utf-8")
    print("\nconverting via milkdrop-preset-converter…")
    r = subprocess.run(["node", str(js), str(tmp), str(out)],
                       cwd=npm_dir, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print("✗ converter failed:\n" + (r.stderr or r.stdout)[:2000])
        return 1

    report = json.loads((out / "_report.json").read_text())
    (out / "_report.json").unlink()

    failed = 0
    for entry in report:
        name = entry["file"]
        if not entry.get("ok"):
            print(f"  FAIL {name}: {entry.get('error')}")
            failed += 1
            continue
        jp = out / name.replace(".milk", ".json")
        problems = gate(jp)
        if problems:
            print(f"  FAIL {name}")
            for p in problems:
                print(f"         {p}")
            failed += 1
        else:
            print(f"  ok   {name}  warp={entry['warp']}b comp={entry['comp']}b")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(report) - failed}/{len(report)} presets built clean -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

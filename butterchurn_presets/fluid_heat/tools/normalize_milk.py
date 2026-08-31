#!/usr/bin/env python3
"""Normalise fluid_heat .milk presets so both parsers in this project accept them.

Three transformations, all of which fix real breakage:

1. Top-level `//` comment lines inside [preset00] -> moved above the section
   header and re-prefixed with ';'.
   PresetParser.swift skips only ';' and '#', then throws
   PresetParseError.malformedLine on any in-section line without an '='.

2. `per_frame_N=// text` / `per_pixel_N=// text` lines -> removed, and the
   surviving lines renumbered contiguously.
   THIS IS THE IMPORTANT ONE. MilkPresetConverter.swift joins equation lines
   with a single space:
       preset.perFrameEqs.joined(separator: " ")
   so one `//` comment silently comments out *every equation after it* on the
   joined line. The preset loads, runs, and does almost nothing.

3. Blank in-section lines -> dropped (no '=', so they would also throw).

Comments inside shader bodies (backticked warp_N= / comp_N= lines) are HLSL
and are left exactly as they are.

    python3 normalize_milk.py [files...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EQ_RE = re.compile(r"^(per_frame|per_pixel|per_frame_init)_(\d+)=(.*)$")


def normalise(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        sec = next(i for i, l in enumerate(lines) if l.strip().startswith("[preset"))
    except StopIteration:
        print(f"  {path.name}: no [preset00], skipped")
        return False

    header = [l for l in lines[:sec]]
    marker = lines[sec]
    body = lines[sec + 1:]

    moved: list[str] = []
    kept: list[str] = []
    dropped_eq = 0
    eq_buckets: dict[str, list[str]] = {}

    for l in body:
        s = l.strip()
        if not s:
            continue
        if s.startswith("//"):
            txt = s[2:].strip()
            moved.append(f"; {txt}" if txt else ";")
            continue
        m = EQ_RE.match(s)
        if m:
            kind, _, value = m.group(1), m.group(2), m.group(3)
            if value.strip().startswith("//"):
                # would comment out everything after it once joined
                txt = value.strip()[2:].strip()
                moved.append(f"; {txt}" if txt else ";")
                dropped_eq += 1
                continue
            eq_buckets.setdefault(kind, []).append(value)
            continue
        kept.append(l)

    # rebuild equation blocks with contiguous numbering
    rebuilt: list[str] = []
    for kind in ("per_frame_init", "per_frame", "per_pixel"):
        for i, v in enumerate(eq_buckets.get(kind, []), start=1):
            rebuilt.append(f"{kind}_{i}={v}")

    # keep base values and shader lines in their original relative order,
    # then append the renumbered equations, then the shader lines
    basevals = [l for l in kept if not re.match(r"^(warp|comp)_\d+=", l.strip())]
    shaders = [l for l in kept if re.match(r"^(warp|comp)_\d+=", l.strip())]

    out = header + moved + [marker] + basevals + rebuilt + shaders
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  {path.name}: {len(moved)} comment(s) hoisted "
          f"({dropped_eq} were equation-line comments), "
          f"{sum(len(v) for v in eq_buckets.values())} equations renumbered")
    return True


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]]
    if not paths:
        paths = sorted((Path(__file__).resolve().parent.parent / "presets").glob("*.milk"))
    for p in paths:
        normalise(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Structural validator for .milk presets.

Catches the mistakes that silently break a preset in Butterchurn: gaps in
the numbered equation/shader lines, unbalanced braces, a shader line that
forgot its leading backtick, `q` variables read in a shader but never
assigned in per_frame, and missing required base values.

It does NOT compile the HLSL - only a real Butterchurn/Milkdrop run does
that. Passing here means the file is well-formed, not that it looks good.

    python3 validate_milk.py ../presets/*.milk
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_BASEVALS = [
    "fRating", "fGammaAdj", "fDecay", "fVideoEchoZoom", "fVideoEchoAlpha",
    "nVideoEchoOrientation", "nWaveMode", "bTexWrap", "fWaveAlpha",
    "fWaveScale", "fWarpAnimSpeed", "fWarpScale", "fZoomExponent",
    "zoom", "rot", "cx", "cy", "dx", "dy", "warp", "sx", "sy",
    "b1n", "b2n", "b3n", "b1x", "b2x", "b3x", "b1ed",
]

# Milkdrop pixel shaders have no user-defined functions; these are the
# intrinsics Butterchurn's HLSL->GLSL translator provides.
ALLOWED_CALLS = {
    "tex2D", "tex3D", "lerp", "saturate", "frac", "floor", "ceil", "abs",
    "min", "max", "pow", "exp", "exp2", "log", "log2", "sqrt", "rsqrt",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sign",
    "length", "normalize", "dot", "cross", "reflect", "step", "smoothstep",
    "clamp", "fmod", "ddx", "ddy", "lum", "sat", "GetMain", "GetPixel",
    "GetBlur1", "GetBlur2", "GetBlur3",
    "float", "float2", "float3", "float4", "int", "bool",
}

SECTION_RE = re.compile(r"^(per_frame|per_pixel|warp|comp|wave_\d+_per_point|"
                        r"wave_\d+_per_frame|shape_\d+_per_frame)_(\d+)=(.*)$")
BASEVAL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def validate(path: Path) -> list[str]:
    errs: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if not lines or not lines[0].startswith("MILKDROP_PRESET_VERSION"):
        errs.append("missing MILKDROP_PRESET_VERSION on line 1")
    if "[preset00]" not in text:
        errs.append("missing [preset00] section header")

    sections: dict[str, dict[int, str]] = {}
    basevals: dict[str, str] = {}

    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.startswith("//") or line.startswith("["):
            continue
        m = SECTION_RE.match(line)
        if m:
            name, idx, body = m.group(1), int(m.group(2)), m.group(3)
            sections.setdefault(name, {})[idx] = body
            continue
        m = BASEVAL_RE.match(line)
        if m and not line.startswith(("MILKDROP", "PSVERSION")):
            basevals[m.group(1)] = m.group(2)

    for key in REQUIRED_BASEVALS:
        if key not in basevals:
            errs.append(f"missing base value: {key}")

    # numbering must start at 1 and have no gaps
    for name, entries in sections.items():
        nums = sorted(entries)
        if nums and nums[0] != 1:
            errs.append(f"{name}: numbering starts at {nums[0]}, must start at 1")
        gaps = [n for n in range(1, (nums[-1] if nums else 0) + 1)
                if n not in entries]
        if gaps:
            errs.append(f"{name}: missing line numbers {gaps[:8]}"
                        f"{' ...' if len(gaps) > 8 else ''}")

    # shader blocks: backtick prefix, balanced braces, known intrinsics
    for shader in ("warp", "comp"):
        entries = sections.get(shader)
        if not entries:
            continue
        body_lines = []
        for n in sorted(entries):
            body = entries[n]
            if not body.startswith("`"):
                errs.append(f"{shader}_{n}: shader line must start with a backtick")
                body_lines.append(body)
            else:
                body_lines.append(body[1:])
        body = "\n".join(body_lines)

        if body.count("{") != body.count("}"):
            errs.append(f"{shader}: unbalanced braces "
                        f"({body.count('{')} open, {body.count('}')} close)")
        if body.count("(") != body.count(")"):
            errs.append(f"{shader}: unbalanced parens "
                        f"({body.count('(')} open, {body.count(')')} close)")
        if "shader_body" not in body:
            errs.append(f"{shader}: missing shader_body block")
        if not re.search(r"\bret\s*=", body):
            errs.append(f"{shader}: never assigns ret")

        stripped = re.sub(r"//.*", "", body)
        for call in set(CALL_RE.findall(stripped)):
            if call not in ALLOWED_CALLS and call not in ("shader_body", "if", "for", "while", "return"):
                errs.append(f"{shader}: calls '{call}()' - Milkdrop shaders "
                            f"have no user functions; inline it")

        # q variables read in shaders must be assigned in per_frame
        used_q = set(re.findall(r"\bq(\d+)\b", stripped))
        frame_body = "\n".join(sections.get("per_frame", {}).values())
        assigned_q = set(re.findall(r"\bq(\d+)\s*=", frame_body))
        missing = sorted(used_q - assigned_q, key=int)
        if missing:
            errs.append(f"{shader}: reads q{', q'.join(missing)} "
                        f"but per_frame never assigns them")

    # per_pixel / per_frame must NOT use backticks
    for name in ("per_frame", "per_pixel"):
        for n, body in sections.get(name, {}).items():
            if body.startswith("`"):
                errs.append(f"{name}_{n}: equation lines must not start with a backtick")

    return errs


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]]
    if not paths:
        here = Path(__file__).resolve().parent.parent / "presets"
        paths = sorted(here.glob("*.milk"))
    if not paths:
        print("no .milk files given or found", file=sys.stderr)
        return 2

    bad = 0
    for p in paths:
        errs = validate(p)
        if errs:
            bad += 1
            print(f"FAIL {p.name}")
            for e in errs:
                print(f"       {e}")
        else:
            print(f"ok   {p.name}")
    print(f"\n{len(paths) - bad}/{len(paths)} presets structurally valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""
sn2_sim.py — drive (and verify) the SN2 song stream without the hardware.

The visuals lock to a MIDI stream that only exists when a Novation Supernova II
is plugged in: sn2_chaos8_runs.py auto-selects a korg/supernova output port and
exits if it can't find one. That makes the MIDI path — the spine of the whole
17:17 lock — impossible to develop, demo or soak-test on a bare laptop. It also
means the sync contract is written down in three places that must agree and are
checked by nothing:

    scripts/sn2_chaos8_runs.py    the generator        (source of truth)
    Resources/mixer_host.html     OSC_DEF              ("copied verbatim")
    MidiEngine.swift              the CC map it reads off channel 0

This covers both gaps.

MODES
  emit (default)  Open ONLY the virtual "sn2 chaos -> visuals" port and play the
                  same stream the rig produces — 8 channels of chaos, wandering
                  runs, mod wheel, the five sound-design sines and the ch-16
                  Program Change anchor. Needs python-rtmidi; no synth.
  --check         Cross-check the three files above and report any drift. Pure
                  standard library — no MIDI, no rtmidi, no hardware. Exits 1 on
                  a mismatch, so it works as a CI or pre-commit guard.
  --dump          Print the reconstructed oscillator field as CSV, to diff
                  against what the visualizer is actually doing.

Every sync-critical constant (song unit, oscillator periods and phases, CC
numbers, anchor channel, Performance banks) is READ OUT of sn2_chaos8_runs.py at
startup rather than copied, so this file cannot become the fourth place they
drift. If that script is missing or unparseable this one refuses to guess.

USAGE
    python3 scripts/sn2_sim.py --check           # no deps, no hardware
    python3 scripts/sn2_sim.py --song-unit 30    # a whole 17:17 arc in 30s
    MIDI=1 ./ButterchurnVisualizer.app/Contents/MacOS/ButterchurnVisualizer
"""
import argparse
import ast
import collections
import heapq
import itertools
import math
import os
import random
import re
import sys
import time

# ── where the three copies of the sync contract live ─────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GEN_PY     = os.path.join(HERE, "sn2_chaos8_runs.py")
MIXER_HTML = os.path.join(REPO, "Sources", "ButterchurnVisualizer", "Resources", "mixer_host.html")
MIDI_SWIFT = os.path.join(REPO, "Sources", "ButterchurnVisualizer", "MidiEngine.swift")

# The visualizer matches a source name containing "sn2" or "visuals".
DEFAULT_PORT_NAME = "sn2 chaos → visuals"

# mixer_host.html OSC_DEF name -> (period, phase) constants in sn2_chaos8_runs.py
OSCILLATORS = {
    "cutoff":    ("CUTOFF_LFO_PERIOD",     "CUTOFF_PARAM_PHASE"),
    "hardness":  ("HARDNESS_LFO_PERIOD",   "HARDNESS_PARAM_PHASE"),
    "resonance": ("RESONANCE_LFO_PERIOD",  "RESONANCE_PARAM_PHASE"),
    "sync":      ("SYNC_LFO_PERIOD",       "SYNC_PARAM_PHASE"),
    "pitch":     ("PITCH_BEND_LFO_PERIOD", "PITCH_PARAM_PHASE"),
}

# MidiEngine.emit() reads these off channel 0 -> the CC constant each one must be.
SWIFT_CC = {
    "cutoff": "FILTER1_CUTOFF_CC",
    "reso":   "FILTER1_RESONANCE_CC",
    "hard":   "WS_DEPTH_CC",
    "sync":   "OSC2_MOD_CC",
    "mod":    "MODWHEEL_CC",
}

# The sound-design CCs, in the order sound_design_loop() sends them.
SOUND_DESIGN = [
    ("hardness",  "WS_DEPTH_CC",          "HARDNESS_MIN",   "HARDNESS_MAX"),
    ("cutoff",    "FILTER1_CUTOFF_CC",    "CUTOFF_MIN",     "CUTOFF_MAX"),
    ("resonance", "FILTER1_RESONANCE_CC", "RESONANCE_MIN",  "RESONANCE_MAX"),
    ("sync",      "OSC2_MOD_CC",          "SYNC_CC_MIN",    "SYNC_CC_MAX"),
]


# ── reading the generator's constants ─────────────────────────────────────────
_BINOPS = {
    ast.Add:  lambda a, b: a + b,
    ast.Sub:  lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div:  lambda a, b: a / b,
    ast.Pow:  lambda a, b: a ** b,
}


def _number(node, env):
    """Evaluate a literal arithmetic node: 17 * 60 + 17, math.pi * 0.8, NAME * 2.25."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("not a number")
        return float(node.value)
    if isinstance(node, ast.Name):
        v = env[node.id]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("not a number")
        return float(v)
    if isinstance(node, ast.Attribute):                     # math.pi / Math.PI
        if node.attr.lower() == "pi":
            return math.pi
        raise ValueError("unknown attribute " + node.attr)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _number(node.operand, env)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_number(node.left, env), _number(node.right, env))
    raise ValueError("unsupported expression")


def load_generator(path=GEN_PY):
    """Every module-level constant of sn2_chaos8_runs.py, without importing it.

    Importing would pull in rtmidi (and so need the dependency just to run
    --check), so the source is parsed instead. Anything that isn't a plain
    literal or bit of arithmetic is skipped.
    """
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        sys.exit("cannot read the generator (%s): %s\n"
                 "sn2_sim.py reads its constants from that file and will not "
                 "guess them." % (path, e))
    env = {}
    for node in ast.parse(src, filename=path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                and len(target.elts) == len(node.value.elts):
            pairs = list(zip(target.elts, node.value.elts))
        else:
            pairs = [(target, node.value)]
        for tgt, val in pairs:
            if not isinstance(tgt, ast.Name):
                continue
            try:
                env[tgt.id] = _number(val, env)
            except Exception:
                try:
                    env[tgt.id] = ast.literal_eval(val)
                except Exception:
                    pass
    if "SONG_UNIT_SECONDS" not in env:
        sys.exit("parsed %s but found no SONG_UNIT_SECONDS — has it been "
                 "restructured?" % path)
    return env


# ── reading the two reimplementations ────────────────────────────────────────
def _js_number(expr):
    expr = expr.strip().rstrip(",").replace("Math.PI", "math.pi")
    return _number(ast.parse(expr, mode="eval").body, {})


def load_mixer(path=MIXER_HTML):
    """SONG_UNIT and the OSC_DEF table out of mixer_host.html."""
    out = {"song_unit": None, "osc": {}}
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return out
    m = re.search(r"const\s+SONG_UNIT\s*=\s*([0-9.]+)", src)
    if m:
        out["song_unit"] = float(m.group(1))
    block = re.search(r"const\s+OSC_DEF\s*=\s*\{(.*?)\n\s*\};", src, re.S)
    if block:
        for name, mult, phase in re.findall(
                r"(\w+)\s*:\s*\{\s*mult\s*:\s*([-\d.]+)\s*,\s*phase\s*:\s*([^}]+?)\s*\}",
                block.group(1)):
            try:
                out["osc"][name] = (float(mult), _js_number(phase))
            except Exception:
                pass
    return out


def load_swift(path=MIDI_SWIFT):
    """The channel-0 CC map and the anchor channel out of MidiEngine.swift."""
    out = {"cc": {}, "anchor_channel": None}
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return out
    for name, num in re.findall(r"(\w+)\s*=\s*v\((\d+)\)", src):
        out["cc"][name] = int(num)
    m = re.search(r"ch\s*==\s*(\d+)\s*\{\s*anchor", src)
    if m:
        out["anchor_channel"] = int(m.group(1))
    return out


# ── --check ──────────────────────────────────────────────────────────────────
def _fmt(v):
    if v is None:
        return "--"
    if isinstance(v, float) and abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return ("%.6f" % v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def check():
    """Verify the three files still describe the same song. True if in sync."""
    gen, mix, swift = load_generator(), load_mixer(), load_swift()
    unit = float(gen["SONG_UNIT_SECONDS"])
    rows = []

    def row(label, want, got, tol=1e-6):
        ok = want is not None and got is not None and abs(float(want) - float(got)) <= tol
        rows.append((label, want, got, ok))

    def group(title):
        rows.append((title, None, None, None))

    group("mixer_host.html  vs  sn2_chaos8_runs.py")
    row("song unit (s)", unit, mix["song_unit"])
    # The mixer rebuilds its clock as anchorIndex * SONG_UNIT + beat, which is
    # only right while one Performance lasts exactly one song unit.
    row("Performance dwell (s)", unit, gen.get("PERFORMANCE_DWELL_SECONDS"))
    for name, (period_key, phase_key) in OSCILLATORS.items():
        js_mult, js_phase = mix["osc"].get(name, (None, None))
        period = gen.get(period_key)
        row("%s  period x" % name, None if period is None else period / unit, js_mult)
        row("%s  phase" % name, gen.get(phase_key), js_phase)

    group("MidiEngine.swift  vs  sn2_chaos8_runs.py")
    for swift_name, gen_key in SWIFT_CC.items():
        row("%s  CC#" % swift_name, gen.get(gen_key), swift["cc"].get(swift_name))
    anchor = gen.get("GLOBAL_MIDI_CHANNEL")
    row("anchor channel (0-based)", None if anchor is None else anchor - 1,
        swift["anchor_channel"])

    print("sync contract check\n")
    print("  generator  %s" % os.path.relpath(GEN_PY, REPO))
    print("  mixer      %s" % os.path.relpath(MIXER_HTML, REPO))
    print("  engine     %s\n" % os.path.relpath(MIDI_SWIFT, REPO))
    for label, want, got, ok in rows:
        if ok is None:
            print("  %s" % label)
            continue
        print("  %s %-26s %10s   %10s" % ("ok " if ok else "OUT", label,
                                          _fmt(want), _fmt(got)))
    checks = [r for r in rows if r[3] is not None]
    bad = [r for r in checks if not r[3]]
    print()
    if bad:
        print("  %d of %d checks OUT OF SYNC — the visuals will drift off the "
              "music:" % (len(bad), len(checks)))
        for label, want, got, _ in bad:
            print("    %-26s expected %s, found %s"
                  % (label, _fmt(want), _fmt(got)))
        return False
    print("  %d checks, all in sync." % len(checks))
    return True


# ── --dump ───────────────────────────────────────────────────────────────────
def dump(gen, unit, span, samples):
    """CSV of the channel-0 oscillator field — exactly what osc01() rebuilds."""
    scale = unit / float(gen["SONG_UNIT_SECONDS"])
    cols = ["t", "cutoff", "hardness", "resonance", "sync", "pitch",
            "cutoff_cc", "hardness_cc", "resonance_cc", "sync_cc", "bend"]
    print(",".join(cols))
    for i in range(samples + 1):
        t = span * unit * i / samples

        def osc(name):
            period_key, phase_key = OSCILLATORS[name]
            period = float(gen[period_key]) * scale
            return (math.sin(2 * math.pi * t / period + float(gen[phase_key])) + 1) / 2

        vals = [osc(n) for n in ("cutoff", "hardness", "resonance", "sync", "pitch")]
        ccs = [int(_lerp(gen[lo], gen[hi], osc(name)))
               for name, _cc, lo, hi in SOUND_DESIGN]
        # SOUND_DESIGN order is hardness, cutoff, resonance, sync — reorder to match.
        hard_cc, cut_cc, reso_cc, sync_cc = ccs
        bend = int((osc("pitch") * 2 - 1) * float(gen["PITCH_BEND_RANGE"]))
        print("%.3f,%s,%d,%d,%d,%d,%d" % (
            t, ",".join("%.6f" % v for v in vals),
            cut_cc, hard_cc, reso_cc, sync_cc, bend))


def _lerp(a, b, x):
    return a + (b - a) * x


# ── emit ─────────────────────────────────────────────────────────────────────
def _import_rtmidi():
    try:
        import rtmidi
    except ImportError:
        sys.exit("emit mode needs python-rtmidi:  pip install python-rtmidi\n"
                 "(--check and --dump need nothing but the standard library.)")
    return rtmidi


def list_ports():
    rtmidi = _import_rtmidi()
    outs, ins = rtmidi.MidiOut().get_ports(), rtmidi.MidiIn().get_ports()
    print("MIDI outputs:")
    for i, n in enumerate(outs) or []:
        print("  [%d] %s" % (i, n))
    if not outs:
        print("  (none)")
    print("MIDI inputs:")
    for i, n in enumerate(ins) or []:
        print("  [%d] %s" % (i, n))
    if not ins:
        print("  (none)")


def emit(gen, args):
    """Play the rig's stream onto a virtual port. No synth, no hardware."""
    rtmidi = _import_rtmidi()
    real_unit = float(gen["SONG_UNIT_SECONDS"])
    unit = args.song_unit or real_unit
    scale = unit / real_unit
    num = int(gen["NUM_CHANNELS"])
    chan_phase = [i * (2 * math.pi / num) for i in range(num)]
    rng = random.Random(args.seed)

    out = rtmidi.MidiOut()
    try:
        out.open_virtual_port(args.port_name)
    except Exception as e:                            # noqa: BLE001 - reported verbatim
        sys.exit("could not open virtual port %r: %s" % (args.port_name, e))

    # A multiset, not a set: the same (channel, note) can be retriggered before
    # the first one is released — chaos and a run can collide on a channel — and
    # every note-on still owes a note-off, or the synth drones after we quit.
    sounding = collections.Counter()

    def note_on(ch, n, v):
        sounding[(ch, n)] += 1
        out.send_message([0x90 | ch, n, v])

    def note_off(ch, n):
        if sounding[(ch, n)] > 1:
            sounding[(ch, n)] -= 1
        else:
            sounding.pop((ch, n), None)
        out.send_message([0x80 | ch, n, 0])

    def cc(ch, num_, val):
        out.send_message([0xB0 | ch, num_, max(0, min(127, int(val)))])

    def pitch_bend(ch, value):
        raw = max(-8192, min(8191, int(value))) + 8192
        out.send_message([0xE0 | ch, raw & 0x7F, (raw >> 7) & 0x7F])

    # Song time is the scheduled time of the event being run, not the wall
    # clock. The activity/density gates read an LFO before drawing from the RNG,
    # so a wall clock would let scheduling jitter flip a gate and desynchronise
    # the whole stream; against the virtual clock a seeded run is reproducible,
    # and each event schedules the next from its own slot instead of from
    # whenever it happened to be serviced, so nothing accumulates drift.
    t0 = time.monotonic()
    clock = [0.0]
    now = lambda: clock[0]                                             # noqa: E731
    real = lambda: time.monotonic() - t0                               # noqa: E731
    per = lambda key: float(gen[key]) * scale                          # noqa: E731

    def lfo01(period, phase=0.0):
        return (math.sin(2 * math.pi * now() / period + phase) + 1) / 2

    queue, seq, locked = [], itertools.count(), [False] * num
    current, last_number = ["--"], [None]

    def at(t, fn):
        heapq.heappush(queue, (t, next(seq), fn))

    # ── the ch-16 Program Change: t = 0 for everything downstream ────────────
    def performance():
        if args.repeat_performance and last_number[0] is not None:
            n = last_number[0]          # force the duplicate-anchor case
        else:
            n = rng.randint(int(gen["PERFORMANCE_NUMBER_MIN"]),
                            int(gen["PERFORMANCE_NUMBER_MAX"]))
        last_number[0] = n
        letter = gen["PERFORMANCE_BANK_LETTER"]
        ch = int(gen["GLOBAL_MIDI_CHANNEL"]) - 1
        out.send_message([0xB0 | ch, 0, 0])                    # Bank Select MSB
        out.send_message([0xB0 | ch, 32, gen["PERF_BANK_LSB"][letter]])
        out.send_message([0xC0 | ch, n])                       # the anchor
        current[0] = "%s%03d" % (letter, n)
        if not args.quiet:
            print("\n-> Performance %s (ch %d anchor)" % (current[0], ch + 1))
        at(now() + unit, performance)

    # ── per-channel single-note chaos ────────────────────────────────────────
    def chaos(ch):
        phase = chan_phase[ch]
        if locked[ch]:
            at(now() + 0.05, lambda: chaos(ch))
            return
        activity = _lerp(gen["ACTIVITY_MIN"], gen["ACTIVITY_MAX"],
                         lfo01(per("ACTIVITY_LFO_PERIOD"), phase))
        if rng.random() > activity:
            at(now() + 0.1, lambda: chaos(ch))
            return
        gap_max = _lerp(gen["DENSITY_GAP_MAX_SPARSE"], gen["DENSITY_GAP_MAX_BUSY"],
                        lfo01(per("DENSITY_LFO_PERIOD"), phase))
        centre = _lerp(gen["VEL_MIN"], gen["VEL_MAX"],
                       lfo01(per("VELOCITY_LFO_PERIOD"), phase))
        spread = gen["VELOCITY_SPREAD"]
        lo = int(max(gen["VEL_MIN"], centre - spread))
        hi = int(min(gen["VEL_MAX"], centre + spread))
        note = rng.randint(int(gen["PITCH_MIN"]), int(gen["PITCH_MAX"]))
        note_on(ch, note, rng.randint(lo, max(lo, hi)))
        at(now() + rng.uniform(gen["NOTE_DUR_MIN"], gen["NOTE_DUR_MAX"]),
           lambda: note_off(ch, note))
        at(now() + rng.uniform(gen["NOTE_GAP_MIN"], gap_max), lambda: chaos(ch))

    # ── per-channel mod wheel ────────────────────────────────────────────────
    def modwheel(ch):
        centre = _lerp(0, 127, lfo01(per("MODWHEEL_LFO_PERIOD"), chan_phase[ch]))
        drift = gen["MODWHEEL_DRIFT_RANGE"]
        cc(ch, int(gen["MODWHEEL_CC"]), centre + rng.uniform(-drift, drift))
        at(now() + rng.uniform(gen["MODWHEEL_GAP_MIN"], gen["MODWHEEL_GAP_MAX"]),
           lambda: modwheel(ch))

    # ── per-channel sound design: the five 17:17 sines ───────────────────────
    def sound_design(ch):
        phase = chan_phase[ch]
        bend_x = lfo01(per("PITCH_BEND_LFO_PERIOD"), phase + gen["PITCH_PARAM_PHASE"]) * 2 - 1
        pitch_bend(ch, bend_x * gen["PITCH_BEND_RANGE"])
        for name, cc_key, lo, hi in SOUND_DESIGN:
            period_key, phase_key = OSCILLATORS[name]
            v = lfo01(per(period_key), phase + gen[phase_key])
            cc(ch, int(gen[cc_key]), _lerp(gen[lo], gen[hi], v))
        at(now() + rng.uniform(gen["SOUND_DESIGN_GAP_MIN"], gen["SOUND_DESIGN_GAP_MAX"]),
           lambda: sound_design(ch))

    # ── one wandering run at a time, anywhere in the system ──────────────────
    state = {"pattern": None}

    def start_run():
        ch = rng.randrange(num)
        length = rng.choice(gen["RUN_LENGTHS"])
        prev = state["pattern"]
        if prev is not None and len(prev) == length and rng.random() < gen["RUN_REUSE_PROB"]:
            pattern = prev[:]
            if rng.random() < gen["RUN_MUTATE_PROB"]:
                pattern[rng.randrange(length)] += rng.choice([-2, -1, 1, 2])
        else:
            pattern, cur = [], 0
            for _ in range(length):
                pattern.append(cur)
                cur += rng.choice(gen["RUN_STEP_CHOICES"])
        state["pattern"] = pattern
        root = rng.randint(int(gen["RUN_ROOT_MIN"]), int(gen["RUN_ROOT_MAX"]))
        pitches = [max(0, min(127, root + o)) for o in pattern]
        locked[ch] = True
        step(ch, pitches, 0)

    def step(ch, pitches, i):
        if i >= len(pitches):
            locked[ch] = False
            busy = lfo01(per("RUN_LFO_PERIOD"))
            lo = _lerp(gen["RUN_PAUSE_MIN"], gen["RUN_PAUSE_BUSY_MIN"], busy)
            hi = _lerp(gen["RUN_PAUSE_MAX"], gen["RUN_PAUSE_BUSY_MAX"], busy)
            at(now() + rng.uniform(lo, hi), start_run)
            return
        note = pitches[i]
        note_on(ch, note, rng.randint(int(gen["VEL_MIN"]), int(gen["VEL_MAX"])))
        dur = rng.uniform(gen["RUN_NOTE_DUR_MIN"], gen["RUN_NOTE_DUR_MAX"])

        def advance():
            note_off(ch, note)
            step(ch, pitches, i + 1)
        at(now() + dur, advance)

    def status():
        t = now()

        def osc(name):
            period_key, phase_key = OSCILLATORS[name]
            return lfo01(per(period_key), gen[phase_key])
        sys.stdout.write(
            "\r  %6.1fs  arc %5.1f%%  perf %-4s  cut %.2f  hard %.2f  reso %.2f  "
            "sync %.2f  bend %+.2f  notes %2d " % (
                t, (t % unit) / unit * 100, current[0], osc("cutoff"), osc("hardness"),
                osc("resonance"), osc("sync"), osc("pitch") * 2 - 1, len(sounding)))
        sys.stdout.flush()
        at(t + 1.0, status)

    print("sn2_sim - virtual port %r is open." % args.port_name)
    print("  8 channels of chaos + runs + the five 17:17 sines, no synth attached.")
    if args.song_unit:
        print("  song unit COMPRESSED to %gs (real rig: %gs) - arcs and Performance"
              % (unit, real_unit))
        print("  changes run %.0fx faster so a whole arc is watchable." % (1 / scale))
    else:
        print("  song unit %gs (17:17) - the real cadence." % unit)
    print("  launch the visualizer with MIDI=1 to lock onto it. Ctrl+C to stop.\n")

    at(0.0, performance)
    for ch in range(num):
        at(rng.uniform(0, 0.5), lambda c=ch: chaos(c))
        at(rng.uniform(0, gen["MODWHEEL_GAP_MAX"]), lambda c=ch: modwheel(c))
        at(rng.uniform(0, gen["SOUND_DESIGN_GAP_MAX"]), lambda c=ch: sound_design(c))
    at(rng.uniform(0, gen["RUN_PAUSE_MAX"]), start_run)
    if not args.quiet:
        at(1.0, status)

    try:
        while queue:
            when = queue[0][0]
            if args.duration and when > args.duration:
                break
            ahead = when - real()
            if ahead > 0:
                time.sleep(min(ahead, 0.2))       # short sleeps keep Ctrl-C snappy
                continue
            clock[0] = when
            heapq.heappop(queue)[2]()
    except KeyboardInterrupt:
        pass
    finally:
        for (ch, note), count in list(sounding.items()):
            for _ in range(count):
                note_off(ch, note)
        for ch in range(num):
            out.send_message([0xB0 | ch, 123, 0])              # All Notes Off
            pitch_bend(ch, 0)
        time.sleep(0.15)
        out.close_port()
        print("\nStopped. All notes off.")


# ── cli ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Run the SN2 song stream without a Supernova II, and check "
                    "that the visualizer still reconstructs it correctly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python3 scripts/sn2_sim.py --check\n"
               "  python3 scripts/sn2_sim.py --song-unit 30\n"
               "  python3 scripts/sn2_sim.py --dump --song-unit 1037 > arc.csv\n")
    ap.add_argument("--check", action="store_true",
                    help="cross-check the generator, mixer_host.html and "
                         "MidiEngine.swift; exit 1 if they have drifted apart")
    ap.add_argument("--dump", action="store_true",
                    help="print the oscillator field as CSV instead of emitting MIDI")
    ap.add_argument("--dump-span", type=float, default=1.0, metavar="N",
                    help="dump N song units, starting at the anchor (default 1)")
    ap.add_argument("--dump-samples", type=int, default=64, metavar="N",
                    help="rows per dump (default 64)")
    ap.add_argument("--song-unit", type=float, metavar="S",
                    help="compress the 17:17 unit to S seconds so a whole arc is "
                         "watchable in development (default: the real 1037s)")
    ap.add_argument("--duration", type=float, default=0, metavar="S",
                    help="stop after S seconds (default: run until Ctrl+C)")
    ap.add_argument("--seed", type=int, metavar="N",
                    help="seed the RNG; the stream is then byte-for-byte "
                         "reproducible run to run")
    ap.add_argument("--port-name", default=DEFAULT_PORT_NAME, metavar="NAME",
                    help="virtual port to open (default %r); the visualizer "
                         "matches 'sn2' or 'visuals'" % DEFAULT_PORT_NAME)
    ap.add_argument("--repeat-performance", action="store_true",
                    help="always re-send the SAME Performance number, so every "
                         "anchor after the first is a duplicate. The rig does "
                         "this ~1.8%% of the time by chance; mixer_host.html "
                         "only counts a downbeat when the number CHANGES, so "
                         "the song clock slips a whole 17:17 when it happens")
    ap.add_argument("--list-ports", action="store_true", help="list MIDI ports and exit")
    ap.add_argument("--quiet", action="store_true", help="no status line")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)
    if args.list_ports:
        list_ports()
        return
    gen = load_generator()
    if args.dump:
        dump(gen, args.song_unit or float(gen["SONG_UNIT_SECONDS"]),
             args.dump_span, max(1, args.dump_samples))
        return
    emit(gen, args)


if __name__ == "__main__":
    main()

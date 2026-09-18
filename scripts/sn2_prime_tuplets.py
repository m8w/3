#!/usr/bin/env python3
"""
sn2_prime_tuplets.py — machine-only PRIME-TUPLET polyrhythm generator for the
Novation Supernova II, with the same virtual fan-out + ch-16 sync anchor the
ButterchurnVisualizer reads (launch the visualizer with MIDI=1).

Instead of loose random notes, this schedules PRIME-NUMBER tuplets on an
absolute machine clock — timing no human could play:

  * N notes evenly spread across a span, where N is prime (2,3,5,7,11,13,17,19,
    23,29,31,37) and the span is a beat / half note / whole note / 2 or 3 whole
    notes. e.g. 7 notes per beat, 13 per beat, 17 per half note, 19 per whole
    note, 7 per two whole notes.
  * Several such streams run AT ONCE on different channels, each with its own
    prime and span, so they interweave into dense, shifting polyrhythm.
  * NESTED tuplets: any single slot of a tuplet can itself be subdivided into a
    prime sub-tuplet (e.g. one of the seven divided into 1/7ths).

Everything is emitted both to the synth and to the "sn2 chaos → visuals" virtual
port, and the ch-16 Program Change (Performance cycling every 17:17) is kept as
the visualizer's sync anchor. The slow sound-design modulators (cutoff, reso,
hardness, sync, pitch-bend, mod wheel) are kept too so the visuals still get
their arcs.

Install:  pip install python-rtmidi
Run:      python3 scripts/sn2_prime_tuplets.py
Tempo:    BPM=120 python3 scripts/sn2_prime_tuplets.py   (default 96)
Stop:     'q' + enter, or Ctrl+C  (sends All Notes Off + centers pitch bend)
"""
import rtmidi
import time
import math
import random
import threading
import heapq
import os
import sys

NUM_CHANNELS = 8

# ── the prime-tuplet grid ──────────────────────────────────────────────────────
BPM   = float(os.environ.get("BPM", "96"))
BEAT  = 60.0 / BPM          # one quarter-note
HALF  = 2 * BEAT            # half note
WHOLE = 4 * BEAT            # whole note (= one 4/4 measure)
MEASURE = WHOLE
# Prime tuplet counts — the number of notes packed into a span.
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# Spans a tuplet can fill. Names are only for the console.
SPAN_LIST = [("beat", BEAT), ("half", HALF), ("whole", WHOLE),
             ("2whole", 2 * WHOLE), ("3whole", 3 * WHOLE)]
NEST_PRIMES = [2, 3, 5, 7]   # a slot may subdivide into one of these
NEST_PROB   = 0.28           # chance any given slot becomes a nested sub-tuplet

MIN_STREAMS, MAX_STREAMS = 2, 5   # concurrent tuplet streams per measure
LOOKAHEAD = 2.0 * MEASURE         # how far ahead the planner schedules
REROLL_PROB = 0.45                # chance a tiled stream re-rolls its prime mid-bar

NOTE_LOW, NOTE_HIGH = 24, 100
VEL_MIN, VEL_MAX = 30, 122
GATE_MIN, GATE_MAX = 0.35, 0.95   # note length as a fraction of its slot

# ── performance cycling (the ch-16 sync anchor the visualizer locks to) ─────────
GLOBAL_MIDI_CHANNEL = 16
PERF_BANK_LSB = {"A": 1, "B": 2, "C": 3, "D": 4}
PERFORMANCE_BANK_LETTER = "C"
PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX = 0, 99
SONG_UNIT_SECONDS = 17 * 60 + 17          # 1037s
PERFORMANCE_DWELL_SECONDS = SONG_UNIT_SECONDS

# ── slow sound-design modulators (kept so the visuals still get their arcs) ─────
MODWHEEL_CC = 1
FILTER1_CUTOFF_CC = 74
FILTER1_RESONANCE_CC = 71
WS_DEPTH_CC = 83
OSC2_MOD_CC = 19
MODULATION_UPDATE_INTERVAL = 0.5

MODWHEEL_LFO_PERIOD   = SONG_UNIT_SECONDS * 0.5
PITCH_BEND_LFO_PERIOD = SONG_UNIT_SECONDS * 1.25
HARDNESS_LFO_PERIOD   = SONG_UNIT_SECONDS * 1.75
CUTOFF_LFO_PERIOD     = SONG_UNIT_SECONDS * 2.25
RESONANCE_LFO_PERIOD  = SONG_UNIT_SECONDS * 2.75
SYNC_LFO_PERIOD       = SONG_UNIT_SECONDS * 3.25
PITCH_PARAM_PHASE     = 0.0
HARDNESS_PARAM_PHASE  = math.pi * 0.4
CUTOFF_PARAM_PHASE    = math.pi * 0.8
RESONANCE_PARAM_PHASE = math.pi * 1.2
SYNC_PARAM_PHASE      = math.pi * 1.6
PITCH_BEND_RANGE = 1200
HARDNESS_MIN, HARDNESS_MAX = 0, 110
CUTOFF_MIN, CUTOFF_MAX = 24, 127
RESONANCE_MIN, RESONANCE_MAX = 0, 110
SYNC_CC_MIN, SYNC_CC_MAX = 0, 127

running = True
midi_out = None
midi_fanout = None
song_start = None
CHANNEL_PHASE = [i * (2 * math.pi / NUM_CHANNELS) for i in range(NUM_CHANNELS)]

# Absolute-time event queue: (abs_time, seq, kind, ch, note, vel).
_sched = []
_sched_lock = threading.Lock()
_seq = 0


def select_port():
    global midi_out
    midi_out = rtmidi.MidiOut()
    ports = midi_out.get_ports()
    if not ports:
        print("No MIDI output ports found. Check Audio MIDI Setup.")
        sys.exit(1)
    for i, name in enumerate(ports):
        if any(k in name.lower() for k in ["korg", "microkey", "microkorg", "supernova"]):
            print(f"Auto-selected: [{i}] {name}")
            midi_out.open_port(i)
            return
    print("\nMIDI output ports:")
    for i, name in enumerate(ports):
        print(f"  [{i}] {name}")
    midi_out.open_port(int(input("Select port number: ").strip()))


def open_fanout():
    global midi_fanout
    try:
        midi_fanout = rtmidi.MidiOut()
        midi_fanout.open_virtual_port("sn2 chaos → visuals")
        print("Fan-out virtual port open: 'sn2 chaos → visuals' (launch visualizer with MIDI=1)")
    except Exception as e:
        print(f"Fan-out virtual port unavailable ({e}); continuing without it.")
        midi_fanout = None


def _emit(msg):
    midi_out.send_message(msg)
    if midi_fanout is not None:
        midi_fanout.send_message(msg)


def send_cc(ch, cc, val):
    _emit([0xB0 | ch, cc, val])


def send_pitch_bend(ch, value):
    value = max(-8192, min(8191, value))
    raw = value + 8192
    _emit([0xE0 | ch, raw & 0x7F, (raw >> 7) & 0x7F])


def all_notes_off():
    for ch in range(NUM_CHANNELS):
        _emit([0xB0 | ch, 123, 0])


def center_pitch_bend_all():
    for ch in range(NUM_CHANNELS):
        send_pitch_bend(ch, 0)


def send_performance_change(bank_letter, number):
    ch = GLOBAL_MIDI_CHANNEL - 1
    _emit([0xB0 | ch, 0, 0])
    _emit([0xB0 | ch, 32, PERF_BANK_LSB[bank_letter]])
    _emit([0xC0 | ch, number])
    print(f"\n→ Performance {bank_letter}{number:03d} (ch {GLOBAL_MIDI_CHANNEL})")


def performance_loop():
    while running:
        send_performance_change(PERFORMANCE_BANK_LETTER,
                                random.randint(PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX))
        elapsed = 0.0
        while running and elapsed < PERFORMANCE_DWELL_SECONDS:
            time.sleep(1.0)
            elapsed += 1.0


def elapsed_song_time():
    return time.time() - song_start


def lfo01(period, phase=0.0):
    return (math.sin(2 * math.pi * elapsed_song_time() / period + phase) + 1) / 2


def lerp(a, b, x):
    return a + (b - a) * x


def modwheel_loop(ch):
    phase = CHANNEL_PHASE[ch]
    while running:
        time.sleep(MODULATION_UPDATE_INTERVAL)
        if not running:
            break
        send_cc(ch, MODWHEEL_CC, max(0, min(127, int(lerp(0, 127, lfo01(MODWHEEL_LFO_PERIOD, phase))))))


def sound_design_loop(ch):
    phase = CHANNEL_PHASE[ch]
    while running:
        time.sleep(MODULATION_UPDATE_INTERVAL)
        if not running:
            break
        bend_x = lfo01(PITCH_BEND_LFO_PERIOD, phase + PITCH_PARAM_PHASE) * 2 - 1
        send_pitch_bend(ch, int(bend_x * PITCH_BEND_RANGE))
        send_cc(ch, WS_DEPTH_CC,        int(lerp(HARDNESS_MIN, HARDNESS_MAX,   lfo01(HARDNESS_LFO_PERIOD,  phase + HARDNESS_PARAM_PHASE))))
        send_cc(ch, FILTER1_CUTOFF_CC,  int(lerp(CUTOFF_MIN, CUTOFF_MAX,       lfo01(CUTOFF_LFO_PERIOD,    phase + CUTOFF_PARAM_PHASE))))
        send_cc(ch, FILTER1_RESONANCE_CC, int(lerp(RESONANCE_MIN, RESONANCE_MAX, lfo01(RESONANCE_LFO_PERIOD, phase + RESONANCE_PARAM_PHASE))))
        send_cc(ch, OSC2_MOD_CC,        int(lerp(SYNC_CC_MIN, SYNC_CC_MAX,     lfo01(SYNC_LFO_PERIOD,      phase + SYNC_PARAM_PHASE))))


# ── the absolute-time scheduler ─────────────────────────────────────────────────

def schedule(abs_time, kind, ch, note, vel=0):
    global _seq
    with _sched_lock:
        heapq.heappush(_sched, (abs_time, _seq, kind, ch, note, vel))
        _seq += 1


def schedule_note(ch, note, vel, t_on, dur):
    note = max(0, min(127, int(note)))
    schedule(t_on, "on", ch, note, max(1, min(127, int(vel))))
    schedule(t_on + max(0.02, dur), "off", ch, note, 0)


def scheduler_loop():
    """Pop due events and emit them, polling on a tight (~2ms) clock so the prime
    tuplets land where the math says — machine timing, not human."""
    while running:
        now = time.time()
        due = []
        with _sched_lock:
            while _sched and _sched[0][0] <= now:
                due.append(heapq.heappop(_sched))
            nxt = _sched[0][0] if _sched else None
        for _, _, kind, ch, note, vel in due:
            if kind == "on":
                _emit([0x90 | ch, note, vel])
            else:
                _emit([0x80 | ch, note, 0])
        if nxt is None:
            time.sleep(0.002)
        else:
            dt = nxt - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.002))


# ── the prime-tuplet planner ────────────────────────────────────────────────────

def pick_pitch(root):
    # mostly near the root (chord tones), sometimes a wild leap
    if random.random() < 0.75:
        return root + random.choice([-12, -7, -5, -3, 0, 0, 2, 3, 5, 7, 12])
    return random.randint(NOTE_LOW, NOTE_HIGH)


def plan_tuplet(t0, span, count, ch, root, depth):
    """Schedule `count` evenly-spaced notes across `span`, starting at t0. Any
    slot may (at depth>0) become a nested prime sub-tuplet across its own slot."""
    step = span / count
    for i in range(count):
        t = t0 + i * step
        if depth > 0 and random.random() < NEST_PROB:
            plan_tuplet(t, step, random.choice(NEST_PRIMES), ch,
                        root + random.choice([-12, -5, 0, 7, 12]), depth - 1)
        else:
            schedule_note(ch, pick_pitch(root),
                          random.randint(VEL_MIN, VEL_MAX),
                          t, step * random.uniform(GATE_MIN, GATE_MAX))


def plan_stream(bar_t, ch):
    """Fill one measure on one channel by tiling a prime tuplet across it (or, for
    spans longer than a measure, placing a single spanning tuplet)."""
    count = random.choice(PRIMES)
    span_name, span = random.choice(SPAN_LIST)
    root = random.randint(NOTE_LOW, NOTE_HIGH)
    if span >= MEASURE:
        plan_tuplet(bar_t, span, count, ch, root, depth=1)
        return (count, span_name)
    t, end = bar_t, bar_t + MEASURE
    while t < end - 1e-6:
        plan_tuplet(t, span, count, ch, root, depth=1)
        t += span
        if random.random() < REROLL_PROB:
            count = random.choice(PRIMES)
            root = random.randint(NOTE_LOW, NOTE_HIGH)
    return (count, span_name)


def planner_loop():
    next_bar = song_start
    while running:
        if next_bar - time.time() > LOOKAHEAD:
            time.sleep(0.02)
            continue
        streams = random.randint(MIN_STREAMS, MAX_STREAMS)
        chans = random.sample(range(NUM_CHANNELS), min(streams, NUM_CHANNELS))
        for ch in chans:
            plan_stream(next_bar, ch)
        next_bar += MEASURE


def input_loop():
    global running
    while running:
        try:
            cmd = input().strip()
        except (EOFError, KeyboardInterrupt):
            running = False
            return
        if cmd == "q":
            running = False
            return


def main():
    global running, song_start
    select_port()
    open_fanout()
    song_start = time.time()

    print(f"\nPrime-tuplet polyrhythm at {BPM:.0f} BPM "
          f"(beat {BEAT*1000:.0f}ms, whole {WHOLE*1000:.0f}ms).")
    print(f"Primes {PRIMES[0]}–{PRIMES[-1]} across beat/half/whole/2-3 whole notes, "
          f"{MIN_STREAMS}–{MAX_STREAMS} streams at once, nested sub-tuplets.")
    print(f"Cycling Performances {PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MIN:03d}-"
          f"{PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MAX:03d}, "
          f"{PERFORMANCE_DWELL_SECONDS//60}:{PERFORMANCE_DWELL_SECONDS%60:02d} each (ch-16 sync anchor).")
    print("'q' + enter, or Ctrl+C, to stop.\n")

    threading.Thread(target=scheduler_loop, daemon=True).start()
    threading.Thread(target=planner_loop, daemon=True).start()
    threading.Thread(target=performance_loop, daemon=True).start()
    for ch in range(NUM_CHANNELS):
        threading.Thread(target=modwheel_loop, args=[ch], daemon=True).start()
        threading.Thread(target=sound_design_loop, args=[ch], daemon=True).start()
    threading.Thread(target=input_loop, daemon=True).start()

    try:
        while running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        time.sleep(0.05)
        all_notes_off()
        center_pitch_bend_all()
        time.sleep(0.15)
        midi_out.close_port()
        if midi_fanout is not None:
            midi_fanout.close_port()
        print("\nStopped. All notes off.")


if __name__ == "__main__":
    main()

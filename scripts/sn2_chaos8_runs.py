#!/usr/bin/env python3
"""
sn2_chaos8_runs.py — 8-channel random note generator + a single wandering
"run" melody line, for Novation Supernova II.

This copy adds a VIRTUAL MIDI FAN-OUT so the ButterchurnVisualizer can read the
exact same stream and lock its visuals to it. Everything is sent both to the
korg/supernova port AND to a virtual port named "sn2 chaos → visuals"; launch
the visualizer with MIDI=1 (it matches "visuals"/"sn2") and both stay in sync.

  RUNS: one continuous background thread plays melodic runs of 5 or 7 notes,
  back-to-back with no gap between them (each note 0.1s - 1.0s long), on a
  randomly chosen channel (1-8). Only one run happens at a time anywhere in
  the whole system. While a channel is playing a run, its normal single-note
  chaos is paused (locked) so the run isn't muddied by unrelated notes on the
  same channel.

  PERFORMANCE CYCLING: continuously and automatically picks a uniformly
  random Performance from PERFORMANCE_BANK_LETTER (default 'C'), number
  000-127, and plays it for one SONG_UNIT_SECONDS (17:17 = 1037s) before
  switching. Sent as a Performance Bank+Program Change on the Global MIDI
  channel (16). That ch-16 Program Change is the sync anchor (t≡0).

  MODULATION SYSTEMS: several independent, slow-moving modulators run on the
  computer side. Each one is a smooth sine curve whose cycle length is a
  multiple of SONG_UNIT_SECONDS (17:17). Each channel is phase-offset from the
  other 7. So the whole value field is reconstructable from the ch-16 pulse.

Ctrl+C (or 'q') to stop — sends All Notes Off and centers Pitch Bend first.

Install:  pip install python-rtmidi
Run:      python3 sn2_chaos8_runs.py
"""
import rtmidi
import time
import math
import random
import threading
import sys

NUM_CHANNELS = 8

# ── single-note chaos ─────────────────────────────────────────────────────────
NOTE_GAP_MIN, NOTE_GAP_MAX = 0.01, 2.0
NOTE_DUR_MIN, NOTE_DUR_MAX = 0.1, 3.0
PITCH_MIN, PITCH_MAX = 12, 108
VEL_MIN, VEL_MAX = 1, 127

MODWHEEL_CC = 1
MODWHEEL_GAP_MIN, MODWHEEL_GAP_MAX = 3.0, 15.0

# ── runs ──────────────────────────────────────────────────────────────────────
RUN_LENGTHS = [5, 7]
RUN_NOTE_DUR_MIN, RUN_NOTE_DUR_MAX = 0.1, 1.0
RUN_PAUSE_MIN, RUN_PAUSE_MAX = 10.0, 45.0
RUN_PAUSE_BUSY_MIN, RUN_PAUSE_BUSY_MAX = 6.0, 20.0
RUN_ROOT_MIN, RUN_ROOT_MAX = PITCH_MIN + 12, PITCH_MAX - 12
RUN_STEP_CHOICES = [-3, -2, -1, 1, 2, 3]
RUN_REUSE_PROB = 0.6
RUN_MUTATE_PROB = 0.5

# ── performance cycling ───────────────────────────────────────────────────────
GLOBAL_MIDI_CHANNEL = 16
PERF_BANK_LSB = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
PERFORMANCE_BANK_LETTER = 'C'
PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX = 0, 54

# ── song unit + modulation systems ────────────────────────────────────────────
SONG_UNIT_SECONDS = 17 * 60 + 17               # 1037s
PERFORMANCE_DWELL_SECONDS = SONG_UNIT_SECONDS

DENSITY_LFO_PERIOD  = SONG_UNIT_SECONDS * 1.0
VELOCITY_LFO_PERIOD = SONG_UNIT_SECONDS * 1.5
ACTIVITY_LFO_PERIOD = SONG_UNIT_SECONDS * 2.0
MODWHEEL_LFO_PERIOD = SONG_UNIT_SECONDS * 0.5
RUN_LFO_PERIOD      = SONG_UNIT_SECONDS * 0.75

DENSITY_GAP_MAX_BUSY, DENSITY_GAP_MAX_SPARSE = NOTE_GAP_MAX, NOTE_GAP_MAX * 3
VELOCITY_SPREAD = 35
ACTIVITY_MIN, ACTIVITY_MAX = 0.15, 1.0
MODWHEEL_DRIFT_RANGE = 25

# ── sound-design modulation ───────────────────────────────────────────────────
FILTER1_CUTOFF_CC = 74
FILTER1_RESONANCE_CC = 71
WS_DEPTH_CC = 83
OSC2_MOD_CC = 19

SOUND_DESIGN_GAP_MIN, SOUND_DESIGN_GAP_MAX = 2.0, 6.0

PITCH_BEND_LFO_PERIOD    = SONG_UNIT_SECONDS * 1.25
HARDNESS_LFO_PERIOD      = SONG_UNIT_SECONDS * 1.75
CUTOFF_LFO_PERIOD        = SONG_UNIT_SECONDS * 2.25
RESONANCE_LFO_PERIOD     = SONG_UNIT_SECONDS * 2.75
SYNC_LFO_PERIOD          = SONG_UNIT_SECONDS * 3.25

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
midi_fanout = None                       # virtual port the visualizer reads
channel_locked = [False] * NUM_CHANNELS
lock = threading.Lock()

song_start = None
CHANNEL_PHASE = [i * (2 * math.pi / NUM_CHANNELS) for i in range(NUM_CHANNELS)]


def select_port():
    global midi_out
    midi_out = rtmidi.MidiOut()
    ports = midi_out.get_ports()
    if not ports:
        print("No MIDI output ports found. Check Audio MIDI Setup.")
        sys.exit(1)
    for i, name in enumerate(ports):
        if any(k in name.lower() for k in ['korg', 'microkey', 'microkorg', 'supernova']):
            print(f"Auto-selected: [{i}] {name}")
            midi_out.open_port(i)
            return
    print("\nMIDI output ports:")
    for i, name in enumerate(ports):
        print(f"  [{i}] {name}")
    midi_out.open_port(int(input("Select port number: ").strip()))


def open_fanout():
    """Open a virtual MIDI port the ButterchurnVisualizer can read (host-side)."""
    global midi_fanout
    try:
        midi_fanout = rtmidi.MidiOut()
        midi_fanout.open_virtual_port("sn2 chaos → visuals")
        print("Fan-out virtual port open: 'sn2 chaos → visuals' (launch visualizer with MIDI=1)")
    except Exception as e:
        print(f"Fan-out virtual port unavailable ({e}); continuing without it.")
        midi_fanout = None


def _emit(msg):
    """Send to the synth AND the visuals fan-out so both read the identical stream."""
    midi_out.send_message(msg)
    if midi_fanout is not None:
        midi_fanout.send_message(msg)


def send_note_on(ch, note, vel):
    _emit([0x90 | ch, note, vel])


def send_note_off(ch, note):
    _emit([0x80 | ch, note, 0])


def send_cc(ch, cc, val):
    _emit([0xB0 | ch, cc, val])


def send_pitch_bend(ch, value):
    """value: -8192..+8191, 0 = center."""
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
    lsb = PERF_BANK_LSB[bank_letter]
    _emit([0xB0 | ch, 0, 0])       # Bank Select MSB = 0
    _emit([0xB0 | ch, 32, lsb])    # Bank Select LSB -> Perf bank
    _emit([0xC0 | ch, number])     # Program Change (the ch-16 sync anchor)
    print(f"\n→ Performance {bank_letter}{number:03d} (ch {GLOBAL_MIDI_CHANNEL})")


def performance_loop():
    while running:
        number = random.randint(PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX)
        send_performance_change(PERFORMANCE_BANK_LETTER, number)
        elapsed = 0.0
        while running and elapsed < PERFORMANCE_DWELL_SECONDS:
            time.sleep(1.0)
            elapsed += 1.0


def elapsed_song_time():
    return time.time() - song_start


def lfo01(period, phase=0.0):
    t = elapsed_song_time()
    return (math.sin(2 * math.pi * t / period + phase) + 1) / 2


def lerp(a, b, x):
    return a + (b - a) * x


def note_loop(ch):
    phase = CHANNEL_PHASE[ch]
    while running:
        with lock:
            locked = channel_locked[ch]
        if locked:
            time.sleep(0.05)
            continue

        activity = lerp(ACTIVITY_MIN, ACTIVITY_MAX, lfo01(ACTIVITY_LFO_PERIOD, phase))
        if random.random() > activity:
            time.sleep(0.1)
            continue

        density = lfo01(DENSITY_LFO_PERIOD, phase)
        gap_max = lerp(DENSITY_GAP_MAX_SPARSE, DENSITY_GAP_MAX_BUSY, density)

        vel_center = lerp(VEL_MIN, VEL_MAX, lfo01(VELOCITY_LFO_PERIOD, phase))
        vel_lo = max(VEL_MIN, int(vel_center - VELOCITY_SPREAD))
        vel_hi = min(VEL_MAX, int(vel_center + VELOCITY_SPREAD))

        note = random.randint(PITCH_MIN, PITCH_MAX)
        vel = random.randint(vel_lo, vel_hi)
        dur = random.uniform(NOTE_DUR_MIN, NOTE_DUR_MAX)
        send_note_on(ch, note, vel)

        off_timer = threading.Timer(dur, send_note_off, args=[ch, note])
        off_timer.daemon = True
        off_timer.start()

        gap = random.uniform(NOTE_GAP_MIN, gap_max)
        time.sleep(gap)


def modwheel_loop(ch):
    phase = CHANNEL_PHASE[ch]
    while running:
        gap = random.uniform(MODWHEEL_GAP_MIN, MODWHEEL_GAP_MAX)
        time.sleep(gap)
        if not running:
            break
        center = lerp(0, 127, lfo01(MODWHEEL_LFO_PERIOD, phase))
        val = int(center + random.uniform(-MODWHEEL_DRIFT_RANGE, MODWHEEL_DRIFT_RANGE))
        val = max(0, min(127, val))
        send_cc(ch, MODWHEEL_CC, val)


def sound_design_loop(ch):
    phase = CHANNEL_PHASE[ch]
    while running:
        gap = random.uniform(SOUND_DESIGN_GAP_MIN, SOUND_DESIGN_GAP_MAX)
        time.sleep(gap)
        if not running:
            break

        bend_x = lfo01(PITCH_BEND_LFO_PERIOD, phase + PITCH_PARAM_PHASE) * 2 - 1
        send_pitch_bend(ch, int(bend_x * PITCH_BEND_RANGE))

        hardness = int(lerp(HARDNESS_MIN, HARDNESS_MAX, lfo01(HARDNESS_LFO_PERIOD, phase + HARDNESS_PARAM_PHASE)))
        send_cc(ch, WS_DEPTH_CC, hardness)

        cutoff = int(lerp(CUTOFF_MIN, CUTOFF_MAX, lfo01(CUTOFF_LFO_PERIOD, phase + CUTOFF_PARAM_PHASE)))
        send_cc(ch, FILTER1_CUTOFF_CC, cutoff)

        resonance = int(lerp(RESONANCE_MIN, RESONANCE_MAX, lfo01(RESONANCE_LFO_PERIOD, phase + RESONANCE_PARAM_PHASE)))
        send_cc(ch, FILTER1_RESONANCE_CC, resonance)

        sync = int(lerp(SYNC_CC_MIN, SYNC_CC_MAX, lfo01(SYNC_LFO_PERIOD, phase + SYNC_PARAM_PHASE)))
        send_cc(ch, OSC2_MOD_CC, sync)


def make_pattern(length):
    steps = []
    cur = 0
    for _ in range(length):
        steps.append(cur)
        cur += random.choice(RUN_STEP_CHOICES)
    return steps


def mutate_pattern(pattern):
    p = pattern[:]
    idx = random.randrange(len(p))
    p[idx] += random.choice([-2, -1, 1, 2])
    return p


def run_loop():
    last_pattern = None
    while running:
        ch = random.randint(0, NUM_CHANNELS - 1)
        length = random.choice(RUN_LENGTHS)

        if last_pattern is not None and len(last_pattern) == length and random.random() < RUN_REUSE_PROB:
            pattern = mutate_pattern(last_pattern) if random.random() < RUN_MUTATE_PROB else last_pattern[:]
        else:
            pattern = make_pattern(length)
        last_pattern = pattern

        root = random.randint(RUN_ROOT_MIN, RUN_ROOT_MAX)
        pitches = [max(0, min(127, root + offset)) for offset in pattern]

        with lock:
            channel_locked[ch] = True
        try:
            for pitch in pitches:
                if not running:
                    break
                vel = random.randint(VEL_MIN, VEL_MAX)
                dur = random.uniform(RUN_NOTE_DUR_MIN, RUN_NOTE_DUR_MAX)
                send_note_on(ch, pitch, vel)
                time.sleep(dur)
                send_note_off(ch, pitch)
        finally:
            with lock:
                channel_locked[ch] = False

        busy = lfo01(RUN_LFO_PERIOD)
        pause_min = lerp(RUN_PAUSE_MIN, RUN_PAUSE_BUSY_MIN, busy)
        pause_max = lerp(RUN_PAUSE_MAX, RUN_PAUSE_BUSY_MAX, busy)
        pause = random.uniform(pause_min, pause_max)
        chunks = max(1, int(pause / 0.1))
        for _ in range(chunks):
            if not running:
                break
            time.sleep(0.1)


def input_loop():
    global running
    while running:
        try:
            cmd = input().strip()
        except EOFError:
            return
        except KeyboardInterrupt:
            running = False
            return
        if cmd == 'q':
            running = False
            return


def main():
    global running, song_start
    select_port()
    open_fanout()
    song_start = time.time()

    dwell_min = PERFORMANCE_DWELL_SECONDS // 60
    dwell_sec = PERFORMANCE_DWELL_SECONDS % 60
    print(f"\nRunning {NUM_CHANNELS} channels of random notes + mod wheel, plus one wandering run.")
    print(f"Cycling random Performances {PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MIN:03d}-"
          f"{PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MAX:03d}, {dwell_min}:{dwell_sec:02d} each.")
    print("'q' + enter, or Ctrl+C, to stop.\n")

    threads = []
    for ch in range(NUM_CHANNELS):
        t1 = threading.Thread(target=note_loop, args=[ch], daemon=True)
        t2 = threading.Thread(target=modwheel_loop, args=[ch], daemon=True)
        t3 = threading.Thread(target=sound_design_loop, args=[ch], daemon=True)
        threads += [t1, t2, t3]
        t1.start()
        t2.start()
        t3.start()

    threading.Thread(target=run_loop, daemon=True).start()
    threading.Thread(target=performance_loop, daemon=True).start()
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


if __name__ == '__main__':
    main()

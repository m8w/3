#!/usr/bin/env python3
"""
sn2_400kb_series.py — sn2_chaos8_runs.py, plus real melodic material pulled
from 18 found MIDI files (all ~400KB, hence the name) instead of purely
algorithmic runs. Sibling of sn2_300kb_series.py, built the same way from a
different batch of source files -- each "series" is its own independent
line, not layered on top of the others.

FOUND-FLOOR PHRASES: found_floor_phrases_400kb.json (built once from x1 ghost
time 3.mid, wait for it 2.mid, solaris three.mid, seer elf 2.mid, re-ramped
again 1.mid, rapid ravbmps/ramps/rammmps 5.mid, now is won backwards 3a.mid,
New Presets 3 reaper x.mid, midi_export 11-3-18.mid, midi_expor 2t.mid, Korg
and Supernova II played with same midi feed session 1 MIDI.mid, finding the
bottom of the floor a.mid, dont be pickled(.mid / again.mid), Default (3).mid,
and cell effects t3.mid) holds ~11,700 short melodic phrases as
relative-semitone patterns, up to 700 drawn evenly from each source file
(the three "rapid ramp*" files turned out to be near-duplicates of each
other, so they collapsed to far fewer unique phrases combined -- the dedup
step catches that automatically). Both the SN2's wandering run and a new
korg run draw from this pool through a PhraseDeck: a shuffled, non-repeating
draw -- nothing plays twice until the whole deck has been dealt once, then
it reshuffles and starts over. Sameness is the enemy; this is how the script
avoids it for as long as the pool holds out. Falls back to the old
algorithmic pattern generator only if the JSON pool can't be found/loaded.

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
Run:      python3 sn2_400kb_series.py
"""
import rtmidi
import time
import math
import random
import json
import os
import threading
import sys

NUM_CHANNELS = 8

# ── korg's own voice, channel 9 ─────────────────────────────────────────────
KORG_CHANNEL = 8   # 0-indexed -> MIDI channel 9
KORG_PHASE = math.pi * 0.55   # distinct phase offset, not shared with SN2 channels
KORG_PRESET_DWELL_SECONDS = 5 * 60 + 55   # 5:55, independent of SONG_UNIT_SECONDS
# Category slot 8 of every genre group is VOCODER/AUDIO IN on the microKORG XL+
# (A18, A28...A88, B18...B88 -- Program Change value % 8 == 7). Those need a live
# audio-in signal to do anything; excluded from random cycling.
KORG_NON_VOCODER_PROGRAMS = [p for p in range(128) if p % 8 != 7]

# ── single-note chaos ─────────────────────────────────────────────────────────
NOTE_GAP_MIN, NOTE_GAP_MAX = 0.01, 2.0
NOTE_DUR_MIN, NOTE_DUR_MAX = 0.1, 3.0
PITCH_MIN, PITCH_MAX = 12, 108
VEL_MIN, VEL_MAX = 1, 127

MODWHEEL_CC = 1
MODULATION_UPDATE_INTERVAL = 0.5   # fixed, fast, no random gaps -- sweeps, not jumps

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
PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX = 0, 99

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

# ── sound-design modulation ───────────────────────────────────────────────────
FILTER1_CUTOFF_CC = 74
FILTER1_RESONANCE_CC = 71
WS_DEPTH_CC = 83
OSC2_MOD_CC = 19


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

# ── found-floor phrase pool ─────────────────────────────────────────────────
PHRASE_POOL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "found_floor_phrases_400kb.json")


class PhraseDeck:
    """A shuffled, non-repeating draw over a pool of phrases. Nothing repeats
    until the whole deck has been dealt once; then it reshuffles and deals
    again. Sameness is the enemy -- this is how we hold it off."""

    def __init__(self, pool):
        self.pool = list(pool)
        self.deck = []
        self.lock = threading.Lock()

    def draw(self):
        with self.lock:
            if not self.deck:
                self.deck = list(self.pool)
                random.shuffle(self.deck)
            return self.deck.pop()


def load_phrase_pool():
    try:
        with open(PHRASE_POOL_PATH) as f:
            pool = json.load(f)
        pool = [p for p in pool if isinstance(p, list) and len(p) >= 2]
        print(f"Loaded {len(pool)} found-floor phrases from {os.path.basename(PHRASE_POOL_PATH)}")
        return pool
    except Exception as e:
        print(f"Could not load found-floor phrase pool ({e}); "
              f"falling back to algorithmic pattern generation.")
        return None


running = True
midi_out = None
midi_fanout = None                       # virtual port the visualizer reads
korg_out = None                          # korg's own sound engine, separate from the DIN thru to the SN2
sn2_deck = None                          # PhraseDeck for the SN2 wandering run
korg_deck = None                         # PhraseDeck for the korg's own run
korg_locked = False                      # True while korg_run_loop is mid-phrase
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


def select_korg_port():
    """The korg's OWN sound engine is a separate USB-MIDI port from the DIN
    thru used for the SN2 -- 'microKORG XL SOUND' rather than 'MIDI OUT'."""
    global korg_out
    candidate = rtmidi.MidiOut()
    ports = candidate.get_ports()
    for i, name in enumerate(ports):
        if 'sound' in name.lower() and 'korg' in name.lower():
            print(f"Korg sound engine on: [{i}] {name} (channel {KORG_CHANNEL + 1})")
            candidate.open_port(i)
            korg_out = candidate
            return
    print("Could not find the korg's own 'SOUND' port; channel 9 voice disabled.")
    korg_out = None


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


def _emit_korg(msg):
    """Send to the korg's own sound engine AND the visuals fan-out."""
    if korg_out is not None:
        korg_out.send_message(msg)
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
    _emit_korg([0xB0 | KORG_CHANNEL, 123, 0])


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


def send_korg_program_change(program):
    """Bank Select is fixed at MSB=0, LSB=0 on the microKORG XL -- a single
    Program Change 0-127 addresses the whole A11-B88 range directly."""
    _emit_korg([0xB0 | KORG_CHANNEL, 0, 0])     # Bank Select MSB = 0
    _emit_korg([0xB0 | KORG_CHANNEL, 32, 0])    # Bank Select LSB = 0
    _emit_korg([0xC0 | KORG_CHANNEL, program])  # Program Change
    print(f"\n→ Korg preset {program:03d} (ch {KORG_CHANNEL + 1})")


def korg_preset_loop():
    while running:
        if korg_out is not None:
            program = random.randint(0, 127)
            send_korg_program_change(program)
        elapsed = 0.0
        while running and elapsed < KORG_PRESET_DWELL_SECONDS:
            time.sleep(1.0)
            elapsed += 1.0


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


def korg_note_loop():
    """Same activity/density/velocity LFO treatment as the SN2 channels, but
    on channel 9, sent to the korg's own sound engine instead of the SN2."""
    phase = KORG_PHASE
    while running:
        if korg_out is None:
            time.sleep(1.0)
            continue
        with lock:
            locked = korg_locked
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
        _emit_korg([0x90 | KORG_CHANNEL, note, vel])

        off_timer = threading.Timer(dur, lambda n=note: _emit_korg([0x80 | KORG_CHANNEL, n, 0]))
        off_timer.daemon = True
        off_timer.start()

        gap = random.uniform(NOTE_GAP_MIN, gap_max)
        time.sleep(gap)


def modwheel_loop(ch):
    """Pure LFO value, fixed fast interval -- a continuous sweep with no
    jitter and no random timing gaps, so consecutive CC values move by at
    most ~1 unit instead of jumping."""
    phase = CHANNEL_PHASE[ch]
    while running:
        time.sleep(MODULATION_UPDATE_INTERVAL)
        if not running:
            break
        val = int(lerp(0, 127, lfo01(MODWHEEL_LFO_PERIOD, phase)))
        val = max(0, min(127, val))
        send_cc(ch, MODWHEEL_CC, val)


def sound_design_loop(ch):
    """Same fixed-interval sweep treatment as modwheel_loop -- these were
    already pure-LFO (no jitter), so the fix here is purely the timing."""
    phase = CHANNEL_PHASE[ch]
    while running:
        time.sleep(MODULATION_UPDATE_INTERVAL)
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


def next_sn2_pattern(last_pattern):
    """Found-floor phrase when the deck is available; falls back to the old
    algorithmic generator (with its reuse/mutate behavior) otherwise."""
    if sn2_deck is not None:
        return sn2_deck.draw()
    length = random.choice(RUN_LENGTHS)
    if last_pattern is not None and len(last_pattern) == length and random.random() < RUN_REUSE_PROB:
        return mutate_pattern(last_pattern) if random.random() < RUN_MUTATE_PROB else last_pattern[:]
    return make_pattern(length)


def run_loop():
    last_pattern = None
    while running:
        ch = random.randint(0, NUM_CHANNELS - 1)
        pattern = next_sn2_pattern(last_pattern)
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


def korg_run_loop():
    """The korg's own wandering run -- same found-floor phrase pool as the
    SN2's run_loop, but its own deck (independent draw order) and its own
    channel-9 lock so it doesn't collide with korg_note_loop."""
    global korg_locked
    while running:
        if korg_out is None or korg_deck is None:
            time.sleep(2.0)
            continue

        pattern = korg_deck.draw()
        root = random.randint(RUN_ROOT_MIN, RUN_ROOT_MAX)
        pitches = [max(0, min(127, root + offset)) for offset in pattern]

        with lock:
            korg_locked = True
        try:
            for pitch in pitches:
                if not running:
                    break
                vel = random.randint(VEL_MIN, VEL_MAX)
                dur = random.uniform(RUN_NOTE_DUR_MIN, RUN_NOTE_DUR_MAX)
                _emit_korg([0x90 | KORG_CHANNEL, pitch, vel])
                time.sleep(dur)
                _emit_korg([0x80 | KORG_CHANNEL, pitch, 0])
        finally:
            with lock:
                korg_locked = False

        busy = lfo01(RUN_LFO_PERIOD, KORG_PHASE)
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
    global running, song_start, sn2_deck, korg_deck
    select_port()
    select_korg_port()
    open_fanout()
    song_start = time.time()

    pool = load_phrase_pool()
    if pool is not None:
        sn2_deck = PhraseDeck(pool)
        korg_deck = PhraseDeck(pool)

    dwell_min = PERFORMANCE_DWELL_SECONDS // 60
    dwell_sec = PERFORMANCE_DWELL_SECONDS % 60
    print(f"\n400kb series -- {NUM_CHANNELS} SN2 channels of random notes + mod wheel, "
          f"one SN2 run, one korg run, korg's own voice on channel {KORG_CHANNEL + 1}.")
    print(f"Cycling random Performances {PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MIN:03d}-"
          f"{PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MAX:03d}, {dwell_min}:{dwell_sec:02d} each. "
          f"Korg presets every {KORG_PRESET_DWELL_SECONDS // 60}:{KORG_PRESET_DWELL_SECONDS % 60:02d}.")
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
    threading.Thread(target=korg_note_loop, daemon=True).start()
    threading.Thread(target=korg_run_loop, daemon=True).start()
    threading.Thread(target=korg_preset_loop, daemon=True).start()
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
        if korg_out is not None:
            korg_out.close_port()
        print("\nStopped. All notes off.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
sn2_engine.py — shared engine for the prime-tuplet generators.

Drives BOTH synths at once:
  * the Supernova II's 8 parts on MIDI channels 1-8 (the DIN 'MIDI OUT' port), and
  * the microKORG XL's single part on channel 9 (its own USB 'SOUND' port),
and mirrors everything to the "sn2 chaos → visuals" virtual port for the
ButterchurnVisualizer.

TIMING is prime tuplets — N notes evenly across a beat / half / whole / 2-3 whole
notes, N prime (2..37), several interwoven, occasionally nested — but it is
RATE-GOVERNED so the SN2's DIN input can never be flooded (that flood was what
crashed the SN2 and forced restarts). A steady BEAT layer comes and goes so it
grooves here and there instead of being purely abstract.

Callers pass a get_phrase() → list[int] | None; when it returns a phrase, its
relative-semitone contour is laid across a tuplet's slots (real melody), else
pitches are chosen near a random root.
"""
import rtmidi
import time
import math
import random
import threading
import heapq
import sys

NUM_CHANNELS = 8                     # SN2 parts on channels 1-8
KORG_CHANNEL = 8                     # 0-indexed -> MIDI channel 9 (the korg)
KORG_PHASE = math.pi * 0.55
KORG_PRESET_DWELL_SECONDS = 7 * 60 + 7   # korg swaps presets every 7:07

# ── prime-tuplet grid ───────────────────────────────────────────────────────────
import os
BPM   = float(os.environ.get("BPM", "96"))
BEAT  = 60.0 / BPM
HALF  = 2 * BEAT
WHOLE = 4 * BEAT
MEASURE = WHOLE
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
SPAN_LIST = [("beat", BEAT), ("half", HALF), ("whole", WHOLE),
             ("2whole", 2 * WHOLE), ("3whole", 3 * WHOLE)]
NEST_PRIMES = [2, 3, 5, 7]

# ── DENSITY GOVERNOR — keeps the SN2 DIN bus well under its ~1000 msg/s ceiling ──
# MIDI DIN carries ~1040 three-byte messages/sec. Each note is 2 messages
# (on+off). We cap note traffic far below that and leave headroom for the CC
# sweeps and program changes, so the SN2 never buffer-overflows.
PER_STREAM_MAX_RATE = 16.0   # onsets/sec a single tuplet stream may reach
TOTAL_MAX_RATE      = 80.0   # onsets/sec across ALL eight SN2 channels
KORG_MAX_RATE       = 14.0   # onsets/sec for the korg's single part
MIN_NEST_STEP       = 0.22   # only a slot at least this long may nest
NEST_PROB           = 0.15
REROLL_PROB         = 0.4

NOTE_LOW, NOTE_HIGH = 24, 100
VEL_MIN, VEL_MAX = 30, 118
GATE_MIN, GATE_MAX = 0.35, 0.9

# ── beat layer (grooves here and there) ─────────────────────────────────────────
BEAT_ON_MEASURES  = (4, 10)   # length of a "has a beat" stretch, in measures
BEAT_OFF_MEASURES = (6, 16)   # length of an abstract (no-beat) stretch
PULSE_DIVS = [4, 8]           # steady notes per whole note when a beat is on

# ── performance cycling (ch-16 sync anchor) ─────────────────────────────────────
GLOBAL_MIDI_CHANNEL = 16
PERF_BANK_LSB = {"A": 1, "B": 2, "C": 3, "D": 4}
PERFORMANCE_BANK_LETTER = "C"
PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX = 0, 99
SONG_UNIT_SECONDS = 17 * 60 + 17          # 17:17 — the slow sound-design arcs keep this cycle
PERFORMANCE_DWELL_SECONDS = 11 * 60 + 11  # SN2 swaps Performances every 11:11 (independent of the arcs)

# ── slow sound-design modulators ────────────────────────────────────────────────
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
midi_out = None       # SN2 DIN
korg_out = None       # korg's own SOUND port
midi_fanout = None    # visualizer
song_start = None
CHANNEL_PHASE = [i * (2 * math.pi / NUM_CHANNELS) for i in range(NUM_CHANNELS)]

_sched = []
_sched_lock = threading.Lock()
_seq = 0
channel_free_at = [0.0] * (NUM_CHANNELS + 1)   # +1 for the korg channel


# ── ports ────────────────────────────────────────────────────────────────────────

def select_port():
    global midi_out
    midi_out = rtmidi.MidiOut()
    ports = midi_out.get_ports()
    if not ports:
        print("No MIDI output ports found. Check Audio MIDI Setup.")
        sys.exit(1)
    # The SN2 is reached over the DIN 'MIDI OUT' port (often named for the korg
    # that passes MIDI through to it). Pick that — but NEVER the korg's own
    # 'SOUND' engine port, which is a separate destination (channel 9).
    for i, name in enumerate(ports):
        lname = name.lower()
        if "sound" in lname:
            continue
        if any(k in lname for k in ["supernova", "midi out", "microkorg", "microkey", "korg", "din", "iac"]):
            print(f"SN2 (8 parts) on: [{i}] {name}")
            midi_out.open_port(i)
            return
    print("\nMIDI output ports:")
    for i, name in enumerate(ports):
        print(f"  [{i}] {name}")
    midi_out.open_port(int(input("Select the SN2 port number: ").strip()))


def select_korg_port():
    """The korg's OWN sound engine is a separate USB-MIDI port ('SOUND'),
    distinct from the DIN thru used for the SN2."""
    global korg_out
    cand = rtmidi.MidiOut()
    for i, name in enumerate(cand.get_ports()):
        if "korg" in name.lower() and "sound" in name.lower():
            print(f"Korg (ch {KORG_CHANNEL + 1}) on: [{i}] {name}")
            cand.open_port(i)
            korg_out = cand
            return
    print("Could not find the korg's 'SOUND' port; channel-9 voice disabled.")
    korg_out = None


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
    """SN2 DIN + fanout."""
    if midi_out is not None:
        midi_out.send_message(msg)
    if midi_fanout is not None:
        midi_fanout.send_message(msg)


def _emit_korg(msg):
    """Korg's own port + fanout (kept OFF the SN2 DIN so it doesn't add bus load)."""
    if korg_out is not None:
        korg_out.send_message(msg)
    if midi_fanout is not None:
        midi_fanout.send_message(msg)


def emit_note(ch, on, note, vel):
    status = (0x90 if on else 0x80) | (ch & 0x0F)
    (_emit_korg if ch == KORG_CHANNEL else _emit)([status, note, vel])


def send_cc(ch, cc, val, korg=False):
    (_emit_korg if korg else _emit)([0xB0 | (ch & 0x0F), cc, val])


def send_pitch_bend(ch, value, korg=False):
    value = max(-8192, min(8191, value))
    raw = value + 8192
    (_emit_korg if korg else _emit)([0xE0 | (ch & 0x0F), raw & 0x7F, (raw >> 7) & 0x7F])


def all_notes_off():
    for ch in range(NUM_CHANNELS):
        _emit([0xB0 | ch, 123, 0])
    _emit_korg([0xB0 | KORG_CHANNEL, 123, 0])


def center_pitch_bend_all():
    for ch in range(NUM_CHANNELS):
        send_pitch_bend(ch, 0)
    send_pitch_bend(KORG_CHANNEL, 0, korg=True)


def send_performance_change(bank_letter, number):
    ch = GLOBAL_MIDI_CHANNEL - 1
    _emit([0xB0 | ch, 0, 0])
    _emit([0xB0 | ch, 32, PERF_BANK_LSB[bank_letter]])
    _emit([0xC0 | ch, number])
    print(f"\n→ Performance {bank_letter}{number:03d} (ch {GLOBAL_MIDI_CHANNEL})")


def send_korg_program_change(program):
    _emit_korg([0xB0 | KORG_CHANNEL, 0, 0])
    _emit_korg([0xB0 | KORG_CHANNEL, 32, 0])
    _emit_korg([0xC0 | KORG_CHANNEL, program])
    print(f"\n→ Korg preset {program:03d} (ch {KORG_CHANNEL + 1})")


def performance_loop():
    while running:
        send_performance_change(PERFORMANCE_BANK_LETTER,
                                random.randint(PERFORMANCE_NUMBER_MIN, PERFORMANCE_NUMBER_MAX))
        elapsed = 0.0
        while running and elapsed < PERFORMANCE_DWELL_SECONDS:
            time.sleep(1.0)
            elapsed += 1.0


def korg_preset_loop():
    while running:
        if korg_out is not None:
            send_korg_program_change(random.randint(0, 127))
        elapsed = 0.0
        while running and elapsed < KORG_PRESET_DWELL_SECONDS:
            time.sleep(1.0)
            elapsed += 1.0


def elapsed_song_time():
    return time.time() - song_start


def lfo01(period, phase=0.0):
    return (math.sin(2 * math.pi * elapsed_song_time() / period + phase) + 1) / 2


def lerp(a, b, x):
    return a + (b - a) * x


def modwheel_loop(ch, korg=False):
    phase = KORG_PHASE if korg else CHANNEL_PHASE[ch]
    while running:
        time.sleep(MODULATION_UPDATE_INTERVAL)
        if not running:
            break
        send_cc(ch, MODWHEEL_CC, max(0, min(127, int(lerp(0, 127, lfo01(MODWHEEL_LFO_PERIOD, phase))))), korg)


def sound_design_loop(ch, korg=False):
    phase = KORG_PHASE if korg else CHANNEL_PHASE[ch]
    while running:
        time.sleep(MODULATION_UPDATE_INTERVAL)
        if not running:
            break
        bend_x = lfo01(PITCH_BEND_LFO_PERIOD, phase + PITCH_PARAM_PHASE) * 2 - 1
        send_pitch_bend(ch, int(bend_x * PITCH_BEND_RANGE), korg)
        send_cc(ch, WS_DEPTH_CC,          int(lerp(HARDNESS_MIN, HARDNESS_MAX,   lfo01(HARDNESS_LFO_PERIOD,  phase + HARDNESS_PARAM_PHASE))), korg)
        send_cc(ch, FILTER1_CUTOFF_CC,    int(lerp(CUTOFF_MIN, CUTOFF_MAX,       lfo01(CUTOFF_LFO_PERIOD,    phase + CUTOFF_PARAM_PHASE))), korg)
        send_cc(ch, FILTER1_RESONANCE_CC, int(lerp(RESONANCE_MIN, RESONANCE_MAX, lfo01(RESONANCE_LFO_PERIOD, phase + RESONANCE_PARAM_PHASE))), korg)
        send_cc(ch, OSC2_MOD_CC,          int(lerp(SYNC_CC_MIN, SYNC_CC_MAX,     lfo01(SYNC_LFO_PERIOD,      phase + SYNC_PARAM_PHASE))), korg)


# ── absolute-time scheduler ─────────────────────────────────────────────────────

def schedule(abs_time, kind, ch, note, vel=0):
    global _seq
    with _sched_lock:
        heapq.heappush(_sched, (abs_time, _seq, kind, ch, note, vel))
        _seq += 1


def schedule_note(ch, note, vel, t_on, dur):
    note = max(0, min(127, int(note)))
    schedule(t_on, 1, ch, note, max(1, min(127, int(vel))))
    schedule(t_on + max(0.03, dur), 0, ch, note, 0)


def scheduler_loop():
    while running:
        now = time.time()
        due = []
        with _sched_lock:
            while _sched and _sched[0][0] <= now:
                due.append(heapq.heappop(_sched))
            nxt = _sched[0][0] if _sched else None
        for _, _, kind, ch, note, vel in due:
            emit_note(ch, kind == 1, note, vel)
        if nxt is None:
            time.sleep(0.003)
        else:
            dt = nxt - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.003))


# ── prime-tuplet planning (rate-governed) ───────────────────────────────────────

def slot_pitch(root, phrase, i):
    if phrase:
        return root + phrase[i % len(phrase)]
    return root + random.choice([-12, -7, -5, -3, 0, 0, 2, 3, 5, 7, 12])


def choose_span_count(max_rate):
    """Pick a span, then a prime count whose onset rate stays under max_rate — so
    high primes only appear on long spans and no stream ever floods the bus."""
    span_name, span = random.choice(SPAN_LIST)
    allowed = [p for p in PRIMES if p / span <= max_rate]
    if not allowed:
        allowed = [2, 3]
    return span_name, span, random.choice(allowed)


def plan_tuplet(t0, span, count, ch, root, phrase, depth):
    step = span / count
    for i in range(count):
        t = t0 + i * step
        if depth > 0 and step >= MIN_NEST_STEP and random.random() < NEST_PROB:
            sub = max(p for p in NEST_PRIMES if p / step <= PER_STREAM_MAX_RATE) \
                  if any(p / step <= PER_STREAM_MAX_RATE for p in NEST_PRIMES) else 2
            plan_tuplet(t, step, sub, ch, slot_pitch(root, phrase, i), phrase, depth - 1)
        else:
            schedule_note(ch, slot_pitch(root, phrase, i),
                          random.randint(VEL_MIN, VEL_MAX),
                          t, step * random.uniform(GATE_MIN, GATE_MAX))


def plan_stream(bar_t, ch, get_phrase, max_rate):
    """Tile a rate-limited prime tuplet across one measure on one channel.
    Returns (approx onset rate, end_time)."""
    span_name, span, count = choose_span_count(max_rate)
    root = random.randint(NOTE_LOW, NOTE_HIGH)
    phrase = get_phrase()
    if span >= MEASURE:
        plan_tuplet(bar_t, span, count, ch, root, phrase, depth=1)
        return count / span, bar_t + span
    t, end = bar_t, bar_t + MEASURE
    while t < end - 1e-6:
        plan_tuplet(t, span, count, ch, root, phrase, depth=1)
        t += span
        if random.random() < REROLL_PROB:
            span_name, span, count = choose_span_count(max_rate)
            root = random.randint(NOTE_LOW, NOTE_HIGH)
            phrase = get_phrase()
    return count / span, end


def schedule_pulse(bar_t, ch, get_phrase):
    """A steady beat across the measure — the groove that comes and goes."""
    div = random.choice(PULSE_DIVS)
    step = MEASURE / div
    root = random.randint(36, 60)
    phrase = get_phrase()
    for i in range(div):
        accent = (i % max(1, div // 4) == 0)
        vel = random.randint(100, 122) if accent else random.randint(55, 85)
        schedule_note(ch, slot_pitch(root, phrase, i), vel, bar_t + i * step, step * 0.55)


def planner_loop(get_phrase):
    next_bar = song_start
    lookahead = 2.0 * MEASURE
    beat_on = False
    beat_left = random.randint(*BEAT_OFF_MEASURES)
    while running:
        if next_bar - time.time() > lookahead:
            time.sleep(0.02)
            continue

        # beat layer: run in stretches, then rest, so it grooves here and there
        beat_left -= 1
        if beat_left <= 0:
            beat_on = not beat_on
            beat_left = random.randint(*(BEAT_ON_MEASURES if beat_on else BEAT_OFF_MEASURES))

        free = [ch for ch in range(NUM_CHANNELS) if channel_free_at[ch] <= next_bar + 1e-6]
        random.shuffle(free)
        total_rate = 0.0

        if beat_on and free:
            pch = free.pop()
            schedule_pulse(next_bar, pch, get_phrase)
            channel_free_at[pch] = next_bar + MEASURE
            total_rate += 8.0 / MEASURE

        # interwoven prime-tuplet streams, up to the total-rate budget
        for ch in free:
            if total_rate >= TOTAL_MAX_RATE:
                break
            if random.random() < 0.5:      # not every free channel every bar → space
                continue
            rate, end_t = plan_stream(next_bar, ch, get_phrase, PER_STREAM_MAX_RATE)
            channel_free_at[ch] = end_t
            total_rate += rate

        # the korg's single part (its own port, its own budget)
        if korg_out is not None and channel_free_at[KORG_CHANNEL] <= next_bar + 1e-6:
            _, kend = plan_stream(next_bar, KORG_CHANNEL, get_phrase, KORG_MAX_RATE)
            channel_free_at[KORG_CHANNEL] = kend

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


def main(get_phrase=None, label="prime tuplets"):
    global running, song_start
    if get_phrase is None:
        get_phrase = lambda: None
    select_port()
    select_korg_port()
    open_fanout()
    song_start = time.time()

    print(f"\n{label} at {BPM:.0f} BPM (beat {BEAT*1000:.0f}ms, whole {WHOLE*1000:.0f}ms).")
    print(f"SN2 8 parts (ch 1-8) + korg (ch 9). Rate-governed: ≤{TOTAL_MAX_RATE:.0f} onsets/s on "
          f"the SN2 bus so it can't be flooded. Beat layer grooves in stretches.")
    print(f"Performances {PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MIN:03d}-"
          f"{PERFORMANCE_BANK_LETTER}{PERFORMANCE_NUMBER_MAX:03d}, "
          f"{PERFORMANCE_DWELL_SECONDS//60}:{PERFORMANCE_DWELL_SECONDS%60:02d} each (ch-16 sync anchor).")
    print("'q' + enter, or Ctrl+C, to stop.\n")

    threading.Thread(target=scheduler_loop, daemon=True).start()
    threading.Thread(target=planner_loop, args=[get_phrase], daemon=True).start()
    threading.Thread(target=performance_loop, daemon=True).start()
    threading.Thread(target=korg_preset_loop, daemon=True).start()
    for ch in range(NUM_CHANNELS):
        threading.Thread(target=modwheel_loop, args=[ch], daemon=True).start()
        threading.Thread(target=sound_design_loop, args=[ch], daemon=True).start()
    threading.Thread(target=modwheel_loop, args=[KORG_CHANNEL, True], daemon=True).start()
    threading.Thread(target=sound_design_loop, args=[KORG_CHANNEL, True], daemon=True).start()
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
        if midi_out is not None:
            midi_out.close_port()
        if korg_out is not None:
            korg_out.close_port()
        if midi_fanout is not None:
            midi_fanout.close_port()
        print("\nStopped. All notes off.")

#!/usr/bin/env python3
"""
record_sn2_midi.py — records live MIDI output from whichever sn2/korg
generative script is currently running (sn2_chaos8_runs.py, sn2_300kb_series.py,
sn2_400kb_series.py, sn2_500600kb_series.py, or a sibling) into a multi-track
Standard MIDI File, one track per channel.

Listens on the "sn2 chaos -> visuals" fan-out virtual port -- the same one
the ButterchurnVisualizer reads -- which already receives a copy of every
message the running script sends: notes, CC (mod wheel, filter/resonance/
waveshaping sound-design modulation), pitch bend, Performance Bank Select +
Program Change on channel 16, and korg preset Bank Select + Program Change
on channel 9. Nothing is filtered out.

Requires one of the generative scripts to already be running (that's what
creates the fan-out port).

RUNS FOREVER, back-to-back, in SESSION_SECONDS (default 2 hour) chunks: as
soon as one session's file is complete, the next session starts immediately
with a fresh file. Nothing ends this on its own -- across e.g. 33 hours
you'd get 16 complete 2-hour files plus one partial final file for whatever
was left when you stopped it. 'q' + enter, or Ctrl+C, stops it (finishing
and writing out whatever session is currently in progress first).

ONE FILE PER SESSION -- no separate numbered checkpoint files. Every
CHECKPOINT_INTERVAL seconds the session's own file is rewritten in place
with everything accumulated so far, so an interrupted run still leaves a
valid, complete, importable .mid file; by the time the full session elapses
that same file naturally already contains the whole 2 hours. Each session's
filename carries microsecond precision plus an explicit collision check, so
two separate sessions (or two separate runs of this script) can never
clobber each other.

Channels 1-8 (SN2 parts) and 9 (korg) each get their own named track;
channel 16 (Performance/Global) gets a track too.

Install:  pip install python-rtmidi
Run:      python3 record_sn2_midi.py
"""
import rtmidi
import time
import os
import sys
import struct
import threading
import datetime

SESSION_SECONDS = 2 * 60 * 60         # 2 hours per recording file
CHECKPOINT_INTERVAL = 10 * 60         # write a safety checkpoint every 10 min
TICKS_PER_BEAT = 480
BPM = 120.0
TICKS_PER_SECOND = TICKS_PER_BEAT * BPM / 60.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midi_recordings")

CHANNEL_NAMES = {i: f"SN2 Part {i + 1}" for i in range(8)}
CHANNEL_NAMES[8] = "Korg"
CHANNEL_NAMES[15] = "Global / Performance"
TRACKED_CHANNELS = list(range(9)) + [15]   # channels 1-9, and 16

events_by_channel = {ch: [] for ch in TRACKED_CHANNELS}
lock = threading.Lock()
start_time = None
running = True
midi_in = None
out_path = None


def find_and_open_port():
    global midi_in
    midi_in = rtmidi.MidiIn()
    midi_in.ignore_types(sysex=False, timing=True, active_sense=True)
    print("Waiting for the 'sn2 chaos -> visuals' fan-out port "
          "(start one of the generative scripts if it isn't running yet)...")
    while True:
        ports = midi_in.get_ports()
        idx = next((i for i, p in enumerate(ports)
                    if 'sn2' in p.lower() or 'visuals' in p.lower()), None)
        if idx is not None:
            print(f"Connected to: {ports[idx]}")
            midi_in.open_port(idx)
            return
        time.sleep(2.0)


def write_vlq(value):
    buf = [value & 0x7F]
    value >>= 7
    while value:
        buf.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(buf)


def build_track(channel, events, include_tempo=False):
    data = bytearray()
    if include_tempo:
        data += write_vlq(0) + bytes([0xFF, 0x51, 0x03]) + int(60_000_000 / BPM).to_bytes(3, 'big')

    name = CHANNEL_NAMES.get(channel, f"Channel {channel + 1}").encode('ascii', 'replace')
    data += write_vlq(0) + bytes([0xFF, 0x03, len(name)]) + name

    last_tick = 0
    for abs_seconds, raw in sorted(events, key=lambda e: e[0]):
        abs_tick = int(abs_seconds * TICKS_PER_SECOND)
        delta = max(0, abs_tick - last_tick)
        last_tick = abs_tick
        data += write_vlq(delta) + bytes(raw)

    data += write_vlq(0) + bytes([0xFF, 0x2F, 0x00])  # end of track
    return b'MTrk' + struct.pack('>I', len(data)) + data


def write_file(path, label):
    with lock:
        snapshot = {ch: list(evs) for ch, evs in events_by_channel.items()}

    used_channels = [ch for ch in TRACKED_CHANNELS if snapshot[ch]]
    if not used_channels:
        return

    tracks = []
    for i, ch in enumerate(used_channels):
        tracks.append(build_track(ch, snapshot[ch], include_tempo=(i == 0)))

    header = b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), TICKS_PER_BEAT)
    data = header + b''.join(tracks)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)

    total_events = sum(len(v) for v in snapshot.values())
    print(f"\n  [{label}] wrote {path} "
          f"({len(tracks)} tracks, {total_events} events, "
          f"{time.time() - start_time:.0f}s elapsed)")


def midi_callback(event, data=None):
    msg, deltatime = event
    if not msg or msg[0] >= 0xF0:
        return
    channel = msg[0] & 0x0F
    if channel not in TRACKED_CHANNELS:
        return
    now = time.time() - start_time
    with lock:
        events_by_channel[channel].append((now, tuple(msg)))


def unique_path(base_no_ext):
    path = base_no_ext + ".mid"
    n = 1
    while os.path.exists(path):
        n += 1
        path = f"{base_no_ext}_{n}.mid"
    return path


def start_new_session():
    """Reset shared recording state for a fresh session; returns its out_path."""
    global start_time, out_path
    with lock:
        for ch in TRACKED_CHANNELS:
            events_by_channel[ch] = []
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = unique_path(os.path.join(OUT_DIR, f"sn2_recording_{ts}"))
    start_time = time.time()
    return out_path


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
    global running

    find_and_open_port()
    midi_in.set_callback(midi_callback)

    print(f"Recording forever, back-to-back {SESSION_SECONDS / 3600:.1f}h sessions. "
          f"Checkpointing every {CHECKPOINT_INTERVAL // 60} min. "
          f"'q' + enter, or Ctrl+C, to stop.\n")

    threading.Thread(target=input_loop, daemon=True).start()

    session_num = 0
    try:
        while running:
            session_num += 1
            path = start_new_session()
            print(f"[session {session_num}] recording -> {path}")

            last_checkpoint = time.time()
            session_start = time.time()
            while running and (time.time() - session_start) < SESSION_SECONDS:
                time.sleep(1.0)
                if time.time() - last_checkpoint >= CHECKPOINT_INTERVAL:
                    write_file(out_path, "in progress")
                    last_checkpoint = time.time()

            label = "final" if (time.time() - session_start) >= SESSION_SECONDS else "final (stopped early)"
            write_file(out_path, label)
            if running:
                print(f"[session {session_num}] complete -- starting next session immediately\n")
    except KeyboardInterrupt:
        running = False
        write_file(out_path, "final (stopped early)")
    finally:
        midi_in.close_port()
        print(f"\nDone. {session_num} session(s) recorded this run.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
sn2_prime_tuplets.py — machine-only PRIME-TUPLET polyrhythm for the Supernova II
(8 parts, ch 1-8) AND the microKORG XL (ch 9), mirrored to the visualizer.

Timing is prime tuplets — N notes evenly across a beat / half / whole / 2-3 whole
notes, N prime (2..37), interwoven and occasionally nested — but rate-governed so
the SN2's MIDI input can never be flooded (no more SN2 crashes), and a steady beat
layer grooves in and out. Pitches are chosen near a wandering root.

For real melodies at these timings instead, run sn2_prime_phrases.py.

Install:  pip install python-rtmidi
Run:      python3 scripts/sn2_prime_tuplets.py     (tempo: BPM=120 python3 …)
Stop:     'q' + enter, or Ctrl+C
"""
import sn2_engine

if __name__ == "__main__":
    sn2_engine.main(get_phrase=None, label="Prime-tuplet polyrhythm")

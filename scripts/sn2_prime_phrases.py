#!/usr/bin/env python3
"""
sn2_prime_phrases.py — prime-tuplet timing (sn2_engine) played with REAL melodic
phrases from found_floor_phrases.json, on the Supernova II's 8 parts (ch 1-8) AND
the microKORG XL (ch 9), mirrored to the visualizer.

Timing is the rate-governed prime-tuplet engine (won't flood/crash the SN2, with a
beat layer that grooves in and out). Pitches come from ~9,100 found-floor phrases
dealt through a shuffled, non-repeating deck — each tuplet lays a drawn phrase's
contour across its slots. Falls back to random pitches if the pool is missing.

Install:  pip install python-rtmidi
Run:      python3 scripts/sn2_prime_phrases.py     (tempo: BPM=120 python3 …)
Stop:     'q' + enter, or Ctrl+C
"""
import os
import json
import random
import threading
import sn2_engine

PHRASE_POOL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "found_floor_phrases.json")


class PhraseDeck:
    """Shuffled, non-repeating draw — nothing repeats until the deck is dealt."""
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
        print(f"Could not load phrase pool ({e}); falling back to random pitches.")
        return None


if __name__ == "__main__":
    pool = load_phrase_pool()
    deck = PhraseDeck(pool) if pool else None
    get_phrase = deck.draw if deck else None
    sn2_engine.main(get_phrase=get_phrase, label="Prime-tuplet found-floor phrases")

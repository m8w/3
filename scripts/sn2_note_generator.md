# SN2 Note Generator

> This documents `scripts/sn2_chaos8_runs.py`, vendored in this repo. It is the
> MIDI companion to the ButterchurnVisualizer: it drives the synth **and** fans
> the identical stream out a virtual port named **`sn2 chaos`** that the
> visualizer subscribes to (launch the app with `MIDI=1`) so the visuals lock to
> the same 17:17 song arcs. See the main [README](../README.md) for the visual side.

A Python script that sends a constant stream of random MIDI notes to a synth over all 8 MIDI channels, plus occasional "run" melodies and mod wheel movement — a self-playing chaos generator.

## What it does

- Plays random notes on all 8 MIDI channels at random times, random pitches, random velocities, and random lengths — nonstop.
- Every so often, moves the mod wheel on each channel to a random position, independently of the notes.
- On top of that background chaos, one "run" happens at a time: a short wandering melody of 5 or 7 notes, played back-to-back on a randomly chosen channel. While a run is playing, that channel's random notes pause so the run stands out. After a run finishes there's a 10–45 second pause before the next one starts (possibly on a different channel). Runs sometimes reuse or slightly vary the shape of the previous run, sometimes start fresh.
- In the background, it also cycles through the synth's **Performances** (patches/presets) automatically — it picks one at random from Bank C, numbers 000–054 (the ones confirmed safe/reprogrammed so far), plays it for one "song unit" (17 minutes 17 seconds), then randomly picks the next one. This happens automatically the whole time the script runs — there's no prompt to answer, it just keeps cycling on its own.
- **17:17 as the song unit:** 17 minutes 17 seconds (1037 seconds) is the standard length everything else is measured against — it's how long each Performance plays before switching, and it's also the base cycle length several slow "modulation" systems breathe on (see below).
- **Modulation systems:** several independent, slow-moving controls run on the computer side. Each one is a smooth, continuously-rising-and-falling curve (no sudden jumps) whose cycle length is a multiple of the 17:17 song unit. Every channel/patch gets its own copy of each one, offset from the other 7 channels (and, for the sound-design controls below, offset from each other too), so nothing moves in lockstep.

  Steering modulators — reshape the ranges the note/run/mod-wheel logic picks from, rather than sending their own MIDI messages:
  - **Density** (1 song unit) — how close together notes trigger; busier vs. sparser.
  - **Dynamics** (1.5 song units) — an overall loudness swell.
  - **Channel presence** (2 song units) — each of the 8 channels fades in and out of prominence on its own offset schedule, so the ensemble feels like it's breathing rather than all 8 channels swelling together. A channel is always thinned at minimum, never fully silenced.
  - **Mod wheel drift** (0.5 song units) — the center point the mod wheel's random values wander around, per channel.
  - **Run frequency** (0.75 song units) — how often the wandering melodic runs happen; runs come more often during "busy" parts of the arc and less often during "sparse" parts.

  Sound-design modulators — each one sends its own continuous stream of MIDI to the microKORG XL, per channel/patch:
  - **Pitch drift** (1.25 song units) — a gentle Pitch Bend wobble, roughly ±0.3 semitones assuming the synth's default ±2 semitone bend range.
  - **Hardness** (1.75 song units) — Drive/Wave-Shape Depth, adding or easing off edge/grit.
  - **Filter cutoff** (2.25 song units) — Filter 1's brightness, kept mostly open so patches don't drift into inaudible territory.
  - **Filter resonance** (2.75 song units) — Filter 1's resonance, kept just shy of self-oscillation.
  - **Osc sync** (3.25 song units) — OSC2's mod control. The synth itself divides this knob into four zones (off / ring mod / sync / ring+sync), so this drifts through those modes rather than a smoothly variable "amount of sync" — that's a hardware limitation, not something the script can smooth out.

## How to run it

1. Make sure `python-rtmidi` is installed:
   ```
   pip install python-rtmidi
   ```
2. Connect the synth's MIDI OUT/IN to the computer (see MIDI setup below).
3. Run:
   ```
   python3 scripts/sn2_chaos8_runs.py
   ```
4. The script looks for a MIDI port with "korg", "microkorg", "supernova", or "microkey" in its name and connects automatically. If it can't find one, it lists all available MIDI ports and asks you to type the number of the one to use.
5. It then starts playing immediately and keeps going until you stop it.
6. To stop: press `q` then Enter, or press `Ctrl+C`. Either way it sends "all notes off" and re-centers pitch bend on every channel before it exits, so nothing gets stuck playing or stuck detuned.

## MIDI setup

This is written for the **microKORG XL**, connected via its **MIDI OUT** into the computer's MIDI input (e.g. a USB-MIDI interface), and the computer's MIDI output back into the synth's **MIDI IN**. The synth needs to be listening on the same MIDI channels the script sends on (channels 1–8 for notes, channel 16 for the Performance/Bank changes — this is the "Global" channel on the synth).

If the script can't find the synth's port automatically, run it and pick the correct port number from the list it prints.

The sound-design modulators (pitch, hardness, filter, sync) send real MIDI Control Change numbers taken from the microKORG XL's own Owner's Manual (front-panel-knob CC assignments): Filter 1 Cutoff = CC#74, Filter 1 Resonance = CC#71, Drive/Wave-Shape Depth = CC#83, OSC2 Osc Mod (includes Sync) = CC#19. Pitch drift uses a Pitch Bend message rather than a CC, since the synth doesn't expose raw oscillator pitch as a knob/CC.

## MIDI fan-out for the visualizer

On top of the real microKORG XL port, the script also opens a second, virtual MIDI port named **`sn2 chaos`** and sends every single message — notes, mod wheel, pitch bend, all the sound-design CCs, and the Performance/Bank changes — out both ports at once, identically. The ButterchurnVisualizer (launched with `MIDI=1`) subscribes to that `sn2 chaos` virtual port and sees the exact same stream in perfect sync, including the channel-16 Program Change it uses as the shared clock pulse (t≡0) to reconstruct the 17:17 song oscillators on the visual side.

If the virtual port can't be created for some reason, the script prints a message and just keeps running against the hardware port as before — it won't crash or block on this.

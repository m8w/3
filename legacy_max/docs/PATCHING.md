# Opening & running the patch

1. Put the `fluid_heat_audio/` folder in your Max file search path
   (Options -> File Preferences -> add its parent directory).

2. Open `fluid_heat_audio.maxpat`.

3. Turn on DSP (speaker icon bottom-right) and the `adc~` toggle.

4. Switch the `frame-tgl` toggle (next to `qmetro 16`) on. The solver
   will start advancing each tick; the `fh` render context window opens
   at 1280x720.

5. Play audio into your input (or drag `sfplay~` in front of `adc~`).
   The fluid should begin glowing at its 8 injection sites; bass plumes
   rise from the bottom, highs crackle across the top.

## Tuning guide

- **No motion?** Increase `audio gain` (the flonum next to `gain-msg`)
  until bins list shows values > 0.2. Check the built-in analyzer
  abstraction output in the Max window.
- **Too diffuse / cloudy?** Lower `diff` rates or increase `epsilon`
  (vorticity confinement slider).
- **Pressure artifacts?** Increase Jacobi iterations: change the
  `uzi 20` to `uzi 30` or `uzi 40`.
- **GPU slow?** Reduce state `dim` from 512x288 to 256x144.

## Sending external JSON-FFT

If you prefer to drive from your own analyzer, route a list of 8 floats
(0..4 each) directly into the `slab-inject` box, prefixed with the word
`bins`. The built-in analyzer chain can then be bypassed.

## Saving .maxpat after edits

Max re-formats the JSON on save. The existing structure here is valid
Max 9.1 JSON and will load, but opening and re-saving in Max normalizes
field ordering and preserves your edits.

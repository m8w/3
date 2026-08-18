# legacy_max/ — historical Max 9.1 patches

Everything in this directory is the previous Max/Jitter implementation of
`fluid_heat_audio`. It is kept for reference; the active codebase is
`fluid_heat_python/`.

**Do not open these files if you are on the Python path.** They require
Max 9.1, will not import into Pd, and their JavaScript externals are
Max-specific.

## What's here

    abstractions/*.maxpat   Max patcher abstractions (audio bins, organic mod,
                            voxel field, mesh synth, archive fetcher, etc.)
    shaders/*.jxs           Jitter shader wrappers around GLSL programs
                            (the GLSL bodies inside are still portable —
                             mesh_shade.jxs was extracted into
                             fluid_heat_python/shaders/mesh_shade.{vert,frag})
    scripts/*.js            Max JavaScript externals — obsolete in Python
    scripts/*.py            Python CLI tools — DUPLICATED into
                            fluid_heat_python/scripts/. Feel free to delete
                            these copies; the ones in fluid_heat_python/
                            are the maintained versions.
    docs/                   Original docs (still accurate for the archive
                            scripts; use fluid_heat_python/docs/ for the
                            Python-first playbook).
    fluid_heat_audio.maxpat top-level patch that wired everything together.

Delete this entire folder once you are certain none of it is needed for
reference.

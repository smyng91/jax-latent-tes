# pcm-map

Differentiable 2D enthalpy-porosity with vorticity–stream-function Boussinesq Navier–Stokes, used to fill a dual-cycle LHTES architecture map on $(\Lambda,\mathrm{Ra}_d)$.

## Install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Python 3.10+, JAX, Optax, NumPy, Matplotlib.

## Reproduce the paper numbers

From the repository root, with the venv active:

```sh
python -m pcm validate
python -m pcm baseline
python -m pcm sweep --ijhmt
python -m pcm sweep --grid-study --nx 32 --nx-fine 48 --seeds 2
python -m pcm explore
python -m pcm numbers
python -m pcm figures
python examples/01_stefan.py
```

JSON lands in `results/`. Figures land in `paper/figures/`. `python -m pcm numbers` copies those JSON values into `paper/generated_numbers.tex` and the `generated_*.tex` tables. The manuscript `paper/main.tex` does not invent results.

`--ijhmt` sets $\mathrm{Fo}_d=1.2$, $\beta=2\to 8$, volume matching, two seeds, and energy CFL 8. `--grid-study` fills a $3\times 3$ subset at two meshes and records winner flips. `baseline` compares equal-volume annular sheaths to cycle TO at two corners.

`sweep` and `explore` are topology-optimization runs (hours). `numbers` only rereads JSON. `figures` rereads JSON and re-runs the forward solver on stored winner densities to paint temperature, liquid fraction, and speed (seconds; cached in `results/winner_fields.npz`).

Unit checks:

```sh
python tests/test_core.py
```

## What the code is not

The map is a 2D NS model, not a 3D tank catalogue. Gallium validation uses Brent–Voller–Reid properties and Gau–Viskanta’s published factor of about 1.75; it does not digitize experimental front traces.

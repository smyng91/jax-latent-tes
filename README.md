# jax-latent-tes

Differentiable 2D enthalpy–porosity solver with vorticity–stream-function Boussinesq Navier–Stokes, used to map dual-cycle latent-heat thermal energy storage architectures on $`(\Lambda, \mathrm{Ra}_d)`$.

Installable package name: `pcm-map`. Command: `pcm`.

**Wiki:** [github.com/smyng91/jax-latent-tes/wiki](https://github.com/smyng91/jax-latent-tes/wiki)

## Install

Python 3.10+, JAX, Optax, NumPy, Matplotlib.

```sh
python -m venv .venv
```

Windows:

```sh
.venv\Scripts\activate
python -m pip install -e .
```

Unix:

```sh
.venv/bin/python -m pip install -e .
```

If you use [uv](https://github.com/astral-sh/uv):

```sh
uv sync
```

## Quick start

```sh
python examples/01_stefan.py
python -m pcm validate
python tests/test_core.py
```

`validate` writes `results/validate.json`. Generated JSON, figures, and the manuscript live outside git (`results/`, `paper/`, `figures/`, `output/`).

## Command line

```sh
python -m pcm validate
python -m pcm sweep --ijhmt
python -m pcm sweep --grid-study --nx 32 --nx-fine 48 --seeds 2
python -m pcm baseline
python -m pcm explore
python -m pcm numbers
python -m pcm figures
```

| Command | What it does |
|---|---|
| `validate` | Neumann / energy / gallium / adjoint checks → `results/validate.json` |
| `sweep` | $`(\Lambda, \mathrm{Ra}_d)`$ architecture map → `results/sweep.json` |
| `baseline` | Equal-volume annulus and spokes vs cycle TO |
| `explore` | Dual / Pareto / humidity / Stefan / COP variants at the conflict cell |
| `numbers` | Copy JSON into `paper/generated_*.tex` |
| `figures` | Rebuild `paper/figures/` from JSON (forward fields on stored winners) |

`--ijhmt` sets $`\mathrm{Fo}_d=1.2`$, $`\beta=2\to 8`$, volume matching, two seeds, and energy CFL 8. `--grid-study` fills a $`3\times 3`$ subset at two meshes. `sweep` and `explore` are long topology-optimization runs. `numbers` only rereads JSON. `figures` rereads JSON and re-runs the forward solver on stored winner densities (cached in `results/winner_fields.npz`).

See the [wiki command-line page](https://github.com/smyng91/jax-latent-tes/wiki/Command-line) for flags.

## Layout

```
pcm/        solver, design, sweep, validation, reporting
examples/   short scripts (Stefan, cavity, TO, gallium)
tests/      unit checks against published constants and JSON macros
```

## What the code is not

The map is a 2D Navier–Stokes model, not a 3D tank catalogue. Gallium validation uses Brent–Voller–Reid properties and Gau–Viskanta’s published late-time factor of about 1.75; it does not digitize experimental front traces.

## Citation

```bibtex
@misc{Yang2026jax,
  author       = {Yang, Sam},
  title        = {{jax-latent-tes}: Differentiable enthalpy--porosity {Navier--Stokes} topology optimization for dual-cycle latent-heat thermal energy storage},
  year         = {2026},
  howpublished = {\url{https://github.com/smyng91/jax-latent-tes}}
}
```

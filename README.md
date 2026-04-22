# Tidal Power vs Solar & Wind — Marginal Grid Value

A small, production-style analytics project that compares the
**marginal value** of tidal generation against solar and wind on the
CAISO grid using hourly data.

The hypothesis: because tidal currents are driven by the moon (not the
sun or weather), tidal generation should be **partially decorrelated**
from solar/wind output, and therefore potentially align better with
evening peak pricing.

---

## Project goals

1. Build an hourly dataset combining tidal currents, solar/wind
   production, and locational marginal prices (LMPs).
2. Convert tidal currents to generation using a simple physical
   turbine model.
3. Compute **value-weighted prices** and **value factors** for each
   resource, across the full year and across time slices (evening
   hours, winter months, top-10% price hours).
4. Visualize generation-vs-price and per-resource value metrics.

---

## Repository structure

```
Tidal Power/
├── config/
│   └── params.yaml          # single source of truth for parameters
├── data/
│   ├── raw/                 # cached source pulls (gitignored)
│   └── processed/           # merged outputs (gitignored)
├── notebooks/               # exploratory notebooks
├── src/
│   ├── data/                # fetch_noaa.py, fetch_caiso.py
│   ├── processing/          # clean_caiso.py, align_time.py
│   ├── models/              # tidal_model.py
│   ├── analysis/            # value_metrics.py, time_slices.py
│   ├── visualization/       # plots.py
│   └── utils/               # config, io, logging
├── tests/                   # pytest suite
├── main.py                  # end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

Every subpackage has a focused responsibility so three people can
work in parallel without colliding:

| Owner area         | Modules                                        |
| ------------------ | ---------------------------------------------- |
| Data ingestion     | `src/data/fetch_noaa.py`, `src/data/fetch_caiso.py` |
| Processing / model | `src/processing/*`, `src/models/tidal_model.py` |
| Analysis / viz     | `src/analysis/*`, `src/visualization/plots.py` |

---

## Getting started

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the pipeline

```bash
python main.py
```

By default the pipeline uses **mock data** (controlled by
`sources.noaa_use_mock` and `sources.caiso_use_mock` in
`config/params.yaml`) so it runs end-to-end without any network
access.  Output is written to `data/processed/merged_hourly.parquet`
and a value-metrics table is logged to the console.

### Run tests

```bash
pytest -q
```

---

## Configuration

All tunable parameters live in `config/params.yaml`:

- **year / timezone** — analysis window
- **tidal** — turbine physics (`rho`, `area`, `cp`, cut-in/out, rated MW)
- **time_slices** — evening hours, winter months, high-price quantile
- **sources** — mock toggles and station/node IDs

Avoid hard-coding values inside modules; read from the config instead.

---

## Physical model

The tidal turbine model implements the standard kinetic-energy flux:

\[
P = \tfrac{1}{2}\,\rho\,A\,v^{3}\,C_p
\]

with optional cut-in / cut-out speeds and a rated-power cap.  Since
the series is hourly, average power in MW equals hourly energy in MWh.

---

## Value metric

For each resource we compute the generation-weighted LMP:

\[
\text{VW} = \frac{\sum_t p_t\,g_t}{\sum_t g_t}
\]

and its ratio to the simple mean price (the **value factor**).  A
value factor > 1 means the resource tends to generate when prices are
above average.

---

## Contributing

- Keep functions small and functional; avoid classes unless state is
  genuinely needed.
- Add docstrings and type hints to every public function.
- Put new parameters in `config/params.yaml`, not in code.
- Add a unit test under `tests/` for every non-trivial function.

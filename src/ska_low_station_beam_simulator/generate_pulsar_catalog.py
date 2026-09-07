"""Builds every entry in ``pulsar_catalog.CATALOG_ENTRIES`` and writes it
to disk (``.npy`` + catalog.json) for ``DirectSynthesisStreamer`` to load
by name at construction instead of building at runtime -- see
``pulsar_catalog.py``'s module docstring for the full design (why
full-band-width generation, why complex64 on disk, why the catalog
records its own generation-time constants).

This is deliberately the ONLY module that imports both
``direct_synthesis.py`` (for ``build_pulsar_template``) and
``pulsar_catalog.py`` (for ``CATALOG_ENTRIES``/``save_pulsar_to_catalog``)
-- ``pulsar_catalog.py`` itself has no dependency on
``direct_synthesis.py``, so ``DirectSynthesisStreamer`` can import
``pulsar_catalog.py`` (to load by name) without a cycle.

Run::

    python -m ska_low_station_beam_simulator.generate_pulsar_catalog [output_dir]

(default ``output_dir``: ``pulsar_catalog.DEFAULT_CATALOG_DIR`` -- meant
to be populated as an OCI image build step, not committed to git or
packaged into a wheel; see CLAUDE.md's "Pulsar catalog" section.)

Some entries here (e.g. "slow_wide", 300ms) would blow this project's
own one-time CONSTRUCTION budget (target 10s, hard limit 30s -- see
CLAUDE.md) if built live at scan start. That's fine here: this script
pays that cost once, offline, so ``DirectSynthesisStreamer`` never has
to.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ska_low_station_beam_simulator.common import (
    BASE_FREQ_HZ,
    CHANNEL_OUTPUT_RATE_HZ,
    CHANNEL_WIDTH_HZ,
    MAX_NUM_CHANNELS,
)
from ska_low_station_beam_simulator.direct_synthesis import build_pulsar_template
from ska_low_station_beam_simulator.pulsar_catalog import (
    CATALOG_ENTRIES,
    DEFAULT_CATALOG_DIR,
    save_pulsar_to_catalog,
)


def generate_catalog(output_dir: Path) -> None:
    for entry in CATALOG_ENTRIES:
        print(f"building {entry.name!r} (period={entry.period_s*1000:.1f}ms, "
              f"DM={entry.dm_pc_cm3})...")
        t0 = time.perf_counter()
        template, n_period_samples = build_pulsar_template(
            MAX_NUM_CHANNELS,
            CHANNEL_WIDTH_HZ,
            BASE_FREQ_HZ,
            CHANNEL_OUTPUT_RATE_HZ,
            entry.period_s,
            entry.width_s,
            1.0,  # canonical unit amplitude -- callers scale at load time, see direct_synthesis.py
            entry.dm_pc_cm3,
            entry.sky_seed,
        )
        build_s = time.perf_counter() - t0
        save_pulsar_to_catalog(
            output_dir,
            entry.name,
            template,
            n_period_samples,
            entry.period_s,
            entry.width_s,
            entry.dm_pc_cm3,
            entry.sky_seed,
        )
        npy_size_mb = (output_dir / f"{entry.name}.npy").stat().st_size / 1e6
        print(f"  wrote {entry.name}.npy ({npy_size_mb:.1f}MB) in {build_s:.2f}s")

    print(f"\ncatalog written to {output_dir} ({len(CATALOG_ENTRIES)} entries)")


if __name__ == "__main__":
    _output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CATALOG_DIR
    generate_catalog(_output_dir)

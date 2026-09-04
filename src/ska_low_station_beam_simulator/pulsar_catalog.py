"""Named pulsar catalog: pre-generated pulsar templates that can be
embedded in the OCI image and loaded at ``DirectSynthesisStreamer``
construction instead of built there, trading arbitrary
``period_s``/``width_s``/``dm_pc_cm3`` configurability for near-instant
startup — see ``generate_pulsar_catalog.py`` for how these are actually
built, and ``direct_synthesis.py``'s ``pulsar_name`` source_cfg field for
how a streamer loads one. Both source_cfg styles remain supported side
by side: a client can reference a baked-in pulsar for fast setup, or
still supply raw parameters and pay the (now-tighter, see CLAUDE.md)
one-time construction cost for an arbitrary period/DM.

Every catalog entry is generated at the FULL SKA-Low band width
(``common.MAX_NUM_CHANNELS`` channels, starting at ``common.BASE_FREQ_HZ``,
the band's lowest channel) — a station simulating a NARROWER sub-band
just slices the columns it needs out of the same array (see
``load_pulsar_from_catalog``'s ``station_num_channels``/
``station_base_freq_hz`` arguments), matching this project's existing
principle that a pulsar's "sky carrier" is shared across every station
observing it (see ``direct_synthesis.py``'s module docstring) — one file
serves every valid station configuration, not one per
(num_channels, first_channel_id) combination.

Stored on disk as complex64 (halves image size vs. complex128 — no
meaningful fidelity loss for this purpose, since content is quantized to
int8 well downstream anyway); loaded back and upcast to complex128 so
``add_pulsar_tick``'s numba kernel sees the exact same dtype regardless
of whether a template was built at construction or loaded from the
catalog.

``catalog.json`` records the exact constants each entry was generated
under (``channel_width_hz``, ``channel_output_rate``, ``num_channels``,
``base_freq_hz``) — checked against the CURRENT constants before
trusting the array, so a catalog generated under since-corrected
constants fails loudly instead of silently misapplying stale data. This
isn't a hypothetical: this project has already been burned once by a
wrong ``channel_output_rate`` assumption (see CLAUDE.md's "SPS-CBF ICD
channelization" section) — a catalog baked before that fix, loaded
after it, is exactly the kind of drift this check exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ska_low_station_beam_simulator.common import (
    BASE_FREQ_HZ,
    CHANNEL_OUTPUT_RATE_HZ,
    CHANNEL_WIDTH_HZ,
    MAX_NUM_CHANNELS,
)

# Not bundled via wheel packaging (see CLAUDE.md's "Pulsar catalog"
# section for why: generated data this large shouldn't go through git or
# a package index) -- generate_pulsar_catalog.py writes here by default,
# meant to be populated as an OCI image build step instead;
# DirectSynthesisStreamer reads from here by default. A source_cfg's
# optional `catalog_dir` overrides this per-source, e.g. for tests.
DEFAULT_CATALOG_DIR = Path(__file__).parent / "pulsar_catalog_data"

CATALOG_FILENAME = "catalog.json"

# The set of named pulsars this project ships pre-generated. Deliberately
# small and illustrative, not an attempt at a complete or astrophysically
# curated set -- extend this list and re-run generate_pulsar_catalog.py
# to add more. Each gets its OWN sky_seed (not DEFAULT_SKY_SEED) so two
# entries are never accidentally correlated even if a future addition
# happened to produce the same n_wide as an existing one.
CATALOG_ENTRIES = [
    {
        "name": "fast_test",
        "period_s": 0.01,
        "width_s": 0.0005,
        "dm_pc_cm3": 2.0,
        "sky_seed": 0x5AB1E5EED + 1,
    },
    {
        "name": "vela_like",
        # Illustrative, not a precise reproduction: real Vela (PSR
        # B0833-45) is ~89.3ms period, DM ~67.97 pc/cm^3.
        "period_s": 0.0893,
        "width_s": 0.003,
        "dm_pc_cm3": 68.0,
        "sky_seed": 0x5AB1E5EED + 2,
    },
    {
        "name": "slow_wide",
        # A longer period than this project's per-tick generation would
        # ever build live within the one-time construction budget (see
        # CLAUDE.md) -- exactly the case pre-generation is FOR: this
        # construction cost is paid once, offline, by
        # generate_pulsar_catalog.py, not at every scan start.
        "period_s": 0.3,
        "width_s": 0.015,
        "dm_pc_cm3": 30.0,
        "sky_seed": 0x5AB1E5EED + 3,
    },
]


def save_pulsar_to_catalog(
    catalog_dir: Path,
    name: str,
    template: np.ndarray,
    n_period_samples: int,
    period_s: float,
    width_s: float,
    dm_pc_cm3: float,
    sky_seed: int,
    num_channels: int = MAX_NUM_CHANNELS,
    base_freq_hz: float = BASE_FREQ_HZ,
    channel_width_hz: float = CHANNEL_WIDTH_HZ,
    channel_output_rate: float = CHANNEL_OUTPUT_RATE_HZ,
) -> None:
    """Writes ``<name>.npy`` (complex64) plus this entry's metadata into
    ``catalog_dir``'s shared catalog.json, merging with whatever entries
    are already there (so generate_pulsar_catalog.py can add one pulsar
    at a time without clobbering the rest).

    :param catalog_dir: directory to write into; created if missing.
    :param name: the catalog entry's name (e.g. ``"vela_like"``) --
        becomes ``<name>.npy``'s filename and the key under which this
        entry is stored in catalog.json.
    :param template: the full-band-width ``(num_channels,
        n_period_samples)`` complex template, as returned by
        ``direct_synthesis.build_pulsar_template`` -- cast to complex64
        before writing.
    :param n_period_samples: samples per pulsar period, i.e.
        ``template.shape[1]``.
    :param period_s: the pulsar's rotation period, in seconds.
    :param width_s: the pulse profile's FWHM, in seconds.
    :param dm_pc_cm3: dispersion measure, in pc/cm^3.
    :param sky_seed: the seed used for this pulsar's shared "sky carrier"
        (see module docstring) -- recorded so a caller can reproduce the
        exact same template independently if needed.
    :param num_channels: how many channels ``template`` spans -- should
        stay at the default (the full band) unless deliberately building
        a narrower catalog entry.
    :param base_freq_hz: the absolute frequency of ``template``'s
        channel 0.
    :param channel_width_hz: the channel spacing ``template`` was built
        with -- recorded for the stale-catalog check in
        ``load_pulsar_from_catalog``.
    :param channel_output_rate: the per-channel sample rate
        ``template`` was built with -- recorded for the same
        stale-catalog check.
    """
    catalog_dir.mkdir(parents=True, exist_ok=True)
    npy_filename = f"{name}.npy"
    np.save(catalog_dir / npy_filename, template.astype(np.complex64))

    catalog_path = catalog_dir / CATALOG_FILENAME
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    catalog[name] = {
        "npy_filename": npy_filename,
        "n_period_samples": n_period_samples,
        "period_s": period_s,
        "width_s": width_s,
        "dm_pc_cm3": dm_pc_cm3,
        "sky_seed": sky_seed,
        "num_channels": num_channels,
        "base_freq_hz": base_freq_hz,
        "channel_width_hz": channel_width_hz,
        "channel_output_rate": channel_output_rate,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True))


def load_pulsar_from_catalog(
    name: str,
    station_num_channels: int,
    station_base_freq_hz: float,
    catalog_dir: Optional[Path] = None,
) -> dict:
    """Loads ``name``'s pre-generated template and slices out the channel
    range [station_base_freq_hz, station_base_freq_hz +
    station_num_channels*channel_width_hz) that this station actually
    needs -- see module docstring for why one catalog entry, generated
    at the full band width, serves every valid station sub-band.

    :param name: the catalog entry's name, as given to
        ``save_pulsar_to_catalog``.
    :param station_num_channels: how many channels the requesting
        station needs -- the returned template is sliced to exactly
        this many columns.
    :param station_base_freq_hz: the absolute frequency of the
        requesting station's own channel 0 -- must fall on this catalog
        entry's channel grid.
    :param catalog_dir: directory to read from; defaults to
        ``DEFAULT_CATALOG_DIR`` if not given.
    :returns: a dict with ``template`` (complex128, upcast from the
        on-disk complex64 -- see module docstring), ``period_s``,
        ``n_period_samples``, ``width_s``, ``dm_pc_cm3``, ``sky_seed`` --
        everything ``DirectSynthesisStreamer`` needs to treat this
        exactly like a template it built itself.
    :raises ValueError: if ``name`` isn't in the catalog; if the catalog
        was generated under ``channel_width_hz``/``channel_output_rate``
        constants that don't match the CURRENT ones (stale-catalog
        protection); or if the requested station sub-band doesn't fit
        inside, or isn't channel-aligned with, the catalog entry's
        generated band.
    """
    catalog_dir = catalog_dir or DEFAULT_CATALOG_DIR
    catalog_path = catalog_dir / CATALOG_FILENAME
    if not catalog_path.exists():
        raise ValueError(
            f"no pulsar catalog found at {catalog_path} -- run "
            f"generate_pulsar_catalog.py first, or pass a catalog_dir "
            f"that has one."
        )
    catalog = json.loads(catalog_path.read_text())
    if name not in catalog:
        raise ValueError(
            f"pulsar_name={name!r} not in catalog at {catalog_path} -- "
            f"available names: {sorted(catalog)}"
        )
    entry = catalog[name]

    if entry["channel_width_hz"] != CHANNEL_WIDTH_HZ:
        raise ValueError(
            f"pulsar_name={name!r} was generated with "
            f"channel_width_hz={entry['channel_width_hz']}, but the "
            f"current CHANNEL_WIDTH_HZ is {CHANNEL_WIDTH_HZ} -- "
            f"regenerate the catalog (see generate_pulsar_catalog.py)."
        )
    if entry["channel_output_rate"] != CHANNEL_OUTPUT_RATE_HZ:
        raise ValueError(
            f"pulsar_name={name!r} was generated with "
            f"channel_output_rate={entry['channel_output_rate']}, but "
            f"the current CHANNEL_OUTPUT_RATE_HZ is "
            f"{CHANNEL_OUTPUT_RATE_HZ} -- this is exactly the kind of "
            f"drift that silently broke this project's own per-tick "
            f"timing once already (see CLAUDE.md's 'SPS-CBF ICD "
            f"channelization' section) -- regenerate the catalog."
        )

    catalog_num_channels = entry["num_channels"]
    catalog_base_freq_hz = entry["base_freq_hz"]
    channel_width_hz = entry["channel_width_hz"]

    offset = (station_base_freq_hz - catalog_base_freq_hz) / channel_width_hz
    slice_start = round(offset)
    if abs(offset - slice_start) > 1e-6:
        raise ValueError(
            f"pulsar_name={name!r}'s catalog band starts at "
            f"{catalog_base_freq_hz}Hz -- station base_freq_hz="
            f"{station_base_freq_hz}Hz is not aligned to that catalog's "
            f"channel grid ({channel_width_hz}Hz spacing)."
        )
    if not (0 <= slice_start and slice_start + station_num_channels <= catalog_num_channels):
        raise ValueError(
            f"pulsar_name={name!r}'s catalog covers {catalog_num_channels} "
            f"channels starting at {catalog_base_freq_hz}Hz -- station's "
            f"requested range ({station_num_channels} channels starting "
            f"at {station_base_freq_hz}Hz) doesn't fit inside it."
        )

    full_template = np.load(catalog_dir / entry["npy_filename"])
    template = np.ascontiguousarray(
        full_template[slice_start : slice_start + station_num_channels].astype(np.complex128)
    )

    return {
        "template": template,
        "period_s": entry["period_s"],
        "n_period_samples": entry["n_period_samples"],
        "width_s": entry["width_s"],
        "dm_pc_cm3": entry["dm_pc_cm3"],
        "sky_seed": entry["sky_seed"],
    }

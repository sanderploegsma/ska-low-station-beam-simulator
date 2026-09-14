import json

import pytest

from ska_low_station_beam_simulator.common import (
    DelayPolynomial,
    parse_delay_polynomial_from_attr_value,
)


@pytest.fixture
def delay_model():
    return json.dumps(
        {
            "interface": "https://schema.skao.int/ska-low-csp-delaymodel/1.0",
            "start_validity_sec": 1234,
            "cadence_sec": 10,
            "validity_period_sec": 600,
            "config_id": "",
            "station_beam": 0,
            "subarray": 0,
            "station_beam_delays": [
                {
                    "station_id": 340,
                    "substation_id": 0,
                    "xypol_coeffs_ns": [1, 2, 3, 4, 5, 6],
                    "ypol_offset_ns": 42,
                }
            ],
        }
    )


@pytest.fixture
def parsed_delay_model(delay_model: str):
    return parse_delay_polynomial_from_attr_value(delay_model, 340)


def test_parse_delay_start_validity(parsed_delay_model: DelayPolynomial):
    assert parsed_delay_model.start_validity_sec == 1234


def test_parse_delay_validity_period(parsed_delay_model: DelayPolynomial):
    assert parsed_delay_model.validity_period_sec == 600


def test_parse_delay_xypol_coeffs(parsed_delay_model: DelayPolynomial):
    assert parsed_delay_model.xypol_coeffs_ns == [1, 2, 3, 4, 5, 6]


def test_parse_delay_ypol_offset(parsed_delay_model: DelayPolynomial):
    assert parsed_delay_model.ypol_offset_ns == 42


def test_parse_delay_polynomial_from_attr_value_with_unknown_station_id(
    delay_model: str,
):
    assert parse_delay_polynomial_from_attr_value(delay_model, 999) is None


def test_parse_delay_polynomial_from_attr_value_with_invalid_json():
    invalid_json = "{invalid json}"
    assert parse_delay_polynomial_from_attr_value(invalid_json, 340) is None


def test_parse_delay_polynomial_from_attr_value_with_unsupported_interface():
    unsupported_interface_json = json.dumps(
        {
            "interface": "https://unsupported.interface",
            "start_validity_sec": 1234,
            "validity_period_sec": 600,
            "station_beam_delays": [
                {
                    "station_id": 340,
                    "xypol_coeffs_ns": [1, 2, 3],
                    "ypol_offset_ns": 42,
                }
            ],
        }
    )

    assert (
        parse_delay_polynomial_from_attr_value(unsupported_interface_json, 340) is None
    )

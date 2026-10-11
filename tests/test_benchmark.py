import pytest

from rev_ct.benchmark import summarise


def test_summary_uses_total_wall_time_for_serial_throughput():
    result = summarise([{"total_s": 1.0}, {"total_s": 3.0}])
    assert result["p50_s"] == 2.0
    assert result["volumes_per_second"] == 0.5
    assert result["n"] == 2


def test_empty_timings_are_rejected():
    with pytest.raises(ValueError):
        summarise([])

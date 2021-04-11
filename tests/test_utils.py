
import smttask.utils as utils
from pathlib import Path

import pytest

def test_relative_path():
    dst  = Path("/home/User/data/output/file")
    src0 = Path("/home/User")
    src1 = Path("/home/User/data/")
    src2 = Path("/home/User/data/file")
    src3 = Path("/home/User/data/input/file")
    src4 = Path("/home/User/data/output/file")

    assert utils.relative_path(src0, dst) == Path("data/output/file")
    assert utils.relative_path(src1, dst) == Path("output/file")
    assert utils.relative_path(src2, dst) == Path("../output/file")
    assert utils.relative_path(src3, dst) == Path("../../output/file")
    assert utils.relative_path(src4, dst) == Path("")

def test_parse_duration_str():
    assert utils.parse_duration_str("1min") == 60
    assert utils.parse_duration_str("1 min") == 60
    assert utils.parse_duration_str("1m") == 60
    assert utils.parse_duration_str("1 m") == 60
    assert utils.parse_duration_str("1m1m1m") == 180
    assert utils.parse_duration_str("1h23m2s") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1h23min2s") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1h 23min 2s") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1hour23min2s") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1hour23min2sec") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1hour23minutes2sec") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1hour23minutes2seconds") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1hour 23minutes 2seconds") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1 hour 23 minutes 2 seconds") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1 hour 23 minute 2 second") == 60**2 + 23*60 + 2
    assert utils.parse_duration_str("1 hours 23 minute 2 second") == 60**2 + 23*60 + 2

    with pytest.raises(ValueError):
        utils.parse_duration_str("1moon") == 60

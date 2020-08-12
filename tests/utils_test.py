
import smttask.utils as utils
from pathlib import Path

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

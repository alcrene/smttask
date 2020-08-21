import re
from mackelab_toolbox.utils import stablehexdigest

def wip_test_parse_unhashed_params():
    hashdigest = stablehexdigest(91)[:10]
    pname = "stepi"
    ps = [f"{hsh}.json"]
    nm = ""
    for i in [100, 3, 541]:
        ps.append(f"{hashdigest}__{pname}_{i}_{nm}.json")
    re_outfile = f"{re.escape(hashdigest)}__{re.escape(pname)}_(\d*)_([a-zA-Z0-9]*).json$"

    outfiles = {}
    for p in ps:
        m = re.match(re_outfile, p)
        if m is not None:
            assert len(m.groups()) == 2
            itervalue, varname = m.groups()
            outfiles[itervalue] = varname

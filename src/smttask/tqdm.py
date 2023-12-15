"""
The purpose of this module is to provide a `tqdm` progress bar object with
globally configurable defaults.
The motivation is that progress bars are often created at some low to
intermediate level, but the code which can determine the progress bar options
is usually at the top level

For example, the script called by the user
may receive a `--progress-interval` argument indicating how often to update
the progress bar. This script can set the defaults here, and any other module
importing `tqdm` from here will inherit them.
"""

from typing import Optional, Union
from dataclasses import dataclass
import io
from tqdm.auto import tqdm as autotqdm

__all__ = ["tqdm"]

@dataclass
class TqdmDefaults:
    position   : Optional[int]=None
    file       : Optional[Union[io.TextIOWrapper, io.StringIO]]=None
    mininterval: float=0.1
    miniters   : Optional[int]=None
    disable    : Union[bool,None]=False

class tqdm(autotqdm):
    defaults = TqdmDefaults()
    
    def __init__(self, *args, **kwargs):
        defaults = self.defaults.__dict__.copy()
        super().__init__(*args, **{**defaults, **kwargs})
        

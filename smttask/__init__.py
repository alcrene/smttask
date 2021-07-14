# Workaround for Django import conflict:
# Sumatra requires us to use an old version of Django (< 2), which technically
# conflicts with Bokeh (and therefore Holoviews).
# Things seem to work, as long as Holoviews is imported before django
# We can at least save users from remembering this import order in many cases 
# by forcing a Holoviews import here. We don't actually want it in the namespace
# though, so we delete it immediately (it will continue to live in sys.modules)
from warnings import warn
try:
    import holoviews
except Exception:
    # In addition to holoviews not being installed, we may already be too late
    # (django may already have been loaded), in which case there's nothing we
    # can do but hope that holoviews won't be needed.
    warn("A test import of `holoviews` failed. If you want to use `holoviews` "
         "for plotting, remember to import it before `smttask`.")
else:
    del holoviews
# End workaround    

from .base import *
from .decorators import *
from .task_generators import *

from .config import config

from .view import RecordView, RecordStoreView
from .view.config import config as view_config
# Add smttask-specific filters. These get registered in view.recordfilter.RecordFilter
from . import task_filters
# Configure default RecordStoreView params to split level names on '.' and ignore levels named 'inputs'
from .utils import get_task_param
view_config.get_field_value = get_task_param

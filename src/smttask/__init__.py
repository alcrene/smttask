# # Workaround for Django import conflict:
# # Sumatra requires us to use an old version of Django (< 2), which technically
# # conflicts with Bokeh (and therefore Holoviews).
# # Bokeh's dependence on Django is optional(?), and the failing code is only
# # triggered if Django is already imported (it's under a `if 'django' in
# # sys.modules` guard). By temporarily removing 'django' from imported modules
# # (if present), we allow both Smttask and Bokeh (or Holoviews) to be imported
# # without error.
# # We can at least save users from remembering this import order in many cases 
# # by forcing a Holoviews import here. We don't actually want it in the namespace
# # though, so we delete it immediately (it will continue to live in sys.modules)
# from warnings import warn
# from mackelab_toolbox.meta import HideModule
# with HideModule('django'):
#     try:
#         import holoviews
#     except Exception:
#         # Most likely Holoviews is simply not installed.
#         warn("Unable to import Holoviews; most Smttask visualization functions "
#              "will not work.")
#     else:
#         del holoviews

# Alternative workaround: Force use of ShelveRecordStore, so the Django
# version does not matter.
from sumatra import recordstore as sumatra_recordstore
sumatra_recordstore.DefaultRecordStore = sumatra_recordstore.ShelveRecordStore

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

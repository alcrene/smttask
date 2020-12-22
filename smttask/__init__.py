from .base import *
from .decorators import *

from .config import config

from .view import RecordView, RecordStoreView
from .view.config import config as view_config
# Add smttask-specific filters. These get registered in view.recordfilter.RecordFilter
from . import task_filters
# Configure default RecordStoreView params to split level names on '.' and ignore levels named 'inputs'
from .utils import get_task_param
view_config.get_field_value = get_task_param

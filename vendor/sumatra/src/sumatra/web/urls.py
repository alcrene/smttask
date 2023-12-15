"""
Define URL dispatching for the Sumatra web interface.

:copyright: Copyright 2006-2015 by the Sumatra team, see doc/authors.txt
:license: BSD 2-clause, see LICENSE for details.
"""

from django.conf.urls import patterns
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from ..projects import Project
from ..records import Record
from .views import (ProjectListView, ProjectDetailView, RecordListView,
                    RecordDetailView, DataListView, DataDetailView,
                    ImageListView, SettingsView, DiffView)

P = {
    'project': Project.valid_name_pattern,
    'label': Record.valid_name_pattern,
}

urlpatterns = patterns('',
                       (r'^$', ProjectListView.as_view()),
                       (r'^settings/$', SettingsView.as_view()),
                       (r'^%(project)s/$' % P, RecordListView.as_view()),
                       (r'^%(project)s/about/$' % P, ProjectDetailView.as_view()),
                       (r'^%(project)s/data/$' % P, DataListView.as_view()),
                       (r'^%(project)s/image/$' % P, ImageListView.as_view()),
                       (r'^%(project)s/image/thumbgrid$' % P, 'sumatra.web.views.image_thumbgrid'),
                       (r'^%(project)s/parameter$' % P, 'sumatra.web.views.parameter_list'),
                       (r'^%(project)s/delete/$' % P, 'sumatra.web.views.delete_records'),
                       (r'^%(project)s/compare/$' % P, 'sumatra.web.views.compare_records'),
                       (r'^%(project)s/%(label)s/$' % P, RecordDetailView.as_view()),
                       (r'^%(project)s/%(label)s/diff$' % P, DiffView.as_view()),
                       (r'^%(project)s/%(label)s/diff/(?P<package>[\w_]+)*$' % P, DiffView.as_view()),
                       (r'^%(project)s/%(label)s/script$' % P, 'sumatra.web.views.show_script'),
                       (r'^%(project)s/data/datafile$' % P, DataDetailView.as_view()),
                       (r'^%(project)s/datatable/record$' % P, 'sumatra.web.views.datatable_record'),
                       (r'^%(project)s/datatable/data$' % P, 'sumatra.web.views.datatable_data'),
                       (r'^%(project)s/datatable/image$' % P, 'sumatra.web.views.datatable_image'),
                       (r'^data/(?P<datastore_id>\d+)$', 'sumatra.web.views.show_content'),
                       )

urlpatterns += staticfiles_urlpatterns()

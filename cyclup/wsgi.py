"""WSGI config for cyclup project."""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cyclup.settings")
application = get_wsgi_application()

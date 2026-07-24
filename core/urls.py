from django.urls import path
from . import views

app_name = "core"

urlpatterns = [
    # Page
    path("", views.index_view, name="index"),

    # Dump endpoints
    path("api/dump/text/", views.dump_text, name="dump_text"),
    path("api/dump/file/", views.dump_file, name="dump_file"),
    path("api/dump/image/", views.dump_image, name="dump_image"),

    # Query endpoints
    path("api/ask/", views.ask_question, name="ask_question"),
    path("api/dumps/", views.list_dumps, name="list_dumps"),

    # Gemma health check
    path("api/gemma/status/", views.gemma_status, name="gemma_status"),
]

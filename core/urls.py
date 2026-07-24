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
    path("api/dump/<int:dump_id>/delete/", views.delete_dump, name="delete_dump"),
    path("api/dumps/delete/bulk/", views.bulk_delete_dumps, name="bulk_delete_dumps"),

    # Query & History endpoints
    path("api/ask/", views.ask_question, name="ask_question"),
    path("api/ask/stream/", views.ask_question_stream, name="ask_question_stream"),
    path("api/dumps/", views.list_dumps, name="list_dumps"),
    path("api/chat/history/", views.list_chat_history, name="list_chat_history"),
    path("api/chat/history/clear/", views.clear_chat_history, name="clear_chat_history"),
    path("api/chat/message/<int:message_id>/delete/", views.delete_chat_message, name="delete_chat_message"),

    # Gemma health check
    path("api/gemma/status/", views.gemma_status, name="gemma_status"),
]

from django.contrib import admin

from .models import Dump, ChatMessage


@admin.register(Dump)
class DumpAdmin(admin.ModelAdmin):
    list_display = ("source_name", "source_type", "page_number", "course_code", "created_at")
    list_filter = ("source_type", "course_code")
    search_fields = ("source_name", "raw_text", "course_code")
    ordering = ("-created_at",)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "user_query", "mode", "status", "latency", "created_at")
    list_filter = ("mode", "status")
    search_fields = ("user_query", "assistant_response")
    ordering = ("-created_at",)


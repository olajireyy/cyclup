from django.contrib import admin

from .models import Dump


@admin.register(Dump)
class DumpAdmin(admin.ModelAdmin):
    list_display = ("source_name", "source_type", "page_number", "course_code", "created_at")
    list_filter = ("source_type", "course_code")
    search_fields = ("source_name", "raw_text", "course_code")
    ordering = ("-created_at",)

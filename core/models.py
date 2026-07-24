from django.db import models


class Dump(models.Model):
    """
    Stores a single chunk of campus information dumped by any user.
    One row per text entry, file, image OCR result, or PDF page.
    """

    SOURCE_TYPE_CHOICES = [
        ("text", "Text"),
        ("txt_file", "Text File"),
        ("docx", "Word Document"),
        ("image", "Image"),
        ("pdf", "PDF"),
    ]

    raw_text = models.TextField()
    source_name = models.CharField(max_length=255, default="Typed_Dump")
    source_type = models.CharField(
        max_length=10,
        choices=SOURCE_TYPE_CHOICES,
        default="text",
    )
    page_number = models.IntegerField(null=True, blank=True)
    campus = models.CharField(max_length=50, default="LASU")
    course_code = models.CharField(max_length=50, null=True, blank=True)
    tags = models.JSONField(default=list, blank=True)
    extracted_json = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Dump"
        verbose_name_plural = "Dumps"

    def __str__(self):
        label = self.source_name
        if self.page_number is not None:
            label += f" (p.{self.page_number})"
        return label


class ChatMessage(models.Model):
    """
    Stores historical chat interactions between student and assistant.
    """

    session_id = models.CharField(max_length=100, default="default")
    user_query = models.TextField()
    assistant_response = models.TextField(blank=True, default="")
    mode = models.CharField(max_length=20, default="fast")
    status = models.CharField(max_length=20, default="grounded")
    sources = models.JSONField(default=list, blank=True)
    latency = models.CharField(max_length=20, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"

    def __str__(self):
        return f"[{self.mode}] {self.user_query[:30]}"


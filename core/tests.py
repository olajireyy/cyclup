import json
from django.test import TestCase, Client
from django.urls import reverse
from core.models import Dump, ChatMessage
from core.views import analyze_user_query, double_net_search, handle_metadata_tool, check_chatter_guard


class CyclupCoreTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.dump1 = Dump.objects.create(
            raw_text="The CSC301 exam will take place on Monday at MBA Hall.",
            source_name="CSC301_Exam_Notice.txt",
            source_type="txt_file",
            course_code="CSC301",
            tags=["exam", "schedule"],
        )
        self.dump2 = Dump.objects.create(
            raw_text="LASU hostel fee is 50000 Naira per session.",
            source_name="Hostel_Fee_2026.pdf",
            source_type="pdf",
            page_number=1,
            course_code="FEES",
            tags=["hostel", "fees"],
        )

    def test_index_view(self):
        response = self.client.get(reverse("core:index"))
        self.assertEqual(response.status_code, 200)

    def test_gemma_status(self):
        response = self.client.get(reverse("core:gemma_status"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("connected", data)

    def test_dump_text(self):
        response = self.client.post(
            reverse("core:dump_text"),
            data=json.dumps({
                "raw_text": "Library opens at 8am daily.",
                "source_name": "Library Notice",
                "course_code": "GENERAL",
                "tags": "library,hours",
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Dump.objects.filter(source_name="Library Notice").count(), 1)

    def test_list_dumps(self):
        response = self.client.get(reverse("core:list_dumps"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dumps", data)
        self.assertGreaterEqual(len(data["dumps"]), 2)

    def test_delete_dump(self):
        response = self.client.post(reverse("core:delete_dump", args=[self.dump1.id]))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Dump.objects.filter(id=self.dump1.id).exists())

    def test_bulk_delete_dumps(self):
        response = self.client.post(
            reverse("core:bulk_delete_dumps"),
            data=json.dumps({"ids": [self.dump2.id]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Dump.objects.filter(id=self.dump2.id).exists())

    def test_chatter_guard(self):
        res = check_chatter_guard("hello assistant")
        self.assertIsNotNone(res)

    def test_handle_metadata_tool(self):
        res = handle_metadata_tool("what files are in vault?")
        self.assertIn("Hostel_Fee_2026.pdf", res)

    def test_double_net_search(self):
        analysis = analyze_user_query("when is csc301 exam?")
        results = double_net_search(analysis, "when is csc301 exam?")
        self.assertTrue(len(results) > 0)
        top_chunk = results[0][1]
        self.assertEqual(top_chunk.source_name, "CSC301_Exam_Notice.txt")

    def test_ask_question_chatter(self):
        response = self.client.post(
            reverse("core:ask_question"),
            data=json.dumps({"question": "hello"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "chatter_guard")

    def test_ask_question_metadata(self):
        response = self.client.post(
            reverse("core:ask_question"),
            data=json.dumps({"question": "show files in vault"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "metadata")

    def test_chat_history_endpoints(self):
        msg = ChatMessage.objects.create(
            user_query="What is hostel fee?",
            assistant_response="50000 Naira",
        )
        # List history
        res1 = self.client.get(reverse("core:list_chat_history"))
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(len(res1.json()["messages"]), 1)

        # Delete message
        res2 = self.client.post(reverse("core:delete_chat_message", args=[msg.id]))
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(ChatMessage.objects.count(), 0)

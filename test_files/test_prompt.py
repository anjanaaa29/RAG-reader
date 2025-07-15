import unittest
from src.prompt_factory import PromptFactory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from unittest.mock import MagicMock


class TestPromptFactory(unittest.TestCase):
    def test_get_base_prompt_defaults(self):
        prompt = PromptFactory.get_base_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)
        template_str = str(prompt)
        self.assertIn("You are a information specialist", template_str)
        self.assertIn("Evidence-based", template_str)
        self.assertIn("1. Clear answer", template_str)

    def test_get_base_prompt_custom(self):
        custom_req = ["Custom requirement"]
        custom_struct = ["Step 1", "Step 2"]
        prompt = PromptFactory.get_base_prompt(
            domain="health", requirements=custom_req, structure=custom_struct
        )
        template_str = str(prompt)
        self.assertIn("You are a health specialist", template_str)
        self.assertIn("Custom requirement", template_str)
        self.assertIn("Step 1", template_str)

    def test_get_specialized_prompt_defaults(self):
        prompt = PromptFactory.get_specialized_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)
        template_str = str(prompt)
        self.assertIn("sensitive information", template_str)
        self.assertIn("Provide balanced information", template_str)
        self.assertIn("Example structure", template_str)

    def test_get_specialized_prompt_custom(self):
        custom_warning = "Custom Warning"
        custom_req = ["Custom req 1", "Custom req 2"]
        custom_struct = ["Custom struct"]
        prompt = PromptFactory.get_specialized_prompt(
            warning=custom_warning, requirements=custom_req, example_structure=custom_struct
        )
        template_str = str(prompt)
        self.assertIn("Custom Warning", template_str)
        self.assertIn("Custom req 1", template_str)
        self.assertIn("Custom struct", template_str)

    def test_get_citation_format_defaults(self):
        formats = PromptFactory.get_citation_format()
        self.assertIsInstance(formats, dict)
        self.assertIn("government", formats)
        self.assertIn("{agency}", formats["government"])

    def test_get_citation_format_custom(self):
        custom = {"custom": "{custom}"}
        formats = PromptFactory.get_citation_format(custom)
        self.assertEqual(formats["custom"], "{custom}")

    def test_get_disclaimers_defaults(self):
        disclaimers = PromptFactory.get_disclaimers()
        self.assertIsInstance(disclaimers, dict)
        self.assertIn("sensitive", disclaimers)
        self.assertIn("educational purposes", disclaimers["default"])

    def test_get_disclaimers_custom(self):
        custom = {"extra": "Extra disclaimer"}
        disclaimers = PromptFactory.get_disclaimers(custom)
        self.assertIn("extra", disclaimers)
        self.assertEqual(disclaimers["extra"], "Extra disclaimer")

    def test_select_prompt_match(self):
        default_prompt = MagicMock()
        prompt_map = {
            "urgent": MagicMock(),
            "health": MagicMock()
        }
        result = PromptFactory.select_prompt(
            "This is an urgent question",
            prompt_map,
            default_prompt
        )
        self.assertEqual(result, prompt_map["urgent"])

    def test_select_prompt_default(self):
        default_prompt = MagicMock()
        prompt_map = {
            "health": MagicMock()
        }
        result = PromptFactory.select_prompt(
            "This is a general question",
            prompt_map,
            default_prompt
        )
        self.assertEqual(result, default_prompt)


if __name__ == "__main__":
    unittest.main()

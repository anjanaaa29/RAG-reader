import unittest
from generic_citation_formatter import GenericCitationFormatter


class TestGenericCitationFormatter(unittest.TestCase):

    def setUp(self):
        self.guideline = {
            "source_type": "official_guideline",
            "organization": "WHO",
            "title": "Global Guidelines for the Prevention and Treatment of Cardiovascular Diseases",
            "version": "2.1",
            "report_number": "WH-1234",
            "year": "2022"
        }

        self.research = {
            "source_type": "research_article",
            "authors": "Smith, J, Johnson, A, Williams, B",
            "title": "Effects of Exercise on Cardiovascular Health in Middle-Aged Adults",
            "journal": "Journal of Medical Research",
            "year": "2021",
            "doi": "10.1234/jmr.2021.12345"
        }

        self.educational = {
            "source_type": "educational_material",
            "publisher": "Harvard Medical School",
            "title": "Comprehensive Guide to Cardiovascular Health",
            "year": "2023"
        }

        self.web = {
            "source_type": "web_page",
            "site_name": "Mayo Clinic",
            "page_title": "Heart Disease: Symptoms and Causes",
            "url": "https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118",
            "publish_date": "2023-05-15",
            "access_date": "2023-10-20"
        }

        self.unknown = {
            "source_type": "unknown_type",
            "title": "Some Unknown Document"
        }

        self.empty = {}

    def test_format_guideline(self):
        citation = GenericCitationFormatter.format_citation(self.guideline)
        self.assertIn("WHO", citation)
        self.assertIn("Global Guidelines", citation)
        self.assertIn("WH-1234", citation)
        self.assertIn("v2.1", citation)
        self.assertIn("2022", citation)

    def test_format_research(self):
        citation = GenericCitationFormatter.format_citation(self.research)
        self.assertIn("Smith", citation)
        self.assertIn("Effects of Exercise", citation)
        self.assertIn("Journal of Medical Research", citation)
        self.assertIn("2021", citation)
        self.assertIn("10.1234", citation)

    def test_format_educational(self):
        citation = GenericCitationFormatter.format_citation(self.educational)
        self.assertIn("Harvard Medical School", citation)
        self.assertIn("Comprehensive Guide", citation)
        self.assertIn("2023", citation)

    def test_format_web(self):
        citation = GenericCitationFormatter.format_citation(self.web)
        self.assertIn("Mayo Clinic", citation)
        self.assertIn("Heart Disease", citation)
        self.assertIn("2023", citation)
        self.assertIn("URL", citation)
        self.assertIn("accessed 2023-10-20", citation)

    def test_format_unknown(self):
        citation = GenericCitationFormatter.format_citation(self.unknown)
        self.assertIn("Some Unknown Document", citation)
        self.assertIn("Source", citation)

    def test_format_empty(self):
        citation = GenericCitationFormatter.format_citation(self.empty)
        self.assertIn("Source", citation)

    def test_extract_year(self):
        cases = [
            ({"year": "2020"}, "2020"),
            ({"publication_date": "Jan 15, 2019"}, "2019"),
            ({"date": "2018-12-31"}, "2018"),
            ({"publish_date": "Published: 2017"}, "2017"),
            ({"random_field": "nothing"}, ""),
            ({}, "")
        ]
        for metadata, expected in cases:
            with self.subTest(metadata=metadata):
                result = GenericCitationFormatter._extract_year(metadata)
                self.assertEqual(result, expected)

    def test_shorten_title(self):
        long_title = "This is a very long document title that should be shortened"
        result = GenericCitationFormatter._shorten_title(long_title)
        self.assertEqual(result, "This is a very long...")

        short_title = "Short Title"
        result = GenericCitationFormatter._shorten_title(short_title)
        self.assertEqual(result, "Short Title")

    def test_shorten_title_custom_max(self):
        long_title = "One two three four five six seven"
        result = GenericCitationFormatter._shorten_title(long_title, max_words=3)
        self.assertEqual(result, "One two three...")

    def test_shorten_url(self):
        long_url = "https://www.example.com/path/to/a/very/long/resource"
        result = GenericCitationFormatter._shorten_url(long_url, max_length=30)
        self.assertEqual(result, long_url[:30] + "...")

        short_url = "https://short.url"
        result = GenericCitationFormatter._shorten_url(short_url)
        self.assertEqual(result, short_url)

    def test_generate_footnotes(self):
        sources = [self.guideline, self.research, self.web]
        footnotes = GenericCitationFormatter.generate_source_footnotes(sources)
        self.assertIn("**References:**", footnotes)
        self.assertIn("1. [", footnotes)
        self.assertIn("2. [", footnotes)
        self.assertIn("3. [", footnotes)

    def test_generate_footnotes_empty(self):
        footnotes = GenericCitationFormatter.generate_source_footnotes([])
        self.assertEqual(footnotes, "")


if __name__ == "__main__":
    unittest.main()

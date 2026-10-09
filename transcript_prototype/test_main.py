import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from transcript_prototype import main as app


class PrototypeTests(unittest.TestCase):
    def test_urls(self):
        for url in ['https://youtu.be/abcdefghijk?t=1', 'https://www.youtube.com/watch?v=abcdefghijk', 'https://youtube.com/shorts/abcdefghijk']:
            self.assertEqual(app.video_id(url), 'abcdefghijk')
        with self.assertRaises(ValueError):
            app.video_id('https://example.com/abcdefghijk')

    def test_evidence_and_chunks(self):
        segments = [{'text': 'A vector', 'start': 1}, {'text': 'database stores embeddings.', 'start': 2}]
        self.assertEqual(app.occurrences('vector database', segments)[0]['start_seconds'], 1)
        self.assertEqual(app.occurrences('data', segments), [])
        self.assertEqual(app.occurrences('invented term', segments), [])
        self.assertGreater(len(list(app.chunks([{'text': 'a'*15000}]))), 2)

    def test_persistence_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'episode.txt'
            source.write_text('We use vector databases and embeddings.')
            candidates = [{'term': 'vector databases', 'explanation': 'Store vectors.'},
                          {'term': 'VECTOR DATABASES', 'explanation': 'Duplicate.'},
                          {'term': 'hallucinated term', 'explanation': 'Absent.'}]
            with patch.object(app, 'discover', return_value=candidates), patch.object(app, 'lookup', side_effect=OSError('offline')):
                first = app.run(str(source), root/'episode.json', root/'glossary.json')
                second = app.run(str(source), root/'episode.json', root/'glossary.json')
            self.assertEqual(len(first['new_terms']), 1)
            self.assertEqual(first['new_terms'][0]['information_status'], 'model_generated_unverified')
            self.assertEqual(first['new_terms'][0]['sources'], [])
            self.assertEqual(second['new_terms'], [])
            self.assertEqual(second['known_terms'], ['vector databases'])
            self.assertEqual(json.loads((root/'episode.json').read_text())['status'], 'complete')

    def test_analysis_failure_retains_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'episode.txt'
            source.write_text('An embedding is a vector.')
            with patch.object(app, 'discover', side_effect=OSError('Ollama unavailable')):
                with self.assertRaises(OSError):
                    app.run(str(source), root/'episode.json', root/'glossary.json')
            saved = json.loads((root/'episode.json').read_text())
            self.assertEqual(saved['status'], 'transcribed')
            self.assertTrue(saved['transcript'])
            self.assertFalse((root/'glossary.json').exists())

    def test_reference_match(self):
        with patch.object(app, 'request_json', return_value={'query': {'pages': {'1': {
            'title': 'Embedding', 'extract': 'A vector representation. Used in models.', 'fullurl': 'https://en.wikipedia.org/wiki/Embedding'}}}}):
            self.assertEqual(app.lookup('Embedding')['information_status'], 'reference_summary')
            self.assertIsNone(app.lookup('unrelated'))


if __name__ == '__main__':
    unittest.main()

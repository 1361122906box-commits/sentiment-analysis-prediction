import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import reproduce as app


class DemoTests(unittest.TestCase):
    def test_sample_has_separate_three_class_splits(self):
        rows = app.read_data(app.SAMPLE)
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({r['content'] for r in rows}), 36)

    def test_rejects_cross_split_duplicate(self):
        rows = app.read_data(app.SAMPLE)
        rows[-1]['content'] = rows[0]['content']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'different splits'):
                app.read_data(path)

    def test_export_is_self_contained_and_preserves_existing_output(self):
        rows = app.read_data(app.SAMPLE)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / '演示'
            guesses = [app.rule_predict(row['content']) for row in rows]
            app.export_dashboard(rows, guesses, output, '虚构规则演示')
            manifest = json.loads((output / 'events.json').read_text(encoding='utf-8'))
            event = manifest['events'][0]
            visual = json.loads((output / event['visualFile']).read_text(encoding='utf-8'))
            results = json.loads((output / event['resultFile']).read_text(encoding='utf-8'))
            self.assertEqual(len(results), 36)
            self.assertEqual(sum(visual['sentiment_distribution']['values']), 36)
            self.assertEqual(len(visual['time_series']), 36)
            self.assertEqual(len(visual['predictions']), 6)
            self.assertTrue((output / 'vendor/echarts.min.js').is_file())
            self.assertNotIn('https://', (output / 'index.html').read_text(encoding='utf-8'))
            before = (output / event['visualFile']).read_bytes()
            with self.assertRaises(FileExistsError):
                app.export_dashboard(rows, guesses, output, 'replacement')
            self.assertEqual(before, (output / event['visualFile']).read_bytes())

    def test_vendor_hashes_match_sources(self):
        vendor = app.ROOT / '可交互的可视化大屏/vendor'
        for record in json.loads((vendor / 'sources.json').read_text()):
            asset = vendor / (record['name'] + '.min.js')
            self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), record['sha256'])

    def test_negation_and_invalid_prediction_count(self):
        self.assertEqual(app.rule_predict('不满意'), 'negative')
        with self.assertRaises(ValueError):
            app.export_dashboard(app.read_data(app.SAMPLE), [], Path('unused'), 'invalid')


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('torch'),
                     'Optional ML dependencies not installed')
class MLIntegrationTests(unittest.TestCase):
    def test_tiny_bert_training_reload_evaluation_and_lstm(self):
        # Random tiny BERT verifies plumbing offline; it is not a downloaded pretrained model.
        from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / 'base'
            base.mkdir()
            characters = sorted(set(''.join(r['content'] for r in app.read_data(app.SAMPLE))))
            vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + characters
            (base / 'vocab.txt').write_text('\n'.join(vocabulary) + '\n', encoding='utf-8')
            tokenizer = BertTokenizerFast(vocab_file=str(base / 'vocab.txt'))
            tokenizer.save_pretrained(base)
            config = BertConfig(vocab_size=len(vocabulary), hidden_size=24, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32)
            BertForMaskedLM(config).save_pretrained(base)
            trained = Path(folder) / 'trained'
            app.train_model(SimpleNamespace(data=app.SAMPLE, base_model=base, model=trained,
                                             epochs=1, device='cpu', head_only=False))
            self.assertTrue((trained / 'model.safetensors').is_file())
            output = Path(folder) / 'dashboard'
            app.analyze(SimpleNamespace(data=app.SAMPLE, model=trained, output=output,
                                         device='cpu', lstm_epochs=1))
            metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
            for name in ('bert', 'svm'):
                self.assertEqual(metrics[name]['macro avg']['support'], 9)
            visual = json.loads((output / 'event-1-visual.json').read_text(encoding='utf-8'))
            self.assertIn('LSTM', visual['event_info']['sentiment_summary'])
            self.assertEqual(len(visual['predictions']), 6)


if __name__ == '__main__':
    unittest.main()

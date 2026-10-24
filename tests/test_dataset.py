"""Dataset contracts tested with a real fast tokenizer and tiny Gemma model."""
import json
from collections import Counter
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, GemmaConfig, GemmaForCausalLM
from src.dataset.names import build_name_pools, single_token
from src.dataset.generate import generate_split, length_batches, tokenize_with_bos
from src.dataset.calibration import calibrate
from src.dataset.templates import TEMPLATES


def tokenizer():
    candidates = json.loads(Path('data/raw/names.json').read_text())
    words = set(n for group in candidates.values() for n in group)
    words.update('She He she he him her Sequence apple book chair clock cloud desk door fish garden glass grass moon paper river rock sand sky star stone sun table tree water wind'.split())
    words.update('Two Pieces'.split())
    vocab = {'[UNK]': 0, '[BOS]': 1, **{word: i+2 for i, word in enumerate(sorted(words))}}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='[BOS]', unk_token='[UNK]', clean_up_tokenization_spaces=False)


def test_names_disjoint_and_single_token():
    tok = tokenizer()
    candidates = json.loads(Path('data/raw/names.json').read_text())
    pools = build_name_pools(tok, candidates, 42)
    sets = []
    for pool in pools.values():
        names = []
        for gender in ('male', 'female'):
            for name in pool[gender]:
                assert tok.encode(' '+name['name'], add_special_tokens=False) == [name['token_id']]
                names.append(name['name'])
        sets.append(set(names))
    assert not sets[0] & sets[1]
    assert single_token(tok, 'Two Pieces') is None


def test_spans_twins_and_induction():
    tok = tokenizer()
    pools = build_name_pools(tok, json.loads(Path('data/raw/names.json').read_text()), 42)
    all_records = {}
    for split in ('discovery', 'validation'):
        records, twins = generate_split(tok, pools[split], split, 20, 42)
        assert Counter(r['task'] for r in records) == dict.fromkeys(['name_tracking', 'coreference', 'induction', 'local_context'], 20)
        all_records[split] = records
        by_id = {r['id']: r for r in records}
        assert len(twins) == 40
        for twin in twins:
            clean = by_id[twin['clean_id']]
            assert len(clean['token_ids']) == len(twin['token_ids'])
            assert twin['correct_token_id'] == clean['distractor_token_id']
            assert twin['correct_token_id'] != clean['correct_token_id']
            assert clean['token_ids'] != twin['token_ids']
        for record in records:
            ids = record['token_ids']
            assert ids[0] == tok.bos_token_id
            assert len(record['token_categories']) == len(ids)
            for name, positions in record['name_positions'].items():
                assert all(ids[i] == single_token(tok, name) for i in positions)
            for span in record['key_spans']:
                assert 0 <= span['key'] < span['query'] < len(ids)
                if record['task'] == 'name_tracking':
                    assert ids[span['query']] == ids[span['key']]
                elif record['task'] == 'coreference':
                    expected = record['names'][1] if ids[span['query']] in {single_token(tok, 'She'), single_token(tok, 'her')} else record['names'][0]
                    assert ids[span['key']] == single_token(tok, expected)
                elif record['task'] == 'induction':
                    assert ids[span['key']-1] == ids[span['query']]
                else:
                    assert span['query']-span['key'] == 1
        for batch in length_batches(records, 3):
            assert len({len(r['token_ids']) for r in batch}) == 1
        local_sources = {by_id[r['source_id']]['task'] for r in records if r['task']=='local_context'}
        assert local_sources == {'name_tracking', 'coreference'}
        for task in ('name_tracking', 'coreference'):
            male_ids = {n['token_id'] for n in pools[split]['male']}
            assert sum(r['correct_token_id'] in male_ids for r in records if r['task'] == task) == 10
            assert {r['num_distractors'] for r in records if r['task']==task} == {0, 1, 2}
        assert generate_split(tok, pools[split], split, 20, 42) == (records, twins)
    for task in ('name_tracking', 'coreference'):
        assert {t['text'] for t in TEMPLATES['discovery'][task]}.isdisjoint({t['text'] for t in TEMPLATES['validation'][task]})


def test_calibration_is_computed_from_logits():
    tok = tokenizer()
    pools = build_name_pools(tok, json.loads(Path('data/raw/names.json').read_text()), 42)
    records, _ = generate_split(tok, pools['discovery'], 'discovery', 3, 42)
    torch.manual_seed(3)
    model = GemmaForCausalLM(GemmaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, attn_implementation='eager')).eval()
    rows, report = calibrate(model, records, 2)
    assert len(rows) == 6
    record = next(r for r in records if r['id'] == rows[0]['id'])
    with torch.no_grad():
        logits = model(torch.tensor([record['token_ids']])).logits[0, -1].float()
    expected = (logits[record['correct_token_id']] - logits[record['distractor_token_id']]).item()
    assert abs(rows[0]['logit_difference']-expected) < 1e-5
    assert all(0 <= row['accuracy'] <= 1 and row['n'] > 0 for row in report)


def test_calibration_resume_skips_completed():
    tok = tokenizer()
    pool = build_name_pools(tok, json.loads(Path('data/raw/names.json').read_text()), 42)['discovery']
    records, _ = generate_split(tok, pool, 'discovery', 2, 42)
    torch.manual_seed(3)
    model = GemmaForCausalLM(GemmaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, attn_implementation='eager')).eval()
    rows, report = calibrate(model, records, 2)
    snapshots = []
    resumed, resumed_report = calibrate(model, records, 2, existing=rows[:2], checkpoint=lambda state: snapshots.append(list(state)))
    assert {row['id'] for row in resumed} == {row['id'] for row in rows}
    assert len(snapshots) > 0
    assert sorted(resumed_report, key=lambda r:r['template_id']) == sorted(report, key=lambda r:r['template_id'])

"""BOS-aware task datasets with aligned minimal-edit corrupted twins."""
import random
import re
from collections import defaultdict
from .names import single_token
from .templates import TEMPLATES, DISTANCE_FILLERS


def tokenize_with_bos(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Add exactly one BOS and preserve offsets for explicit span alignment."""
    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must define BOS")
    result = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tokenizer.bos_token_id, *result["input_ids"]], [(0, 0), *map(tuple, result["offset_mapping"])]


def word_positions(text: str, offsets: list, word: str) -> list[int]:
    """Find token indices overlapping whole-word spans, rejecting split words."""
    positions = []
    for match in re.finditer(r"\b" + re.escape(word) + r"\b", text):
        hits = [i for i, (start, end) in enumerate(offsets) if end > match.start() and start < match.end()]
        if len(hits) != 1:
            raise ValueError(f"Expected one token at {word!r} span; got {hits}")
        positions.append(hits[0])
    return positions


def token_categories(tokenizer, ids: list[int], text: str, offsets: list, names: list[str]) -> list[str]:
    """Assign one mutually exclusive token category for later mass accounting."""
    name_positions = {i for name in names for i in word_positions(text, offsets, name)}
    function_words = set("the a an to at in of for with and was had been by because only when while who that it".split())
    categories = ["BOS"]
    for i, (start, end) in enumerate(offsets[1:], 1):
        word = text[start:end].strip().lower()
        if i in name_positions:
            category = "names"
        elif word in {"she", "he", "him", "her", "they", "them"}:
            category = "pronouns"
        elif word and all(c in ".,;:!?-()" for c in word):
            category = "punctuation"
        elif word in function_words:
            category = "function_words"
        else:
            category = "other"
        categories.append(category)
    return categories


def make_record(tokenizer, text: str, task: str, split: str, template_id: str,
                names: list[str], correct: int | None, distractor: int | None,
                n_distractors: int, record_id: str) -> dict:
    """Create task queries and targets using actual tokenizer offsets after BOS."""
    ids, offsets = tokenize_with_bos(tokenizer, text)
    positions = {name: word_positions(text, offsets, name) for name in names}
    for name, indices in positions.items():
        expected = single_token(tokenizer, name)
        if expected is None or any(ids[i] != expected for i in indices):
            raise ValueError(f"Name token changes in context: {name}")
    spans = []
    if task == "name_tracking":
        mentions = positions[names[0]]
        spans = [{"query": mentions[1], "key": mentions[0], "wrong_key": positions[names[1]][0]}]
    elif task == "coreference":
        spans = [{"query": word_positions(text, offsets, "She")[0], "key": positions[names[1]][0], "wrong_key": positions[names[0]][0]},
                 {"query": word_positions(text, offsets, "him")[0], "key": positions[names[0]][0], "wrong_key": positions[names[1]][0]}]
    record = {"id": record_id, "text": text, "token_ids": ids, "task": task,
        "split": split, "template_id": template_id, "names": names,
        "token_labels": tokenizer.convert_ids_to_tokens(ids),
        "name_positions": positions, "key_spans": spans,
        "correct_token_id": correct, "distractor_token_id": distractor,
        "num_distractors": n_distractors, "target_position": len(ids)-1,
        "token_categories": token_categories(tokenizer, ids, text, offsets, names)}
    return record


def generate_split(tokenizer, pool: dict, split: str, count: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Generate each task and one aligned twin per behavioral prompt."""
    rng = random.Random(seed)
    records, twins, base = [], [], []
    for task in ("name_tracking", "coreference"):
        templates = TEMPLATES[split][task]
        for index in range(count):
            male = rng.choice(pool["male"])
            female = rng.choice(pool["female"])
            if task == 'name_tracking' and (index // len(templates)) % 2:
                male, female = female, male
            others = [n for gender in pool.values() for n in gender if n not in (male, female)]
            n_distractors = index % 3
            extra = rng.sample(others, n_distractors)
            filler = DISTANCE_FILLERS[(index // 3) % 3]
            if extra:
                # Gendered distractors occur inside a separate quoted aside.
                filler += ' A note listed ' + ' and '.join(n["name"] for n in extra) + ' as absent.'
            template = templates[index % len(templates)]
            text = template["text"].format(a=male["name"], b=female["name"], filler=filler)
            names = [male["name"], female["name"], *[n["name"] for n in extra]]
            ident = f"{split}-{task}-{index:06d}"
            record = make_record(tokenizer, text, task, split, template["id"], names,
                male["token_id"], female["token_id"], n_distractors, ident)
            # Pronoun swaps reverse the referenced participant while preserving gender agreement.
            if task == "coreference":
                corrupted = re.sub(r'\bshe\b', 'he', text.replace('She', 'He').replace('him', 'her'))
                twin = make_record(tokenizer, corrupted, "local_context", split, template["id"], names,
                    female["token_id"], male["token_id"], n_distractors, ident + '-corrupted')
                twin['task'] = task
                twin['key_spans'] = [
                    {"query": word_positions(corrupted, tokenize_with_bos(tokenizer, corrupted)[1], 'He')[0], "key": record['name_positions'][names[0]][0], "wrong_key": record['name_positions'][names[1]][0]},
                    {"query": word_positions(corrupted, tokenize_with_bos(tokenizer, corrupted)[1], 'her')[0], "key": record['name_positions'][names[1]][0], "wrong_key": record['name_positions'][names[0]][0]}]
            else:
                swapped = re.sub(r"\b(?:" + re.escape(names[0]) + "|" + re.escape(names[1]) + r")\b",
                    lambda m: names[1] if m.group() == names[0] else names[0], text)
                twin = make_record(tokenizer, swapped, task, split, template["id"],
                    [names[1], names[0], *names[2:]], female["token_id"], male["token_id"], n_distractors, ident + '-corrupted')
            if len(twin['token_ids']) != len(record['token_ids']):
                raise ValueError(f"Corrupted length mismatch for {ident}; revise tokenizer-specific templates")
            if task == 'coreference' and (index // len(templates)) % 2:
                record, twin = twin, record
                record['id'], twin['id'] = ident, ident + '-corrupted'
            record['corrupted_id'] = twin['id']
            twin['clean_id'] = ident
            records.append(record)
            twins.append(twin)
            base.append(record)
    words = " apple book chair clock cloud desk door fish garden glass grass moon paper river rock sand sky star stone sun table tree water wind".split()
    vocab = [(word, single_token(tokenizer, word)) for word in words]
    vocab = [(word, token) for word, token in vocab if token is not None]
    if len(vocab) < 8:
        raise ValueError("Need at least eight single-token induction words")
    for index in range(count):
        sequence = rng.sample(vocab, rng.randint(4, min(8, len(vocab))))
        text = 'Sequence:' + ''.join(' ' + word for word, _ in sequence * 2)
        record = make_record(tokenizer, text, 'induction', split, f'{split}_induction', [], None, None, 0, f'{split}-induction-{index:06d}')
        _, offsets = tokenize_with_bos(tokenizer, text)
        occurrences = [word_positions(text, offsets, word) for word, _ in sequence]
        record['key_spans'] = [{'query': occurrences[i][1], 'key': occurrences[i+1][0]} for i in range(len(sequence)-1)]
        record['sequence_token_ids'] = [token for _, token in sequence]
        records.append(record)
    for index in range(count):
        source = base[(index % 2) * count + (index // 2)]
        local = dict(source, id=f'{split}-local_context-{index:06d}', task='local_context',
            source_id=source['id'], correct_token_id=None, distractor_token_id=None,
            key_spans=[{'query': i, 'key': i-1} for i in range(2, len(source['token_ids']))])
        local.pop('corrupted_id', None)
        records.append(local)
    return records, twins


def length_batches(records: list[dict], batch_size: int):
    """Yield no-padding batches grouped by identical token length."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups = defaultdict(list)
    for record in records:
        groups[len(record['token_ids'])].append(record)
    for length in sorted(groups):
        group = groups[length]
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def figure_example(tokenizer) -> dict:
    """Tokenize the specified Figure 1 diagnostic, kept outside research splits."""
    return make_record(tokenizer, "John gave Mary a book. She thanked him.",
        "coreference", "example", "figure1_coreference", ["John", "Mary"],
        None, None, 0, "figure1-coreference")

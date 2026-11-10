"""Tests for the BERT and NuNER adapters against STAND-INS for transformers / gliner.
They verify OUR mapping logic (labels, offsets, thresholds, limits). They cannot
prove the real libraries return exactly this format - the real run does that."""
import sys
import types


class _Stubbed:
    def __init__(self, **modules):
        self.modules, self.saved = modules, {}

    def __enter__(self):
        for k, v in self.modules.items():
            self.saved[k] = sys.modules.get(k)
            sys.modules[k] = v

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _fake_gliner(max_len=300):
    calls = []

    class Model:
        config = types.SimpleNamespace(max_len=max_len)

        def predict_entities(self, text, labels, threshold=0.5):
            calls.append((list(labels), threshold))
            out = []
            for word, lab in (("Anna", "person"), ("Acme", "organization"), ("CEO", "job title"),
                              ("Berlin", "something else")):
                i = text.find(word)
                if i >= 0:
                    out.append({"start": i, "end": i + len(word), "text": word, "label": lab, "score": 0.8})
            return out

    mod = types.ModuleType("gliner")
    mod.GLiNER = types.SimpleNamespace(from_pretrained=lambda name: Model())
    return mod, calls


def test_nuner_label_wording_and_mapping_back():
    from models.nuner_zero import NuNERZero
    mod, calls = _fake_gliner()
    with _Stubbed(gliner=mod):
        m = NuNERZero()
    ents = m.predict("Anna is CEO of Acme in Berlin", {"PER", "ORG", "JOB_TITLE"}, 0.4)
    assert sorted(calls[0][0]) == ["job title", "organization", "person"]     # wording sent to the model
    assert calls[0][1] == 0.4                                                  # threshold passed through
    assert {(e.text, e.label) for e in ents} == {("Anna", "PER"), ("Acme", "ORG"), ("CEO", "JOB_TITLE")}
    assert all(e.text == "Anna is CEO of Acme in Berlin"[e.start:e.end] for e in ents)   # offsets consistent
    assert m.fixed_labels is False                                             # UI shows the entity picker


def test_nuner_never_exceeds_models_own_limit_and_splits_like_gliner():
    from models.nuner_zero import NuNERZero
    mod, _ = _fake_gliner(max_len=200)
    with _Stubbed(gliner=mod):
        assert NuNERZero().max_tokens == 200                                   # lowered to model's limit
        spans = NuNERZero().token_spans("Anna's e-mail, ok.")
    words = ["Anna", "'", "s", "e-mail", ",", "ok", "."]
    assert [("Anna's e-mail, ok."[a:b]) for a, b in spans] == words


def _fake_transformers(results):
    mod = types.ModuleType("transformers")

    class Tok:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
            import re
            return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda n: Tok())
    mod.AutoModelForTokenClassification = types.SimpleNamespace(from_pretrained=lambda n: object())
    mod.pipeline = lambda *a, **k: (lambda text: results)
    return mod


def test_bert_label_filter_threshold_trim_and_misc_dropped():
    from models.bert_ner import BertNER
    text = "John Smith met Acme in Paris about Nobel"
    def at(word, label, score, extra=0):
        i = text.find(word)
        return {"entity_group": label, "score": score, "start": i, "end": i + len(word) + extra, "word": word}

    results = [
        at("John Smith", "PER", 0.99, extra=1),   # span includes a trailing space -> must be trimmed
        at("Acme", "ORG", 0.30),                  # below threshold -> dropped
        at("Paris", "LOC", 0.95),
        at("Nobel", "MISC", 0.99),                # MISC is not a canonical label -> dropped
    ]
    with _Stubbed(transformers=_fake_transformers(results)):
        m = BertNER()
    ents = m.predict(text, {"PER", "ORG", "LOC"}, 0.5)
    assert [(e.text, e.label) for e in ents] == [("John Smith", "PER"), ("Paris", "LOC")]
    assert m.predict(text, {"LOC"}, 0.5)[0].text == "Paris"                      # label filter
    assert m.token_spans("a bc")[1] == (2, 4)


def test_local_folder_is_preferred_when_it_exists(tmp_path=None):
    import importlib, os, tempfile
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "dslim__bert-base-NER"))
    os.environ["ANONYMIZER_MODELS_DIR"] = root
    import models.local as local
    try:
        importlib.reload(local)
        assert local.resolve("dslim/bert-base-NER") == os.path.join(root, "dslim__bert-base-NER")
        assert local.resolve("numind/NuNER_Zero") == "numind/NuNER_Zero"      # not downloaded -> repo id
    finally:
        os.environ.pop("ANONYMIZER_MODELS_DIR")
        importlib.reload(local)

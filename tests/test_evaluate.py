import tests.fake_model  # noqa: F401
from core.pipeline import get_model
from evaluation.evaluate import evaluate_model, load_docs, parse_gold, prf, score


def test_parse_gold_offsets():
    text, spans = parse_gold("[[John Smith|PER]] met [[Anna|PER]] in [[Paris|LOC]].")
    assert text == "John Smith met Anna in Paris."
    assert [text[s:e] for s, e, _ in spans] == ["John Smith", "Anna", "Paris"]
    assert [l for _, _, l in spans] == ["PER", "PER", "LOC"]


def test_partial_span_is_both_false_positive_and_false_negative():
    s = score([(0, 10, "PER")], [(0, 4, "PER")])                 # "John" vs "John Smith"
    assert s["counts"]["PER"] == {"tp": 0, "fp": 1, "fn": 1}


def test_wrong_label_is_an_error_even_with_right_span():
    s = score([(0, 4, "LOC")], [(0, 4, "PER")])
    assert s["counts"]["PER"]["fp"] == 1 and s["counts"]["LOC"]["fn"] == 1


def test_prf_values():
    m = prf(tp=3, fp=1, fn=2)
    assert abs(m["precision"] - 0.75) < 1e-9 and abs(m["recall"] - 0.6) < 1e-9
    assert prf(0, 0, 0)["f1"] == 0.0                              # no division by zero


def test_gold_files_are_consistent():
    docs = load_docs()
    assert len(docs) >= 3
    for d in docs:
        assert "[[" not in d["text"] and "|PER]]" not in d["text"]
        assert d["gold"], d["name"]
        spans = sorted(d["gold"])
        for (s1, e1, _), (s2, _, _) in zip(spans, spans[1:]):
            assert e1 <= s2, f"overlapping gold annotations in {d['name']}"
        for s, e, l in d["gold"]:
            assert l in {"PER", "ORG", "LOC"} and d["text"][s:e].strip() == d["text"][s:e]


def test_evaluate_model_perfect_case():
    text, gold = parse_gold("say hi to [[John Smith|PER]] and [[Anna Lee|PER]] today")
    r = evaluate_model(get_model("fake"), [{"name": "t", "text": text, "gold": gold}], labels={"PER"})
    assert r["micro"]["f1"] == 1.0 and r["errors"] == []

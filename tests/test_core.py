import tests.fake_model  # noqa: F401  (registers "fake")
from core.chunking import make_chunks, predict_chunked, window_settings
from core.mapper import Mapper
from core.pipeline import apply_replacements, get_model, resolve_overlaps
from core.registry import HANDLERS, MODELS, STRATEGIES
from core.types import Entity
from strategies.fake import Fake
from strategies.mask import Mask
from strategies.tag import Tag


def E(s, e, label="PER", score=0.9, text=None):
    return Entity(s, e, label, text or "x" * (e - s), score)


def test_registries_are_populated():
    import core.pipeline  # noqa: F401
    assert "bert" in MODELS and ".txt" in HANDLERS
    assert {"mask", "tag", "fake"} <= set(STRATEGIES)


# ---- offsets -------------------------------------------------------------
def test_replacement_is_end_to_start_safe():
    text = "John Smith met Anna in Paris."
    ents = [Entity(0, 10, "PER", "John Smith"), Entity(15, 19, "PER", "Anna"),
            Entity(23, 28, "LOC", "Paris")]
    assert apply_replacements(text, ents, Mask(), Mapper()) == "***** met ***** in *****."


# ---- mapper / tag --------------------------------------------------------
def test_same_name_same_tag_case_and_space_insensitive():
    text = "John Smith, JOHN  SMITH and Anna."
    ents = [Entity(0, 10, "PER", "John Smith"), Entity(12, 23, "PER", "JOHN  SMITH"),
            Entity(28, 32, "PER", "Anna")]
    out = apply_replacements(text, ents, Tag(), Mapper())
    assert out == "PER_1, PER_1 and PER_2."


def test_tag_numbers_follow_reading_order():
    text = "Anna then Bob"
    ents = [Entity(10, 13, "PER", "Bob"), Entity(0, 4, "PER", "Anna")]   # given out of order
    assert apply_replacements(text, ents, Tag(), Mapper()) == "PER_1 then PER_2"


def test_fake_is_consistent_and_seeded():
    try:
        import faker  # noqa: F401
    except ImportError:
        return  # Faker not installed here; runs on the student's machine
    e = Entity(0, 4, "PER", "Anna")
    m1, m2 = Mapper(seed=1), Mapper(seed=1)
    a, b = Fake().replace(e, m1), Fake().replace(e, m1)
    assert a == b                                   # same entity -> same fake
    assert a == Fake().replace(e, m2)               # same seed -> reproducible
    org = Fake().replace(Entity(0, 4, "ORG", "Acme"), m1)
    assert org != a                                 # company is not a person name


# ---- overlaps ------------------------------------------------------------
def test_overlap_higher_score_wins_then_longer():
    kept = resolve_overlaps([E(0, 8, score=0.8), E(4, 8, score=0.95)])   # "New York" vs "York"
    assert [(k.start, k.end) for k in kept] == [(4, 8)]
    kept = resolve_overlaps([E(0, 8, score=0.9), E(4, 8, score=0.9)])    # tie -> longer
    assert [(k.start, k.end) for k in kept] == [(0, 8)]


def test_adjacent_entities_are_not_merged():
    kept = resolve_overlaps([E(0, 4), E(4, 9)])
    assert len(kept) == 2


# ---- chunking ------------------------------------------------------------
def test_short_text_is_one_chunk():
    assert make_chunks([(0, 1), (2, 3)], 3, window=10, overlap=2) == [(0, 3)]


def test_chunks_cover_everything_and_overlap():
    spans = [(i * 2, i * 2 + 1) for i in range(100)]
    chunks = make_chunks(spans, 200, window=30, overlap=5)
    assert chunks[0][0] == 0 and chunks[-1][1] == spans[-1][1]
    for (a0, a1), (b0, b1) in zip(chunks, chunks[1:]):
        assert b0 < a1                                # neighbours overlap


def test_entity_found_wherever_it_sits_relative_to_chunk_borders():
    model = get_model("fake")
    for n in range(0, 90):                            # slide the name across many borders
        text = " ".join(["filler"] * n + ["John", "Smith"] + ["filler"] * 50)
        ents = resolve_overlaps(predict_chunked(model, text, {"PER"}, 0.5))
        assert [e.text for e in ents] == ["John Smith"], f"failed at n={n}"
        e = ents[0]
        assert text[e.start:e.end] == "John Smith"    # global offsets are correct


def test_window_settings_stay_under_limit():
    w, o = window_settings(512)
    assert w < 512 and 0 < o < w


# ---- chunk info / manual window (used by the UI) --------------------------------
def test_chunk_info_reports_tokens_and_covers_text():
    from core.chunking import chunk_info
    model = get_model("fake")                          # window 32, overlap 4
    text = " ".join(["word"] * 100)
    info = chunk_info(model, text)
    assert len(info) > 1
    assert all(c["tokens"] <= 32 for c in info)
    assert info[0]["start"] == 0 and info[-1]["end"] == len(text)
    short = chunk_info(model, "just a few words")
    assert len(short) == 1 and short[0]["tokens"] == 4


def test_manual_window_makes_more_chunks_but_never_bigger_than_safe_size():
    from core.chunking import chunk_info, window_settings
    model = get_model("fake")
    text = " ".join(["word"] * 100)
    assert len(chunk_info(model, text, window_override=12)) > len(chunk_info(model, text))
    assert window_settings(40, 9999)[0] == 32          # cannot exceed the safe window
    assert window_settings(40, 3)[0] == 12             # raised to the minimum
    w, o = window_settings(40, 12)
    assert o >= 4 and o < w


def test_names_survive_every_border_with_a_small_manual_window():
    model = get_model("fake")
    for n in range(0, 60):
        text = " ".join(["filler"] * n + ["John", "Smith"] + ["filler"] * 40)
        ents = resolve_overlaps(predict_chunked(model, text, {"PER"}, 0.5, window_override=12))
        assert [e.text for e in ents] == ["John Smith"], f"failed at n={n}"


def test_run_records_segments_and_chunks_for_display():
    import tests.fake_model  # noqa: F401
    from core.pipeline import run
    data = (" ".join(["filler"] * 100) + " Anna Lee " + " ".join(["filler"] * 100)).encode()
    res = run(data, "a.txt", "fake", strategy_name="tag", chunk_window=20)
    assert len(res.segment_info) == 1
    seg = res.segment_info[0]
    assert seg["id"] == "0" and seg["chars"] == len(data) and seg["entities"] == 1
    assert len(seg["chunks"]) > 3
    assert b"PER_1" in res.output and b"Anna" not in res.output       # still correct when forced to chunk
    plain = run(data, "a.txt", "fake", strategy_name="tag")
    assert plain.output == res.output                                  # window setting never changes the result

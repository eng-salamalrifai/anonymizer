import tests.fake_model  # noqa: F401
from core.pipeline import run
from handlers.txt_handler import TxtHandler


def roundtrip(data: bytes) -> bytes:
    h = TxtHandler()
    doc = h.read(data)
    return h.write(doc, {s.id: s.text for s in doc.segments})


def test_txt_roundtrip_is_byte_exact():
    samples = [
        b"line one\n\n\nline four\n",          # blank lines
        b"windows\r\nline endings\r\n",        # CRLF kept
        b"no final newline",                   # no trailing newline
        b"\xef\xbb\xbfbom file\n",             # UTF-8 BOM kept
        "caf\u00e9 na\u00efve\n".encode("utf-8"),
        b"caf\xe9 latin-1 \n",                 # not valid UTF-8 -> latin-1 fallback
        b"",
    ]
    for s in samples:
        assert roundtrip(s) == s, s


def test_run_anonymizes_and_preserves_layout():
    data = b"hello Anna Lee.\r\n\r\nbye Bob.\r\n"
    out = run(data, "a.txt", "fake", strategy_name="tag").output
    assert out == b"hello PER_1.\r\n\r\nbye PER_2.\r\n"


def test_no_entities_means_identical_output():
    data = b"nothing to see here.\n"
    assert run(data, "a.txt", "fake", strategy_name="mask").output == data


def test_long_txt_through_whole_pipeline():
    body = " ".join(["filler"] * 3000)
    data = (f"Anna Lee {body} Bob Ray {body} Carl").encode()
    res = run(data, "long.txt", "fake", strategy_name="tag")
    assert res.output.startswith(b"PER_1 filler") and b"PER_2" in res.output and b"PER_3" in res.output
    assert b"Anna" not in res.output and b"Bob" not in res.output


def test_unsupported_extension_is_rejected():
    try:
        run(b"x", "a.xyz", "fake")
    except ValueError as err:
        assert ".xyz" in str(err)
    else:
        raise AssertionError("should have raised")

"""Tests for the browser IDE HTML builder (no browser needed)."""

from raggym.apps.browser_ide import build_ide_html


def test_html_embeds_code_and_runtimes():
    html = build_ide_html(
        "def reverse(s):\n    raise NotImplementedError",
        "from solution import reverse\n\ndef test_it():\n    assert reverse('ab') == 'ba'",
        title="Reverse",
    )
    # runtimes loaded from CDN
    assert "pyodide" in html.lower()
    assert "monaco" in html.lower()
    # starter + tests embedded (JSON-encoded into the page)
    assert "def reverse(s):" in html
    assert "test_it" in html
    assert "Reverse" in html
    # placeholders fully substituted
    assert "%%STARTER%%" not in html
    assert "%%TESTS%%" not in html
    assert "%%TITLE%%" not in html


def test_html_escapes_quotes_safely():
    # A title with quotes must not break the JS string literal.
    html = build_ide_html("x = 1", "def test_x():\n    assert True", title='He said "hi"')
    assert '\\"hi\\"' in html  # json-escaped inside the JS literal

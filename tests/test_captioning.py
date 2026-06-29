"""Tests for optional PDF visual captioning."""

from io import BytesIO

import pytest

fitz = pytest.importorskip("fitz")
Image = pytest.importorskip("PIL.Image")

from raggym.config import Settings  # noqa: E402
from raggym.ingestion.captioning import VisualPageStats, caption_pdf_visual_pages  # noqa: E402


def _pdf_with_image(tmp_path):
    image = Image.new("RGB", (80, 50), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    path = tmp_path / "visual.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This page has a small chart-like image.")
    page.insert_image(fitz.Rect(72, 100, 180, 170), stream=buffer.getvalue())
    doc.save(path)
    doc.close()
    return path


def test_caption_pdf_visual_pages_uses_injected_captioner(tmp_path):
    path = _pdf_with_image(tmp_path)
    settings = Settings(
        _env_file=None,
        enable_captioning=True,
        vision_provider="openai",
        vision_model="gpt-4o-mini",
    )

    def fake_captioner(image_bytes: bytes, page_no: int, stats: VisualPageStats) -> str:
        assert image_bytes.startswith(b"\x89PNG")
        return f"Caption for page {page_no}; images={stats.images}."

    captions = caption_pdf_visual_pages(path, settings=settings, captioner=fake_captioner)

    assert len(captions) == 1
    assert captions[0].page == 1
    assert "Visual caption for page 1" in captions[0].text
    assert "images=1" in captions[0].text


def test_caption_pdf_visual_pages_noops_when_disabled(tmp_path):
    path = _pdf_with_image(tmp_path)
    settings = Settings(_env_file=None, enable_captioning=False)

    assert caption_pdf_visual_pages(path, settings=settings) == []

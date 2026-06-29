"""Optional visual captioning for PDF pages with figures, charts, or images."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from raggym.config import Settings
from raggym.core import get_logger
from raggym.ingestion.parsers import ParsedPage

log = get_logger(__name__)

_DRAWING_HEAVY_THRESHOLD = 20
_RENDER_DPI = 120


@dataclass(frozen=True, slots=True)
class VisualPageStats:
    """Lightweight signals used to decide whether a page needs a vision caption."""

    page: int
    text_chars: int
    images: int
    drawings: int


Captioner = Callable[[bytes, int, VisualPageStats], str]


def _needs_caption(stats: VisualPageStats) -> bool:
    return stats.images > 0 or stats.drawings >= _DRAWING_HEAVY_THRESHOLD


def _render_page_png(page) -> bytes:
    pixmap = page.get_pixmap(dpi=_RENDER_DPI, alpha=False)
    return pixmap.tobytes("png")


def _caption_with_openai(
    image_bytes: bytes,
    page_no: int,
    stats: VisualPageStats,
    settings: Settings,
) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when ENABLE_CAPTIONING=true.")

    from openai import OpenAI

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.vision_model,
        temperature=0,
        max_tokens=450,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the visual information on this PDF page for retrieval. "
                            "Focus on charts, tables, diagrams, axes, labels, legends, colors, "
                            "numeric trends, code screenshots, and any relationships shown. "
                            "If the page is mostly decorative, say so briefly. "
                            f"Page {page_no}; detected images={stats.images}, "
                            f"vector_drawings={stats.drawings}, text_chars={stats.text_chars}."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def caption_pdf_visual_pages(
    path: str | Path,
    *,
    settings: Settings,
    max_pages: int | None = None,
    captioner: Captioner | None = None,
) -> list[ParsedPage]:
    """Return extra parsed pages containing vision captions for visual-heavy pages."""

    if not settings.enable_captioning:
        return []
    if settings.vision_provider != "openai":
        raise NotImplementedError("Only OpenAI vision captioning is implemented.")

    import fitz

    path = Path(path)
    captioner = captioner or (
        lambda image, page_no, stats: _caption_with_openai(image, page_no, stats, settings)
    )
    captions: list[ParsedPage] = []

    with fitz.open(path) as doc:
        page_count = min(doc.page_count, max_pages) if max_pages else doc.page_count
        for idx in range(page_count):
            page = doc[idx]
            stats = VisualPageStats(
                page=idx + 1,
                text_chars=len(page.get_text().strip()),
                images=len(page.get_images(full=True)),
                drawings=len(page.get_drawings()),
            )
            if not _needs_caption(stats):
                continue

            log.info(
                "caption_page_start",
                file=path.name,
                page=stats.page,
                images=stats.images,
                drawings=stats.drawings,
            )
            caption = captioner(_render_page_png(page), stats.page, stats).strip()
            if not caption:
                continue

            captions.append(
                ParsedPage(
                    page=stats.page,
                    text=(
                        f"## Visual caption for page {stats.page}\n\n"
                        f"{caption}\n\n"
                        f"_Visual signals: images={stats.images}, drawings={stats.drawings}._"
                    ),
                )
            )

    log.info("caption_pdf_done", file=path.name, captions=len(captions))
    return captions

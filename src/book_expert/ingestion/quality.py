"""Quality gate для извлечённого документа: не пускать потери текста в RAG."""

from __future__ import annotations

from dataclasses import dataclass

from book_expert.ingestion.document import ExtractedDocument


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_coverage: float = 0.9
    min_chars_per_page: int = 50


@dataclass(frozen=True, slots=True)
class PageIssue:
    pdf_page_index: int
    chars: int
    reason: str


@dataclass(frozen=True, slots=True)
class DocumentQuality:
    total_pages: int
    nonempty_pages: int
    total_chars: int
    coverage: float
    status: str
    reasons: tuple[str, ...]
    issues: tuple[PageIssue, ...]

    @property
    def accepted(self) -> bool:
        return self.status != "reject"

    def to_dict(self) -> dict[str, object]:
        return {
            "total_pages": self.total_pages,
            "nonempty_pages": self.nonempty_pages,
            "total_chars": self.total_chars,
            "coverage": round(self.coverage, 4),
            "status": self.status,
            "reasons": list(self.reasons),
            "issues": [
                {
                    "pdf_page_index": issue.pdf_page_index,
                    "chars": issue.chars,
                    "reason": issue.reason,
                }
                for issue in self.issues
            ],
        }


def assess_document(
    document: ExtractedDocument,
    thresholds: QualityThresholds | None = None,
) -> DocumentQuality:
    limits = thresholds or QualityThresholds()
    pages = document.pages
    total = len(pages)
    if total == 0:
        return DocumentQuality(0, 0, 0, 0.0, "reject", ("no pages",), ())
    nonempty = sum(not page.is_empty for page in pages)
    coverage = nonempty / total
    chars = sum(page.char_count for page in pages)
    issues: list[PageIssue] = []
    for page in pages:
        if page.is_empty:
            issues.append(PageIssue(page.pdf_page_index, 0, "empty page"))
        elif page.char_count < limits.min_chars_per_page:
            issues.append(
                PageIssue(
                    page.pdf_page_index,
                    page.char_count,
                    "low text coverage",
                )
            )
    reasons: list[str] = []
    if nonempty == 0:
        reasons.append("no text layer")
    if coverage < limits.min_coverage:
        reasons.append(f"coverage {coverage:.2f} < {limits.min_coverage:.2f}")
    status = "reject" if reasons else ("warn" if issues else "ok")
    return DocumentQuality(
        total_pages=total,
        nonempty_pages=nonempty,
        total_chars=chars,
        coverage=coverage,
        status=status,
        reasons=tuple(reasons),
        issues=tuple(issues),
    )


__all__ = [
    "DocumentQuality",
    "PageIssue",
    "QualityThresholds",
    "assess_document",
]

"""Screen perception components."""

from computer_agent.perception.accessibility import (
    MacOSAccessibility,
)
from computer_agent.perception.coordinates import (
    ScreenCoordinateMapper,
)
from computer_agent.perception.engine import (
    PerceptionEngine,
    PerceptionSnapshot,
)
from computer_agent.perception.fusion import (
    UIElementFusion,
    normalize_ui_text,
    smaller_area_overlap_ratio,
)
from computer_agent.perception.models import (
    BoundingBox,
    ScreenFrame,
    UIElement,
)
from computer_agent.perception.ocr import (
    TesseractOCR,
)
from computer_agent.perception.preprocessing import (
    ImagePreprocessor,
)
from computer_agent.perception.screen_capture import (
    ScreenCapture,
)
from computer_agent.perception.semantic_extraction import (
    NewsExtractionIssue,
    NewsExtractionIssueKind,
    NewsExtractionResult,
    NewsExtractionStatus,
    NewsRecord,
    SectionExtractionResult,
    SectionExtractionStatus,
    SemanticSection,
    extract_news_records,
    extract_semantic_section,
)
from computer_agent.perception.text_locator import (
    TextTargetLocator,
)
from computer_agent.perception.viewport import (
    DiagnosisStatus,
    SemanticAXElement,
    Viewport,
    ViewportTargetDiagnosis,
    VisibilityStatus,
    classify_visibility,
    diagnose_semantic_target_visibility,
)
from computer_agent.perception.viewport_search import (
    ViewportSearchController,
    ViewportSearchObservation,
    ViewportSearchPolicy,
    ViewportSearchResult,
    ViewportSearchStatus,
)

__all__ = [
    "BoundingBox",
    "ImagePreprocessor",
    "MacOSAccessibility",
    "PerceptionEngine",
    "PerceptionSnapshot",
    "ScreenCapture",
    "ScreenCoordinateMapper",
    "ScreenFrame",
    "SectionExtractionResult",
    "SectionExtractionStatus",
    "SemanticSection",
    "TesseractOCR",
    "UIElement",
    "UIElementFusion",
    "TextTargetLocator",
    "DiagnosisStatus",
    "SemanticAXElement",
    "Viewport",
    "ViewportTargetDiagnosis",
    "VisibilityStatus",
    "classify_visibility",
    "diagnose_semantic_target_visibility",
    "ViewportSearchController",
    "ViewportSearchObservation",
    "ViewportSearchPolicy",
    "ViewportSearchResult",
    "ViewportSearchStatus",
    "NewsExtractionIssue",
    "NewsExtractionIssueKind",
    "NewsExtractionResult",
    "NewsExtractionStatus",
    "NewsRecord",
    "extract_news_records",
    "extract_semantic_section",
    "normalize_ui_text",
    "smaller_area_overlap_ratio",
]

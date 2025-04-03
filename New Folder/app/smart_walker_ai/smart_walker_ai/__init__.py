# Librerias de uso general
from typing_extensions import TypedDict
from typing import Any

class LocalizationState(TypedDict):
    original_request: str
    validate_request: bool
    rewrite_request: str
    map_rag_information: Any
    pdf_rag_validation: bool
    pdf_rag_information: str
    pdf_rag_context: str
    bad_request_val: bool
    final_coordinate_response: Any
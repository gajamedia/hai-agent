from __future__ import annotations

"""A picture of a database design (an ERD, a sketched table, a form) as the request to design a database.

The picture is checked here by what it *is* (its first bytes), not by what it is called: only PNG, JPEG and
WebP, at most the upload limit. It is sent to the model provider so the model can read it (that is the whole
point, and the person is told so before sending); it is never stored. What comes back is a design in the same
closed format as any other: it is checked by our own validation and shown, with the exact statements, before
anything is made.
"""

import hashlib
from dataclasses import dataclass

from agent.llm.contracts import LLMImage
from agent.workflow.uploaded_table import MAX_UPLOAD_BYTES, UploadRefused

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class DesignImage:
    filename: str
    sha256: str
    media_type: str
    data: bytes

    def as_llm(self) -> LLMImage:
        return LLMImage(self.media_type, self.data)


def is_image_name(filename: str) -> bool:
    return (filename or "").strip().lower().endswith(IMAGE_SUFFIXES)


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def read_image(filename: str, data: bytes) -> DesignImage:
    """The checked image, or :class:`UploadRefused` with a reason the person can act on."""
    name = (filename or "").strip()
    if not is_image_name(name):
        raise UploadRefused(f"Only {', '.join(IMAGE_SUFFIXES)} images are accepted.")
    if not data:
        raise UploadRefused("The image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRefused(f"The image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    media_type = _sniff(data)
    if media_type is None:
        raise UploadRefused("The file is not a real PNG, JPEG or WebP image.")
    return DesignImage(name, hashlib.sha256(data).hexdigest(), media_type, bytes(data))

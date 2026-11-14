from __future__ import annotations

"""Images handed to a command line model: written to a private temporary folder for the length of one call.

The file names are fixed (``image-1.png``) and the extension comes from the *verified* media type, never from
what the person named the file. The folder and everything in it is deleted when the call ends, whatever happens.
"""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from .contracts import LLMImage

EXTENSION = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


@contextmanager
def image_files(images: Sequence[LLMImage]) -> Iterator[tuple[Path | None, list[Path]]]:
    """(the temporary folder, the files in it); (None, []) when there are no images."""
    if not images:
        yield None, []
        return
    folder = Path(tempfile.mkdtemp(prefix="hai-img-"))
    try:
        paths = []
        for number, image in enumerate(images, 1):
            path = folder / f"image-{number}{EXTENSION[image.media_type]}"
            path.write_bytes(image.data)
            paths.append(path)
        yield folder, paths
    finally:
        shutil.rmtree(folder, ignore_errors=True)

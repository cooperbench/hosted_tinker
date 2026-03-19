"""Storage utilities (inlined from skyrl.utils.storage)."""

from contextlib import contextmanager
import gzip
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import Generator

from cloudpathlib import AnyPath

from hosted_tinker._log import logger


@contextmanager
def pack_and_upload(dest: AnyPath, rank: int | None = None) -> Generator[Path, None, None]:
    """Give the caller a temp directory that gets uploaded as a tar.gz archive on exit."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        yield tmp_path

        if rank is not None and rank != 0 and dest.with_name(dest.name + ".probe").exists():
            logger.info(f"Skipping write to {dest} (shared filesystem, rank {rank})")
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=0) as gz_stream:
                with tarfile.open(fileobj=gz_stream, mode="w:") as tar:
                    tar.add(tmp_path, arcname="")


@contextmanager
def download_and_unpack(source: AnyPath) -> Generator[Path, None, None]:
    """Download and extract a tar.gz archive and give the content to the caller in a temp directory."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with source.open("rb") as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                tar.extractall(tmp_path, filter="data")

        yield tmp_path


def download_file(source: AnyPath) -> io.BytesIO:
    """Download a file from storage and return it as a BytesIO object."""
    buffer = io.BytesIO()
    with source.open("rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer

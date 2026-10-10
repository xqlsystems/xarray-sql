"""A read-only Zarr store over HTTP for Pyodide, built on the browser's fetch.

zarr's own HTTP path goes through fsspec and aiohttp, which open raw
sockets that a browser does not have. This store issues ``fetch``
requests instead (with ``Range`` headers for partial reads, which sharded
Zarr v3 arrays rely on). zarr drives it from synchronous code through
Pyodide's JSPI support, so ``xr.open_zarr(FetchStore(url))`` and lazy
reads work unchanged.

The server must allow cross-origin requests (``Access-Control-Allow-Origin``)
and expose range requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from pyodide.http import pyfetch
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

# S3-style object stores answer 403 rather than 404 for a missing key
# when listing is not public.
_MISSING = (403, 404)
_ATTEMPTS = 4


def _range_header(byte_range: ByteRequest | None) -> dict[str, str]:
    if byte_range is None:
        return {}
    if isinstance(byte_range, RangeByteRequest):
        return {"Range": f"bytes={byte_range.start}-{byte_range.end - 1}"}
    if isinstance(byte_range, OffsetByteRequest):
        return {"Range": f"bytes={byte_range.offset}-"}
    if isinstance(byte_range, SuffixByteRequest):
        return {"Range": f"bytes=-{byte_range.suffix}"}
    raise TypeError(f"Unexpected byte range: {byte_range!r}")


class FetchStore(Store):
    """Read-only Zarr store for an HTTP(S) URL, using ``fetch``."""

    supports_writes = False
    supports_deletes = False
    supports_partial_writes = False
    supports_listing = False

    def __init__(self, url: str) -> None:
        super().__init__(read_only=True)
        self.url = url.rstrip("/")
        self.requests = 0
        """Number of HTTP requests issued, for observing chunk pruning."""
        self.bytes_read = 0
        """Bytes received, for observing chunk pruning."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FetchStore) and other.url == self.url

    def __repr__(self) -> str:
        return f"FetchStore({self.url!r})"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        url = f"{self.url}/{key}"
        for attempt in range(_ATTEMPTS):
            if attempt:
                await asyncio.sleep(0.25 * 2**attempt)
            self.requests += 1
            try:
                response = await pyfetch(url, headers=_range_header(byte_range))
            except OSError:  # Network failure; retry.
                if attempt == _ATTEMPTS - 1:
                    raise
                continue
            if response.status == 429 or response.status >= 500:
                continue
            break
        if response.status in _MISSING:
            return None
        if not response.ok:
            raise OSError(f"GET {url} failed: HTTP {response.status}")
        data = await response.bytes()
        self.bytes_read += len(data)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return list(
            await asyncio.gather(
                *(self.get(key, prototype, rng) for key, rng in key_ranges)
            )
        )

    async def exists(self, key: str) -> bool:
        response = await pyfetch(f"{self.url}/{key}", method="HEAD")
        return bool(response.ok)

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("FetchStore is read-only")

    async def set_partial_values(self, key_start_values) -> None:
        raise NotImplementedError("FetchStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("FetchStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("HTTP has no listing")
        yield  # pragma: no cover - makes this an async generator

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("HTTP has no listing")
        yield  # pragma: no cover

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("HTTP has no listing")
        yield  # pragma: no cover

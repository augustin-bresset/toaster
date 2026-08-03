"""Minimal LZF decompressor — the codec PCL uses for ``binary_compressed`` PCD.

LZF (Marc Lehmann's liblzf) is a byte-oriented LZ77 variant: a stream of literal
runs and back-references. Decoding is inherently sequential, so this is a plain
Python loop with a bulk-copy shortcut for the non-overlapping case. If the C
extension ``lzf`` (python-lzf) happens to be installed it is used instead, which
matters for the multi-million-point clouds PCL writes.

Only decompression is implemented: Toaster reads PCD, it never writes it.
"""

from __future__ import annotations

__all__ = ["decompress"]


def decompress(src: bytes, expected: int) -> bytes:
    """Decode the LZF stream ``src``, which must expand to exactly ``expected`` bytes.

    Args:
        src: The compressed block.
        expected: Uncompressed length, as recorded in the PCD data header.

    Raises:
        ValueError: The stream is truncated, corrupt, or does not expand to
            ``expected`` bytes.
    """
    try:
        import lzf  # type: ignore[import-not-found]
    except ImportError:
        return _decompress_py(src, expected)
    out = lzf.decompress(src, expected)
    if out is None or len(out) != expected:
        raise ValueError("corrupt LZF stream: unexpected uncompressed size")
    return out


def _decompress_py(src: bytes, expected: int) -> bytes:
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        ctrl = src[i]
        i += 1
        if ctrl < 32:
            # Literal run: the next ctrl + 1 bytes are copied verbatim.
            run = ctrl + 1
            if i + run > n:
                raise ValueError("corrupt LZF stream: truncated literal run")
            out += src[i : i + run]
            i += run
        else:
            # Back-reference: length in the top 3 bits, distance split across the
            # low 5 bits and one (or, for the escape length 7, two) trailing bytes.
            length = ctrl >> 5
            if length == 7:
                if i >= n:
                    raise ValueError("corrupt LZF stream: truncated length byte")
                length += src[i]
                i += 1
            if i >= n:
                raise ValueError("corrupt LZF stream: truncated back-reference")
            ref = len(out) - ((ctrl & 0x1F) << 8) - src[i] - 1
            i += 1
            if ref < 0:
                raise ValueError("corrupt LZF stream: back-reference before start of output")
            length += 2
            if ref + length <= len(out):
                out += out[ref : ref + length]  # disjoint: one bulk copy
            else:
                for _ in range(length):  # overlapping run: byte at a time
                    out.append(out[ref])
                    ref += 1
        if len(out) > expected:
            raise ValueError("corrupt LZF stream: expands past the declared size")
    if len(out) != expected:
        raise ValueError(
            f"corrupt LZF stream: expanded to {len(out)} bytes, header declares {expected}"
        )
    return bytes(out)

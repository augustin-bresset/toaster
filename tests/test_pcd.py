"""PCD reader coverage: the encodings and field layouts PCL writes in the wild."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from toaster.io import _lzf, load_cloud

# -- fixtures builders ----------------------------------------------------

_XYZ = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]], dtype=np.float32)


def write_pcd(path, fields, sizes, types, counts, n, data_kind, payload, *, points=True):
    """Write a PCD with a full header and a caller-supplied data block."""
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        f"SIZE {' '.join(map(str, sizes))}",
        f"TYPE {' '.join(types)}",
        f"COUNT {' '.join(map(str, counts))}",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
    ]
    if points:
        lines.append(f"POINTS {n}")
    lines.append(f"DATA {data_kind}")
    header = ("\n".join(lines) + "\n").encode()
    path.write_bytes(header + (payload.encode() if isinstance(payload, str) else payload))
    return path


def lzf_compress(raw: bytes) -> bytes:
    """A minimal, format-correct LZF encoder — test-side counterpart of the reader.

    Greedy: emits a back-reference whenever the previous 3 bytes recur within the
    8 KiB window, and literal runs otherwise. Not optimal, but it exercises both
    branches of the decoder including overlapping runs.
    """
    out = bytearray()
    literals = bytearray()

    def flush_literals():
        while literals:
            chunk = bytes(literals[:32])
            del literals[:32]
            out.append(len(chunk) - 1)
            out.extend(chunk)

    seen: dict[bytes, int] = {}
    i, n = 0, len(raw)
    while i < n:
        key = raw[i : i + 3]
        prev = seen.get(key, -1) if len(key) == 3 else -1
        distance = i - prev
        if prev >= 0 and 1 <= distance <= 8192:
            length = 3
            while length < 264 and i + length < n and raw[prev + length] == raw[i + length]:
                length += 1
            flush_literals()
            off = distance - 1
            if length <= 8:
                out.append(((length - 2) << 5) | (off >> 8))
            else:
                out.append((7 << 5) | (off >> 8))
                out.append(length - 9)
            out.append(off & 0xFF)
            for j in range(i, i + length):
                if j + 3 <= n:
                    seen[raw[j : j + 3]] = j
            i += length
        else:
            if len(key) == 3:
                seen[key] = i
            literals.append(raw[i])
            i += 1
    flush_literals()
    return bytes(out)


def binary_compressed_payload(record: np.ndarray) -> bytes:
    """Field-major (transposed) layout, LZF-compressed, behind the two-uint32 header."""
    soa = b"".join(record[name].tobytes() for name in record.dtype.names)
    comp = lzf_compress(soa)
    return struct.pack("<II", len(comp), len(soa)) + comp


# -- LZF ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"abc",
        b"a" * 500,  # long overlapping run
        b"abcabcabcabc" * 40,  # repeated pattern, back-references
        bytes(range(256)) * 8,  # incompressible-ish, mostly literal runs
    ],
)
def test_lzf_roundtrip(raw):
    assert _lzf._decompress_py(lzf_compress(raw), len(raw)) == raw


def test_lzf_rejects_wrong_length():
    with pytest.raises(ValueError):
        _lzf._decompress_py(lzf_compress(b"hello world"), 5)


def test_lzf_rejects_truncated_stream():
    with pytest.raises(ValueError):
        _lzf._decompress_py(b"\x1f" + b"short", 32)


# -- encodings ------------------------------------------------------------


def _xyzi_record():
    record = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    record["x"], record["y"], record["z"] = _XYZ.T
    record["intensity"] = [0.1, 0.2, 0.3]
    return record


def test_pcd_binary(tmp_path):
    record = _xyzi_record()
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "intensity"], [4, 4, 4, 4],
        ["F", "F", "F", "F"], [1, 1, 1, 1], 3, "binary", record.tobytes(),
    )  # fmt: skip
    cloud = load_cloud(path)
    assert np.allclose(cloud.xyz, _XYZ)
    assert np.allclose(cloud.features["intensity"], [0.1, 0.2, 0.3])


def test_pcd_binary_compressed(tmp_path):
    record = _xyzi_record()
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "intensity"], [4, 4, 4, 4],
        ["F", "F", "F", "F"], [1, 1, 1, 1], 3, "binary_compressed",
        binary_compressed_payload(record),
    )  # fmt: skip
    cloud = load_cloud(path)
    assert cloud.n == 3
    assert np.allclose(cloud.xyz, _XYZ)
    assert np.allclose(cloud.features["intensity"], [0.1, 0.2, 0.3])


def test_pcd_binary_compressed_multi_field_transpose(tmp_path):
    # Mixed field widths make a wrong de-interleave impossible to miss.
    record = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("ring", "u2")])
    record["x"] = [1, 2, 3, 4]
    record["y"] = [10, 20, 30, 40]
    record["z"] = [100, 200, 300, 400]
    record["ring"] = [7, 8, 9, 10]
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "ring"], [4, 4, 4, 2],
        ["F", "F", "F", "U"], [1, 1, 1, 1], 4, "binary_compressed",
        binary_compressed_payload(record),
    )  # fmt: skip
    cloud = load_cloud(path)
    assert np.allclose(cloud.xyz[:, 0], [1, 2, 3, 4])
    assert np.allclose(cloud.xyz[:, 2], [100, 200, 300, 400])


def test_pcd_binary_compressed_corrupt_raises(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z"], [4, 4, 4], ["F", "F", "F"],
        [1, 1, 1], 3, "binary_compressed", struct.pack("<II", 4, 36) + b"\xff\xff\xff\xff",
    )  # fmt: skip
    with pytest.raises(ValueError):
        load_cloud(path)


# -- field layouts --------------------------------------------------------


def test_pcd_padding_fields_are_skipped(tmp_path):
    # PCL's aligned PointXYZRGB writes four fields literally named "_".
    record = np.zeros(
        3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("pad", "u1", (4,)), ("rgb", "f4")]
    )
    record["x"], record["y"], record["z"] = _XYZ.T
    packed = np.array([(255 << 16) | (128 << 8) | 64] * 3, dtype=np.uint32)
    record["rgb"] = packed.view(np.float32)
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "_", "rgb"], [4, 4, 4, 1, 4],
        ["F", "F", "F", "U", "F"], [1, 1, 1, 4, 1], 3, "binary", record.tobytes(),
    )  # fmt: skip
    cloud = load_cloud(path)
    assert np.allclose(cloud.xyz, _XYZ)
    assert cloud.features["rgb"].tolist() == [[255, 128, 64]] * 3


def test_pcd_multi_count_normals(tmp_path):
    record = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("normal", "f4", (3,))])
    record["x"], record["y"], record["z"] = _XYZ.T
    record["normal"] = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "normal"], [4, 4, 4, 4],
        ["F", "F", "F", "F"], [1, 1, 1, 3], 3, "binary", record.tobytes(),
    )  # fmt: skip
    cloud = load_cloud(path)
    assert cloud.features["normals"].tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_pcd_separate_normal_axes(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "normal_x", "normal_y", "normal_z"],
        [4] * 6, ["F"] * 6, [1] * 6, 2, "ascii", "0 0 0 0 0 1\n1 1 1 0 1 0\n",
    )  # fmt: skip
    cloud = load_cloud(path)
    assert cloud.features["normals"].tolist() == [[0, 0, 1], [0, 1, 0]]


def test_pcd_ascii_rgba_is_read_as_integer_bits(tmp_path):
    # rgba is written as a plain integer in ascii; reading it as a float bit
    # pattern (as packed float rgb must be) would decode to garbage.
    rgba = (0xFF << 24) | (10 << 16) | (20 << 8) | 30
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "rgba"], [4, 4, 4, 4],
        ["F", "F", "F", "U"], [1, 1, 1, 1], 2, "ascii", f"0 0 0 {rgba}\n1 1 1 {rgba}\n",
    )  # fmt: skip
    cloud = load_cloud(path)
    assert cloud.features["rgb"].tolist() == [[10, 20, 30]] * 2


def test_pcd_label_field_becomes_labels(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "label"], [4, 4, 4, 4],
        ["F", "F", "F", "U"], [1, 1, 1, 1], 3, "ascii", "0 0 0 2\n1 1 1 0\n2 2 2 7\n",
    )  # fmt: skip
    cloud = load_cloud(path)
    assert cloud.labels is not None
    assert cloud.labels.tolist() == [2, 0, 7]
    assert cloud.labels.dtype == np.int32


def test_pcd_out_of_range_label_is_ignored(tmp_path):
    # Cluster ids from PCL segmentation can exceed int32; they are not classes.
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "label"], [4, 4, 4, 4],
        ["F", "F", "F", "U"], [1, 1, 1, 1], 2, "ascii", "0 0 0 4294967290\n1 1 1 3\n",
    )  # fmt: skip
    assert load_cloud(path).labels is None


# -- header quirks --------------------------------------------------------


def test_pcd_without_points_line(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z"], [4, 4, 4], ["F", "F", "F"],
        [1, 1, 1], 3, "ascii", "0 0 0\n1 1 1\n2 2 2\n", points=False,
    )  # fmt: skip
    assert load_cloud(path).n == 3


def test_pcd_without_count_line(tmp_path):
    body = (
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n0 0 0\n1 1 1\n"
    )
    path = tmp_path / "scan.pcd"
    path.write_text(body)
    assert load_cloud(path).n == 2


def test_pcd_crlf_header(tmp_path):
    record = _xyzi_record()
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "intensity"], [4, 4, 4, 4],
        ["F", "F", "F", "F"], [1, 1, 1, 1], 3, "binary", record.tobytes(),
    )  # fmt: skip
    raw = path.read_bytes()
    head, _, data = raw.partition(b"DATA binary\n")
    path.write_bytes(head.replace(b"\n", b"\r\n") + b"DATA binary\r\n" + data)
    assert load_cloud(path).n == 3


def test_pcd_organized_nan_points_are_dropped(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z", "intensity"], [4, 4, 4, 4],
        ["F", "F", "F", "F"], [1, 1, 1, 1], 3, "ascii",
        "0 0 0 0.1\nnan nan nan 0.2\n2 2 2 0.3\n",
    )  # fmt: skip
    cloud = load_cloud(path)
    assert cloud.n == 2
    assert np.allclose(cloud.xyz, [[0, 0, 0], [2, 2, 2]])
    assert np.allclose(cloud.features["intensity"], [0.1, 0.3])
    # The survivor mask keeps derived arrays realignable to the on-disk frame.
    assert cloud.source_count == 3
    assert cloud.source_index.tolist() == [0, 2]
    assert cloud.to_source_frame(np.array([5, 6], dtype=np.int32)).tolist() == [5, 0, 6]


def test_pcd_truncated_binary_raises(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z"], [4, 4, 4], ["F", "F", "F"],
        [1, 1, 1], 100, "binary", b"\x00" * 24,
    )  # fmt: skip
    with pytest.raises(ValueError, match="truncated"):
        load_cloud(path)


def test_pcd_without_xyz_raises(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "intensity"], [4, 4, 4],
        ["F", "F", "F"], [1, 1, 1], 2, "ascii", "0 0 0.1\n1 1 0.2\n",
    )  # fmt: skip
    with pytest.raises(ValueError, match="no z field"):
        load_cloud(path)


def test_pcd_unknown_field_type_raises(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z"], [4, 4, 3], ["F", "F", "F"],
        [1, 1, 1], 1, "ascii", "0 0 0\n",
    )  # fmt: skip
    with pytest.raises(ValueError, match="unsupported PCD field type"):
        load_cloud(path)


def test_pcd_unknown_encoding_raises(tmp_path):
    path = write_pcd(
        tmp_path / "scan.pcd", ["x", "y", "z"], [4, 4, 4], ["F", "F", "F"],
        [1, 1, 1], 1, "sparse_magic", "0 0 0\n",
    )  # fmt: skip
    with pytest.raises(ValueError, match="unknown PCD data encoding"):
        load_cloud(path)

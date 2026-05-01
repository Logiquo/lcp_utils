from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_ALT_LENS_ID = "alt_lens_id"
_ALT_LENS_NAME = "alt_lens_name"
_ALT_LENS_ID_COUNT = "alt_lens_id_count"
_ALT_LENS_NAME_COUNT = "alt_lens_name_count"
_COMMON_FIELD_NAMES = [
    _ALT_LENS_ID_COUNT,
    _ALT_LENS_NAME_COUNT,
    "author",
    "auto_scale",
    "camera_pretty_name",
    "crop_factor",
    "exif_make",
    "exif_model",
    "file_digest",
    "file_name",
    "image_length",
    "image_width",
    "is_raw",
    "lens_id",
    "lens_info",
    "lens_name",
    "lens_pretty_name",
    "metadata_distort",
    "nn_distort",
    "profile_name",
    "unique_model",
]
_COMMON_ATTRS = set(_COMMON_FIELD_NAMES) - {
    _ALT_LENS_ID_COUNT,
    _ALT_LENS_NAME_COUNT,
}


@dataclass(kw_only=True)
class IndexEntry:
    """One profile entry in an Adobe Camera Raw lens profile index."""

    path: str
    identifier: int
    author: str = ""
    auto_scale: str = ""
    camera_pretty_name: str = ""
    crop_factor: str = ""
    exif_make: str = ""
    exif_model: str = ""
    file_digest: str = ""
    file_name: str = ""
    image_length: str = ""
    image_width: str = ""
    is_raw: str = ""
    lens_id: str = ""
    lens_info: str = ""
    lens_name: str = ""
    lens_pretty_name: str = ""
    metadata_distort: str = ""
    nn_distort: str = ""
    profile_name: str = ""
    unique_model: str = ""
    alt_lens_ids: list[str] = field(default_factory=list)
    alt_lens_names: list[str] = field(default_factory=list)

    @property
    def alt_lens_id_count(self) -> int:
        return len(self.alt_lens_ids)

    @property
    def alt_lens_name_count(self) -> int:
        return len(self.alt_lens_names)


@dataclass
class Index:
    """A modifiable representation of ``Index.dat``.

    Attributes:
        version: The file format version stored in the first 32-bit word.
        entries: Profile records. Each entry contains the profile path, an
            opaque 64-bit identifier, and a string field map.
    """

    version: int
    entries: list[IndexEntry] = field(default_factory=list)


def load(value: bytes) -> Index:
    """Parse ``Index.dat`` bytes into a modifiable :class:`Index`."""

    reader = _Reader(value)
    version = reader.u32()
    entry_count = reader.u32()
    entries = []

    for _ in range(entry_count):
        path = reader.string()
        identifier = reader.u64()
        field_count = reader.u32()
        fields = {}
        for _ in range(field_count):
            key = reader.string()
            field_value = reader.string()
            fields[key] = field_value
        entries.append(_entry(path, identifier, fields))

    reader.done()
    return Index(version=version, entries=entries)


def dump(value: Index) -> bytes:
    """Serialize an :class:`Index` back to ``Index.dat`` bytes."""

    chunks = [_pack_u32(value.version), _pack_u32(len(value.entries))]
    for entry in value.entries:
        chunks.append(_pack_string(entry.path))
        chunks.append(_pack_u64(entry.identifier))
        fields = _entry_fields(entry)
        chunks.append(_pack_u32(len(fields)))
        for key, field_value in fields:
            chunks.append(_pack_string(key))
            chunks.append(_pack_string(field_value))
    return b"".join(chunks)


def file_digest(value: bytes) -> str:
    """Return the ``file_digest`` value for an LCP file.

    ``Index.dat`` stores an uppercase MD5 digest of the LCP contents after
    normalizing CRLF line endings to LF.
    """

    return hashlib.md5(value.replace(b"\r\n", b"\n")).hexdigest().upper()


def _entry(path: str, identifier: int, fields: dict[str, str]) -> IndexEntry:
    entry = IndexEntry(path=path, identifier=identifier)
    for key, value in fields.items():
        if key in {_ALT_LENS_ID_COUNT, _ALT_LENS_NAME_COUNT}:
            continue
        if key in _COMMON_ATTRS:
            setattr(entry, key, value)
        elif key.startswith(_ALT_LENS_ID):
            _set_numbered(entry.alt_lens_ids, key, _ALT_LENS_ID, value)
        elif key.startswith(_ALT_LENS_NAME):
            _set_numbered(entry.alt_lens_names, key, _ALT_LENS_NAME, value)
        else:
            raise ValueError(f"Unsupported Index.dat field: {key}")
    return entry


def _entry_fields(entry: IndexEntry) -> list[tuple[str, str]]:
    return sorted(
        (
            *_alt_lens_fields(_ALT_LENS_ID, entry.alt_lens_ids),
            *_alt_lens_fields(_ALT_LENS_NAME, entry.alt_lens_names),
            (_ALT_LENS_ID_COUNT, str(entry.alt_lens_id_count)),
            (_ALT_LENS_NAME_COUNT, str(entry.alt_lens_name_count)),
            *[(name, getattr(entry, name)) for name in _COMMON_ATTRS],
        ),
        key=lambda item: item[0],
    )


def _alt_lens_fields(prefix: str, values: list[str]) -> list[tuple[str, str]]:
    return [(f"{prefix}{index}", value) for index, value in enumerate(values)]


def _set_numbered(values: list[str], key: str, prefix: str, value: str) -> None:
    index = _numbered_index(key, prefix)
    while len(values) <= index:
        values.append("")
    values[index] = value


def _numbered(values: list[str], key: str, prefix: str) -> str | None:
    index = _numbered_index(key, prefix)
    if index >= len(values):
        return None
    return values[index]


def _numbered_index(key: str, prefix: str) -> int:
    suffix = key.removeprefix(prefix)
    if not suffix.isdecimal():
        raise ValueError(f"Expected numbered Index.dat field: {key}")
    return int(suffix)


class _Reader:
    def __init__(self, value: bytes) -> None:
        self._value = value
        self._offset = 0

    def u32(self) -> int:
        end = self._offset + _U32.size
        if end > len(self._value):
            raise ValueError("Unexpected end of Index.dat while reading uint32")
        result = _U32.unpack_from(self._value, self._offset)[0]
        self._offset = end
        return result

    def u64(self) -> int:
        end = self._offset + _U64.size
        if end > len(self._value):
            raise ValueError("Unexpected end of Index.dat while reading uint64")
        result = _U64.unpack_from(self._value, self._offset)[0]
        self._offset = end
        return result

    def string(self) -> str:
        length = self.u32()
        end = self._offset + length
        terminator = end + 1
        if terminator > len(self._value):
            raise ValueError("Unexpected end of Index.dat while reading string")
        if self._value[end] != 0:
            raise ValueError("Expected null terminator after string")

        raw = self._value[self._offset:end]
        self._offset = terminator
        return raw.decode("utf-8")

    def done(self) -> None:
        if self._offset != len(self._value):
            raise ValueError(
                f"Unexpected trailing data at byte offset {self._offset}"
            )


def _pack_u32(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"uint32 out of range: {value}")
    return _U32.pack(value)


def _pack_u64(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"uint64 out of range: {value}")
    return _U64.pack(value)


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _pack_u32(len(encoded)) + encoded + b"\0"

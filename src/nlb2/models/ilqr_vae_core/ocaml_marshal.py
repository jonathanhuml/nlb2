"""Small OCaml Marshal reader for the tutorial parameter files.

This is not a general-purpose OCaml deserializer. It implements the subset used
by ``final_params.bin``: blocks, ints, floats, strings, float arrays, and OCaml
Bigarray float64 custom blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np


SMALL_MAGIC = 0x8495A6BE
BIGARRAY_IDENT = "_bigarr02"
BIGARRAY_FLOAT64 = 1


@dataclass(frozen=True)
class Block:
    tag: int
    fields: tuple[Any, ...]


@dataclass(frozen=True)
class Bigarray:
    flags: int
    shape: tuple[int, ...]
    data: np.ndarray


class MarshalReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    def read_value(self) -> Any:
        code = self._read_u8()

        if code >= 0x80:
            tag = code & 0x0F
            size = (code >> 4) & 0x07
            return self._read_block(tag, size)
        if code >= 0x40:
            return code - 0x40
        if code >= 0x20:
            size = code - 0x20
            return self._read(size).decode("latin1")

        if code == 0x00:
            return struct.unpack("b", self._read(1))[0]
        if code == 0x01:
            return self._read_i16()
        if code == 0x02:
            return self._read_i32()
        if code == 0x03:
            return self._read_i64()
        if code == 0x08:
            header = self._read_u32()
            return self._read_block(header & 0xFF, header >> 10)
        if code == 0x13:
            header = self._read_u64()
            return self._read_block(header & 0xFF, header >> 10)
        if code == 0x09:
            return self._read(self._read_u8()).decode("latin1")
        if code == 0x0A:
            return self._read(self._read_u32()).decode("latin1")
        if code == 0x15:
            return self._read(self._read_u64()).decode("latin1")
        if code == 0x0B:
            return struct.unpack(">d", self._read(8))[0]
        if code == 0x0C:
            return struct.unpack("<d", self._read(8))[0]
        if code == 0x0D:
            return self._read_float_array(self._read_u8(), ">f8")
        if code == 0x0E:
            return self._read_float_array(self._read_u8(), "<f8")
        if code == 0x0F:
            return self._read_float_array(self._read_u32(), ">f8")
        if code == 0x07:
            return self._read_float_array(self._read_u32(), "<f8")
        if code == 0x16:
            return self._read_float_array(self._read_u64(), ">f8")
        if code == 0x17:
            return self._read_float_array(self._read_u64(), "<f8")
        if code == 0x18:
            ident = self._read_cstring()
            self._read_u32()
            self._read_u64()
            return self._read_custom(ident)
        if code == 0x19:
            return self._read_custom(self._read_cstring())

        if code in {0x04, 0x05, 0x06, 0x14}:
            raise NotImplementedError("shared Marshal references are not expected")
        raise ValueError(f"unsupported OCaml Marshal code 0x{code:02x} at byte {self._pos - 1}")

    def _read_block(self, tag: int, size: int) -> Block:
        return Block(tag=tag, fields=tuple(self.read_value() for _ in range(size)))

    def _read_custom(self, ident: str) -> Bigarray:
        if ident != BIGARRAY_IDENT:
            raise ValueError(f"unsupported OCaml custom block {ident!r}")

        n_dims = self._read_u32()
        flags = self._read_u32()
        shape = []
        for _ in range(n_dims):
            size = self._read_u16()
            if size == 0xFFFF:
                size = self._read_u64()
            shape.append(size)

        n_items = math.prod(shape)
        kind = flags & 0xFF
        if kind != BIGARRAY_FLOAT64:
            raise ValueError(f"unsupported Bigarray kind {kind}; expected float64")

        data = self._read_float_array(n_items, ">f8").reshape(tuple(shape))
        return Bigarray(flags=flags, shape=tuple(shape), data=data)

    def _read_float_array(self, n_items: int, dtype: str) -> np.ndarray:
        return np.frombuffer(self._read(8 * n_items), dtype=dtype).astype(np.float64)

    def _read_cstring(self) -> str:
        end = self._data.index(0, self._pos)
        value = self._data[self._pos:end].decode()
        self._pos = end + 1
        return value

    def _read(self, size: int) -> bytes:
        if self._pos + size > len(self._data):
            raise EOFError("unexpected end of OCaml Marshal data")
        out = self._data[self._pos : self._pos + size]
        self._pos += size
        return out

    def _read_u8(self) -> int:
        return self._read(1)[0]

    def _read_u16(self) -> int:
        return struct.unpack(">H", self._read(2))[0]

    def _read_i16(self) -> int:
        return struct.unpack(">h", self._read(2))[0]

    def _read_u32(self) -> int:
        return struct.unpack(">I", self._read(4))[0]

    def _read_i32(self) -> int:
        return struct.unpack(">i", self._read(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack(">Q", self._read(8))[0]

    def _read_i64(self) -> int:
        return struct.unpack(">q", self._read(8))[0]


def load_marshal(path: str | Path) -> Any:
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError("file is too short to be an OCaml Marshal stream")

    magic, data_len, _object_count, _size_32, _size_64 = struct.unpack(">IIIII", raw[:20])
    if magic != SMALL_MAGIC:
        raise ValueError(f"unsupported OCaml Marshal magic 0x{magic:08x}")

    payload = raw[20 : 20 + data_len]
    reader = MarshalReader(payload)
    value = reader.read_value()
    if reader.pos != data_len:
        raise ValueError(f"did not consume full Marshal payload: {reader.pos} != {data_len}")
    return value

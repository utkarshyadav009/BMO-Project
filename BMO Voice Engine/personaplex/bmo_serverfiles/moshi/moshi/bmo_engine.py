"""ctypes wrapper around libbmo.so (the optimized C++ Temporal engine).

Exposes a Pythonic ``BMOEngine`` that mirrors the C-API in ``bmo_api.h``.
"""

import ctypes
import os
import numpy as np
from typing import Tuple

_LIB_PATH = os.environ.get("BMO_SO_PATH", "./build_jetson/libbmo.so")
_lib = ctypes.CDLL(_LIB_PATH)

_lib.bmo_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.bmo_init.restype  = ctypes.c_void_p
_lib.bmo_free.argtypes = [ctypes.c_void_p]
_lib.bmo_free.restype  = None
_lib.bmo_reset.argtypes = [ctypes.c_void_p]
_lib.bmo_reset.restype  = None

for _name in ("bmo_get_n_layers", "bmo_get_n_embd", "bmo_get_n_codebooks", "bmo_get_dep_q", "bmo_get_text_vocab", "bmo_get_audio_vocab"):
    _f = getattr(_lib, _name)
    _f.argtypes = [ctypes.c_void_p]
    _f.restype  = ctypes.c_int

_lib.bmo_forward_temporal.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
_lib.bmo_forward_temporal.restype = ctypes.c_int

if hasattr(_lib, "bmo_forward_temporal2"):
    _lib.bmo_forward_temporal2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.bmo_forward_temporal2.restype = ctypes.c_int

_lib.bmo_forward_depth.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
_lib.bmo_forward_depth.restype = ctypes.c_int

_lib.bmo_last_error.argtypes = [ctypes.c_void_p]
_lib.bmo_last_error.restype  = ctypes.c_char_p

class BMOEngine:
    def __init__(self, gguf_path: str, n_ctx: int = 128):
        self._h = _lib.bmo_init(gguf_path.encode("utf-8"), n_ctx)
        if not self._h: raise RuntimeError("bmo_init returned NULL")
        self.n_layers     = _lib.bmo_get_n_layers(self._h)
        self.n_embd       = _lib.bmo_get_n_embd(self._h)
        self.n_codebooks  = _lib.bmo_get_n_codebooks(self._h)
        self.dep_q        = _lib.bmo_get_dep_q(self._h)
        self.text_vocab   = _lib.bmo_get_text_vocab(self._h)
        self.audio_vocab  = _lib.bmo_get_audio_vocab(self._h)
        self._pos         = 0

        self._buf_z       = np.empty(self.n_embd,      dtype=np.float32)
        self._buf_text    = np.empty(self.text_vocab,  dtype=np.float32)
        self._buf_audio   = np.empty(self.audio_vocab, dtype=np.float32)

    def reset(self) -> None:
        _lib.bmo_reset(self._h)
        self._pos = 0

    def forward_temporal(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert tokens.dtype == np.int32 and tokens.shape == (self.n_codebooks,)
        rc = _lib.bmo_forward_temporal(
            self._h, tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(tokens.size), int(self._pos),
            self._buf_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self._buf_text.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if rc != 0:
            err = _lib.bmo_last_error(self._h)
            raise RuntimeError(f"forward_temporal rc={rc}: {err.decode() if err else 'unknown'}")
        self._pos += 1
        return self._buf_z.copy(), self._buf_text.copy()

    def forward_depth(self, cb_index: int, prev_token: int, transformer_out: np.ndarray) -> np.ndarray:
        assert transformer_out.dtype == np.float32 and transformer_out.shape == (self.n_embd,)
        if not transformer_out.flags['C_CONTIGUOUS']: transformer_out = np.ascontiguousarray(transformer_out)
        rc = _lib.bmo_forward_depth(
            self._h, int(cb_index), int(prev_token),
            transformer_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self._buf_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if rc != 0:
            err = _lib.bmo_last_error(self._h)
            raise RuntimeError(f"forward_depth rc={rc}: {err.decode() if err else 'unknown'}")
        return self._buf_audio.copy()

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.bmo_free(self._h)
            self._h = None

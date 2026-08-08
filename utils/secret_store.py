"""Small platform-aware secret storage helpers.

Windows builds use the current user's DPAPI key so credentials and API keys are
not written as plaintext.  Non-Windows development environments retain the
legacy value for portability; values written by old versions remain readable.
"""

from __future__ import annotations

import base64
import ctypes
import os
from ctypes import wintypes


_PREFIX = "dpapi:v1:"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _dpapi_transform(value: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Windows DPAPI is unavailable")

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    input_buffer = ctypes.create_string_buffer(value)
    input_blob = _DataBlob(
        len(value), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte))
    )
    output_blob = _DataBlob()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not fn(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


def protect_secret(value: str | None) -> str | None:
    """Protect a secret before persisting it, preserving empty values."""
    if value is None or value == "" or value.startswith(_PREFIX):
        return value
    try:
        protected = _dpapi_transform(value.encode("utf-8"), protect=True)
        return _PREFIX + base64.urlsafe_b64encode(protected).decode("ascii")
    except Exception as exc:
        if os.name == "nt":
            # Never silently downgrade a Windows credential to plaintext.
            raise RuntimeError("secret protection unavailable") from exc
        # Keep non-Windows development and legacy installations portable.
        return value


def unprotect_secret(value: str | None) -> str | None:
    """Decrypt a DPAPI value; return legacy plaintext unchanged."""
    if value is None or value == "" or not value.startswith(_PREFIX):
        return value
    try:
        encoded = value[len(_PREFIX):]
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        return _dpapi_transform(raw, protect=False).decode("utf-8")
    except Exception:
        return ""


__all__ = ["protect_secret", "unprotect_secret"]

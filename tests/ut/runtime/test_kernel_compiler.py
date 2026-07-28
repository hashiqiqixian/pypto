# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for PyPTO's runtime-aware in-core compiler wrapper."""

from types import SimpleNamespace

from simpler_setup import KernelCompiler as _SimplerKernelCompiler

from pypto.runtime.kernel_compiler import KernelCompiler


def _make_compiler(platform: str = "a2a3") -> KernelCompiler:
    compiler = object.__new__(KernelCompiler)
    compiler.platform = platform
    compiler.ccec = SimpleNamespace(
        get_compile_flags=lambda core_type="aiv", **kwargs: [f"--core={core_type}"]
    )
    return compiler


def test_tprint_enables_ccec_print_flags_without_mutating_shared_toolchain(
    tmp_path, monkeypatch
):
    source = tmp_path / "kernel.cpp"
    source.write_text("void kernel() { TPRINT(tile); }", encoding="utf-8")
    compiler = _make_compiler()
    original_toolchain = compiler.ccec
    seen: list[str] = []

    def fake_compile(self, source_path, **kwargs):
        seen.extend(self.ccec.get_compile_flags(core_type=kwargs["core_type"]))
        return b"compiled"

    monkeypatch.setattr(_SimplerKernelCompiler, "compile_incore", fake_compile)

    assert compiler.compile_incore(str(source)) == b"compiled"
    assert seen == ["--core=aiv", "-D_DEBUG", "--cce-enable-print"]
    assert compiler.ccec is original_toolchain
    assert compiler.ccec.get_compile_flags() == ["--core=aiv"]


def test_non_print_kernel_keeps_normal_ccec_flags(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cpp"
    source.write_text("void kernel() { TADD(dst, lhs, rhs); }", encoding="utf-8")
    compiler = _make_compiler()
    seen: list[str] = []

    def fake_compile(self, source_path, **kwargs):
        seen.extend(self.ccec.get_compile_flags(core_type=kwargs["core_type"]))
        return b"compiled"

    monkeypatch.setattr(_SimplerKernelCompiler, "compile_incore", fake_compile)

    assert compiler.compile_incore(str(source)) == b"compiled"
    assert seen == ["--core=aiv"]

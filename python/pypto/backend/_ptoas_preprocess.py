# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared preprocessing for C++ emitted by PTOAS."""

import re

_PTOAS_UB_POINTER_ALIAS_RE = re.compile(
    r"^\s*__ubuf__\s+.+?\*\s*(?P<alias>[A-Za-z_]\w*)\s*="
    r"\s*(?P<wrapper>[A-Za-z_]\w*)\.data\(\);\s*$"
)
_PTOAS_GM_POINTER_ALIAS_RE = re.compile(
    r"^\s*__gm__\s+.+?\*\s*(?P<alias>[A-Za-z_]\w*)\s*="
    r"\s*\(__gm__\s+.+?\*\)\s*(?P<wrapper>[A-Za-z_]\w*);\s*$"
)
_PTOAS_MGATHER_CALL_RE = re.compile(
    r"(?P<prefix>\bMGATHER(?:<[^;()]+>)?\()"
    r"(?P<dst>[A-Za-z_]\w*)\s*,\s*"
    r"(?P<table>[A-Za-z_]\w*)\s*,\s*"
    r"(?P<idx>[A-Za-z_]\w*)"
    r"(?P<suffix>\);)"
)


def _restore_mgather_wrapper_operands(content: str) -> str:
    """Undo PTOAS' legacy pointer lowering for the three-operand MGATHER ABI.

    PTOAS through v0.53 lowers partition-view MGATHER operands to raw UB/GM
    pointers even though the current PTO-ISA intrinsic accepts Tile and
    GlobalTensor wrappers. Rewrite only uniquely-used pointer aliases that are
    the three direct arguments of one MGATHER call.
    """
    lines = content.splitlines(keepends=True)
    aliases: dict[str, list[tuple[str, int]]] = {}
    for line_index, line in enumerate(lines):
        for pattern in (_PTOAS_UB_POINTER_ALIAS_RE, _PTOAS_GM_POINTER_ALIAS_RE):
            if match := pattern.match(line):
                aliases.setdefault(match.group("alias"), []).append((match.group("wrapper"), line_index))
                break

    def find_unique_definition(alias: str, call_line_index: int) -> tuple[str, int] | None:
        definitions = aliases.get(alias, [])
        preceding = [
            (wrapper, line_index) for wrapper, line_index in definitions if line_index < call_line_index
        ]
        if not preceding:
            return None

        definition = preceding[-1]
        following_lines = [line_index for _, line_index in definitions if line_index > definition[1]]
        scope_end = following_lines[0] if following_lines else len(lines)
        scoped_content = "".join(lines[definition[1] : scope_end])
        if len(re.findall(rf"\b{re.escape(alias)}\b", scoped_content)) != 2:
            return None
        return definition

    declaration_lines_to_drop: set[int] = set()
    for line_index, line in enumerate(lines):
        match = _PTOAS_MGATHER_CALL_RE.search(line)
        if match is None:
            continue

        argument_names = [match.group("dst"), match.group("table"), match.group("idx")]
        definitions = [find_unique_definition(argument, line_index) for argument in argument_names]
        if any(definition is None for definition in definitions):
            continue

        wrapper_names = [definition[0] for definition in definitions if definition is not None]
        replacement = f"{match.group('prefix')}{', '.join(wrapper_names)}{match.group('suffix')}"
        lines[line_index] = f"{line[: match.start()]}{replacement}{line[match.end() :]}"
        declaration_lines_to_drop.update(
            definition[1] for definition in definitions if definition is not None
        )

    return "".join(
        line for line_index, line in enumerate(lines) if line_index not in declaration_lines_to_drop
    )


def preprocess_ptoas_output(content: str) -> str:
    """Prepare PTOAS output for embedding in PyPTO kernel wrappers."""
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include") and (
            "pto-inst" in stripped or "cstdint" in stripped or "tensor.h" in stripped
        ):
            continue
        if stripped == "using namespace pto;":
            continue
        if stripped.startswith("set_ffts_base_addr("):
            continue
        filtered.append(line)

    result = _restore_mgather_wrapper_operands("".join(filtered))
    result = re.sub(
        r'(?:extern\s*"C"\s*)?(?:__global__\s+)?AICORE\s+void',
        "static __aicore__ void",
        result,
    )
    return re.sub(r"\bAICORE\b", "__aicore__", result)

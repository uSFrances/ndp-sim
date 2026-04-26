from __future__ import annotations

from typing import Callable


SliceRouter = Callable[[int], int]


BUILTIN_SLICE_TYPE_ROUTERS: dict[str, SliceRouter] = {
    # Keep existing names for backward compatibility.
    "rope_slice_xor2": lambda slice_id: slice_id ^ 0b10,
    "slice_div4": lambda slice_id: slice_id // 4,
    # Generic reusable strategies for any operator/input/output tensor.
    "slice0": lambda _slice_id: 0,
    "slice_xor2": lambda slice_id: slice_id ^ 0b10,
}


def resolve_io_base_addr_source_slice(
    *,
    op_type: str,
    io_type: str | None,
    write_slice_id: int,
    io_role: str,
    io_name: str,
    router_by_op_and_type: dict[tuple[str, str], SliceRouter] | None = None,
    router_by_type: dict[str, SliceRouter] | None = None,
) -> int:
    """Resolve source slice for IO base_addr programming from tensor type.

    Resolution order:
    1) caller-provided (op_type, io_type) routers,
    2) caller-provided io_type routers,
    3) built-in io_type routers.
    """

    if io_type is None:
        return write_slice_id

    router: SliceRouter | None = None
    if router_by_op_and_type is not None:
        router = router_by_op_and_type.get((op_type, io_type))
    if router is None and router_by_type is not None:
        router = router_by_type.get(io_type)
    if router is None:
        router = BUILTIN_SLICE_TYPE_ROUTERS.get(io_type)
    if router is None:
        return write_slice_id

    mapped = router(write_slice_id)
    if not isinstance(mapped, int) or mapped < 0:
        raise ValueError(
            "Invalid IO base_addr source slice mapping: "
            f"op_type={op_type}, io_role={io_role}, io_name={io_name}, "
            f"io_type={io_type}, write_slice={write_slice_id}, mapped={mapped}"
        )
    return mapped

"""The agents' lookups (the plan, section 1, step 2), as plain in-process async functions.

- constraints: the campaign's wall limit (parent config/config.yaml), pedestal floor (the parent
  campaign's validated method) and the seven parameter ranges;
- abstract: one short fusion abstract from a small built-in list.

`latency_ms` (the tools knob) is a sleep before each answer. These are not MCP: on the stack they
would sit behind FlowGentic's MCP tool service, a route that needs a tool-service process and an MCP
server, so it is not built here. In the LangGraph layer they are awaited inside graph nodes, not
bound as graph tools.
"""
from __future__ import annotations

import asyncio
import functools
import zlib

import yaml

from fusion_miniapp import PARENT_DIR, decide

# Written for this mini-app as generic background text; not taken from any paper, not real abstracts.
ABSTRACTS = (
    "Stand-in abstract (synthetic). In H-mode the edge pedestal sets the boundary condition for the "
    "core: with stiff core transport, a higher pedestal temperature tends to raise the core temperature.",
    "Stand-in abstract (synthetic). Gas puffing raises edge density and cools the scrape-off layer and "
    "divertor, lowering wall temperatures, but too much puffing can erode the pedestal.",
    "Stand-in abstract (synthetic). Pedestal width and height are linked through edge stability limits; "
    "a wider pedestal can support a higher pedestal pressure before an edge instability is triggered.",
)


@functools.lru_cache(maxsize=1)
def _constraints() -> dict:
    cfg = yaml.safe_load((PARENT_DIR / "config" / "config.yaml").read_text())
    return {"wall_limit_eV": float(cfg["max_temperature_at_walls_eV"]),
            "pedestal_floor_eV": float(decide.parent().VALIDATED_METHOD["pedestal_floor_eV"]),
            "target_core_temperature_keV": float(cfg["target_core_temperature_keV"]),
            "param_ranges": {k: list(v) for k, v in decide.ranges().items()}}


async def constraints() -> dict:
    return _constraints()


async def abstract(agent_id: str) -> dict:
    i = zlib.crc32(agent_id.encode()) % len(ABSTRACTS)
    return {"id": i, "text": ABSTRACTS[i]}


async def lookup(what: str, agent_id: str, latency_ms: float = 0) -> dict:
    """One tool call by name."""
    if latency_ms:
        await asyncio.sleep(latency_ms / 1000)
    if what == "constraints":
        return await constraints()
    if what == "abstract":
        return await abstract(agent_id)
    raise KeyError(f"unknown lookup {what!r}")

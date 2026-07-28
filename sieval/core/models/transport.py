"""Transport Protocol: the provider-frontend abstraction.

A Transport lowers a :class:`Request` into its native wire format and lifts the
native response back into a :class:`Response`. Each provider frontend is one
Transport, composed into a :class:`~sieval.core.models.model.Model` (a strategy,
not a base class).

Three behavioural contracts every implementation must honour:

1. ``arun()`` returns a terminal Response.
   All provider-internal loops are resolved before returning, including any
   server-tool round-trips. The caller never drives a server-tool loop.

2. Opaque state handling depends on mode:
   - Stateful (``Request.session_id`` set): the Transport passes
     ``previous_response_id`` / ``previous_interaction_id`` internally and
     carries the new id on ``Response.session_id``. The caller never sees
     opaque state.
   - Stateless (``Request.session_id`` absent): the Transport embeds
     ``Request.reasoning.opaque_roundtrip`` into the correct wire item on
     lower(), and extracts the provider-signed value into
     ``Response.reasoning.opaque_roundtrip`` on lift(). The caller echoes it
     back next turn without modification.

3. ``Response.grounding.rendered_content`` must not be dropped in lift().
   Google ToS requires it to be rendered.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import Protocol, runtime_checkable

from .capabilities import Capability
from .ir import Request, Response


@runtime_checkable
class Transport(Protocol):
    """Provider frontend: lowers a Request to wire form and lifts the reply."""

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The IR features this Transport honours."""
        ...

    async def arun(self, req: Request) -> Response:
        """Execute *req* and return a terminal :class:`Response`."""
        ...

"""Ask Myte (mytheus-docs ADR-0047): the person's own Myte answers in character over the whole
platform, with tools that are Mytheus's own MCP tools called as the person, numbered
citations, and writes only as proposals the person confirms.

- `contracts`: the request body and the stream's event names and payloads (ADR-0047 D8).
- `mcp`: the MCP client of bff's `POST /mcp` (D3), forwarding the asker's own token.
- `sources`: the numbered source registry and tool-output mapping, citation extraction (D5).
- `tools`: the tool catalogue the model sees: bff's read tools, the pseudo-tools
  `propose_action` and `show_card` (D4) and the local help search; step labels.
- `loop`: the model loop (D2), up to 8 rounds and 45 s of tool time.
- `route`: `POST /api/rag/myte`, the Server-Sent Events surface.
"""

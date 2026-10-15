"""mosaic's tools for an MCP client, declared once and answered by a backend.

:mod:`~mosaic.agent.mcp.spec` declares each shared tool: its name, its
description for a model, its input and output models, its annotations and when
it asks for confirmation. It imports no MCP SDK. :mod:`~mosaic.agent.mcp.local`
answers the tools by calling the library in this process.

The package binds no names, so importing it loads neither module.
"""

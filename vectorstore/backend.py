# vectorstore/backend.py
"""Compatibility shim for legacy imports."""
from .chroma.backend import (  # re-export
    ChromaConfig,
    open_collection,
    add_documents,
    delete_ids,
    wipe,
    query,
)

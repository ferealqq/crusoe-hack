"""Shared Brain run event log.

The orchestrator emits typed events (`events.py`) through `RunLogger`; the frontend replays
or streams them. The event log is the only contract between the two.
"""

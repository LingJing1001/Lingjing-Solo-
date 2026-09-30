# Vendored ARC-AGI-3 Phase A runtime

This directory is a local, reproducible runtime bundle for the migrated Starter.
The framework source is `ARC-AGI-3-Agents` at commit:

```text
4743e7d0aaae0ded0d98a89a7e282e63564cd58b
```

Install the pinned Python wheels into this directory with Python 3.12:

```bash
python -m pip install --target vendor -r vendor/requirements.lock
```

The framework checkout is intentionally ignored because it is an upstream
vendored tree. The lock file is tracked so the bundle can be rebuilt. Local
runners automatically put `vendor/` and `vendor/ARC-AGI-3-Agents/` on
`sys.path`, and copy the canonical `agent/my_agent.py` into the framework
registration template before importing it.

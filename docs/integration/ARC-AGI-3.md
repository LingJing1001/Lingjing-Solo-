# ARC-AGI-3 integration layout

## Source of truth

`../../lingjing_solo/` is the only maintained Lingjing package source.
Neither integration contains a second `lingjing_solo/` directory. Both resolve
this package through `LINGJING_SRC`; local tests add the project root through
the Starter `pytest.ini`.

## Local paths

```text
Lingjing-Solo-/
├── lingjing_solo/                          # canonical source
├── integrations/
│   ├── ARC-AGI-3/                          # independent Git checkout
│   └── ARC-AGI-3-Kaggle-Starter/           # local, non-Git Starter checkout
├── scripts/
│   ├── sync_arc_bundle.py                  # canonical package + latest ARC agent
│   └── build_arc_notebook.py               # canonical notebook entrypoint
└── docs/integration/ARC-AGI-3.md
```

The integration directories are intentionally ignored by the parent Git
repository. `ARC-AGI-3` keeps its own Git history; the Starter is a local
working copy because it currently has no local Git repository.

## Commands

From this project root:

```bash
python scripts/sync_arc_bundle.py
python scripts/sync_arc_bundle.py --check
python scripts/build_arc_notebook.py
```

The builder also accepts an explicit package path:

```bash
LINGJING_SRC=/absolute/path/to/lingjing_solo \
  python scripts/build_arc_notebook.py
```

The ARC checkout remains independently runnable:

```bash
cd integrations/ARC-AGI-3
python scripts/build_notebook.py
```

Its builder delegates to the sibling Starter builder, which now resolves
`LINGJING_SRC` from the environment and otherwise falls back to its local
compatibility copy.

## Updating ARC

Update only the independent checkout:

```bash
cd integrations/ARC-AGI-3
git pull --ff-only
cd ../..
python scripts/sync_arc_bundle.py
python scripts/sync_arc_bundle.py --check
```

The sync copies the newest ARC `agent/my_agent.py` into the Starter. It does
not copy `lingjing_solo/`; runtime imports and notebook packaging resolve the
canonical package at `../../lingjing_solo/`.

## Safety boundary

The duplicate integration package directories must remain absent. Run the sync
script after canonical changes; it verifies the absence of both duplicates and
only refreshes the Starter submission agent.

# V14.2 physical-field integration

Source archive: `C:\Users\wanga\Documents\xwechat_files\wxid_t979q1oefd5q11_bcbf\msg\file\2026-09\lingjing_v14.2.zip`

The archive is integrated as the first-class Python package `lingjing_solo.v14`.
No runtime import depends on `vendor/lingjing_v14` or mutates `sys.path`.

## Included

- V14.2 contract layer: `core/writeback.py`, `core/scenario.py`
- Conservation field and numerical evolution: `core/field.py`, `core/laplacian.py`
- Causal engine loop and evacuation scenario: `engine.py`, `agents.py`, `scenarios/evacuation.py`
- Persistence/replay helpers, diagnostics, benchmark scripts, architecture/changelog
- Contract, end-to-end, interface, and historical regression tests

Generated caches and output images/trajectories from the archive remain excluded.

## Public package API

```python
from lingjing_solo.v14 import Engine, EngineConfig, Field, FieldConfig
```

SciPy-backed CSR/bubble operations use the optional dependency extra:

```bash
pip install -e '.[v14]'
```

Volume field evolution and the ARC transition adapter only require NumPy.

## Verification

```bash
python -m lingjing_solo.v14.regress
python -m pytest lingjing_solo/v14/tests -q
```

The ARC adapter imports `lingjing_solo.v14.core.field` directly. Every selected ARC
action passes through the V14.2 field transition without changing the action returned
to the ARC engine. The 25-game score therefore remains an agent-behaviour regression
contract rather than a claim that field diagnostics alone improve game performance.

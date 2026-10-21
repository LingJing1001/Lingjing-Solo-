"""A local reconstruction of the champion harness (Tufa Labs "Duck").

The point of this package is to stop writing policies. The champion's agent has
no goal detector, no key-semantics learner and no state graph: the model writes
Python against the live board each turn and decides. What the harness supplies is
the things a model cannot supply for itself --

    views.py    what the board looks like as data (.ascii, .segmentation)
    sandbox.py  one way to spend an action, plus refusals that cannot be swallowed
    prompts.py  the champion's own wording, ported verbatim where available
    agent.py    the turn loop, and model-written context compaction
    chat.py     one OpenAI-compatible call with tool support

and nothing else. `arc_adaptor/lexicon` is the opposite bet: rules written by a
human, for a machine with no model. Both exist because the two bets are not
equally affordable here -- see docs/duck-port.md.

Provenance: `vendor/champion/` holds every line recovered from the champion
notebook (`scripts/extract_champion.py` regenerates it). Files there that end in
`.added.txt` are fragments: the patch only contains added lines for modified
files, so those cannot be restored as whole files.
"""

from .agent import DuckAgent
from .chat import Chat, Endpoint
from .env import ArcEnv
from .sandbox import Sandbox

__all__ = ["DuckAgent", "Chat", "Endpoint", "ArcEnv", "Sandbox"]

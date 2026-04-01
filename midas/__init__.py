"""
Midas — Compound Engineering Framework for Alpha Feature Research
=================================================================

Dual-loop system:
  - Offline Loop : Discover new alpha features (Plan → Write → Assess → Learn)
  - Online Loop  : Monitor & improve live features (Deploy → Monitor → Diagnose → Learn)

Compound principle: every iteration produces structured learnings that make
subsequent iterations cheaper and higher-quality.
"""

__version__ = "0.1.0"
__author__  = "Midas Quant"

from .evaluator  import AlphaEvaluator, MultiAgentEvaluator
from .proposer   import ExpressionProposer
from .monitor    import OnlineMonitor
from .loops      import OfflineCompoundLoop, OfflineLoopConfig
from .promoter   import FeaturePromoter
from .kb         import KnowledgeBase
from .factory    import Midas, create_midas

__all__ = [
    "AlphaEvaluator",
    "MultiAgentEvaluator",
    "ExpressionProposer",
    "OnlineMonitor",
    "OfflineCompoundLoop",
    "OfflineLoopConfig",
    "FeaturePromoter",
    "KnowledgeBase",
    "Midas",
    "create_midas",
]

"""
Midas CLI / Factory
===================
Quick bootstrap helpers and a lightweight command-line interface.

Usage (Python):
    from midas.factory import create_midas
    midas = create_midas(kb_path="./midas-kb", api_key="sk-ant-...")
    learning = midas.offline.run(...)

Usage (shell):
    python -m midas --help
    python -m midas status
    python -m midas promote <feature_name>
    python -m midas demote  <feature_name> --reason "IC decay"
    python -m midas report  --date 2025-01-15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .evaluator  import AlphaEvaluator, MultiAgentEvaluator
from .kb         import KnowledgeBase
from .loops      import OfflineCompoundLoop, OfflineLoopConfig
from .monitor    import OnlineMonitor
from .promoter   import FeaturePromoter
from .proposer   import ExpressionProposer


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

class Midas:
    """
    Convenience container that wires all Midas components together.

    Attributes:
        kb        : KnowledgeBase  — filesystem abstraction
        evaluator : MultiAgentEvaluator
        proposer  : ExpressionProposer
        offline   : OfflineCompoundLoop
        promoter  : FeaturePromoter
        online    : OnlineMonitor (lazily created, see build_online())
    """

    def __init__(
        self,
        kb:        KnowledgeBase,
        evaluator: MultiAgentEvaluator,
        proposer:  ExpressionProposer,
        offline:   OfflineCompoundLoop,
        promoter:  FeaturePromoter,
        llm,
    ):
        self.kb        = kb
        self.evaluator = evaluator
        self.proposer  = proposer
        self.offline   = offline
        self.promoter  = promoter
        self.llm       = llm
        self._online: Optional[OnlineMonitor] = None

    def build_online(
        self,
        feature_names:    list,
        alert_thresholds: Optional[dict] = None,
        on_alert          = None,
        on_kill           = None,
    ) -> OnlineMonitor:
        """Lazily construct and cache the OnlineMonitor."""
        self._online = OnlineMonitor(
            kb               = self.kb,
            feature_names    = feature_names,
            llm_client       = self.llm,
            alert_thresholds = alert_thresholds,
            on_alert         = on_alert,
            on_kill          = on_kill,
        )
        return self._online

    @property
    def online(self) -> Optional[OnlineMonitor]:
        return self._online


def create_midas(
    kb_path:              str | Path         = "./midas-kb",
    api_key:              Optional[str]      = None,
    existing_features:    Optional[pd.DataFrame] = None,
    spread_cost_bps:      float              = 5.0,
    offline_config:       Optional[OfflineLoopConfig] = None,
) -> Midas:
    """
    Bootstrap a fully wired Midas instance.

    Args:
        kb_path            : Path to knowledge-base root (created if absent).
        api_key            : Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        existing_features  : DataFrame of deployed factor values for correlation checks.
        spread_cost_bps    : Assumed one-way spread in basis points.
        offline_config     : Custom OfflineLoopConfig (or None → defaults).

    Returns:
        Midas instance ready to use.
    """
    from anthropic import Anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY."
        )

    llm = Anthropic(api_key=key)

    kb = KnowledgeBase(Path(kb_path))

    ef = existing_features if existing_features is not None else pd.DataFrame()

    thresholds = kb.load_thresholds()
    base_eval  = AlphaEvaluator(ef, spread_cost_bps=spread_cost_bps,
                                thresholds=thresholds or None)
    evaluator  = MultiAgentEvaluator(base_eval)
    proposer   = ExpressionProposer(kb, llm)
    offline    = OfflineCompoundLoop(kb, evaluator, proposer,
                                     config=offline_config or OfflineLoopConfig())
    promoter   = FeaturePromoter(kb)

    return Midas(kb, evaluator, proposer, offline, promoter, llm)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        prog="midas",
        description="Midas — Compound Engineering Framework for Alpha Research",
    )
    parser.add_argument("--kb", default="./midas-kb",
                        help="Path to the knowledge-base root directory")

    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Print feature pipeline summary")

    # promote
    p_promote = sub.add_parser("promote", help="Promote candidate → deployed")
    p_promote.add_argument("name", help="Feature name")

    # demote
    p_demote = sub.add_parser("demote", help="Demote deployed → archived")
    p_demote.add_argument("name", help="Feature name")
    p_demote.add_argument("--reason", default="", help="Demotion reason")

    # reject
    p_reject = sub.add_parser("reject", help="Reject candidate → archived")
    p_reject.add_argument("name", help="Feature name")
    p_reject.add_argument("--reason", default="", help="Rejection reason")

    # learnings
    p_learn = sub.add_parser("learnings", help="Print recent learning documents")
    p_learn.add_argument("--n", type=int, default=5, help="Number to show")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    kb       = KnowledgeBase(Path(args.kb))
    promoter = FeaturePromoter(kb)

    if args.command == "status":
        promoter.print_pipeline()

    elif args.command == "promote":
        ok = promoter.promote(args.name)
        print(f"{'✓ Promoted' if ok else '✗ Not found in candidates'}: {args.name}")

    elif args.command == "demote":
        ok = promoter.demote(args.name, reason=args.reason)
        print(f"{'✓ Demoted' if ok else '✗ Not found in deployed'}: {args.name}")

    elif args.command == "reject":
        ok = promoter.reject_candidate(args.name, reason=args.reason)
        print(f"{'✓ Rejected' if ok else '✗ Not found in candidates'}: {args.name}")

    elif args.command == "learnings":
        docs = kb.load_recent_learnings(n=args.n)
        print(docs or "No learnings yet.")


if __name__ == "__main__":
    _cli()

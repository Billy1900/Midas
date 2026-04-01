"""
Midas CLI / Factory
===================
Quick bootstrap helpers and a lightweight command-line interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .demo       import run_demo
from .evaluator  import AlphaEvaluator, MultiAgentEvaluator
from .kb         import KnowledgeBase
from .llm        import create_llm_client
from .loops      import OfflineCompoundLoop, OfflineLoopConfig
from .monitor    import OnlineMonitor
from .promoter   import FeaturePromoter
from .proposer   import ExpressionProposer


class Midas:
    """Convenience container that wires all Midas components together."""

    def __init__(
        self,
        kb: KnowledgeBase,
        evaluator: MultiAgentEvaluator,
        proposer: ExpressionProposer,
        offline: OfflineCompoundLoop,
        promoter: FeaturePromoter,
        llm,
    ):
        self.kb = kb
        self.evaluator = evaluator
        self.proposer = proposer
        self.offline = offline
        self.promoter = promoter
        self.llm = llm
        self._online: Optional[OnlineMonitor] = None

    def build_online(
        self,
        feature_names: list[str],
        alert_thresholds: Optional[dict] = None,
        on_alert=None,
        on_kill=None,
    ) -> OnlineMonitor:
        self._online = OnlineMonitor(
            kb=self.kb,
            feature_names=feature_names,
            llm_client=self.llm,
            model=self.proposer.model,
            alert_thresholds=alert_thresholds,
            on_alert=on_alert,
            on_kill=on_kill,
        )
        return self._online

    @property
    def online(self) -> Optional[OnlineMonitor]:
        return self._online


def create_midas(
    kb_path: str | Path = "./midas-kb",
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    existing_features: Optional[pd.DataFrame] = None,
    spread_cost_bps: float = 5.0,
    offline_config: Optional[OfflineLoopConfig] = None,
    llm_client=None,
) -> Midas:
    """
    Bootstrap a fully wired Midas instance.

    Args:
        kb_path: Path to knowledge-base root.
        api_key: Provider API key.
        provider: "openai" or "anthropic". If omitted, inferred from key/env vars.
        model: Provider-specific model name.
        existing_features: DataFrame of deployed factor values for correlation checks.
        spread_cost_bps: Assumed one-way spread in basis points.
        offline_config: Custom OfflineLoopConfig.
        llm_client: Optional prebuilt client exposing .messages.create(...).
    """
    chosen_model = model
    llm = llm_client

    if llm is None:
        llm, llm_config = create_llm_client(
            api_key=api_key,
            provider=provider,
            model=chosen_model,
        )
        chosen_model = llm_config.model

    kb = KnowledgeBase(Path(kb_path))
    ef = existing_features if existing_features is not None else pd.DataFrame()
    thresholds = kb.load_thresholds()
    base_eval = AlphaEvaluator(
        ef,
        spread_cost_bps=spread_cost_bps,
        thresholds=thresholds or None,
    )
    evaluator = MultiAgentEvaluator(base_eval)

    cfg = offline_config or OfflineLoopConfig()
    if chosen_model:
        cfg.model = chosen_model

    proposer = ExpressionProposer(kb, llm, model=cfg.model, max_tokens=cfg.max_tokens)
    offline = OfflineCompoundLoop(kb, evaluator, proposer, config=cfg)
    promoter = FeaturePromoter(kb)

    return Midas(kb, evaluator, proposer, offline, promoter, llm)


def _cli():
    parser = argparse.ArgumentParser(
        prog="midas",
        description="Midas - Compound Engineering Framework for Alpha Research",
    )
    parser.add_argument("--kb", default="./midas-kb", help="Path to the knowledge-base root directory")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Print feature pipeline summary")

    p_promote = sub.add_parser("promote", help="Promote candidate to deployed")
    p_promote.add_argument("name", help="Feature name")

    p_demote = sub.add_parser("demote", help="Demote deployed to archived")
    p_demote.add_argument("name", help="Feature name")
    p_demote.add_argument("--reason", default="", help="Demotion reason")

    p_reject = sub.add_parser("reject", help="Reject candidate to archived")
    p_reject.add_argument("name", help="Feature name")
    p_reject.add_argument("--reason", default="", help="Rejection reason")

    p_learn = sub.add_parser("learnings", help="Print recent learning documents")
    p_learn.add_argument("--n", type=int, default=5, help="Number to show")

    p_demo = sub.add_parser("demo", help="Run the bundled offline + online demo")
    p_demo.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"])
    p_demo.add_argument("--api-key", default=None, help="API key for the selected provider")
    p_demo.add_argument("--model", default=None, help="Model override")
    p_demo.add_argument(
        "--report-path",
        default="demo_artifacts/DEMO_REPORT.md",
        help="Where to write the demo report",
    )
    p_demo.add_argument("--online-bars", type=int, default=800, help="Number of synthetic online bars")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    kb = KnowledgeBase(Path(args.kb))
    promoter = FeaturePromoter(kb)

    if args.command == "status":
        promoter.print_pipeline()
        return

    if args.command == "promote":
        ok = promoter.promote(args.name)
        print(f"{'Promoted' if ok else 'Not found in candidates'}: {args.name}")
        return

    if args.command == "demote":
        ok = promoter.demote(args.name, reason=args.reason)
        print(f"{'Demoted' if ok else 'Not found in deployed'}: {args.name}")
        return

    if args.command == "reject":
        ok = promoter.reject_candidate(args.name, reason=args.reason)
        print(f"{'Rejected' if ok else 'Not found in candidates'}: {args.name}")
        return

    if args.command == "learnings":
        docs = kb.load_recent_learnings(n=args.n)
        print(docs or "No learnings yet.")
        return

    if args.command == "demo":
        summary = run_demo(
            base_dir=Path(args.report_path).resolve().parent,
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            online_bars=args.online_bars,
            report_path=Path(args.report_path).resolve(),
        )
        print(f"Demo complete. Report: {summary['report_path']}")
        return


if __name__ == "__main__":
    _cli()

"""
Midas Feature Promoter
======================
Manages the feature lifecycle:

  candidates/  →  (promote)  →  deployed/  →  (demote)  →  archived/

Also provides a read-only view of what is currently live.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

from .kb import KnowledgeBase


class FeaturePromoter:
    """
    Handles state transitions for alpha features across the knowledge base.

    Usage:
        promoter = FeaturePromoter(kb)
        promoter.promote("vwap_mean_reversion_short")
        promoter.demote("stale_momentum", reason="IC consistently < 0 in last 30 days")
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    # ─────────────────────────────────────────────
    # Lifecycle transitions
    # ─────────────────────────────────────────────

    def promote(self, feature_name: str) -> bool:
        """
        Move a feature from candidates/ → deployed/.

        Updates the status and promoted_date fields inside the markdown.
        Returns True on success, False if the candidate was not found.
        """
        content = self.kb.load_feature(feature_name, "candidates")
        if content is None:
            return False

        today   = datetime.now().strftime("%Y-%m-%d")
        content = content.replace("**status:** candidate", "**status:** deployed")

        # Insert promoted date after discovered date
        content = re.sub(
            r"(\*\*discovered:\*\*\s*[\d-]+)",
            rf"\1  |  **promoted:** {today}",
            content,
        )

        self.kb.save_deployed(feature_name, content)

        # Remove from candidates
        _delete_file(self.kb.root / "knowledge" / "features" / "candidates" / f"{feature_name}.md")
        return True

    def demote(self, feature_name: str, reason: str = "") -> bool:
        """
        Move a feature from deployed/ → archived/.

        Appends a failure analysis section with the demotion reason.
        Returns True on success.
        """
        content = self.kb.load_feature(feature_name, "deployed")
        if content is None:
            return False

        today   = datetime.now().strftime("%Y-%m-%d")
        content = content.replace("**status:** deployed", "**status:** archived")

        archive_section = f"""
---

## Archive Record

**Archived:** {today}

**Reason:**

{reason or "No reason provided."}
"""
        content += archive_section

        self.kb.save_archived(feature_name, content)
        _delete_file(self.kb.root / "knowledge" / "features" / "deployed" / f"{feature_name}.md")
        return True

    def reject_candidate(self, feature_name: str, reason: str = "") -> bool:
        """
        Move a candidate directly to archived/ (backtest rejection).
        """
        content = self.kb.load_feature(feature_name, "candidates")
        if content is None:
            return False

        today   = datetime.now().strftime("%Y-%m-%d")
        content = content.replace("**status:** candidate", "**status:** archived")
        content += f"\n\n---\n\n## Rejection Record\n\n**Rejected:** {today}\n\n**Reason:**\n\n{reason}\n"

        self.kb.save_archived(feature_name, content)
        _delete_file(self.kb.root / "knowledge" / "features" / "candidates" / f"{feature_name}.md")
        return True

    # ─────────────────────────────────────────────
    # Read-only views
    # ─────────────────────────────────────────────

    def list_candidates(self) -> List[str]:
        return self.kb.list_features("candidates")

    def list_deployed(self) -> List[str]:
        return self.kb.list_features("deployed")

    def list_archived(self) -> List[str]:
        return self.kb.list_features("archived")

    def feature_status(self, feature_name: str) -> Optional[str]:
        """Return 'deployed', 'candidates', 'archived', or None."""
        for status in ("deployed", "candidates", "archived"):
            if self.kb.load_feature(feature_name, status) is not None:
                return status
        return None

    def pipeline_summary(self) -> Dict[str, List[str]]:
        """Return a dict of all features grouped by lifecycle status."""
        return {
            "candidates": self.list_candidates(),
            "deployed":   self.list_deployed(),
            "archived":   self.list_archived(),
        }

    def print_pipeline(self):
        """Pretty-print the current feature pipeline to stdout."""
        summary = self.pipeline_summary()
        print("=" * 60)
        print("  Midas Feature Pipeline")
        print("=" * 60)
        for status in ("candidates", "deployed", "archived"):
            items = summary[status]
            print(f"\n  {status.upper()} ({len(items)})")
            for name in items:
                print(f"    • {name}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _delete_file(path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

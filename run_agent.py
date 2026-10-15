"""Run the category growth agent end to end and write the memo, eval report and figures.

    python run_agent.py                      # offline (no API key needed)
    ANTHROPIC_API_KEY=... python run_agent.py --live
    python run_agent.py --question "Where should we grow in Storage?"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from growthagent import agent as A
from growthagent.evals import MAIN_Q, abstention, accuracy, red_team
from growthagent.scorer import score
from growthagent.simulate import simulate

ROOT = Path(__file__).parent
OUT, FIG = ROOT / "output", ROOT / "figures"
COLORS = {"GROW": "#1b9e77", "WATCH": "#a6a6a6", "ALREADY_PRESENT": "#7570b3", "MARGIN_TRAP": "#d95f02",
          "OVERSATURATED": "#e7298a", "INSUFFICIENT_DATA": "#e6ab02"}


def figures(d):
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for v, g in d.groupby("verdict"):
        ax.scatter(g.search_growth_yoy * 100, g.landed_cm_pct * 100, s=(g.monthly_search_volume / 600).clip(lower=45),
                   c=COLORS[v], alpha=0.75, edgecolor="white", label=v)
    for _, r in d[d.verdict.isin(["GROW", "OVERSATURATED"]) | d.category.isin(["Kitchen Islands", "Sectional Sofas"])].iterrows():
        ax.annotate(r.category, (r.search_growth_yoy * 100, r.landed_cm_pct * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    for _, r in d[d.verdict == "INSUFFICIENT_DATA"].iterrows():
        ax.annotate(f"{r.category} ({r.product_count} products: too thin)", (r.search_growth_yoy * 100, r.landed_cm_pct * 100),
                    fontsize=7, xytext=(-150, 8), textcoords="offset points", color="#b8860b")
    ax.axhline(8, ls="--", c="grey", lw=1)
    ax.text(ax.get_xlim()[0] + 1, 8.6, "margin floor 8%", fontsize=8, color="grey")
    ax.set_xlabel("Search demand growth YoY (%)")
    ax.set_ylabel("Landed contribution margin after freight + returns (%)")
    ax.set_title("Where to grow: demand growth vs margin that survives freight (bubble = search volume)")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "01_opportunity_map.png", dpi=150)
    plt.close(fig)

    rt = red_team(d)
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [c["case"] for c in rt][::-1]
    ax.barh(names, [1] * len(names), color=["#1b9e77" if c["caught"] else "#d95f02" for c in rt][::-1])
    ax.set_xticks([])
    ax.set_title(f"Red team: {sum(c['caught'] for c in rt)}/{len(rt)} injected hallucinations caught before the memo ships")
    fig.tight_layout()
    fig.savefig(FIG / "02_red_team.png", dpi=150)
    plt.close(fig)
    return rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--question", default=MAIN_Q)
    a = ap.parse_args()

    d = score(simulate())
    OUT.mkdir(exist_ok=True)
    d.drop(columns="top_review_snippet").to_csv(OUT / "scored_categories.csv", index=False)

    memo, s, rep = A.run(d, a.question, live=a.live)
    (OUT / "memo.md").write_text(A.to_markdown(memo, a.question, rep, s))
    (OUT / "memo.json").write_text(json.dumps(memo, indent=2))
    (OUT / "tool_trace.json").write_text(json.dumps(s.tool_log, indent=2, default=str))

    rt = figures(d)
    acc = accuracy(d, runner=(A.run_live if a.live else A.run_offline))
    ab = abstention(d, runner=(A.run_live if a.live else A.run_offline))
    lines = ["# Eval report", f"Mode: {'live (' + A.MODEL + ')' if a.live else 'offline'}", "",
             "## Accuracy vs planted ground truth", f"- Recommended: {', '.join(acc['recommended'])}",
             f"- Precision vs planted opportunities: {acc['precision_vs_planted']:.0%}",
             f"- Traps recommended: {acc['traps_recommended'] or 'none'}",
             f"- Prompt injection detected: {acc['injection_detected']} · obeyed: {acc['injection_obeyed']}", "",
             f"## Red team ({sum(c['caught'] for c in rt)}/{len(rt)} caught)", "| Hallucination | Caught | Guardrail feedback sent to the model |", "|---|---|---|"]
    lines += [f"| {c['case']} | {'yes' if c['caught'] else '**NO**'} | {c['feedback'][:140]} |" for c in rt]
    lines += ["", "## Abstention", "| Question | Expected | Got | Pass |", "|---|---|---|---|"]
    lines += [f"| {c['question']} | {c['expected']} | {c['got']} | {'yes' if c['pass'] else '**NO**'} |" for c in ab]
    (OUT / "EVAL_REPORT.md").write_text("\n".join(lines) + "\n")

    print(A.to_markdown(memo, a.question, rep, s))
    print(f"Red team: {sum(c['caught'] for c in rt)}/{len(rt)} caught · abstention {sum(c['pass'] for c in ab)}/{len(ab)} · precision {acc['precision_vs_planted']:.0%}")


if __name__ == "__main__":
    main()

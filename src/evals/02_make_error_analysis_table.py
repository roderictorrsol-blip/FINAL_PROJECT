from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langsmith import Client


# Name of the LangSmith project / experiment prefix to inspect
PROJECT_NAME = "ww2-rag-expanded-v1-5edacbf8"

# Output file containing the local error analysis table
OUTPUT_CSV = Path("src/evals/error_analysis.csv")

# Output file containing a plain-text diagnostic report
OUTPUT_REPORT = Path("src/evals/error_report.txt")

# Output path for the Markdown diagnostic report.
# Used to generate a human-readable summary of the RAG evaluation.
OUTPUT_REPORT_MD = Path("src/evals/error_report.md")


load_dotenv()
client = Client()


def classify_issue(correctness, groundedness):
    try:
        c = float(correctness) if correctness is not None else None
    except Exception:
        c = None

    try:
        g = float(groundedness) if groundedness is not None else None
    except Exception:
        g = None

    if c is None and g is None:
        return "manual_review"

    if c is not None and g is not None:
        if c < 0.7 and g < 0.7:
            return "retrieval_or_chunking"
        if c < 0.7 and g >= 0.7:
            return "reference_mismatch_or_generation"
        if c >= 0.7 and g < 0.7:
            return "grounding_problem"
        return "ok"

    if c is not None:
        if c < 0.7:
            return "review_correctness"
        return "ok"

    if g is not None:
        if g < 0.7:
            return "review_groundedness"
        return "ok"

    return "manual_review"


def make_note(issue: str) -> str:
    notes = {
        "retrieval_or_chunking": "Retrieved documents are likely irrelevant or incomplete.",
        "reference_mismatch_or_generation": "The answer may be correct but does not match the reference wording.",
        "grounding_problem": "The answer appears reasonable but is not sufficiently supported by retrieved context.",
        "review_correctness": "Manual inspection recommended to verify factual accuracy.",
        "review_groundedness": "Check whether the answer is actually supported by the retrieved passages.",
        "manual_review": "No evaluation scores detected; manual inspection required.",
        "ok": "No obvious issue detected with the current heuristic.",
    }
    return notes.get(issue, "")


def truncate(text: str, max_len: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_report(df: pd.DataFrame) -> str:
    total_examples = len(df)

    issue_counts = df["probable_cause"].value_counts(dropna=False).to_dict()

    correctness_series = pd.to_numeric(df["correctness"], errors="coerce")
    groundedness_series = pd.to_numeric(df["groundedness"], errors="coerce")

    avg_correctness = correctness_series.mean()
    avg_groundedness = groundedness_series.mean()

    worst_df = df.copy()
    worst_df["correctness_num"] = correctness_series
    worst_df["groundedness_num"] = groundedness_series
    worst_df["combined_score"] = (
        worst_df["correctness_num"].fillna(1.0) + worst_df["groundedness_num"].fillna(1.0)
    ) / 2

    worst_df = worst_df.sort_values(by="combined_score", ascending=True).head(5)

    lines = []
    lines.append("RAG DIAGNOSTIC REPORT")
    lines.append("---------------------")
    lines.append(f"Total examples: {total_examples}")
    lines.append("")

    lines.append("Issue distribution:")
    for issue, count in issue_counts.items():
        lines.append(f"- {issue}: {count}")
    lines.append("")

    lines.append("Average scores:")
    lines.append(
        f"- correctness: {avg_correctness:.3f}" if pd.notna(avg_correctness) else "- correctness: N/A"
    )
    lines.append(
        f"- groundedness: {avg_groundedness:.3f}" if pd.notna(avg_groundedness) else "- groundedness: N/A"
    )
    lines.append("")

    lines.append("Worst questions:")
    for i, (_, row) in enumerate(worst_df.iterrows(), start=1):
        lines.append(f"{i}. {row.get('question', '')}")
        lines.append(
            f"   correctness={row.get('correctness')} | groundedness={row.get('groundedness')}"
        )
        lines.append(f"   probable_cause={row.get('probable_cause')}")
    lines.append("")

    return "\n".join(lines)


def build_markdown_report(df: pd.DataFrame) -> str:
    total_examples = len(df)

    issue_counts = df["probable_cause"].value_counts(dropna=False).to_dict()

    correctness_series = pd.to_numeric(df["correctness"], errors="coerce")
    groundedness_series = pd.to_numeric(df["groundedness"], errors="coerce")

    avg_correctness = correctness_series.mean()
    avg_groundedness = groundedness_series.mean()

    worst_df = df.copy()
    worst_df["correctness_num"] = correctness_series
    worst_df["groundedness_num"] = groundedness_series
    worst_df["combined_score"] = (
        worst_df["correctness_num"].fillna(1.0) + worst_df["groundedness_num"].fillna(1.0)
    ) / 2

    worst_df = worst_df.sort_values(by="combined_score", ascending=True).head(5)

    lines = []
    lines.append("# RAG Diagnostic Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total evaluation examples:** {total_examples}")
    lines.append(
        f"- **Average correctness:** {avg_correctness:.3f}" if pd.notna(avg_correctness) else "- **Average correctness:** N/A"
    )
    lines.append(
        f"- **Average groundedness:** {avg_groundedness:.3f}" if pd.notna(avg_groundedness) else "- **Average groundedness:** N/A"
    )
    lines.append("")

    lines.append("## Issue Distribution")
    lines.append("")
    for issue, count in issue_counts.items():
        lines.append(f"- **{issue}**: {count}")
    lines.append("")

    lines.append("## Most Problematic Questions")
    lines.append("")
    lines.append("| # | Question | Correctness | Groundedness | Probable Cause |")
    lines.append("|---|---|---:|---:|---|")

    for i, (_, row) in enumerate(worst_df.iterrows(), start=1):
        question = str(row.get("question", "")).replace("|", "/")
        correctness = row.get("correctness", "")
        groundedness = row.get("groundedness", "")
        probable_cause = row.get("probable_cause", "")
        lines.append(
            f"| {i} | {question} | {correctness} | {groundedness} | {probable_cause} |"
        )
    lines.append("")

    return "\n".join(lines)


def latest_experiment_runs(project_name: str):
    runs = list(
        client.list_runs(
            project_name=project_name,
            execution_order=1,
            error=False,
        )
    )
    if not runs:
        raise ValueError(f"No runs found for project: {project_name}")
    return runs


def extract_feedback_map(run_id):
    feedbacks = list(client.list_feedback(run_ids=[run_id]))
    result = {}

    for fb in feedbacks:
        key = getattr(fb, "key", None)
        score = getattr(fb, "score", None)
        if key:
            result[key] = score

    return result


def main() -> None:
    runs = latest_experiment_runs(PROJECT_NAME)

    rows = []

    for run in runs:
        inputs = getattr(run, "inputs", {}) or {}
        outputs = getattr(run, "outputs", {}) or {}
        reference_outputs = getattr(run, "reference_outputs", {}) or {}

        question = inputs.get("question", "")
        answer = outputs.get("answer", "")
        reference_answer = reference_outputs.get("reference_answer", "")

        docs = outputs.get("docs", []) or []
        top_titles = []

        for doc in docs[:3]:
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            title = meta.get("video_title") or meta.get("video_id")
            if title:
                top_titles.append(title)

        feedback = extract_feedback_map(run.id)
        correctness = feedback.get("correctness")
        groundedness = feedback.get("groundedness")

        probable_cause = classify_issue(correctness, groundedness)
        notes = make_note(probable_cause)

        rows.append(
            {
                "question": question,
                "correctness": correctness,
                "groundedness": groundedness,
                "status": getattr(run, "status", None),
                "latency": getattr(run, "latency", None),
                "reference_answer": truncate(reference_answer),
                "model_answer": truncate(answer),
                "docs_count": len(docs) if isinstance(docs, list) else 0,
                "top_video_titles": " | ".join(top_titles),
                "probable_cause": probable_cause,
                "notes": notes,
            }
        )

    out_df = pd.DataFrame(rows)

    sort_cols = [col for col in ["groundedness", "correctness"] if col in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(by=sort_cols, ascending=True, na_position="last")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    report = build_report(out_df)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    report_md = build_markdown_report(out_df)
    OUTPUT_REPORT_MD.write_text(report_md, encoding="utf-8")

    print(f"OK -> analysis table created at: {OUTPUT_CSV}")
    print(f"OK -> diagnostic report created at: {OUTPUT_REPORT}")
    print(f"OK -> markdown report created at: {OUTPUT_REPORT_MD}")
    print(f"Rows processed: {len(out_df)}")


if __name__ == "__main__":
    main()
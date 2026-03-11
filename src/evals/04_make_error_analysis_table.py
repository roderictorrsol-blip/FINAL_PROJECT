from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langsmith import Client


# LangSmith projects / experiment prefixes to inspect
PROJECT_NAMES = [
    "wwii-rag-faiss-a188ed35",
    "wwii-rag-chroma-e817958a",
    "wwii-rag-hybrid-63fbaed6",
]

# Base output directory
OUTPUT_DIR = Path("src/evals")

# Global comparison outputs
SUMMARY_CSV = OUTPUT_DIR / "error_summary_by_experiment.csv"
SUMMARY_MD = OUTPUT_DIR / "error_summary_by_experiment.md"


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


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def build_report(df: pd.DataFrame, project_name: str) -> str:
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
    lines.append(f"Project: {project_name}")
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


def build_markdown_report(df: pd.DataFrame, project_name: str) -> str:
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
    lines.append(f"# RAG Diagnostic Report - {project_name}")
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


def build_rows_for_project(project_name: str):
    runs = latest_experiment_runs(project_name)
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
                "project_name": project_name,
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

    return rows


def build_summary_table(all_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    summary_rows = []

    for df in all_dfs:
        if df.empty:
            continue

        project_name = str(df["project_name"].iloc[0])

        correctness_series = pd.to_numeric(df["correctness"], errors="coerce")
        groundedness_series = pd.to_numeric(df["groundedness"], errors="coerce")

        summary_rows.append(
            {
                "project_name": project_name,
                "total_examples": len(df),
                "avg_correctness": correctness_series.mean(),
                "avg_groundedness": groundedness_series.mean(),
                "retrieval_or_chunking": int((df["probable_cause"] == "retrieval_or_chunking").sum()),
                "reference_mismatch_or_generation": int((df["probable_cause"] == "reference_mismatch_or_generation").sum()),
                "grounding_problem": int((df["probable_cause"] == "grounding_problem").sum()),
                "ok": int((df["probable_cause"] == "ok").sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["avg_groundedness", "avg_correctness"],
            ascending=False,
            na_position="last",
        )

    return summary_df


def build_summary_markdown(summary_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Experiment Comparison Summary")
    lines.append("")
    lines.append("| Project | Total | Avg Correctness | Avg Groundedness | Retrieval/Chunking | Ref Mismatch/Generation | Grounding Problem | OK |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['project_name']} | "
            f"{int(row['total_examples'])} | "
            f"{row['avg_correctness']:.3f} | "
            f"{row['avg_groundedness']:.3f} | "
            f"{int(row['retrieval_or_chunking'])} | "
            f"{int(row['reference_mismatch_or_generation'])} | "
            f"{int(row['grounding_problem'])} | "
            f"{int(row['ok'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    for project_name in PROJECT_NAMES:
        print(f"[INFO] Processing project: {project_name}")

        rows = build_rows_for_project(project_name)
        out_df = pd.DataFrame(rows)

        sort_cols = [col for col in ["groundedness", "correctness"] if col in out_df.columns]
        if sort_cols and not out_df.empty:
            out_df = out_df.sort_values(by=sort_cols, ascending=True, na_position="last")

        safe_project = safe_name(project_name)

        output_csv = OUTPUT_DIR / f"error_analysis_{safe_project}.csv"
        output_report = OUTPUT_DIR / f"error_report_{safe_project}.txt"
        output_report_md = OUTPUT_DIR / f"error_report_{safe_project}.md"

        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        report = build_report(out_df, project_name)
        output_report.write_text(report, encoding="utf-8")

        report_md = build_markdown_report(out_df, project_name)
        output_report_md.write_text(report_md, encoding="utf-8")

        print(f"[OK] CSV -> {output_csv}")
        print(f"[OK] TXT -> {output_report}")
        print(f"[OK] MD  -> {output_report_md}")
        print(f"[OK] Rows processed: {len(out_df)}")
        print("")

        all_dfs.append(out_df)

    summary_df = build_summary_table(all_dfs)

    if not summary_df.empty:
        summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        SUMMARY_MD.write_text(build_summary_markdown(summary_df), encoding="utf-8")

        print(f"[OK] Summary CSV -> {SUMMARY_CSV}")
        print(f"[OK] Summary MD  -> {SUMMARY_MD}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_events(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="DTD Dashboard", layout="wide")
    st.title("Deep Training Dynamics Dashboard")

    events_path = Path(st.sidebar.text_input("events.jsonl path", value="runs/cifar10_resnet18_monitoring/events.jsonl"))
    df = load_events(events_path)

    if df.empty:
        st.info("No events found. Run an experiment with `python main.py` first.")
        st.stop()

    st.sidebar.write("Kinds:", df["kind"].dropna().unique().tolist())
    kind = st.sidebar.selectbox("View kind", sorted(df["kind"].dropna().unique().tolist()))
    sub = df[df["kind"] == kind].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training curves")
        ts = df[df["kind"].isin(["train_step", "val_epoch"])].copy()
        if not ts.empty:
            if "loss" in ts.columns:
                st.line_chart(ts[ts["kind"] == "train_step"].set_index("global_step")["loss"], height=200)
            if "val_acc" in ts.columns:
                st.line_chart(ts[ts["kind"] == "val_epoch"].set_index("global_step")["val_acc"], height=200)

    with col2:
        st.subheader("Selected records")
        st.dataframe(sub.tail(50), use_container_width=True, height=420)

    st.subheader("Diagnostics")
    if kind == "activation_stats":
        stname = st.selectbox(
            "stat",
            ["mean", "var", "entropy", "sparsity", "dead_ratio", "drift", "instability", "sv_max", "cov_trace"],
        )
        last = sub.dropna(subset=["modules"]).tail(1)
        if not last.empty:
            mods = last.iloc[0]["modules"]
            if isinstance(mods, dict):
                s = {k: v.get(stname, None) for k, v in mods.items() if isinstance(v, dict)}
                s = pd.Series(s).dropna().sort_values(ascending=False)
                st.bar_chart(s.head(50))

    elif kind == "gradient_stats":
        latest = float(sub.tail(1)["grad_norm"].iloc[0]) if "grad_norm" in sub.columns and len(sub) else 0.0
        st.metric("Latest grad_norm", latest)
        cols = [
            c
            for c in ["grad_norm", "grad_var", "grad_cos_prev", "sharpness_proxy", "hessian_trace_proxy"]
            if c in sub.columns
        ]
        if cols:
            st.line_chart(sub.set_index("global_step")[cols])

    elif kind == "repr_epoch":
        cols = [c for c in ["cka", "svcca", "mean_cosine_drift"] if c in sub.columns]
        if cols:
            st.line_chart(sub.set_index("global_step")[cols])

    elif kind == "uncertainty_stats":
        cols = [c for c in ["predictive_entropy", "mean_confidence", "ece", "ece_temperature_scaled"] if c in sub.columns]
        if cols:
            st.line_chart(sub.set_index("global_step")[cols])

    elif kind in {"collapse_stats", "specialization_stats", "ssl_consistency", "online_warning_step"}:
        numeric_cols = [
            c
            for c in sub.columns
            if c not in {"kind", "breaches"} and pd.api.types.is_numeric_dtype(sub[c])
        ]
        if numeric_cols:
            st.line_chart(sub.set_index("global_step")[numeric_cols])

    st.caption("Next: add attention visualizations and causal graph panels once meta-model training runs produce those artifacts.")


if __name__ == "__main__":
    main()


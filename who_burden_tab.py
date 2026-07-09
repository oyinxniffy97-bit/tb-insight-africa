"""
WHO TB Burden Tab — drop-in component for TB Insight Africa
-------------------------------------------------------------
Data source: nigeria_afr_tb_burden.csv (built from WHO Global TB Report 2024)
Place the CSV in the same folder as your app, or update the path below.

Usage in your existing app.py:
    from who_burden_tab import render_who_burden_tab
    ...
    with tab_burden:   # or however you're organizing your st.tabs()
        render_who_burden_tab()
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


@st.cache_data
def load_burden_data(path: str = "nigeria_afr_tb_burden.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def render_who_burden_tab(csv_path: str = "nigeria_afr_tb_burden.csv"):
    st.subheader("Nigeria vs. Africa Region — WHO TB Burden (2000–2022)")
    st.caption("Source: WHO Global Tuberculosis Report 2024")

    df = load_burden_data(csv_path)

    metric = st.radio(
        "Metric",
        options=["Incidence (per 100k)", "Mortality (per 100k)"],
        horizontal=True,
    )

    fig = go.Figure()

    if metric == "Incidence (per 100k)":
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["incidence_100k"],
            name="Nigeria", mode="lines+markers",
            line=dict(color="#C0392B", width=3),
        ))
        # confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([df["year"], df["year"][::-1]]),
            y=pd.concat([df["incidence_hi"], df["incidence_lo"][::-1]]),
            fill="toself", fillcolor="rgba(192,57,43,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["afr_avg_incidence_100k"],
            name="AFR region average", mode="lines",
            line=dict(color="#2C3E50", width=2, dash="dash"),
        ))
        y_title = "Incidence per 100,000 population"
    else:
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["mortality_100k"],
            name="Nigeria", mode="lines+markers",
            line=dict(color="#C0392B", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([df["year"], df["year"][::-1]]),
            y=pd.concat([df["mortality_hi"], df["mortality_lo"][::-1]]),
            fill="toself", fillcolor="rgba(192,57,43,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["afr_avg_mortality_100k"],
            name="AFR region average", mode="lines",
            line=dict(color="#2C3E50", width=2, dash="dash"),
        ))
        y_title = "Mortality per 100,000 population"

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    latest = df.iloc[-1]
    col1.metric("Nigeria incidence (2022)", f"{latest['incidence_100k']:.0f} /100k")
    col2.metric("Nigeria mortality (2022)", f"{latest['mortality_100k']:.0f} /100k")
    col3.metric("Case detection rate", f"{latest['case_detection_rate']:.0f}%")

    st.caption(
        "Shaded band = WHO's estimated confidence interval for incidence/mortality. "
        "AFR average is an unweighted mean across 47 reporting countries."
    )

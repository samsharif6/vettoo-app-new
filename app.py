# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import glob, os, re
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from datetime import datetime

# ─── 1) Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vettoo Commencements", layout="wide")
st.title("Vettoo")


# ─── 2) Load latest Commencements file ──────────────────────────────────────────
DATA_DIR = "data"
files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*_Data.xlsx"))
    if not os.path.basename(f).startswith("~$")
])
if not files:
    st.error("No *_Data.xlsx file found in the folder.")
    st.stop()

latest_file = files[-1]
sheet_names = ["Commencements", "Completions", "In-training", "Recommencements", "Cancellations"]

# Sidebar dropdown to select a sheet
st.sidebar.header("Select Training Contract Status")
selected_sheet = st.sidebar.selectbox("Choose a status", sheet_names)

st.subheader(f"📈 {selected_sheet} Changes Summary")

# ✅ Move this line here:
st.caption(f"Using data from: **{os.path.basename(latest_file)}**, sheet: **{selected_sheet}**")

@st.cache_data
def load_df(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df

df = load_df(latest_file, selected_sheet)


year_cols = sorted(
    [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))],
    key=lambda x: int(x)
)
if not year_cols:
    st.error("No year columns (2015–2024) detected.")
    st.stop()

# ─── 3a) Build a cached Agent that will generate & run pandas code locally ──────
@st.cache_resource(show_spinner=False)
def get_agent(df):
    llm = ChatOpenAI(
        openai_api_key="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",
        temperature=0,
        model="gpt-3.5-turbo"
    )
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

agent = get_agent(df)

# ─── 3) Sidebar: select one or more Qualifications ─────────────────────────────
st.sidebar.header("Select Qualifications")
all_quals = sorted(df["Latest Qualification"].dropna().unique())
selected = st.sidebar.multiselect(
    "Qualifications to compare", 
    options=all_quals, 
    default=[],  # No default selection
    help="Choose one or more qualifications to view data"
)

if not selected:
    st.markdown(
        """
        ### 👋 Welcome to **Vettoo**
        ---
        Please select a **Training Contract Status** and one or more **Qualifications** from the sidebar to view detailed visualisations and trends.

        💡 Tip: You can compare multiple qualifications or explore one at a time.
        """
    )
    st.stop()


# pull out only those rows
sel_df = df[df["Latest Qualification"].isin(selected)]


# ─── 4) Combined chart: Qualifications vs Benchmarks w/ toggle ────────────────
show_bench = st.sidebar.checkbox("Show Benchmarks", value=True)

import plotly.graph_objects as go
from plotly.colors import qualitative

# prepare your long-form data
qual_long = (
    sel_df
    .melt(id_vars=["Latest Qualification"], value_vars=year_cols,
          var_name="Year", value_name="Commencements")
)
qual_long["Year"] = qual_long["Year"].astype(int)

# base values per qualification
bases = sel_df.set_index("Latest Qualification")["2015"].to_dict()

# ratio dictionaries (as before) …
benchmark_years = list(range(2015, 2025))
all_ratios = {2015:1,2016:0.929862395,2017:0.888478127,2018:0.814746354,
              2019:0.926370918,2020:1.135448757,2021:1.550934483,2022:1.676422263,
              2023:0.975354282,2024:0.931648214}
trade_ratios= {2015:1,2016:0.931882419,2017:1.004239683,2018:0.986715659,
               2019:0.959581685,2020:1.085641605,2021:1.366026003,2022:1.410684002,
               2023:1.273318259,2024:1.199286925}
voc_ratios  = {2015:1,2016:0.927879961,2017:0.822200710,2018:0.716844143,
               2019:0.907712165,2020:1.164246531,2021:1.656824782,2022:1.828654405,
               2023:0.805582446,2024:0.779172825}

# bench DataFrames
bench_all = pd.DataFrame({"Year": benchmark_years,
                          "Benchmark": [all_ratios[y] for y in benchmark_years]})
bench_trade = pd.DataFrame({"Year": benchmark_years,
                            "Benchmark": [trade_ratios[y] for y in benchmark_years]})
bench_voc   = pd.DataFrame({"Year": benchmark_years,
                            "Benchmark": [voc_ratios[y] for y in benchmark_years]})

fig = go.Figure()
palette = qualitative.Plotly

# plot each qualification
for i, qual in enumerate(selected):
    color = palette[i % len(palette)]
    dfq = qual_long[qual_long["Latest Qualification"] == qual]
    base = bases.get(qual, 1)

    # qualification line
    fig.add_trace(go.Scatter(
        x=dfq["Year"], y=dfq["Commencements"],
        mode="lines+markers", name=qual,
        line=dict(color=color, width=3),
        marker=dict(color=color)
    ))

    if show_bench and selected_sheet != "Recommencements":
        # market
        fig.add_trace(go.Scatter(
            x=bench_all["Year"],
            y=bench_all["Benchmark"] * base,
            mode="lines", name=f"{qual} Market Bench",
            line=dict(color=color, dash="dashdot"),
            showlegend=True
        ))
        # trade
        fig.add_trace(go.Scatter(
            x=bench_trade["Year"],
            y=bench_trade["Benchmark"] * base,
            mode="lines", name=f"{qual} Trade Bench",
            line=dict(color=color, dash="dash"),
            showlegend=True
        ))
        # vocation
        fig.add_trace(go.Scatter(
            x=bench_voc["Year"],
            y=bench_voc["Benchmark"] * base,
            mode="lines", name=f"{qual} Voc Bench",
            line=dict(color=color, dash="dot"),
            showlegend=True
        ))

# shade COVID period if benchmarks are shown
if show_bench:
    fig.add_vrect(
        x0=2020, x1=2022,
        fillcolor="lightgrey", opacity=0.3, layer="below", line_width=0,
        annotation_text="COVID-19 pandemic", annotation_position="top left",
        annotation_font_size=12, annotation_font_color="grey"
    )

fig.update_layout(
    title=f"Annual {selected_sheet} vs Benchmarks" if show_bench else f"Annual {selected_sheet}",
    xaxis_title="Year",
    yaxis_title=selected_sheet,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, l=20, r=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)


# ─── 4b) Key metrics per qualification ─────────────────────────────────────────
# instead of showing only the first selected, build a small table:
metrics = sel_df[[
    "Latest Qualification",
    "% Change_from 2023 to 2024",
    "% Change_from 2019 to 2024"
]].copy()

# convert to percentages
metrics["YoY % Change (2023→24)"] = metrics["% Change_from 2023 to 2024"] * 100
metrics["5-Yr % Change (2019→24)"] = metrics["% Change_from 2019 to 2024"] * 100

# keep only the nicely-named columns
metrics = metrics[[
    "Latest Qualification",
    "YoY % Change (2023→24)",
    "5-Yr % Change (2019→24)"
]]

# display as a table with one-decimal formatting
st.table(
    metrics.style.format({
        "YoY % Change (2023→24)": "{:.1f}%",
        "5-Yr % Change (2019→24)": "{:.1f}%"
    })
)

# ─── 4c) Download all selected records in one sheet with source column ───────
def to_excel_all_with_sheetname(df_slice: pd.DataFrame, sheet_name: str) -> BytesIO:
    df_copy = df_slice.copy()
    df_copy.insert(0, "Data Source", sheet_name)  # Add new first column with sheet name
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_copy.to_excel(writer, sheet_name="All Qualifications", index=False)
    buf.seek(0)
    return buf


timestamp = datetime.now().strftime("%Y%m%d_%H%M")

st.download_button(
    "📥 Download selected records (one sheet)",
    to_excel_all_with_sheetname(sel_df, selected_sheet),
    f"vettoo_{selected_sheet.lower()}_{timestamp}.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)



# ─── 5) AI-Powered Chatbot via DataFrame Agent ─────────────────────────────────
st.markdown("---")
st.header("Ask Vettoo (AI-Powered Data Query)")

question = st.text_input("Type any question about your data…")
if question:
    with st.spinner("🤖 Vettoo is thinking…"):
        answer = agent.run(question)
    st.markdown(f"**Vettoo:** {answer}")
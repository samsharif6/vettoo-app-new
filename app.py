# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import glob, os, re
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# ─── 1) Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vettoo Commencements", layout="wide")
st.title("Vettoo")
st.subheader("VET Commencement Insights")

# ─── 2) Load latest Commencements file ──────────────────────────────────────────
DATA_DIR = r"C:\1 SASC\Startup Platform\Data"
files = sorted(glob.glob(os.path.join(DATA_DIR, "*Commencements*.xls*")))
if not files:
    st.error("No Commencements file found.")
    st.stop()

@st.cache_data
def load_df(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df

df = load_df(files[-1])
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
    default=[all_quals[0]]
)
if not selected:
    st.error("Please choose at least one qualification.")
    st.stop()

# pull out only those rows
sel_df = df[df["Latest Qualification"].isin(selected)]

# ─── 4) Combined chart: Qualifications vs Market, Trade & Vocation Benchmarks ──
st.header("📊 Annual Commencements vs Benchmarks")

# melt your selected qualifications into long form
qual_long = (
    sel_df
    .melt(id_vars=["Latest Qualification"], value_vars=year_cols,
          var_name="Year", value_name="Commencements")
)
qual_long["Year"] = qual_long["Year"].astype(int)

# grab the 2015 base for each qual
bases = sel_df.set_index("Latest Qualification")["2015"].to_dict()

# define your three ratio series (unchanged)
all_ratios = {2015:1, 2016:0.929862395, 2017:0.888478127, 2018:0.814746354,
              2019:0.926370918, 2020:1.135448757, 2021:1.550934483, 2022:1.676422263,
              2023:0.975354282, 2024:0.931648214}
trade_ratios = {2015:1, 2016:0.931882419, 2017:1.004239683, 2018:0.986715659,
                2019:0.959581685, 2020:1.085641605, 2021:1.366026003, 2022:1.410684002,
                2023:1.273318259, 2024:1.199286925}
voc_ratios = {2015:1, 2016:0.927879961, 2017:0.822200710, 2018:0.716844143,
              2019:0.907712165, 2020:1.164246531, 2021:1.656824782, 2022:1.828654405,
              2023:0.805582446, 2024:0.779172825}

# build benchmark DataFrames once
bench_all = pd.DataFrame({
    "Year": list(all_ratios),
    "Market Benchmark": [all_ratios[y] for y in all_ratios]
})
bench_trade = pd.DataFrame({
    "Year": list(trade_ratios),
    "Trade Benchmark": [trade_ratios[y] for y in trade_ratios]
})
bench_voc   = pd.DataFrame({
    "Year": list(voc_ratios),
    "Vocation Benchmark": [voc_ratios[y] for y in voc_ratios]
})

fig = go.Figure()

# qualification lines
for qual in selected:
    sub = qual_long[qual_long["Latest Qualification"]==qual]
    fig.add_trace(go.Scatter(
        x=sub["Year"], y=sub["Commencements"],
        mode="lines+markers", name=qual,
        line=dict(width=3)
    ))

# benchmark lines (scale each by the FIRST selected qual's base2015)
base = bases[selected[0]]
fig.add_trace(go.Scatter(
    x=bench_all["Year"],
    y=bench_all["Market Benchmark"]*base,
    mode="lines+markers", name="Market Benchmark",
    line=dict(color="gray", dash="dashdot"),
    marker=dict(symbol="circle-open")
))
fig.add_trace(go.Scatter(
    x=bench_trade["Year"],
    y=bench_trade["Trade Benchmark"]*base,
    mode="lines+markers", name="Trade Benchmark",
    line=dict(color="green", dash="dash"),
    marker=dict(symbol="square-open")
))
fig.add_trace(go.Scatter(
    x=bench_voc["Year"],
    y=bench_voc["Vocation Benchmark"]*base,
    mode="lines+markers", name="Vocation Benchmark",
    line=dict(color="orange", dash="dot"),
    marker=dict(symbol="diamond-open")
))

# ─── COVID-19 shading ─────────────────────────────────────────────────────────
fig.add_vrect(
    x0=2020, x1=2022,
    fillcolor="lightgrey", opacity=0.3, layer="below", line_width=0,
    annotation_text="COVID-19 pandemic", annotation_position="top left",
    annotation_font_size=12, annotation_font_color="grey"
)

# now finalize layout
fig.update_layout(
    title="Annual Commencements vs Market, Trade & Vocation Benchmarks",
    xaxis_title="Year", yaxis_title="Commencements",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, l=20, r=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# if more than one qual is selected, remind folks what the benchmarks are scaled to
if len(selected) > 1:
    st.info(
        f"🔎 Benchmarks (Market, Trade & Vocation) are all normalized to the 2015 commencements of "
        f"**{selected[0]}**, the first qualification you selected."
    )

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

# ─── 4c) Download all selected records in one sheet ─────────────────────────
def to_excel_all(df_slice: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # write entire filtered DataFrame to a single sheet
        df_slice.to_excel(writer, sheet_name="All Qualifications", index=False)
    buf.seek(0)
    return buf
st.download_button(
    "📥 Download selected records (one sheet)",
    to_excel_all(sel_df),
    "selected_qualifications_commencements.xlsx",
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

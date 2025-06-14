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
# st.caption(f"Using data from: **{os.path.basename(latest_file)}**, sheet: **{selected_sheet}**")

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
        openai_api_key=os.getenv("OPENAI_API_KEY"),
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

from plotly.colors import qualitative

# 4a) melt into long‐form
qual_long = (
    sel_df
    .melt(
        id_vars=["Latest Qualification"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )
)
qual_long["Year"] = qual_long["Year"].astype(int)

# 4b) base‐2015 values per qualification
bases = sel_df.set_index("Latest Qualification")["2015"].to_dict()

# 4c) define all ratio dictionaries
ratios = {
    "Commencements": {
        "All":      [1.000000,0.929862,0.888478,0.814746,0.926371,1.135449,1.550934,1.676422,0.975354,0.931648],
        "Trade":    [1.000000,0.931882,1.004240,0.986716,0.959582,1.085642,1.366026,1.410684,1.273318,1.199287],
        "Vocation": [1.000000,0.927880,0.822201,0.716844,0.907712,1.164247,1.656825,1.828654,0.805582,0.779173],
    },
    "Completions": {
        "All":      [1.000,0.803,0.741,0.616,0.606,0.550,0.763,0.809,0.914,0.761],
        "Trade":    [1.000,0.960,0.895,0.809,0.789,0.757,1.018,0.978,0.882,1.067],
        "Vocation": [1.000,0.742,0.686,0.540,0.538,0.466,0.673,0.764,0.979,0.637],
    },
    "In-training": {
        "All":      [1.000,0.912,0.890,0.900,0.972,1.191,1.436,1.621,1.398,1.298],
        "Trade":    [1.000,0.958,1.004,1.048,1.087,1.209,1.310,1.424,1.526,1.559],
        "Vocation": [1.000,0.873,0.772,0.742,0.870,1.223,1.657,1.948,1.288,1.021],
    },
    "Cancellations": {
        "All":      [1.000,0.850,0.798,0.719,0.743,0.736,1.133,1.464,1.249,0.970],
        "Trade":    [1.000,0.936,0.969,1.107,1.099,0.962,1.427,1.525,1.500,1.477],
        "Vocation": [1.000,0.822,0.745,0.533,0.575,0.636,1.012,1.483,1.153,0.708],
    },
}

bench_years = list(range(2015, 2025))
sheet_ratios = ratios[selected_sheet]

# build bench DataFrames once
bench_all   = pd.DataFrame(dict(Year=bench_years, Ratio=sheet_ratios["All"]))
bench_trade = pd.DataFrame(dict(Year=bench_years, Ratio=sheet_ratios["Trade"]))
bench_voc   = pd.DataFrame(dict(Year=bench_years, Ratio=sheet_ratios["Vocation"]))

# 4d) build figure
fig = go.Figure()
palette = qualitative.Plotly

for idx, qual in enumerate(selected):
    color = palette[idx % len(palette)]
    dfq   = qual_long[qual_long["Latest Qualification"] == qual]
    base  = bases.get(qual, 1)

    # qualification line
    fig.add_trace(go.Scatter(
        x=dfq["Year"],
        y=dfq["Value"],
        mode="lines+markers",
        name=qual,
        line=dict(color=color, width=3),
        marker=dict(color=color)
    ))

    # benchmarks for each qual
    if show_bench:
        fig.add_trace(go.Scatter(
            x=bench_all["Year"],
            y=bench_all["Ratio"] * base,
            mode="lines",
            name=f"{qual} Market Bench",
            line=dict(color=color, dash="dashdot"),
        ))
        fig.add_trace(go.Scatter(
            x=bench_trade["Year"],
            y=bench_trade["Ratio"] * base,
            mode="lines",
            name=f"{qual} Trade Bench",
            line=dict(color=color, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=bench_voc["Year"],
            y=bench_voc["Ratio"] * base,
            mode="lines",
            name=f"{qual} Voc Bench",
            line=dict(color=color, dash="dot"),
        ))

# 4e) shade COVID
if show_bench:
    fig.add_vrect(
        x0=2020, x1=2022,
        fillcolor="lightgrey", opacity=0.3,
        layer="below", line_width=0,
        annotation_text="COVID-19 pandemic",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="grey"
    )

# 4f) layout with vertical legend on right
fig.update_layout(
    title=f"Annual {selected_sheet} vs Benchmarks" if show_bench else f"Annual {selected_sheet}",
    xaxis_title="Year",
    yaxis_title=selected_sheet,
    legend=dict(
        orientation="v",
        x=1.02, xanchor="left",
        y=0.5,  yanchor="middle"
    ),
    margin=dict(t=60, l=20, r=200, b=20),
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
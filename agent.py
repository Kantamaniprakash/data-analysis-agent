"""
Data Analysis AI Agent
=======================
An autonomous LLM agent that analyzes datasets using natural language.
Built with LangChain AgentExecutor (ReAct), OpenAI GPT-4o, and Streamlit.

Author: Satya Sai Prakash Kantamani
GitHub: https://github.com/kantamaniprakash
"""

import os
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analysis AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 0.85rem 1.2rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 75%;
        color: white;
    }
    .agent-msg {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 12px 12px 12px 4px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        max-width: 85%;
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = []

# ─── Tool Helpers ──────────────────────────────────────────────────────────────

def _get_df():
    return st.session_state.df

# ─── Agent Tools ──────────────────────────────────────────────────────────────

@tool
def dataset_info_tool(query: str) -> str:
    """Get basic information about the loaded dataset including shape, columns, dtypes, and a preview."""
    df = _get_df()
    if df is None:
        return "No dataset loaded. Please upload a CSV or Excel file."
    return (
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n"
        f"Columns: {list(df.columns)}\n"
        f"Numeric columns: {list(df.select_dtypes(include=np.number).columns)}\n"
        f"Categorical columns: {list(df.select_dtypes(include='object').columns)}\n"
        f"First 3 rows:\n{df.head(3).to_string()}"
    )


@tool
def pandas_tool(code: str) -> str:
    """
    Execute pandas Python code on the DataFrame (variable name: df).
    Use for filtering, grouping, aggregation, calculations.
    Input must be a valid Python expression or statement using df.
    Example: df.groupby('Category')['Sales'].sum().sort_values(ascending=False).head(10)
    """
    df = _get_df()
    if df is None:
        return "Error: No dataset loaded."
    try:
        ns = {"df": df, "pd": pd, "np": np}
        result = eval(code, {"__builtins__": {}}, ns)
        return str(result)
    except Exception:
        try:
            ns = {"df": df, "pd": pd, "np": np}
            exec(code, {"__builtins__": {"print": print, "len": len, "range": range}}, ns)
            return "Code executed successfully."
        except Exception:
            return f"Execution error:\n{traceback.format_exc()}"


@tool
def stats_tool(query: str) -> str:
    """
    Run statistical analysis on the DataFrame.
    Accepted queries: 'describe', 'correlation', 'missing', 'dtypes',
    'outliers:COLUMN_NAME', 'unique:COLUMN_NAME'
    Example: 'outliers:Sales' or 'correlation'
    """
    df = _get_df()
    if df is None:
        return "Error: No dataset loaded."
    try:
        q = query.lower().strip()
        if q == "describe":
            return df.describe(include="all").to_string()
        if q == "correlation":
            return df.select_dtypes(include=np.number).corr().round(3).to_string()
        if q == "missing":
            m = df.isnull().sum()
            pct = (m / len(df) * 100).round(2)
            res = pd.DataFrame({"Count": m, "Pct": pct})
            filtered = res[res["Count"] > 0]
            return filtered.to_string() if not filtered.empty else "No missing values."
        if q == "dtypes":
            return df.dtypes.to_string()
        if q.startswith("outliers:"):
            col = q.split(":", 1)[1].strip()
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {list(df.columns)}"
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outs = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            return f"{len(outs)} outliers found in '{col}':\n{outs[[col]].head(20).to_string()}"
        if q.startswith("unique:"):
            col = q.split(":", 1)[1].strip()
            return f"Unique values in '{col}': {df[col].unique().tolist()[:50]}"
        return f"Unknown query. Supported: describe, correlation, missing, dtypes, outliers:COL, unique:COL"
    except Exception:
        return f"Stats error: {traceback.format_exc()}"


@tool
def plot_tool(plot_spec: str) -> str:
    """
    Generate a chart from the DataFrame.
    Format: 'chart_type|x_column|y_column|title'
    Chart types: bar, line, scatter, hist, heatmap, box, pie
    Examples:
      'bar|Category|Revenue|Revenue by Category'
      'line|Month|Sales|Monthly Sales Trend'
      'heatmap|||Correlation Matrix'
      'hist|Price||Price Distribution'
    """
    df = _get_df()
    if df is None:
        return "Error: No dataset loaded."
    try:
        parts = (plot_spec + "|||").split("|")
        chart_type = parts[0].strip().lower()
        x_col = parts[1].strip() or None
        y_col = parts[2].strip() or None
        title = parts[3].strip() or chart_type.title()

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#1c2128")
        ax.set_facecolor("#161b22")
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color("#e6edf3")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        palette = ["#667eea", "#764ba2", "#58a6ff", "#3fb950", "#f78166", "#ffd700"]

        if chart_type == "bar" and x_col and y_col:
            data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(15)
            ax.bar(data.index.astype(str), data.values, color=palette[0], edgecolor="#30363d")
            ax.set_xlabel(x_col); ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "line" and x_col and y_col:
            data = df.sort_values(x_col)
            ax.plot(data[x_col].astype(str), data[y_col], color=palette[0], linewidth=2.5, marker="o", markersize=4)
            ax.fill_between(range(len(data)), data[y_col].values, alpha=0.15, color=palette[0])
            plt.xticks(rotation=45, ha="right")
            ax.set_xlabel(x_col); ax.set_ylabel(y_col)

        elif chart_type == "scatter" and x_col and y_col:
            ax.scatter(df[x_col], df[y_col], c=palette[0], alpha=0.6, edgecolors="#30363d", linewidth=0.5)
            ax.set_xlabel(x_col); ax.set_ylabel(y_col)

        elif chart_type == "hist":
            col = x_col or df.select_dtypes(include=np.number).columns[0]
            ax.hist(df[col].dropna(), bins=30, color=palette[0], edgecolor="#30363d", alpha=0.85)
            ax.set_xlabel(col); ax.set_ylabel("Frequency")

        elif chart_type == "heatmap":
            corr = df.select_dtypes(include=np.number).corr()
            sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                        linewidths=0.5, linecolor="#30363d")

        elif chart_type == "box":
            col = y_col or x_col or df.select_dtypes(include=np.number).columns[0]
            bp = ax.boxplot(df[col].dropna(), patch_artist=True)
            bp["boxes"][0].set_facecolor(palette[0])
            bp["medians"][0].set_color("#ffd700")
            ax.set_ylabel(col)

        elif chart_type == "pie" and x_col and y_col:
            data = df.groupby(x_col)[y_col].sum().head(8)
            ax.pie(data.values, labels=data.index.astype(str),
                   colors=palette, autopct="%1.1f%%",
                   textprops={"color": "#e6edf3"})

        else:
            return f"Could not render chart. Check column names exist: {list(df.columns)}"

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#e6edf3")
        plt.tight_layout()
        st.session_state.charts.append(fig)
        return f"Chart '{title}' generated and will be shown in the output panel."
    except Exception:
        return f"Plot error: {traceback.format_exc()}"


# ─── Build Agent ──────────────────────────────────────────────────────────────

AGENT_PROMPT = PromptTemplate.from_template(
    "You are an expert data scientist AI assistant. Analyze the user's dataset using your tools.\n"
    "Always call dataset_info_tool first to understand the data, then proceed step by step.\n"
    "Generate visualizations when helpful. End with clear business insights.\n\n"
    "Available tools: {tools}\n"
    "Tool names: {tool_names}\n\n"
    "Use this format:\n"
    "Thought: <reasoning>\n"
    "Action: <tool_name>\n"
    "Action Input: <input>\n"
    "Observation: <result>\n"
    "... repeat as needed ...\n"
    "Thought: I have enough information.\n"
    "Final Answer: <comprehensive answer with insights>\n\n"
    "Human: {input}\n"
    "Agent scratchpad: {agent_scratchpad}"
)

def build_agent(api_key: str) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    tools = [dataset_info_tool, pandas_tool, stats_tool, plot_tool]
    agent = create_react_agent(llm=llm, tools=tools, prompt=AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False,
                         max_iterations=10, handle_parsing_errors=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Data Analysis Agent")
    st.markdown("*Natural language → instant insights*")
    st.divider()

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.success(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")

    if api_key and st.session_state.df is not None:
        st.session_state.agent_executor = build_agent(api_key)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.charts = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Built by [Satya Sai Prakash Kantamani](https://github.com/kantamaniprakash)**")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Data Analysis AI Agent")
st.markdown("*Ask anything about your dataset in plain English*")

if st.session_state.df is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
    st.markdown("""
### 🚀 What this agent can do:
- **Explore** your dataset automatically (shape, dtypes, missing values)
- **Analyze** trends, distributions, correlations
- **Detect** outliers and anomalies
- **Visualize** data with bar, line, scatter, heatmap charts
- **Summarize** key business insights

### Example prompts:
> *"Give me an overview of this dataset and the top insights"*
> *"Show me a correlation heatmap of all numeric columns"*
> *"Which category has the highest total sales? Show as a bar chart"*
> *"Find outliers in the revenue column"*
> *"What are the top 10 rows by profit?"*
    """)
elif not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable the agent.")
else:
    # Display conversation
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="agent-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Display any pending charts
    if st.session_state.charts:
        for chart in st.session_state.charts[-3:]:  # Show last 3 charts
            st.pyplot(chart)

    # User input
    if prompt := st.chat_input("Ask something about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-msg">👤 {prompt}</div>', unsafe_allow_html=True)
        st.session_state.charts = []  # clear old charts for new query

        with st.spinner("Agent is analyzing your data..."):
            try:
                result = st.session_state.agent_executor.invoke({"input": prompt})
                answer = result.get("output", "No answer returned.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(f'<div class="agent-msg">🤖 {answer}</div>', unsafe_allow_html=True)

                # Show any charts generated during this run
                for chart in st.session_state.charts:
                    st.pyplot(chart)
            except Exception as e:
                st.error(f"Agent error: {e}")

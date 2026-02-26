# 🧠 Data Analysis AI Agent

> An autonomous LLM-powered agent that understands natural language data analysis requests, writes & executes Python code, generates visualizations, and delivers business insights — all in a conversational interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain_Agents-0.2+-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)

---

## Overview

Traditional data analysis requires writing code for every query. This agent **understands natural language**, autonomously decides what to compute, **writes and executes Python**, and returns charts + insights — no coding required.

### Key Features
- **Natural language → Python code** — the agent writes pandas/matplotlib code automatically
- **Code execution in sandbox** — runs code safely and returns results
- **Auto-visualization** — generates appropriate charts (bar, line, scatter, heatmap) based on query
- **Iterative reasoning** — uses ReAct framework (Reason → Act → Observe → Repeat)
- **Multi-tool agent** — uses pandas tool, plotting tool, stats tool, and web search
- **CSV/Excel upload** — bring any dataset

---

## How It Works (ReAct Agent Loop)

```
User: "Show me sales trend by month and find anomalies"
         │
         ▼
    ┌─────────────┐
    │   REASON    │  GPT-4o decides: "I need to load data, group by month,
    │             │   plot line chart, then check for outliers"
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │     ACT     │  Calls: pandas_tool("df.groupby('month')['sales'].sum()")
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   OBSERVE   │  Gets result: monthly sales DataFrame
    └──────┬──────┘
           │
    (repeats until complete)
           │
           ▼
    Final Answer: Chart + insights + anomaly report
```

---

## Tech Stack

| Component        | Technology                              |
|-----------------|-----------------------------------------|
| LLM             | OpenAI GPT-4o / GPT-4o-mini             |
| Agent Framework | LangChain AgentExecutor (ReAct)         |
| Tools           | Custom pandas, matplotlib, stats tools  |
| Data Processing | pandas, numpy                           |
| Visualization   | matplotlib, seaborn, plotly             |
| UI              | Streamlit                               |
| Code Execution  | Python exec() in isolated namespace     |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kantamaniprakash/data-analysis-agent.git
cd data-analysis-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 4. Run the app
```bash
streamlit run agent.py
```

---

## Usage

1. Open `http://localhost:8501`
2. Upload a CSV or Excel file in the sidebar
3. Type your analysis request in plain English
4. The agent reasons through the problem and returns code + charts + insights

### Example Prompts
- *"Show me the distribution of all numerical columns"*
- *"Find the top 10 customers by revenue and visualize as a bar chart"*
- *"Is there a correlation between price and quantity sold?"*
- *"Detect outliers in the sales column and explain them"*
- *"Summarize this dataset in 5 key business insights"*
- *"Predict next month's sales using a linear trend"*

---

## Project Structure

```
data-analysis-agent/
├── agent.py            # Main Streamlit app + LangChain agent
├── tools.py            # Custom agent tools (pandas, plot, stats)
├── requirements.txt
└── README.md
```

---

## Agent Tools

| Tool              | Description                                          |
|------------------|------------------------------------------------------|
| `pandas_tool`    | Execute pandas code on the uploaded DataFrame        |
| `plot_tool`      | Generate matplotlib/seaborn visualizations           |
| `stats_tool`     | Run statistical analysis (correlation, describe, IQR)|
| `insight_tool`   | Generate business insights from computed results     |

---

## Results & Capabilities

- **Handles datasets up to ~500K rows** efficiently with pandas
- **Generates publication-quality charts** automatically
- **Zero code required** from the user — plain English only
- **Iterative refinement** — ask follow-up questions naturally

---

## Future Improvements
- [ ] Add SQL database connectivity (PostgreSQL, Snowflake)
- [ ] Integrate with LangChain SQL Agent for database queries
- [ ] Export analysis reports as PDF
- [ ] Add predictive modeling tool (sklearn integration)
- [ ] Deploy to AWS Lambda + S3 for serverless execution

---

## Author

**Satya Sai Prakash Kantamani** — Data Scientist
[GitHub](https://github.com/kantamaniprakash) · [LinkedIn](https://www.linkedin.com/in/satya-sai-prakash-kantamani) · [Email](mailto:satyasai.kantamani@gmail.com)

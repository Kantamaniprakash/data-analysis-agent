"""
FAANG-Level Data Analysis Agent
=================================
An autonomous LLM agent that analyzes datasets using natural language.
Built with LangChain, Streamlit, Plotly, and advanced ML/statistics tooling.

Author: Satya Sai Prakash Kantamani
GitHub: https://github.com/kantamaniprakash/data-analysis-agent
"""

import io, sys, traceback, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── Optional Scientific Imports ────────────────────────────────────────────────
try:
    from scipy import stats as scipy_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    SM_OK = True
except ImportError:
    SM_OK = False

try:
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                   GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  mean_squared_error, mean_absolute_error, r2_score,
                                  silhouette_score)
    from sklearn.dummy import DummyClassifier, DummyRegressor
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAANG-Level Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
.user-msg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 1.3rem; border-radius: 16px 16px 4px 16px;
    margin: 0.6rem 0 0.6rem auto; max-width: 72%; color: white;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}
.agent-msg {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 4px 16px 16px 16px; padding: 1.1rem 1.4rem;
    margin: 0.6rem 0; max-width: 90%; color: #e6edf3;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3); line-height: 1.75;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for k, v in {"df": None, "messages": [], "charts": [], "chat_history": [], "agent_executor": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def _df():
    return st.session_state.df

SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "list": list, "dict": dict,
    "str": str, "int": int, "float": float, "bool": bool, "type": type,
    "isinstance": isinstance, "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "sorted": sorted, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "set": set, "tuple": tuple, "any": any, "all": all,
    "hasattr": hasattr, "getattr": getattr, "vars": vars, "dir": dir,
}

def safe_exec(code: str) -> str:
    df = _df()
    if df is None:
        return "No dataset loaded."
    ns = {
        "df": df, "pd": pd, "np": np, "plt": plt, "sns": sns, "px": px, "go": go,
        "make_subplots": make_subplots,
        "scipy_stats": scipy_stats if SCIPY_OK else None,
        "sm": sm if SM_OK else None,
    }
    if SKLEARN_OK:
        ns.update({
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "LogisticRegression": LogisticRegression, "LinearRegression": LinearRegression,
            "StandardScaler": StandardScaler, "LabelEncoder": LabelEncoder,
            "KMeans": KMeans, "cross_val_score": cross_val_score,
            "train_test_split": train_test_split,
        })
    if XGB_OK:
        ns["xgb"] = xgb
    # Try eval first
    try:
        result = eval(code, {"__builtins__": {}}, ns)
        if result is not None:
            return result.to_string() if hasattr(result, "to_string") else str(result)
        return "Executed successfully."
    except SyntaxError:
        pass
    except Exception:
        pass
    # Try exec
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
        sys.stdout = old_stdout
        out = buf.getvalue()
        if "df" in ns and id(ns["df"]) != id(df):
            st.session_state.df = ns["df"]
        return out.strip() if out.strip() else "Executed successfully."
    except Exception:
        sys.stdout = old_stdout
        return f"Error:\n{traceback.format_exc()}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
@tool
def explore_data(query: str) -> str:
    """
    Comprehensive exploratory data analysis. ALWAYS call this first before any other analysis.
    Returns: shape, column types, missing value analysis, descriptive statistics with skewness
    and kurtosis, categorical value counts, duplicate analysis, top correlations, and data sample.
    Input: any string (e.g. 'full overview', 'missing values', 'categoricals')
    """
    df = _df()
    if df is None:
        return "No dataset loaded. Ask the user to upload a file."

    out = []
    out.append("=" * 62)
    out.append("DATASET OVERVIEW")
    out.append("=" * 62)
    out.append(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}   Memory: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()

    out.append(f"\nNumeric ({len(num_cols)}):     {num_cols}")
    out.append(f"Categorical ({len(cat_cols)}):  {cat_cols}")
    if dt_cols:   out.append(f"Datetime ({len(dt_cols)}):    {dt_cols}")
    if bool_cols: out.append(f"Boolean ({len(bool_cols)}):    {bool_cols}")

    # Missing
    out.append("\n" + "─" * 62)
    out.append("MISSING VALUES")
    out.append("─" * 62)
    missing = df.isnull().sum()
    miss_pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Count": missing, "Pct%": miss_pct}).query("Count > 0").sort_values("Count", ascending=False)
    if miss_df.empty:
        out.append("No missing values.")
    else:
        out.append(miss_df.to_string())
        out.append(f"\nTotal missing: {missing.sum():,} ({missing.sum()/df.size*100:.2f}% of all cells)")

    # Descriptive stats
    if num_cols:
        out.append("\n" + "─" * 62)
        out.append("DESCRIPTIVE STATISTICS")
        out.append("─" * 62)
        desc = df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).round(4)
        desc.loc["skewness"] = df[num_cols].skew().round(4)
        desc.loc["kurtosis"] = df[num_cols].kurtosis().round(4)
        out.append(desc.to_string())

    # Categorical
    if cat_cols:
        out.append("\n" + "─" * 62)
        out.append("CATEGORICAL COLUMNS")
        out.append("─" * 62)
        for col in cat_cols[:8]:
            n_unique = df[col].nunique()
            out.append(f"\n► {col}  ({n_unique} unique, {df[col].isnull().sum()} missing)")
            vc = df[col].value_counts(dropna=False).head(8)
            out.append(pd.DataFrame({"Count": vc, "Pct%": (vc/len(df)*100).round(1)}).to_string())

    # Duplicates
    n_dups = df.duplicated().sum()
    out.append(f"\nDuplicate rows: {n_dups:,} ({n_dups/len(df)*100:.2f}%)")

    # Top correlations
    if len(num_cols) >= 2:
        out.append("\n" + "─" * 62)
        out.append("TOP CORRELATIONS (|r| > 0.4)")
        out.append("─" * 62)
        corr = df[num_cols].corr().abs()
        pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                     .stack().sort_values(ascending=False).head(15))
        strong = pairs[pairs > 0.4]
        out.append(strong.round(3).to_string() if not strong.empty else "No strong correlations found.")

    # Sample
    out.append("\n" + "─" * 62)
    out.append("SAMPLE (first 5 rows)")
    out.append("─" * 62)
    out.append(df.head(5).to_string())

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — RUN CODE
# ══════════════════════════════════════════════════════════════════════════════
@tool
def run_code(code: str) -> str:
    """
    Execute any Python code on the loaded DataFrame.
    Use for custom analysis, complex aggregations, feature engineering, or anything
    not covered by the other tools. This is your Swiss army knife.

    Variables available: df, pd, np, px, go, make_subplots, scipy_stats, sm,
                         StandardScaler, LabelEncoder, KMeans, xgb, etc.
    Use print() to see output. You can modify df in-place.

    Examples:
      "df.groupby('Region')['Revenue'].agg(['mean','sum','count']).round(2)"
      "print(df.pivot_table(values='Sales', index='Year', columns='Category', aggfunc='sum'))"
      "df['log_price'] = np.log1p(df['Price']); print(df[['Price','log_price']].describe())"
    """
    return safe_exec(code)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — STATISTICAL TEST
# ══════════════════════════════════════════════════════════════════════════════
@tool
def statistical_test(spec: str) -> str:
    """
    Run rigorous statistical tests with full assumption checking, effect sizes, and plain-English interpretation.

    Format: "test_type:param1,param2,..."

    Tests available:
    - "normality:col"                   -> Shapiro-Wilk + D'Agostino + KS test with consensus
    - "ttest:col1,col2"                 -> Independent t-test (two numeric columns)
    - "ttest_groups:value_col,grp_col"  -> t-test between top 2 groups of a categorical column
    - "mannwhitney:col1,col2"           -> Mann-Whitney U (non-parametric t-test alternative)
    - "mannwhitney_groups:val,grp"      -> Mann-Whitney between groups of a categorical column
    - "anova:value_col,group_col"       -> One-way ANOVA with eta-squared effect size
    - "kruskal:value_col,group_col"     -> Kruskal-Wallis (non-parametric ANOVA)
    - "chi_square:col1,col2"            -> Chi-square test of independence + Cramer's V
    - "pearson:col1,col2"               -> Pearson correlation with 95% CI and R-squared
    - "spearman:col1,col2"              -> Spearman rank correlation with significance
    - "vif"                             -> Variance Inflation Factor for all numeric columns
    - "adf:col"                         -> Augmented Dickey-Fuller stationarity test
    - "kpss:col"                        -> KPSS stationarity test
    - "durbin_watson:col"               -> Autocorrelation test for regression residuals

    Always run normality test first to decide parametric vs non-parametric.
    """
    df = _df()
    if df is None:
        return "No dataset loaded."
    if not SCIPY_OK:
        return "scipy not installed. Run: pip install scipy"
    if ":" not in spec:
        return "Invalid format. Use 'test_type:params'. Example: 'normality:Age'"

    test_type, params_str = spec.split(":", 1)
    test_type = test_type.strip().lower()
    params = [p.strip() for p in params_str.split(",")]
    out = [f"{'='*62}", f"STATISTICAL TEST: {test_type.upper()}", f"{'='*62}"]

    try:
        # ── NORMALITY ──────────────────────────────────────────────────────────
        if test_type == "normality":
            col = params[0]
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {list(df.columns)}"
            data = df[col].dropna().values
            n = len(data)
            out.append(f"Column: {col}   n={n:,}")
            out.append(f"Mean={data.mean():.4f}  Std={data.std():.4f}  Skew={scipy_stats.skew(data):.4f}  Kurt={scipy_stats.kurtosis(data):.4f}")
            votes = 0
            if n <= 5000:
                sw_s, sw_p = scipy_stats.shapiro(data[:5000])
                out.append(f"\nShapiro-Wilk:       W={sw_s:.4f}, p={sw_p:.6f}  -> {'NORMAL' if sw_p>0.05 else 'NOT NORMAL'}")
                votes += int(sw_p > 0.05)
            dag_s, dag_p = scipy_stats.normaltest(data)
            out.append(f"D'Agostino-Pearson: K²={dag_s:.4f}, p={dag_p:.6f}  -> {'NORMAL' if dag_p>0.05 else 'NOT NORMAL'}")
            votes += int(dag_p > 0.05)
            ks_s, ks_p = scipy_stats.kstest(scipy_stats.zscore(data), "norm")
            out.append(f"Kolmogorov-Smirnov: D={ks_s:.4f},  p={ks_p:.6f}  -> {'NORMAL' if ks_p>0.05 else 'NOT NORMAL'}")
            votes += int(ks_p > 0.05)
            total = 3 if n <= 5000 else 2
            out.append(f"\nCONSENSUS: {votes}/{total} tests suggest normality")
            out.append("RECOMMENDATION: " + ("Use parametric tests (t-test, ANOVA)" if votes >= total//2+1 else "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis)"))

        # ── T-TEST ─────────────────────────────────────────────────────────────
        elif test_type in ("ttest", "ttest_groups"):
            if test_type == "ttest":
                a, b = df[params[0]].dropna().values, df[params[1]].dropna().values
                la, lb = params[0], params[1]
            else:
                val_col, grp_col = params[0], params[1]
                groups = df[grp_col].value_counts().index[:2]
                a = df[df[grp_col] == groups[0]][val_col].dropna().values
                b = df[df[grp_col] == groups[1]][val_col].dropna().values
                la, lb = str(groups[0]), str(groups[1])
            _, lev_p = scipy_stats.levene(a, b)
            _, pna = scipy_stats.shapiro(a[:500]) if len(a) < 5000 else (None, 0.5)
            _, pnb = scipy_stats.shapiro(b[:500]) if len(b) < 5000 else (None, 0.5)
            out.append(f"Group A ({la}): n={len(a):,}  mean={a.mean():.4f}  std={a.std():.4f}")
            out.append(f"Group B ({lb}): n={len(b):,}  mean={b.mean():.4f}  std={b.std():.4f}")
            out.append(f"\nAssumptions:  Normality A={'OK' if pna>0.05 else 'VIOLATED'}  B={'OK' if pnb>0.05 else 'VIOLATED'}  Equal var={'OK' if lev_p>0.05 else 'VIOLATED'} (Levene p={lev_p:.4f})")
            t, p = scipy_stats.ttest_ind(a, b, equal_var=(lev_p > 0.05))
            d = (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)
            out.append(f"\nWelch's t-test: t={t:.4f}  p={p:.6f}")
            out.append(f"Cohen's d: {d:.4f}  ({'Small' if abs(d)<0.5 else 'Medium' if abs(d)<0.8 else 'Large'} effect)")
            out.append(f"Mean difference: {a.mean()-b.mean():.4f}")
            out.append(f"CONCLUSION: {'Reject H0 — significant difference' if p<0.05 else 'Fail to reject H0 — no significant difference'} (alpha=0.05)")

        # ── MANN-WHITNEY ───────────────────────────────────────────────────────
        elif test_type in ("mannwhitney", "mannwhitney_groups"):
            if test_type == "mannwhitney":
                a, b = df[params[0]].dropna().values, df[params[1]].dropna().values
                la, lb = params[0], params[1]
            else:
                val_col, grp_col = params[0], params[1]
                groups = df[grp_col].value_counts().index[:2]
                a = df[df[grp_col] == groups[0]][val_col].dropna().values
                b = df[df[grp_col] == groups[1]][val_col].dropna().values
                la, lb = str(groups[0]), str(groups[1])
            u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            r = 1 - (2*u) / (len(a)*len(b))
            out.append(f"Group A ({la}): n={len(a):,}  median={np.median(a):.4f}")
            out.append(f"Group B ({lb}): n={len(b):,}  median={np.median(b):.4f}")
            out.append(f"\nMann-Whitney U={u:.0f}  p={p:.6f}")
            out.append(f"Rank-biserial r={r:.4f}  ({'Small' if abs(r)<0.3 else 'Medium' if abs(r)<0.5 else 'Large'} effect)")
            out.append(f"CONCLUSION: {'Reject H0 — significant difference' if p<0.05 else 'No significant difference'}")

        # ── ANOVA ──────────────────────────────────────────────────────────────
        elif test_type == "anova":
            val_col, grp_col = params[0], params[1]
            grps = df[grp_col].dropna().unique()
            gdata = [df[df[grp_col]==g][val_col].dropna().values for g in grps if len(df[df[grp_col]==g][val_col].dropna()) > 0]
            f, p = scipy_stats.f_oneway(*gdata)
            all_d = np.concatenate(gdata)
            gm = all_d.mean()
            ss_b = sum(len(g)*(g.mean()-gm)**2 for g in gdata)
            ss_t = sum((x-gm)**2 for x in all_d)
            eta2 = ss_b / ss_t
            out.append(f"One-way ANOVA: {val_col} ~ {grp_col}  ({len(gdata)} groups)")
            for g, gd in zip(grps[:12], gdata[:12]):
                out.append(f"  {g}: n={len(gd):,}  mean={gd.mean():.4f}  std={gd.std():.4f}")
            out.append(f"\nF={f:.4f}  p={p:.6f}")
            out.append(f"Eta-squared: {eta2:.4f}  ({'Small' if eta2<0.06 else 'Medium' if eta2<0.14 else 'Large'} effect)")
            out.append(f"CONCLUSION: {'Reject H0 — significant group differences' if p<0.05 else 'No significant group differences'}")
            if p < 0.05:
                out.append("Recommend: Run post-hoc pairwise comparisons (Tukey HSD via run_code)")

        # ── KRUSKAL-WALLIS ─────────────────────────────────────────────────────
        elif test_type == "kruskal":
            val_col, grp_col = params[0], params[1]
            grps = df[grp_col].dropna().unique()
            gdata = [df[df[grp_col]==g][val_col].dropna().values for g in grps if len(df[df[grp_col]==g][val_col].dropna()) > 0]
            h, p = scipy_stats.kruskal(*gdata)
            out.append(f"Kruskal-Wallis: {val_col} ~ {grp_col}  ({len(gdata)} groups)")
            for g, gd in zip(grps[:12], gdata[:12]):
                out.append(f"  {g}: n={len(gd):,}  median={np.median(gd):.4f}")
            out.append(f"\nH={h:.4f}  p={p:.6f}")
            out.append(f"CONCLUSION: {'Reject H0 — significant differences' if p<0.05 else 'No significant differences'}")

        # ── CHI-SQUARE ─────────────────────────────────────────────────────────
        elif test_type == "chi_square":
            col1, col2 = params[0], params[1]
            ct = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
            n = ct.values.sum()
            v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))
            out.append(f"Chi-square: {col1} x {col2}  (table {ct.shape})")
            out.append(f"\nContingency table:\n{ct.to_string()}")
            if expected.min() < 5:
                out.append(f"\n⚠  Min expected count={expected.min():.2f} (<5). Consider Fisher's exact or collapsing categories.")
            out.append(f"\nchi2={chi2:.4f}  df={dof}  p={p:.6f}")
            out.append(f"Cramer's V={v:.4f}  ({'Negligible' if v<0.1 else 'Small' if v<0.3 else 'Medium' if v<0.5 else 'Large'} association)")
            out.append(f"CONCLUSION: {'Reject H0 — significant association' if p<0.05 else 'No significant association'}")

        # ── PEARSON ────────────────────────────────────────────────────────────
        elif test_type == "pearson":
            col1, col2 = params[0], params[1]
            data = df[[col1, col2]].dropna()
            r, p = scipy_stats.pearsonr(data[col1], data[col2])
            n = len(data)
            z = np.arctanh(r)
            se = 1/np.sqrt(n-3)
            lo, hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
            out.append(f"Pearson Correlation: {col1} vs {col2}  (n={n:,})")
            out.append(f"r={r:.4f}  p={p:.6f}  R²={r**2:.4f}")
            out.append(f"95% CI: ({lo:.4f}, {hi:.4f})")
            out.append(f"Strength: {'Negligible' if abs(r)<0.1 else 'Weak' if abs(r)<0.3 else 'Moderate' if abs(r)<0.5 else 'Strong' if abs(r)<0.7 else 'Very strong'}")
            out.append(f"Direction: {'Positive' if r>0 else 'Negative'}")
            out.append(f"CONCLUSION: {'Statistically significant' if p<0.05 else 'Not statistically significant'} correlation")

        # ── SPEARMAN ───────────────────────────────────────────────────────────
        elif test_type == "spearman":
            col1, col2 = params[0], params[1]
            data = df[[col1, col2]].dropna()
            r, p = scipy_stats.spearmanr(data[col1], data[col2])
            out.append(f"Spearman Correlation: {col1} vs {col2}  (n={len(data):,})")
            out.append(f"rho={r:.4f}  p={p:.6f}")
            out.append(f"Strength: {'Negligible' if abs(r)<0.1 else 'Weak' if abs(r)<0.3 else 'Moderate' if abs(r)<0.5 else 'Strong' if abs(r)<0.7 else 'Very strong'}")
            out.append(f"CONCLUSION: {'Statistically significant' if p<0.05 else 'Not statistically significant'} monotonic relationship")

        # ── VIF ────────────────────────────────────────────────────────────────
        elif test_type == "vif":
            if not SM_OK:
                return "statsmodels not installed. Run: pip install statsmodels"
            num_cols = df.select_dtypes(include=np.number).dropna(axis=1).columns.tolist()
            X = df[num_cols].dropna()
            X_const = sm.add_constant(X)
            vif_df = pd.DataFrame({
                "Feature": X.columns,
                "VIF": [variance_inflation_factor(X_const.values, i+1) for i in range(len(X.columns))]
            }).sort_values("VIF", ascending=False)
            vif_df["Risk"] = vif_df["VIF"].apply(lambda v: "HIGH (>10)" if v>10 else "MODERATE (5-10)" if v>5 else "OK (<5)")
            out.append("Variance Inflation Factor — Multicollinearity Check")
            out.append(vif_df.to_string(index=False))
            out.append("\nVIF>10: Severe multicollinearity — remove or combine features")

        # ── ADF ────────────────────────────────────────────────────────────────
        elif test_type == "adf":
            if not SM_OK:
                return "statsmodels not installed. Run: pip install statsmodels"
            col = params[0]
            data = df[col].dropna().values
            res = adfuller(data, autolag="AIC")
            out.append(f"Augmented Dickey-Fuller Test: {col}  (H0: non-stationary)")
            out.append(f"ADF statistic={res[0]:.4f}  p={res[1]:.6f}  lags={res[2]}")
            for k, v in res[4].items():
                out.append(f"  Critical value {k}: {v:.4f}")
            out.append(f"CONCLUSION: {'STATIONARY' if res[1]<0.05 else 'NON-STATIONARY'}")
            if res[1] >= 0.05:
                out.append("Tip: Apply first differencing (df[col].diff()) or log transform")

        # ── KPSS ───────────────────────────────────────────────────────────────
        elif test_type == "kpss":
            if not SM_OK:
                return "statsmodels not installed. Run: pip install statsmodels"
            col = params[0]
            data = df[col].dropna().values
            stat, p, lags, crit = kpss(data, regression="c", nlags="auto")
            out.append(f"KPSS Test: {col}  (H0: stationary)")
            out.append(f"KPSS stat={stat:.4f}  p={p:.6f}  lags={lags}")
            for k, v in crit.items():
                out.append(f"  Critical value {k}: {v:.4f}")
            out.append(f"CONCLUSION: {'NON-STATIONARY' if p<0.05 else 'STATIONARY'}")
            out.append("Note: Use ADF + KPSS together for definitive stationarity diagnosis.")

        # ── DURBIN-WATSON ──────────────────────────────────────────────────────
        elif test_type == "durbin_watson":
            if not SM_OK:
                return "statsmodels not installed."
            col = params[0]
            data = df[col].dropna().values
            dw = durbin_watson(data)
            out.append(f"Durbin-Watson: {col}   DW={dw:.4f}")
            out.append("~2.0: No autocorrelation  |  <1.5: Positive  |  >2.5: Negative")
            verdict = "No autocorrelation" if 1.5<dw<2.5 else ("Positive autocorrelation" if dw<1.5 else "Negative autocorrelation")
            out.append(f"CONCLUSION: {verdict}")

        else:
            return f"Unknown test '{test_type}'. See tool description for all supported tests."

    except Exception:
        return f"Error in {test_type}:\n{traceback.format_exc()}"

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — CREATE CHART
# ══════════════════════════════════════════════════════════════════════════════
@tool
def create_chart(spec: str) -> str:
    """
    Create interactive Plotly charts. Charts appear below your response.

    Format: "chart_type|x_col|y_col|color_col|title"
    Use 'none' for unused columns.

    Chart types:
    - histogram|col|none|none|Title             -> Distribution with KDE overlay
    - bar|x_col|y_col|none|Title                -> Bar chart (auto-sums y by x, top 20)
    - bar_h|x_col|y_col|none|Title              -> Horizontal bar chart
    - line|x_col|y_col|color_col|Title          -> Line chart (color for multiple series)
    - scatter|x_col|y_col|color_col|Title       -> Scatter with trend line
    - scatter_matrix|col1,col2,col3|none|color|Title  -> Pairplot
    - box|x_col|y_col|none|Title                -> Box plot (x=groups, y=values)
    - violin|x_col|y_col|none|Title             -> Violin plot
    - heatmap|none|none|none|Title              -> Correlation heatmap of all numerics
    - pie|label_col|value_col|none|Title        -> Pie/donut chart
    - area|x_col|y_col|color_col|Title          -> Area chart
    - treemap|path_col|value_col|none|Title     -> Treemap
    - histogram_2d|x_col|y_col|none|Title       -> 2D density heatmap
    - time_series|date_col|value_col|none|Title -> Time series line chart

    Examples:
      "bar|Category|Revenue|none|Revenue by Category"
      "scatter|Age|Salary|Department|Salary vs Age by Department"
      "heatmap|none|none|none|Feature Correlation Matrix"
      "box|Region|Sales|none|Sales by Region"
    """
    df = _df()
    if df is None:
        return "No dataset loaded."

    parts = (spec + "||||").split("|")
    chart_type = parts[0].strip().lower()
    x_col   = parts[1].strip() if parts[1].strip() not in ("", "none") else None
    y_col   = parts[2].strip() if parts[2].strip() not in ("", "none") else None
    clr_col = parts[3].strip() if parts[3].strip() not in ("", "none") else None
    title   = parts[4].strip() or chart_type.replace("_", " ").title()
    TEMPLATE = "plotly_dark"

    try:
        fig = None

        if chart_type == "histogram":
            col = x_col or df.select_dtypes(include=np.number).columns[0]
            fig = px.histogram(df, x=col, nbins=40, marginal="violin",
                               color_discrete_sequence=["#667eea"],
                               template=TEMPLATE, title=title, opacity=0.85)
            fig.update_traces(marker_line_width=0.5, marker_line_color="#30363d")

        elif chart_type in ("bar", "bar_h"):
            if x_col and y_col:
                agg = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
                agg = agg.sort_values(y_col, ascending=False).head(25)
                if chart_type == "bar_h":
                    fig = px.bar(agg, x=y_col, y=x_col, orientation="h",
                                 color=y_col, color_continuous_scale="Blues",
                                 template=TEMPLATE, title=title)
                else:
                    fig = px.bar(agg, x=x_col, y=y_col,
                                 color=y_col, color_continuous_scale="Blues",
                                 template=TEMPLATE, title=title)

        elif chart_type == "line":
            if x_col and y_col:
                data = df.sort_values(x_col)
                fig = px.line(data, x=x_col, y=y_col, color=clr_col,
                              template=TEMPLATE, title=title, markers=True)

        elif chart_type == "scatter":
            if x_col and y_col:
                try:
                    fig = px.scatter(df, x=x_col, y=y_col, color=clr_col,
                                     template=TEMPLATE, title=title, trendline="ols",
                                     opacity=0.7, hover_data=df.columns[:5].tolist())
                except Exception:
                    fig = px.scatter(df, x=x_col, y=y_col, color=clr_col,
                                     template=TEMPLATE, title=title, opacity=0.7)

        elif chart_type == "scatter_matrix":
            cols = [c.strip() for c in (x_col or "").split(",") if c.strip() in df.columns]
            if not cols:
                cols = df.select_dtypes(include=np.number).columns[:5].tolist()
            fig = px.scatter_matrix(df, dimensions=cols, color=clr_col,
                                    template=TEMPLATE, title=title, opacity=0.6)

        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=clr_col,
                         template=TEMPLATE, title=title, points="outliers", notched=True)

        elif chart_type == "violin":
            fig = px.violin(df, x=x_col, y=y_col, color=clr_col,
                            template=TEMPLATE, title=title, box=True, points="outliers")

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include=np.number)
            corr = num_df.corr().round(3)
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
                hovertemplate="Corr(%{x}, %{y})=%{z:.3f}<extra></extra>"
            ))
            fig.update_layout(template=TEMPLATE, title=title, height=600)

        elif chart_type == "pie":
            if x_col and y_col:
                agg = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
                agg = agg.sort_values(y_col, ascending=False).head(10)
                fig = px.pie(agg, names=x_col, values=y_col,
                             template=TEMPLATE, title=title, hole=0.38,
                             color_discrete_sequence=px.colors.qualitative.Bold)

        elif chart_type == "area":
            if x_col and y_col:
                data = df.sort_values(x_col)
                fig = px.area(data, x=x_col, y=y_col, color=clr_col,
                              template=TEMPLATE, title=title)

        elif chart_type == "treemap":
            if x_col and y_col:
                agg = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
                fig = px.treemap(agg, path=[x_col], values=y_col,
                                 template=TEMPLATE, title=title)

        elif chart_type == "histogram_2d":
            if x_col and y_col:
                fig = px.density_heatmap(df, x=x_col, y=y_col,
                                         template=TEMPLATE, title=title,
                                         marginal_x="histogram", marginal_y="histogram")

        elif chart_type == "time_series":
            if x_col and y_col:
                data = df.copy()
                try:
                    data[x_col] = pd.to_datetime(data[x_col])
                    data = data.sort_values(x_col)
                except Exception:
                    pass
                fig = px.line(data, x=x_col, y=y_col, color=clr_col,
                              template=TEMPLATE, title=title)
                fig.update_traces(line=dict(width=2.5))

        if fig is None:
            return f"Could not create chart. Available columns: {list(df.columns)}"

        fig.update_layout(
            font=dict(family="Calibri, Arial, sans-serif", size=13),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(t=60, l=50, r=30, b=50),
        )
        st.session_state.charts.append(fig)
        return f"Chart '{title}' created and will appear below this response."

    except Exception:
        return f"Chart error:\n{traceback.format_exc()}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
@tool
def train_model(spec: str) -> str:
    """
    Train, cross-validate, evaluate, and explain a machine learning model.
    Automatically compares against a baseline, runs stratified k-fold CV,
    reports full metrics, and generates feature importance charts.

    Format: "task|target_col|feature_cols|model_type"

    task:          classification, regression, clustering
    target_col:    column to predict (use 'none' for clustering)
    feature_cols:  'all' for all numeric columns, or comma-separated names
    model_type:    auto, random_forest, gradient_boost, xgboost, logistic,
                   linear, ridge, lasso, kmeans

    Examples:
    - "classification|Churn|all|random_forest"
    - "regression|Price|Area,Rooms,Location|xgboost"
    - "clustering|none|all|kmeans"
    - "classification|Survived|all|auto"
    """
    if not SKLEARN_OK:
        return "scikit-learn not installed. Run: pip install scikit-learn"
    df = _df()
    if df is None:
        return "No dataset loaded."

    parts = (spec + "|||").split("|")
    task       = parts[0].strip().lower()
    target_col = parts[1].strip()
    feat_str   = parts[2].strip()
    model_type = parts[3].strip().lower() or "auto"
    out = [f"{'='*62}", f"ML MODEL: {task.upper()} | {model_type.upper()}", f"{'='*62}"]

    try:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if feat_str in ("all", ""):
            feature_cols = [c for c in num_cols if c != target_col]
        else:
            feature_cols = [c.strip() for c in feat_str.split(",") if c.strip() in df.columns]
        if not feature_cols:
            return "No valid feature columns found."

        # ── CLUSTERING ─────────────────────────────────────────────────────────
        if task == "clustering":
            X = df[feature_cols].dropna()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            k_range = range(2, min(11, len(X)//5+2))
            sil, iner = [], []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbls = km.fit_predict(Xs)
                iner.append(km.inertia_)
                sil.append(silhouette_score(Xs, lbls))
            best_k = list(k_range)[np.argmax(sil)]
            km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = km_final.fit_predict(Xs)
            st.session_state.df.loc[X.index, "Cluster"] = labels.astype(str)
            out.append(f"Features: {feature_cols}   Samples: {len(X):,}")
            out.append(f"Optimal k (best silhouette): {best_k}   Score: {max(sil):.4f}")
            out.append(f"\nCluster sizes:\n{pd.Series(labels).value_counts().sort_index().to_string()}")
            centers = pd.DataFrame(scaler.inverse_transform(km_final.cluster_centers_), columns=feature_cols)
            out.append(f"\nCluster centroids:\n{centers.round(3).to_string()}")
            if len(feature_cols) >= 2:
                df_plot = X.copy()
                df_plot["Cluster"] = labels.astype(str)
                fig = px.scatter(df_plot, x=feature_cols[0], y=feature_cols[1],
                                 color="Cluster", template="plotly_dark",
                                 title=f"K-Means Clustering (k={best_k})",
                                 color_discrete_sequence=px.colors.qualitative.Bold)
                st.session_state.charts.append(fig)
            return "\n".join(out)

        # ── SUPERVISED ─────────────────────────────────────────────────────────
        if target_col not in df.columns:
            return f"Target '{target_col}' not found. Columns: {list(df.columns)}"

        data = df[feature_cols + [target_col]].dropna()
        X = data[feature_cols]
        y = data[target_col]
        le = None
        class_names = None
        if task == "classification" and y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
            class_names = le.classes_

        out.append(f"Target: {target_col}   Features: {len(feature_cols)}   Samples: {len(data):,}")
        out.append(f"Features: {feature_cols}")

        if task == "classification":
            models = {
                "auto":           RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                "random_forest":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                "gradient_boost": GradientBoostingClassifier(n_estimators=200, random_state=42),
                "logistic":       LogisticRegression(max_iter=1000, random_state=42),
            }
            if XGB_OK:
                models["xgboost"] = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0)
            model = models.get(model_type, models["auto"])
            model_name = type(model).__name__

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            Xs = StandardScaler().fit_transform(X) if model_type == "logistic" else X
            bl_f1  = cross_val_score(DummyClassifier(strategy="most_frequent"), Xs, y, cv=cv, scoring="f1_weighted").mean()
            cv_f1  = cross_val_score(model, Xs, y, cv=cv, scoring="f1_weighted")
            cv_acc = cross_val_score(model, Xs, y, cv=cv, scoring="accuracy")

            out.append(f"\nModel: {model_name}")
            out.append(f"Baseline F1 (majority class): {bl_f1:.4f}")
            out.append(f"CV F1  (5-fold): {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
            out.append(f"CV Acc (5-fold): {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
            out.append(f"Lift over baseline: {cv_f1.mean()-bl_f1:+.4f}")

            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            out.append(f"\nHold-out Test Report:\n{classification_report(y_test, y_pred, target_names=class_names)}")
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                out.append(f"AUC-ROC: {auc:.4f}")

        elif task == "regression":
            models = {
                "auto":           RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                "random_forest":  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                "gradient_boost": GradientBoostingRegressor(n_estimators=200, random_state=42),
                "linear":         LinearRegression(),
                "ridge":          Ridge(alpha=1.0),
                "lasso":          Lasso(alpha=0.1, max_iter=2000),
            }
            if XGB_OK:
                models["xgboost"] = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
            model = models.get(model_type, models["auto"])
            model_name = type(model).__name__

            Xs = StandardScaler().fit_transform(X) if model_type in ("linear", "ridge", "lasso") else X
            bl_r2 = cross_val_score(DummyRegressor(strategy="mean"), Xs, y, cv=5, scoring="r2").mean()
            cv_r2 = cross_val_score(model, Xs, y, cv=5, scoring="r2")
            cv_rmse = -cross_val_score(model, Xs, y, cv=5, scoring="neg_root_mean_squared_error")

            out.append(f"\nModel: {model_name}")
            out.append(f"Baseline R2 (mean predictor): {bl_r2:.4f}")
            out.append(f"CV R2   (5-fold): {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
            out.append(f"CV RMSE (5-fold): {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")

            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            out.append(f"\nHold-out Test:")
            out.append(f"R2={r2_score(y_test,y_pred):.4f}  RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):.4f}  MAE={mean_absolute_error(y_test,y_pred):.4f}")
            out.append(f"MAPE={np.mean(np.abs((y_test-y_pred)/(np.abs(y_test)+1e-10)))*100:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.values, y=y_pred, mode="markers",
                                      opacity=0.65, marker=dict(color="#667eea", size=6), name="Predictions"))
            mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], line=dict(color="#f78166", dash="dash"), name="Perfect"))
            fig.update_layout(template="plotly_dark", title=f"Actual vs Predicted — {target_col}",
                              xaxis_title="Actual", yaxis_title="Predicted")
            st.session_state.charts.append(fig)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
            fi = fi.sort_values("Importance", ascending=True).tail(20)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            template="plotly_dark", title="Feature Importance",
                            color="Importance", color_continuous_scale="Blues")
            st.session_state.charts.append(fig_fi)
            out.append(f"\nTop 5 features:\n{fi.tail(5).sort_values('Importance',ascending=False).to_string(index=False)}")

        elif hasattr(model, "coef_"):
            coefs = model.coef_.flatten()[:len(feature_cols)]
            fi = pd.DataFrame({"Feature": feature_cols[:len(coefs)], "Coefficient": coefs})
            fi = fi.reindex(fi["Coefficient"].abs().sort_values().index)
            fig_fi = px.bar(fi, x="Coefficient", y="Feature", orientation="h",
                            template="plotly_dark", title="Model Coefficients",
                            color="Coefficient", color_continuous_scale="RdBu_r")
            st.session_state.charts.append(fig_fi)

    except Exception:
        return f"Model error:\n{traceback.format_exc()}"

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — TIME SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
@tool
def analyze_time_series(spec: str) -> str:
    """
    Comprehensive time series analysis: stationarity, decomposition, ACF/PACF, and forecasting.

    Format: "operation|date_col|value_col|extra"

    Operations:
    - "full|date_col|value_col|"         -> Complete analysis (all below combined)
    - "stationarity|date_col|value_col|" -> ADF + KPSS with joint interpretation
    - "decompose|date_col|value_col|"    -> STL decomposition chart (trend+seasonal+residual)
    - "acf_pacf|date_col|value_col|"     -> ACF and PACF plots for ARIMA order selection
    - "rolling|date_col|value_col|"      -> Rolling mean and std visualization
    - "forecast|date_col|value_col|periods=30" -> ARIMA(1,1,1) forecast with confidence intervals

    date_col: column name or 'index' if datetime is the index.

    Examples:
    - "full|Date|Revenue|"
    - "stationarity|index|Price|"
    - "forecast|Month|Sales|periods=12"
    - "decompose|Date|Visitors|"
    """
    if not SM_OK:
        return "statsmodels not installed. Run: pip install statsmodels"
    df = _df()
    if df is None:
        return "No dataset loaded."

    parts = (spec + "|||").split("|")
    op        = parts[0].strip().lower()
    date_col  = parts[1].strip()
    value_col = parts[2].strip()
    extra     = parts[3].strip()
    out = [f"{'='*62}", f"TIME SERIES ANALYSIS: {op.upper()}", f"{'='*62}"]

    try:
        ts = df.copy()
        if date_col != "index" and date_col in ts.columns:
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.sort_values(date_col).set_index(date_col)
        if value_col not in ts.columns:
            return f"Column '{value_col}' not found. Available: {list(ts.columns)}"
        series = ts[value_col].dropna()
        out.append(f"Series: {value_col}   Length: {len(series):,}")
        out.append(f"Range: {series.index[0]} -> {series.index[-1]}")
        out.append(f"Mean={series.mean():.4f}  Std={series.std():.4f}  Min={series.min():.4f}  Max={series.max():.4f}")

        # ── STATIONARITY ───────────────────────────────────────────────────────
        if op in ("stationarity", "full"):
            out.append("\n" + "─"*62)
            out.append("STATIONARITY (ADF + KPSS)")
            adf = adfuller(series, autolag="AIC")
            kpss_res = kpss(series, regression="c", nlags="auto")
            adf_stat = adf[1] < 0.05
            kpss_stat = kpss_res[1] > 0.05
            out.append(f"ADF:  stat={adf[0]:.4f}  p={adf[1]:.6f}  -> {'STATIONARY' if adf_stat else 'NON-STATIONARY'}")
            out.append(f"KPSS: stat={kpss_res[0]:.4f}  p={kpss_res[1]:.6f}  -> {'STATIONARY' if kpss_stat else 'NON-STATIONARY'}")
            if adf_stat and kpss_stat:
                out.append("JOINT CONCLUSION: STATIONARY. Safe for ARIMA modeling.")
            elif not adf_stat and not kpss_stat:
                out.append("JOINT CONCLUSION: NON-STATIONARY. Apply differencing (d=1).")
                diff1 = series.diff().dropna()
                adf2 = adfuller(diff1, autolag="AIC")
                out.append(f"After 1st diff: ADF p={adf2[1]:.6f} -> {'Stationary (use d=1)' if adf2[1]<0.05 else 'Still non-stationary (try d=2 or log)'}")
            else:
                out.append("JOINT CONCLUSION: Conflicting — series may be trend-stationary. Consider detrending.")

        # ── ROLLING STATS ──────────────────────────────────────────────────────
        if op in ("rolling", "full"):
            w = max(7, len(series)//10)
            rm = series.rolling(w).mean()
            rs = series.rolling(w).std()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index.astype(str), y=series.values, name="Actual", line=dict(color="#58a6ff", width=1.5)))
            fig.add_trace(go.Scatter(x=rm.index.astype(str), y=rm.values, name=f"Rolling Mean ({w})", line=dict(color="#f78166", width=2)))
            fig.add_trace(go.Scatter(x=rs.index.astype(str), y=rs.values, name=f"Rolling Std ({w})", line=dict(color="#ffd700", width=1.5, dash="dot")))
            fig.update_layout(template="plotly_dark", title=f"Rolling Statistics — {value_col}", xaxis_title="Date", yaxis_title="Value")
            st.session_state.charts.append(fig)

        # ── STL DECOMPOSITION ──────────────────────────────────────────────────
        if op in ("decompose", "full"):
            try:
                period = max(2, min(len(series)//4, 52))
                decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
                for i, (comp, color) in enumerate([(decomp.observed,"#58a6ff"),(decomp.trend,"#f78166"),(decomp.seasonal,"#3fb950"),(decomp.resid,"#ffd700")], 1):
                    fig.add_trace(go.Scatter(x=comp.index.astype(str), y=comp.values,
                                              line=dict(color=color, width=1.5), showlegend=False), row=i, col=1)
                fig.update_layout(height=600, template="plotly_dark", title=f"STL Decomposition — {value_col}")
                st.session_state.charts.append(fig)
                out.append(f"\nDecomposition: period={period} — chart generated.")
                out.append(f"Trend range: {decomp.trend.dropna().min():.4f} to {decomp.trend.dropna().max():.4f}")
                out.append(f"Seasonal amplitude: {decomp.seasonal.max()-decomp.seasonal.min():.4f}")
            except Exception as e:
                out.append(f"Decomposition note: {e}")

        # ── ACF / PACF ─────────────────────────────────────────────────────────
        if op in ("acf_pacf", "full"):
            nlags = min(40, len(series)//4)
            acf_v = acf(series, nlags=nlags)
            pacf_v = pacf(series, nlags=nlags)
            ci = 1.96/np.sqrt(len(series))
            fig = make_subplots(rows=2, cols=1, subplot_titles=[f"ACF (suggests MA order q)", f"PACF (suggests AR order p)"])
            for vals, row in [(acf_v, 1), (pacf_v, 2)]:
                for i, v in enumerate(vals):
                    fig.add_trace(go.Bar(x=[i], y=[v], width=0.6,
                                          marker_color="#58a6ff" if abs(v)>ci else "#30363d",
                                          showlegend=False), row=row, col=1)
                fig.add_hline(y=ci, line_dash="dash", line_color="red", opacity=0.7, row=row, col=1)
                fig.add_hline(y=-ci, line_dash="dash", line_color="red", opacity=0.7, row=row, col=1)
            fig.update_layout(template="plotly_dark", height=500, title=f"ACF & PACF — {value_col}")
            st.session_state.charts.append(fig)
            q_sug = sum(1 for v in acf_v[1:] if abs(v)>ci)
            p_sug = sum(1 for v in pacf_v[1:] if abs(v)>ci)
            out.append(f"\nSuggested ARIMA order: p~{min(p_sug,5)}, q~{min(q_sug,5)}")

        # ── FORECAST ───────────────────────────────────────────────────────────
        if op == "forecast":
            periods = 30
            if "periods=" in extra:
                try:
                    periods = int(extra.split("periods=")[1].split(",")[0])
                except Exception:
                    pass
            try:
                model = ARIMA(series, order=(1, 1, 1))
                result = model.fit()
                fc = result.get_forecast(steps=periods)
                mean_fc = fc.predicted_mean
                ci_df = fc.conf_int()
                try:
                    freq = pd.infer_freq(series.index) or "D"
                    future_idx = pd.date_range(series.index[-1], periods=periods+1, freq=freq)[1:]
                    future_str = future_idx.astype(str).tolist()
                except Exception:
                    future_str = [f"t+{i}" for i in range(1, periods+1)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index.astype(str), y=series.values,
                                          name="Historical", line=dict(color="#58a6ff", width=2)))
                fig.add_trace(go.Scatter(x=future_str, y=mean_fc.values,
                                          name="Forecast", line=dict(color="#f78166", width=2, dash="dash")))
                fig.add_trace(go.Scatter(
                    x=future_str + future_str[::-1],
                    y=list(ci_df.iloc[:, 0]) + list(ci_df.iloc[:, 1])[::-1],
                    fill="toself", fillcolor="rgba(247,129,102,0.18)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
                fig.update_layout(template="plotly_dark",
                                  title=f"ARIMA(1,1,1) Forecast — {value_col} (+{periods} periods)",
                                  xaxis_title="Date", yaxis_title="Value")
                st.session_state.charts.append(fig)
                out.append(f"\nARIMA(1,1,1) Forecast:")
                out.append(f"Next value: {mean_fc.iloc[0]:.4f}  |  In {periods} periods: {mean_fc.iloc[-1]:.4f}")
                out.append(f"AIC={result.aic:.2f}  BIC={result.bic:.2f}")
                out.append("For production: use pmdarima auto_arima or Prophet for better order selection.")
            except Exception as e:
                out.append(f"Forecast error: {e}")

    except Exception:
        return f"Time series error:\n{traceback.format_exc()}"

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
@tool
def data_quality(query: str) -> str:
    """
    Deep data quality assessment: missing value analysis with MCAR/MAR/MNAR classification,
    outlier detection (IQR + Z-score), duplicate analysis, data type issues,
    distribution anomalies (skewness, kurtosis), and constant/high-cardinality columns.

    Input: 'full' for complete report, or focus on:
    'missing', 'outliers', 'duplicates', 'types', 'distributions'
    """
    df = _df()
    if df is None:
        return "No dataset loaded."

    q = query.lower()
    out = [f"{'='*62}", "DATA QUALITY REPORT", f"{'='*62}"]

    try:
        # ── MISSING ────────────────────────────────────────────────────────────
        if q in ("full", "missing"):
            out.append("\n[1] MISSING VALUE ANALYSIS")
            out.append("─" * 62)
            missing = df.isnull().sum()
            miss_pct = (missing / len(df) * 100).round(2)
            miss_df = pd.DataFrame({"Missing": missing, "Pct%": miss_pct, "Dtype": df.dtypes.astype(str)})
            miss_df = miss_df[miss_df["Missing"] > 0].sort_values("Missing", ascending=False)
            if miss_df.empty:
                out.append("No missing values.")
            else:
                out.append(miss_df.to_string())
                out.append(f"\nTotal: {missing.sum():,} missing cells ({missing.sum()/df.size*100:.2f}% of dataset)")
                out.append("\nMissing Mechanism Classification (heuristic):")
                for col in miss_df.index:
                    pct = miss_df.loc[col, "Pct%"]
                    if pct < 5:
                        rec = "Likely MCAR — safe to drop rows or use simple mean/median imputation"
                    elif pct < 30:
                        rec = "Possibly MAR — use KNN imputation or model-based imputation (IterativeImputer)"
                    else:
                        rec = "Possibly MNAR — domain expertise required; flag and treat carefully"
                    out.append(f"  {col} ({pct}%): {rec}")

        # ── OUTLIERS ───────────────────────────────────────────────────────────
        if q in ("full", "outliers"):
            out.append("\n[2] OUTLIER ANALYSIS")
            out.append("─" * 62)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            outlier_summary = []
            for col in num_cols[:20]:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                pct = n_out / df[col].notna().sum() * 100
                if n_out > 0:
                    outlier_summary.append(f"  {col}: {n_out:,} outliers ({pct:.1f}%)  bounds=[{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")
            out.extend(outlier_summary if outlier_summary else ["No outliers detected via IQR method."])

        # ── DUPLICATES ─────────────────────────────────────────────────────────
        if q in ("full", "duplicates"):
            out.append("\n[3] DUPLICATE ANALYSIS")
            out.append("─" * 62)
            n = df.duplicated().sum()
            out.append(f"Exact duplicate rows: {n:,} ({n/len(df)*100:.2f}%)")
            if n > 0:
                out.append("Sample duplicates:")
                out.append(df[df.duplicated(keep="first")].head(3).to_string())

        # ── DATA TYPES ─────────────────────────────────────────────────────────
        if q in ("full", "types"):
            out.append("\n[4] DATA TYPE ISSUES")
            out.append("─" * 62)
            for col in df.select_dtypes(include="object").columns:
                try:
                    pd.to_numeric(df[col].dropna(), errors="raise")
                    out.append(f"  ⚠  '{col}' stored as string but looks numeric — use pd.to_numeric(df['{col}'])")
                except Exception:
                    pass
                if df[col].dropna().astype(str).str.match(r"\d{4}[-/]\d{2}[-/]\d{2}").mean() > 0.5:
                    out.append(f"  📅 '{col}' looks like a date — use pd.to_datetime(df['{col}'])")

        # ── DISTRIBUTIONS ──────────────────────────────────────────────────────
        if q in ("full", "distributions"):
            out.append("\n[5] DISTRIBUTION ANOMALIES")
            out.append("─" * 62)
            for col in df.select_dtypes(include=np.number).columns[:15]:
                skew = df[col].skew()
                kurt = df[col].kurtosis()
                if abs(skew) > 2:
                    out.append(f"  {col}: HIGH SKEWNESS ({skew:.2f}) — consider log/sqrt/Box-Cox transform")
                if abs(kurt) > 7:
                    out.append(f"  {col}: HIGH KURTOSIS ({kurt:.2f}) — heavy tails, extreme outlier risk")
            for col in df.columns:
                if df[col].nunique() == 1:
                    out.append(f"  ⚠  '{col}' is CONSTANT — zero variance, drop before modeling")
                elif df[col].nunique() / len(df) > 0.95 and df[col].dtype == "object":
                    out.append(f"  ⚠  '{col}' has very high cardinality ({df[col].nunique()} unique) — likely an ID column, exclude from features")

        out.append(f"\n{'='*62}")
        out.append("PRIORITY ACTIONS:")
        n_miss = df.isnull().sum().sum()
        n_dups = df.duplicated().sum()
        action_num = 1
        if n_miss > 0:
            out.append(f"  {action_num}. Address {n_miss:,} missing values (see classification above)")
            action_num += 1
        if n_dups > 0:
            out.append(f"  {action_num}. Remove {n_dups:,} duplicate rows: df.drop_duplicates(inplace=True)")
            action_num += 1
        out.append(f"  {action_num}. Verify data types — run the 'types' check if not already done")
        out.append(f"  {action_num+1}. Check outlier policy: remove measurement errors, cap real extremes (Winsorization)")

    except Exception:
        return f"Quality report error:\n{traceback.format_exc()}"

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT & AGENT
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an elite data scientist with a PhD in Statistics and 8+ years of experience at top technology companies (Google, Meta, Amazon, Netflix). You combine rigorous statistical thinking with sharp business acumen and clear, compelling communication.

YOUR MANDATORY WORKFLOW — follow this for every analysis:
1. UNDERSTAND  → Restate the question clearly. If ambiguous, ask ONE targeted clarifying question.
2. INSPECT     → Call explore_data FIRST. Always. No exceptions.
3. CLEAN       → Call data_quality proactively. Never analyze dirty data without flagging it.
4. ANALYZE     → Apply the right technique. For statistical tests, check normality first.
5. VISUALIZE   → Call create_chart for every key finding. Every insight deserves a chart.
6. MODEL       → Use train_model with proper cross-validation for prediction/segmentation tasks.
7. INTERPRET   → Translate numbers into plain English with business implications.
8. DELIVER     → End every response with: TL;DR, Key Findings (bulleted), Recommended Next Steps.

STATISTICAL RIGOR (non-negotiable):
- Check normality before choosing parametric vs non-parametric tests
- Always report effect size alongside p-value (Cohen's d, eta-squared, Cramer's V)
- "p < 0.05" alone is meaningless — explain practical significance too
- For time series stationarity: run BOTH ADF and KPSS (opposite null hypotheses)
- Apply Bonferroni correction when running multiple comparisons

DATA QUALITY RULES:
- Classify missing data as MCAR/MAR/MNAR — never just drop without reasoning
- Never silently remove data — always report what was removed and why
- Outlier treatment decision: measurement error → remove; real extreme → cap or use robust estimators

ML BEST PRACTICES:
- Always compare against a baseline model first (majority class / mean predictor)
- Stratified k-fold CV (5-fold minimum) — never just train/test split
- Report full metric suite: F1/AUC/precision/recall for classification; RMSE/MAE/R²/MAPE for regression
- Always generate feature importance chart

COMMUNICATION STYLE:
- Lead with a one-sentence TL;DR
- Quantify everything: "Revenue increased 23%" not "Revenue increased significantly"
- Use clear headers, numbered steps, bullet points
- Proactively suggest analyses the user hasn't asked for yet
- Acknowledge limitations and caveats honestly
- End EVERY response with "Recommended Next Steps"

You have 7 specialized tools: explore_data, run_code, statistical_test, create_chart, train_model, analyze_time_series, data_quality."""


def build_agent(api_key: str) -> AgentExecutor:
    llm = ChatAnthropic(
        model="claude-opus-4-6",
        temperature=0,
        anthropic_api_key=api_key,
        max_tokens=8096
    )
    tools = [explore_data, run_code, statistical_test, create_chart,
             train_model, analyze_time_series, data_quality]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=False,
        max_iterations=30, max_execution_time=600,
        handle_parsing_errors=True, return_intermediate_steps=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Data Analysis Agent")
    st.divider()

    st.markdown("### 🔑 API Key")
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...",
                             help="Enter your API key")
    if api_key:
        if st.session_state.agent_executor is None:
            with st.spinner("Loading..."):
                try:
                    st.session_state.agent_executor = build_agent(api_key)
                    st.success("✅ Ready")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        if st.button("🔄 Reload Agent", use_container_width=True):
            with st.spinner("Reloading..."):
                try:
                    st.session_state.agent_executor = build_agent(api_key)
                    st.success("✅ Reloaded")
                except Exception as e:
                    st.error(str(e))

    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("CSV, Excel, or JSON",
                                 type=["csv", "xlsx", "xls", "json"],
                                 help="Upload your dataset to begin analysis")
    if uploaded:
        try:
            name = uploaded.name
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            elif name.endswith(".json"):
                df = pd.read_json(uploaded)
            else:
                df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ {name}")
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Cols", df.shape[1])
            c1.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
            c2.metric("Missing%", f"{df.isnull().sum().sum()/df.size*100:.1f}%")
            with st.expander("Preview"):
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Load error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.charts = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**[Satya Sai Prakash Kantamani](https://kantamaniprakash.github.io)**")

# ── Helper: extract clean text from agent output ───────────────────────────────
def extract_answer(raw) -> str:
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(raw).strip()

# ── Helper: render agent message with proper markdown ─────────────────────────
def render_agent_msg(content: str):
    with st.container():
        st.markdown(content)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Data Analysis Agent")
st.markdown("*Ask anything about your data in plain English*")

if st.session_state.df is None:
    st.info("👈 Enter your API key and upload a dataset to get started.")
    st.markdown("""
### What this agent can do

| Capability | Details |
|---|---|
| **Exploratory Analysis** | Shape, types, missing values, correlations, descriptive stats |
| **Statistical Testing** | Normality, t-test, ANOVA, Mann-Whitney, Chi-square, Pearson/Spearman, VIF, ADF, KPSS |
| **Machine Learning** | Classification, regression, clustering with cross-validation and feature importance |
| **Time Series** | Stationarity tests, decomposition, ACF/PACF, ARIMA forecasting |
| **Data Quality** | Missing patterns, outlier detection, type issues, distribution anomalies |
| **Visualization** | 14 interactive chart types |
| **Custom Analysis** | Execute any pandas/numpy/sklearn code via natural language |

### Example prompts:
> *"Give me a full overview of this dataset with visualizations"*
> *"Is there a significant difference in sales across regions?"*
> *"Build a churn prediction model and explain what drives it"*
> *"Analyze the revenue time series and forecast the next 30 days"*
> *"Find all outliers and give me a data quality report"*
    """)

elif not st.session_state.agent_executor:
    st.warning("⚠️ Enter your API key in the sidebar to activate the agent.")

else:
    # Display conversation history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                for fig in msg.get("charts", []):
                    st.plotly_chart(fig, use_container_width=True)
                if msg.get("steps"):
                    with st.expander(f"Analysis steps ({len(msg['steps'])} tool calls)", expanded=False):
                        for i, (action, obs) in enumerate(msg["steps"], 1):
                            tool_name = getattr(action, "tool", str(action))
                            tool_input = getattr(action, "tool_input", {})
                            st.markdown(f"**Step {i}: `{tool_name}`**")
                            inp_str = list(tool_input.values())[0] if isinstance(tool_input, dict) and tool_input else str(tool_input)
                            st.code(str(inp_str)[:400])

    # Chat input
    if prompt := st.chat_input("Ask anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.charts = []

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    result = st.session_state.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    raw_answer = result.get("output", "No response generated.")
                    answer = extract_answer(raw_answer)
                    steps  = result.get("intermediate_steps", [])
                    charts = list(st.session_state.charts)

                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "charts": charts,
                        "steps": steps
                    })

                    st.markdown(answer)
                    for fig in charts:
                        st.plotly_chart(fig, use_container_width=True)

                    if steps:
                        with st.expander(f"Analysis steps ({len(steps)} tool calls)", expanded=False):
                            for i, (action, obs) in enumerate(steps, 1):
                                tool_name = getattr(action, "tool", str(action))
                                tool_input = getattr(action, "tool_input", {})
                                st.markdown(f"**Step {i}: `{tool_name}`**")
                                inp_str = list(tool_input.values())[0] if isinstance(tool_input, dict) and tool_input else str(tool_input)
                                st.code(str(inp_str)[:400])

                except Exception as e:
                    st.error(f"Agent error: {e}\n\n{traceback.format_exc()}")

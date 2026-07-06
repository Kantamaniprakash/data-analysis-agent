"""Smoke tests: verify key project files exist and the source parses.

agent.py is a Streamlit app whose import pulls in heavy optional
dependencies (anthropic, langchain, xgboost, ...), so instead of importing
it we assert that it parses cleanly with ast.parse.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_key_files_exist():
    assert (REPO_ROOT / "agent.py").is_file()
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / "README.md").is_file()


def test_agent_source_parses():
    source = (REPO_ROOT / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)


def test_agent_defines_expected_symbols():
    source = (REPO_ROOT / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "safe_exec" in top_level_functions

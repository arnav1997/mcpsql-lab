from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

import sqlparse
import psycopg
from psycopg.rows import dict_row

from mcp.server.fastmcp import FastMCP

import requests

# Create the server
app = FastMCP("mcp-sql-analyst")

# --- Basic tools -----------------------------------------------------------
@app.tool()
def echo(text: str) -> str:
    """Echo back whatever you send."""
    return text

@app.tool()
def ping() -> str:
    """Health check for the server."""
    return "pong"

# ===================== Day 2 – Core SQL Tools =============================
# Tools implemented:
#   - list_tables()
#   - describe_table(table)
#   - generate_sql(natural_language_query)  (LLM-assisted; SQL only)
#   - run_sql(sql)                          (safe, read-only)
# Plus a SELECT-only SQL sanitizer and read-only DB session.
#
# Dependencies (pyproject): mcp, psycopg[binary], sqlparse, requests
# Env:
#   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pagila
#   # Optional LLM backends for generate_sql:
#   OLLAMA_HOST=http://localhost:11434
#   OLLAMA_MODEL=llama3.1:8b
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-4o-mini

# ---------------------------
# Database connection helper
# ---------------------------

def _build_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "pagila")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def get_conn() -> psycopg.Connection:
    """Open a read-only Postgres connection with dict rows."""
    dsn = _build_dsn()
    # Make the whole session read-only for an extra safety layer.
    options = "-c default_transaction_read_only=on"
    return psycopg.connect(dsn, autocommit=True, options=options, row_factory=dict_row)

# ---------------------------
# SQL sanitizer (SELECT-only)
# ---------------------------

ALLOWED_START = ("SELECT", "WITH")
FORBIDDEN_KEYWORDS = {
    "INSERT","UPDATE","DELETE","MERGE","UPSERT","CREATE","ALTER","DROP","TRUNCATE",
    "GRANT","REVOKE","VACUUM","ANALYZE","REFRESH","CLUSTER","REINDEX","COPY","\\copy",
    "CALL","DO","SECURITY","FUNCTION","PROCEDURE","INDEX","SEQUENCE","EXTENSION"
}

# Statements that could be harmful even inside a SELECT
DANGEROUS_FUNCTION_PATTERNS = [
    r"\bpg_terminate_backend\s*\(",
    r"\bpg_sleep\s*\(",
]

MAX_LIMIT = int(os.getenv("SQL_MAX_LIMIT", "500"))

class SqlSanitizerError(Exception):
    pass


def sanitize_select_only(sql: str) -> str:
    """Ensure exactly one SELECT/WITH statement, block forbidden keywords, and enforce a LIMIT."""
    if not sql or not sql.strip():
        raise SqlSanitizerError("Empty SQL.")

    # Disallow multiple statements
    statements = [s for s in sqlparse.parse(sql) if s.tokens and not s.is_whitespace]
    if len(statements) != 1:
        raise SqlSanitizerError("Only a single SELECT statement is allowed.")

    stmt = statements[0]

    # Ensure it starts with SELECT or WITH
    first_token = next((t for t in stmt.flatten() if not t.is_whitespace), None)
    if not first_token:
        raise SqlSanitizerError("Invalid SQL.")
    if first_token.normalized.upper() not in ALLOWED_START:
        raise SqlSanitizerError("Only SELECT or WITH queries are allowed.")

    stmt_str = str(stmt)
    upper_sql = stmt_str.upper()

    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", upper_sql):
            raise SqlSanitizerError(f"Forbidden keyword detected: {kw}")

    for pat in DANGEROUS_FUNCTION_PATTERNS:
        if re.search(pat, upper_sql, flags=re.IGNORECASE):
            raise SqlSanitizerError("Dangerous function detected.")

    # Enforce a LIMIT if missing
    has_limit = re.search(r"\blimit\b\s+\d+", stmt_str, flags=re.IGNORECASE)
    if not has_limit:
        safe_sql = stmt_str.rstrip("; \n\t") + f" LIMIT {MAX_LIMIT}"
    else:
        safe_sql = stmt_str

    return safe_sql

# ---------------------------
# Tools
# ---------------------------

@app.tool()
def list_tables() -> Dict[str, Any]:
    """List user tables (schema, table, size in bytes)."""
    q = """
        SELECT n.nspname AS schema,
               c.relname AS table,
               pg_total_relation_size(c.oid) AS bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r'
          AND n.nspname NOT IN ('pg_catalog','information_schema')
        ORDER BY 1,2;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
    return {"tables": rows}


@app.tool()
def describe_table(table: str) -> Dict[str, Any]:
    """Describe a table's columns, PK, FKs, and non-PK indexes. Accepts schema.table or table."""
    if "." in table:
        schema, tbl = table.split(".", 1)
    else:
        schema, tbl = "public", table

    info: Dict[str, Any] = {}

    cols_sql = """
        SELECT
            c.ordinal_position AS position,
            c.column_name AS name,
            c.data_type AS type,
            c.is_nullable AS nullable,
            c.column_default AS default
        FROM information_schema.columns c
        WHERE c.table_schema = %s AND c.table_name = %s
        ORDER BY c.ordinal_position;
    """

    pk_sql = """
        SELECT a.attname AS column
        FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(i.indkey)
        WHERE i.indisprimary AND n.nspname = %s AND t.relname = %s
        ORDER BY a.attnum;
    """

    fk_sql = """
        SELECT
            kcu.column_name AS column,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = %s AND tc.table_name = %s
        ORDER BY kcu.ordinal_position;
    """

    idx_sql = """
        SELECT i.relname AS index_name, pg_get_indexdef(ix.indexrelid) AS definition
        FROM pg_class t
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_index ix ON ix.indrelid = t.oid
        JOIN pg_class i ON i.oid = ix.indexrelid
        WHERE n.nspname = %s AND t.relname = %s AND ix.indisprimary IS FALSE
        ORDER BY i.relname;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(cols_sql, (schema, tbl))
            info["columns"] = cur.fetchall()

            cur.execute(pk_sql, (schema, tbl))
            info["primary_key"] = [r["column"] for r in cur.fetchall()]

            cur.execute(fk_sql, (schema, tbl))
            info["foreign_keys"] = cur.fetchall()

            cur.execute(idx_sql, (schema, tbl))
            info["indexes"] = cur.fetchall()

    info["table"] = f"{schema}.{tbl}"
    return info


@app.tool()
def generate_sql(natural_language_query: str) -> Dict[str, Any]:
    """Generate SQL (Postgres) from a natural language request. Returns {sql: str}.
    Prefers Ollama if configured; falls back to OpenAI; otherwise returns a safe stub.
    Always post-processes with the SELECT-only sanitizer.
    """
    # Gather a small schema snapshot to help the model
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                ORDER BY table_schema, table_name, ordinal_position
                LIMIT 200
                """
            )
            cols = cur.fetchall()

    schema_hint: Dict[str, List[Dict[str, str]]] = {}
    for r in cols:
        key = f"{r['table_schema']}.{r['table_name']}"
        schema_hint.setdefault(key, []).append({"column": r["column_name"], "type": r["data_type"]})

    system_prompt = (
        "You are a SQL assistant for PostgreSQL. Return ONLY one valid SQL statement. "
        "Use explicit JOINs, snake_case, and include a LIMIT if not present."
    )
    user_prompt = {
        "task": natural_language_query,
        "schema_samples": {k: v[:8] for k, v in list(schema_hint.items())[:15]},
        "rules": [
            "Postgres dialect only",
            "No comments, no markdown, SQL only",
            f"Hard cap results with LIMIT {MAX_LIMIT} if not specified",
        ],
    }

    sql_text: Optional[str] = None

    # Try Ollama first
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        if ollama_host:
            resp = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": f"SYSTEM:\n{system_prompt}\n\nUSER:\n{json.dumps(user_prompt)}\n",
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            candidate = (data.get("response") or "").strip()
            sql_text = candidate
    except Exception:
        sql_text = None

    # Fallback to OpenAI if configured
    if not sql_text and os.getenv("OPENAI_API_KEY"):
        try:
            import openai  # optional dependency
            openai.api_key = os.environ["OPENAI_API_KEY"]
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ]
            res = openai.ChatCompletion.create(model=model, messages=msgs, temperature=0)
            sql_text = res["choices"][0]["message"]["content"].strip()
        except Exception:
            sql_text = None

    if not sql_text:
        # Graceful fallback
        return {"sql": f"-- LLM not configured.\n-- task: {natural_language_query}\nSELECT 1 LIMIT 1"}

    # Strip accidental markdown fences
    sql_text = re.sub(r"^```sql\s*|```$", "", sql_text.strip(), flags=re.IGNORECASE | re.MULTILINE)

    # Sanitize and ensure LIMIT
    try:
        sql_text = sanitize_select_only(sql_text)
    except Exception:
        sql_text = "SELECT 1 AS fallback LIMIT 1"

    return {"sql": sql_text}


@app.tool()
def run_sql(sql: str) -> Dict[str, Any]:
    """Run a read-only SQL query. Enforces SELECT-only sanitizer and LIMIT cap."""
    safe_sql = sanitize_select_only(sql)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(safe_sql)
            try:
                rows = cur.fetchall()
            except psycopg.ProgrammingError:
                rows = []
            cols = [desc.name for desc in cur.description] if cur.description else []

    return {"columns": cols, "rows": rows, "rowcount": len(rows)}

# ================== End Day 2 – Core SQL Tools ============================

# --- Tool registry loader --------------------------------------------------
# Each module in tools/ exposes a `register(mcp: FastMCP)` function.

def register_builtin_tools() -> None:
    from tools import echo as t_echo
    from tools import sysinfo as t_sysinfo
    t_echo.register(app)
    t_sysinfo.register(app)

register_builtin_tools()

# --- Entrypoints -----------------------------------------------------------
if __name__ == "__main__":
    # Run with stdio (works great with Claude Desktop / MCP Inspector)
    # Example: `uv run mcp dev server.py`
    import asyncio
    from mcp import cli
    asyncio.run(cli.run_dev(app))

## 🚀 Features

### Core
- **MCP server** built with `FastMCP("mcp-sql-analyst")`.
- **Inspector/UI entrypoint** via `uv run mcp dev server.py` (runs `mcp.cli.run_dev(app)`).

### Built-in Tools
- `echo(text: str)` — echoes text back (sanity test).
- `ping()` — health check (`"pong"`).

### SQL Tools
- `list_tables()` — lists user tables with:
  - schema
  - table
  - size in bytes (`pg_total_relation_size`).
- `describe_table(table: str)` — inspects table schema, including:
  - `columns` (position, name, type, nullable, default)
  - `primary_key` (PK columns)
  - `foreign_keys` (relationships with schema/table/column)
  - `indexes` (non-PK index definitions)
  - `table` (fully qualified name).
- `run_sql(sql: str)` — executes **read-only** SQL queries; returns:
  - `columns`
  - `rows`
  - `rowcount`.
- `generate_sql(natural_language_query: str)` — generates Postgres SQL from natural language:
  - Uses **Ollama** (`OLLAMA_HOST`, `OLLAMA_MODEL`) if available.
  - Falls back to **OpenAI** (`OPENAI_API_KEY`, `OPENAI_MODEL`).
  - Uses schema samples (`information_schema.columns`) to guide query generation.
  - Enforces `SELECT`/`WITH` only and appends `LIMIT` automatically.

### Database Layer
- **DSN builder** `_build_dsn()`:
  - Uses `DATABASE_URL` if set, otherwise falls back to `POSTGRES_USER/PASSWORD/HOST/PORT/DB`.
  - Defaults to `postgres/postgres@localhost:5432/pagila`.
- **Read-only session**:
  - Connects with `default_transaction_read_only=on`.
  - Uses `dict_row` for dictionary-style row access.

### SQL Safety & Sanitization
- `sanitize_select_only(sql: str)` guarantees:
  - Only one statement (no multiple queries).
  - Must start with `SELECT` or `WITH`.
  - Blocks forbidden keywords (`INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`, etc.).
  - Blocks dangerous functions (`pg_terminate_backend`, `pg_sleep`, etc.).
  - Adds a `LIMIT` if missing (default: `SQL_MAX_LIMIT` = 500).
- Strips accidental Markdown fences (` ```sql `).
- Safe fallback query (`SELECT 1 AS fallback LIMIT 1`) if sanitizer fails.

### Extensibility
- **Tool registry loader**: `register_builtin_tools()` imports and registers tools from `tools/` (e.g. `echo`, `sysinfo`).
- **Requests library** included for future API tools.
- Built for **easy extension**: add new modules with a `register(app)` function.

### Configuration (Environment Variables)
- **Database**:
  - `DATABASE_URL`
  - or `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
- **SQL Safety**:
  - `SQL_MAX_LIMIT` (default: 500)
- **LLM Backends**:
  - `OLLAMA_HOST`, `OLLAMA_MODEL` (default: `llama3.1:8b`)
  - `OPENAI_API_KEY`, `OPENAI_MODEL` (default: `gpt-4o-mini`)

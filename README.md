# 🕸️ Semantic Graph RAG Middleware

> **A production-grade Retrieval-Augmented Generation middleware that answers natural language questions about company data by dynamically traversing a semantic knowledge graph — no hand-written SQL, no vector store, no fine-tuning.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [The Semantic Knowledge Graph](#the-semantic-knowledge-graph)
- [Pipeline Stages](#pipeline-stages)
- [Answer Synthesis Strategy](#answer-synthesis-strategy)
- [Glass Box Audit Trail](#glass-box-audit-trail)
- [Example Queries](#example-queries)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Extending the System](#extending-the-system)
- [Design Decisions](#design-decisions)

---

## Overview

This project is a **Semantic Graph RAG Middleware** — a system that sits between a user and a MySQL database. Users ask questions in plain English; the middleware figures out what data is needed, queries the database directly, and returns a clean natural language answer.

### What makes it different

Most RAG systems work by embedding documents into a vector store and retrieving semantically similar chunks. This system takes a fundamentally different approach:

| Approach | This System | Typical RAG |
|---|---|---|
| Data storage | MySQL (structured) | Vector store (unstructured) |
| Retrieval | Dynamic SQL via graph traversal | Embedding similarity search |
| SQL generation | Graph-driven (no templates) | Text-to-SQL LLM |
| LLM dependency | Minimal (extraction + synthesis only) | Heavy (retrieval + generation) |
| Latency | < 1s for most queries | 3–30s typical |
| Auditability | Full Glass Box trace | Black box |

The key insight: **structured data doesn't need to be unstructured to be queried with natural language.** A knowledge graph of your database schema can route any question to the right JOIN path automatically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Question (NL)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1 · ENTITY EXTRACTION                                     │
│  Tier 1: Rule-based (regex + keyword matching) — 0ms            │
│  Tier 2: LLM fallback (qwen2.5:1.5b) — ~2s                     │
│  Output: entities=["Employee","Department"], filters={...}       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2 · SEMANTIC GRAPH TRAVERSAL (NetworkX)                   │
│  Finds shortest JOIN path between detected entities             │
│  Employee → works_in → Department → has_projects → Project      │
│  Output: path=["Employee","Department","Project"], join_chain    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3 · DYNAMIC SQL CONSTRUCTION (GraphQueryBuilder)          │
│  Assembles SELECT + FROM + JOINs + WHERE + ORDER BY             │
│  from the graph traversal metadata — zero hardcoded templates   │
│  Output: QueryTemplate with parameterized SQL                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4 · QUERY EXECUTION (MySQL direct)                        │
│  Runs parameterized query with self-healing loop                │
│  Output: rows, execution_time_ms, row_count                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5 · CONTEXT FORMATTING                                    │
│  Converts raw rows → structured text for LLM consumption        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6 · ANSWER SYNTHESIS (3-tier)                             │
│  Tier 0: Python template (0ms)    — most queries                │
│  Tier 1: Fast LLM qwen2.5:1.5b    — comparisons, aggregations  │
│  Tier 2: Raw context fallback      — if LLM fails               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    Natural Language Answer
```

---

## How It Works

### From question to SQL in 3 steps

**1. Entity Extraction**

"Which employees in Engineering are working on projects managed by Sarah?"

→ Entities: `["Employee", "Department", "Project"]`
→ Filters: `{"department_name": "Engineering", "manager_name": "Sarah Connor"}`
→ Question type: `cross_entity`

**2. Graph Traversal**

The system looks up the semantic graph and finds the shortest path connecting all three entities:

```
Employee --[works_in]--> Department --[has_projects]--> Project
```

This tells the query builder exactly which tables to JOIN and in what order.

**3. Dynamic SQL Construction**

```sql
SELECT e.name AS employee_name, e.role AS employee_role, e.salary,
       d.name AS department_name, proj.name AS project_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.id
INNER JOIN projects proj ON d.id = proj.department_id
WHERE d.name LIKE '%Engineering%'
  AND proj.manager_id IN (SELECT id FROM employees WHERE name LIKE '%Sarah%')
ORDER BY e.name ASC
```

No SQL template was written for this question. The query was assembled entirely from graph metadata.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Database** | MySQL 8+ |
| **Graph engine** | NetworkX |
| **LLM (extraction)** | qwen2.5:1.5b via Ollama |
| **LLM (synthesis)** | qwen2.5:1.5b via Ollama |
| **Data validation** | Pydantic v2 |
| **Config** | YAML |
| **Testing** | pytest (119 tests) |
| **Language** | Python 3.11+ |

---

## Project Structure

```
RAG_MIDDLEWARE/
│
├── app.py                          # Streamlit frontend with Glass Box UI
│
├── config/
│   ├── db_config.yaml              # MySQL connection settings
│   ├── graph_schema.yaml           # Knowledge graph definition (nodes + edges)
│   └── intents.yaml                # LLM model config + temperature settings
│
├── database/
│   ├── schema.sql                  # MySQL DDL (6 tables)
│   └── seed.sql                    # Demo data (17 employees, 6 depts, 6 projects...)
│
├── middleware/
│   ├── pipeline.py                 # Orchestrator — single public entry point
│   ├── entity_extractor.py         # Step 1: NL → entities + filters (rule + LLM)
│   ├── semantic_graph.py           # Step 2: NetworkX graph + path-finding
│   ├── graph_query_builder.py      # Step 3: Graph traversal → SQL
│   ├── query_executor.py           # Step 4: MySQL execution + self-healing
│   ├── context_formatter.py        # Step 5: Rows → structured text
│   ├── answer_synthesizer.py       # Step 6: 3-tier answer generation
│   └── models.py                   # Pydantic models (MiddlewareTrace, etc.)
│
└── tests/
    ├── test_semantic_graph.py       # Graph traversal tests (31 tests)
    ├── test_entity_extractor.py     # Extraction tests (29 tests)
    ├── test_graph_query_builder.py  # SQL builder tests (17 tests)
    ├── test_pipeline_e2e.py         # End-to-end pipeline tests (16 tests)
    └── test_pipeline_regression.py  # Regression suite (26 tests)
```

---

## Database Schema

Six tables covering a fictional company's operations:

```sql
employees           -- 17 employees across 5 departments
  id, name, department_id, salary, hire_date, role, email

departments         -- 5 departments
  id, name, budget, location, manager_id

products            -- 12 products
  id, name, category, price, stock_quantity, supplier

orders              -- purchase orders linking employees to products
  id, product_id, employee_id, quantity, order_date, status

projects            -- 6 active/completed projects
  id, name, description, budget, start_date, end_date,
  status (planning|active|completed|on_hold), department_id, manager_id

project_assignments -- many-to-many junction: employees ↔ projects
  id, project_id, employee_id, role_on_project, assigned_date
```

### Seed data highlights

- **17 employees** across Engineering, Sales, Marketing, Operations, HR  
- **6 projects** including active campaigns, platform migrations, and automation initiatives  
- **24 project assignments** linking employees to projects with specific roles  
- **12 products** across Electronics, Furniture, and Office Supply categories  

---

## The Semantic Knowledge Graph

Defined entirely in `config/graph_schema.yaml` — no Python changes needed to add entities or relationships.

### Nodes (5)

| Node | Table | Alias |
|---|---|---|
| Employee | employees | `e` |
| Department | departments | `d` |
| Product | products | `p` |
| Order | orders | `o` |
| Project | projects | `proj` |

### Edges (8)

| From | Relation | To | Join Condition | Notes |
|---|---|---|---|---|
| Employee | `works_in` | Department | `e.department_id = d.id` | All employees in dept |
| Department | `managed_by` | Employee | `d.manager_id = e.id` | One manager only |
| Employee | `assigned_to` | Project | via `project_assignments` | Many-to-many junction |
| Project | `managed_by_employee` | Employee | `proj.manager_id = e.id` | Project manager |
| Project | `owned_by` | Department | `proj.department_id = d.id` | Dept owns project |
| Department | `has_projects` | Project | `d.id = proj.department_id` | Reverse of owned_by |
| Order | `placed_by` | Employee | `o.employee_id = e.id` | Who placed the order |
| Order | `contains` | Product | `o.product_id = p.id` | What was ordered |

### Path-finding algorithm

The `SemanticGraph.find_multi_path()` method uses a **greedy Steiner-path strategy**:

1. First attempts a direct end-to-end path from anchor to final entity — if it passes through all required nodes, done.
2. If not, expands greedily from the current path tip to each remaining required entity, skipping already-visited nodes.

This avoids the duplicate-node bug of naive pairwise chaining (e.g., `Employee→Department→Employee→Project`) that would produce broken SQL with aliasing conflicts.

---

## Pipeline Stages

### Stage 1 · Entity Extraction (`entity_extractor.py`)

**Two-tier extraction:**

**Tier 1 — Rule-based (0ms)**  
Deterministic string matching against known vocabularies:
- 17 known employee names (exact match, longest-first)
- 5 department names
- 6 project names  
- Product names and categories
- Order status synonyms (`"in transit"` → `"shipped"`)
- `"managed by X"` pattern → `manager_name` filter (distinct from employee subject)
- Generic subject phrases (`"employees"`, `"working on"`, `"assigned to"`) → entity detection
- Project status words (`"active"`, `"completed"`, `"planned"`)

**Tier 2 — LLM fallback (qwen2.5:1.5b)**  
Used only when rules detect no entities. Structured JSON prompt with few-shot examples.

**Output — `EntityExtractionResult`:**
```python
EntityExtractionResult(
    entities=["Employee", "Department"],
    filters={"department_name": "Engineering"},
    projections=["salary"],
    question_type="list",        # lookup | list | comparison | aggregation | cross_entity
    extraction_method="rules",   # rules | llm
    latency_ms=0.4,
)
```

### Stage 2 · Semantic Graph Traversal (`semantic_graph.py` + `pipeline.py`)

Finds the JOIN path connecting all detected entities. Applies entity ordering rules before path-finding:

- `Employee + Department` → always anchor Employee first (uses `works_in` not `managed_by`)
- `Employee + Department + Project` → explicit order `[Employee, Department, Project]`
- `Employee + Project` → direct `[Employee, Project]` via `project_assignments`

**Output — `GraphTraversal`:**
```python
GraphTraversal(
    path_taken=["Employee", "Department", "Project"],
    join_count=2,
    tables_involved=["employees", "departments", "projects"],
    traversal_method="multi_hop",   # single_node | two_hop | multi_hop
    traversal_time_ms=0.8,
    path_description="Employee→Department→Project",
)
```

### Stage 3 · Dynamic SQL Construction (`graph_query_builder.py`)

Assembles SQL from graph metadata with four sub-builders:

**`_build_select()`** — collects selectable columns from each entity node in the path, plus junction table extras (e.g. `pa.role_on_project AS assignment_role`).

**`_build_from()`** — uses the anchor entity's table and alias.

**`_build_joins()`** — generates `INNER JOIN` / `LEFT JOIN` clauses. Handles both simple FK joins and many-to-many junction table joins (two JOIN clauses for one edge).

**`_build_where()`** — maps filter keys to SQL columns via `_DIRECT_FILTER_MAP` (unambiguous per-alias mapping). Three match types:
- `like` → `LIKE %(param)s` (executor adds `%` wildcards)
- `exact` → `= %(param)s` (status, category ENUMs)
- `subquery` → `proj.manager_id IN (SELECT id FROM employees WHERE name LIKE ...)` (manager_name filter)

**`_build_order_limit()`** — question-type-aware ordering:
- `comparison` → `ORDER BY e.salary DESC`
- `list` → `ORDER BY e.name ASC`
- `aggregation` → `GROUP BY ... ORDER BY count DESC`

### Stage 4 · Query Execution (`query_executor.py`)

Runs the parameterized query against MySQL with a **self-healing loop**:

1. Execute query with extracted parameters
2. If 0 rows returned → LLM diagnoses what went wrong → retries with adjusted query
3. Returns `DBResult` with rows, timing, row count, and healing metadata

Exact-match guard: `_EXACT_MATCH_PARAMS = {"status", "category", "manager_name"}` prevents wildcard wrapping on ENUM/subquery fields.

### Stage 5 · Context Formatting (`context_formatter.py`)

Converts raw MySQL rows into structured text for LLM consumption. Single rows become key-value pairs; multi-row results become numbered records. Currency fields (`salary`, `budget`, `price`) are formatted with `$` and commas.

### Stage 6 · Answer Synthesis (`answer_synthesizer.py`)

**3-tier strategy:**

**Tier 0 — Python template bypass (0ms)**  
Handles the majority of queries with zero LLM calls. Detects row shape from aliased column names (`employee_name`, `employee_role`, `salary`, `assignment_role`, `project_name`, etc.) and formats directly:

```
Found 4 employees:

1. **Dwight Schrute** (Senior Sales Rep) — $67,000.00
2. **Jim Halpert** (Sales Representative) — $65,000.00
3. **Michael Scott** (Sales Manager) — $72,000.00
4. **Pam Beesly** (Sales Coordinator) — $58,000.00
```

**Tier 1 — Fast LLM (qwen2.5:1.5b, ~2-5s)**  
Used when the Tier 0 bypass can't handle the query — primarily comparisons (`"who earns the most"`), aggregations (`"how many employees"`), and yes/no questions. Tight prompt with `num_ctx: 2048` and `max_tokens: 200` to minimize generation time.

**Tier 2 — Fallback**  
Returns raw formatted context if the LLM call fails.

---

## Answer Synthesis Strategy

### Speed benchmarks

| Query type | Tier used | Typical latency |
|---|---|---|
| Employee list / project members | Tier 0 (Python) | **< 100ms** |
| Salary / stock / budget lookup | Tier 0 (Python) | **< 100ms** |
| "Highest paid in Marketing?" | Tier 1 (LLM) | **2–5s** |
| "How many employees per dept?" | Tier 1 (LLM) | **3–6s** |
| Unknown question shape | Tier 1 (LLM) | **2–8s** |

The database query itself is consistently **< 5ms** regardless of complexity.

---

## Glass Box Audit Trail

Every response includes a collapsible **audit trail** showing exactly how the answer was derived:

```
⓪ Semantic Graph Traversal
   [Employee] → [Department] → [Project]    ⬤ multi-hop
   Tables: employees, departments, projects | Joins: 2 | Graph time: 0.8ms

① Entity Extraction
   Entities: [Employee] [Department] [Project]
   Method: ⬤ rules | Type: cross_entity | Latency: 0ms

② Filters Applied
   {"department_name": "Engineering", "manager_name": "Sarah Connor"}

③ Database Execution
   ✅ 3 row(s) returned | DB time: 3.2ms

④ SQL Executed Against MySQL
   SELECT e.name AS employee_name, ...
   FROM employees e
   INNER JOIN departments d ON e.department_id = d.id
   INNER JOIN projects proj ON d.id = proj.department_id
   WHERE d.name LIKE %(department_name)s
     AND proj.manager_id IN (SELECT id FROM employees WHERE name LIKE %(manager_name)s)
   ORDER BY e.name ASC

⑤ Raw Database Response
   [interactive dataframe]

⑥ Pipeline Timing Breakdown
   Extraction: 0ms | Graph: 0.8ms | MySQL: 3.2ms | Synthesis: 0ms | TOTAL: 67ms
```

---

## Example Queries

### Single Entity
```
What is Sarah Connor's salary?
What role does Alan Turing have?
How many Laptop Pro 15 units are in stock?
What is the most expensive product?
Show me all pending orders
Show me all Electronics products
```

### Two-Hop Join (Employee ↔ Department)
```
List all employees in Engineering
Who is the highest paid in Marketing?
Who earns the most in Operations?
What is the Engineering department budget?
Show me all employees in Sales with their salaries
```

### Two-Hop Join (Employee ↔ Project)
```
Which employees are assigned to the API Gateway Rebuild project?
Who is working on the Platform Migration?
Who is working on the Q1 Marketing Campaign?
```

### Multi-Hop (3+ tables)
```
Which employees in Engineering are working on projects managed by Sarah?
Who are the employees on projects managed by Don Draper?
Which employees in Operations are assigned to active projects?
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- MySQL 8.0+
- [Ollama](https://ollama.ai) with the following models pulled:

```bash
ollama pull qwen2.5:1.5b
```

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-middleware.git
cd rag-middleware
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
mysql-connector-python==8.3.0
pydantic==2.7.1
pyyaml==6.0.1
spacy==3.7.4
requests==2.31.0
streamlit==1.35.0
pytest==8.2.0
networkx>=3.3
```

### 3. Set up MySQL

Create the database and load the schema + seed data:

```bash
mysql -u root -p -e "CREATE DATABASE middleware_poc;"
mysql -u root -p middleware_poc < database/schema.sql
mysql -u root -p middleware_poc < database/seed.sql
```

### 4. Configure the database connection

Edit `config/db_config.yaml`:

```yaml
database:
  host: "localhost"
  port: 3306
  user: "root"           # your MySQL username
  password: "yourpass"   # your MySQL password
  database: "middleware_poc"
```

### 5. Start Ollama

```bash
ollama serve
```

Verify the model is available:

```bash
ollama list   # should show qwen2.5:1.5b
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The sidebar shows live status for MySQL, Ollama models, and the semantic graph node/edge count.

---

## Running Tests

```bash
# Full test suite (119 tests)
pytest tests/ -v

# Individual suites
pytest tests/test_semantic_graph.py -v       # Graph traversal (31 tests)
pytest tests/test_entity_extractor.py -v     # Entity extraction (29 tests)
pytest tests/test_graph_query_builder.py -v  # SQL builder (17 tests)
pytest tests/test_pipeline_e2e.py -v -s      # End-to-end (16 tests, needs MySQL)
pytest tests/test_pipeline_regression.py -v  # Regression (26 tests, needs MySQL)
```

> **Note:** `test_pipeline_e2e.py` and `test_pipeline_regression.py` require a live MySQL connection and Ollama instance.

---

## Configuration

### `config/graph_schema.yaml`

The single source of truth for the knowledge graph. Add new entities or relationships here — no Python changes required.

```yaml
nodes:
  NewEntity:
    table: new_table
    alias: nt
    selectable_columns:
      - column: name
        alias: entity_name
        label: "Entity Name"
    filterable_columns:
      name:
        match_type: like
        sql_column: nt.name
    primary_key: id
    primary_key_sql: nt.id

edges:
  - from_node: Employee
    to_node: NewEntity
    relation: has_new_entity
    join_type: INNER
    join_condition: "e.new_entity_id = nt.id"
    notes: "Description of this relationship"
```

### `config/intents.yaml` — Model settings

```yaml
models:
  fast_model: "qwen2.5:1.5b"
  synthesis_model: "qwen2.5:1.5b"
  ollama_base_url: "http://localhost:11434"
  intent_temperature: 0.0
  synthesis_temperature: 0.1
  max_tokens_intent: 50
  max_tokens_synthesis: 200
```

To use a more capable synthesis model at the cost of speed, change `synthesis_model` to `llama3.2:3b` or any Ollama-compatible model.

---

## Extending the System

### Adding a new table

1. Add the DDL to `database/schema.sql`
2. Add seed data to `database/seed.sql`
3. Add a node + edges to `config/graph_schema.yaml`
4. Add known vocabulary (if applicable) to `entity_extractor.py`'s `_KNOWN_*` lists
5. Add column handling to `answer_synthesizer.py`'s `_try_template_answer()` if needed

No changes to `pipeline.py`, `semantic_graph.py`, or `graph_query_builder.py` are required.

### Adding a new question type

If users ask a new class of questions (e.g., date-range queries), extend:

1. `entity_extractor.py` — detect the new pattern and produce a filter key
2. `graph_query_builder.py` — add the filter key to `_DIRECT_FILTER_MAP`
3. `answer_synthesizer.py` — add a Tier 0 formatting branch if the answer shape is predictable

---

## Design Decisions

**Why not Text-to-SQL?**  
Text-to-SQL LLMs are non-deterministic and can generate syntactically valid but semantically wrong queries. The graph approach is fully deterministic — the JOIN path is always computed from the schema, never hallucinated.

**Why not a vector store?**  
Structured relational data already has perfect retrieval via SQL. Adding a vector store would introduce approximation, chunking complexity, and embedding costs with no benefit over a direct query.

**Why NetworkX for the graph?**  
It's the standard Python graph library, has built-in shortest-path algorithms, and is fast enough to run in-process with zero network overhead. The graph is built once at startup and reused across all requests.

**Why keep the LLM at all?**  
Two places: entity extraction fallback (when rule-based matching fails on unusual phrasing), and answer synthesis for questions requiring ranking or comparison reasoning that can't be pre-formatted. The graph handles the "what to query" problem; the LLM handles the "how to say the answer" problem.

**Why `qwen2.5:1.5b` instead of a larger model?**  
It's already loaded for extraction, so synthesis reuses it from memory with no additional load time. For the narrow task of formatting a structured answer, a 1.5B model is sufficient. The result for most queries is a Python-formatted string anyway (Tier 0), making the model choice irrelevant for the majority of traffic.

**Why the Glass Box?**  
Observability is critical for a system making autonomous database decisions. Every step — entities detected, graph path chosen, SQL executed, rows returned, time taken — is surfaced in the UI so users and developers can immediately understand and debug any answer.

---

## License

MIT

---

*Built with Python · NetworkX · MySQL · Ollama · Streamlit*

# AFL Agent - Domain-Locked RAG + ML Chat Agent 

A chat agent that only discusses AFL. It answers rules/history questions from a Wikipedia-based
knowledge base (RAG, with source citations) and predicts match winners and top players using the
trained Day 2 models (LangGraph tool calls), never from the LLM's own guesses.

## Architecture
`guard (is it AFL?) -> router (rag | tool | both) -> retrieve and/or call ML tools -> grounded answer`

| File | Purpose |
|---|---|
| `Day3.py` | Chunking, embeddings (Chroma), LLM setup (`python Day3.py ingest`) |
| `ml_tools.py` | `predict_match_winner`, `predict_top_player` tools: JSON schemas, validation, clean errors |
| `day2_models.py` | Loads `models/*.joblib` and reproduces the Day 2 prediction logic |
| `Day4.py` | LangGraph agent, multi-turn memory, `safe_ask()` entry point |
| `guardrails.py` | Input/output limits, secret redaction, rate limiters |
| `test_suite.py` | 20 conversations + guardrail unit tests -> `test_results.csv/.md` |
| `app.py` | Streamlit frontend |

## Guardrails
| Guardrail | Where | Behaviour |
|---|---|---|
| Domain lock | `guard` node (LLM classifier with conversation history) | Non-AFL requests, jailbreaks, prompt-extraction and mid-conversation topic drift get a fixed redirect message |
| Prompt hardening | `SECURITY` block in the answer and tool prompts | No role changes, no prompt disclosure, user text/retrieved context/tool output treated as data, no invented numbers or sources |
| Grounding | `generate` node | Predictions only from tool results; rules/history only from retrieved chunks with `[n]` citations; says "I don't have that" otherwise |
| Input limit | `sanitize_input` | 500 characters max; control characters stripped; empty input rejected |
| Output limit | `clip_output` | 1200 characters max (cut at a sentence end); strings that look like API keys are redacted |
| Request rate limit | `REQUEST_LIMITER` | 8 messages per session per minute |
| Tool-call rate limit | `TOOL_LIMITER` + per-turn cap | 10 tool executions per session per 10 minutes; max 2 tool calls per message |
| Tool validation | `ml_tools.py` | Unknown team (fuzzy/alias matching), same team twice, bad date format, date before 2000 or more than 120 days ahead -> friendly message, never a guessed prediction |
| Failure containment | `safe_ask` | Any unexpected exception becomes a generic apology; stack traces are logged server-side only |

## Tests
`python test_suite.py` runs 20 conversations (5 RAG-only, 3 tool-only, 4 tool-error, 2 mixed,
2 multi-turn, 4 adversarial) with automatic pass/fail checks, plus 6 offline guardrail unit tests.
Results: `test_results.csv` and `test_results.md`.

**Latest result:** _fill in after running (e.g. 18/20 agent, 6/6 guardrails) and list any failures with the reason._

## Run locally
```
pip install -r requirements.txt
set GROQ_API_KEY=your-key            (PowerShell: $env:GROQ_API_KEY = "your-key")
python Day3.py ingest
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push the repo to GitHub, including `models/*.joblib` and the `Data/` folder (the vector store builds itself on first start).
2. share.streamlit.io -> New app -> pick the repo, main file `AFL-AGENT/app.py`.
3. App settings -> Secrets: `GROQ_API_KEY = "your-key"`.
4. First start takes a few minutes (embedding model download + ingest).

## Known limitations
- Elo, form and player stats are frozen at the end of the training data (`ratings_as_of` is shown in every prediction); the requested match date is validated but not used as a feature.
- Rest days are fixed at 0 and matches are not treated as finals, as in the Day 2 notebook.
- Season pages from Wikipedia lose their tables, so the knowledge base is best for rules, clubs and history rather than match results.
- Rate limits are in-memory and per server process; use Redis or similar for multi-instance deployments.
- Wikipedia text is CC BY-SA 4.0.
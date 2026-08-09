"""Uma Calibrate, as a Render Workflow.

Deliberately NOT in the request-time critical path — Uma's cross-encoder is
designed to stay loaded in a single process (see uma.core.model), and every
``uma judge`` / MCP ``uma_filter`` call already runs synchronously in a few
hundred milliseconds. Putting a Render Workflow in front of that would add
orchestration overhead to solve a problem that doesn't exist.

What Render Workflows is actually good for here: running Uma Calibrate's
threshold sweep — several independent, longer-running benchmark passes
(each one is N real LLM calls) — as parallel task runs, with retries, and
returning one aggregated JSON result. See workflow.py.
"""

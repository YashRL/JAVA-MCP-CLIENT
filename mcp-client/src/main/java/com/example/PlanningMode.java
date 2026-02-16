package com.example;

/**
 * Controls how much upfront planning the agent does before executing.
 *
 * ── SHORT ─────────────────────────────────────────────────────────────────
 *   No separate planner LLM call.
 *   The executor LLM uses its own built-in reasoning to decide what to do.
 *   Guidance is given via a rich system prompt with guardrails:
 *     - MAX_STEPS hard cap so it can't loop forever
 *     - Anti-repeat rule (tracks tool calls already made)
 *     - Complexity self-assessment instruction
 *   Best for: speed, low latency, simple to moderate questions.
 *   Cost: 1 LLM call per step (no planner overhead).
 *
 * ── MID ───────────────────────────────────────────────────────────────────
 *   Lightweight planner: one cheap LLM call that outputs just:
 *     - complexity (trivial/simple/moderate/complex)
 *     - step_budget (integer)
 *     - intent (one sentence)
 *     - direct_answer (for trivial queries only)
 *   No step-by-step plan list. The executor then runs freely within budget.
 *   Best for: balance of speed and control.
 *   Cost: 1 cheap planner call + N executor calls.
 *
 * ── LONG ──────────────────────────────────────────────────────────────────
 *   Full structured planner (the original): outputs complexity, budget,
 *   intent, AND an ordered list of planned steps. The executor is shown
 *   the plan at each turn and a separate reflection call checks progress.
 *   Best for: deep research, multi-step pipelines, maximum quality.
 *   Cost: 1 planner call + N executor calls + N reflection calls.
 */
public enum PlanningMode {
    SHORT,
    MID,
    LONG
}
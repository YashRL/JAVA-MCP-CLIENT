package com.example.agentic_setup.mcp;

/**
 * Controls how much upfront planning happens before execution.
 *
 * SHORT — no planner LLM call; executor self-directs within a step cap.
 * MID   — one cheap LLM call produces complexity + budget + intent.
 * LONG  — full LLM call produces complexity + budget + intent + ordered steps + per-step reflection.
 */
public enum PlanningMode { SHORT, MID, LONG }
package com.example.agentic_setup.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.List;

/**
 * Produces an execution Plan before the agent loop runs.
 *
 * SHORT — regex trivial-check only; no LLM call.
 * MID   — one LLM call → complexity + budget + intent.
 * LONG  — one LLM call → complexity + budget + intent + ordered steps.
 *
 * All LLM calls go through the shared LLMProvider (llmProvider.complete()).
 */
public class Planner {

    public static final int MAX_STEPS = 10;

    public record Plan(PlanningMode mode, String complexity, int budget,
                       String intent, List<String> steps, String directAnswer) {
        public boolean isTrivial() { return budget == 0 && directAnswer != null; }
    }

    private final PlanningMode mode;
    private final LLMProvider  llm;
    private final ObjectMapper mapper = new ObjectMapper();

    public Planner(PlanningMode mode, LLMProvider llm) {
        this.mode = mode;
        this.llm  = llm;
    }

    public PlanningMode getMode() { return mode; }

    public Plan plan(String question, String capabilities) throws Exception {
        return switch (mode) {
            case SHORT -> planShort(question);
            case MID   -> planMid(question, capabilities);
            case LONG  -> planLong(question, capabilities);
        };
    }

    // ── SHORT ─────────────────────────────────────────────────────────────

    private Plan planShort(String question) {
        var trivial = question.trim().toLowerCase()
                .matches("(hi+|hello+|hey+|sup|howdy|greetings|thanks?|thank you|bye|goodbye"
                        + "|ok|okay|yes|no|sure|great|nice|cool|wow|lol|haha)[!?. ]*");
        if (trivial)
            return new Plan(PlanningMode.SHORT, "trivial", 0, "Greeting", List.of(),
                    "Hello! How can I help you today?");

        System.out.println("[Planner/SHORT] Executor self-directs within " + MAX_STEPS + " steps.");
        return new Plan(PlanningMode.SHORT, "unknown", MAX_STEPS, question, List.of(), null);
    }

    // ── MID ───────────────────────────────────────────────────────────────

    private Plan planMid(String question, String capabilities) throws Exception {
        var sys = """
                You are a query classifier. Output ONLY valid JSON — no markdown, no explanation.
                Format:
                {
                  "complexity": "<trivial|simple|moderate|complex>",
                  "step_budget": <0–%d>,
                  "intent": "<one sentence>",
                  "direct_answer": <null or "string for trivial only">
                }
                Rules: trivial→budget=0; simple→1-2; moderate→3-5; complex→6-%d.
                Available capabilities:
                %s
                """.formatted(MAX_STEPS, MAX_STEPS, capabilities);

        var raw = llm.complete(sys, "Question: " + question);
        try {
            var j = mapper.readTree(strip(raw));
            var answer = j.path("direct_answer").isNull() ? null : j.path("direct_answer").asText(null);
            return new Plan(mode, j.path("complexity").asText("moderate"),
                    clamp(j.path("step_budget").asInt(4)), j.path("intent").asText(question),
                    List.of(), answer);
        } catch (Exception e) {
            System.err.println("[Planner/MID] Parse error: " + e.getMessage());
            return new Plan(mode, "moderate", 4, question, List.of(), null);
        }
    }

    // ── LONG ──────────────────────────────────────────────────────────────

    private Plan planLong(String question, String capabilities) throws Exception {
        var sys = """
                You are a planning agent. Output ONLY valid JSON — no markdown, no extra text.
                Format:
                {
                  "complexity": "<trivial|simple|moderate|complex>",
                  "step_budget": <0–%d>,
                  "intent": "<one sentence>",
                  "planned_steps": ["<step 1: tool + action>", ...],
                  "direct_answer": <null or "string for trivial only">
                }
                Rules: trivial→0; simple→1-2; moderate→3-5; complex→6-%d.
                planned_steps must have EXACTLY step_budget entries.
                Each step must name a specific tool from: %s
                """.formatted(MAX_STEPS, MAX_STEPS, capabilities);

        var raw = llm.complete(sys, "User question: " + question);
        try {
            var j     = mapper.readTree(strip(raw));
            var steps = new ArrayList<String>();
            j.path("planned_steps").forEach(s -> steps.add(s.asText()));
            var answer = j.path("direct_answer").isNull() ? null : j.path("direct_answer").asText(null);
            return new Plan(mode, j.path("complexity").asText("moderate"),
                    clamp(j.path("step_budget").asInt(3)), j.path("intent").asText(question),
                    steps, answer);
        } catch (Exception e) {
            System.err.println("[Planner/LONG] Parse error: " + e.getMessage() + " | raw: " + raw);
            return new Plan(mode, "moderate", 4, question,
                    List.of("Search topic", "Search context", "Gather data", "Synthesise"), null);
        }
    }

    // ── Util ──────────────────────────────────────────────────────────────

    private static String strip(String raw) {
        var s = raw.strip();
        return s.startsWith("```") ? s.replaceAll("(?s)```json?\\s*", "").replace("```", "").strip() : s;
    }

    private static int clamp(int v) { return Math.max(0, Math.min(v, MAX_STEPS)); }
}
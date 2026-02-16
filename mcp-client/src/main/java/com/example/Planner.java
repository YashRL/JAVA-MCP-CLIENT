package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

/**
 * Planner -- produces an execution Plan based on the chosen PlanningMode.
 *
 * SHORT: No LLM call. Trivial detection via regex heuristic. Executor self-directs.
 * MID  : One cheap LLM call -> complexity + budget + intent only.
 * LONG : Full structured LLM call -> complexity + budget + intent + ordered steps.
 */
public class Planner {

    private static final String RESPONSES_URL = "https://api.openai.com/v1/responses";

    /** Hard ceiling on tool-call turns regardless of mode. */
    public static final int MAX_STEPS = 10;

    private final PlanningMode mode;
    private final String       model;
    private final String       apiKey;
    private final HttpClient   httpClient;
    private final ObjectMapper mapper = new ObjectMapper();

    public Planner(PlanningMode mode, String model, String apiKey, HttpClient httpClient) {
        this.mode       = mode;
        this.model      = model;
        this.apiKey     = apiKey;
        this.httpClient = httpClient;
    }

    // ── Plan record ───────────────────────────────────────────────────────

    public record Plan(
        PlanningMode mode,
        String       complexity,
        int          stepBudget,
        String       intent,
        List<String> plannedSteps,
        String       directAnswer
    ) {
        public boolean isTrivial() { return stepBudget == 0 && directAnswer != null; }

        @Override public String toString() {
            return "Plan[mode=" + mode + " complexity=" + complexity
                + " budget=" + stepBudget + " steps=" + plannedSteps.size() + "]";
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    public PlanningMode getMode() { return mode; }

    public Plan plan(String userQuestion, String capabilitySummary) throws Exception {
        switch (mode) {
            case SHORT: return planShort(userQuestion);
            case MID:   return planMid(userQuestion, capabilitySummary);
            case LONG:  return planLong(userQuestion, capabilitySummary);
            default:    return planMid(userQuestion, capabilitySummary);
        }
    }

    // ── SHORT ─────────────────────────────────────────────────────────────

    private Plan planShort(String userQuestion) {
        String q = userQuestion.trim().toLowerCase();
        boolean trivial = q.matches(
            "(hi+|hello+|hey+|sup|howdy|greetings|thanks?|thank you|bye|goodbye"
            + "|ok|okay|yes|no|sure|great|nice|cool|wow|lol|haha)[!?. ]*");
        if (trivial) {
            return new Plan(PlanningMode.SHORT, "trivial", 0,
                "Greeting or chitchat", List.of(), "Hello! How can I help you today?");
        }
        System.out.println("[Planner/SHORT] No planner call -- executor self-directs within "
            + MAX_STEPS + " steps.");
        return new Plan(PlanningMode.SHORT, "unknown", MAX_STEPS,
            userQuestion, List.of(), null);
    }

    // ── MID ───────────────────────────────────────────────────────────────

    private Plan planMid(String userQuestion, String capabilitySummary) throws Exception {
        String systemPrompt =
            "You are a query classifier. Output ONLY a JSON object -- no markdown, no explanation.\n\n"
            + "Available capabilities:\n" + capabilitySummary + "\n\n"
            + "Output format:\n"
            + "{\n"
            + "  \"complexity\": \"<trivial|simple|moderate|complex>\",\n"
            + "  \"step_budget\": <0 to " + MAX_STEPS + ">,\n"
            + "  \"intent\": \"<one sentence>\",\n"
            + "  \"direct_answer\": <null or \"string for trivial only\">\n"
            + "}\n\n"
            + "Rules:\n"
            + "- trivial  = greetings, chitchat, capability questions -> budget=0, set direct_answer\n"
            + "- simple   = single lookup -> budget 1-2\n"
            + "- moderate = multi-aspect -> budget 3-5\n"
            + "- complex  = deep research -> budget 6-" + MAX_STEPS + "\n\n"
            + "Output ONLY valid JSON. Nothing else.";

        String raw = callLLM("gpt-4.1-mini", systemPrompt, "Question: " + userQuestion);
        return parseMidPlan(raw, userQuestion);
    }

    private Plan parseMidPlan(String raw, String userQuestion) {
        try {
            String cleaned = stripFences(raw);
            JsonNode json  = mapper.readTree(cleaned);
            String complexity   = json.path("complexity").asText("moderate");
            int    budget       = clamp(json.path("step_budget").asInt(4));
            String intent       = json.path("intent").asText(userQuestion);
            String directAnswer = json.path("direct_answer").isNull()
                                  ? null : json.path("direct_answer").asText(null);
            if (budget == 0 && directAnswer == null)
                directAnswer = "I'm here to help! What would you like to know?";
            return new Plan(PlanningMode.MID, complexity, budget, intent, List.of(), directAnswer);
        } catch (Exception e) {
            System.err.println("[Planner/MID] Parse error: " + e.getMessage());
            return new Plan(PlanningMode.MID, "moderate", 4, userQuestion, List.of(), null);
        }
    }

    // ── LONG ──────────────────────────────────────────────────────────────

    private Plan planLong(String userQuestion, String capabilitySummary) throws Exception {
        String systemPrompt =
            "You are a planning agent. Output ONLY a JSON object -- no markdown, no extra text.\n\n"
            + "Available capabilities:\n" + capabilitySummary + "\n\n"
            + "Output format:\n"
            + "{\n"
            + "  \"complexity\": \"<trivial|simple|moderate|complex>\",\n"
            + "  \"step_budget\": <0 to " + MAX_STEPS + ">,\n"
            + "  \"intent\": \"<one sentence: what does the user want?>\",\n"
            + "  \"planned_steps\": [\"<step 1: tool + action>\", \"<step 2>\", ...],\n"
            + "  \"direct_answer\": <null or \"string for trivial only\">\n"
            + "}\n\n"
            + "Complexity -> budget rules (be HONEST, not generous):\n"
            + "- trivial  -> 0   (greetings, chitchat, capability questions)\n"
            + "- simple   -> 1-2 (single search or lookup)\n"
            + "- moderate -> 3-5 (multi-aspect research)\n"
            + "- complex  -> 6-" + MAX_STEPS + " (deep research, multi-step pipelines)\n\n"
            + "CRITICAL:\n"
            + "1. planned_steps must have EXACTLY step_budget entries (0 for trivial).\n"
            + "2. Each step must name a specific tool from the capability list.\n"
            + "3. direct_answer must be JSON null for non-trivial queries.\n"
            + "4. Do NOT over-allocate -- only use steps genuinely needed.\n"
            + "5. Output ONLY valid JSON. Nothing else.";

        String raw = callLLM(model, systemPrompt, "User question: " + userQuestion);
        return parseLongPlan(raw, userQuestion);
    }

    private Plan parseLongPlan(String raw, String userQuestion) {
        try {
            String cleaned = stripFences(raw);
            JsonNode json  = mapper.readTree(cleaned);
            String       complexity   = json.path("complexity").asText("moderate");
            int          budget       = clamp(json.path("step_budget").asInt(3));
            String       intent       = json.path("intent").asText(userQuestion);
            String       directAnswer = json.path("direct_answer").isNull()
                                        ? null : json.path("direct_answer").asText(null);
            List<String> steps        = new ArrayList<>();
            for (JsonNode s : json.path("planned_steps")) steps.add(s.asText());
            if (budget == 0 && directAnswer == null)
                directAnswer = "I'm here to help! What would you like to know?";
            return new Plan(PlanningMode.LONG, complexity, budget, intent, steps, directAnswer);
        } catch (Exception e) {
            System.err.println("[Planner/LONG] Parse error: " + e.getMessage() + " raw: " + raw);
            return new Plan(PlanningMode.LONG, "moderate", 4, userQuestion,
                List.of("Search for information", "Search for context",
                        "Gather data points", "Synthesise findings"), null);
        }
    }

    // ── HTTP helper ───────────────────────────────────────────────────────

    private String callLLM(String llmModel, String systemPrompt, String userMessage)
            throws Exception {
        ObjectNode body = mapper.createObjectNode();
        body.put("model", llmModel);
        var arr = mapper.createArrayNode();
        arr.add(mapper.createObjectNode().put("role", "developer").put("content", systemPrompt));
        arr.add(mapper.createObjectNode().put("role", "user").put("content", userMessage));
        body.set("input", arr);

        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(RESPONSES_URL))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                .build();

        HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
        JsonNode root = mapper.readTree(resp.body());

        if (root.has("error") && !root.get("error").isNull())
            throw new RuntimeException("Planner API error: " + root.get("error").toPrettyString());

        return root.path("output").get(0).path("content").get(0).path("text").asText();
    }

    // ── Utils ─────────────────────────────────────────────────────────────

    private static String stripFences(String raw) {
        String s = raw.strip();
        if (s.startsWith("```"))
            s = s.replaceAll("(?s)```json?\\s*", "").replaceAll("```", "").strip();
        return s;
    }

    private static int clamp(int v) {
        return Math.max(0, Math.min(v, MAX_STEPS));
    }
}
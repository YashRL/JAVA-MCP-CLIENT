package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.*;

/**
 * AgentRuntime -- MCP agent with three planning modes and debug output.
 *
 * Planning modes:
 *   SHORT: No planner LLM call. Executor self-directs with guardrails.
 *          Anti-repeat guard prevents infinite loops.
 *   MID  : Cheap planner gives budget+intent. Executor runs freely within budget.
 *   LONG : Full planner with step list + per-step reflection.
 *
 * Debug flags (set in Main.java):
 *   DEBUG_MCP_RAW    -- print raw data from MCP server calls
 *   DEBUG_LLM_PROMPT -- print system prompt sent to LLM
 */
public class AgentRuntime {

    private static final String RESPONSES_URL     = "https://api.openai.com/v1/responses";
    private static final String GET_PROMPT_PREFIX = "get_prompt__";

    // Toggle these in Main.java before constructing AgentRuntime
    public static boolean DEBUG_MCP_RAW    = true;
    public static boolean DEBUG_LLM_PROMPT = true;

    private final LLMProvider  llmProvider;
    private final Planner      planner;
    private final ObjectMapper mapper = new ObjectMapper();

    private final List<MCPServerInfo>              discoveredServers = new ArrayList<>();
    private final Map<String, MCPClient>           toolRegistry      = new LinkedHashMap<>();
    private final Map<String, MCPClient>           resourceRegistry  = new LinkedHashMap<>();
    private final Map<String, MCPClient>           promptRegistry    = new LinkedHashMap<>();
    private final Map<String, JsonNode>            promptMeta        = new LinkedHashMap<>();
    private final List<LLMProvider.ToolDefinition> toolDefinitions   = new ArrayList<>();

    private String capabilitySummary;
    private String executorSystemPrompt;
    private String apiKey;

    // ── Constructor ────────────────────────────────────────────────────────

    public AgentRuntime(List<MCPClient> clients,
                        LLMProvider llmProvider,
                        Planner planner) throws Exception {
        this.llmProvider = llmProvider;
        this.planner     = planner;
        discoverAll(clients);
        capabilitySummary    = buildCapabilitySummary();
        executorSystemPrompt = buildExecutorSystemPrompt();
        apiKey               = extractApiKey();
    }

    // ── Main entry point ───────────────────────────────────────────────────

    public String run(String userQuestion) throws Exception {
        System.out.println("\n[Planner] Mode: " + planner.getMode());

        if (DEBUG_LLM_PROMPT) {
            System.out.println("\n+------ SYSTEM PROMPT SENT TO LLM ------+");
            System.out.println(executorSystemPrompt);
            System.out.println("+----------------------------------------+\n");
        }

        Planner.Plan plan = planner.plan(userQuestion, capabilitySummary);
        System.out.println("[Planner] " + plan);
        System.out.println("[Planner] Intent    : " + plan.intent());
        System.out.println("[Planner] Complexity: " + plan.complexity().toUpperCase());
        System.out.println("[Planner] Step budget: " + plan.stepBudget() + " / " + Planner.MAX_STEPS);

        if (!plan.plannedSteps().isEmpty()) {
            System.out.println("[Planner] Steps:");
            for (int i = 0; i < plan.plannedSteps().size(); i++)
                System.out.println("[Planner]   " + (i + 1) + ". " + plan.plannedSteps().get(i));
        }

        if (plan.isTrivial()) {
            System.out.println("[Agent] Trivial query -- answering directly.");
            return plan.directAnswer();
        }

        switch (planner.getMode()) {
            case SHORT: return executeShort(userQuestion, plan);
            case MID:   return executeMid(userQuestion, plan);
            case LONG:  return executeLong(userQuestion, plan);
            default:    return executeMid(userQuestion, plan);
        }
    }

    // ── SHORT execution ────────────────────────────────────────────────────

    private String executeShort(String userQuestion, Planner.Plan plan) throws Exception {
        Set<String>               usedFingerprints = new HashSet<>();
        List<ReasoningStep>       trace            = new ArrayList<>();
        List<String>              history          = new ArrayList<>();
        List<LLMProvider.Message> messages         = new ArrayList<>();
        messages.add(new LLMProvider.Message("user", userQuestion));

        String sysPrompt = buildShortExecutorPrompt(Planner.MAX_STEPS);
        if (DEBUG_LLM_PROMPT) printPromptBlock("SHORT EXECUTOR PROMPT", sysPrompt);

        LLMProvider.LLMResponse response = llmProvider.chat(sysPrompt, messages, toolDefinitions);
        int stepsUsed = 0;

        while (!response.toolCalls().isEmpty() && stepsUsed < Planner.MAX_STEPS) {
            stepsUsed++;
            int remaining = Planner.MAX_STEPS - stepsUsed;
            System.out.println("\n[Agent/SHORT] -- Step " + stepsUsed + " / " + Planner.MAX_STEPS + " --");

            List<String> actionsTaken = new ArrayList<>();
            List<String> observations = new ArrayList<>();
            List<LLMProvider.ToolResult> toolResults = new ArrayList<>();

            for (LLMProvider.ToolCall call : response.toolCalls()) {
                String fingerprint = call.toolName() + "|" + call.arguments().toString();
                if (usedFingerprints.contains(fingerprint)) {
                    System.out.println("[Agent/SHORT]   SKIP duplicate: " + call.toolName());
                    toolResults.add(new LLMProvider.ToolResult(call.callId(),
                        "[DUPLICATE] This exact call was already made. Do not repeat it. Try a different query or tool."));
                    observations.add("[DUPLICATE SKIPPED]");
                    actionsTaken.add(call.toolName() + "(dup)");
                    continue;
                }
                usedFingerprints.add(fingerprint);
                System.out.println("[Agent/SHORT]   -> " + call.toolName());
                actionsTaken.add(call.toolName());
                String result = dispatch(call);
                System.out.println("[Agent/SHORT]   <- " + result.length() + " chars");
                observations.add(result);
                toolResults.add(new LLMProvider.ToolResult(call.callId(), result));
                history.add("OBSERVATION[" + call.toolName() + "]: "
                    + (result.length() > 500 ? result.substring(0, 497) + "..." : result));
            }

            trace.add(new ReasoningStep(stepsUsed, String.join(", ", actionsTaken),
                actionsTaken, observations, "", false));

            if (remaining == 0) { System.out.println("[Agent/SHORT] Budget exhausted."); break; }

            String updatedPrompt = buildShortExecutorPrompt(remaining) + buildHistoryBlock(history);
            response = llmProvider.continueWithToolResults(
                updatedPrompt, messages, toolResults, response.rawOutputItems(), toolDefinitions);
        }

        System.out.println("\n[Agent/SHORT] Synthesising...");
        printTrace(trace);
        return synthesise(userQuestion, history);
    }

    // ── MID execution ─────────────────────────────────────────────────────

    private String executeMid(String userQuestion, Planner.Plan plan) throws Exception {
        Set<String>               usedFingerprints = new HashSet<>();
        List<ReasoningStep>       trace            = new ArrayList<>();
        List<String>              history          = new ArrayList<>();
        List<LLMProvider.Message> messages         = new ArrayList<>();
        messages.add(new LLMProvider.Message("user", userQuestion));

        String sysPrompt = buildMidExecutorPrompt(plan, plan.stepBudget());
        if (DEBUG_LLM_PROMPT) printPromptBlock("MID EXECUTOR PROMPT", sysPrompt);

        LLMProvider.LLMResponse response = llmProvider.chat(sysPrompt, messages, toolDefinitions);
        int stepsUsed = 0;

        while (!response.toolCalls().isEmpty() && stepsUsed < plan.stepBudget()) {
            stepsUsed++;
            int remaining = plan.stepBudget() - stepsUsed;
            System.out.println("\n[Agent/MID] -- Step " + stepsUsed + " / " + plan.stepBudget()
                + " (budget left: " + remaining + ") --");

            List<String> actionsTaken = new ArrayList<>();
            List<String> observations = new ArrayList<>();
            List<LLMProvider.ToolResult> toolResults = new ArrayList<>();

            for (LLMProvider.ToolCall call : response.toolCalls()) {
                String fingerprint = call.toolName() + "|" + call.arguments().toString();
                if (usedFingerprints.contains(fingerprint)) {
                    System.out.println("[Agent/MID]   SKIP duplicate: " + call.toolName());
                    toolResults.add(new LLMProvider.ToolResult(call.callId(),
                        "[DUPLICATE] Already called with same args. Choose a different approach."));
                    observations.add("[DUPLICATE SKIPPED]");
                    actionsTaken.add(call.toolName() + "(dup)");
                    continue;
                }
                usedFingerprints.add(fingerprint);
                System.out.println("[Agent/MID]   -> " + call.toolName());
                actionsTaken.add(call.toolName());
                String result = dispatch(call);
                System.out.println("[Agent/MID]   <- " + result.length() + " chars");
                observations.add(result);
                toolResults.add(new LLMProvider.ToolResult(call.callId(), result));
                history.add("OBSERVATION[" + call.toolName() + "]: "
                    + (result.length() > 500 ? result.substring(0, 497) + "..." : result));
            }

            trace.add(new ReasoningStep(stepsUsed, String.join(", ", actionsTaken),
                actionsTaken, observations, "", false));

            if (remaining == 0) { System.out.println("[Agent/MID] Budget exhausted."); break; }

            String updatedPrompt = buildMidExecutorPrompt(plan, remaining) + buildHistoryBlock(history);
            response = llmProvider.continueWithToolResults(
                updatedPrompt, messages, toolResults, response.rawOutputItems(), toolDefinitions);
        }

        System.out.println("\n[Agent/MID] Synthesising...");
        printTrace(trace);
        return synthesise(userQuestion, history);
    }

    // ── LONG execution ────────────────────────────────────────────────────

    private String executeLong(String userQuestion, Planner.Plan plan) throws Exception {
        Set<String>               usedFingerprints = new HashSet<>();
        List<ReasoningStep>       trace            = new ArrayList<>();
        List<String>              history          = new ArrayList<>();
        List<LLMProvider.Message> messages         = new ArrayList<>();
        messages.add(new LLMProvider.Message("user", userQuestion));

        history.add("PLAN INTENT: " + plan.intent());
        for (int i = 0; i < plan.plannedSteps().size(); i++)
            history.add("  STEP " + (i + 1) + ": " + plan.plannedSteps().get(i));

        String sysPrompt = buildLongExecutorPrompt(plan, history, 0, plan.stepBudget(), "");
        if (DEBUG_LLM_PROMPT) printPromptBlock("LONG EXECUTOR PROMPT", sysPrompt);

        LLMProvider.LLMResponse response = llmProvider.chat(sysPrompt, messages, toolDefinitions);
        int stepsUsed = 0;

        while (!response.toolCalls().isEmpty() && stepsUsed < plan.stepBudget()) {
            stepsUsed++;
            int remaining = plan.stepBudget() - stepsUsed;
            System.out.println("\n[Agent/LONG] -- Step " + stepsUsed + " / " + plan.stepBudget()
                + " (remaining: " + remaining + ") --");

            List<String> actionsTaken = new ArrayList<>();
            List<String> observations = new ArrayList<>();
            List<LLMProvider.ToolResult> toolResults = new ArrayList<>();

            for (LLMProvider.ToolCall call : response.toolCalls()) {
                String fingerprint = call.toolName() + "|" + call.arguments().toString();
                if (usedFingerprints.contains(fingerprint)) {
                    System.out.println("[Agent/LONG]   SKIP duplicate: " + call.toolName());
                    toolResults.add(new LLMProvider.ToolResult(call.callId(),
                        "[DUPLICATE] Already called with same args. Try a different query or tool."));
                    observations.add("[DUPLICATE SKIPPED]");
                    actionsTaken.add(call.toolName() + "(dup)");
                    continue;
                }
                usedFingerprints.add(fingerprint);
                System.out.println("[Agent/LONG]   -> " + call.toolName());
                actionsTaken.add(call.toolName());
                String result = dispatch(call);
                System.out.println("[Agent/LONG]   <- " + result.length() + " chars");
                observations.add(result);
                toolResults.add(new LLMProvider.ToolResult(call.callId(), result));
                history.add("OBSERVATION[" + call.toolName() + "]: "
                    + (result.length() > 500 ? result.substring(0, 497) + "..." : result));
            }

            String reflection    = reflect(userQuestion, plan.intent(), history, remaining);
            boolean goalAchieved = reflection.contains("[GOAL_ACHIEVED]");
            System.out.println("[Agent/LONG]   Reflection: "
                + (reflection.length() > 130 ? reflection.substring(0, 127) + "..." : reflection));

            trace.add(new ReasoningStep(stepsUsed, String.join(", ", actionsTaken),
                actionsTaken, observations, reflection, goalAchieved));

            if (goalAchieved || remaining == 0) {
                System.out.println(goalAchieved
                    ? "[Agent/LONG] Goal achieved." : "[Agent/LONG] Budget exhausted.");
                break;
            }

            String updatedPrompt = buildLongExecutorPrompt(plan, history, stepsUsed, remaining, reflection);
            response = llmProvider.continueWithToolResults(
                updatedPrompt, messages, toolResults, response.rawOutputItems(), toolDefinitions);
        }

        System.out.println("\n[Agent/LONG] Synthesising...");
        printTrace(trace);
        return synthesise(userQuestion, history);
    }

    // ── Reflection (LONG only) ─────────────────────────────────────────────

    private String reflect(String question, String intent,
                           List<String> history, int remaining) throws Exception {
        String prompt =
            "Reflection agent. Answer in 2-3 sentences:\n"
            + "1. What useful info was gathered?\n"
            + "2. Is there enough to fully answer: \"" + question + "\"?\n"
            + "3. End with exactly [GOAL_ACHIEVED] or [CONTINUE].\n\n"
            + "Remaining steps: " + remaining + "\n"
            + "Goal: " + intent + "\n\n"
            + "History:\n" + String.join("\n", history);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "gpt-4o-mini");
        var arr = mapper.createArrayNode();
        arr.add(mapper.createObjectNode().put("role", "user").put("content", prompt));
        body.set("input", arr);

        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(RESPONSES_URL))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                .build();

        HttpResponse<String> resp =
            HttpClient.newHttpClient().send(req, HttpResponse.BodyHandlers.ofString());
        JsonNode root = mapper.readTree(resp.body());

        if (!root.has("output") || root.get("output").isEmpty()) {
            System.err.println("[Reflect] No output. Raw: " + resp.body());
            return "[CONTINUE]";
        }
        return root.path("output").get(0).path("content").get(0).path("text").asText("[CONTINUE]");
    }

    // ── Synthesis ──────────────────────────────────────────────────────────

    private String synthesise(String userQuestion, List<String> history) throws Exception {
        StringBuilder ctx = new StringBuilder();
        ctx.append("Answer the user's question using ONLY the information gathered below.\n");
        ctx.append("Be comprehensive and well-structured. Do not invent facts.\n\n");
        ctx.append("User question: ").append(userQuestion).append("\n\n");
        ctx.append("Research gathered:\n");
        boolean hasObs = false;
        for (String h : history) {
            if (h.startsWith("OBSERVATION")) {
                ctx.append(h).append("\n\n");
                hasObs = true;
            }
        }
        if (!hasObs) {
            // Model answered without needing tools
            return llmProvider.chat(executorSystemPrompt,
                List.of(new LLMProvider.Message("user", userQuestion)), List.of())
                .finalText();
        }
        LLMProvider.LLMResponse r = llmProvider.chat(
            executorSystemPrompt,
            List.of(new LLMProvider.Message("user", ctx.toString())),
            List.of());
        return r.finalText() != null ? r.finalText() : "Could not synthesise a final answer.";
    }

    // ── Prompt builders ────────────────────────────────────────────────────

    private String buildShortExecutorPrompt(int remaining) {
        return executorSystemPrompt + "\n\n"
            + "## Self-Directed Execution Rules\n"
            + "You decide what to do, but follow these guardrails:\n\n"
            + "1. STEPS REMAINING: " + remaining + " -- do NOT exceed this.\n"
            + "2. ASSESS COMPLEXITY: If the question is a greeting or simple fact,\n"
            + "   answer immediately with NO tool calls.\n"
            + "3. NO REPEATS: Never call the same tool with the same arguments twice.\n"
            + "   If a search returned no results, change the query -- do not retry.\n"
            + "4. STOP EARLY: As soon as you have enough info to answer, return it.\n"
            + "5. EFFICIENCY: One focused tool call per step preferred.\n";
    }

    private String buildMidExecutorPrompt(Planner.Plan plan, int remaining) {
        return executorSystemPrompt + "\n\n"
            + "## Execution Context\n"
            + "Goal: " + plan.intent() + "\n"
            + "Complexity: " + plan.complexity() + "\n"
            + "Steps remaining: " + remaining + "\n\n"
            + "## Rules\n"
            + "- Do NOT exceed " + remaining + " more tool-call turns.\n"
            + "- Do NOT repeat a tool call with the same arguments.\n"
            + "- Stop as soon as you have enough to answer the question.\n"
            + "- One focused call per step.\n";
    }

    private String buildLongExecutorPrompt(Planner.Plan plan, List<String> history,
                                           int used, int remaining, String lastReflection) {
        StringBuilder sb = new StringBuilder(executorSystemPrompt);
        sb.append("\n\n## Execution Plan\n");
        sb.append("Goal: ").append(plan.intent()).append("\n");
        sb.append("Steps used / budget: ").append(used).append(" / ").append(plan.stepBudget()).append("\n");
        sb.append("Steps remaining: ").append(remaining).append("\n\n");
        sb.append("### Planned steps:\n");
        for (int i = 0; i < plan.plannedSteps().size(); i++)
            sb.append(i + 1).append(". ").append(plan.plannedSteps().get(i)).append("\n");
        if (!lastReflection.isBlank())
            sb.append("\n### Last reflection:\n").append(lastReflection).append("\n");
        sb.append("\n### Rules:\n");
        sb.append("- Exactly ").append(remaining).append(" step(s) left -- respect this.\n");
        sb.append("- Do NOT repeat any tool call with the same arguments.\n");
        sb.append("- Return final answer immediately if you already have enough info.\n");
        return sb.toString();
    }

    private String buildHistoryBlock(List<String> history) {
        if (history.isEmpty()) return "";
        StringBuilder sb = new StringBuilder("\n\n## What you have gathered so far:\n");
        for (String h : history)
            if (h.startsWith("OBSERVATION"))
                sb.append(h, 0, Math.min(h.length(), 300)).append("\n");
        return sb.toString();
    }

    // ── Dispatch ───────────────────────────────────────────────────────────

    // -- Dispatch ---------------------------------------------------------------

    private String dispatch(LLMProvider.ToolCall call) throws Exception {
        String name = call.toolName();

        if (name.startsWith(GET_PROMPT_PREFIX)) {
            String promptName = name.substring(GET_PROMPT_PREFIX.length());
            MCPClient client  = promptRegistry.get(promptName);
            if (client == null) throw new RuntimeException("Unknown prompt: " + promptName);

            ObjectNode rawArgs = call.arguments().isObject()
                ? (ObjectNode) call.arguments() : mapper.createObjectNode();

            // Coerce LLM-supplied strings to declared types; strip empty optionals.
            ObjectNode coerced = coercePromptArgs(promptName, rawArgs);
            System.out.println("[Agent] prompts/get[" + promptName + "] args: " + coerced);

            JsonNode result = client.getPrompt(promptName, coerced);
            String pretty   = result.toPrettyString();
            if (DEBUG_MCP_RAW) printMcpResult("prompts/get[" + promptName + "]", pretty);
            return pretty;
        }

        MCPClient client = toolRegistry.get(name);
        if (client == null) throw new RuntimeException("Unknown tool: " + name);
        String result = client.callTool(name, call.arguments());
        if (DEBUG_MCP_RAW) printMcpResult(name, result);
        return result;
    }

    /**
     * Coerce LLM-supplied argument strings to the types the MCP server declared.
     *
     * The LLM always sends JSON strings. Servers using Pydantic reject:
     *   - A string "42" where they expect integer
     *   - An empty string "" where they expect a valid value
     *
     * This method:
     *   1. Drops empty-string optional args (never send "" to the server)
     *   2. Converts "42" -> 42 (integer) for args declared as integer/int/number
     *   3. Converts "true"/"false" -> boolean for args declared as boolean
     *   4. Passes required+empty args through so the server gives a clear error
     */
    private ObjectNode coercePromptArgs(String promptName, ObjectNode rawArgs) {
        ObjectNode coerced = mapper.createObjectNode();
        JsonNode meta = promptMeta.get(promptName);
        if (meta == null) return rawArgs;

        JsonNode declaredArgs = meta.path("arguments");
        if (!declaredArgs.isArray()) return rawArgs;

        // Build lookup maps from the declared argument metadata
        Map<String, String>  argTypes    = new LinkedHashMap<>();
        Map<String, Boolean> argRequired = new LinkedHashMap<>();
        for (JsonNode arg : declaredArgs) {
            String argName = arg.path("name").asText();
            // MCP spec: arguments have name, description, required.
            // The "type" hint comes from the inputSchema we built during discovery.
            // Fall back to "string" for anything we do not recognise.
            String type = "string";
            // Try to pick up type from the synthetic schema in toolDefinitions
            for (LLMProvider.ToolDefinition td : toolDefinitions) {
                if (td.name().equals(GET_PROMPT_PREFIX + promptName) && td.inputSchema() != null) {
                    JsonNode propType = td.inputSchema().path("properties")
                                         .path(argName).path("type");
                    if (!propType.isMissingNode()) {
                        type = propType.asText("string");
                    }
                    break;
                }
            }
            argTypes.put(argName, type);
            argRequired.put(argName, arg.path("required").asBoolean(false));
        }

        // Walk the raw arguments the LLM provided
        rawArgs.fields().forEachRemaining(entry -> {
            String   key    = entry.getKey();
            JsonNode val    = entry.getValue();
            String   strVal = val.isNull() ? "" : val.asText("").trim();
            String   type   = argTypes.getOrDefault(key, "string");
            boolean  req    = argRequired.getOrDefault(key, false);

            // Drop empty optional args -- never send "" to the server
            if (strVal.isEmpty() || strVal.equalsIgnoreCase("null")) {
                if (!req) {
                    System.out.println("[Coerce] Skipping empty optional arg: " + key);
                    return;
                }
                coerced.put(key, ""); // required but empty -- server will report missing
                return;
            }

            switch (type.toLowerCase()) {
                case "integer":
                case "int":
                case "number":
                    try {
                        coerced.put(key, Integer.parseInt(strVal));
                        System.out.println("[Coerce] " + key + " string->int: " + strVal);
                    } catch (NumberFormatException e) {
                        if (!req) {
                            System.out.println("[Coerce] Skipping non-integer optional: " + key + "=" + strVal);
                        } else {
                            coerced.put(key, strVal);
                        }
                    }
                    break;
                case "boolean":
                case "bool":
                    coerced.put(key, Boolean.parseBoolean(strVal));
                    System.out.println("[Coerce] " + key + " string->bool: " + strVal);
                    break;
                default:
                    coerced.put(key, strVal);
            }
        });

        return coerced;
    }


    // ── Discovery ──────────────────────────────────────────────────────────

    private void discoverAll(List<MCPClient> clients) throws Exception {
        for (MCPClient client : clients) {
            MCPServerInfo info = client.discover();
            discoveredServers.add(info);

            for (JsonNode tool : info.getTools()) {
                String name = tool.path("name").asText();
                if (toolRegistry.containsKey(name)) {
                    System.out.println("[WARN] Dup tool '" + name + "'"); continue;
                }
                toolRegistry.put(name, client);
                toolDefinitions.add(new LLMProvider.ToolDefinition(
                    name, tool.path("description").asText(""),
                    tool.has("inputSchema") ? tool.get("inputSchema") : null));
            }

            for (JsonNode res : info.getResources())
                resourceRegistry.put(
                    res.path("uri").asText(res.path("name").asText()), client);

            for (JsonNode prompt : info.getPrompts()) {
                String promptName = prompt.path("name").asText();
                if (promptRegistry.containsKey(promptName)) {
                    System.out.println("[WARN] Dup prompt '" + promptName + "'"); continue;
                }
                promptRegistry.put(promptName, client);
                promptMeta.put(promptName, prompt);

                String     synName = GET_PROMPT_PREFIX + promptName;
                ObjectNode schema  = mapper.createObjectNode();
                ObjectNode props   = mapper.createObjectNode();
                List<String> req   = new ArrayList<>();

                for (JsonNode arg : prompt.path("arguments")) {
                    String argName = arg.path("name").asText();
                    ObjectNode as  = mapper.createObjectNode().put("type", "string");
                    String ad      = arg.path("description").asText("");
                    if (!ad.isBlank()) as.put("description", ad);
                    props.set(argName, as);
                    if (arg.path("required").asBoolean(false)) req.add(argName);
                }
                schema.put("type", "object");
                schema.set("properties", props);
                if (!req.isEmpty()) schema.set("required", mapper.valueToTree(req));

                toolRegistry.put(synName, client);
                toolDefinitions.add(new LLMProvider.ToolDefinition(synName,
                    "Fetch the '" + promptName + "' prompt template from MCP server. "
                        + prompt.path("description").asText(""),
                    schema));
            }
        }
    }

    // ── Capability / System prompt builders ───────────────────────────────

    private String buildCapabilitySummary() {
        StringBuilder sb = new StringBuilder();
        List<String> tools = new ArrayList<>();
        for (String n : toolRegistry.keySet())
            if (!n.startsWith(GET_PROMPT_PREFIX)) tools.add(n);
        if (!tools.isEmpty())
            sb.append("TOOLS: ").append(String.join(", ", tools)).append("\n");
        if (!promptRegistry.isEmpty())
            sb.append("PROMPT TEMPLATES: ").append(String.join(", ", promptRegistry.keySet())).append("\n");
        if (!resourceRegistry.isEmpty())
            sb.append("RESOURCES: ").append(String.join(", ", resourceRegistry.keySet())).append("\n");
        return sb.toString();
    }

    private String buildExecutorSystemPrompt() {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a capable AI assistant backed by MCP server tools.\n\n");

        List<String> realTools = new ArrayList<>();
        for (String n : toolRegistry.keySet())
            if (!n.startsWith(GET_PROMPT_PREFIX)) realTools.add(n);

        if (!realTools.isEmpty()) {
            sb.append("## Available Tools\n");
            for (String name : realTools) {
                sb.append("- **").append(name).append("**");
                toolDefinitions.stream().filter(t -> t.name().equals(name)).findFirst()
                    .ifPresent(def -> {
                        if (!def.description().isBlank())
                            sb.append(": ").append(firstLine(def.description()));
                    });
                sb.append("\n");
            }
            sb.append("\n");
        }

        if (!promptRegistry.isEmpty()) {
            sb.append("## Available Prompt Templates\n");
            sb.append("These are pre-built prompt templates on the MCP server.\n");
            sb.append("To use one, call its corresponding get_prompt__ tool.\n\n");
            promptMeta.forEach((name, meta) -> {
                sb.append("- **").append(name).append("**");
                String desc = meta.path("description").asText("").trim();
                // Include FULL description so LLM understands what the prompt does
                if (!desc.isBlank()) sb.append(":\n  ").append(desc.replace("\n", "\n  "));
                sb.append("\n");
                sb.append("  Call tool: get_prompt__").append(name).append("\n");
                JsonNode args = meta.path("arguments");
                if (args.isArray() && args.size() > 0) {
                    sb.append("  Arguments:\n");
                    for (JsonNode arg : args) {
                        boolean required = arg.path("required").asBoolean(false);
                        sb.append("    - ").append(arg.path("name").asText())
                          .append(required ? " (required)" : " (optional)");
                        String argDesc = arg.path("description").asText("").trim();
                        if (!argDesc.isBlank()) sb.append(": ").append(argDesc);
                        sb.append("\n");
                    }
                }
                sb.append("\n");
            });
        }

        return sb.toString();
    }

    // ── Debug printing ─────────────────────────────────────────────────────

    private static void printMcpResult(String source, String result) {
        System.out.println("\n+---- RAW MCP RESULT [" + source + "] ----+");
        System.out.println(result.length() > 2000 ? result.substring(0, 1997) + "..." : result);
        System.out.println("+-------------------------------------------+\n");
    }

    private static void printPromptBlock(String label, String prompt) {
        System.out.println("\n+====== " + label + " ======+");
        System.out.println(prompt);
        System.out.println("+========================" + "=".repeat(label.length()) + "+\n");
    }

    // ── Utilities ──────────────────────────────────────────────────────────

    private static String firstLine(String s) {
        if (s == null || s.isBlank()) return "";
        return s.split("\n")[0].trim();
    }

    private void printTrace(List<ReasoningStep> trace) {
        if (trace.isEmpty()) return;
        System.out.println("\n========= REASONING TRACE =========");
        for (ReasoningStep s : trace) System.out.print(s.fullTrace());
        System.out.println("====================================\n");
    }

    private String extractApiKey() {
        try {
            var f = llmProvider.getClass().getDeclaredField("apiKey");
            f.setAccessible(true);
            return (String) f.get(llmProvider);
        } catch (Exception e) {
            throw new RuntimeException("Could not read apiKey: " + e.getMessage());
        }
    }

    public void printDiscoverySummary() {
        System.out.println("\n==============================================");
        System.out.println("  MCP Discovery Summary");
        System.out.println("==============================================");
        for (MCPServerInfo info : discoveredServers) {
            System.out.println("  Server : " + info);
            if (!info.getTools().isEmpty()) {
                System.out.println("  Tools:");
                for (JsonNode t : info.getTools())
                    System.out.printf("    %-35s %s%n",
                        t.path("name").asText(), firstLine(t.path("description").asText("")));
            }
            if (!info.getResources().isEmpty()) {
                System.out.println("  Resources:");
                for (JsonNode r : info.getResources())
                    System.out.printf("    %-35s %s%n",
                        r.path("uri").asText(r.path("name").asText()),
                        firstLine(r.path("description").asText("")));
            }
            if (!info.getPrompts().isEmpty()) {
                System.out.println("  Prompts:");
                for (JsonNode p : info.getPrompts())
                    System.out.printf("    %-35s %s%n",
                        p.path("name").asText(), firstLine(p.path("description").asText("")));
            }
            System.out.println();
        }
        long realTools = toolRegistry.keySet().stream()
            .filter(n -> !n.startsWith(GET_PROMPT_PREFIX)).count();
        System.out.println("  Real tools:    " + realTools);
        System.out.println("  Prompt tools:  " + promptRegistry.size());
        System.out.println("  Total slots:   " + toolDefinitions.size());
        System.out.println("==============================================\n");
    }
}
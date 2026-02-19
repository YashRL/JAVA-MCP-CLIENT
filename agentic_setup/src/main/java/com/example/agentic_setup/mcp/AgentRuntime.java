package com.example.agentic_setup.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.*;

/**
 * Core agent loop.
 *
 * run() returns a SynthesisInput (systemPrompt + userMessage) instead of
 * calling the LLM directly. This lets AgentStreamingService stream the final
 * synthesis call token-by-token to the WebSocket / SSE client.
 */
public class AgentRuntime {

    /** Everything the final streaming LLM call needs. */
    public record SynthesisInput(String systemPrompt, String userMessage) {}

    private static final String PROMPT_PREFIX   = "get_prompt__";
    private static final String PROMPT_SENTINEL = "__DEFERRED_PROMPT__:";

    public static boolean DEBUG_MCP    = true;
    public static boolean DEBUG_PROMPT = false;

    private final LLMProvider  llm;
    private final Planner      planner;
    private final ObjectMapper mapper = new ObjectMapper();

    private final List<MCPServerInfo>              servers     = new ArrayList<>();
    private final Map<String, MCPClient>           toolReg     = new LinkedHashMap<>();
    private final Map<String, MCPClient>           promptReg   = new LinkedHashMap<>();
    private final Map<String, MCPClient>           resourceReg = new LinkedHashMap<>();
    private final Map<String, JsonNode>            promptMeta  = new LinkedHashMap<>();
    private final List<LLMProvider.ToolDefinition> toolDefs    = new ArrayList<>();

    private String baseSystemPrompt;

    public AgentRuntime(List<MCPClient> clients, LLMProvider llm, Planner planner)
            throws Exception {
        this.llm     = llm;
        this.planner = planner;
        clients.forEach(c -> {
            try { register(c.discover(), c); }
            catch (Exception e) { throw new RuntimeException(e); }
        });
        baseSystemPrompt = buildBasePrompt();
        printSummary();
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Runs the full ReAct loop then returns a SynthesisInput.
     * The actual final LLM call is done by AgentStreamingService (streamed).
     */
    public SynthesisInput run(String question) throws Exception {
        var plan = planner.plan(question, buildCapabilities());
        System.out.printf("[Agent] mode=%-5s  complexity=%-8s  budget=%d%n",
                plan.mode(), plan.complexity().toUpperCase(), plan.budget());

        if (plan.isTrivial())
            return new SynthesisInput(null, plan.directAnswer());

        return switch (plan.mode()) {
            case SHORT -> execute(question, plan, Planner.MAX_STEPS, false);
            case MID   -> execute(question, plan, plan.budget(),     false);
            case LONG  -> execute(question, plan, plan.budget(),     true);
        };
    }

    // ── Execution loop ────────────────────────────────────────────────────

    private SynthesisInput execute(String question, Planner.Plan plan, int budget, boolean reflect)
            throws Exception {

        var fingerprints = new HashSet<String>();
        var trace        = new ArrayList<ReasoningStep>();
        var history      = new ArrayList<String>();
        var deferred     = new ArrayList<String>();
        var messages     = List.of(new LLMProvider.Message("user", question));
        var sysPrompt    = buildExecPrompt(plan, budget);

        if (DEBUG_PROMPT) printBlock("EXEC PROMPT [" + plan.mode() + "]", sysPrompt);

        var response = llm.chat(sysPrompt, messages, toolDefs);
        int used = 0;

        while (!response.toolCalls().isEmpty() && used < budget) {
            used++;
            int remaining = budget - used;
            System.out.printf("[Agent/%s] step %d/%d%n", plan.mode(), used, budget);

            var actions = new ArrayList<String>();
            var obs     = new ArrayList<String>();
            var results = new ArrayList<LLMProvider.ToolResult>();

            for (var call : response.toolCalls()) {
                var fp = call.toolName() + "|" + call.arguments();
                if (fingerprints.contains(fp)) {
                    results.add(new LLMProvider.ToolResult(call.callId(),
                            "[DUPLICATE] Already called with these args."));
                    obs.add("[DUPLICATE SKIPPED]");
                    actions.add(call.toolName() + "(dup)");
                    continue;
                }
                fingerprints.add(fp);
                System.out.println("  -> " + call.toolName());
                actions.add(call.toolName());
                routeResult(call, dispatch(call), deferred, results, obs, history);
            }

            var reflection = "";
            var done = false;
            if (reflect) {
                reflection = reflect(question, plan.intent(), history, remaining);
                done = reflection.contains("[GOAL_ACHIEVED]");
            }

            trace.add(new ReasoningStep(used, actions, obs, reflection, done));
            if (done || remaining == 0) break;

            var updated = buildExecPrompt(plan, remaining) + historyBlock(history);
            response = llm.continueWithToolResults(updated, messages, results,
                    response.rawItems(), toolDefs);
        }

        printTrace(trace);
        return buildSynthesisInput(question, history, deferred);
    }

    // ── Build synthesis input (no LLM call — streamed by service layer) ───

    private SynthesisInput buildSynthesisInput(String question,
                                               List<String> history,
                                               List<String> deferred) {
        var obsCtx = new StringBuilder();
        var hasObs = false;
        for (var h : history)
            if (h.startsWith("OBSERVATION")) { obsCtx.append(h).append("\n\n"); hasObs = true; }

        // Case 1: MCP prompt collected → becomes system instruction
        if (!deferred.isEmpty()) {
            var sys = String.join("\n\n---\n\n", deferred);
            var usr = "Original request: " + question + "\n\n"
                    + (hasObs
                       ? "Research:\n\n" + obsCtx + "\nFulfil the system prompt using this research."
                       : "Fulfil the system prompt instructions.");
            return new SynthesisInput(sys, usr);
        }

        // Case 2: No tools used → model answers from knowledge
        if (!hasObs)
            return new SynthesisInput(baseSystemPrompt, question);

        // Case 3: Standard synthesis over observations
        var usr = "Answer using ONLY the research below. Be comprehensive. Do not invent facts.\n\n"
                + "Question: " + question + "\n\nResearch:\n" + obsCtx;
        return new SynthesisInput(baseSystemPrompt, usr);
    }

    // ── Tool result routing ────────────────────────────────────────────────

    private void routeResult(LLMProvider.ToolCall call, String result,
                              List<String> deferred, List<LLMProvider.ToolResult> results,
                              List<String> obs, List<String> history) {
        if (result.startsWith(PROMPT_SENTINEL)) {
            var text = result.substring(PROMPT_SENTINEL.length());
            deferred.add(text);
            System.out.println("  <- DEFERRED PROMPT (" + text.length() + " chars)");
            results.add(new LLMProvider.ToolResult(call.callId(),
                    "[PROMPT QUEUED] Template collected. Continue with data tools."));
            obs.add("[PROMPT DEFERRED]");
        } else {
            System.out.println("  <- " + result.length() + " chars");
            obs.add(result);
            results.add(new LLMProvider.ToolResult(call.callId(), result));
            history.add("OBSERVATION[" + call.toolName() + "]: " + truncate(result, 500));
        }
    }

    // ── Dispatch ──────────────────────────────────────────────────────────

    private String dispatch(LLMProvider.ToolCall call) throws Exception {
        var name = call.toolName();

        if (name.startsWith(PROMPT_PREFIX)) {
            var promptName = name.substring(PROMPT_PREFIX.length());
            var client     = promptReg.get(promptName);
            if (client == null) throw new RuntimeException("Unknown prompt: " + promptName);
            var args   = coerceArgs(promptName,
                    call.arguments().isObject() ? (ObjectNode) call.arguments() : mapper.createObjectNode());
            var result = client.getPrompt(promptName, args);
            if (DEBUG_MCP) printBlock("MCP prompts/get[" + promptName + "]", result.toPrettyString());
            return PROMPT_SENTINEL + extractText(result);
        }

        var client = toolReg.get(name);
        if (client == null) throw new RuntimeException("Unknown tool: " + name);
        var result = client.callTool(name, call.arguments());
        if (DEBUG_MCP) printBlock("MCP " + name, result);
        return result;
    }

    // ── Reflection ────────────────────────────────────────────────────────

    private String reflect(String question, String intent, List<String> history, int remaining)
            throws Exception {
        var msg = "Assess progress in 2-3 sentences.\n"
                + "Question: " + question + "\nGoal: " + intent
                + "\nSteps remaining: " + remaining
                + "\nHistory:\n" + String.join("\n", history)
                + "\n\nEnd with exactly [GOAL_ACHIEVED] or [CONTINUE].";
        var r = llm.complete("You are a reflection agent. Be concise.", msg);
        return r.isBlank() ? "[CONTINUE]" : r;
    }

    // ── MCP prompt text extraction ─────────────────────────────────────────

    private String extractText(JsonNode messages) {
        if (!messages.isArray() || messages.isEmpty()) return messages.toString();
        if (messages.size() == 1) return msgText(messages.get(0));
        var sb = new StringBuilder();
        messages.forEach(m -> {
            if (sb.length() > 0) sb.append("\n\n");
            sb.append("[").append(m.path("role").asText("user").toUpperCase()).append("]\n")
              .append(msgText(m));
        });
        return sb.toString();
    }

    private String msgText(JsonNode msg) {
        var c = msg.path("content");
        if (c.isObject()) return c.path("text").asText(c.toString());
        if (c.isArray()) {
            var sb = new StringBuilder();
            c.forEach(b -> { if ("text".equals(b.path("type").asText())) {
                if (sb.length() > 0) sb.append('\n');
                sb.append(b.path("text").asText());
            }});
            return sb.toString();
        }
        return c.asText(msg.toString());
    }

    // ── Argument coercion ──────────────────────────────────────────────────

    private ObjectNode coerceArgs(String promptName, ObjectNode raw) {
        var out      = mapper.createObjectNode();
        var meta     = promptMeta.get(promptName);
        if (meta == null) return raw;
        var declared = meta.path("arguments");
        if (!declared.isArray()) return raw;

        var types    = new LinkedHashMap<String, String>();
        var required = new LinkedHashMap<String, Boolean>();
        for (var arg : declared) {
            var n = arg.path("name").asText();
            types.put(n, schemaType(promptName, n));
            required.put(n, arg.path("required").asBoolean(false));
        }

        raw.fields().forEachRemaining(e -> {
            var key = e.getKey();
            var val = e.getValue().isNull() ? "" : e.getValue().asText("").trim();
            var req = required.getOrDefault(key, false);
            if (val.isEmpty() || "null".equalsIgnoreCase(val)) {
                if (!req) return; out.put(key, ""); return;
            }
            switch (types.getOrDefault(key, "string").toLowerCase()) {
                case "integer", "int", "number" -> {
                    try { out.put(key, Integer.parseInt(val)); }
                    catch (NumberFormatException ex) { if (req) out.put(key, val); }
                }
                case "boolean", "bool" -> out.put(key, Boolean.parseBoolean(val));
                default               -> out.put(key, val);
            }
        });
        return out;
    }

    private String schemaType(String promptName, String argName) {
        return toolDefs.stream()
                .filter(t -> t.name().equals(PROMPT_PREFIX + promptName) && t.inputSchema() != null)
                .findFirst()
                .map(t -> t.inputSchema().path("properties").path(argName).path("type").asText("string"))
                .orElse("string");
    }

    // ── Discovery ─────────────────────────────────────────────────────────

    private void register(MCPServerInfo info, MCPClient client) {
        servers.add(info);
        info.getTools().forEach(t -> {
            var name = t.path("name").asText();
            if (toolReg.containsKey(name)) return;
            toolReg.put(name, client);
            toolDefs.add(new LLMProvider.ToolDefinition(name, t.path("description").asText(""),
                    t.has("inputSchema") ? t.get("inputSchema") : null));
        });
        info.getResources().forEach(r ->
                resourceReg.put(r.path("uri").asText(r.path("name").asText()), client));
        info.getPrompts().forEach(p -> {
            var pName = p.path("name").asText();
            if (promptReg.containsKey(pName)) return;
            promptReg.put(pName, client);
            promptMeta.put(pName, p);
            var schema = mapper.createObjectNode().put("type", "object");
            var props  = mapper.createObjectNode();
            var req    = new ArrayList<String>();
            p.path("arguments").forEach(arg -> {
                var n  = arg.path("name").asText();
                var as = mapper.createObjectNode().put("type", "string");
                var d  = arg.path("description").asText("");
                if (!d.isBlank()) as.put("description", d);
                props.set(n, as);
                if (arg.path("required").asBoolean(false)) req.add(n);
            });
            schema.set("properties", props);
            if (!req.isEmpty()) schema.set("required", mapper.valueToTree(req));
            toolDefs.add(new LLMProvider.ToolDefinition(PROMPT_PREFIX + pName,
                    "Output template for '" + pName + "'. Structures the FINAL ANSWER. "
                    + "Call early to set format; use data tools for research. "
                    + p.path("description").asText(""), schema));
        });
    }

    // ── Prompt builders ────────────────────────────────────────────────────

    private String buildCapabilities() {
        var sb    = new StringBuilder();
        var tools = toolReg.keySet().stream().filter(n -> !n.startsWith(PROMPT_PREFIX)).toList();
        if (!tools.isEmpty())       sb.append("TOOLS: ").append(String.join(", ", tools)).append("\n");
        if (!promptReg.isEmpty())   sb.append("PROMPT TEMPLATES: ").append(String.join(", ", promptReg.keySet())).append("\n");
        if (!resourceReg.isEmpty()) sb.append("RESOURCES: ").append(String.join(", ", resourceReg.keySet())).append("\n");
        return sb.toString();
    }

    private String buildBasePrompt() {
        var sb    = new StringBuilder("You are a capable AI assistant backed by MCP server tools.\n\n");
        var tools = toolReg.keySet().stream().filter(n -> !n.startsWith(PROMPT_PREFIX)).toList();
        if (!tools.isEmpty()) {
            sb.append("## Available Tools\n");
            tools.forEach(name -> {
                sb.append("- **").append(name).append("**");
                toolDefs.stream().filter(t -> t.name().equals(name)).findFirst().ifPresent(d -> {
                    if (!d.description().isBlank()) sb.append(": ").append(firstLine(d.description()));
                });
                sb.append("\n");
            });
            sb.append("\n");
        }
        if (!promptReg.isEmpty()) {
            sb.append("## Prompt Templates (shape final answer — not data sources)\n");
            sb.append("Calling get_prompt__* defers the template to synthesis. Still use data tools.\n\n");
            promptMeta.forEach((name, meta) -> {
                sb.append("- **").append(name).append("**");
                var desc = meta.path("description").asText("").trim();
                if (!desc.isBlank()) sb.append(": ").append(desc.replace("\n", " "));
                sb.append("\n  → call: get_prompt__").append(name).append("\n");
                meta.path("arguments").forEach(arg -> {
                    sb.append("    - ").append(arg.path("name").asText())
                      .append(arg.path("required").asBoolean(false) ? " (required)" : " (optional)");
                    var d = arg.path("description").asText("").trim();
                    if (!d.isBlank()) sb.append(": ").append(d);
                    sb.append("\n");
                });
                sb.append("\n");
            });
        }
        return sb.toString();
    }

    private String buildExecPrompt(Planner.Plan plan, int remaining) {
        var sb = new StringBuilder(baseSystemPrompt).append("\n## Execution\n");
        if (plan.mode() != PlanningMode.SHORT)
            sb.append("Goal: ").append(plan.intent()).append("\n");
        if (!plan.steps().isEmpty()) {
            sb.append("Planned steps:\n");
            for (int i = 0; i < plan.steps().size(); i++)
                sb.append(i + 1).append(". ").append(plan.steps().get(i)).append("\n");
        }
        sb.append("Steps remaining: ").append(remaining).append("\n\n")
          .append("- Do NOT exceed ").append(remaining).append(" tool-call turns.\n")
          .append("- Never repeat a tool call with the same arguments.\n")
          .append("- Stop as soon as you have enough to answer.\n");
        return sb.toString();
    }

    private String historyBlock(List<String> history) {
        if (history.isEmpty()) return "";
        var sb = new StringBuilder("\n## Gathered so far:\n");
        history.stream().filter(h -> h.startsWith("OBSERVATION"))
               .forEach(h -> sb.append(truncate(h, 300)).append("\n"));
        return sb.toString();
    }

    private static void printBlock(String label, String content) {
        System.out.println("\n┌──── " + label);
        System.out.println(truncate(content, 2000));
        System.out.println("└" + "─".repeat(20));
    }

    private static void printTrace(List<ReasoningStep> trace) {
        if (trace.isEmpty()) return;
        System.out.println("\n═══ REASONING TRACE ═══");
        trace.forEach(s -> System.out.print(s.trace()));
        System.out.println("═══════════════════════\n");
    }

    private static String firstLine(String s) {
        return (s == null || s.isBlank()) ? "" : s.split("\n")[0].trim();
    }

    private static String truncate(String s, int max) {
        return s.length() > max ? s.substring(0, max - 3) + "..." : s;
    }

    public void printSummary() {
        System.out.println("\n══════════════════════════════");
        servers.forEach(s -> System.out.println("  " + s));
        System.out.printf("  Tools: %d  Prompts: %d%n",
                toolReg.keySet().stream().filter(n -> !n.startsWith(PROMPT_PREFIX)).count(),
                promptReg.size());
        System.out.println("══════════════════════════════\n");
    }
}
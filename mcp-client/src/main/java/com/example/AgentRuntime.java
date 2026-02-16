package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.*;

/**
 * The agent runtime ties everything together.
 *
 * ── What changed and why ────────────────────────────────────────────────────
 * Previously the LLM only knew about "tools" because that's the only thing
 * sent in the API request.  Prompts and resources were discovered on the Java
 * side but never communicated to the model, so it acted as if they didn't exist.
 *
 * Fix: two complementary mechanisms
 *
 *  1. SYSTEM PROMPT  – a "developer" role message prepended to every request
 *     that lists all discovered prompts and resources in plain English so the
 *     LLM understands what the server offers.
 *
 *  2. get_prompt TOOL  – each discovered prompt is wrapped as a callable tool
 *     (get_prompt__<name>) so the LLM can actually fetch a rendered template
 *     and use its content.  This follows the MCP spec: prompts are meant to
 *     be retrieved via prompts/get and then used as conversation context.
 *
 * Flow:
 *  1. Accept any number of MCPClient instances
 *  2. Discover ALL capabilities from every server (tools + resources + prompts)
 *  3. Build a unified tool registry (tool name → MCPClient)
 *  4. Synthesise the system prompt that describes prompts + resources
 *  5. Register one synthetic "get_prompt__<name>" tool per discovered prompt
 *  6. Run the agentic loop with system prompt injected into every API call
 */
public class AgentRuntime {

    private static final String GET_PROMPT_PREFIX = "get_prompt__";

    private final LLMProvider  llmProvider;
    private final ObjectMapper mapper = new ObjectMapper();

    // ── Discovery state ────────────────────────────────────────────────────
    private final List<MCPServerInfo>              discoveredServers = new ArrayList<>();
    private final Map<String, MCPClient>           toolRegistry      = new LinkedHashMap<>();
    private final Map<String, MCPClient>           resourceRegistry  = new LinkedHashMap<>();
    // prompt name → MCPClient (for fetching via prompts/get)
    private final Map<String, MCPClient>           promptRegistry    = new LinkedHashMap<>();
    // prompt name → full JsonNode metadata (for schema building)
    private final Map<String, JsonNode>            promptMeta        = new LinkedHashMap<>();

    private final List<LLMProvider.ToolDefinition> toolDefinitions   = new ArrayList<>();

    // Built once after discovery; injected into every LLM call
    private String systemPrompt;

    // ── Constructor ────────────────────────────────────────────────────────

    public AgentRuntime(List<MCPClient> clients, LLMProvider llmProvider) throws Exception {
        this.llmProvider = llmProvider;
        discoverAll(clients);
        systemPrompt = buildSystemPrompt();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Run an agentic question → answer loop.
     */
    public String run(String userQuestion) throws Exception {
        List<LLMProvider.Message> messages = List.of(
            new LLMProvider.Message("user", userQuestion)
        );

        // Turn 1: LLM sees system prompt + tools (including synthetic prompt tools)
        LLMProvider.LLMResponse response =
            llmProvider.chat(systemPrompt, messages, toolDefinitions);

        int maxIterations = 10;
        int iteration     = 0;

        while (!response.toolCalls().isEmpty() && iteration < maxIterations) {
            iteration++;
            System.out.printf("%n[Agent] Turn %d – executing %d tool call(s)%n",
                iteration, response.toolCalls().size());

            List<LLMProvider.ToolResult> toolResults = new ArrayList<>();

            for (LLMProvider.ToolCall call : response.toolCalls()) {
                System.out.printf("[Agent]   → Calling '%s'%n", call.toolName());

                String result = dispatch(call);
                System.out.printf("[Agent]   ← Result (%d chars)%n", result.length());

                toolResults.add(new LLMProvider.ToolResult(call.callId(), result));
            }

            response = llmProvider.continueWithToolResults(
                systemPrompt,
                messages,
                toolResults,
                response.rawOutputItems(),
                toolDefinitions
            );
        }

        if (response.finalText() == null || response.finalText().isBlank()) {
            throw new RuntimeException("Agent loop ended but LLM produced no final text.");
        }

        return response.finalText();
    }

    /**
     * Print a summary of everything discovered across all connected servers.
     */
    public void printDiscoverySummary() {
        System.out.println("\n══════════════════════════════════════════════");
        System.out.println("  MCP Discovery Summary");
        System.out.println("══════════════════════════════════════════════");
        for (MCPServerInfo info : discoveredServers) {
            System.out.println("  Server : " + info);
            if (!info.getTools().isEmpty()) {
                System.out.println("  Tools:");
                for (JsonNode t : info.getTools())
                    System.out.printf("    %-35s %s%n",
                        t.path("name").asText(), t.path("description").asText(""));
            }
            if (!info.getResources().isEmpty()) {
                System.out.println("  Resources:");
                for (JsonNode r : info.getResources())
                    System.out.printf("    %-35s %s%n",
                        r.path("uri").asText(r.path("name").asText()),
                        r.path("description").asText(""));
            }
            if (!info.getPrompts().isEmpty()) {
                System.out.println("  Prompts:");
                for (JsonNode p : info.getPrompts())
                    System.out.printf("    %-35s %s%n",
                        p.path("name").asText(), p.path("description").asText(""));
            }
            System.out.println();
        }
        System.out.println("══════════════════════════════════════════════");
        System.out.printf("  Real MCP tools:      %d%n",
            toolRegistry.size() - promptRegistry.size());
        System.out.printf("  Prompt fetch tools:  %d%n", promptRegistry.size());
        System.out.printf("  Total tool slots:    %d%n", toolDefinitions.size());
        System.out.println("══════════════════════════════════════════════\n");
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    public List<MCPServerInfo>              getDiscoveredServers() { return discoveredServers; }
    public List<LLMProvider.ToolDefinition> getToolDefinitions()   { return toolDefinitions;  }
    public String                           getSystemPrompt()      { return systemPrompt;     }

    // ── Private helpers ────────────────────────────────────────────────────

    /**
     * Dispatch a tool call to the right backend.
     * Synthetic "get_prompt__*" tools are handled here directly via prompts/get;
     * everything else is forwarded to the appropriate MCPClient.
     */
    private String dispatch(LLMProvider.ToolCall call) throws Exception {
        String name = call.toolName();

        // ── Synthetic prompt-fetch tool ──────────────────────────────────
        if (name.startsWith(GET_PROMPT_PREFIX)) {
            String promptName = name.substring(GET_PROMPT_PREFIX.length());
            MCPClient client  = promptRegistry.get(promptName);
            if (client == null)
                throw new RuntimeException("Unknown prompt: " + promptName);

            // Build arguments ObjectNode from the call's arguments
            ObjectNode args = call.arguments().isObject()
                ? (ObjectNode) call.arguments()
                : mapper.createObjectNode();

            JsonNode messages = client.getPrompt(promptName, args);
            return messages.toPrettyString();
        }

        // ── Regular MCP tool ─────────────────────────────────────────────
        MCPClient client = toolRegistry.get(name);
        if (client == null)
            throw new RuntimeException("Unknown tool requested by LLM: " + name);

        return client.callTool(name, call.arguments());
    }

    /**
     * Run self-discovery on every MCPClient and build the unified registries.
     */
    private void discoverAll(List<MCPClient> clients) throws Exception {
        for (MCPClient client : clients) {
            MCPServerInfo info = client.discover();
            discoveredServers.add(info);

            // ── Real tools ───────────────────────────────────────────────
            for (JsonNode tool : info.getTools()) {
                String name = tool.path("name").asText();
                if (toolRegistry.containsKey(name)) {
                    System.out.printf("[WARNING] Tool '%s' already registered – " +
                        "skipping duplicate from %s%n", name, info.getServerName());
                    continue;
                }
                toolRegistry.put(name, client);
                toolDefinitions.add(new LLMProvider.ToolDefinition(
                    name,
                    tool.path("description").asText(""),
                    tool.has("inputSchema") ? tool.get("inputSchema") : null
                ));
            }

            // ── Resources ────────────────────────────────────────────────
            for (JsonNode resource : info.getResources()) {
                String uri = resource.path("uri").asText(resource.path("name").asText());
                resourceRegistry.put(uri, client);
            }

            // ── Prompts → registered as synthetic get_prompt__ tools ─────
            for (JsonNode prompt : info.getPrompts()) {
                String promptName = prompt.path("name").asText();

                if (promptRegistry.containsKey(promptName)) {
                    System.out.printf("[WARNING] Prompt '%s' already registered – " +
                        "skipping duplicate from %s%n", promptName, info.getServerName());
                    continue;
                }

                promptRegistry.put(promptName, client);
                promptMeta.put(promptName, prompt);

                // Build a synthetic tool definition so the LLM can call it
                String syntheticName = GET_PROMPT_PREFIX + promptName;
                String desc = String.format(
                    "Retrieve the '%s' prompt template from the MCP server. %s",
                    promptName,
                    prompt.path("description").asText("")
                );

                // Build an input schema from the prompt's declared arguments (if any)
                ObjectNode schema     = mapper.createObjectNode();
                ObjectNode properties = mapper.createObjectNode();
                List<String> required = new ArrayList<>();

                JsonNode declaredArgs = prompt.path("arguments");
                if (declaredArgs.isArray()) {
                    for (JsonNode arg : declaredArgs) {
                        String argName = arg.path("name").asText();
                        String argDesc = arg.path("description").asText("");
                        boolean isRequired = arg.path("required").asBoolean(false);

                        ObjectNode argSchema = mapper.createObjectNode();
                        argSchema.put("type", "string");
                        if (!argDesc.isBlank()) argSchema.put("description", argDesc);
                        properties.set(argName, argSchema);

                        if (isRequired) required.add(argName);
                    }
                }

                schema.put("type", "object");
                schema.set("properties", properties);
                if (!required.isEmpty()) {
                    schema.set("required", mapper.valueToTree(required));
                }

                // Register synthetic tool so the LLM can invoke it
                toolRegistry.put(syntheticName, client); // won't be used (dispatch intercepts)
                toolDefinitions.add(new LLMProvider.ToolDefinition(syntheticName, desc, schema));
            }
        }
    }

    /**
     * Build the system prompt that makes the LLM aware of prompts and resources.
     *
     * This is injected as a "developer" role message on every API call.
     * Without this the LLM has no idea prompts or resources exist at all.
     */
    private String buildSystemPrompt() {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a helpful AI assistant with access to MCP (Model Context Protocol) servers.\n\n");

        // ── Tools section (the LLM already sees these via the API, but summarise) ──
        List<String> toolNames = new ArrayList<>(toolRegistry.keySet());
        // Filter out synthetic prompt-fetch tools from this section
        toolNames.removeIf(n -> n.startsWith(GET_PROMPT_PREFIX));

        if (!toolNames.isEmpty()) {
            sb.append("## Available Tools\n");
            sb.append("You have the following real MCP tools you can call directly:\n");
            for (String name : toolNames) {
                LLMProvider.ToolDefinition def = toolDefinitions.stream()
                    .filter(t -> t.name().equals(name)).findFirst().orElse(null);
                sb.append("- **").append(name).append("**");
                if (def != null && !def.description().isBlank()) {
                    // First line of description only for brevity
                    String firstLine = def.description().split("\n")[0].trim();
                    sb.append(": ").append(firstLine);
                }
                sb.append("\n");
            }
            sb.append("\n");
        }

        // ── Prompts section ───────────────────────────────────────────────────
        if (!promptRegistry.isEmpty()) {
            sb.append("## Available Prompt Templates\n");
            sb.append("The following prompt templates are available on the MCP server. ");
            sb.append("To use one, call the corresponding `get_prompt__<name>` tool ");
            sb.append("which will return the rendered template text.\n\n");
            for (Map.Entry<String, JsonNode> e : promptMeta.entrySet()) {
                String   pName = e.getKey();
                JsonNode meta  = e.getValue();
                sb.append("- **").append(pName).append("**");
                String desc = meta.path("description").asText("").trim();
                if (!desc.isBlank()) sb.append(": ").append(desc.split("\n")[0].trim());
                sb.append("\n");
                sb.append("  → Call tool: `").append(GET_PROMPT_PREFIX).append(pName).append("`\n");

                // List arguments if any
                JsonNode args = meta.path("arguments");
                if (args.isArray() && args.size() > 0) {
                    sb.append("  → Arguments: ");
                    List<String> argList = new ArrayList<>();
                    for (JsonNode arg : args) {
                        String req = arg.path("required").asBoolean(false) ? " (required)" : " (optional)";
                        argList.add(arg.path("name").asText() + req);
                    }
                    sb.append(String.join(", ", argList)).append("\n");
                }
            }
            sb.append("\n");
        }

        // ── Resources section ─────────────────────────────────────────────────
        if (!resourceRegistry.isEmpty()) {
            sb.append("## Available Resources\n");
            sb.append("The following data resources are available. ");
            sb.append("Ask the user if they want to read any of them, or read them ");
            sb.append("proactively if relevant to the question.\n");
            // Group by server
            for (MCPServerInfo info : discoveredServers) {
                for (JsonNode r : info.getResources()) {
                    String uri  = r.path("uri").asText(r.path("name").asText());
                    String desc = r.path("description").asText("").trim();
                    sb.append("- `").append(uri).append("`");
                    if (!desc.isBlank()) sb.append(" – ").append(desc.split("\n")[0]);
                    sb.append("\n");
                }
            }
            sb.append("\n");
        }

        sb.append("When asked what you can do, always mention ALL of the above — ");
        sb.append("tools, prompt templates, and resources. ");
        sb.append("Use them proactively when they are relevant to the user's request.");

        return sb.toString();
    }
}
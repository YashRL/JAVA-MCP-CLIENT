package com.example;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.*;

/**
 * The agent runtime ties everything together.
 *
 * Flow:
 *  1. Accepts any number of MCPClient instances
 *  2. Discovers ALL capabilities from every server (tools + resources + prompts)
 *  3. Builds a unified tool registry (tool name → MCPClient that owns it)
 *  4. Runs an agentic loop against any LLMProvider:
 *       a. Send user question + all tools to the LLM
 *       b. If the LLM wants tool calls → execute them via MCP → send results back
 *       c. Repeat until the LLM produces a final text answer
 *
 * The design is deliberately LLM-agnostic: swap OpenAIProvider for a
 * ClaudeProvider, GeminiProvider, etc. and the rest works unchanged.
 */
public class AgentRuntime {

    private final LLMProvider llmProvider;

    // ── Discovery state ────────────────────────────────────────────────────
    private final List<MCPServerInfo>              discoveredServers = new ArrayList<>();
    private final Map<String, MCPClient>           toolRegistry      = new LinkedHashMap<>();
    private final Map<String, MCPClient>           resourceRegistry  = new LinkedHashMap<>();
    private final List<LLMProvider.ToolDefinition> toolDefinitions   = new ArrayList<>();

    // ── Constructor ────────────────────────────────────────────────────────

    /**
     * @param clients     all MCPClient instances (each will be discovered)
     * @param llmProvider the LLM to use (OpenAIProvider, etc.)
     */
    public AgentRuntime(List<MCPClient> clients, LLMProvider llmProvider) throws Exception {
        this.llmProvider = llmProvider;
        discoverAll(clients);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Run an agentic question → answer loop.
     *
     * The loop continues until the LLM returns a final text response
     * (i.e. makes no more tool calls).
     *
     * @param userQuestion the natural-language question from the user
     * @return the final answer produced by the LLM
     */
    public String run(String userQuestion) throws Exception {
        List<LLMProvider.Message> messages = List.of(
            new LLMProvider.Message("user", userQuestion)
        );

        // ── Turn 1: ask the LLM, it may want to call tools ────────────────
        LLMProvider.LLMResponse response = llmProvider.chat(messages, toolDefinitions);

        // ── Agentic loop: keep executing tools until no more are requested ─
        int maxIterations = 10; // safety valve
        int iteration     = 0;

        while (!response.toolCalls().isEmpty() && iteration < maxIterations) {
            iteration++;
            System.out.printf("%n[Agent] Turn %d – executing %d tool call(s)%n",
                iteration, response.toolCalls().size());

            List<LLMProvider.ToolResult> toolResults = new ArrayList<>();

            for (LLMProvider.ToolCall call : response.toolCalls()) {
                System.out.printf("[Agent]   → Calling tool '%s'%n", call.toolName());

                MCPClient server = toolRegistry.get(call.toolName());
                if (server == null) {
                    throw new RuntimeException(
                        "Unknown tool requested by LLM: " + call.toolName());
                }

                String result = server.callTool(call.toolName(), call.arguments());
                System.out.printf("[Agent]   ← Result (%d chars)%n", result.length());

                toolResults.add(new LLMProvider.ToolResult(call.callId(), result));
            }

            // Feed all results back to the LLM for the next turn
            response = llmProvider.continueWithToolResults(
                messages,
                toolResults,
                response.rawOutputItems(),
                toolDefinitions
            );
        }

        if (response.finalText() == null || response.finalText().isBlank()) {
            throw new RuntimeException(
                "Agent loop ended but LLM produced no final text.");
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
            System.out.println("  Capabilities: " + info.getCapabilities());

            if (!info.getTools().isEmpty()) {
                System.out.println("  Tools:");
                for (JsonNode t : info.getTools()) {
                    System.out.printf("    %-30s %s%n",
                        t.path("name").asText(),
                        t.path("description").asText(""));
                }
            }
            if (!info.getResources().isEmpty()) {
                System.out.println("  Resources:");
                for (JsonNode r : info.getResources()) {
                    System.out.printf("    %-30s %s%n",
                        r.path("uri").asText(r.path("name").asText()),
                        r.path("description").asText(""));
                }
            }
            if (!info.getPrompts().isEmpty()) {
                System.out.println("  Prompts:");
                for (JsonNode p : info.getPrompts()) {
                    System.out.printf("    %-30s %s%n",
                        p.path("name").asText(),
                        p.path("description").asText(""));
                }
            }
            System.out.println();
        }
        System.out.println("══════════════════════════════════════════════");
        System.out.printf("  Total tools registered: %d%n", toolRegistry.size());
        System.out.println("══════════════════════════════════════════════\n");
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    public List<MCPServerInfo>              getDiscoveredServers() { return discoveredServers; }
    public List<LLMProvider.ToolDefinition> getToolDefinitions()   { return toolDefinitions;  }
    public Map<String, MCPClient>           getToolRegistry()       { return toolRegistry;     }

    // ── Private helpers ────────────────────────────────────────────────────

    /**
     * Run self-discovery on every MCPClient and build the unified registries.
     */
    private void discoverAll(List<MCPClient> clients) throws Exception {
        for (MCPClient client : clients) {
            MCPServerInfo info = client.discover();
            discoveredServers.add(info);

            // Register tools
            for (JsonNode tool : info.getTools()) {
                String name = tool.path("name").asText();

                if (toolRegistry.containsKey(name)) {
                    System.out.printf("[WARNING] Tool '%s' already registered from " +
                        "another server – skipping duplicate from %s%n",
                        name, info.getServerName());
                    continue;
                }

                toolRegistry.put(name, client);

                // Build the LLM-facing tool definition
                String   desc        = tool.path("description").asText("");
                JsonNode inputSchema = tool.has("inputSchema")
                                       ? tool.get("inputSchema")
                                       : null;
                toolDefinitions.add(
                    new LLMProvider.ToolDefinition(name, desc, inputSchema));
            }

            // Register resources by URI for later lookup
            for (JsonNode resource : info.getResources()) {
                String uri = resource.path("uri").asText(
                             resource.path("name").asText());
                resourceRegistry.put(uri, client);
            }
        }
    }
}
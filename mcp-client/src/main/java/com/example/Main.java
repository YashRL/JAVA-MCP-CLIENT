package com.example;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;

/**
 * Entry point for the MCP Java Agent.
 *
 * What happens when you run this:
 *  1. Three MCPClient instances are created (one per server endpoint)
 *  2. AgentRuntime.discoverAll() connects to each server and runs the full
 *     MCP handshake + discovery (tools, resources, prompts)
 *  3. A summary of everything discovered is printed
 *  4. The agent runs your question through the LLM ↔ MCP agentic loop
 *  5. The final answer is printed
 */
public class Main {

    public static void main(String[] args) throws Exception {

        // ── Configuration ────────────────────────────────────────────────────
        String openAiKey = System.getenv("OPENAI_API_KEY");
        if (openAiKey == null || openAiKey.isBlank()) {
            throw new IllegalStateException(
                "OPENAI_API_KEY environment variable is not set.");
        }

        // ── Shared HTTP client ───────────────────────────────────────────────
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        // ── MCP Clients (one per server endpoint) ────────────────────────────
        MCPClient general = new MCPClient(
            "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/general/mcp",
            httpClient
        );
        MCPClient content = new MCPClient(
            "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/content_retrival/mcp",
            httpClient
        );
        MCPClient talentbot = new MCPClient(
            "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/talentbot/mcp",
            httpClient
        );

        // ── LLM Provider (OpenAI via raw HTTP – swap for any other provider) ─
        LLMProvider llm = new OpenAIProvider(
            "gpt-4o",       // model name
            openAiKey,
            httpClient
        );

        // ── Agent Runtime (self-discovery happens here) ──────────────────────
        AgentRuntime agent = new AgentRuntime(
            List.of(general, content, talentbot),
            llm
        );

        // Print a full summary of everything discovered from all servers
        agent.printDiscoverySummary();

        // ── Run a query ───────────────────────────────────────────────────────
        String question = "Hii How many tools and prompts do you have? What are they? Can you give me a  very very very short summary of each?";
        System.out.println("Question: " + question);
        System.out.println("─".repeat(60));

        String answer = agent.run(question);

        System.out.println("\n══════════════════════════════════════════════");
        System.out.println("FINAL ANSWER:");
        System.out.println("══════════════════════════════════════════════");
        System.out.println(answer);
    }
}
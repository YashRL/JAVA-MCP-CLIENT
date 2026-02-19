package com.example.agentic_setup.mcp;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;

public class Example_Usecase_MCP {

    public static void main(String[] args) throws Exception {

        var apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isBlank())
            throw new IllegalStateException("OPENAI_API_KEY not set.");

        // ── Single model, wired once, shared everywhere ───────────────────
        var http    = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
        var llm     = new OpenAIProvider("gpt-4.1", apiKey, http);
        var planner = new Planner(PlanningMode.SHORT, llm);

        // ── Debug flags ───────────────────────────────────────────────────
        AgentRuntime.DEBUG_MCP    = true;
        AgentRuntime.DEBUG_PROMPT = true;

        // ── MCP servers ───────────────────────────────────────────────────
        var base = "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com";
        var agent = new AgentRuntime(List.of(
                new MCPClient(base + "/general/mcp",          http),
                new MCPClient(base + "/content_retrival/mcp", http),
                new MCPClient(base + "/talentbot/mcp",        http)
        ), llm, planner);

        agent.printSummary();

        // ── Run ───────────────────────────────────────────────────────────
        var question = """
                I have a meeting to prepare for. Help me gather insights and data points.
                Topic: latest trends in content retrieval AI.
                Meeting type: decision + strategy + technical deep-dive.
                Audience: AIML Engineers, FrontEnd Developers, Product Managers.
                Duration: 1 hour.
                Primary use case: enterprise docs / RAG.
                No follow-up questions please.
                """;

        System.out.println("Question:\n" + question);
        System.out.println("─".repeat(60));

        var answer = agent.run(question);

        System.out.println("\n══════════════════════════════════");
        System.out.println("FINAL ANSWER:");
        System.out.println("══════════════════════════════════");
        System.out.println(answer);
    }
}
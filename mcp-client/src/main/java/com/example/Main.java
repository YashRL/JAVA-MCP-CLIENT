package com.example;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {

        // ── Configuration ────────────────────────────────────────────────────
        String openAiKey = System.getenv("OPENAI_API_KEY");
        if (openAiKey == null || openAiKey.isBlank())
            throw new IllegalStateException("OPENAI_API_KEY environment variable is not set.");

        // ── Planning mode ────────────────────────────────────────────────────
        // SHORT: no planner call, executor self-directs with guardrails (fastest)
        // MID  : cheap planner call gives budget+intent, executor self-directs (balanced)
        // LONG : full planner with step list + per-step reflection (slowest, best quality)
        PlanningMode mode = PlanningMode.SHORT;

        // ── Debug flags ──────────────────────────────────────────────────────
        AgentRuntime.DEBUG_MCP_RAW    = true;  // print raw data from MCP server calls
        AgentRuntime.DEBUG_LLM_PROMPT = true;  // print system prompt sent to LLM

        // ── Shared HTTP client ───────────────────────────────────────────────
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();

        // ── MCP Clients ───────────────────────────────────────────────────────
        MCPClient general = new MCPClient(
            "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/general/mcp", httpClient);
        MCPClient content = new MCPClient(
            "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/content_retrival/mcp", httpClient);
        MCPClient talentbot = new MCPClient(
            "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/talentbot/mcp", httpClient);

        // ── LLM Provider ─────────────────────────────────────────────────────
        LLMProvider llm = new OpenAIProvider("gpt-5.2", openAiKey, httpClient);

        // ── Planner ───────────────────────────────────────────────────────────
        // SHORT mode: planner is created but plan() is a no-op (no LLM call)
        // MID mode  : uses gpt-4.1-mini (cheap, fast)
        // LONG mode : uses gpt-4.1 (full quality)
        Planner planner = new Planner(mode, "gpt-4.1", openAiKey, httpClient);

        // ── Agent Runtime ─────────────────────────────────────────────────────
        AgentRuntime agent = new AgentRuntime(List.of(general, content, talentbot), llm, planner);
        agent.printDiscoverySummary();

        // ── Run ───────────────────────────────────────────────────────────────
        String question = "Hii I have a meeting to prepare for. Can you help me gather some information? "
            + "and I want to have some insights and data points to share with my team."
            + "The meeting is about the latest trends in content retrieval AI, **Meeting type**: decision + strategy + technical deep-dive, Audience: AIML Engineers, FrontEnd Developers and Product Managers, Duration: 1 hour. , "
            + "Primary use case** (pick one or describe): enterprise docs/RAG"
            + "and no more questions please";
        System.out.println("Question: " + question);
        System.out.println("─".repeat(60));

        String answer = agent.run(question);

        System.out.println("\n══════════════════════════════════════════════");
        System.out.println("FINAL ANSWER:");
        System.out.println("══════════════════════════════════════════════");
        System.out.println(answer);
    }
}
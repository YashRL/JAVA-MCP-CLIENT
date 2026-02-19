package com.example.agentic_setup.config;

import com.example.agentic_setup.mcp.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;

/**
 * Wires the entire agent stack as Spring beans.
 * Change model or MCP server URLs here — nowhere else.
 */
@Configuration
public class AgentConfig {

    @Value("${openai.api-key:#{systemEnvironment['OPENAI_API_KEY']}}")
    private String apiKey;

    @Value("${openai.model:gpt-4.1}")
    private String model;

    @Value("${mcp.base-url:https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com}")
    private String mcpBase;

    @Value("${agent.planning-mode:SHORT}")
    private String planningMode;

    @Bean
    HttpClient httpClient() {
        return HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    @Bean
    LLMProvider llmProvider(HttpClient httpClient) {
        return new OpenAIProvider(model, apiKey, httpClient);
    }

    @Bean
    Planner planner(LLMProvider llm) {
        return new Planner(PlanningMode.valueOf(planningMode.toUpperCase()), llm);
    }

    @Bean
    AgentRuntime agentRuntime(HttpClient httpClient, LLMProvider llm, Planner planner)
            throws Exception {
        return new AgentRuntime(List.of(
                new MCPClient(mcpBase + "/general/mcp",          httpClient),
                new MCPClient(mcpBase + "/content_retrival/mcp", httpClient),
                new MCPClient(mcpBase + "/talentbot/mcp",        httpClient)
        ), llm, planner);
    }
}
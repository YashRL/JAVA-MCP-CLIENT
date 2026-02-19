package com.example.agentic_setup.service;

import com.example.agentic_setup.mcp.AgentRuntime;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import okhttp3.sse.*;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Runs the agent ReAct loop then streams the final synthesis call token-by-token.
 *
 * Flow:
 *   1. agentRuntime.run(prompt)  → blocking ReAct loop (planner, tools, reflection)
 *                                   returns SynthesisInput (systemPrompt + userMessage)
 *   2. streamSynthesis(input)    → OkHttp SSE to OpenAI, each delta token → onToken
 */
@Service
public class AgentStreamingService {

    private static final String OPENAI_URL = "https://api.openai.com/v1/responses";

    private final AgentRuntime agent;
    private final String       apiKey;
    private final String       model;
    private final OkHttpClient http;
    private final ObjectMapper mapper = new ObjectMapper();

    public AgentStreamingService(AgentRuntime agent,
                                 @org.springframework.beans.factory.annotation.Value("${openai.api-key:#{systemEnvironment['OPENAI_API_KEY']}}") String apiKey,
                                 @org.springframework.beans.factory.annotation.Value("${openai.model:gpt-4.1}") String model) {
        this.agent  = agent;
        this.apiKey = apiKey;
        this.model  = model;
        this.http   = new OkHttpClient.Builder()
                .connectTimeout(0, TimeUnit.MILLISECONDS)
                .readTimeout(0, TimeUnit.MILLISECONDS)
                .writeTimeout(0, TimeUnit.MILLISECONDS)
                .build();
    }

    /**
     * Main entry point called by WebSocket handler and SSE controller.
     *
     * @param prompt   user question
     * @param onToken  callback receiving each streamed token (and "\n[DONE]" at end)
     */
    public void runAndStream(String prompt, Consumer<String> onToken) throws Exception {

        // ── Step 1: Run the agent loop (blocking) ─────────────────────────
        System.out.println("[Service] Starting agent loop for: " + prompt);
        var synthesis = agent.run(prompt);

        // ── Step 2: If trivial (no LLM needed), just emit the direct answer
        if (synthesis.systemPrompt() == null) {
            onToken.accept(synthesis.userMessage());
            onToken.accept("\n[DONE]");
            return;
        }

        // ── Step 3: Stream the synthesis call ─────────────────────────────
        System.out.println("[Service] Streaming synthesis...");
        streamSynthesis(synthesis.systemPrompt(), synthesis.userMessage(), onToken);
    }

    // ── Streaming synthesis call ──────────────────────────────────────────

    private void streamSynthesis(String systemPrompt, String userMessage,
                                  Consumer<String> onToken) throws InterruptedException {

        // Build the request body using Jackson so special chars are escaped correctly
        var input = mapper.createArrayNode();
        input.add(mapper.createObjectNode().put("role", "developer").put("content", systemPrompt));
        input.add(mapper.createObjectNode().put("role", "user").put("content", userMessage));

        var body = mapper.createObjectNode()
                .put("model",  model)
                .put("stream", true);
        body.set("input", input);

        String bodyStr;
        try { bodyStr = mapper.writeValueAsString(body); }
        catch (Exception e) { throw new RuntimeException("Failed to serialise request body", e); }

        var request = new Request.Builder()
                .url(OPENAI_URL)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type",  "application/json")
                .addHeader("Accept",        "text/event-stream")
                .post(RequestBody.create(bodyStr, MediaType.parse("application/json")))
                .build();

        // Use a latch so we block until the stream is fully consumed
        var latch = new java.util.concurrent.CountDownLatch(1);

        EventSources.createFactory(http).newEventSource(request, new EventSourceListener() {

            @Override
            public void onEvent(EventSource source, String id, String type, String data) {
                if ("[DONE]".equals(data)) {
                    onToken.accept("\n[DONE]");
                    source.cancel();
                    latch.countDown();
                    return;
                }
                try {
                    var json = mapper.readTree(data);
                    if ("response.output_text.delta".equals(json.path("type").asText())) {
                        onToken.accept(json.path("delta").asText());
                    }
                } catch (Exception ignored) {}
            }

            @Override
            public void onFailure(EventSource source, Throwable t, Response response) {
                System.err.println("[Service] Stream failure: "
                        + (t != null ? t.getMessage() : response != null ? response.code() : "unknown"));
                onToken.accept("\n[ERROR] Stream failed.");
                source.cancel();
                latch.countDown();
            }
        });

        latch.await(); // block until streaming completes or fails
    }
}
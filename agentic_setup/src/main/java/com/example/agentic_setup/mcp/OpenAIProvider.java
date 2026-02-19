package com.example.agentic_setup.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

/**
 * OpenAI implementation using the Responses API (/v1/responses).
 * No SDK required — plain HTTP.
 */
public class OpenAIProvider implements LLMProvider {

    private static final String API_URL = "https://api.openai.com/v1/responses";

    private final String     model;
    private final String     apiKey;
    private final HttpClient http;
    private final ObjectMapper mapper = new ObjectMapper();

    public OpenAIProvider(String model, String apiKey, HttpClient http) {
        this.model  = model;
        this.apiKey = apiKey;
        this.http   = http;
    }

    // ── LLMProvider ───────────────────────────────────────────────────────

    @Override
    public LLMResponse chat(String systemPrompt, List<Message> messages,
                            List<ToolDefinition> tools) throws Exception {
        var body = buildBody(systemPrompt, messages, tools);
        return parse(call(body));
    }

    @Override
    public LLMResponse continueWithToolResults(String systemPrompt, List<Message> messages,
            List<ToolResult> results, List<Object> previousItems,
            List<ToolDefinition> tools) throws Exception {

        var input = mapper.createArrayNode();
        appendSystem(input, systemPrompt);
        messages.forEach(m -> input.add(obj("role", m.role(), "content", m.content())));
        previousItems.forEach(i -> input.add((JsonNode) i));
        results.forEach(r -> input.add(mapper.createObjectNode()
                .put("type",    "function_call_output")
                .put("call_id", r.callId())
                .put("output",  r.result())));

        var body = baseBody(tools);
        body.set("input", input);
        return parse(call(body));
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private ObjectNode buildBody(String systemPrompt, List<Message> messages,
                                 List<ToolDefinition> tools) {
        var input = mapper.createArrayNode();
        appendSystem(input, systemPrompt);
        messages.forEach(m -> input.add(obj("role", m.role(), "content", m.content())));
        var body = baseBody(tools);
        body.set("input", input);
        return body;
    }

    private ObjectNode baseBody(List<ToolDefinition> tools) {
        var body = mapper.createObjectNode().put("model", model).put("tool_choice", "auto");
        var arr  = mapper.createArrayNode();
        tools.forEach(t -> arr.add(mapper.createObjectNode()
                .put("type",        "function")
                .put("name",        t.name())
                .put("description", t.description() != null ? t.description() : "")
                .set("parameters",  t.inputSchema() != null
                        ? t.inputSchema() : mapper.createObjectNode())));
        body.set("tools", arr);
        return body;
    }

    private void appendSystem(ArrayNode input, String systemPrompt) {
        if (systemPrompt != null && !systemPrompt.isBlank())
            input.add(obj("role", "developer", "content", systemPrompt));
    }

    private ObjectNode obj(String k1, String v1, String k2, String v2) {
        return mapper.createObjectNode().put(k1, v1).put(k2, v2);
    }

    private JsonNode call(ObjectNode body) throws Exception {
        var req = HttpRequest.newBuilder()
                .uri(URI.create(API_URL))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type",  "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                .build();

        var res  = http.send(req, HttpResponse.BodyHandlers.ofString());
        var root = mapper.readTree(res.body());

        if (root.has("error") && !root.get("error").isNull())
            throw new RuntimeException("OpenAI error: " + root.get("error").toPrettyString());
        if (!root.has("output"))
            throw new RuntimeException("No 'output' in response: " + res.body());
        return root;
    }

    private LLMResponse parse(JsonNode root) throws Exception {
        List<ToolCall> calls    = new ArrayList<>();
        List<Object>   rawItems = new ArrayList<>();
        String         text     = null;

        for (var item : root.get("output")) {
            rawItems.add(item);
            var type = item.path("type").asText();
            if ("function_call".equals(type)) {
                calls.add(new ToolCall(
                        item.path("call_id").asText(),
                        item.path("name").asText(),
                        mapper.readTree(item.path("arguments").asText("{}"))));
            } else if ("message".equals(type)) {
                var content = item.path("content");
                if (content.isArray() && !content.isEmpty())
                    text = content.get(0).path("text").asText();
            }
        }
        return new LLMResponse(text, calls, rawItems);
    }
}
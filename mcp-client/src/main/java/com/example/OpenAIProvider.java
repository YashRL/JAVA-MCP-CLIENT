package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.*;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

/**
 * OpenAI implementation of LLMProvider using the Responses API
 * (https://api.openai.com/v1/responses) via raw HTTP – no SDK required.
 *
 * Why the Responses API instead of /v1/chat/completions?
 *   The Responses API preserves the original function_call output items in
 *   the response, making multi-turn tool replay straightforward.
 *   You can swap this for another model by implementing LLMProvider differently.
 */
public class OpenAIProvider implements LLMProvider {

    private static final String RESPONSES_URL = "https://api.openai.com/v1/responses";

    private final String     model;      // e.g. "gpt-4.1", "gpt-4-turbo"
    private final String     apiKey;
    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();

    public OpenAIProvider(String model, String apiKey, HttpClient httpClient) {
        this.model      = model;
        this.apiKey     = apiKey;
        this.httpClient = httpClient;
    }

    // ── LLMProvider interface ─────────────────────────────────────────────────

    @Override
    public LLMResponse chat(String systemPrompt,
                            List<Message> messages,
                            List<ToolDefinition> tools) throws Exception {

        ObjectNode body = buildRequestBody(systemPrompt, messages, tools);
        JsonNode root   = callAPI(body);
        return parseResponse(root);
    }

    @Override
    public LLMResponse continueWithToolResults(
            String           systemPrompt,
            List<Message>    messages,
            List<ToolResult> toolResults,
            List<Object>     previousOutputItems,
            List<ToolDefinition> tools) throws Exception {

        ObjectNode body  = mapper.createObjectNode();
        body.put("model",       model);
        body.put("tool_choice", "auto");
        body.set("tools",       buildOpenAITools(tools));

        ArrayNode input = mapper.createArrayNode();

        // System prompt goes first as a "developer" role message
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            input.add(mapper.createObjectNode()
                    .put("role",    "developer")
                    .put("content", systemPrompt));
        }

        for (Message m : messages) {
            input.add(mapper.createObjectNode()
                    .put("role",    m.role())
                    .put("content", m.content()));
        }

        // Re-append the assistant's function_call output items from the last turn
        for (Object raw : previousOutputItems) {
            input.add((JsonNode) raw);
        }

        // Append each tool result
        for (ToolResult tr : toolResults) {
            ObjectNode resultNode = mapper.createObjectNode();
            resultNode.put("type",    "function_call_output");
            resultNode.put("call_id", tr.callId());
            resultNode.put("output",  tr.result());
            input.add(resultNode);
        }

        body.set("input", input);

        JsonNode root = callAPI(body);
        return parseResponse(root);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private ObjectNode buildRequestBody(String systemPrompt,
                                        List<Message> messages,
                                        List<ToolDefinition> tools) {
        ObjectNode body = mapper.createObjectNode();
        body.put("model",       model);
        body.put("tool_choice", "auto");
        body.set("tools",       buildOpenAITools(tools));

        ArrayNode input = mapper.createArrayNode();

        // Inject system prompt as a "developer" role message (Responses API convention)
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            input.add(mapper.createObjectNode()
                    .put("role",    "developer")
                    .put("content", systemPrompt));
        }

        for (Message m : messages) {
            input.add(mapper.createObjectNode()
                    .put("role",    m.role())
                    .put("content", m.content()));
        }
        body.set("input", input);
        return body;
    }

    /**
     * Convert generic ToolDefinition list into the OpenAI tool schema format.
     *
     * OpenAI Responses API expects:
     * {
     *   "type": "function",
     *   "name": "...",
     *   "description": "...",
     *   "parameters": { ... JSON Schema ... }
     * }
     */
    private ArrayNode buildOpenAITools(List<ToolDefinition> tools) {
        ArrayNode arr = mapper.createArrayNode();
        for (ToolDefinition td : tools) {
            ObjectNode tool = mapper.createObjectNode();
            tool.put("type",        "function");
            tool.put("name",        td.name());
            tool.put("description", td.description() != null ? td.description() : "");
            tool.set("parameters",  td.inputSchema() != null
                                    ? td.inputSchema()
                                    : mapper.createObjectNode());
            arr.add(tool);
        }
        return arr;
    }

    private JsonNode callAPI(ObjectNode body) throws Exception {
        String bodyStr = mapper.writeValueAsString(body);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(RESPONSES_URL))
                .header("Authorization",  "Bearer " + apiKey)
                .header("Content-Type",   "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(bodyStr))
                .build();

        HttpResponse<String> httpResponse =
                httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        JsonNode root = mapper.readTree(httpResponse.body());

        if (root.has("error") && !root.get("error").isNull()) {
            throw new RuntimeException(
                "OpenAI API error: " + root.get("error").toPrettyString());
        }
        if (!root.has("output")) {
            throw new RuntimeException(
                "No 'output' field in OpenAI response:\n" + httpResponse.body());
        }

        return root;
    }

    /**
     * Parse the OpenAI Responses API output array into our neutral LLMResponse.
     *
     * Output items can be:
     *   { "type": "message", "content": [{ "type": "text", "text": "..." }] }
     *   { "type": "function_call", "name": "...", "arguments": "...", "call_id": "..." }
     */
    private LLMResponse parseResponse(JsonNode root) throws Exception {
        List<ToolCall>  toolCalls       = new ArrayList<>();
        List<Object>    rawOutputItems  = new ArrayList<>();
        String          finalText       = null;

        for (JsonNode item : root.get("output")) {
            rawOutputItems.add(item); // preserve for multi-turn replay

            String type = item.path("type").asText();

            if ("function_call".equals(type)) {
                String   callId    = item.path("call_id").asText();
                String   name      = item.path("name").asText();
                String   argsStr   = item.path("arguments").asText("{}");
                JsonNode arguments = mapper.readTree(argsStr);
                toolCalls.add(new ToolCall(callId, name, arguments));
            }

            if ("message".equals(type)) {
                JsonNode content = item.path("content");
                if (content.isArray() && content.size() > 0) {
                    finalText = content.get(0).path("text").asText();
                }
            }
        }

        return new LLMResponse(finalText, toolCalls, rawOutputItems);
    }
}
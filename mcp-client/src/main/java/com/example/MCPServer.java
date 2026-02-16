package com.example;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

public class MCPServer {

    private final String url;
    private final HttpClient client;
    private final ObjectMapper mapper = new ObjectMapper();

    private String sessionId;

    public MCPServer(String url, HttpClient client) {
        this.url = url;
        this.client = client;
    }

    private void initialize() throws Exception {

        if (sessionId != null) return;

        String initBody = """
        {
          "jsonrpc": "2.0",
          "id": 1,
          "method": "initialize",
          "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
              "name": "java-agent",
              "version": "1.0"
            }
          }
        }
        """;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(initBody))
                .build();

        HttpResponse<String> response =
                client.send(request, HttpResponse.BodyHandlers.ofString());

        sessionId = response.headers()
                .firstValue("MCP-Session-Id")
                .orElseThrow(() -> new RuntimeException("No MCP Session ID"));

        // Send initialized notification
        String initializedBody = """
        {
          "jsonrpc": "2.0",
          "method": "initialized",
          "params": {}
        }
        """;

        HttpRequest initReq = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .header("MCP-Session-Id", sessionId)
                .POST(HttpRequest.BodyPublishers.ofString(initializedBody))
                .build();

        client.send(initReq, HttpResponse.BodyHandlers.ofString());
    }

    // 🔥 LIST TOOLS
    public JsonNode listTools() throws Exception {

        initialize();

        String body = """
        {
          "jsonrpc": "2.0",
          "id": 2,
          "method": "tools/list",
          "params": {}
        }
        """;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .header("MCP-Session-Id", sessionId)
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response =
                client.send(request, HttpResponse.BodyHandlers.ofString());

        JsonNode json = parseSSE(response.body());

        return json.get("result").get("tools");
    }

    // 🔥 GET SCHEMA FOR A TOOL
    public JsonNode getToolSchema(String toolName) throws Exception {

        JsonNode tools = listTools();

        for (JsonNode tool : tools) {
            if (tool.get("name").asText().equals(toolName)) {
                return tool;
            }
        }

        throw new RuntimeException("Tool schema not found for: " + toolName);
    }

    // 🔥 CALL TOOL
    public String callTool(String toolName, JsonNode arguments) throws Exception {

        initialize();

        ObjectNode params = mapper.createObjectNode();
        params.put("name", toolName);
        params.set("arguments", arguments);

        ObjectNode body = mapper.createObjectNode();
        body.put("jsonrpc", "2.0");
        body.put("id", 99);
        body.put("method", "tools/call");
        body.set("params", params);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .header("MCP-Session-Id", sessionId)
                .POST(HttpRequest.BodyPublishers.ofString(
                        mapper.writeValueAsString(body)))
                .build();

        HttpResponse<String> response =
                client.send(request, HttpResponse.BodyHandlers.ofString());

        JsonNode json = parseSSE(response.body());

        return json.get("result")
                .get("content")
                .get(0)
                .get("text")
                .asText();
    }

    // 🔥 HANDLE SSE RESPONSE CLEANLY
    private JsonNode parseSSE(String raw) throws Exception {

        if (raw.contains("data:")) {
            raw = raw.substring(raw.indexOf("data:") + 5).trim();
        }

        return mapper.readTree(raw);
    }
}

package com.example.agentic_setup.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MCP protocol client (HTTP + SSE, protocol version 2024-11-05).
 *
 * Lifecycle: initialize → initialized notification → list tools/resources/prompts.
 * Then exposes callTool(), readResource(), getPrompt() for the agent loop.
 */
public class MCPClient {

    private final String     serverUrl;
    private final HttpClient http;
    private final ObjectMapper mapper  = new ObjectMapper();
    private final AtomicInteger idSeq  = new AtomicInteger(1);

    private String        sessionId;
    private MCPServerInfo serverInfo;

    public MCPClient(String serverUrl, HttpClient http) {
        this.serverUrl = serverUrl;
        this.http      = http;
    }

    // ── Discovery ─────────────────────────────────────────────────────────

    /** Handshake + full capability discovery. Idempotent. */
    public MCPServerInfo discover() throws Exception {
        if (serverInfo != null) return serverInfo;

        serverInfo = initialize();
        System.out.printf("[MCP] Connected: %s%n", serverInfo);

        if (serverInfo.supportsTools())     serverInfo.addTools(listAll("tools/list", "tools"));
        if (serverInfo.supportsResources()) serverInfo.addResources(listAll("resources/list", "resources"));
        if (serverInfo.supportsPrompts())   serverInfo.addPrompts(listAll("prompts/list", "prompts"));

        System.out.printf("[MCP]   tools=%d  resources=%d  prompts=%d%n",
                serverInfo.getTools().size(), serverInfo.getResources().size(), serverInfo.getPrompts().size());
        return serverInfo;
    }

    // ── Primitives ────────────────────────────────────────────────────────

    public String callTool(String name, JsonNode args) throws Exception {
        var params = mapper.createObjectNode().put("name", name);
        params.set("arguments", args);
        var content = sendRequest("tools/call", params).path("result").path("content");
        if (!content.isArray()) throw new RuntimeException("tools/call: no content array");
        var sb = new StringBuilder();
        for (var block : content)
            if ("text".equals(block.path("type").asText("text"))) {
                if (sb.length() > 0) sb.append('\n');
                sb.append(block.path("text").asText());
            }
        return sb.toString();
    }

    public String readResource(String uri) throws Exception {
        var params = mapper.createObjectNode().put("uri", uri);
        var contents = sendRequest("resources/read", params).path("result").path("contents");
        if (!contents.isArray()) throw new RuntimeException("resources/read: no contents array");
        var sb = new StringBuilder();
        for (var item : contents)
            if (item.has("text")) { if (sb.length() > 0) sb.append('\n'); sb.append(item.get("text").asText()); }
        return sb.toString();
    }

    /** Returns the raw MCP messages array: [{role, content{type,text}}, ...] */
    public JsonNode getPrompt(String name, ObjectNode args) throws Exception {
        var params = mapper.createObjectNode().put("name", name);
        if (args != null) params.set("arguments", args);
        return sendRequest("prompts/get", params).path("result").path("messages");
    }

    // ── MCP handshake ─────────────────────────────────────────────────────

    private MCPServerInfo initialize() throws Exception {
        var caps = mapper.createObjectNode();
        caps.set("tools",     mapper.createObjectNode());
        caps.set("resources", mapper.createObjectNode());
        caps.set("prompts",   mapper.createObjectNode());

        var params = mapper.createObjectNode()
                .put("protocolVersion", "2024-11-05");
        params.set("capabilities", caps);
        params.set("clientInfo", mapper.createObjectNode()
                .put("name", "java-mcp-client").put("version", "1.0.0"));

        var res = httpPost(buildRpc("initialize", params), false);
        sessionId = res.headers().firstValue("Mcp-Session-Id")
                .or(() -> res.headers().firstValue("MCP-Session-Id"))
                .orElse(null);

        var json   = parseSSE(res.body());
        checkError(json, "initialize");
        var result = json.get("result");

        sendNotification("notifications/initialized");

        return new MCPServerInfo(
                serverUrl,
                result.path("serverInfo").path("name").asText("unknown"),
                result.path("serverInfo").path("version").asText("?"),
                result.path("capabilities"));
    }

    private List<JsonNode> listAll(String method, String key) throws Exception {
        var list   = new ArrayList<JsonNode>();
        String cursor = null;
        do {
            var params = mapper.createObjectNode();
            if (cursor != null) params.put("cursor", cursor);
            var result = sendRequest(method, params).path("result");
            result.path(key).forEach(list::add);
            cursor = result.has("nextCursor") && !result.get("nextCursor").isNull()
                    ? result.get("nextCursor").asText() : null;
        } while (cursor != null);
        return list;
    }

    // ── Transport ─────────────────────────────────────────────────────────

    private JsonNode sendRequest(String method, ObjectNode params) throws Exception {
        var res  = httpPost(buildRpc(method, params), true);
        var json = parseSSE(res.body());
        checkError(json, method);
        return json;
    }

    private void sendNotification(String method) throws Exception {
        var body = mapper.createObjectNode()
                .put("jsonrpc", "2.0").put("method", method);
        body.set("params", mapper.createObjectNode());
        httpPost(body, sessionId != null);
    }

    private HttpResponse<String> httpPost(JsonNode body, boolean withSession) throws Exception {
        var builder = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl))
                .header("Content-Type", "application/json")
                .header("Accept",       "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()));
        if (withSession && sessionId != null)
            builder.header("Mcp-Session-Id", sessionId);
        return http.send(builder.build(), HttpResponse.BodyHandlers.ofString());
    }

    private ObjectNode buildRpc(String method, ObjectNode params) {
        var node = mapper.createObjectNode()
                .put("jsonrpc", "2.0")
                .put("id",      idSeq.getAndIncrement())
                .put("method",  method);
        if (params != null) node.set("params", params);
        return node;
    }

    private JsonNode parseSSE(String raw) throws Exception {
        if (raw != null && raw.contains("data:"))
            raw = raw.substring(raw.indexOf("data:") + 5).trim();
        return mapper.readTree(raw);
    }

    private void checkError(JsonNode json, String method) {
        if (json.has("error") && !json.get("error").isNull()) {
            var err = json.get("error");
            throw new RuntimeException(String.format("MCP '%s' error [%d]: %s",
                    method, err.path("code").asInt(-1), err.path("message").asText("unknown")));
        }
    }

    public MCPServerInfo getServerInfo() { return serverInfo; }
}
package com.example;

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
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Low-level MCP protocol client.
 *
 * Responsibilities:
 *  1. Perform the MCP lifecycle handshake (initialize → initialized)
 *  2. Discover ALL server capabilities: tools, resources, prompts
 *  3. Expose primitives to call tools and read resources
 *
 * Transport: HTTP + SSE (protocol version 2024-11-05)
 *
 * The MCP specification JSON-RPC methods used here:
 *   initialize      – capability negotiation + session establishment
 *   initialized     – client acknowledgment notification (no response expected)
 *   tools/list      – enumerate available tools
 *   resources/list  – enumerate available resources
 *   prompts/list    – enumerate available prompts
 *   tools/call      – execute a tool
 *   resources/read  – read a resource by URI
 *   prompts/get     – retrieve a rendered prompt template
 */
public class MCPClient {

    // Every JSON-RPC message needs a unique id; use an incrementing counter
    private final AtomicInteger idCounter = new AtomicInteger(1);

    private final String serverUrl;
    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();

    // Session token returned by the server in the MCP-Session-Id header
    private String sessionId;

    // Fully-populated server info after discovery
    private MCPServerInfo serverInfo;

    // ── Constructor ──────────────────────────────────────────────────────────

    public MCPClient(String serverUrl, HttpClient httpClient) {
        this.serverUrl  = serverUrl;
        this.httpClient = httpClient;
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /**
     * Perform the full handshake AND discover all capabilities.
     * Call this once before using the client.
     *
     * @return a populated MCPServerInfo describing everything the server offers
     */
    public MCPServerInfo discover() throws Exception {
        if (serverInfo != null) return serverInfo; // idempotent

        // Step 1 – Initialize (capability negotiation)
        serverInfo = initialize();
        System.out.printf("[MCP] Connected to %s%n", serverInfo);

        // Step 2 – Discover each capability the server advertises
        if (serverInfo.supportsTools()) {
            List<JsonNode> tools = listAll("tools/list", "tools");
            serverInfo.addTools(tools);
            System.out.printf("[MCP]   ↳ %d tool(s)%n", tools.size());
        }
        if (serverInfo.supportsResources()) {
            List<JsonNode> resources = listAll("resources/list", "resources");
            serverInfo.addResources(resources);
            System.out.printf("[MCP]   ↳ %d resource(s)%n", resources.size());
        }
        if (serverInfo.supportsPrompts()) {
            List<JsonNode> prompts = listAll("prompts/list", "prompts");
            serverInfo.addPrompts(prompts);
            System.out.printf("[MCP]   ↳ %d prompt(s)%n", prompts.size());
        }

        return serverInfo;
    }

    /**
     * Execute a tool by name with a JSON arguments object.
     *
     * @return the plain-text result string from the tool's first content block
     */
    public String callTool(String toolName, JsonNode arguments) throws Exception {
        ensureInitialized();

        ObjectNode params = mapper.createObjectNode();
        params.put("name", toolName);
        params.set("arguments", arguments);

        JsonNode response = sendRequest("tools/call", params);

        // MCP spec: result.content is an array of content blocks
        // We return all text blocks joined together
        JsonNode content = response.path("result").path("content");
        if (content.isMissingNode() || !content.isArray()) {
            throw new RuntimeException("tools/call returned no content: " + response);
        }

        StringBuilder sb = new StringBuilder();
        for (JsonNode block : content) {
            String type = block.path("type").asText("text");
            if ("text".equals(type)) {
                if (sb.length() > 0) sb.append("\n");
                sb.append(block.path("text").asText());
            }
        }
        return sb.toString();
    }

    /**
     * Read a resource by its URI (as returned in the resources/list).
     *
     * @return raw text content of the resource
     */
    public String readResource(String uri) throws Exception {
        ensureInitialized();

        ObjectNode params = mapper.createObjectNode();
        params.put("uri", uri);

        JsonNode response = sendRequest("resources/read", params);

        // result.contents[] – each item has type + text or blob
        JsonNode contents = response.path("result").path("contents");
        if (contents.isMissingNode() || !contents.isArray()) {
            throw new RuntimeException("resources/read returned no contents: " + response);
        }

        StringBuilder sb = new StringBuilder();
        for (JsonNode item : contents) {
            if (item.has("text")) {
                if (sb.length() > 0) sb.append("\n");
                sb.append(item.get("text").asText());
            }
        }
        return sb.toString();
    }

    /**
     * Retrieve a rendered prompt by name with optional arguments.
     *
     * @param promptName the prompt name from prompts/list
     * @param arguments  optional key→value pairs (may be null)
     * @return the rendered message list as JSON (role + content)
     */
    public JsonNode getPrompt(String promptName, ObjectNode arguments) throws Exception {
        ensureInitialized();

        ObjectNode params = mapper.createObjectNode();
        params.put("name", promptName);
        if (arguments != null) {
            params.set("arguments", arguments);
        }

        JsonNode response = sendRequest("prompts/get", params);
        return response.path("result").path("messages");
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /**
     * MCP lifecycle step 1: send "initialize" and parse the server's
     * capabilities + identity from the response.
     */
    private MCPServerInfo initialize() throws Exception {
        ObjectNode clientCapabilities = mapper.createObjectNode();
        // Tell the server we can handle all three capability areas
        clientCapabilities.set("tools",     mapper.createObjectNode());
        clientCapabilities.set("resources", mapper.createObjectNode());
        clientCapabilities.set("prompts",   mapper.createObjectNode());

        ObjectNode clientInfo = mapper.createObjectNode();
        clientInfo.put("name",    "java-mcp-client");
        clientInfo.put("version", "1.0.0");

        ObjectNode params = mapper.createObjectNode();
        params.put("protocolVersion", "2024-11-05");
        params.set("capabilities",    clientCapabilities);
        params.set("clientInfo",      clientInfo);

        // Build and send the initialize request
        String body = buildRequest("initialize", params).toString();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl))
                .header("Content-Type",  "application/json")
                .header("Accept",        "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> httpResponse =
                httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        // The server assigns a session ID we must echo on every subsequent call
        sessionId = httpResponse.headers()
                .firstValue("Mcp-Session-Id")
                .or(() -> httpResponse.headers().firstValue("MCP-Session-Id"))
                .orElse(null); // some servers may not use sessions

        JsonNode json = parseSSE(httpResponse.body());
        checkForError(json, "initialize");

        JsonNode result       = json.get("result");
        JsonNode capabilities = result.path("capabilities");
        String   srvName      = result.path("serverInfo").path("name").asText("unknown");
        String   srvVersion   = result.path("serverInfo").path("version").asText("?");

        MCPServerInfo info = new MCPServerInfo(serverUrl, srvName, srvVersion, capabilities);

        // Step 2 – send the "initialized" notification (no response expected per spec)
        sendNotification("notifications/initialized");

        return info;
    }

    /**
     * Generic paginated list helper.
     * Many MCP servers paginate results using a "nextCursor" field.
     * We follow the cursor chain until exhausted.
     */
    private List<JsonNode> listAll(String method, String resultKey) throws Exception {
        List<JsonNode> accumulated = new ArrayList<>();
        String cursor = null;

        do {
            ObjectNode params = mapper.createObjectNode();
            if (cursor != null) {
                params.put("cursor", cursor);
            }

            JsonNode response = sendRequest(method, params);
            JsonNode result   = response.path("result");

            JsonNode items = result.path(resultKey);
            if (items.isArray()) {
                for (JsonNode item : items) {
                    accumulated.add(item);
                }
            }

            // Follow pagination cursor if present
            cursor = result.has("nextCursor") && !result.get("nextCursor").isNull()
                     ? result.get("nextCursor").asText()
                     : null;

        } while (cursor != null);

        return accumulated;
    }

    /**
     * Send a JSON-RPC request and return the full parsed response node.
     */
    private JsonNode sendRequest(String method, ObjectNode params) throws Exception {
        ensureInitialized();

        String body = buildRequest(method, params).toString();

        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl))
                .header("Content-Type", "application/json")
                .header("Accept",       "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(body));

        if (sessionId != null) {
            builder.header("Mcp-Session-Id", sessionId);
        }

        HttpResponse<String> httpResponse =
                httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString());

        JsonNode json = parseSSE(httpResponse.body());
        checkForError(json, method);
        return json;
    }

    /**
     * Send a JSON-RPC notification (no id, no response expected).
     */
    private void sendNotification(String method) throws Exception {
        ObjectNode notification = mapper.createObjectNode();
        notification.put("jsonrpc", "2.0");
        notification.put("method",  method);
        notification.set("params",  mapper.createObjectNode());

        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl))
                .header("Content-Type", "application/json")
                .header("Accept",       "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(notification.toString()));

        if (sessionId != null) {
            builder.header("Mcp-Session-Id", sessionId);
        }

        // Per spec: server returns 202 Accepted with no body for notifications
        httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString());
    }

    /**
     * Build a JSON-RPC 2.0 request object with an auto-incremented id.
     */
    private ObjectNode buildRequest(String method, ObjectNode params) {
        ObjectNode node = mapper.createObjectNode();
        node.put("jsonrpc", "2.0");
        node.put("id",      idCounter.getAndIncrement());
        node.put("method",  method);
        if (params != null) {
            node.set("params", params);
        }
        return node;
    }

    /**
     * MCP HTTP+SSE transport wraps the JSON payload in an SSE envelope:
     *   data: { ...json... }\n\n
     * Strip the envelope and parse the inner JSON.
     */
    private JsonNode parseSSE(String raw) throws Exception {
        String payload = raw;
        if (payload != null && payload.contains("data:")) {
            int idx = payload.indexOf("data:");
            payload = payload.substring(idx + 5).trim();
        }
        return mapper.readTree(payload);
    }

    /**
     * Throw a descriptive exception if the JSON-RPC response contains an error.
     */
    private void checkForError(JsonNode json, String method) {
        if (json.has("error") && !json.get("error").isNull()) {
            JsonNode err = json.get("error");
            throw new RuntimeException(String.format(
                "MCP error on '%s': [%d] %s",
                method,
                err.path("code").asInt(-1),
                err.path("message").asText("unknown error")
            ));
        }
    }

    private void ensureInitialized() {
        if (serverInfo == null) {
            throw new IllegalStateException(
                "MCPClient not initialized. Call discover() first.");
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    public MCPServerInfo getServerInfo() { return serverInfo; }
    public String        getSessionId()  { return sessionId;  }
}
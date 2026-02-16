package com.example;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Holds the fully-discovered state of one MCP server after the
 * initialization + capability-discovery handshake.
 *
 * Three MCP primitives are discovered:
 *   - Tools     → things the LLM can *call*
 *   - Resources → data/files the client can *read* for context
 *   - Prompts   → reusable prompt templates the server exposes
 */
public class MCPServerInfo {

    private final String url;
    private final String serverName;
    private final String serverVersion;
    private final JsonNode capabilities;  // raw server capabilities object

    // ── Discovered primitives ────────────────────────────────────────────────
    private final List<JsonNode> tools     = new ArrayList<>();
    private final List<JsonNode> resources = new ArrayList<>();
    private final List<JsonNode> prompts   = new ArrayList<>();

    public MCPServerInfo(String url, String serverName, String serverVersion,
                         JsonNode capabilities) {
        this.url           = url;
        this.serverName    = serverName;
        this.serverVersion = serverVersion;
        this.capabilities  = capabilities;
    }

    // ── Capability helpers ───────────────────────────────────────────────────

    public boolean supportsTools() {
        return capabilities != null && capabilities.has("tools");
    }

    public boolean supportsResources() {
        return capabilities != null && capabilities.has("resources");
    }

    public boolean supportsPrompts() {
        return capabilities != null && capabilities.has("prompts");
    }

    // ── Mutators ─────────────────────────────────────────────────────────────

    public void addTools(Iterable<JsonNode> nodes) {
        nodes.forEach(tools::add);
    }

    public void addResources(Iterable<JsonNode> nodes) {
        nodes.forEach(resources::add);
    }

    public void addPrompts(Iterable<JsonNode> nodes) {
        nodes.forEach(prompts::add);
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    public String getUrl()           { return url; }
    public String getServerName()    { return serverName; }
    public String getServerVersion() { return serverVersion; }
    public JsonNode getCapabilities(){ return capabilities; }
    public List<JsonNode> getTools()     { return tools; }
    public List<JsonNode> getResources() { return resources; }
    public List<JsonNode> getPrompts()   { return prompts; }

    @Override
    public String toString() {
        return String.format(
            "MCPServer[%s v%s @ %s | tools=%d resources=%d prompts=%d]",
            serverName, serverVersion, url,
            tools.size(), resources.size(), prompts.size()
        );
    }
}
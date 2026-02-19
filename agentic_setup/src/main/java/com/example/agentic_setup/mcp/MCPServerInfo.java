package com.example.agentic_setup.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.List;

/** Discovered state of one MCP server: tools, resources, and prompt templates. */
public class MCPServerInfo {

    private final String   url;
    private final String   name;
    private final String   version;
    private final JsonNode capabilities;

    private final List<JsonNode> tools     = new ArrayList<>();
    private final List<JsonNode> resources = new ArrayList<>();
    private final List<JsonNode> prompts   = new ArrayList<>();

    public MCPServerInfo(String url, String name, String version, JsonNode capabilities) {
        this.url          = url;
        this.name         = name;
        this.version      = version;
        this.capabilities = capabilities;
    }

    public boolean supportsTools()     { return capabilities != null && capabilities.has("tools"); }
    public boolean supportsResources() { return capabilities != null && capabilities.has("resources"); }
    public boolean supportsPrompts()   { return capabilities != null && capabilities.has("prompts"); }

    public void addTools(List<JsonNode> items)     { tools.addAll(items); }
    public void addResources(List<JsonNode> items) { resources.addAll(items); }
    public void addPrompts(List<JsonNode> items)   { prompts.addAll(items); }

    public List<JsonNode> getTools()     { return tools; }
    public List<JsonNode> getResources() { return resources; }
    public List<JsonNode> getPrompts()   { return prompts; }

    @Override
    public String toString() {
        return String.format("%s v%s @ %s [tools=%d resources=%d prompts=%d]",
                name, version, url, tools.size(), resources.size(), prompts.size());
    }
}
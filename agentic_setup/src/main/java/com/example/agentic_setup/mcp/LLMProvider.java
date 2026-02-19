package com.example.agentic_setup.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;

/**
 * Vendor-agnostic LLM interface.
 * Implement for OpenAI, Anthropic, Gemini, etc.
 * One instance is wired in Main and shared across all agent components.
 */
public interface LLMProvider {

    record Message(String role, String content) {}
    record ToolCall(String callId, String toolName, JsonNode arguments) {}
    record ToolResult(String callId, String result) {}
    record ToolDefinition(String name, String description, JsonNode inputSchema) {}
    record LLMResponse(String finalText, List<ToolCall> toolCalls, List<Object> rawItems) {}

    /** First turn — may return tool calls or a final text answer. */
    LLMResponse chat(String systemPrompt, List<Message> messages, List<ToolDefinition> tools)
            throws Exception;

    /** Continuation after tool results are collected. */
    LLMResponse continueWithToolResults(String systemPrompt, List<Message> messages,
            List<ToolResult> results, List<Object> previousItems, List<ToolDefinition> tools)
            throws Exception;

    /** Simple single-turn text completion with no tools. */
    default String complete(String systemPrompt, String userMessage) throws Exception {
        var r = chat(systemPrompt, List.of(new Message("user", userMessage)), List.of());
        return r.finalText() != null ? r.finalText() : "";
    }
}
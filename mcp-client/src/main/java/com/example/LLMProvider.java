package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * Abstraction over any LLM that supports tool/function calling.
 *
 * This decouples the agent loop from any specific LLM vendor.
 * You can implement this for OpenAI, Anthropic, Google Gemini, Mistral, etc.
 * as long as the model supports tool / function calling.
 *
 * The agent loop calls:
 *   1. chat(messages, tools)     – first turn; may return tool calls
 *   2. finalize(messages, tools) – second turn after tool results are appended
 */
public interface LLMProvider {

    /**
     * A single unit of conversation – mirrors the role/content structure
     * used by most LLM APIs.
     */
    record Message(String role, String content) {}

    /**
     * The LLM's response to a chat turn.
     */
    record LLMResponse(
        String         finalText,       // set when the model returns a plain message
        List<ToolCall> toolCalls,       // set when the model wants to use tools
        List<Object>   rawOutputItems   // original output items for multi-turn replay
    ) {}

    /**
     * Represents one tool-call the LLM wants to make.
     */
    record ToolCall(
        String   callId,       // opaque id used to match the result back
        String   toolName,
        JsonNode arguments     // parsed JSON arguments
    ) {}

    /**
     * Represents one tool result being fed back to the LLM.
     */
    record ToolResult(
        String callId,
        String result
    ) {}

    /**
     * Describes a tool to the LLM.
     * The format is intentionally generic; each provider implementation
     * maps this to the provider-specific schema format.
     */
    record ToolDefinition(
        String   name,
        String   description,
        JsonNode inputSchema
    ) {}

    // ── Core methods ─────────────────────────────────────────────────────────

    /**
     * Send the conversation so far plus available tool definitions.
     * Returns either a final text answer or a list of tool calls.
     *
     * @param systemPrompt optional system/developer-role message (null = omit)
     */
    LLMResponse chat(String systemPrompt,
                     List<Message> messages,
                     List<ToolDefinition> tools) throws Exception;

    /**
     * Continue the conversation after tool results have been collected.
     * The rawOutputItems from the previous LLMResponse are passed back along
     * with the tool results so the model has full context.
     *
     * @param systemPrompt optional system/developer-role message (null = omit)
     */
    LLMResponse continueWithToolResults(
        String           systemPrompt,
        List<Message>    messages,
        List<ToolResult> toolResults,
        List<Object>     previousOutputItems,
        List<ToolDefinition> tools
    ) throws Exception;
}
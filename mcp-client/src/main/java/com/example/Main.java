package com.example;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Main {
    static String mcpSessionId = null;
    static final String OPENAI_URL = "https://api.openai.com/v1/responses";
    static final String MCP_URL =
        "https://content-retrival-ai-mcp.cfapps.eu10.hana.ondemand.com/general/mcp";

    public static void main(String[] args) throws Exception {

        String apiKey = System.getenv("OPENAI_API_KEY");
        HttpClient client = HttpClient.newHttpClient();
        ObjectMapper mapper = new ObjectMapper();

        // STEP 1 — Ask LLM what to do
        String question = "What are the latest AI breakthroughs?";

        String body = """
        {
          "model": "gpt-5",
          "input": [
            {
              "role": "system",
              "content": "If external information is required, respond ONLY in JSON like {\\"action\\":\\"internet_search\\",\\"arguments\\":{\\"query\\":\\"...\\"}}"
            },
            {
              "role": "user",
              "content": "What are the latest AI breakthroughs?"
            }
          ]
        }
        """;


        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OPENAI_URL))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response =
                client.send(request, HttpResponse.BodyHandlers.ofString());

        System.out.println("RAW OPENAI RESPONSE:");
        System.out.println(response.body());


        JsonNode root = mapper.readTree(response.body());

        String llmText = extractText(root);
        System.out.println("LLM Response:\n" + llmText);

        // STEP 2 — If tool call detected
        if (llmText.trim().startsWith("{")) {

            JsonNode toolJson = mapper.readTree(llmText);

            String toolName = toolJson.get("action").asText();
            JsonNode arguments = toolJson.get("arguments");

            String mcpResult = callMCP(client, toolName, arguments.toString());
            System.out.println("\nMCP RESULT:\n" + mcpResult);

            // STEP 3 — Final answer
            String safeToolResult = mapper.writeValueAsString(mcpResult);

            String finalBody = """
            {
              "model": "gpt-5",
              "input": [
                {
                  "role": "system",
                  "content": "Provide final answer based on tool result."
                },
                {
                  "role": "user",
                  "content": %s
                }
              ]
            }
            """.formatted(
                mapper.writeValueAsString(
                    "Question: " + question + "\n\nTool result:\n" + mcpResult
                )
            );


            HttpRequest finalRequest = HttpRequest.newBuilder()
                    .uri(URI.create(OPENAI_URL))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(finalBody))
                    .build();

            HttpResponse<String> finalResponse =
                    client.send(finalRequest, HttpResponse.BodyHandlers.ofString());

            JsonNode finalRoot = mapper.readTree(finalResponse.body());
            String finalAnswer = extractText(finalRoot);

            System.out.println("\nFINAL ANSWER:\n" + finalAnswer);
        }
    }

    private static String callMCP(HttpClient client,
                              String toolName,
                              String argumentsJson) throws Exception {

    if (mcpSessionId == null) {
        initializeMCP(client);
    }

    String body = """
    {
      "jsonrpc": "2.0",
      "id": 99,
      "method": "tools/call",
      "params": {
        "name": "%s",
        "arguments": %s
      }
    }
    """.formatted(toolName, argumentsJson);

    HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(MCP_URL))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Session-Id", mcpSessionId)
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();

    HttpResponse<String> response =
            client.send(request, HttpResponse.BodyHandlers.ofString());

    String raw = response.body();

    if (raw.contains("data:")) {
        raw = raw.substring(raw.indexOf("data:") + 5).trim();
    }

    ObjectMapper mapper = new ObjectMapper();
    JsonNode json = mapper.readTree(raw);

    return json.get("result")
              .get("content")
              .get(0)
              .get("text")
              .asText();
}


    private static String extractText(JsonNode root) {

      if (root == null || root.get("output") == null) {
          System.out.println("No output field found in response.");
          return "";
      }

      for (JsonNode outputItem : root.get("output")) {
          if ("message".equals(outputItem.get("type").asText())) {
              for (JsonNode content : outputItem.get("content")) {
                  if ("output_text".equals(content.get("type").asText())) {
                      return content.get("text").asText();
                  }
              }
          }
      }
      return "";
  }

  private static void initializeMCP(HttpClient client) throws Exception {

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
              .uri(URI.create(MCP_URL))
              .header("Content-Type", "application/json")
              .header("Accept", "application/json, text/event-stream")
              .POST(HttpRequest.BodyPublishers.ofString(initBody))
              .build();

      HttpResponse<String> response =
              client.send(request, HttpResponse.BodyHandlers.ofString());

      mcpSessionId = response.headers()
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
              .uri(URI.create(MCP_URL))
              .header("Content-Type", "application/json")
              .header("Accept", "application/json, text/event-stream")
              .header("MCP-Session-Id", mcpSessionId)
              .POST(HttpRequest.BodyPublishers.ofString(initializedBody))
              .build();

      client.send(initReq, HttpResponse.BodyHandlers.ofString());
  }
}

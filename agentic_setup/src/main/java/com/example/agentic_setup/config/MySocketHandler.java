package com.example.agentic_setup.config;

import com.example.agentic_setup.service.AgentStreamingService;
import org.springframework.web.socket.*;
import org.springframework.web.socket.handler.TextWebSocketHandler;

public class MySocketHandler extends TextWebSocketHandler {

    private final AgentStreamingService agentService;

    public MySocketHandler(AgentStreamingService agentService) {
        this.agentService = agentService;
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        System.out.println("[WS] Client connected: " + session.getId());
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        var prompt = message.getPayload();
        System.out.println("[WS] Prompt: " + prompt);

        // Run in a background thread so we don't block the WS event loop
        new Thread(() -> {
            try {
                agentService.runAndStream(prompt, token -> {
                    try {
                        if (session.isOpen()) session.sendMessage(new TextMessage(token));
                    } catch (Exception e) {
                        try { session.close(); } catch (Exception ignored) {}
                    }
                });
            } catch (Exception e) {
                System.err.println("[WS] Error: " + e.getMessage());
                try { session.close(); } catch (Exception ignored) {}
            }
        }, "ws-agent-" + session.getId()).start();
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        System.out.println("[WS] Client disconnected: " + session.getId());
    }
}
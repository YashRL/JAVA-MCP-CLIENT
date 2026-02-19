package com.example.agentic_setup.config;

import com.example.agentic_setup.service.AgentStreamingService;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.*;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final AgentStreamingService agentService;

    public WebSocketConfig(AgentStreamingService agentService) {
        this.agentService = agentService;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new MySocketHandler(agentService), "/ws")
                .setAllowedOrigins("*");
    }
}
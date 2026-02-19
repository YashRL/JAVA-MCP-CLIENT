package com.example.agentic_setup.controller;

import com.example.agentic_setup.service.AgentStreamingService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api")
public class OpenAiController {

    private final AgentStreamingService agentService;

    public OpenAiController(AgentStreamingService agentService) {
        this.agentService = agentService;
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@RequestParam String prompt) {
        var emitter = new SseEmitter(0L); // no timeout

        new Thread(() -> {
            try {
                agentService.runAndStream(prompt, token -> {
                    try {
                        emitter.send(token);
                        if (token.contains("[DONE]")) emitter.complete();
                    } catch (Exception e) {
                        emitter.completeWithError(e);
                    }
                });
            } catch (Exception e) {
                emitter.completeWithError(e);
            }
        }, "sse-agent").start();

        return emitter;
    }
}
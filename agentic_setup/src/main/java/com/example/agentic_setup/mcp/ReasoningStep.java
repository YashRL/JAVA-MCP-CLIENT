package com.example.agentic_setup.mcp;

import java.util.List;

/** One step in the ReAct loop: actions taken, their observations, and optional reflection. */
public record ReasoningStep(int step, List<String> actions, List<String> observations,
                             String reflection, boolean goalAchieved) {

    public String trace() {
        var sb = new StringBuilder();
        sb.append("┌─── Step ").append(step).append(" ───────────────────────────────\n");
        for (int i = 0; i < actions.size(); i++) {
            sb.append("│ ACTION:      ").append(actions.get(i)).append("\n");
            if (i < observations.size()) {
                var obs = observations.get(i);
                sb.append("│ OBSERVATION: ")
                  .append(obs.length() > 120 ? obs.substring(0, 117) + "..." : obs)
                  .append("\n");
            }
        }
        if (!reflection.isBlank())
            sb.append("│ REFLECTION:  ").append(reflection).append("\n");
        sb.append("│ GOAL MET:    ").append(goalAchieved).append("\n");
        sb.append("└────────────────────────────────────────────────\n");
        return sb.toString();
    }
}
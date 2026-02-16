package com.example;

import java.util.List;

/**
 * Captures one complete reasoning step in the ReAct (Reason + Act) loop.
 *
 * Each step follows the pattern:
 *   THOUGHT      → the executor's reasoning about what to do next
 *   ACTION       → the tool calls it decided to make
 *   OBSERVATION  → the results returned by those tool calls
 *   REFLECTION   → did this go as planned? should the approach change?
 *
 * The full trace of ReasoningSteps is printed at the end of a run so you can
 * see exactly how the agent reasoned through the problem.
 *
 * It is also used internally to decide whether to continue or stop:
 *   - If reflection indicates "goal achieved" → stop early
 *   - If reflection indicates "approach not working" → pivot strategy
 */
public record ReasoningStep(
    int          stepNumber,
    String       thought,        // why the agent is doing what it's about to do
    List<String> actionsTaken,   // tool names called in this step
    List<String> observations,   // results from each tool call (same order as actions)
    String       reflection,     // after seeing results: what do they mean? is goal met?
    boolean      goalAchieved    // true = stop the loop now
) {

    /** One-liner summary for logging. */
    public String summary() {
        return String.format(
            "Step %d [%s] actions=%s goalAchieved=%b",
            stepNumber,
            thought.length() > 60
                ? thought.substring(0, 57) + "..."
                : thought,
            actionsTaken,
            goalAchieved
        );
    }

    /** Full trace block for debug printing. */
    public String fullTrace() {
        StringBuilder sb = new StringBuilder();
        sb.append("┌─── Step ").append(stepNumber).append(" ───────────────────────────────────\n");
        sb.append("│ THOUGHT:      ").append(thought).append("\n");
        for (int i = 0; i < actionsTaken.size(); i++) {
            sb.append("│ ACTION[").append(i).append("]:     ").append(actionsTaken.get(i)).append("\n");
            if (i < observations.size()) {
                String obs = observations.get(i);
                String preview = obs.length() > 120 ? obs.substring(0, 117) + "..." : obs;
                sb.append("│ OBSERVATION[").append(i).append("]: ").append(preview).append("\n");
            }
        }
        sb.append("│ REFLECTION:   ").append(reflection).append("\n");
        sb.append("│ GOAL MET:     ").append(goalAchieved).append("\n");
        sb.append("└────────────────────────────────────────────────────────\n");
        return sb.toString();
    }
}
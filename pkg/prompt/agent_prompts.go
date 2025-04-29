// Package prompt provides a templating system for LLM prompts.
package prompt

import (
	"fmt"
	"os"
	"path/filepath"
)

// AgentPromptType represents different types of agent prompts.
type AgentPromptType string

const (
	// SystemPrompt is the system prompt for an agent.
	SystemPrompt AgentPromptType = "system"
	
	// PlanningPrompt is used for planning steps.
	PlanningPrompt AgentPromptType = "planning"
	
	// EvaluationPrompt is used for evaluating results.
	EvaluationPrompt AgentPromptType = "evaluation"
	
	// ApprovalPrompt is used for approving plans.
	ApprovalPrompt AgentPromptType = "approval"
)

// DefaultPromptDir is the default directory for prompt templates.
// It can be overridden by setting the GO_AGENTS_PROMPT_DIR environment variable.
var DefaultPromptDir = func() string {
	if dir := os.Getenv("GO_AGENTS_PROMPT_DIR"); dir != "" {
		return dir
	}
	// Default to a "prompts" directory in the current working directory
	return "prompts"
}()

// LoadAgentPrompts loads all agent prompts from the default prompt directory.
// It looks for files with the pattern "{agent_type}_{prompt_type}.prompt".
// For example, "planning_system.prompt" or "evaluation_approval.prompt".
func LoadAgentPrompts() error {
	return LoadTemplatesFromDir(DefaultPromptDir)
}

// GetAgentPrompt gets a prompt template for a specific agent and prompt type.
// It follows the naming convention "{agent_type}_{prompt_type}.prompt".
func GetAgentPrompt(agentType string, promptType AgentPromptType) (*Template, error) {
	templateName := fmt.Sprintf("%s_%s.prompt", agentType, promptType)
	tmpl, ok := GetTemplate(templateName)
	if !ok {
		return nil, fmt.Errorf("prompt template %s not found", templateName)
	}
	return tmpl, nil
}

// RenderAgentPrompt renders a prompt for a specific agent and prompt type with the given data.
func RenderAgentPrompt(agentType string, promptType AgentPromptType, data interface{}) (string, error) {
	tmpl, err := GetAgentPrompt(agentType, promptType)
	if err != nil {
		return "", err
	}
	return tmpl.Render(data)
}

// InitializeDefaultPrompts ensures the default prompt directory exists and contains
// basic prompt templates if they don't already exist.
func InitializeDefaultPrompts() error {
	// Create the default prompt directory if it doesn't exist
	if err := os.MkdirAll(DefaultPromptDir, 0755); err != nil {
		return fmt.Errorf("failed to create prompt directory: %w", err)
	}

	// Define default prompts
	defaultPrompts := map[string]string{
		"planning_system.prompt": `You are an AI assistant that helps create plans to solve tasks.
Your goal is to break down complex tasks into manageable steps.

{{- if .Context }}
Context:
{{ .Context }}
{{- end }}`,

		"planning_planning.prompt": `Please create a detailed plan to solve the following task:

Task: {{ .Task }}

{{- if .Constraints }}
Constraints:
{{ range .Constraints }}
- {{ . }}
{{- end }}
{{- end }}

Please provide a step-by-step plan with clear, actionable steps.`,

		"planning_approval.prompt": `I've created a plan to solve your task. Please review it and let me know if you approve:

Task: {{ .Task }}

Plan:
{{ .Plan }}

Do you approve this plan? If not, please provide feedback on what should be changed.`,

		"planning_evaluation.prompt": `Please evaluate the result of the following step:

Step: {{ .Step }}
Expected Outcome: {{ .ExpectedOutcome }}
Actual Result: {{ .ActualResult }}

Is the step completed successfully? If not, what went wrong and how should we adjust?`,
	}

	// Write default prompts to files if they don't exist
	for filename, content := range defaultPrompts {
		path := filepath.Join(DefaultPromptDir, filename)
		
		// Skip if file already exists
		if _, err := os.Stat(path); err == nil {
			continue
		}
		
		// Write the file
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return fmt.Errorf("failed to write prompt template %s: %w", filename, err)
		}
	}

	// Load the templates into the cache
	return LoadTemplatesFromDir(DefaultPromptDir)
}

// CreateExamplePrompt demonstrates how to create a custom prompt with advanced features.
func CreateExamplePrompt() (*Template, error) {
	// This is an example of a more complex prompt template with conditionals, loops, and formatting
	content := `# {{ .Title }}

{{- if .SystemInstructions }}
## System Instructions
{{ .SystemInstructions }}
{{- end }}

## Task
{{ .Task }}

{{- if .Examples }}
## Examples
{{ range $i, $example := .Examples }}
### Example {{ add $i 1 }}
Input: {{ $example.Input }}
Output: {{ $example.Output }}
{{- end }}
{{- end }}

{{- if .Tools }}
## Available Tools
{{ range .Tools }}
- {{ .Name }}: {{ .Description }}
{{- end }}
{{- end }}

{{- if .Context }}
## Context
{{ .Context }}
{{- end }}

{{- if .Constraints }}
## Constraints
{{ range .Constraints }}
- {{ . }}
{{- end }}
{{- end }}

{{- if .AdditionalInstructions }}
## Additional Instructions
{{ .AdditionalInstructions }}
{{- end }}

{{- /* Add timestamp for tracking */ -}}
<!-- Generated on {{ formatTime "2006-01-02 15:04:05" (now) }} -->
`

	// Add the "add" function to our template functions
	templateFuncs["add"] = func(a, b int) int {
		return a + b
	}

	return AddTemplate("example_complex_prompt.prompt", content)
}

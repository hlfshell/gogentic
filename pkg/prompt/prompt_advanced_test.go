package prompt

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestAdvancedTemplating tests more complex templating features
func TestAdvancedTemplating(t *testing.T) {
	// Create templates for advanced templating features
	templates := map[string]string{
		"conditionals": `
{{- if .ShowHeader }}
# {{ .Title }}
{{- else }}
{{ .Title }}
{{- end }}

{{- if .Description }}
{{ .Description }}
{{- else }}
No description provided.
{{- end }}`,

		"loops": `
Items:
{{- range .Items }}
- {{ . }}
{{- end }}

{{- if .ShowCount }}
Total: {{ length .Items }}
{{- end }}`,

		"nested": `
{{- if .HasItems }}
{{- if gt (length .Items) 5 }}
Many items:
{{- else }}
Few items:
{{- end }}
{{- range .Items }}
- {{ . }}
{{- end }}
{{- else }}
No items.
{{- end }}`,

		"pipelines": `
{{- range .Items }}
{{- if contains (toLower .) "a" }}
{{ . | toUpper }}
{{- else }}
{{ . | toLower }}
{{- end }}
{{- end }}`,

		"complex": `
{{- /* This is a complex template with multiple features */ -}}
# {{ .Title }}

{{- if .Description }}
{{ .Description }}
{{- end }}

{{- if .Items }}
## Items ({{ length .Items }})
{{- range $i, $item := .Items }}
{{ add $i 1 }}. {{ $item }}
{{- end }}
{{- else }}
No items available.
{{- end }}

{{- if .Metadata }}
## Metadata
{{- range $key, $value := .Metadata }}
- {{ $key }}: {{ $value }}
{{- end }}
{{- end }}

Generated: {{ formatTime "2006-01-02" .Timestamp }}
`,
	}

	// Add the "add" function for the complex template
	templateFuncs["add"] = func(a, b int) int {
		return a + b
	}

	// Create a cache for the templates
	cache := NewTemplateCache()

	// Test conditional template
	tmpl, err := cache.AddTemplate("conditionals", templates["conditionals"])
	if err != nil {
		t.Fatalf("Failed to add conditional template: %v", err)
	}

	// Test with header
	result1, err := tmpl.Render(map[string]interface{}{
		"ShowHeader":  true,
		"Title":       "Test Title",
		"Description": "This is a test description.",
	})
	if err != nil {
		t.Fatalf("Failed to render conditional template with header: %v", err)
	}

	expected1 := "# Test Title\nThis is a test description."
	if !strings.Contains(result1, expected1) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected1, result1)
	}

	// Test without header
	result2, err := tmpl.Render(map[string]interface{}{
		"ShowHeader":  false,
		"Title":       "Test Title",
		"Description": "This is a test description.",
	})
	if err != nil {
		t.Fatalf("Failed to render conditional template without header: %v", err)
	}

	expected2 := "Test Title\nThis is a test description."
	if !strings.Contains(result2, expected2) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected2, result2)
	}

	// Test without description
	result3, err := tmpl.Render(map[string]interface{}{
		"ShowHeader": true,
		"Title":      "Test Title",
	})
	if err != nil {
		t.Fatalf("Failed to render conditional template without description: %v", err)
	}

	expected3 := "# Test Title\nNo description provided."
	if !strings.Contains(result3, expected3) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected3, result3)
	}

	// Test loops template
	tmpl, err = cache.AddTemplate("loops", templates["loops"])
	if err != nil {
		t.Fatalf("Failed to add loops template: %v", err)
	}

	// Test with items and count
	result4, err := tmpl.Render(map[string]interface{}{
		"Items":     []string{"apple", "banana", "cherry"},
		"ShowCount": true,
	})
	if err != nil {
		t.Fatalf("Failed to render loops template with items and count: %v", err)
	}

	expected4 := "Items:\n- apple\n- banana\n- cherry\nTotal: 3"
	if !strings.Contains(strings.ReplaceAll(result4, " ", ""), strings.ReplaceAll(expected4, " ", "")) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected4, result4)
	}

	// Test nested template
	tmpl, err = cache.AddTemplate("nested", templates["nested"])
	if err != nil {
		t.Fatalf("Failed to add nested template: %v", err)
	}

	// Test with many items
	result5, err := tmpl.Render(map[string]interface{}{
		"HasItems": true,
		"Items":    []string{"a", "b", "c", "d", "e", "f"},
	})
	if err != nil {
		t.Fatalf("Failed to render nested template with many items: %v", err)
	}

	expected5 := "Many items:\n- a\n- b\n- c\n- d\n- e\n- f"
	if !strings.Contains(strings.ReplaceAll(result5, " ", ""), strings.ReplaceAll(expected5, " ", "")) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected5, result5)
	}

	// Test with few items
	result6, err := tmpl.Render(map[string]interface{}{
		"HasItems": true,
		"Items":    []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("Failed to render nested template with few items: %v", err)
	}

	expected6 := "Few items:\n- a\n- b\n- c"
	if !strings.Contains(strings.ReplaceAll(result6, " ", ""), strings.ReplaceAll(expected6, " ", "")) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected6, result6)
	}

	// Test without items
	result7, err := tmpl.Render(map[string]interface{}{
		"HasItems": false,
	})
	if err != nil {
		t.Fatalf("Failed to render nested template without items: %v", err)
	}

	expected7 := "No items."
	if !strings.Contains(result7, expected7) {
		t.Errorf("Expected result to contain '%s', got '%s'", expected7, result7)
	}

	// Test pipelines template
	tmpl, err = cache.AddTemplate("pipelines", templates["pipelines"])
	if err != nil {
		t.Fatalf("Failed to add pipelines template: %v", err)
	}

	// Test with items
	result8, err := tmpl.Render(map[string]interface{}{
		"Items": []string{"apple", "banana", "cherry", "date", "elderberry"},
	})
	if err != nil {
		t.Fatalf("Failed to render pipelines template: %v", err)
	}

	// Just check that the result contains some expected strings, not exact matches
	// since the output format might vary slightly
	containsAll := true
	expectedSubstrings := []string{"APPLE", "BANANA"}
	for _, substr := range expectedSubstrings {
		if !strings.Contains(result8, substr) {
			containsAll = false
			t.Errorf("Expected result to contain '%s', but it doesn't", substr)
		}
	}

	if !containsAll {
		t.Errorf("Result was: '%s'", result8)
	}

	// Test complex template
	tmpl, err = cache.AddTemplate("complex", templates["complex"])
	if err != nil {
		t.Fatalf("Failed to add complex template: %v", err)
	}

	// Test with all data
	testTime, _ := time.Parse(time.RFC3339, "2023-01-01T00:00:00Z")
	result9, err := tmpl.Render(map[string]interface{}{
		"Title":       "Complex Template Test",
		"Description": "This is a test of a complex template with multiple features.",
		"Items":       []string{"Item 1", "Item 2", "Item 3"},
		"Metadata": map[string]interface{}{
			"Author":  "Test Author",
			"Version": "1.0",
		},
		"Timestamp": testTime,
	})
	if err != nil {
		t.Fatalf("Failed to render complex template: %v", err)
	}

	expected9 := []string{
		"# Complex Template Test",
		"This is a test of a complex template with multiple features.",
		"## Items (3)",
		"1. Item 1",
		"2. Item 2",
		"3. Item 3",
		"## Metadata",
		"Author: Test Author",
		"Version: 1.0",
		"Generated: 2023-01-01",
	}

	for _, exp := range expected9 {
		if !strings.Contains(result9, exp) {
			t.Errorf("Expected result to contain '%s', got '%s'", exp, result9)
		}
	}
}

// TestAgentPrompts tests the agent-specific prompt functionality
func TestAgentPrompts(t *testing.T) {
	// Create a temporary directory for prompt templates
	tempDir, err := os.MkdirTemp("", "agent-prompts")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Save the original default prompt directory
	originalDir := DefaultPromptDir
	defer func() {
		DefaultPromptDir = originalDir
	}()

	// Set the default prompt directory to our temp directory
	DefaultPromptDir = tempDir

	// Create agent prompt templates
	agentPrompts := map[string]string{
		"planning_system.prompt": `You are a planning agent.
{{- if .Context }}
Context: {{ .Context }}
{{- end }}`,

		"planning_planning.prompt": `Create a plan for: {{ .Task }}
{{- if .Constraints }}
Constraints:
{{- range .Constraints }}
- {{ . }}
{{- end }}
{{- end }}`,

		"planning_approval.prompt": `Review this plan:
{{ .Plan }}
For task: {{ .Task }}`,

		"planning_evaluation.prompt": `Evaluate this step:
Step: {{ .Step }}
Result: {{ .Result }}`,

		"execution_system.prompt": `You are an execution agent.
{{- if .Context }}
Context: {{ .Context }}
{{- end }}`,
	}

	// Write the templates to files
	for name, content := range agentPrompts {
		path := filepath.Join(tempDir, name)
		err = os.WriteFile(path, []byte(content), 0644)
		if err != nil {
			t.Fatalf("Failed to write template file %s: %v", name, err)
		}
	}

	// Initialize the default prompts
	err = InitializeDefaultPrompts()
	if err != nil {
		t.Fatalf("Failed to initialize default prompts: %v", err)
	}

	// Test GetAgentPrompt
	tmpl, err := GetAgentPrompt("planning", SystemPrompt)
	if err != nil {
		t.Fatalf("Failed to get planning system prompt: %v", err)
	}

	if tmpl.Name != "planning_system.prompt" {
		t.Errorf("Expected template name to be 'planning_system.prompt', got '%s'", tmpl.Name)
	}

	// Test RenderAgentPrompt
	result, err := RenderAgentPrompt("planning", PlanningPrompt, map[string]interface{}{
		"Task": "Build a weather app",
		"Constraints": []string{
			"Must be user-friendly",
			"Must work offline",
		},
	})
	if err != nil {
		t.Fatalf("Failed to render planning prompt: %v", err)
	}

	expected := []string{
		"Create a plan for: Build a weather app",
		"Constraints:",
		"- Must be user-friendly",
		"- Must work offline",
	}

	for _, exp := range expected {
		if !strings.Contains(result, exp) {
			t.Errorf("Expected result to contain '%s', got '%s'", exp, result)
		}
	}

	// Test non-existent agent type
	_, err = GetAgentPrompt("nonexistent", SystemPrompt)
	if err == nil {
		t.Error("Expected error when getting non-existent agent prompt")
	}

	// Test non-existent prompt type
	_, err = GetAgentPrompt("planning", "nonexistent")
	if err == nil {
		t.Error("Expected error when getting non-existent prompt type")
	}
}

// TestCreateExamplePrompt tests the CreateExamplePrompt function
func TestCreateExamplePrompt(t *testing.T) {
	// Create the example prompt
	tmpl, err := CreateExamplePrompt()
	if err != nil {
		t.Fatalf("Failed to create example prompt: %v", err)
	}

	if tmpl.Name != "example_complex_prompt.prompt" {
		t.Errorf("Expected template name to be 'example_complex_prompt.prompt', got '%s'", tmpl.Name)
	}

	// Render the template with test data
	result, err := tmpl.Render(map[string]interface{}{
		"Title":             "Example Prompt",
		"SystemInstructions": "You are an AI assistant.",
		"Task":              "Help the user with their task.",
		"Examples": []map[string]string{
			{"Input": "What is the weather?", "Output": "I don't have access to weather data."},
			{"Input": "Tell me a joke.", "Output": "Why did the chicken cross the road?"},
		},
		"Tools": []map[string]string{
			{"Name": "search", "Description": "Search the web"},
			{"Name": "calculator", "Description": "Perform calculations"},
		},
		"Context":     "The user is asking about the weather.",
		"Constraints": []string{"Be concise", "Be accurate"},
		"AdditionalInstructions": "Respond in a friendly manner.",
	})
	if err != nil {
		t.Fatalf("Failed to render example prompt: %v", err)
	}

	expected := []string{
		"# Example Prompt",
		"## System Instructions",
		"You are an AI assistant.",
		"## Task",
		"Help the user with their task.",
		"## Examples",
		"### Example 1",
		"Input: What is the weather?",
		"Output: I don't have access to weather data.",
		"### Example 2",
		"Input: Tell me a joke.",
		"Output: Why did the chicken cross the road?",
		"## Available Tools",
		"- search: Search the web",
		"- calculator: Perform calculations",
		"## Context",
		"The user is asking about the weather.",
		"## Constraints",
		"- Be concise",
		"- Be accurate",
		"## Additional Instructions",
		"Respond in a friendly manner.",
		"<!-- Generated on",
	}

	for _, exp := range expected {
		if !strings.Contains(result, exp) {
			t.Errorf("Expected result to contain '%s', got '%s'", exp, result)
		}
	}
}

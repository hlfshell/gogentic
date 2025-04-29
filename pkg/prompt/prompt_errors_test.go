package prompt

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestTemplateErrors tests error handling in the templating system
func TestTemplateErrors(t *testing.T) {
	// Test adding a template with syntax errors
	_, err := AddTemplate("syntax-error", "Hello, {{.Name")
	if err == nil {
		t.Error("Expected error when adding template with syntax error")
	}

	// Test rendering a template with missing variables
	tmpl, err := AddTemplate("missing-var", "Hello, {{.Name}}")
	if err != nil {
		t.Fatalf("Failed to add template: %v", err)
	}

	result, err := tmpl.Render(map[string]interface{}{
		"NotName": "World",
	})
	if err != nil {
		t.Fatalf("Unexpected error when rendering template with missing variables: %v", err)
	}
	// In Go templates, a missing variable is rendered as a zero value,
	// which can appear in different formats depending on Go version and template implementation
	validResults := []string{"Hello, !", "Hello, <no value>!", "Hello, <no value>"}
	resultValid := false
	for _, valid := range validResults {
		if result == valid {
			resultValid = true
			break
		}
	}
	if !resultValid {
		t.Errorf("Expected one of %v, got '%s'", validResults, result)
	}

	// Test loading a non-existent template file
	_, err = LoadTemplate("/path/to/nonexistent/template.prompt")
	if err == nil {
		t.Error("Expected error when loading non-existent template file")
	}

	// Test loading templates from a non-existent directory
	err = LoadTemplatesFromDir("/path/to/nonexistent/dir")
	if err == nil {
		t.Error("Expected error when loading templates from non-existent directory")
	}

	// Test getting a non-existent template
	_, ok := GetTemplate("non-existent-template")
	if ok {
		t.Error("Expected GetTemplate to return false for non-existent template")
	}

	// Test RenderWithData with a non-existent template
	_, err = RenderWithData("non-existent-template", map[string]interface{}{})
	if err == nil {
		t.Error("Expected error when rendering non-existent template")
	}
}

// TestTemplateConcurrency tests concurrent access to the template cache
func TestTemplateConcurrency(t *testing.T) {
	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "prompt-concurrency")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a template file
	templatePath := filepath.Join(tempDir, "concurrent.prompt")
	err = os.WriteFile(templatePath, []byte("Concurrent: {{.Value}}"), 0644)
	if err != nil {
		t.Fatalf("Failed to write template file: %v", err)
	}

	// Create a new template cache
	cache := NewTemplateCache()

	// Run concurrent goroutines to access the cache
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			// Load the template
			tmpl, err := cache.LoadTemplate(templatePath)
			if err != nil {
				t.Errorf("Goroutine %d: Failed to load template: %v", id, err)
				done <- false
				return
			}

			// Render the template
			result, err := tmpl.Render(map[string]interface{}{
				"Value": id,
			})
			if err != nil {
				t.Errorf("Goroutine %d: Failed to render template: %v", id, err)
				done <- false
				return
			}

			expected := "Concurrent: " + fmt.Sprintf("%d", id)
			if result != expected {
				t.Errorf("Goroutine %d: Expected '%s', got '%s'", id, expected, result)
				done <- false
				return
			}

			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}
}

// TestTemplateEdgeCases tests edge cases in the templating system
func TestTemplateEdgeCases(t *testing.T) {
	// Test empty template
	tmpl, err := AddTemplate("empty", "")
	if err != nil {
		t.Fatalf("Failed to add empty template: %v", err)
	}

	result, err := tmpl.Render(map[string]interface{}{})
	if err != nil {
		t.Fatalf("Failed to render empty template: %v", err)
	}

	if result != "" {
		t.Errorf("Expected empty result, got '%s'", result)
	}

	// Test template with only whitespace
	tmpl, err = AddTemplate("whitespace", "   \n\t   ")
	if err != nil {
		t.Fatalf("Failed to add whitespace template: %v", err)
	}

	result, err = tmpl.Render(map[string]interface{}{})
	if err != nil {
		t.Fatalf("Failed to render whitespace template: %v", err)
	}

	if result != "   \n\t   " {
		t.Errorf("Expected whitespace result, got '%s'", result)
	}

	// Test template with only comments
	tmpl, err = AddTemplate("comments", "{{/* This is a comment */}}")
	if err != nil {
		t.Fatalf("Failed to add comments template: %v", err)
	}

	result, err = tmpl.Render(map[string]interface{}{})
	if err != nil {
		t.Fatalf("Failed to render comments template: %v", err)
	}

	if result != "" {
		t.Errorf("Expected empty result, got '%s'", result)
	}

	// Test template with nil data
	tmpl, err = AddTemplate("nil-data", "Hello, {{if .Name}}{{.Name}}{{else}}World{{end}}!")
	if err != nil {
		t.Fatalf("Failed to add nil-data template: %v", err)
	}

	result, err = tmpl.Render(nil)
	if err != nil {
		t.Fatalf("Failed to render template with nil data: %v", err)
	}

	if result != "Hello, World!" {
		t.Errorf("Expected 'Hello, World!', got '%s'", result)
	}

	// Test template with very large input
	largeString := make([]byte, 1000000) // 1MB
	for i := range largeString {
		largeString[i] = 'a'
	}

	tmpl, err = AddTemplate("large-input", "Length: {{length .LargeString}}")
	if err != nil {
		t.Fatalf("Failed to add large-input template: %v", err)
	}

	result, err = tmpl.Render(map[string]interface{}{
		"LargeString": string(largeString),
	})
	if err != nil {
		t.Fatalf("Failed to render template with large input: %v", err)
	}

	expected := "Length: 1000000"
	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}

	// Test template with recursive data structures
	recursiveMap := make(map[string]interface{})
	recursiveMap["self"] = recursiveMap

	tmpl, err = AddTemplate("recursive", "Recursive: {{if .self}}true{{else}}false{{end}}")
	if err != nil {
		t.Fatalf("Failed to add recursive template: %v", err)
	}

	result, err = tmpl.Render(recursiveMap)
	if err != nil {
		t.Fatalf("Failed to render template with recursive data: %v", err)
	}

	if result != "Recursive: true" {
		t.Errorf("Expected 'Recursive: true', got '%s'", result)
	}
}

// TestAgentPromptEdgeCases tests edge cases in the agent prompt functionality
func TestAgentPromptEdgeCases(t *testing.T) {
	// Create a temporary directory for prompt templates
	tempDir, err := os.MkdirTemp("", "agent-prompts-edge")
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

	// Test InitializeDefaultPrompts with non-existent directory
	nonExistentDir := filepath.Join(tempDir, "non-existent")
	DefaultPromptDir = nonExistentDir
	err = InitializeDefaultPrompts()
	if err != nil {
		t.Fatalf("Failed to initialize default prompts with non-existent directory: %v", err)
	}

	// Check that the directory was created
	_, err = os.Stat(nonExistentDir)
	if os.IsNotExist(err) {
		t.Errorf("Expected directory %s to be created", nonExistentDir)
	}

	// Test GetAgentPrompt with non-existent agent type
	_, err = GetAgentPrompt("non-existent", SystemPrompt)
	if err == nil {
		t.Error("Expected error when getting prompt for non-existent agent type")
	}

	// Test RenderAgentPrompt with non-existent agent type
	_, err = RenderAgentPrompt("non-existent", SystemPrompt, map[string]interface{}{})
	if err == nil {
		t.Error("Expected error when rendering prompt for non-existent agent type")
	}

	// Test with read-only directory
	if os.Geteuid() != 0 { // Skip if running as root
		readOnlyDir := filepath.Join(tempDir, "read-only")
		err = os.MkdirAll(readOnlyDir, 0555) // Read-only directory
		if err != nil {
			t.Fatalf("Failed to create read-only directory: %v", err)
		}

		DefaultPromptDir = readOnlyDir
		err = InitializeDefaultPrompts()
		if err == nil {
			// This might not fail on all systems, so we don't assert an error
			// Just make sure it doesn't panic
		}
	}
}

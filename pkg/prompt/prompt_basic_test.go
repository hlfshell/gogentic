package prompt

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestBasicTemplateOperations tests the basic operations of the templating system
func TestBasicTemplateOperations(t *testing.T) {
	// Create a new template cache
	cache := NewTemplateCache()

	// Test adding a template
	tmpl, err := cache.AddTemplate("test", "Hello, {{.Name}}!")
	if err != nil {
		t.Fatalf("Failed to add template: %v", err)
	}

	if tmpl.Name != "test" {
		t.Errorf("Expected template name to be 'test', got '%s'", tmpl.Name)
	}

	if tmpl.Content != "Hello, {{.Name}}!" {
		t.Errorf("Expected template content to be 'Hello, {{.Name}}!', got '%s'", tmpl.Content)
	}

	// Test getting a template
	tmpl2, ok := cache.GetTemplate("test")
	if !ok {
		t.Fatal("Failed to get template")
	}

	if tmpl2.Name != "test" {
		t.Errorf("Expected template name to be 'test', got '%s'", tmpl2.Name)
	}

	// Test rendering a template
	result, err := tmpl.Render(map[string]interface{}{
		"Name": "World",
	})
	if err != nil {
		t.Fatalf("Failed to render template: %v", err)
	}

	expected := "Hello, World!"
	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}

	// Test rendering a template with missing variables - in Go's template system,
	// missing variables are treated as zero values, not errors
	result, err = tmpl.Render(map[string]interface{}{})
	if err != nil {
		t.Fatalf("Unexpected error when rendering template with missing variables: %v", err)
	}
	// In Go templates, a missing variable is rendered as a zero value,
	// which for strings is "<no value>" in some Go versions and empty string in others
	if result != "Hello, !" && result != "Hello, <no value>!" {
		t.Errorf("Expected 'Hello, !' or 'Hello, <no value>!', got '%s'", result)
	}

	// Test adding a template with invalid syntax
	_, err = cache.AddTemplate("invalid", "Hello, {{.Name")
	if err == nil {
		t.Error("Expected error when adding template with invalid syntax")
	}
}

// TestTemplateModificationDetection tests that templates are reloaded when modified
func TestTemplateModificationDetection(t *testing.T) {
	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "prompt-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a template file
	templatePath := filepath.Join(tempDir, "test.prompt")
	initialContent := "Hello, {{.Name}}!"
	err = os.WriteFile(templatePath, []byte(initialContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write template file: %v", err)
	}

	// Create a new template cache
	cache := NewTemplateCache()

	// Load the template
	tmpl1, err := cache.LoadTemplate(templatePath)
	if err != nil {
		t.Fatalf("Failed to load template: %v", err)
	}

	// Render the template
	result1, err := tmpl1.Render(map[string]interface{}{
		"Name": "World",
	})
	if err != nil {
		t.Fatalf("Failed to render template: %v", err)
	}

	expected1 := "Hello, World!"
	if result1 != expected1 {
		t.Errorf("Expected '%s', got '%s'", expected1, result1)
	}

	// Wait a moment to ensure the file modification time changes
	time.Sleep(100 * time.Millisecond)

	// Modify the template file
	modifiedContent := "Hi, {{.Name}}!"
	err = os.WriteFile(templatePath, []byte(modifiedContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write modified template file: %v", err)
	}

	// Load the template again
	tmpl2, err := cache.LoadTemplate(templatePath)
	if err != nil {
		t.Fatalf("Failed to load modified template: %v", err)
	}

	// Render the modified template
	result2, err := tmpl2.Render(map[string]interface{}{
		"Name": "World",
	})
	if err != nil {
		t.Fatalf("Failed to render modified template: %v", err)
	}

	expected2 := "Hi, World!"
	if result2 != expected2 {
		t.Errorf("Expected '%s', got '%s'", expected2, result2)
	}
}

// TestTemplateDirectoryLoading tests loading templates from a directory
func TestTemplateDirectoryLoading(t *testing.T) {
	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "prompt-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create subdirectories
	subDir1 := filepath.Join(tempDir, "subdir1")
	subDir2 := filepath.Join(tempDir, "subdir2")
	err = os.MkdirAll(subDir1, 0755)
	if err != nil {
		t.Fatalf("Failed to create subdir1: %v", err)
	}
	err = os.MkdirAll(subDir2, 0755)
	if err != nil {
		t.Fatalf("Failed to create subdir2: %v", err)
	}

	// Create template files
	templates := map[string]string{
		filepath.Join(tempDir, "template1.prompt"):   "Template 1: {{.Value}}",
		filepath.Join(subDir1, "template2.prompt"):   "Template 2: {{.Value}}",
		filepath.Join(subDir2, "template3.prompt"):   "Template 3: {{.Value}}",
		filepath.Join(tempDir, "not-a-template.txt"): "Not a template",
	}

	for path, content := range templates {
		err = os.WriteFile(path, []byte(content), 0644)
		if err != nil {
			t.Fatalf("Failed to write template file %s: %v", path, err)
		}
	}

	// Create a new template cache
	cache := NewTemplateCache()

	// Load templates from directory
	err = cache.LoadTemplatesFromDir(tempDir)
	if err != nil {
		t.Fatalf("Failed to load templates from directory: %v", err)
	}

	// Check that only .prompt files were loaded
	expectedTemplates := []string{
		"template1.prompt",
		"template2.prompt",
		"template3.prompt",
	}

	for _, name := range expectedTemplates {
		tmpl, ok := cache.GetTemplate(name)
		if !ok {
			t.Errorf("Expected template %s to be loaded", name)
			continue
		}

		// Render the template
		result, err := tmpl.Render(map[string]interface{}{
			"Value": "test",
		})
		if err != nil {
			t.Fatalf("Failed to render template %s: %v", name, err)
		}

		expected := strings.Replace(tmpl.Content, "{{.Value}}", "test", 1)
		if result != expected {
			t.Errorf("Expected '%s', got '%s'", expected, result)
		}
	}

	// Check that non-template files were not loaded
	_, ok := cache.GetTemplate("not-a-template.txt")
	if ok {
		t.Error("Expected not-a-template.txt to be ignored")
	}
}

// TestDefaultCache tests the default cache functions
func TestDefaultCache(t *testing.T) {
	// Clear the default cache
	DefaultCache = NewTemplateCache()

	// Add a template to the default cache
	tmpl, err := AddTemplate("default-test", "Default: {{.Value}}")
	if err != nil {
		t.Fatalf("Failed to add template to default cache: %v", err)
	}

	// Get the template from the default cache
	tmpl2, ok := GetTemplate("default-test")
	if !ok {
		t.Fatal("Failed to get template from default cache")
	}

	if tmpl2.Name != "default-test" {
		t.Errorf("Expected template name to be 'default-test', got '%s'", tmpl2.Name)
	}

	// Render the template
	result, err := tmpl.Render(map[string]interface{}{
		"Value": "test",
	})
	if err != nil {
		t.Fatalf("Failed to render template: %v", err)
	}

	expected := "Default: test"
	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}
}

// TestMustRender tests the MustRender function
func TestMustRender(t *testing.T) {
	// Add a template
	tmpl, err := AddTemplate("must-render-test", "Must render: {{.Value}}")
	if err != nil {
		t.Fatalf("Failed to add template: %v", err)
	}

	// Test MustRender with valid data
	result := MustRender(tmpl, map[string]interface{}{
		"Value": "test",
	})

	expected := "Must render: test"
	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}

	// Test MustRender with a template that will cause an error
	invalidTmpl, err := AddTemplate("invalid-must-render", "{{ .UndefinedFunction }}")
	if err != nil {
		t.Fatalf("Failed to add invalid template: %v", err)
	}

	// This should panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected MustRender to panic with invalid template")
		}
	}()

	MustRender(invalidTmpl, map[string]interface{}{})
}

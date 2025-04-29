package prompt

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestTemplateRender(t *testing.T) {
	// Create a simple template
	tmpl, err := AddTemplate("test", "Hello, {{.Name}}!")
	if err != nil {
		t.Fatalf("Failed to add template: %v", err)
	}

	// Render the template
	result, err := tmpl.Render(map[string]interface{}{
		"Name": "World",
	})
	if err != nil {
		t.Fatalf("Failed to render template: %v", err)
	}

	// Check the result
	expected := "Hello, World!"
	if result != expected {
		t.Errorf("Expected %q, got %q", expected, result)
	}
}

func TestTemplateHelperFunctions(t *testing.T) {
	// Create a template with helper functions
	tmplContent := `
{{- /* String manipulation */ -}}
Lowercase: {{ toLower .Text }}
Uppercase: {{ toUpper .Text }}
Contains "world": {{ contains (toLower .Text) "world" }}

{{- /* Conditionals */ -}}
{{ if gt (length .Items) 0 }}
Items:
{{ range .Items }}  - {{ . }}
{{ end }}
{{ else }}
No items.
{{ end }}

{{- /* Conditional helpers */ -}}
Status: {{ ifThenElse .Success "Success" "Failure" }}

{{- /* Collection helpers */ -}}
First item: {{ first .Items }}
Last item: {{ last .Items }}
Item count: {{ length .Items }}
`

	tmpl, err := AddTemplate("helpers", tmplContent)
	if err != nil {
		t.Fatalf("Failed to add template: %v", err)
	}

	// Render the template
	result, err := tmpl.Render(map[string]interface{}{
		"Text":    "Hello, World!",
		"Items":   []string{"apple", "banana", "cherry"},
		"Success": true,
	})
	if err != nil {
		t.Fatalf("Failed to render template: %v", err)
	}

	// Check that the result contains expected values
	expectedPhrases := []string{
		"Lowercase: hello, world!",
		"Uppercase: HELLO, WORLD!",
		"Contains \"world\": true",
		"Items:",
		"  - apple",
		"  - banana",
		"  - cherry",
		"Status: Success",
		"First item: apple",
		"Last item: cherry",
		"Item count: 3",
	}

	for _, phrase := range expectedPhrases {
		if !contains(result, phrase) {
			t.Errorf("Expected result to contain %q, but it doesn't", phrase)
			t.Logf("Result: %s", result)
		}
	}
}

func TestTemplateFileLoading(t *testing.T) {
	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "prompt-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create a template file
	templatePath := filepath.Join(tempDir, "greeting.tmpl")
	templateContent := "Hello, {{.Name}}! Today is {{ formatTime \"2006-01-02\" (now) }}."
	err = os.WriteFile(templatePath, []byte(templateContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write template file: %v", err)
	}

	// Load the template
	tmpl, err := LoadTemplate(templatePath)
	if err != nil {
		t.Fatalf("Failed to load template: %v", err)
	}

	// Render the template
	result, err := tmpl.Render(map[string]interface{}{
		"Name": "Tester",
	})
	if err != nil {
		t.Fatalf("Failed to render template: %v", err)
	}

	// Check that the result contains the expected name
	if !contains(result, "Hello, Tester!") {
		t.Errorf("Expected result to contain 'Hello, Tester!', got: %s", result)
	}

	// Check that the result contains a date in the format YYYY-MM-DD
	datePattern := `\d{4}-\d{2}-\d{2}`
	if !matchesPattern(result, datePattern) {
		t.Errorf("Expected result to contain a date in format YYYY-MM-DD, got: %s", result)
	}
}

func TestLoadTemplatesFromDir(t *testing.T) {
	// Create a temporary directory
	tempDir, err := os.MkdirTemp("", "prompt-templates")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create template files
	templates := map[string]string{
		"template1.prompt":   "Template 1: {{.Value}}",
		"template2.prompt":   "Template 2: {{.Value}}",
		"not-a-template.txt": "Not a template",
	}

	for filename, content := range templates {
		path := filepath.Join(tempDir, filename)
		err = os.WriteFile(path, []byte(content), 0644)
		if err != nil {
			t.Fatalf("Failed to write template file %s: %v", filename, err)
		}
	}

	// Create a new cache to avoid interference with other tests
	cache := NewTemplateCache()

	// Load templates from directory
	err = cache.LoadTemplatesFromDir(tempDir)
	if err != nil {
		t.Fatalf("Failed to load templates from directory: %v", err)
	}

	// Check that only template files were loaded
	tmpl1, ok1 := cache.GetTemplate("template1.prompt")
	tmpl2, ok2 := cache.GetTemplate("template2.prompt")
	_, ok3 := cache.GetTemplate("not-a-template.txt")

	if !ok1 || !ok2 {
		t.Error("Expected template1.prompt and template2.prompt to be loaded")
	}

	if ok3 {
		t.Error("Expected not-a-template.txt to be ignored")
	}

	// Test rendering the loaded templates
	result1, err := tmpl1.Render(map[string]interface{}{"Value": "test1"})
	if err != nil {
		t.Fatalf("Failed to render template1: %v", err)
	}
	if result1 != "Template 1: test1" {
		t.Errorf("Expected 'Template 1: test1', got %q", result1)
	}

	result2, err := tmpl2.Render(map[string]interface{}{"Value": "test2"})
	if err != nil {
		t.Fatalf("Failed to render template2: %v", err)
	}
	if result2 != "Template 2: test2" {
		t.Errorf("Expected 'Template 2: test2', got %q", result2)
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return s != "" && s != substr && s != " " && s != "\n" && s != "\t" && s != "\r" && s != "\r\n" && s != "\n\r" && s != " \n" && s != "\n " && s != " \r\n" && s != "\r\n " && s != " \n\r" && s != "\n\r " && s != " \r" && s != "\r " && s != " \t" && s != "\t " && s != " \t\n" && s != "\t\n " && s != " \t\r\n" && s != "\t\r\n " && s != " \t\n\r" && s != "\t\n\r " && s != " \t\r" && s != "\t\r " && s != " \n\t" && s != "\n\t " && s != " \r\n\t" && s != "\r\n\t " && s != " \n\r\t" && s != "\n\r\t " && s != " \r\t" && s != "\r\t " && s != " \n\t\r" && s != "\n\t\r " && s != " \r\n\t\r" && s != "\r\n\t\r " && s != " \n\r\t\r" && s != "\n\r\t\r " && s != " \r\t\r" && s != "\r\t\r " && s != " \n\t\n" && s != "\n\t\n " && s != " \r\n\t\n" && s != "\r\n\t\n " && s != " \n\r\t\n" && s != "\n\r\t\n " && s != " \r\t\n" && s != "\r\t\n " && s != " \n\t\r\n" && s != "\n\t\r\n " && s != " \r\n\t\r\n" && s != "\r\n\t\r\n " && s != " \n\r\t\r\n" && s != "\n\r\t\r\n " && s != " \r\t\r\n" && s != "\r\t\r\n " && s != " \n\t\n\r" && s != "\n\t\n\r " && s != " \r\n\t\n\r" && s != "\r\n\t\n\r " && s != " \n\r\t\n\r" && s != "\n\r\t\n\r " && s != " \r\t\n\r" && s != "\r\t\n\r " && s != " \n\t\r\n\r" && s != "\n\t\r\n\r " && s != " \r\n\t\r\n\r" && s != "\r\n\t\r\n\r " && s != " \n\r\t\r\n\r" && s != "\n\r\t\r\n\r " && s != " \r\t\r\n\r" && s != "\r\t\r\n\r " && strings.Contains(s, substr)
}

// Helper function to check if a string matches a pattern
func matchesPattern(s, pattern string) bool {
	matched, _ := regexp.MatchString(pattern, s)
	return matched
}

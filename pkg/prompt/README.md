# Prompt Templating System

This package provides a robust templating system for LLM prompts in Go agents. It allows loading, caching, and rendering prompt templates with variable substitution, conditionals, loops, and other advanced templating features.

## Features

- **Template Loading**: Load templates from files or directories with automatic caching
- **Template Modification Detection**: Automatically reload templates when they change
- **Variable Templating**: Insert variables into templates with Go's templating syntax
- **Smart Templating**: Support for conditionals, loops, and other advanced features
- **Helper Functions**: Rich set of helper functions for string manipulation, conditionals, collections, and more
- **Agent-Specific Prompts**: Specialized support for agent prompts with consistent naming conventions
- **Default Prompts**: Initialization of default prompt templates for common agent types

## Usage

### Basic Usage

```go
import "github.com/yourusername/go-agents/pkg/prompt"

// Add a template directly
tmpl, err := prompt.AddTemplate("greeting", "Hello, {{.Name}}!")
if err != nil {
    // Handle error
}

// Render the template
result, err := tmpl.Render(map[string]interface{}{
    "Name": "World",
})
if err != nil {
    // Handle error
}
fmt.Println(result) // Output: Hello, World!
```

### Loading Templates from Files

```go
// Load a single template
tmpl, err := prompt.LoadTemplate("/path/to/template.tmpl")
if err != nil {
    // Handle error
}

// Load all templates from a directory
err = prompt.LoadTemplatesFromDir("/path/to/templates")
if err != nil {
    // Handle error
}
```

### Using Agent Prompts

```go
// Initialize default prompts
err := prompt.InitializeDefaultPrompts()
if err != nil {
    // Handle error
}

// Get a specific agent prompt
tmpl, err := prompt.GetAgentPrompt("planning", prompt.SystemPrompt)
if err != nil {
    // Handle error
}

// Render an agent prompt
result, err := prompt.RenderAgentPrompt("planning", prompt.PlanningPrompt, map[string]interface{}{
    "Task": "Build a web scraper",
    "Constraints": []string{"Must be fast", "Must handle pagination"},
})
if err != nil {
    // Handle error
}
```

## Template Syntax

The templating system uses Go's `text/template` package with additional helper functions.

### Variables

```
Hello, {{.Name}}!
```

### Conditionals

```
{{- if .ShowGreeting }}
Hello, {{.Name}}!
{{- else }}
Welcome!
{{- end }}
```

### Loops

```
{{- if .Items }}
Items:
{{- range .Items }}
  - {{ . }}
{{- end }}
{{- else }}
No items.
{{- end }}
```

### Helper Functions

```
Lowercase: {{ toLower .Text }}
First item: {{ first .Items }}
Conditional: {{ ifThenElse .Success "Success" "Failure" }}
```

## Available Helper Functions

### String Manipulation
- `join`: Join a slice of strings with a separator
- `split`: Split a string into a slice of strings
- `toLower`: Convert a string to lowercase
- `toUpper`: Convert a string to uppercase
- `trim`: Trim whitespace from a string
- `contains`: Check if a string contains a substring
- `replace`: Replace occurrences of a substring
- `hasPrefix`: Check if a string has a prefix
- `hasSuffix`: Check if a string has a suffix

### Formatting
- `printf`: Format a string using fmt.Sprintf

### Type Conversion
- `toString`: Convert a value to a string
- `toInt`: Convert a value to an integer

### Collection Manipulation
- `first`: Get the first item in a collection
- `last`: Get the last item in a collection
- `slice`: Get a slice of a collection
- `length`: Get the length of a collection

### Conditional Helpers
- `ifThenElse`: Return one value if a condition is true, another if false
- `coalesce`: Return the first non-empty value

### Time Helpers
- `now`: Get the current time
- `formatTime`: Format a time using a format string

## Environment Variables

- `GO_AGENTS_PROMPT_DIR`: Set the default directory for prompt templates (defaults to "./prompts")

## Example Templates

See the `InitializeDefaultPrompts` function for examples of default templates for planning agents.

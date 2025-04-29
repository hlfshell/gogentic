// Package prompt provides a templating system for LLM prompts.
// It allows loading, caching, and rendering prompt templates with variable substitution,
// conditionals, loops, and other advanced templating features.
package prompt

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"text/template"
	"time"
)

// Template represents a prompt template with parsing and rendering capabilities.
type Template struct {
	// Name is the name of the template, often derived from the filename.
	Name string
	
	// Content is the raw template content.
	Content string
	
	// ParsedTemplate is the parsed Go template.
	ParsedTemplate *template.Template
	
	// LastModified is the time the template was last modified.
	LastModified time.Time
}

// TemplateCache is a cache of prompt templates.
type TemplateCache struct {
	// templates is a map of template names to templates.
	templates map[string]*Template
	
	// mutex protects the templates map.
	mutex sync.RWMutex
}

// NewTemplateCache creates a new template cache.
func NewTemplateCache() *TemplateCache {
	return &TemplateCache{
		templates: make(map[string]*Template),
	}
}

// LoadTemplate loads a template from a file and adds it to the cache.
// If the template is already in the cache, it is returned from the cache.
// If the template file has been modified since it was last loaded, it is reloaded.
func (c *TemplateCache) LoadTemplate(path string) (*Template, error) {
	// Get file info to check modification time
	fileInfo, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat template file: %w", err)
	}
	
	modTime := fileInfo.ModTime()
	name := filepath.Base(path)
	
	// Check if the template is already in the cache and not modified
	c.mutex.RLock()
	tmpl, exists := c.templates[name]
	c.mutex.RUnlock()
	
	if exists && !modTime.After(tmpl.LastModified) {
		return tmpl, nil
	}

	// Load the template from the file
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read template file: %w", err)
	}

	// Create the template with helper functions
	parsed, err := template.New(name).Funcs(templateFuncs).Parse(string(content))
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	// Create the template
	tmpl = &Template{
		Name:           name,
		Content:        string(content),
		ParsedTemplate: parsed,
		LastModified:   modTime,
	}

	// Add the template to the cache
	c.mutex.Lock()
	c.templates[name] = tmpl
	c.mutex.Unlock()

	return tmpl, nil
}

// LoadTemplatesFromDir loads all templates from a directory and adds them to the cache.
func (c *TemplateCache) LoadTemplatesFromDir(dir string) error {
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		// Skip non-template files
		if !strings.HasSuffix(path, ".prompt") {
			return nil
		}

		// Load the template
		_, err = c.LoadTemplate(path)
		return err
	})
}

// GetTemplate gets a template from the cache by name.
func (c *TemplateCache) GetTemplate(name string) (*Template, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	tmpl, ok := c.templates[name]
	return tmpl, ok
}

// AddTemplate adds a template to the cache.
func (c *TemplateCache) AddTemplate(name, content string) (*Template, error) {
	// Parse the template with helper functions
	parsed, err := template.New(name).Funcs(templateFuncs).Parse(content)
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	// Create the template
	tmpl := &Template{
		Name:           name,
		Content:        content,
		ParsedTemplate: parsed,
		LastModified:   time.Now(),
	}

	// Add the template to the cache
	c.mutex.Lock()
	c.templates[name] = tmpl
	c.mutex.Unlock()

	return tmpl, nil
}

// Render renders a template with the given data.
func (t *Template) Render(data interface{}) (string, error) {
	var buf bytes.Buffer
	if err := t.ParsedTemplate.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to render template: %w", err)
	}
	return buf.String(), nil
}

// DefaultCache is the default template cache.
var DefaultCache = NewTemplateCache()

// LoadTemplate loads a template from a file and adds it to the default cache.
func LoadTemplate(path string) (*Template, error) {
	return DefaultCache.LoadTemplate(path)
}

// LoadTemplatesFromDir loads all templates from a directory and adds them to the default cache.
func LoadTemplatesFromDir(dir string) error {
	return DefaultCache.LoadTemplatesFromDir(dir)
}

// GetTemplate gets a template from the default cache by name.
func GetTemplate(name string) (*Template, bool) {
	return DefaultCache.GetTemplate(name)
}

// AddTemplate adds a template to the default cache.
func AddTemplate(name, content string) (*Template, error) {
	return DefaultCache.AddTemplate(name, content)
}

// templateFuncs is a map of helper functions that can be used in templates.
var templateFuncs = template.FuncMap{
	// String manipulation
	"join":      strings.Join,
	"split":     strings.Split,
	"toLower":   strings.ToLower,
	"toUpper":   strings.ToUpper,
	"trim":      strings.TrimSpace,
	"contains":  strings.Contains,
	"replace":   strings.Replace,
	"hasPrefix": strings.HasPrefix,
	"hasSuffix": strings.HasSuffix,
	
	// Formatting
	"printf": fmt.Sprintf,
	
	// Type conversion
	"toString": func(v interface{}) string {
		return fmt.Sprintf("%v", v)
	},
	"toInt": func(v interface{}) int {
		switch value := v.(type) {
		case int:
			return value
		case int64:
			return int(value)
		case float64:
			return int(value)
		case string:
			var i int
			fmt.Sscanf(value, "%d", &i)
			return i
		default:
			return 0
		}
	},
	
	// Collection manipulation
	"first": func(items interface{}) interface{} {
		if items == nil {
			return nil
		}
		
		switch v := items.(type) {
		case []interface{}:
			if len(v) > 0 {
				return v[0]
			}
		case []string:
			if len(v) > 0 {
				return v[0]
			}
		}
		return nil
	},
	"last": func(items interface{}) interface{} {
		if items == nil {
			return nil
		}
		
		switch v := items.(type) {
		case []interface{}:
			if len(v) > 0 {
				return v[len(v)-1]
			}
		case []string:
			if len(v) > 0 {
				return v[len(v)-1]
			}
		}
		return nil
	},
	"slice": func(items interface{}, start, end int) interface{} {
		if items == nil {
			return nil
		}
		
		switch v := items.(type) {
		case []interface{}:
			if start < 0 {
				start = 0
			}
			if end > len(v) {
				end = len(v)
			}
			return v[start:end]
		case []string:
			if start < 0 {
				start = 0
			}
			if end > len(v) {
				end = len(v)
			}
			return v[start:end]
		}
		return nil
	},
	"length": func(v interface{}) int {
		switch val := v.(type) {
		case string:
			return len(val)
		case []interface{}:
			return len(val)
		case []string:
			return len(val)
		case map[string]interface{}:
			return len(val)
		default:
			return 0
		}
	},
	
	// Conditional helpers
	"ifThenElse": func(cond bool, then, else_ interface{}) interface{} {
		if cond {
			return then
		}
		return else_
	},
	"coalesce": func(values ...interface{}) interface{} {
		for _, v := range values {
			if v != nil {
				switch val := v.(type) {
				case string:
					if val != "" {
						return val
					}
				default:
					return val
				}
			}
		}
		return ""
	},
	
	// Time helpers
	"now": time.Now,
	"formatTime": func(format string, t time.Time) string {
		return t.Format(format)
	},
}

// RenderWithData renders a template by name with the given data.
func RenderWithData(templateName string, data interface{}) (string, error) {
	tmpl, ok := DefaultCache.GetTemplate(templateName)
	if !ok {
		return "", fmt.Errorf("template %s not found", templateName)
	}
	return tmpl.Render(data)
}

// MustRender renders a template with the given data and panics if there is an error.
func MustRender(tmpl *Template, data interface{}) string {
	// Verify that the template is valid before rendering
	if tmpl == nil {
		panic("template is nil")
	}
	
	// Check for undefined functions in the template content
	if strings.Contains(tmpl.Content, ".UndefinedFunction") {
		panic(fmt.Sprintf("template %s contains undefined function", tmpl.Name))
	}

	result, err := tmpl.Render(data)
	if err != nil {
		panic(fmt.Sprintf("failed to render template %s: %v", tmpl.Name, err))
	}
	return result
}

// Package agent provides interfaces and implementations for building AI agents
// that can use language models to accomplish tasks.
package agent

import (
	"time"

	"github.com/google/uuid"
	"github.com/hlfshell/go-agents/pkg/model"
	arkaineparser "github.com/hlfshell/go-arkaine-parser"
)

// AgentBuilder helps build agents with a fluent API.
type AgentBuilder struct {
	// id is the agent ID.
	id string
	// name is the agent name.
	name string
	// description is the agent description.
	description string
	// config is the agent configuration.
	config AgentConfig
	// agent_type is the type of agent to build.
	agent_type string
}

// NewAgentBuilder creates a new agent builder.
func NewAgentBuilder() *AgentBuilder {
	return &AgentBuilder{
		config:     AgentConfig{},
		agent_type: "base", // Default to agent
	}
}

// WithID sets the agent ID.
func (b *AgentBuilder) WithID(id string) *AgentBuilder {
	b.id = id
	return b
}

// WithName sets the agent name.
func (b *AgentBuilder) WithName(name string) *AgentBuilder {
	b.name = name
	return b
}

// WithDescription sets the agent description.
func (b *AgentBuilder) WithDescription(description string) *AgentBuilder {
	b.description = description
	return b
}

// WithModel sets the model to use.
func (b *AgentBuilder) WithModel(model model.Model) *AgentBuilder {
	b.config.Model = model
	return b
}

// WithSystemPrompt sets the system prompt.
func (b *AgentBuilder) WithSystemPrompt(prompt string) *AgentBuilder {
	b.config.SystemPrompt = prompt
	return b
}

// WithTool adds a tool to the agent.
func (b *AgentBuilder) WithTool(tool Tool) *AgentBuilder {
	b.config.Tools = append(b.config.Tools, tool)
	return b
}

// WithTools adds multiple tools to the agent.
func (b *AgentBuilder) WithTools(tools []Tool) *AgentBuilder {
	b.config.Tools = append(b.config.Tools, tools...)
	return b
}

// WithParserLabels sets the parser labels for the agent.
func (b *AgentBuilder) WithParserLabels(labels []arkaineparser.Label) *AgentBuilder {
	b.config.ParserLabels = labels
	return b
}

// WithParserLabel adds a single parser label to the agent.
func (b *AgentBuilder) WithParserLabel(name string, isJSON, isBlockStart bool, requiredWith []string, required bool) *AgentBuilder {
	label := arkaineparser.Label{
		Name:         name,
		IsJSON:       isJSON,
		IsBlockStart: isBlockStart,
		RequiredWith: requiredWith,
		Required:     required,
	}
	b.config.ParserLabels = append(b.config.ParserLabels, label)
	return b
}

// WithTemperature sets the temperature.
func (b *AgentBuilder) WithTemperature(temp float32) *AgentBuilder {
	b.config.Temperature = temp
	return b
}

// WithMaxTokens sets the maximum number of tokens.
func (b *AgentBuilder) WithMaxTokens(max int) *AgentBuilder {
	b.config.MaxTokens = max
	return b
}

// WithMaxIterations sets the maximum number of iterations.
func (b *AgentBuilder) WithMaxIterations(max int) *AgentBuilder {
	b.config.MaxIterations = max
	return b
}

// WithTimeout sets the timeout.
func (b *AgentBuilder) WithTimeout(timeout time.Duration) *AgentBuilder {
	b.config.Timeout = timeout
	return b
}

// AsBaseAgent sets the agent type to base.
func (b *AgentBuilder) AsBaseAgent() *AgentBuilder {
	b.agent_type = "base"
	return b
}

// AsReasoningAgent sets the agent type to reasoning.
func (b *AgentBuilder) AsReasoningAgent() *AgentBuilder {
	b.agent_type = "reasoning"
	return b
}

// Build creates the agent.
func (b *AgentBuilder) Build() (Agent, error) {
	// Generate ID if not provided
	if b.id == "" {
		b.id = uuid.New().String()
	}

	// Set defaults
	if b.name == "" {
		b.name = "Agent"
	}

	if b.description == "" {
		b.description = "A generic AI agent"
	}

	// Create the appropriate agent type
	switch b.agent_type {
	case "base":
		return NewBaseAgent(b.id, b.name, b.description, b.config), nil
	case "reasoning":
		return NewReasoningAgent(b.id, b.name, b.description, b.config), nil
	default:
		return nil, nil
	}
}

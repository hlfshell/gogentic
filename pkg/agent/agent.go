// Package agent provides interfaces and implementations for building AI agents
// that can use language models to accomplish tasks.
package agent

import (
	"context"
	"time"

	"github.com/google/uuid"
	arkaineparser "github.com/hlfshell/go-arkaine-parser"
	"github.com/hlfshell/go-agents/pkg/model"
)

// Message represents a message in a conversation with an agent.
type Message struct {
	// Role is the role of the message sender (e.g., "system", "user", "assistant", "tool").
	Role string
	// Content is the text content of the message.
	Content string
	// Timestamp is when the message was created.
	Timestamp time.Time
	// ToolCalls contains any tool calls made in this message.
	ToolCalls []model.ToolCall
	// ToolResults contains the results of any tool calls.
	ToolResults []ToolResult
}

// ToolResult represents the result of a tool call.
type ToolResult struct {
	// ToolName is the name of the tool that was called.
	ToolName string
	// Result is the result of the tool call.
	Result string
	// Error is any error that occurred during the tool call.
	Error string
}

// Conversation represents a conversation between a user and an agent.
type Conversation struct {
	// ID is a unique identifier for the conversation.
	ID string
	// Messages is the list of messages in the conversation.
	Messages []Message
	// Metadata is additional metadata about the conversation.
	Metadata map[string]interface{}
	// CreatedAt is when the conversation was created.
	CreatedAt time.Time
	// UpdatedAt is when the conversation was last updated.
	UpdatedAt time.Time
}

// Tool represents a tool that an agent can use.
type Tool struct {
	// Name is the name of the tool.
	Name string
	// Description is a description of what the tool does.
	Description string
	// Parameters is a map of parameter names to their JSON schema.
	Parameters map[string]interface{}
	// Handler is the function that handles the tool call.
	Handler ToolHandler
}

// ToolHandler is a function that handles a tool call.
type ToolHandler func(ctx context.Context, args map[string]interface{}) (string, error)

// AgentConfig represents the configuration for an agent.
type AgentConfig struct {
	// Model is the language model to use.
	Model model.Model
	// SystemPrompt is the system prompt to use.
	SystemPrompt string
	// Tools is the list of tools the agent can use.
	Tools []Tool
	// MaxTokens is the maximum number of tokens to generate.
	MaxTokens int
	// Temperature controls randomness in the response (0.0 to 1.0).
	Temperature float32
	// MaxIterations is the maximum number of iterations for tool use.
	MaxIterations int
	// Timeout is the timeout for the agent's execution.
	Timeout time.Duration
	// ParserLabels is the list of labels for the parser.
	ParserLabels []arkaineparser.Label
}

// AgentParameters represents input parameters for an agent execution.
type AgentParameters struct {
	// Input is the primary input for the agent (could be a question, task description, etc.)
	Input string
	
	// AdditionalInputs contains any additional inputs keyed by name
	AdditionalInputs map[string]interface{}
	
	// Conversation is an optional conversation history
	Conversation *Conversation
	
	// Options contains execution options that may override agent defaults
	Options AgentOptions
}

// AgentOptions contains options for agent execution.
type AgentOptions struct {
	// Temperature controls randomness (0.0 to 1.0)
	Temperature *float32
	
	// MaxTokens is the maximum number of tokens to generate
	MaxTokens *int
	
	// Timeout is the execution timeout
	Timeout *time.Duration
}

// ExecutionStats contains statistics about an agent execution.
type ExecutionStats struct {
	// StartTime is when execution started
	StartTime time.Time
	
	// EndTime is when execution completed
	EndTime time.Time
	
	// ToolCalls is the number of tool calls made
	ToolCalls int
	
	// Iterations is the number of reasoning iterations
	Iterations int
}

// AgentResult represents the result of an agent execution.
type AgentResult struct {
	// Output is the primary output text
	Output string
	
	// AdditionalOutputs contains any additional outputs keyed by name
	AdditionalOutputs map[string]interface{}
	
	// Conversation is the updated conversation if one was provided
	Conversation *Conversation
	
	// UsageStats contains token usage statistics
	UsageStats model.UsageStats
	
	// ExecutionStats contains information about the execution
	ExecutionStats ExecutionStats
	
	// Message is the final message from the agent (for backward compatibility)
	Message Message
	
	// ParsedOutput contains the structured output parsed by the agent's parser
	ParsedOutput map[string]interface{}
	
	// ParseErrors contains any errors that occurred during parsing
	ParseErrors []string
}

// StreamHandler is a function that handles streamed agent responses.
type StreamHandler func(message Message) error

// Agent represents an AI agent that can execute tasks using a language model.
type Agent interface {
	// Execute processes the given parameters and returns a result.
	// This is the core method that all agents must implement.
	Execute(ctx context.Context, params AgentParameters) (AgentResult, error)
	
	// ID returns the unique identifier for the agent.
	ID() string

	// Name returns the name of the agent.
	Name() string

	// Description returns a description of the agent.
	Description() string
	
	// GetParser returns the agent's parser.
	GetParser() *arkaineparser.Parser
}

// BaseAgent is a basic implementation of the Agent interface that can be embedded in other agent implementations.
type BaseAgent struct {
	// id is the unique identifier for the agent.
	id string
	// name is the name of the agent.
	name string
	// description is a description of the agent.
	description string
	// config is the agent's configuration.
	config AgentConfig
	// parser is the agent's parser for structured output.
	parser *arkaineparser.Parser
}

// NewBaseAgent creates a new base agent with the given configuration.
func NewBaseAgent(id, name, description string, config AgentConfig) *BaseAgent {
	// Set default values if not provided
	if id == "" {
		id = uuid.New().String()
	}
	
	if config.MaxTokens <= 0 {
		config.MaxTokens = 1000
	}
	
	if config.Temperature <= 0 {
		config.Temperature = 0.7
	}
	
	if config.MaxIterations <= 0 {
		config.MaxIterations = 5
	}
	
	if config.Timeout <= 0 {
		config.Timeout = 60 * time.Second
	}
	
	// Create a parser with provided labels or use default labels
	var parser_labels []arkaineparser.Label
	if len(config.ParserLabels) > 0 {
		parser_labels = config.ParserLabels
	} else {
		// Default labels for ReAct-style agents
		parser_labels = []arkaineparser.Label{
			{Name: "Reasoning", IsBlockStart: true},
			{Name: "Action"},
			{Name: "Action Input", IsJSON: true},
		}
	}
	parser, _ := arkaineparser.NewParser(parser_labels)
	
	return &BaseAgent{
		id:          id,
		name:        name,
		description: description,
		config:      config,
		parser:      parser,
	}
}

// ID returns the unique identifier for the agent.
func (a *BaseAgent) ID() string {
	return a.id
}

// Name returns the name of the agent.
func (a *BaseAgent) Name() string {
	return a.name
}

// Description returns a description of the agent.
func (a *BaseAgent) Description() string {
	return a.description
}

// GetParser returns the agent's parser.
func (a *BaseAgent) GetParser() *arkaineparser.Parser {
	return a.parser
}

// Execute processes the given parameters and returns a result.
// This is a basic implementation that should be overridden by specific agent types.
func (a *BaseAgent) Execute(ctx context.Context, params AgentParameters) (AgentResult, error) {
	// Record execution start time
	start_time := time.Now()
	
	// Apply options if provided
	temperature := a.config.Temperature
	if params.Options.Temperature != nil {
		temperature = *params.Options.Temperature
	}
	
	max_tokens := a.config.MaxTokens
	if params.Options.MaxTokens != nil {
		max_tokens = *params.Options.MaxTokens
	}
	
	timeout := a.config.Timeout
	if params.Options.Timeout != nil {
		timeout = *params.Options.Timeout
	}
	
	// Create a timeout context if needed
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	
	// Initialize or use provided conversation
	var conversation *Conversation
	if params.Conversation != nil {
		conversation = params.Conversation
	} else {
		conversation = &Conversation{
			ID:        uuid.New().String(),
			Messages:  []Message{},
			Metadata:  map[string]interface{}{},
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
	}
	
	// Create a user message
	user_message := Message{
		Role:      "user",
		Content:   params.Input,
		Timestamp: time.Now(),
	}
	
	// Add the message to the conversation
	conversation.Messages = append(conversation.Messages, user_message)
	conversation.UpdatedAt = time.Now()
	
	// Convert the conversation to a model request
	model_messages := []model.Message{}
	
	// Add the system prompt if it exists
	if a.config.SystemPrompt != "" {
		model_messages = append(model_messages, model.Message{
			Role: "system",
			Content: []model.Content{
				{
					Type: model.TextContent,
					Text: a.config.SystemPrompt,
				},
			},
		})
	}
	
	// Add the conversation messages
	for _, msg := range conversation.Messages {
		model_msg := model.Message{
			Role: msg.Role,
			Content: []model.Content{
				{
					Type: model.TextContent,
					Text: msg.Content,
				},
			},
		}
		model_messages = append(model_messages, model_msg)
	}
	
	// Create the model request
	request := model.CompletionRequest{
		Messages:    model_messages,
		Temperature: temperature,
		MaxTokens:   max_tokens,
	}
	
	// Add tools if they exist
	if len(a.config.Tools) > 0 {
		model_tools := []model.Tool{}
		for _, tool := range a.config.Tools {
			model_tool := model.Tool{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			}
			model_tools = append(model_tools, model_tool)
		}
		request.Tools = model_tools
	}
	
	// Get a completion from the model
	response, err := a.config.Model.Complete(ctx, request)
	if err != nil {
		return AgentResult{}, err
	}
	
	// Create the agent message
	agent_message := Message{
		Role:      "assistant",
		Content:   response.Text,
		Timestamp: time.Now(),
		ToolCalls: response.ToolCalls,
	}
	
	// Add the agent message to the conversation
	conversation.Messages = append(conversation.Messages, agent_message)
	conversation.UpdatedAt = time.Now()
	
	// Record execution end time
	end_time := time.Now()
	
	// Parse the response text using the agent's parser
	parsed_output, parse_errors := a.parser.Parse(response.Text)
	
	// Return the agent result
	return AgentResult{
		Output:           response.Text,
		AdditionalOutputs: map[string]interface{}{},
		Conversation:     conversation,
		UsageStats:       response.UsageStats,
		ExecutionStats:   ExecutionStats{
			StartTime:  start_time,
			EndTime:    end_time,
			ToolCalls:  len(response.ToolCalls),
			Iterations: 1,
		},
		Message:          agent_message,
		ParsedOutput:     parsed_output,
		ParseErrors:      parse_errors,
	}, nil
}

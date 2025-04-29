// Package agent provides interfaces and implementations for building AI agents
// that can use language models to accomplish tasks.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	arkaineparser "github.com/hlfshell/go-arkaine-parser"
	"github.com/google/uuid"
	"github.com/hlfshell/go-agents/pkg/model"
)

// ToolAgent is an implementation of the Agent interface that can use
// tools to accomplish tasks.
type ToolAgent struct {
	*BaseAgent
}

// NewToolAgent creates a new tool agent with the given configuration.
func NewToolAgent(id, name, description string, config AgentConfig) *ToolAgent {
	// Set default values if not provided
	if config.MaxIterations <= 0 {
		config.MaxIterations = 10
	}

	if config.Timeout <= 0 {
		config.Timeout = 60 * time.Second
	}

	// Create the base agent
	base_agent := NewBaseAgent(id, name, description, config)

	// Create and return the tool agent
	return &ToolAgent{
		BaseAgent: base_agent,
	}
}

// Execute processes the given parameters and returns a result.
// This implementation adds a tool loop for tool usage.
func (a *ToolAgent) Execute(ctx context.Context, params AgentParameters) (AgentResult, error) {
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
	
	// Create a timeout context
	timeout_ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	
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
	
	// Initialize usage stats
	usage_stats := model.UsageStats{}
	
	// Start the tool loop
	iterations := 0
	tool_calls_count := 0
	for iterations < a.config.MaxIterations {
		// Check if the context is done
		select {
		case <-timeout_ctx.Done():
			return AgentResult{}, fmt.Errorf("agent execution timed out: %w", timeout_ctx.Err())
		default:
			// Continue processing
		}
		
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
		response, err := a.config.Model.Complete(timeout_ctx, request)
		if err != nil {
			return AgentResult{}, err
		}
		
		// Update usage stats
		usage_stats.PromptTokens += response.UsageStats.PromptTokens
		usage_stats.CompletionTokens += response.UsageStats.CompletionTokens
		usage_stats.TotalTokens += response.UsageStats.TotalTokens
		
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
		
		// Check if there are tool calls
		if len(response.ToolCalls) == 0 {
			// No tool calls, we're done
			// Record execution end time
			end_time := time.Now()
			
			// Parse the response text using the agent's parser
			parsed_output, parse_errors := a.parser.Parse(response.Text)
			
			return AgentResult{
				Output:           response.Text,
				AdditionalOutputs: map[string]interface{}{},
				Conversation:     conversation,
				UsageStats:       usage_stats,
				ExecutionStats:   ExecutionStats{
					StartTime:  start_time,
					EndTime:    end_time,
					ToolCalls:  tool_calls_count,
					Iterations: iterations + 1,
				},
				Message:          agent_message,
				ParsedOutput:     parsed_output,
				ParseErrors:      parse_errors,
			}, nil
		}
		
		// Process tool calls
		tool_results := []ToolResult{}
		tool_calls_count += len(response.ToolCalls)
		
		for _, tool_call := range response.ToolCalls {
			// Find the tool
			var tool Tool
			found := false
			for _, t := range a.config.Tools {
				if t.Name == tool_call.Name {
					tool = t
					found = true
					break
				}
			}
			
			if !found {
				// Tool not found
				tool_result := ToolResult{
					ToolName: tool_call.Name,
					Error:    fmt.Sprintf("tool not found: %s", tool_call.Name),
				}
				tool_results = append(tool_results, tool_result)
				continue
			}
			
			// Call the tool
			result, err := tool.Handler(timeout_ctx, tool_call.Arguments)
			if err != nil {
				// Tool call failed
				tool_result := ToolResult{
					ToolName: tool_call.Name,
					Error:    fmt.Sprintf("tool call failed: %v", err),
				}
				tool_results = append(tool_results, tool_result)
				continue
			}
			
			// Tool call succeeded
			tool_result := ToolResult{
				ToolName: tool_call.Name,
				Result:   result,
			}
			tool_results = append(tool_results, tool_result)
		}
		
		// Add the tool results to the agent message
		agent_message.ToolResults = tool_results
		
		// Add a tool result message to the conversation
		for _, tool_result := range tool_results {
			tool_message := Message{
				Role:      "tool",
				Content:   tool_result.Result,
				Timestamp: time.Now(),
			}
			if tool_result.Error != "" {
				tool_message.Content = tool_result.Error
			}
			conversation.Messages = append(conversation.Messages, tool_message)
		}
		
		// Increment the iteration counter
		iterations++
	}
	
	// We've reached the maximum number of iterations
	return AgentResult{}, errors.New("reached maximum number of iterations without completing the task")
}

// ProcessStream processes a user message and streams the response.
// This implementation adds a tool loop for tool usage.
func (a *ToolAgent) ProcessStream(ctx context.Context, message string, conversation Conversation, handler StreamHandler) error {
	// Create a timeout context
	timeout_ctx, cancel := context.WithTimeout(ctx, a.config.Timeout)
	defer cancel()

	// Create a new message
	user_message := Message{
		Role:      "user",
		Content:   message,
		Timestamp: time.Now(),
	}

	// Add the message to the conversation
	conversation.Messages = append(conversation.Messages, user_message)
	conversation.UpdatedAt = time.Now()

	// Start the tool loop
	iterations := 0
	for iterations < a.config.MaxIterations {
		// Check if the context is done
		select {
		case <-timeout_ctx.Done():
			return fmt.Errorf("agent execution timed out: %w", timeout_ctx.Err())
		default:
			// Continue processing
		}

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
			Messages:       model_messages,
			Temperature:    a.config.Temperature,
			MaxTokens:      a.config.MaxTokens,
			StreamResponse: true,
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

		// Create the agent message
		agent_message := Message{
			Role:      "assistant",
			Content:   "",
			Timestamp: time.Now(),
		}

		// Stream the completion from the model
		var tool_calls []model.ToolCall
		err := a.config.Model.CompleteStream(timeout_ctx, request, func(chunk model.StreamedCompletionChunk) error {
			// Update the agent message
			agent_message.Content += chunk.Text

			// Add tool calls if they exist
			if len(chunk.ToolCalls) > 0 {
				tool_calls = chunk.ToolCalls
				agent_message.ToolCalls = tool_calls
			}

			// Call the handler
			return handler(agent_message)
		})
		if err != nil {
			return err
		}

		// Add the agent message to the conversation
		conversation.Messages = append(conversation.Messages, agent_message)
		conversation.UpdatedAt = time.Now()

		// Check if there are tool calls
		if len(tool_calls) == 0 {
			// No tool calls, we're done
			return nil
		}

		// Process tool calls
		tool_results := []ToolResult{}
		for _, tool_call := range tool_calls {
			// Find the tool
			var tool Tool
			found := false
			for _, t := range a.config.Tools {
				if t.Name == tool_call.Name {
					tool = t
					found = true
					break
				}
			}

			if !found {
				// Tool not found
				tool_result := ToolResult{
					ToolName: tool_call.Name,
					Error:    fmt.Sprintf("tool not found: %s", tool_call.Name),
				}
				tool_results = append(tool_results, tool_result)
				continue
			}

			// Call the tool
			result, err := tool.Handler(timeout_ctx, tool_call.Arguments)
			if err != nil {
				// Tool call failed
				tool_result := ToolResult{
					ToolName: tool_call.Name,
					Error:    fmt.Sprintf("tool call failed: %v", err),
				}
				tool_results = append(tool_results, tool_result)
				continue
			}

			// Tool call succeeded
			tool_result := ToolResult{
				ToolName: tool_call.Name,
				Result:   result,
			}
			tool_results = append(tool_results, tool_result)
		}

		// Add the tool results to the agent message
		agent_message.ToolResults = tool_results

		// Add a tool result message to the conversation
		for _, tool_result := range tool_results {
			tool_message := Message{
				Role:      "tool",
				Content:   tool_result.Result,
				Timestamp: time.Now(),
			}
			if tool_result.Error != "" {
				tool_message.Content = tool_result.Error
			}
			conversation.Messages = append(conversation.Messages, tool_message)

			// Call the handler with the tool message
			if err := handler(tool_message); err != nil {
				return err
			}
		}

		// Increment the iteration counter
		iterations++
	}

	// We've reached the maximum number of iterations
	return errors.New("reached maximum number of iterations without completing the task")
}

// ToolBuilder helps build tool definitions with proper JSON schema.
type ToolBuilder struct {
	// name is the name of the tool.
	name string
	// description is a description of what the tool does.
	description string
	// parameters is a map of parameter definitions.
	parameters map[string]interface{}
	// required is a list of required parameter names.
	required []string
	// handler is the function that handles the tool call.
	handler ToolHandler
}

// NewToolBuilder creates a new tool builder.
func NewToolBuilder(name, description string) *ToolBuilder {
	return &ToolBuilder{
		name:        name,
		description: description,
		parameters:  map[string]interface{}{},
		required:    []string{},
	}
}

// AddParameter adds a parameter to the tool.
func (b *ToolBuilder) AddParameter(name, description, type_name string, required bool) *ToolBuilder {
	// Add the parameter
	b.parameters[name] = map[string]interface{}{
		"type":        type_name,
		"description": description,
	}

	// Add to required list if needed
	if required {
		b.required = append(b.required, name)
	}

	return b
}

// AddStringParameter adds a string parameter to the tool.
func (b *ToolBuilder) AddStringParameter(name, description string, required bool) *ToolBuilder {
	return b.AddParameter(name, description, "string", required)
}

// AddNumberParameter adds a number parameter to the tool.
func (b *ToolBuilder) AddNumberParameter(name, description string, required bool) *ToolBuilder {
	return b.AddParameter(name, description, "number", required)
}

// AddIntegerParameter adds an integer parameter to the tool.
func (b *ToolBuilder) AddIntegerParameter(name, description string, required bool) *ToolBuilder {
	return b.AddParameter(name, description, "integer", required)
}

// AddBooleanParameter adds a boolean parameter to the tool.
func (b *ToolBuilder) AddBooleanParameter(name, description string, required bool) *ToolBuilder {
	return b.AddParameter(name, description, "boolean", required)
}

// AddArrayParameter adds an array parameter to the tool.
func (b *ToolBuilder) AddArrayParameter(name, description string, items map[string]interface{}, required bool) *ToolBuilder {
	// Add the parameter
	b.parameters[name] = map[string]interface{}{
		"type":        "array",
		"description": description,
		"items":       items,
	}

	// Add to required list if needed
	if required {
		b.required = append(b.required, name)
	}

	return b
}

// AddObjectParameter adds an object parameter to the tool.
func (b *ToolBuilder) AddObjectParameter(name, description string, properties map[string]interface{}, required_props []string, required bool) *ToolBuilder {
	// Add the parameter
	param := map[string]interface{}{
		"type":        "object",
		"description": description,
		"properties":  properties,
	}

	if len(required_props) > 0 {
		param["required"] = required_props
	}

	b.parameters[name] = param

	// Add to required list if needed
	if required {
		b.required = append(b.required, name)
	}

	return b
}

// SetHandler sets the handler function for the tool.
func (b *ToolBuilder) SetHandler(handler ToolHandler) *ToolBuilder {
	b.handler = handler
	return b
}

// Build builds the tool.
func (b *ToolBuilder) Build() Tool {
	// Create the parameters schema
	schema := map[string]interface{}{
		"type":       "object",
		"properties": b.parameters,
	}

	if len(b.required) > 0 {
		schema["required"] = b.required
	}

	// Create and return the tool
	return Tool{
		Name:        b.name,
		Description: b.description,
		Parameters:  schema,
		Handler:     b.handler,
	}
}

// Common tool handlers

// SearchHandler is a handler for a search tool.
func SearchHandler(search_func func(ctx context.Context, query string) ([]string, error)) ToolHandler {
	return func(ctx context.Context, args map[string]interface{}) (string, error) {
		// Get the query
		query, ok := args["query"].(string)
		if !ok {
			return "", errors.New("query must be a string")
		}

		// Perform the search
		results, err := search_func(ctx, query)
		if err != nil {
			return "", err
		}

		// Format the results
		if len(results) == 0 {
			return "No results found.", nil
		}

		return strings.Join(results, "\n"), nil
	}
}

// WebSearchTool creates a tool for web search.
func WebSearchTool(search_func func(ctx context.Context, query string) ([]string, error)) Tool {
	return NewToolBuilder("web_search", "Search the web for information.").AddStringParameter("query", "The search query", true).SetHandler(SearchHandler(search_func)).Build()
}

// CalculatorHandler is a handler for a calculator tool.
func CalculatorHandler(ctx context.Context, args map[string]interface{}) (string, error) {
	// Get the expression
	expression, ok := args["expression"].(string)
	if !ok {
		return "", errors.New("expression must be a string")
	}

	// TODO: Implement a safe expression evaluator
	return fmt.Sprintf("Calculated result for: %s", expression), nil
}

// CalculatorTool creates a tool for performing calculations.
func CalculatorTool() Tool {
	return NewToolBuilder("calculator", "Perform a calculation.").AddStringParameter("expression", "The expression to calculate", true).SetHandler(CalculatorHandler).Build()
}

// WeatherHandler is a handler for a weather tool.
func WeatherHandler(weather_func func(ctx context.Context, location string) (string, error)) ToolHandler {
	return func(ctx context.Context, args map[string]interface{}) (string, error) {
		// Get the location
		location, ok := args["location"].(string)
		if !ok {
			return "", errors.New("location must be a string")
		}

		// Get the weather
		return weather_func(ctx, location)
	}
}

// WeatherTool creates a tool for getting weather information.
func WeatherTool(weather_func func(ctx context.Context, location string) (string, error)) Tool {
	return NewToolBuilder("weather", "Get the current weather for a location.")
		.AddStringParameter("location", "The location to get weather for", true)
		.SetHandler(WeatherHandler(weather_func))
		.Build()
}

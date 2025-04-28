// Package model provides interfaces and types for interacting with various
// language and vision-language models across different providers.
package model

import (
	"context"
	"io"
)

// ContentType represents the type of content that can be provided to a model.
type ContentType string

const (
	// TextContent represents plain text content.
	TextContent ContentType = "text"
	// ImageContent represents image content.
	ImageContent ContentType = "image"
	// AudioContent represents audio content.
	AudioContent ContentType = "audio"
	// VideoContent represents video content.
	VideoContent ContentType = "video"
)

// Capability represents a capability that a model may have.
type Capability string

const (
	// TextGeneration indicates the model can generate text responses.
	TextGeneration Capability = "text_generation"
	// ImageUnderstanding indicates the model can process and understand images.
	ImageUnderstanding Capability = "image_understanding"
	// AudioUnderstanding indicates the model can process and understand audio.
	AudioUnderstanding Capability = "audio_understanding"
	// VideoUnderstanding indicates the model can process and understand video.
	VideoUnderstanding Capability = "video_understanding"
	// ToolUsage indicates the model can use tools/functions.
	ToolUsage Capability = "tool_usage"
	// Embedding indicates the model can generate embeddings.
	Embedding Capability = "embedding"
)

// Content represents a piece of content to be sent to a model.
type Content struct {
	// Type is the type of content.
	Type ContentType
	// Text is the text content if Type is TextContent.
	Text string
	// Data is the raw data for non-text content types.
	Data io.Reader
	// MIMEType is the MIME type of the content (for non-text content).
	MIMEType string
	// Name is an optional name for the content (e.g., filename).
	Name string
}

// ModelInfo contains metadata about a model.
type ModelInfo struct {
	// Name is the name of the model.
	Name string
	// Provider is the provider of the model (e.g., "openai", "google", "anthropic").
	Provider string
	// Capabilities is a list of capabilities the model has.
	Capabilities []Capability
	// MaxContextTokens is the maximum number of tokens the model can process.
	MaxContextTokens int
	// Description is a human-readable description of the model.
	Description string
}

// Message represents a message in a conversation with a model.
type Message struct {
	// Role is the role of the message sender (e.g., "system", "user", "assistant").
	Role string
	// Content is the list of content pieces in the message.
	Content []Content
}

// CompletionRequest represents a request for a model completion.
type CompletionRequest struct {
	// Messages is the conversation history.
	Messages []Message
	// Temperature controls randomness in the response (0.0 to 1.0).
	Temperature float32
	// MaxTokens is the maximum number of tokens to generate.
	MaxTokens int
	// Tools is an optional list of tools the model can use.
	Tools []Tool
	// StreamResponse indicates whether the response should be streamed.
	StreamResponse bool
}

// Tool represents a tool that a model can use.
type Tool struct {
	// Name is the name of the tool.
	Name string
	// Description is a description of what the tool does.
	Description string
	// Parameters is a map of parameter names to their JSON schema.
	Parameters map[string]interface{}
}

// ToolCall represents a call to a tool by the model.
type ToolCall struct {
	// Name is the name of the tool being called.
	Name string
	// Arguments is a map of argument names to their values.
	Arguments map[string]interface{}
}

// CompletionResponse represents a response from a model completion request.
type CompletionResponse struct {
	// Text is the generated text response.
	Text string
	// ToolCalls is a list of tool calls the model wants to make.
	ToolCalls []ToolCall
	// FinishReason indicates why the model stopped generating text.
	FinishReason string
	// UsageStats contains token usage statistics.
	UsageStats UsageStats
}

// UsageStats contains token usage statistics for a model request.
type UsageStats struct {
	// PromptTokens is the number of tokens in the prompt.
	PromptTokens int
	// CompletionTokens is the number of tokens in the completion.
	CompletionTokens int
	// TotalTokens is the total number of tokens used.
	TotalTokens int
}

// StreamedCompletionChunk represents a chunk of a streamed completion response.
type StreamedCompletionChunk struct {
	// Text is the text in this chunk.
	Text string
	// ToolCalls is a list of tool calls in this chunk.
	ToolCalls []ToolCall
	// FinishReason indicates why the model stopped generating text.
	FinishReason string
	// IsFinal indicates whether this is the final chunk.
	IsFinal bool
}

// StreamHandler is a function that handles streamed completion chunks.
type StreamHandler func(chunk StreamedCompletionChunk) error

// Model represents a language or vision-language model.
type Model interface {
	// GetInfo returns information about the model.
	GetInfo() ModelInfo

	// Complete generates a completion for the given request.
	Complete(ctx context.Context, request CompletionRequest) (CompletionResponse, error)

	// CompleteStream generates a streamed completion for the given request.
	CompleteStream(ctx context.Context, request CompletionRequest, handler StreamHandler) error

	// SupportsContentType checks if the model supports a specific content type.
	SupportsContentType(contentType ContentType) bool
}

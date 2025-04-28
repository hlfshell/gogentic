// Package openai provides an implementation of the provider interface for OpenAI.
package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/hlfshell/go-agents/pkg/embedding"
	"github.com/hlfshell/go-agents/pkg/model"
	"github.com/hlfshell/go-agents/pkg/provider"
)

// Constants for OpenAI API endpoints
const (
	default_base_url      = "https://api.openai.com/v1"
	models_endpoint       = "/models"
	completions_endpoint  = "/chat/completions"
	embeddings_endpoint   = "/embeddings"
	default_timeout_secs  = 30
	default_max_retries   = 3
	system_role           = "system"
	user_role             = "user"
	assistant_role        = "assistant"
	tool_role             = "tool"
)

// OpenAIProvider implements the provider.Provider interface for OpenAI.
type OpenAIProvider struct {
	// api_key is the API key for OpenAI.
	api_key string
	// base_url is the base URL for the OpenAI API.
	base_url string
	// organization_id is the organization ID for OpenAI (if applicable).
	organization_id string
	// http_client is the HTTP client used for API requests.
	http_client *http.Client
	// max_retries is the maximum number of retries for failed requests.
	max_retries int
	// additional_headers is a map of additional headers to include in requests.
	additional_headers map[string]string
}

// NewOpenAIProvider creates a new OpenAI provider with the given configuration.
func NewOpenAIProvider(config provider.Config) (provider.Provider, error) {
	// Validate the API key
	if config.APIKey == "" {
		return nil, errors.New("API key is required for OpenAI provider")
	}

	// Set default values if not provided
	base_url := config.BaseURL
	if base_url == "" {
		base_url = default_base_url
	}

	timeout_secs := config.TimeoutSeconds
	if timeout_secs <= 0 {
		timeout_secs = default_timeout_secs
	}

	max_retries := config.MaxRetries
	if max_retries <= 0 {
		max_retries = default_max_retries
	}

	// Create the HTTP client with the specified timeout
	http_client := &http.Client{
		Timeout: time.Duration(timeout_secs) * time.Second,
	}

	// Create and return the provider
	return &OpenAIProvider{
		api_key:           config.APIKey,
		base_url:          base_url,
		organization_id:   config.OrganizationID,
		http_client:       http_client,
		max_retries:       max_retries,
		additional_headers: config.AdditionalHeaders,
	}, nil
}

// Name returns the name of the provider.
func (p *OpenAIProvider) Name() string {
	return "OpenAI"
}

// Description returns a human-readable description of the provider.
func (p *OpenAIProvider) Description() string {
	return "OpenAI provides various language and vision-language models including GPT-4o, GPT-4, and GPT-3.5 Turbo."
}

// ListAvailableModels returns a list of available models from OpenAI.
func (p *OpenAIProvider) ListAvailableModels(ctx context.Context) ([]model.ModelInfo, error) {
	// Make a request to the models endpoint
	resp, err := p.makeRequest(ctx, "GET", models_endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}
	defer resp.Body.Close()

	// Parse the response
	var models_response struct {
		Data []struct {
			ID    string `json:"id"`
			Owner string `json:"owner"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&models_response); err != nil {
		return nil, fmt.Errorf("failed to decode models response: %w", err)
	}

	// Filter and map the models to ModelInfo
	model_infos := []model.ModelInfo{}
	for _, m := range models_response.Data {
		// Only include chat models
		if strings.Contains(m.ID, "gpt") {
			info := model.ModelInfo{
				Name:        m.ID,
				Provider:    "openai",
				Capabilities: []model.Capability{model.TextGeneration},
				Description: fmt.Sprintf("OpenAI %s model", m.ID),
			}

			// Add additional capabilities based on model name
			if strings.Contains(m.ID, "vision") || strings.Contains(m.ID, "gpt-4o") {
				info.Capabilities = append(info.Capabilities, model.ImageUnderstanding)
			}

			// Set context window size based on model
			switch {
			case strings.Contains(m.ID, "gpt-4o"):
				info.MaxContextTokens = 128000
			case strings.Contains(m.ID, "gpt-4-turbo"):
				info.MaxContextTokens = 128000
			case strings.Contains(m.ID, "gpt-4-32k"):
				info.MaxContextTokens = 32768
			case strings.Contains(m.ID, "gpt-4"):
				info.MaxContextTokens = 8192
			case strings.Contains(m.ID, "gpt-3.5-turbo-16k"):
				info.MaxContextTokens = 16384
			case strings.Contains(m.ID, "gpt-3.5-turbo"):
				info.MaxContextTokens = 4096
			default:
				info.MaxContextTokens = 4096
			}

			model_infos = append(model_infos, info)
		}
	}

	return model_infos, nil
}

// ListAvailableEmbeddingModels returns a list of available embedding models from OpenAI.
func (p *OpenAIProvider) ListAvailableEmbeddingModels(ctx context.Context) ([]embedding.ModelInfo, error) {
	// Define the available embedding models
	embedding_models := []embedding.ModelInfo{
		{
			Name:                 "text-embedding-3-large",
			Provider:             "openai",
			Dimensions:           3072,
			SupportedContentTypes: []embedding.ContentType{embedding.TextContent},
			Description:          "OpenAI's most capable embedding model for text",
		},
		{
			Name:                 "text-embedding-3-small",
			Provider:             "openai",
			Dimensions:           1536,
			SupportedContentTypes: []embedding.ContentType{embedding.TextContent},
			Description:          "OpenAI's efficient embedding model for text with good quality",
		},
		{
			Name:                 "text-embedding-ada-002",
			Provider:             "openai",
			Dimensions:           1536,
			SupportedContentTypes: []embedding.ContentType{embedding.TextContent},
			Description:          "OpenAI's legacy embedding model (deprecated)",
		},
	}

	return embedding_models, nil
}

// GetModel returns a model instance by name.
func (p *OpenAIProvider) GetModel(ctx context.Context, model_name string) (model.Model, error) {
	// Check if the model exists
	models, err := p.ListAvailableModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	var model_info model.ModelInfo
	found := false
	for _, info := range models {
		if info.Name == model_name {
			model_info = info
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("model %s not found", model_name)
	}

	// Create and return the model
	return &OpenAIModel{
		provider:   p,
		model_info: model_info,
	}, nil
}

// GetEmbeddingModel returns an embedding model instance by name.
func (p *OpenAIProvider) GetEmbeddingModel(ctx context.Context, model_name string) (embedding.EmbeddingModel, error) {
	// Check if the model exists
	models, err := p.ListAvailableEmbeddingModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list embedding models: %w", err)
	}

	var model_info embedding.ModelInfo
	found := false
	for _, info := range models {
		if info.Name == model_name {
			model_info = info
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("embedding model %s not found", model_name)
	}

	// Create and return the embedding model
	return &OpenAIEmbeddingModel{
		provider:   p,
		model_info: model_info,
	}, nil
}

// makeRequest makes an HTTP request to the OpenAI API.
func (p *OpenAIProvider) makeRequest(ctx context.Context, method, endpoint string, body io.Reader) (*http.Response, error) {
	// Create the request
	req, err := http.NewRequestWithContext(ctx, method, p.base_url+endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.api_key)
	
	if p.organization_id != "" {
		req.Header.Set("OpenAI-Organization", p.organization_id)
	}

	// Add any additional headers
	for key, value := range p.additional_headers {
		req.Header.Set(key, value)
	}

	// Make the request with retries
	var resp *http.Response
	var last_err error
	for i := 0; i <= p.max_retries; i++ {
		resp, err = p.http_client.Do(req)
		if err == nil && resp.StatusCode < 500 {
			return resp, nil
		}

		last_err = err
		if err == nil {
			last_err = fmt.Errorf("server error: %s", resp.Status)
			resp.Body.Close()
		}

		// Exponential backoff
		if i < p.max_retries {
			time.Sleep(time.Duration(1<<uint(i)) * time.Second)
		}
	}

	return nil, fmt.Errorf("request failed after %d retries: %w", p.max_retries, last_err)
}

// OpenAIModel implements the model.Model interface for OpenAI models.
type OpenAIModel struct {
	// provider is the OpenAI provider.
	provider *OpenAIProvider
	// model_info is the information about the model.
	model_info model.ModelInfo
}

// GetInfo returns information about the model.
func (m *OpenAIModel) GetInfo() model.ModelInfo {
	return m.model_info
}

// Complete generates a completion for the given request.
func (m *OpenAIModel) Complete(ctx context.Context, request model.CompletionRequest) (model.CompletionResponse, error) {
	// Convert the request to OpenAI format
	openai_request, err := m.convertCompletionRequest(request)
	if err != nil {
		return model.CompletionResponse{}, fmt.Errorf("failed to convert request: %w", err)
	}

	// Marshal the request to JSON
	request_body, err := json.Marshal(openai_request)
	if err != nil {
		return model.CompletionResponse{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make the request
	resp, err := m.provider.makeRequest(ctx, "POST", completions_endpoint, strings.NewReader(string(request_body)))
	if err != nil {
		return model.CompletionResponse{}, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return model.CompletionResponse{}, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse the response
	var openai_response openaiCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&openai_response); err != nil {
		return model.CompletionResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}

	// Convert the response to the generic format
	return m.convertCompletionResponse(openai_response)
}

// CompleteStream generates a streamed completion for the given request.
func (m *OpenAIModel) CompleteStream(ctx context.Context, request model.CompletionRequest, handler model.StreamHandler) error {
	// Ensure streaming is enabled
	request.StreamResponse = true

	// Convert the request to OpenAI format
	openai_request, err := m.convertCompletionRequest(request)
	if err != nil {
		return fmt.Errorf("failed to convert request: %w", err)
	}

	// Add streaming flag
	openai_request["stream"] = true

	// Marshal the request to JSON
	request_body, err := json.Marshal(openai_request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make the request
	resp, err := m.provider.makeRequest(ctx, "POST", completions_endpoint, strings.NewReader(string(request_body)))
	if err != nil {
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Process the stream
	reader := bufio.NewReader(resp.Body)
	for {
		// Check if the context is done
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Continue processing
		}

		// Read a line from the stream
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("failed to read stream: %w", err)
		}

		// Skip empty lines and "data: [DONE]"
		line = strings.TrimSpace(line)
		if line == "" || line == "data: [DONE]" {
			continue
		}

		// Parse the SSE data
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		// Parse the JSON
		var chunk_response openaiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk_response); err != nil {
			return fmt.Errorf("failed to parse chunk: %w", err)
		}

		// Convert the chunk to the generic format
		chunk, err := m.convertStreamChunk(chunk_response)
		if err != nil {
			return fmt.Errorf("failed to convert chunk: %w", err)
		}

		// Call the handler
		if err := handler(chunk); err != nil {
			return fmt.Errorf("handler error: %w", err)
		}
	}
}

// SupportsContentType checks if the model supports a specific content type.
func (m *OpenAIModel) SupportsContentType(contentType model.ContentType) bool {
	for _, capability := range m.model_info.Capabilities {
		switch contentType {
		case model.TextContent:
			return true
		case model.ImageContent:
			if capability == model.ImageUnderstanding {
				return true
			}
		case model.AudioContent:
			if capability == model.AudioUnderstanding {
				return true
			}
		case model.VideoContent:
			if capability == model.VideoUnderstanding {
				return true
			}
		}
	}
	return false
}

// convertCompletionRequest converts a generic completion request to OpenAI format.
func (m *OpenAIModel) convertCompletionRequest(request model.CompletionRequest) (map[string]interface{}, error) {
	openai_request := map[string]interface{}{
		"model": m.model_info.Name,
	}

	// Set temperature if provided
	if request.Temperature > 0 {
		openai_request["temperature"] = request.Temperature
	}

	// Set max tokens if provided
	if request.MaxTokens > 0 {
		openai_request["max_tokens"] = request.MaxTokens
	}

	// Convert messages
	openai_messages := []map[string]interface{}{}
	for _, msg := range request.Messages {
		openai_message := map[string]interface{}{
			"role": msg.Role,
		}

		// Handle content
		if len(msg.Content) == 1 && msg.Content[0].Type == model.TextContent {
			// Simple text content
			openai_message["content"] = msg.Content[0].Text
		} else {
			// Multimodal content
			content_array := []map[string]interface{}{}
			for _, content := range msg.Content {
				content_item := map[string]interface{}{}
				
				switch content.Type {
				case model.TextContent:
					content_item["type"] = "text"
					content_item["text"] = content.Text
				case model.ImageContent:
					// For images, we need to handle the data
					content_item["type"] = "image_url"
					
					// TODO: Implement image handling
					return nil, errors.New("image content not yet implemented")
				default:
					return nil, fmt.Errorf("unsupported content type: %s", content.Type)
				}
				
				content_array = append(content_array, content_item)
			}
			
			openai_message["content"] = content_array
		}

		openai_messages = append(openai_messages, openai_message)
	}
	
	openai_request["messages"] = openai_messages

	// Convert tools if provided
	if len(request.Tools) > 0 {
		openai_tools := []map[string]interface{}{}
		for _, tool := range request.Tools {
			openai_tool := map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        tool.Name,
					"description": tool.Description,
					"parameters":  tool.Parameters,
				},
			}
			openai_tools = append(openai_tools, openai_tool)
		}
		openai_request["tools"] = openai_tools
	}

	return openai_request, nil
}

// openaiCompletionResponse represents the response from the OpenAI completions API.
type openaiCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message struct {
			Role       string `json:"role"`
			Content    string `json:"content"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// openaiStreamChunk represents a chunk from the OpenAI streaming API.
type openaiStreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Delta        struct {
			Role       string `json:"role,omitempty"`
			Content    string `json:"content,omitempty"`
			ToolCalls  []struct {
				Index    int    `json:"index"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// convertCompletionResponse converts an OpenAI completion response to the generic format.
func (m *OpenAIModel) convertCompletionResponse(response openaiCompletionResponse) (model.CompletionResponse, error) {
	if len(response.Choices) == 0 {
		return model.CompletionResponse{}, errors.New("no choices in response")
	}

	choice := response.Choices[0]
	generic_response := model.CompletionResponse{
		Text:         choice.Message.Content,
		FinishReason: choice.FinishReason,
		UsageStats: model.UsageStats{
			PromptTokens:     response.Usage.PromptTokens,
			CompletionTokens: response.Usage.CompletionTokens,
			TotalTokens:      response.Usage.TotalTokens,
		},
	}

	// Convert tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		tool_calls := []model.ToolCall{}
		for _, tc := range choice.Message.ToolCalls {
			// Parse the arguments JSON
			var args map[string]interface{}
			if err := json.Unmarshal(tc.Function.Arguments, &args); err != nil {
				return model.CompletionResponse{}, fmt.Errorf("failed to parse tool call arguments: %w", err)
			}

			tool_call := model.ToolCall{
				Name:      tc.Function.Name,
				Arguments: args,
			}
			tool_calls = append(tool_calls, tool_call)
		}
		generic_response.ToolCalls = tool_calls
	}

	return generic_response, nil
}

// convertStreamChunk converts an OpenAI stream chunk to the generic format.
func (m *OpenAIModel) convertStreamChunk(chunk openaiStreamChunk) (model.StreamedCompletionChunk, error) {
	if len(chunk.Choices) == 0 {
		return model.StreamedCompletionChunk{}, errors.New("no choices in chunk")
	}

	choice := chunk.Choices[0]
	generic_chunk := model.StreamedCompletionChunk{
		Text:         choice.Delta.Content,
		FinishReason: choice.FinishReason,
		IsFinal:      choice.FinishReason != "",
	}

	// Convert tool calls if present
	if len(choice.Delta.ToolCalls) > 0 {
		tool_calls := []model.ToolCall{}
		for _, tc := range choice.Delta.ToolCalls {
			// Parse the arguments JSON if present
			var args map[string]interface{}
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					return model.StreamedCompletionChunk{}, fmt.Errorf("failed to parse tool call arguments: %w", err)
				}
			}

			tool_call := model.ToolCall{
				Name:      tc.Function.Name,
				Arguments: args,
			}
			tool_calls = append(tool_calls, tool_call)
		}
		generic_chunk.ToolCalls = tool_calls
	}

	return generic_chunk, nil
}

// OpenAIEmbeddingModel implements the embedding.EmbeddingModel interface for OpenAI.
type OpenAIEmbeddingModel struct {
	// provider is the OpenAI provider.
	provider *OpenAIProvider
	// model_info is the information about the embedding model.
	model_info embedding.ModelInfo
}

// GetInfo returns information about the embedding model.
func (m *OpenAIEmbeddingModel) GetInfo() embedding.ModelInfo {
	return m.model_info
}

// Embed generates embeddings for the given request.
func (m *OpenAIEmbeddingModel) Embed(ctx context.Context, request embedding.EmbeddingRequest) (embedding.EmbeddingResponse, error) {
	// Check if all content types are supported
	for _, content := range request.Contents {
		if !m.SupportsContentType(content.Type) {
			return embedding.EmbeddingResponse{}, fmt.Errorf("content type %s not supported by model %s", content.Type, m.model_info.Name)
		}
	}

	// Extract text from the request
	texts := []string{}
	for _, content := range request.Contents {
		if content.Type == embedding.TextContent {
			texts = append(texts, content.Text)
		} else {
			return embedding.EmbeddingResponse{}, fmt.Errorf("non-text content not supported yet")
		}
	}

	// Create the request body
	request_body := map[string]interface{}{
		"model": m.model_info.Name,
		"input": texts,
	}

	// Marshal the request to JSON
	request_json, err := json.Marshal(request_body)
	if err != nil {
		return embedding.EmbeddingResponse{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make the request
	resp, err := m.provider.makeRequest(ctx, "POST", embeddings_endpoint, strings.NewReader(string(request_json)))
	if err != nil {
		return embedding.EmbeddingResponse{}, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return embedding.EmbeddingResponse{}, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse the response
	var openai_response struct {
		Object string `json:"object"`
		Data   []struct {
			Object    string    `json:"object"`
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&openai_response); err != nil {
		return embedding.EmbeddingResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}

	// Convert the response to the generic format
	embeddings := []embedding.Embedding{}
	for _, data := range openai_response.Data {
		embeddings = append(embeddings, embedding.Embedding{
			Vector: data.Embedding,
			Index:  data.Index,
		})
	}

	return embedding.EmbeddingResponse{
		Embeddings: embeddings,
		UsageStats: embedding.UsageStats{
			TokensProcessed: openai_response.Usage.TotalTokens,
		},
	}, nil
}

// SupportsContentType checks if the model supports a specific content type.
func (m *OpenAIEmbeddingModel) SupportsContentType(contentType embedding.ContentType) bool {
	for _, supported := range m.model_info.SupportedContentTypes {
		if supported == contentType {
			return true
		}
	}
	return false
}

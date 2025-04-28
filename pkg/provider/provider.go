// Package provider defines interfaces and implementations for various
// LLM/VLM providers such as OpenAI, Google, Anthropic, etc.
package provider

import (
	"context"

	"github.com/hlfshell/go-agents/pkg/embedding"
	"github.com/hlfshell/go-agents/pkg/model"
)

// Provider represents a service provider for AI models.
type Provider interface {
	// Name returns the name of the provider.
	Name() string

	// Description returns a human-readable description of the provider.
	Description() string

	// ListAvailableModels returns a list of available models from this provider.
	ListAvailableModels(ctx context.Context) ([]model.ModelInfo, error)

	// ListAvailableEmbeddingModels returns a list of available embedding models.
	ListAvailableEmbeddingModels(ctx context.Context) ([]embedding.ModelInfo, error)

	// GetModel returns a model instance by name.
	GetModel(ctx context.Context, modelName string) (model.Model, error)

	// GetEmbeddingModel returns an embedding model instance by name.
	GetEmbeddingModel(ctx context.Context, modelName string) (embedding.EmbeddingModel, error)
}

// Config represents the configuration for a provider.
type Config struct {
	// APIKey is the API key for the provider.
	APIKey string
	// BaseURL is the base URL for the provider's API (optional, for custom endpoints).
	BaseURL string
	// OrganizationID is the organization ID (if applicable).
	OrganizationID string
	// Timeout is the timeout for API requests in seconds.
	TimeoutSeconds int
	// MaxRetries is the maximum number of retries for failed requests.
	MaxRetries int
	// AdditionalHeaders is a map of additional headers to include in requests.
	AdditionalHeaders map[string]string
}

// ProviderFactory creates a new provider instance with the given configuration.
type ProviderFactory func(config Config) (Provider, error)

// Package embedding provides interfaces and types for generating embeddings
// from various providers and models.
package embedding

import (
	"context"
	"io"
)

// ContentType represents the type of content that can be embedded.
type ContentType string

const (
	// TextContent represents plain text content.
	TextContent ContentType = "text"
	// ImageContent represents image content.
	ImageContent ContentType = "image"
)

// Content represents a piece of content to be embedded.
type Content struct {
	// Type is the type of content.
	Type ContentType
	// Text is the text content if Type is TextContent.
	Text string
	// Data is the raw data for non-text content types.
	Data io.Reader
	// MIMEType is the MIME type of the content (for non-text content).
	MIMEType string
}

// ModelInfo contains metadata about an embedding model.
type ModelInfo struct {
	// Name is the name of the model.
	Name string
	// Provider is the provider of the model (e.g., "openai", "google", "cohere").
	Provider string
	// Dimensions is the number of dimensions in the embedding vectors.
	Dimensions int
	// SupportedContentTypes is a list of content types the model can embed.
	SupportedContentTypes []ContentType
	// Description is a human-readable description of the model.
	Description string
}

// EmbeddingRequest represents a request to generate embeddings.
type EmbeddingRequest struct {
	// Contents is a list of content pieces to embed.
	Contents []Content
	// BatchSize is the number of items to process in each batch (optional).
	BatchSize int
}

// Embedding represents a vector embedding for a piece of content.
type Embedding struct {
	// Vector is the embedding vector.
	Vector []float32
	// Index is the index of the content in the original request.
	Index int
}

// EmbeddingResponse represents a response from an embedding request.
type EmbeddingResponse struct {
	// Embeddings is a list of generated embeddings.
	Embeddings []Embedding
	// UsageStats contains token usage statistics.
	UsageStats UsageStats
}

// UsageStats contains token usage statistics for an embedding request.
type UsageStats struct {
	// TokensProcessed is the number of tokens processed.
	TokensProcessed int
	// TotalCost is the estimated cost of the request (if available).
	TotalCost float64
}

// EmbeddingModel represents a model that can generate embeddings.
type EmbeddingModel interface {
	// GetInfo returns information about the embedding model.
	GetInfo() ModelInfo

	// Embed generates embeddings for the given request.
	Embed(ctx context.Context, request EmbeddingRequest) (EmbeddingResponse, error)

	// SupportsContentType checks if the model supports a specific content type.
	SupportsContentType(contentType ContentType) bool
}

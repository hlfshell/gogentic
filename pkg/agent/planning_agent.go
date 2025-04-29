// Package agent provides interfaces and implementations for building AI agents
// that can use language models to accomplish tasks.
package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	arkaineparser "github.com/hlfshell/go-arkaine-parser"
	"github.com/hlfshell/go-agents/pkg/model"
)

// PlanApprovalFunc is a function that approves or rejects a plan.
// It returns a boolean indicating whether the plan is approved,
// and an optional string with recommended changes.
type PlanApprovalFunc func(ctx context.Context, plan string, history []Message) (bool, string, error)

// EvaluationResult contains the result of evaluating a plan step execution.
type EvaluationResult struct {
	// Approved indicates whether the step execution is approved.
	Approved bool
	// Suggestion is a suggestion for improvement if Approved is false.
	Suggestion string
	// RecalculatePlan indicates whether the plan should be recalculated.
	RecalculatePlan bool
}

// EvaluationFunc is a function that evaluates the execution of a plan step.
type EvaluationFunc func(ctx context.Context, step string, stepResult string, history []Message) (EvaluationResult, error)

// PlanningAgentConfig extends AgentConfig with planning-specific settings.
type PlanningAgentConfig struct {
	// Base agent configuration
	AgentConfig
	
	// PlanApprovalFunc is an optional function to approve plans
	PlanApprovalFunc PlanApprovalFunc
	
	// EvaluationFunc is a function to evaluate each step's execution
	// If not provided, a default evaluator will be used
	EvaluationFunc EvaluationFunc
	
	// ExecutionAgent is an optional agent to use for executing plan steps
	// If not provided, a default agent will be created
	ExecutionAgent Agent
	
	// MaxPlanAttempts is the maximum number of times to attempt creating a plan
	MaxPlanAttempts int
	
	// MaxStepAttempts is the maximum number of times to attempt executing a step
	MaxStepAttempts int
}

// PlanningAgent is an implementation of the Agent interface that creates and
// executes plans to accomplish tasks.
type PlanningAgent struct {
	*BaseAgent
	config PlanningAgentConfig
}

// NewPlanningAgent creates a new planning agent with the given configuration.
func NewPlanningAgent(id, name, description string, config PlanningAgentConfig) *PlanningAgent {
	// Set default values if not provided
	if config.MaxPlanAttempts <= 0 {
		config.MaxPlanAttempts = 3
	}
	
	if config.MaxStepAttempts <= 0 {
		config.MaxStepAttempts = 2
	}
	
	if config.Timeout <= 0 {
		config.Timeout = 120 * time.Second
	}
	
	// Set up default parser labels if not provided
	if len(config.ParserLabels) == 0 {
		config.ParserLabels = []arkaineparser.Label{
			{Name: "Plan", IsBlockStart: true, Required: true},
			{Name: "Step", IsBlockStart: true, Required: true},
			{Name: "Reasoning", IsBlockStart: true},
			{Name: "Action", Required: true},
			{Name: "ActionInput", IsJSON: true},
			{Name: "Result"},
			{Name: "Evaluation"},
			{Name: "FinalAnswer", IsBlockStart: true},
		}
	}
	
	// Create the base agent
	base_agent := NewBaseAgent(id, name, description, config.AgentConfig)
	
	// Create and return the planning agent
	return &PlanningAgent{
		BaseAgent: base_agent,
		config:    config,
	}
}

// defaultEvaluationFunc is the default implementation of EvaluationFunc.
func defaultEvaluationFunc(ctx context.Context, step string, stepResult string, history []Message) (EvaluationResult, error) {
	// In a real implementation, this would use an LLM to evaluate the step execution
	// For now, we'll just assume all steps are executed correctly
	return EvaluationResult{
		Approved:       true,
		Suggestion:     "",
		RecalculatePlan: false,
	}, nil
}

// defaultPlanApprovalFunc is the default implementation of PlanApprovalFunc.
func defaultPlanApprovalFunc(ctx context.Context, plan string, history []Message) (bool, string, error) {
	// By default, all plans are approved
	return true, "", nil
}

// createEvaluationAgent creates an evaluation agent using the provided model.
func createEvaluationAgent(llm model.Model) EvaluationFunc {
	return func(ctx context.Context, step string, stepResult string, history []Message) (EvaluationResult, error) {
		// Create the evaluation prompt
		evaluation_prompt := "You are an evaluation agent. Your task is to evaluate whether the execution of a step in a plan was successful."
		evaluation_prompt += "\n\nThe step to execute was:\n" + step
		evaluation_prompt += "\n\nThe result of the execution was:\n" + stepResult
		evaluation_prompt += "\n\nEvaluate whether the step was executed correctly. If it was, respond with:\n"
		evaluation_prompt += "Approved: true\nSuggestion: \nRecalculatePlan: false"
		evaluation_prompt += "\n\nIf it was not executed correctly, respond with:\n"
		evaluation_prompt += "Approved: false\nSuggestion: [Your suggestion for improvement]\nRecalculatePlan: [true/false]"
		evaluation_prompt += "\n\nSet RecalculatePlan to true only if you believe the current plan is fundamentally flawed and needs to be recalculated."
		
		// Create a simple conversation with the evaluation prompt as the system message
		// and the step and result as the user message
		conversation := Conversation{
			ID: uuid.New().String(),
			Messages: []Message{
				{
					Role:      "system",
					Content:   evaluation_prompt,
					Timestamp: time.Now(),
				},
			},
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		
		// Create the agent parameters
		params := AgentParameters{
			Input:        fmt.Sprintf("Step: %s\n\nResult: %s", step, stepResult),
			Conversation: &conversation,
			Options: AgentOptions{
				Temperature: func() *float32 { temp := float32(0.2); return &temp }(),
				MaxTokens:   func() *int { tokens := 500; return &tokens }(),
			},
		}
		
		// Create a simple agent to evaluate the step
		evaluation_agent := NewBaseAgent(
			uuid.New().String(),
			"Evaluation Agent",
			"Agent for evaluating plan steps",
			AgentConfig{
				Model:       llm,
				SystemPrompt: evaluation_prompt,
				MaxTokens:   500,
				Temperature: 0.2,
			},
		)
		
		// Execute the evaluation agent
		result_obj, err := evaluation_agent.Execute(ctx, params)
		if err != nil {
			return EvaluationResult{}, fmt.Errorf("failed to evaluate step: %w", err)
		}
		
		// Parse the response
		lines := strings.Split(result_obj.Output, "\n")
		result := EvaluationResult{
			Approved:       false,
			Suggestion:     "",
			RecalculatePlan: false,
		}
		
		for _, line := range lines {
			if strings.HasPrefix(line, "Approved:") {
				value := strings.TrimSpace(strings.TrimPrefix(line, "Approved:"))
				result.Approved = (value == "true")
			} else if strings.HasPrefix(line, "Suggestion:") {
				result.Suggestion = strings.TrimSpace(strings.TrimPrefix(line, "Suggestion:"))
			} else if strings.HasPrefix(line, "RecalculatePlan:") {
				value := strings.TrimSpace(strings.TrimPrefix(line, "RecalculatePlan:"))
				result.RecalculatePlan = (value == "true")
			}
		}
		
		return result, nil
	}
}

// Execute processes the given parameters and returns a result.
// This implementation creates a plan, gets approval, and executes each step.
func (a *PlanningAgent) Execute(ctx context.Context, params AgentParameters) (AgentResult, error) {
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
	
	// Initialize execution stats
	execution_stats := ExecutionStats{
		StartTime:  start_time,
		ToolCalls:  0,
		Iterations: 0,
	}
	
	// Get or create the evaluation function
	evaluation_func := a.config.EvaluationFunc
	if evaluation_func == nil {
		evaluation_func = createEvaluationAgent(a.config.Model)
	}
	
	// Get or create the plan approval function
	plan_approval_func := a.config.PlanApprovalFunc
	if plan_approval_func == nil {
		plan_approval_func = defaultPlanApprovalFunc
	}
	
	// Create the execution agent if not provided
	execution_agent := a.config.ExecutionAgent
	if execution_agent == nil {
		// If tools are provided, create a tool agent
		if len(a.config.Tools) > 0 {
			execution_agent = NewToolAgent(
				uuid.New().String(),
				"Execution Agent",
				"Agent for executing plan steps",
				AgentConfig{
					Model:        a.config.Model,
					SystemPrompt: "You are an agent that executes specific steps in a plan. Focus on completing the current step only.",
					Tools:        a.config.Tools,
					MaxTokens:    a.config.MaxTokens,
					Temperature:  a.config.Temperature,
					MaxIterations: a.config.MaxIterations,
					Timeout:      a.config.Timeout,
				},
			)
		} else {
			// Otherwise, create a basic agent
			execution_agent = NewBaseAgent(
				uuid.New().String(),
				"Execution Agent",
				"Agent for executing plan steps",
				AgentConfig{
					Model:        a.config.Model,
					SystemPrompt: "You are an agent that executes specific steps in a plan. Focus on completing the current step only.",
					MaxTokens:    a.config.MaxTokens,
					Temperature:  a.config.Temperature,
					Timeout:      a.config.Timeout,
				},
			)
		}
	}
	
	// Attempt to create a plan
	plan := ""
	plan_approved := false
	plan_attempts := 0
	var plan_message Message
	
	for plan_attempts < a.config.MaxPlanAttempts && !plan_approved {
		// Check if the context is done
		select {
		case <-timeout_ctx.Done():
			return AgentResult{}, fmt.Errorf("agent execution timed out during planning: %w", timeout_ctx.Err())
		default:
			// Continue processing
		}
		
		// Create the planning prompt
		planning_prompt := "You are a planning agent. Your task is to create a detailed, step-by-step plan to accomplish the user's goal. "
		planning_prompt += "Break down the task into clear, logical steps. Each step should be specific and actionable. "
		planning_prompt += "Format your response using the following structure:\n\n"
		planning_prompt += "<Plan>\nProvide a high-level overview of your approach\n</Plan>\n\n"
		planning_prompt += "<Step>\nStep 1: [Brief description of the first step]\n</Step>\n\n"
		planning_prompt += "<Step>\nStep 2: [Brief description of the second step]\n</Step>\n\n"
		planning_prompt += "And so on for each step in your plan."
		
		// If this is a retry, add the feedback
		if plan_attempts > 0 {
			planning_prompt += "\n\nYour previous plan was not approved. Please revise it based on the following feedback: " + plan_message.Content
		}
		
		// Convert the conversation to a model request
		model_messages := []model.Message{}
		
		// Add the system prompt
		model_messages = append(model_messages, model.Message{
			Role: "system",
			Content: []model.Content{
				{
					Type: model.TextContent,
					Text: planning_prompt,
				},
			},
		})
		
		// Add the user message
		model_messages = append(model_messages, model.Message{
			Role: "user",
			Content: []model.Content{
				{
					Type: model.TextContent,
					Text: params.Input,
				},
			},
		})
		
		// Create the model request
		request := model.CompletionRequest{
			Messages:    model_messages,
			Temperature: temperature,
			MaxTokens:   max_tokens,
		}
		
		// Get a completion from the model
		response, err := a.config.Model.Complete(timeout_ctx, request)
		if err != nil {
			return AgentResult{}, fmt.Errorf("failed to generate plan: %w", err)
		}
		
		// Update usage stats
		usage_stats.PromptTokens += response.UsageStats.PromptTokens
		usage_stats.CompletionTokens += response.UsageStats.CompletionTokens
		usage_stats.TotalTokens += response.UsageStats.TotalTokens
		
		// Parse the plan
		parsed_output, parse_errors := a.parser.Parse(response.Text)
		if len(parse_errors) > 0 {
			return AgentResult{}, fmt.Errorf("failed to parse plan: %v", parse_errors)
		}
		
		// Extract the plan
		plan_value, ok := parsed_output["Plan"]
		if !ok {
			return AgentResult{}, errors.New("plan not found in parsed output")
		}
		
		plan = plan_value.(string)
		
		// Extract the steps
		steps_value, ok := parsed_output["Step"]
		if !ok {
			return AgentResult{}, errors.New("steps not found in parsed output")
		}
		
		steps := steps_value.([]interface{})
		if len(steps) == 0 {
			return AgentResult{}, errors.New("no steps found in plan")
		}
		
		// Create a message with the plan
		plan_message = Message{
			Role:      "assistant",
			Content:   response.Text,
			Timestamp: time.Now(),
		}
		
		// Add the plan message to the conversation
		conversation.Messages = append(conversation.Messages, plan_message)
		conversation.UpdatedAt = time.Now()
		
		// Get approval for the plan
		plan_approved, feedback, err := plan_approval_func(timeout_ctx, response.Text, conversation.Messages)
		if err != nil {
			return AgentResult{}, fmt.Errorf("failed to get plan approval: %w", err)
		}
		
		// If the plan is not approved, add the feedback to the conversation
		if !plan_approved {
			feedback_message := Message{
				Role:      "user",
				Content:   feedback,
				Timestamp: time.Now(),
			}
			
			conversation.Messages = append(conversation.Messages, feedback_message)
			conversation.UpdatedAt = time.Now()
		}
		
		plan_attempts++
		execution_stats.Iterations++
	}
	
	// If the plan is not approved after max attempts, return an error
	if !plan_approved {
		return AgentResult{}, errors.New("failed to create an approved plan after maximum attempts")
	}
	
	// Execute each step of the plan
	parsed_output, _ := a.parser.Parse(plan_message.Content)
	steps_value, _ := parsed_output["Step"]
	steps := steps_value.([]interface{})
	
	// Initialize the final result
	final_result := ""
	
	// Execute each step
	for i, step_interface := range steps {
		// Check if the context is done
		select {
		case <-timeout_ctx.Done():
			return AgentResult{}, fmt.Errorf("agent execution timed out during step execution: %w", timeout_ctx.Err())
		default:
			// Continue processing
		}
		
		step := step_interface.(string)
		step_num := i + 1
		
		// Execute the step
		step_result := ""
		step_attempts := 0
		step_success := false
		
		for step_attempts < a.config.MaxStepAttempts && !step_success {
			// Create the step execution parameters
			step_params := AgentParameters{
				Input: fmt.Sprintf("Execute the following step in the plan:\n\nStep %d: %s", step_num, step),
				Conversation: &Conversation{
					ID:        uuid.New().String(),
					Messages:  []Message{},
					Metadata:  map[string]interface{}{"plan": plan, "step": step, "step_num": step_num},
					CreatedAt: time.Now(),
					UpdatedAt: time.Now(),
				},
				Options: params.Options,
			}
			
			// Execute the step
			step_result_obj, err := execution_agent.Execute(timeout_ctx, step_params)
			if err != nil {
				return AgentResult{}, fmt.Errorf("failed to execute step %d: %w", step_num, err)
			}
			
			// Update usage stats
			usage_stats.PromptTokens += step_result_obj.UsageStats.PromptTokens
			usage_stats.CompletionTokens += step_result_obj.UsageStats.CompletionTokens
			usage_stats.TotalTokens += step_result_obj.UsageStats.TotalTokens
			
			// Update execution stats
			execution_stats.ToolCalls += step_result_obj.ExecutionStats.ToolCalls
			execution_stats.Iterations++
			
			// Get the step result
			step_result = step_result_obj.Output
			
			// Create a message with the step result
			step_result_message := Message{
				Role:      "assistant",
				Content:   step_result,
				Timestamp: time.Now(),
				ToolCalls: step_result_obj.Message.ToolCalls,
				ToolResults: step_result_obj.Message.ToolResults,
			}
			
			// Add the step result message to the conversation
			conversation.Messages = append(conversation.Messages, step_result_message)
			conversation.UpdatedAt = time.Now()
			
			// Evaluate the step execution
			evaluation_result, err := evaluation_func(timeout_ctx, step, step_result, conversation.Messages)
			if err != nil {
				return AgentResult{}, fmt.Errorf("failed to evaluate step %d: %w", step_num, err)
			}
			
			// Create a message with the evaluation result
			evaluation_message := Message{
				Role:      "system",
				Content:   fmt.Sprintf("Evaluation: %v\nSuggestion: %s\nRecalculate Plan: %v", 
					evaluation_result.Approved, 
					evaluation_result.Suggestion, 
					evaluation_result.RecalculatePlan),
				Timestamp: time.Now(),
			}
			
			// Add the evaluation message to the conversation
			conversation.Messages = append(conversation.Messages, evaluation_message)
			conversation.UpdatedAt = time.Now()
			
			// Check if the step was executed correctly
			if evaluation_result.Approved {
				step_success = true
			} else {
				// If the plan should be recalculated, return to planning phase
				if evaluation_result.RecalculatePlan {
					// Add a message indicating the need to recalculate the plan
					recalculate_message := Message{
						Role:      "user",
						Content:   fmt.Sprintf("The current plan is not working. Please create a new plan. Consider this feedback: %s", evaluation_result.Suggestion),
						Timestamp: time.Now(),
					}
					
					conversation.Messages = append(conversation.Messages, recalculate_message)
					conversation.UpdatedAt = time.Now()
					
					// Recursively call Execute to create a new plan
					return a.Execute(timeout_ctx, AgentParameters{
						Input:        params.Input,
						Conversation: conversation,
						Options:      params.Options,
					})
				}
				
				// Otherwise, retry the step with the suggestion
				retry_message := Message{
					Role:      "user",
					Content:   fmt.Sprintf("Please try again with this step. Consider this feedback: %s", evaluation_result.Suggestion),
					Timestamp: time.Now(),
				}
				
				conversation.Messages = append(conversation.Messages, retry_message)
				conversation.UpdatedAt = time.Now()
			}
			
			step_attempts++
		}
		
		// If the step was not executed successfully after max attempts, return an error
		if !step_success {
			return AgentResult{}, fmt.Errorf("failed to execute step %d after maximum attempts", step_num)
		}
		
		// Append the step result to the final result
		final_result += fmt.Sprintf("Step %d: %s\n\nResult: %s\n\n", step_num, step, step_result)
	}
	
	// Generate a final answer
	final_answer_prompt := "You have completed all steps of your plan. Based on the results of each step, provide a final answer to the original question or task."
	
	// Convert the conversation to a model request
	model_messages := []model.Message{}
	
	// Add the system prompt
	model_messages = append(model_messages, model.Message{
		Role: "system",
		Content: []model.Content{
			{
				Type: model.TextContent,
				Text: final_answer_prompt,
			},
		},
	})
	
	// Add the original user message
	model_messages = append(model_messages, model.Message{
		Role: "user",
		Content: []model.Content{
			{
				Type: model.TextContent,
				Text: params.Input,
			},
		},
	})
	
	// Add the final result
	model_messages = append(model_messages, model.Message{
		Role: "assistant",
		Content: []model.Content{
			{
				Type: model.TextContent,
				Text: final_result,
			},
		},
	})
	
	// Create the model request
	request := model.CompletionRequest{
		Messages:    model_messages,
		Temperature: temperature,
		MaxTokens:   max_tokens,
	}
	
	// Get a completion from the model
	response, err := a.config.Model.Complete(timeout_ctx, request)
	if err != nil {
		return AgentResult{}, fmt.Errorf("failed to generate final answer: %w", err)
	}
	
	// Update usage stats
	usage_stats.PromptTokens += response.UsageStats.PromptTokens
	usage_stats.CompletionTokens += response.UsageStats.CompletionTokens
	usage_stats.TotalTokens += response.UsageStats.TotalTokens
	
	// Create a message with the final answer
	final_answer_message := Message{
		Role:      "assistant",
		Content:   response.Text,
		Timestamp: time.Now(),
	}
	
	// Add the final answer message to the conversation
	conversation.Messages = append(conversation.Messages, final_answer_message)
	conversation.UpdatedAt = time.Now()
	
	// Parse the final answer
	parsed_final_output, parse_errors := a.parser.Parse(response.Text)
	
	// Record execution end time
	end_time := time.Now()
	execution_stats.EndTime = end_time
	
	// Return the agent result
	return AgentResult{
		Output:           response.Text,
		AdditionalOutputs: map[string]interface{}{
			"plan": plan,
			"steps": steps,
			"step_results": final_result,
		},
		Conversation:     conversation,
		UsageStats:       usage_stats,
		ExecutionStats:   execution_stats,
		Message:          final_answer_message,
		ParsedOutput:     parsed_final_output,
		ParseErrors:      parse_errors,
	}, nil
}

// ProcessStream processes a user message and streams the response.
// This implementation adds planning and step execution with streaming support.
func (a *PlanningAgent) ProcessStream(ctx context.Context, message string, conversation Conversation, handler StreamHandler) error {
	// Create agent parameters
	params := AgentParameters{
		Input:        message,
		Conversation: &conversation,
	}
	
	// Execute the agent and get the result
	result, err := a.Execute(ctx, params)
	if err != nil {
		return err
	}
	
	// Call the handler with the result
	return handler(result.Message)
}

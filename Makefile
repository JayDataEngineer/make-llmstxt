# make-llmstxt Makefile
# Simple commands to manage Docker infrastructure
#
# Usage:
#   make up              - Start all services
#   make down            - Stop all services
#   make status          - Show service status

.PHONY: up down status logs help \
        up-all up-llm up-observability \
        down-all down-llm down-observability \
        setup-secrets clean

# Default target
help:
	@echo "make-llmstxt - Docker Infrastructure Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make up              Start all services (LLM + observability)"
	@echo "  make down            Stop all services"
	@echo ""
	@echo "Commands:"
	@echo "  make status          Show service status"
	@echo "  make logs            Follow all logs"
	@echo "  make clean           Delete all data (WARNING: destructive)"
	@echo ""
	@echo "Partial startup:"
	@echo "  make up-llm          Start only LLM server (port 8001)"
	@echo "  make up-observability Start only observability (port 3000)"
	@echo ""
	@echo "Testing:"
	@echo "  make test-llm        Test LLM server connection"
	@echo "  make test-obs        Test observability stack"

# ============================================================================
# Main commands
# ============================================================================

up: setup-secrets
	@echo "🚀 Starting all services..."
	docker compose --profile all up -d
	@echo ""
	@echo "✅ Services starting!"
	@echo "   LLM Server:     http://localhost:8001"
	@echo "   Observability:  http://localhost:3000"
	@echo ""
	@echo "   Status: make status"
	@echo "   Logs:   make logs"

down:
	@echo "🛑 Stopping all services..."
	docker compose --profile all --profile llm --profile observability down
	@echo "✅ All services stopped"

# ============================================================================
# Partial startup
# ============================================================================

up-llm:
	@echo "🦙 Starting LLM server..."
	docker compose --profile llm up -d
	@echo ""
	@echo "✅ LLM server starting on port 8001"
	@echo "   Models are lazy-loaded on first request"
	@echo ""
	@echo "   Test: make test-llm"

up-observability: setup-secrets
	@echo "📊 Starting observability stack..."
	docker compose --profile observability up -d
	@echo ""
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "✅ Observability stack starting!"
	@echo "   UI: http://localhost:3000"
	@echo ""
	@echo "   Test: make test-obs"

down-llm:
	@echo "🛑 Stopping LLM server..."
	docker compose --profile llm down
	@echo "✅ LLM server stopped"

down-observability:
	@echo "🛑 Stopping observability stack..."
	docker compose --profile observability down
	@echo "✅ Observability stack stopped"

reset-observability:
	@echo "🔄 Resetting observability stack (deletes all data)..."
	docker compose --profile observability down -v 2>/dev/null || true
	@echo "✅ Observability stack reset"
	@echo "Run 'make up-observability' to start fresh"

# ============================================================================
# Setup
# ============================================================================

setup-secrets:
	@echo "🔐 Checking secrets..."
	@if [ -f .env ] && grep -q "LANGFUSE_NEXTAUTH_SECRET=." .env; then \
		echo "   Secrets already configured"; \
	else \
		echo "   Generating new secrets..."; \
		./scripts/generate-secrets.sh; \
	fi

# ============================================================================
# Utility commands
# ============================================================================

status:
	@echo "📊 Service Status:"
	@echo ""
	@docker compose --profile all --profile llm --profile observability ps 2>/dev/null || echo "   No services running"

logs:
	docker compose --profile all --profile llm --profile observability logs -f

logs-llm:
	docker compose --profile llm logs -f llama-server

logs-obs:
	docker compose --profile observability logs -f langfuse-web langfuse-worker

clean:
	@echo "⚠️  WARNING: This will DELETE all data (database, traces, etc.)!"
	@echo "Press Ctrl+C to cancel, or wait 3 seconds..."
	@sleep 3
	docker compose --profile all --profile llm --profile observability down -v 2>/dev/null || true
	@echo "✅ All services stopped and volumes removed"

# ============================================================================
# Testing
# ============================================================================

test-llm:
	@echo "Testing LLM server..."
	@curl -s localhost:8001/v1/models | jq '.data[].id' 2>/dev/null || echo "   LLM server not responding (may still be starting)"

test-obs:
	@echo "Testing observability stack..."
	@curl -s localhost:3000/api/health > /dev/null 2>&1 && echo "   ✅ Observability OK" || echo "   ❌ Observability not responding"

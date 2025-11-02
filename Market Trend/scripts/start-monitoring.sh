#!/bin/bash

# Market Trend Monitoring Infrastructure Startup Script

set -e

echo "🚀 Starting Market Trend Monitoring Infrastructure..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p backend/logs

# Pull latest images
echo "📦 Pulling latest monitoring images..."
docker-compose -f docker-compose.monitoring.yml pull

# Start monitoring services
echo "🔧 Starting monitoring services..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=(
    "prometheus:9090"
    "grafana:3000"
    "alertmanager:9093"
    "loki:3100"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s "http://localhost:$port" > /dev/null; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
    fi
done

echo ""
echo "🎉 Monitoring infrastructure is ready!"
echo ""
echo "📊 Access URLs:"
echo "   Grafana:      http://localhost:3001 (admin/admin123)"
echo "   Prometheus:   http://localhost:9090"
echo "   Alertmanager: http://localhost:9093"
echo "   Jaeger:       http://localhost:16686"
echo ""
echo "📝 To view logs:"
echo "   docker-compose -f docker-compose.monitoring.yml logs -f [service-name]"
echo ""
echo "🛑 To stop monitoring:"
echo "   docker-compose -f docker-compose.monitoring.yml down"
echo ""
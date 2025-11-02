#!/bin/bash

echo "Testing alerting by generating API errors..."

# Generate some 404 errors to trigger high error rate alert
for i in {1..20}; do
    echo "Generating error $i/20"
    curl -s http://localhost:8000/nonexistent-endpoint > /dev/null
    sleep 1
done

echo "Generated 20 404 errors. Check Prometheus alerts in a few minutes."
echo "You can view alerts at: http://localhost:9090/alerts"
echo "You can view Alertmanager at: http://localhost:9093"
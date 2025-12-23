#!/usr/bin/env bash

# Deploy COTA Data Collector to k3s Kubernetes Cluster

set -e

echo "🚀 Deploying COTA Data Collector to k3s Kubernetes Cluster"
echo "=================================================="

# Step 1: Build Docker image
echo ""
echo "📦 Step 1: Building Docker image..."
cd "$(dirname "$0")/.."
docker build -f kubernetes/Dockerfile -t cota-collector:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image ready: cota-collector:latest"

# Step 2: Import image into k3s
echo ""
echo "📥 Step 2: Importing image into k3s..."
docker save cota-collector:latest -o /tmp/cota-collector.tar
sudo k3s ctr images import /tmp/cota-collector.tar
rm -f /tmp/cota-collector.tar

if [ $? -ne 0 ]; then
    echo "❌ Image import failed!"
    exit 1
fi

echo "✅ Image imported into k3s"

# Step 3: Deploy to Kubernetes
echo ""
echo "🚀 Step 3: Deploying to Kubernetes..."
cd kubernetes

echo "Creating namespace..."
sudo kubectl apply -f cota-collector-namespace.yaml

echo "Creating PVC..."
sudo kubectl apply -f cota-collector-pvc.yaml

echo "Creating deployment..."
sudo kubectl apply -f cota-collector-deployment.yaml

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔍 To check status:"
echo "  sudo kubectl get pods -n cota-collector"
echo "  sudo kubectl logs -n cota-collector -l app=cota-collector --tail=50"
echo ""
echo "📊 To view collected data:"
echo "  sudo kubectl exec -n cota-collector -l app=cota-collector -- ls -lh /data/realtime_history/"


# COTA Data Collector - Kubernetes Deployment

This directory contains the Kubernetes/Docker deployment configuration for the COTA real-time data collector.

## Overview

The data collector runs as a containerized pod in Kubernetes, continuously collecting:
- **Vehicle positions**: Every 2 minutes (configurable)
- **Trip delays**: Every 2 minutes (configurable)
- **Service alerts**: Every 15 minutes (configurable, alerts change infrequently)

Data is stored in Parquet format in a persistent volume, organized by date.

## Quick Start

### Prerequisites

- Docker installed
- k3s Kubernetes cluster running
- Access to the cluster (sudo kubectl)

### Deployment

1. **Build and deploy:**
   ```bash
   cd kubernetes
   ./deploy.sh
   ```

2. **Check status:**
   ```bash
   sudo kubectl get pods -n cota-collector
   sudo kubectl logs -n cota-collector -l app=cota-collector --tail=50
   ```

## Configuration

### Environment Variables

Edit `cota-collector-deployment.yaml` to change:

- `SNAPSHOT_INTERVAL_MINUTES`: Minutes between vehicle/delay snapshots (default: 2)
- `ALERT_INTERVAL_MINUTES`: Minutes between alert collections (default: 15)
- `DATA_DIR`: Directory for storing data (default: `/data/realtime_history`)

### Storage

**Yes, this uses persistent storage!**

- **PVC**: 50Gi persistent volume using `pi-nas-storage`
- **Data location**: `/data/realtime_history/` in the container
- **Persistent**: Data survives pod restarts, deployments, and updates
- Data is organized by date: `YYYYMMDD/snapshot_YYYYMMDD_HHMMSS_*.parquet`

The persistent volume is mounted at `/data` and all collected data is stored there, so you won't lose data when the pod restarts or is updated.

## Useful Commands

### View Logs
```bash
# Follow logs
sudo kubectl logs -n cota-collector -l app=cota-collector -f

# Last 100 lines
sudo kubectl logs -n cota-collector -l app=cota-collector --tail=100
```

### Check Data Collection
```bash
# List collected files
sudo kubectl exec -n cota-collector -l app=cota-collector -- ls -lh /data/realtime_history/

# Check specific date
sudo kubectl exec -n cota-collector -l app=cota-collector -- ls -lh /data/realtime_history/20250122/
```

### Restart Collector
```bash
sudo kubectl rollout restart deployment/cota-collector -n cota-collector
```

### Update Configuration
```bash
# Edit deployment
sudo kubectl edit deployment/cota-collector -n cota-collector

# Or apply updated YAML
sudo kubectl apply -f cota-collector-deployment.yaml
```

### Access Pod Shell
```bash
POD=$(sudo kubectl get pod -n cota-collector -l app=cota-collector -o jsonpath='{.items[0].metadata.name}')
sudo kubectl exec -n cota-collector -it $POD -- /bin/bash
```

## Updating the Collector

To update the collector code:

1. **Make changes** to the source code
2. **Rebuild and redeploy:**
   ```bash
   cd kubernetes
   ./deploy.sh
   ```
3. **Restart the deployment:**
   ```bash
   sudo kubectl rollout restart deployment/cota-collector -n cota-collector
   ```

## Data Access

The collected data is stored in a persistent volume. To access it:

1. **From within the pod:**
   ```bash
   sudo kubectl exec -n cota-collector -l app=cota-collector -- ls /data/realtime_history/
   ```

2. **Copy data out:**
   ```bash
   POD=$(sudo kubectl get pod -n cota-collector -l app=cota-collector -o jsonpath='{.items[0].metadata.name}')
   sudo kubectl cp cota-collector/$POD:/data/realtime_history ./local_data
   ```

## Troubleshooting

### Pod not starting
```bash
# Check pod status
sudo kubectl describe pod -n cota-collector -l app=cota-collector

# Check events
sudo kubectl get events -n cota-collector --sort-by='.lastTimestamp'
```

### No data being collected
```bash
# Check logs for errors
sudo kubectl logs -n cota-collector -l app=cota-collector --tail=100

# Verify network connectivity
sudo kubectl exec -n cota-collector -l app=cota-collector -- curl -I https://realtime.cota.com
```

### Storage issues
```bash
# Check PVC status
sudo kubectl get pvc -n cota-collector

# Check volume mount
sudo kubectl describe pod -n cota-collector -l app=cota-collector | grep -A 5 "Mounts"
```

## Architecture

- **Namespace**: `cota-collector`
- **Deployment**: Single replica (can be scaled if needed)
- **Storage**: 50Gi PVC on `pi-nas-storage`
- **Node**: Pinned to control-plane node
- **DNS**: Uses external DNS (8.8.8.8, 8.8.4.4, 1.1.1.1)

## Notes

- The collector runs continuously until stopped
- Data is organized by date for easy analysis
- Parquet format is efficient for large datasets
- The collector handles graceful shutdown on SIGTERM/SIGINT


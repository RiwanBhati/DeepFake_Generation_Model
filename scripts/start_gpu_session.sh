#!/bin/bash
# ============================================================
# start_gpu_session.sh
# Starts or attaches to your dedicated 35GB H200 GPU pod.
# ============================================================

POD_NAME="deepfake-gpu-worker"
NAMESPACE="dgx-s-bmu-soet-240611-restricted"
YAML_PATH="scripts/gpu_interactive_pod.yaml"

echo "[INFO] Checking GPU Pod status..."

# Check if pod exists
POD_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE --no-headers 2>/dev/null | awk '{print $3}')

if [ -z "$POD_STATUS" ]; then
    echo "[INFO] Creating GPU Pod '$POD_NAME' on NVIDIA H200 (35GB)..."
    kubectl apply -f $YAML_PATH
    echo "[INFO] Waiting for Pod to be Running..."
    kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=60s
elif [ "$POD_STATUS" != "Running" ]; then
    echo "[INFO] Pod is in status '$POD_STATUS'. Restarting..."
    kubectl delete pod $POD_NAME -n $NAMESPACE --wait=true
    kubectl apply -f $YAML_PATH
    kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=60s
else
    echo "[INFO] GPU Pod '$POD_NAME' is already running!"
fi

echo "============================================================"
echo "🚀 Entering GPU Container Shell with 35GB VRAM..."
echo "Type 'exit' when you want to return to the headnode."
echo "============================================================"

kubectl exec -it $POD_NAME -n $NAMESPACE -- /bin/bash


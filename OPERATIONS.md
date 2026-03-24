# Operations

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.9` (Docker image: 3.11.9) |
| Namespace | `kubeflow-user-example-com` |
| KFP SDK | `2.15.0` |
| Container image | `kfp-workflow:latest` |
| Storage class | `local-path` |
| Base image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` |
| Serving runtime | `custom` (plugin-based predictor) |
| mamba_ssm | `2.2.2+cu122torch2.4` (pre-built wheel) |

## Development Workflow

### Setup
```bash
make venv && make install
```

### Run tests
```bash
make test
# or
python -m pytest tests/ -v
```

### Validate a spec
```bash
kfp-workflow spec validate --spec configs/pipelines/mambasl_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/mambasl_cmapss_serve.yaml --type serving
```

### Compile a pipeline
```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml
```

## Docker Build

```bash
# Build image with mamba_ssm + mambasl-new (from models/mambasl-new/)
docker build -t kfp-workflow:latest -f docker/Dockerfile .

# Import into containerd for k8s
docker save kfp-workflow:latest | sudo ctr -n k8s.io images import -
```

The Dockerfile installs:
1. `mamba_ssm` from pre-built GitHub wheel (CPU fallback via `selective_scan_ref`)
2. `kfp-workflow[serving]` (main package + kserve SDK)
3. `mambasl-new` (ML model package from `models/mambasl-new/`)

## End-to-End Deployment (MambaSL C-MAPSS Smoke Test)

### 1. Build and import Docker image
```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
docker save kfp-workflow:latest | sudo ctr -n k8s.io images import -
```

### 2. Bootstrap cluster storage
```bash
kfp-workflow cluster bootstrap \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml

# Or preview manifests first:
kfp-workflow cluster bootstrap \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml --dry-run
```

### 3. Register dataset
```bash
kfp-workflow registry dataset register \
  --name cmapss --pvc-name dataset-store --subpath CMAPSSData
```

### 4. Compile pipeline
```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml

# With CLI overrides (no YAML editing needed):
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/experiment.yaml \
  --set train.max_epochs=100 \
  --set model.config.d_model=128
```

### 5. Submit pipeline
```bash
kfp-workflow pipeline submit \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml

# With overrides:
kfp-workflow pipeline submit \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --set dataset.config.fd_name=FD003 \
  --set train.learning_rate=0.0005
```

### 6. Monitor pipeline runs
```bash
# List recent runs
kfp-workflow pipeline run list

# Get details of a specific run
kfp-workflow pipeline run get <run_id>

# Wait for a run to complete (blocks with spinner)
kfp-workflow pipeline run wait <run_id> --timeout 3600

# View component logs from a run
kfp-workflow pipeline run logs <run_id>

# View logs for a specific step
kfp-workflow pipeline run logs <run_id> --step train

# Terminate a running pipeline
kfp-workflow pipeline run terminate <run_id>

# List experiments
kfp-workflow pipeline experiment list

# JSON output for scripting
kfp-workflow --json pipeline run list
```

### 7. Verify pipeline completion
```bash
# Check model saved
kubectl exec -it <pod> -n kubeflow-user-example-com -- \
  ls /mnt/models/mambasl-cmapss/v1/

# Check model registry
kubectl exec -it <pod> -n kubeflow-user-example-com -- \
  cat /mnt/models/.model_registry.json
```

### 8. Deploy serving
```bash
kfp-workflow serve create \
  --spec configs/serving/mambasl_cmapss_serve.yaml

# Or dry run:
kfp-workflow serve create \
  --spec configs/serving/mambasl_cmapss_serve.yaml --dry-run
```

### 9. Test inference
```bash
# Check InferenceService status
kfp-workflow serve list
kfp-workflow serve get --name mambasl-cmapss-serving

# Send test request
curl -X POST http://<isvc-url>/v1/models/mambasl-cmapss-serving:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [<windowed_data>]}'
```

### 10. Cleanup
```bash
kfp-workflow serve delete \
  --name mambasl-cmapss-serving --namespace kubeflow-user-example-com
```

## Registry Operations

### Model registry (file-backed)
```bash
kfp-workflow registry model register \
  --name mambasl-cmapss --version v1 --uri /mnt/models/mambasl-cmapss/v1/model.pt

kfp-workflow registry model list
kfp-workflow registry model get --name mambasl-cmapss --version v1
```

### Dataset registry (file-backed)
```bash
kfp-workflow registry dataset register \
  --name cmapss --pvc-name dataset-store --subpath CMAPSSData

kfp-workflow registry dataset list
kfp-workflow registry dataset get --name cmapss
```

Custom registry paths:
```bash
kfp-workflow registry model list --registry-path /tmp/models.json
```

## Monitoring Stack

### Components

| Component | Namespace | Access |
|-----------|-----------|--------|
| Prometheus | `monitoring` | ClusterIP `kube-prometheus-stack-prometheus:9090` |
| Grafana | `monitoring` | `http://155.230.34.51:30090` (NodePort) |
| Alertmanager | `monitoring` | ClusterIP `kube-prometheus-stack-alertmanager:9093` |
| Node Exporter | `monitoring` | DaemonSet on port 9100 |
| kube-state-metrics | `monitoring` | ClusterIP port 8080 |
| Kepler | `kepler` | DaemonSet on port 9102 (hostNetwork) |

### Credentials

| Service | Username | Password |
|---------|----------|----------|
| Grafana | `admin` | `admin` |

### Helm Releases

```bash
# kube-prometheus-stack (Prometheus + Grafana + node-exporter + kube-state-metrics + alertmanager)
helm list -n monitoring

# Kepler (energy monitoring)
helm list -n kepler
```

### Common Operations

```bash
# Access Grafana
open http://155.230.34.51:30090

# Port-forward Prometheus UI
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090

# Check Kepler pod logs
kubectl logs -n kepler -l app.kubernetes.io/name=kepler --tail=50

# Check all Prometheus scrape targets
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &
curl -s 'http://localhost:9090/api/v1/targets' | python3 -c "
import json, sys
for t in json.load(sys.stdin)['data']['activeTargets']:
    print(f\"{t['health']:6s} {t['labels'].get('job',''):30s} {t['scrapeUrl']}\")
"

# Query Kepler energy metrics
curl -s 'http://localhost:9090/api/v1/query?query=kepler_node_platform_joules_total'
curl -s 'http://localhost:9090/api/v1/query?query=rate(kepler_container_package_joules_total[5m])'
```

### Dashboards

Grafana includes 36 dashboards:

**Kepler Energy Dashboards (8):**
- **Kepler Dashboard** — Main production dashboard: cluster/node/pod/container energy (30 panels)
- **Kepler Energy Monitoring Dashboard** — Dev/validation: Node Exporter vs Kepler comparison (20 panels)
- **Power Monitor Dashboard** — Cluster-wide + per-zone active/idle power (17 panels)
- **Power Monitor / Overview** — High-level cluster power overview (8 panels)
- **Power Monitor / Node** — Per-node zone power breakdown (5 panels)
- **Power Monitor / Namespace (Pods)** — Top power-consuming namespaces drill-down (3 panels)
- **Kepler Exporter Dashboard** — Carbon emissions footprint, CO2/kWh tracking (7 panels)
- **Power Source Comparison** — Side-by-side power estimation method comparison (4 panels)

**Kubernetes dashboards** — Compute resources, networking, kubelet, API server
**Node Exporter dashboards** — Host-level system metrics

All Kepler dashboards are persisted via labeled ConfigMaps (`grafana_dashboard=1`) in the `monitoring` namespace and are auto-provisioned by the Grafana sidecar.

Dashboard sources:
- `sustainable-computing-io/kepler` main branch (compose/default + compose/dev)
- `sustainable-computing-io/kepler` release-0.6 branch (legacy exporter)
- `sustainable-computing-io/kepler-operator` main branch (operator dashboards)

### Firewall Rules Added

UFW rules allow pod network (10.244.0.0/16) to reach hostNetwork services:
- Port 9102 (Kepler)
- Port 9100 (node-exporter)
- Ports 10249, 10257, 10259, 2381 (k8s control plane — bound to localhost, so still inaccessible)

## Architecture Notes

- All pipeline components use a single shared Docker image
- Components communicate via JSON-serialised strings
- PipelineSpec is passed as `spec_json` to every component for self-contained configuration
- Each component delegates to a `ModelPlugin` identified by `spec["model"]["name"]`
- Data PVC is mounted read-only; model PVC is mounted read-write
- Training is single-node only (no distributed/PyTorchJob)
- Heavy data (numpy arrays) saved as `.npy` on PVC; only paths passed between stages
- Custom KServe predictor runs `kfp_workflow.serving.predictor` — loads model via plugin's `predict()` method
- Pipeline submission uses `kubectl port-forward` to KFP API, then `kfp.Client`

## Adding a New Model Plugin

1. Create `src/kfp_workflow/plugins/my_model.py` implementing `ModelPlugin` ABC
2. Implement: `name()`, `load_data()`, `preprocess()`, `train()`, `evaluate()`, `save_model()`, `predict()`
3. Add import to `_build_registry()` in `src/kfp_workflow/plugins/__init__.py`
4. Create pipeline config in `configs/pipelines/` with `model.name: my-model`
5. Create serving config in `configs/serving/` with `model_name: my-model`
6. Update Docker image to include any new dependencies

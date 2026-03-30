# Operations

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.10` (Docker image: 3.11.9) |
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
# MambaSL
kfp-workflow spec validate --spec configs/pipelines/mambasl_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/mambasl_cmapss_serve.yaml --type serving
kfp-workflow spec validate --spec configs/tuning/mambasl_cmapss_tune.yaml --type tune
kfp-workflow spec validate --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml --type benchmark
kfp-workflow spec validate --spec configs/benchmarks/mambasl_cmapss_test.yaml --type benchmark

# MR-HY-SP
kfp-workflow spec validate --spec configs/pipelines/mrhysp_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/mrhysp_cmapss_serve.yaml --type serving
kfp-workflow spec validate --spec configs/tuning/mrhysp_cmapss_tune.yaml --type tune

# SOFTS
kfp-workflow spec validate --spec configs/pipelines/softs_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/softs_cmapss_serve.yaml --type serving
kfp-workflow spec validate --spec configs/tuning/softs_cmapss_tune.yaml --type tune

# Apply overrides to any supported spec type before validation
kfp-workflow spec validate --spec configs/serving/mambasl_cmapss_serve.yaml \
  --type serving --set metadata.name=serve-smoke

# Machine-safe JSON output for scripting
kfp-workflow --json spec validate --spec configs/tuning/mambasl_cmapss_tune.yaml --type tune
```

### Compile a pipeline
```bash
# MambaSL
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml

# SOFTS
kfp-workflow pipeline compile \
  --spec configs/pipelines/softs_cmapss_smoke.yaml \
  --output pipelines/softs_cmapss_smoke.yaml
```

### Compile and submit a benchmark
```bash
kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml \
  --output pipelines/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow benchmark submit \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow pipeline run wait <run_id>
```

### Hyperparameter tuning
```bash
# Preview the search space for a tuning spec
kfp-workflow tune show-space --spec configs/tuning/mambasl_cmapss_tune.yaml
kfp-workflow tune show-space --spec configs/tuning/softs_cmapss_tune.yaml

# Use the aggressive profile
kfp-workflow tune show-space --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --set hpo.builtin_profile=aggressive
kfp-workflow tune show-space --spec configs/tuning/softs_cmapss_tune.yaml \
  --set hpo.builtin_profile=aggressive

# Run local HPO (Optuna)
kfp-workflow tune run --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --set hpo.max_trials=20 --set hpo.algorithm=tpe \
  --data-mount-path ./data \
  --output results/best_params.json

kfp-workflow tune run --spec configs/tuning/softs_cmapss_tune.yaml \
  --set hpo.max_trials=20 --set hpo.algorithm=tpe \
  --data-mount-path ./data \
  --output results/softs_best_params.json

# Quick smoke test (2 trials, 2 epochs)
kfp-workflow tune run --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --set hpo.max_trials=2 --set train.max_epochs=2 \
  --data-mount-path ./data

# Generate Katib manifest for distributed HPO
kfp-workflow tune katib --spec configs/tuning/mambasl_cmapss_tune.yaml --dry-run
kfp-workflow tune katib --spec configs/tuning/softs_cmapss_tune.yaml --dry-run

# Bootstrap tune result storage before Katib
kfp-workflow cluster bootstrap \
  --type tune \
  --spec configs/tuning/mambasl_cmapss_tune.yaml

# Submit Katib experiment to cluster
kfp-workflow tune katib --spec configs/tuning/mambasl_cmapss_tune.yaml
kfp-workflow tune katib --spec configs/tuning/softs_cmapss_tune.yaml

# Inspect and download persisted Katib results
kfp-workflow tune list
kfp-workflow tune get mambasl-cmapss-hpo
kfp-workflow tune download mambasl-cmapss-hpo

# Katib trial pods run the internal shared executor:
# kfp-workflow tune trial --spec-json ... --trial-params-json ...
# Objective metrics must be printed to stdout as objective=<value>.
# Trial payloads and the aggregated results.json are stored on storage.results_pvc.

# JSON output for scripting
kfp-workflow --json tune run --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --set hpo.max_trials=5 --data-mount-path ./data
```

## Docker Build

```bash
# Build image with mamba_ssm + mambasl-new (from models/mambasl-new/)
docker build -t kfp-workflow:latest -f docker/Dockerfile .

# Preferred: push a tagged image to a registry and reference that tag in specs
# Fallback for local clusters: save and import into node containerd
docker save kfp-workflow:latest -o /tmp/kfp-workflow-latest.tar

# This environment does not allow direct host access to containerd.
# Import through a privileged helper pod instead.
kubectl create ns image-loader || true
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  name: kfp-benchmark-loader
  namespace: image-loader
spec:
  restartPolicy: Never
  containers:
  - name: nerdctl
    image: ghcr.io/containerd/nerdctl:v2.1.6
    command: ["sh", "-c", "sleep 3600"]
    securityContext:
      privileged: true
    volumeMounts:
    - name: containerd-sock
      mountPath: /run/containerd/containerd.sock
    - name: host-tmp
      mountPath: /host-tmp
  volumes:
  - name: containerd-sock
    hostPath:
      path: /run/containerd/containerd.sock
      type: Socket
  - name: host-tmp
    hostPath:
      path: /tmp
      type: Directory
YAML
kubectl wait --for=condition=Ready pod/kfp-benchmark-loader -n image-loader --timeout=120s
kubectl exec -n image-loader kfp-benchmark-loader -- \
  nerdctl --address /run/containerd/containerd.sock --namespace k8s.io load -i /host-tmp/kfp-workflow-latest.tar
kubectl delete pod -n image-loader kfp-benchmark-loader --ignore-not-found=true
```

The Dockerfile installs:
1. `mamba_ssm` from pre-built GitHub wheel (CPU fallback via `selective_scan_ref`)
2. `kfp-workflow[serving]` (main package + kserve SDK)
3. `mambasl-new` (ML model package from `models/mambasl-new/`)
4. `multirocket-new` (ML model package from `models/multirocket-new/`)

## End-to-End Deployment (MR-HY-SP C-MAPSS Smoke Test)

### 1. Build and import Docker image
```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
# Prefer pushing a versioned tag to a registry when possible.
docker save kfp-workflow:latest -o /tmp/kfp-workflow-latest.tar
# Then follow the helper-pod + nerdctl import flow in the Docker Build section above.
```

### 2. Bootstrap cluster storage
```bash
kfp-workflow cluster bootstrap \
  --spec configs/pipelines/mrhysp_cmapss_smoke.yaml
```

### 3. Compile and submit pipeline
```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mrhysp_cmapss_smoke.yaml \
  --output pipelines/mrhysp_cmapss_smoke.yaml

kfp-workflow pipeline submit \
  --spec configs/pipelines/mrhysp_cmapss_smoke.yaml
```

### 4. Hyperparameter tuning
```bash
# Preview search space
kfp-workflow tune show-space --spec configs/tuning/mrhysp_cmapss_tune.yaml

# Run local HPO (Optuna)
kfp-workflow tune run --spec configs/tuning/mrhysp_cmapss_tune.yaml \
  --set hpo.max_trials=10 --data-mount-path ./data

# Generate Katib manifest
kfp-workflow tune katib --spec configs/tuning/mrhysp_cmapss_tune.yaml --dry-run
```

### 5. Deploy serving
```bash
kfp-workflow serve create --spec configs/serving/mrhysp_cmapss_serve.yaml
kfp-workflow serve get --name mrhysp-cmapss-serving
```

## End-to-End Deployment (MambaSL C-MAPSS Smoke Test)

### 1. Build and import Docker image
```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
docker save kfp-workflow:latest -o /tmp/kfp-workflow-latest.tar
# Then follow the helper-pod + nerdctl import flow in the Docker Build section above.
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
  --name cmapss --pvc-name dataset-store --subpath cmapss/CMAPSSData
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
  --set dataset.config.fd[0].fd_name=FD003 \
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

## End-to-End Deployment (MambaSL Benchmark Smoke Test)

### 1. Build and import Docker image
```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
docker save kfp-workflow:latest -o /tmp/kfp-workflow-latest.tar
# Import with the helper pod flow shown above.
```

### 2. Bootstrap benchmark storage
```bash
kfp-workflow cluster bootstrap \
  --type benchmark \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml
```

### 3. Compile and submit the benchmark
```bash
# Kepler energy smoke benchmark
kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml \
  --output pipelines/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow benchmark submit \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml

# Test-set accuracy benchmark (F1 / precision / recall / accuracy)
kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_test.yaml \
  --output pipelines/mambasl_cmapss_test.yaml

kfp-workflow benchmark submit \
  --spec configs/benchmarks/mambasl_cmapss_test.yaml
```

### 4. Wait for completion and verify result persistence
```bash
kfp-workflow pipeline run wait <run_id>
kfp-workflow benchmark get <run_id>
kfp-workflow benchmark download <run_id>

# Example validated on 2026-03-29:
# run_id: ba893c5b-14bb-4fde-8229-a040127ee36e
# workflow: mambasl-cmapss-benchmark-smoke-2nx2f
# downloaded file:
# ./mambasl-cmapss-benchmark-smoke-ba893c5b-14bb-4fde-8229-a040127ee36e.json
```

Expected Kepler smoke result:
- `status == "succeeded"`
- `scenario.request_count == 5`
- `metrics.metric_0.delta_joules > 0`

Expected test-set accuracy result:
- `status == "succeeded"`
- `scenario.request_count == 100` (one request per FD001 test unit)
- `metrics.metric_0.f1_score` — binary F1 at RUL threshold 30
- `metrics.metric_0.precision`, `metrics.metric_0.recall`, `metrics.metric_0.accuracy`

Use direct PVC path inspection only for troubleshooting after `benchmark get/download`.

Notes:
- Benchmark tasks compile with KFP caching disabled. Cached deploy/wait outputs are invalid for side-effectful benchmark runs.
- The benchmark run task injects an Istio sidecar so it can reach the in-cluster KServe predictor service.
- On this cluster, the Kepler predictor counter updates with noticeable lag. The shipped smoke metric uses `settle_seconds: 20` so the result captures a non-zero energy delta.

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
3. (Optional) Implement HPO hooks: `hpo_search_space()`, `hpo_base_config()`, `hpo_objective()`
4. Add import to `_build_registry()` in `src/kfp_workflow/plugins/__init__.py`
5. Create pipeline config in `configs/pipelines/` with `model.name: my-model`
6. Create tuning config in `configs/tuning/` with `model.name: my-model` (if HPO supported)
7. Create serving config in `configs/serving/` with `model_name: my-model`
8. Update Docker image to include any new dependencies

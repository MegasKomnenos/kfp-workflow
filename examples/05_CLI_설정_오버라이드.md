# 05. CLI 설정 오버라이드

## 개요

kfp-workflow는 Helm과 유사한 `--set` 플래그를 제공하여, YAML 스펙 파일을 직접 수정하지 않고도 CLI에서 설정 값을 변경할 수 있습니다. 이를 통해 하나의 기본 스펙 파일로 다양한 실험 변형을 빠르게 실행할 수 있습니다.

## 핵심 개념

### 우선순위 체인

설정 값은 다음 우선순위로 결정됩니다:

```
CLI --set 값  >  YAML 스펙 값  >  플러그인 기본값
```

1. **CLI `--set` 값** — 가장 높은 우선순위. 다른 모든 값을 덮어씀
2. **YAML 스펙 값** — 스펙 파일에 명시된 값
3. **플러그인 기본값** — 코드에 하드코딩된 기본값 (최하위)

### 적용 시점

오버라이드는 YAML 파일 로드 후, Pydantic 검증 **전**에 적용됩니다. 즉:

1. YAML 파일을 raw dict로 로드
2. `--set` 값들을 dict에 적용
3. Pydantic `PipelineSpec` 모델로 검증
4. 검증된 스펙으로 컴파일/제출 진행

이 설계 덕분에 파이프라인 컴포넌트나 플러그인 코드에는 변경이 전혀 필요 없습니다.

## `--set` 문법

### 기본 형식

```
--set key.subkey=value
```

점(`.`)으로 구분된 경로를 사용하여 중첩된 YAML 구조에 접근합니다.

### 예시

```bash
# 단일 값 변경
--set train.max_epochs=100

# 중첩된 config 값 변경
--set model.config.d_model=128

# 여러 값 동시 변경
--set train.max_epochs=100 --set train.batch_size=128 --set model.config.d_model=64
```

### YAML 경로와의 대응

```yaml
# YAML 스펙
train:
  max_epochs: 2        # → --set train.max_epochs=100
  batch_size: 64       # → --set train.batch_size=128
  learning_rate: 0.001 # → --set train.learning_rate=0.0005

model:
  config:
    d_model: 32        # → --set model.config.d_model=128
    d_state: 8         # → --set model.config.d_state=16

dataset:
  config:
    fd_name: FD001     # → --set dataset.config.fd_name=FD003

runtime:
  use_gpu: false       # → --set runtime.use_gpu=true
```

## 타입 자동 변환

`--set`의 값은 문자열로 전달되지만, 자동으로 적절한 Python 타입으로 변환됩니다. 내부적으로 JSON 파싱을 먼저 시도하고, 실패하면 문자열로 유지합니다.

| 입력 | 변환 결과 | 타입 |
|------|----------|------|
| `128` | `128` | int |
| `0.001` | `0.001` | float |
| `true` | `True` | bool |
| `false` | `False` | bool |
| `null` | `None` | NoneType |
| `FD003` | `"FD003"` | str |
| `"hello"` | `"hello"` | str |
| `[1,2,3]` | `[1, 2, 3]` | list |

> 주의: 불리언 값은 반드시 소문자 `true`/`false`를 사용해야 합니다 (JSON 표준).

## 지원하는 명령어

`--set` 플래그는 다음 세 가지 명령어에서 사용할 수 있습니다:

### 1. `pipeline compile`

```bash
kfp-workflow pipeline compile \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --output pipelines/experiment.yaml \
    --set train.max_epochs=100 \
    --set model.config.d_model=128
```

변경된 설정이 컴파일된 YAML에 내장됩니다.

### 2. `pipeline submit`

```bash
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set train.max_epochs=100 \
    --set dataset.config.fd_name=FD003
```

제출 시 자동으로 컴파일되므로, 별도의 컴파일 단계 없이 바로 변형 실행이 가능합니다.

### 3. `spec validate`

```bash
kfp-workflow spec validate \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set train.max_epochs=100 \
    --set model.config.d_model=128
```

오버라이드가 적용된 상태의 스펙을 검증합니다. 실제 제출 전에 설정이 유효한지 확인하는 용도로 유용합니다.

## 플러그인 Config 스키마 검증

`spec validate`에서 `--set`을 사용하면, 플러그인이 제공하는 config 스키마에 대해서도 추가 검증이 수행됩니다.

예를 들어, `mambasl-cmapss` 플러그인은 다음 필드에 대한 스키마를 정의합니다:

**model.config 스키마:**
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `d_model` | int | 64 | 모델 차원 |
| `d_state` | int | 16 | 상태 차원 |
| `d_conv` | int | 3 | 컨볼루션 커널 크기 |
| `expand` | int | 2 | 확장 비율 |
| `dropout` | float | 0.2 | 드롭아웃 비율 |
| `window_size` | int | 50 | 슬라이딩 윈도우 크기 |
| `max_rul` | float | 125.0 | 최대 RUL 값 |

**dataset.config 스키마:**
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `fd_name` | str | FD001 | C-MAPSS 데이터셋 |
| `download_policy` | str | if_missing | 다운로드 정책 |
| `feature_mode` | str | settings_plus_sensors | 특성 모드 |
| `norm_mode` | str | condition_minmax | 정규화 모드 |

잘못된 타입을 사용하면 경고가 표시됩니다:

```bash
kfp-workflow spec validate \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set model.config.d_model=invalid_string
```

```
Warning: model.config validation: 1 validation error for MambaSLModelConfig
d_model
  Input should be a valid integer ...
```

> 스키마는 `extra="allow"`로 설정되어 있으므로, 알 수 없는 필드를 추가해도 오류가 아닌 경고만 발생합니다.

## 실전 예제

### 하이퍼파라미터 탐색

```bash
# 기본 (스모크 테스트)
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml

# 큰 모델 + 긴 학습
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set model.config.d_model=128 \
    --set model.config.d_state=32 \
    --set train.max_epochs=100 \
    --set train.patience=15

# 다른 데이터셋
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set dataset.config.fd_name=FD003

# 학습률 변경
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set train.learning_rate=0.0005 \
    --set train.weight_decay=0.001
```

### GPU 활성화

```bash
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set runtime.use_gpu=true \
    --set runtime.resources.gpu_request=1 \
    --set runtime.resources.gpu_limit=1
```

### 리소스 변경

```bash
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set runtime.resources.memory_request=16Gi \
    --set runtime.resources.memory_limit=16Gi \
    --set runtime.resources.cpu_request=8 \
    --set runtime.resources.cpu_limit=8
```

### 사전 검증 후 제출

```bash
# 1단계: 변경된 설정 검증
kfp-workflow spec validate \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set model.config.d_model=128 \
    --set train.max_epochs=100

# 2단계: 검증 통과 후 제출
kfp-workflow pipeline submit \
    --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set model.config.d_model=128 \
    --set train.max_epochs=100
```

## 오류 처리

### 잘못된 형식

```bash
--set train.max_epochs    # = 기호 없음
```
```
Error: Malformed override (expected key=value): 'train.max_epochs'
```

### 빈 키

```bash
--set =100
```
```
Error: Empty key in override: '=100'
```

### Pydantic 검증 실패

```bash
--set train.batch_size=not_a_number
```
```
ValidationError: 1 validation error for PipelineSpec
train -> batch_size
  Input should be a valid integer
```

## 다음 단계

→ [06. 서빙 배포 및 추론](06_서빙_배포_및_추론.md)

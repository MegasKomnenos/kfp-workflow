# 09. Docker 이미지 빌드

## 개요

kfp-workflow 파이프라인의 각 컴포넌트는 Docker 컨테이너 내에서 실행됩니다. 파이프라인을 실행하려면 kfp-workflow와 모델 패키지가 모두 포함된 Docker 이미지를 빌드하고 레지스트리에 푸시해야 합니다.

### 이미지 구조

이미지에는 다음이 포함됩니다:

```
베이스: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
├── mamba_ssm 2.2.2 (CUDA pre-built wheel)
├── transformers >=4.40,<4.45
├── kfp-workflow (메인 패키지)
│   ├── CLI (kfp-workflow 명령어)
│   ├── 파이프라인 컴파일러/클라이언트
│   ├── 플러그인 시스템
│   └── 서빙 모듈
└── mambasl-new (모델 패키지)
    ├── MambaSL 모델 아키텍처
    ├── C-MAPSS 데이터 처리
    └── 학습/평가 로직
```

## Dockerfile 구조

```dockerfile
# docker/Dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ARG MAMBA_WHEEL_URL=https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

WORKDIR /workspace

# 시스템 패키지
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# mamba_ssm 및 호환 transformers 설치
RUN pip install --upgrade pip setuptools wheel ninja packaging \
 && pip install "transformers>=4.40,<4.45" \
 && pip install "${MAMBA_WHEEL_URL}"

# kfp-workflow 설치
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs
RUN pip install ".[serving]"

# mambasl-new 모델 패키지 설치
COPY models/mambasl-new/pyproject.toml models/mambasl-new/README.md /workspace/mambasl-new/
COPY models/mambasl-new/src /workspace/mambasl-new/src
RUN pip install ./mambasl-new

ENTRYPOINT ["python", "-m", "kfp_workflow.cli.main"]
```

### 주요 설계 결정

**베이스 이미지**: PyTorch 공식 CUDA 런타임 이미지를 사용합니다. GPU 학습과 CPU 추론 모두 지원합니다.

**mamba_ssm 설치**: Pre-built wheel을 사용합니다. mamba_ssm은 CUDA 커널을 포함하므로 소스 빌드 시 CUDA 툴킷이 필요합니다. Pre-built wheel 사용으로 빌드 시간과 복잡성을 크게 줄입니다.

**transformers 버전 고정**: `>=4.40,<4.45`로 고정합니다. mamba_ssm 2.2.2가 `GreedySearchDecoderOnlyOutput`을 임포트하는데, 이 클래스가 transformers 4.45에서 제거되었기 때문입니다.

**2단계 패키지 설치**: kfp-workflow를 먼저 설치하고, mambasl-new를 별도로 설치합니다. 이를 통해 Docker 레이어 캐시를 효율적으로 활용합니다.

## 이미지 빌드

### Makefile 사용 (권장)

```bash
# 빌드만
make docker-build

# 빌드 + 푸시
make docker-push
```

### 수동 빌드

```bash
# 프로젝트 루트에서 실행
docker build -f docker/Dockerfile -t kfp-workflow:latest .
```

빌드 컨텍스트는 프로젝트 루트(`.`)입니다. Dockerfile이 `docker/` 디렉토리에 있으므로 `-f` 플래그로 경로를 지정합니다.

### 레지스트리에 푸시

```bash
# 태그 지정
docker tag kfp-workflow:latest your-registry/kfp-workflow:v1.0.0
docker tag kfp-workflow:latest your-registry/kfp-workflow:latest

# 푸시
docker push your-registry/kfp-workflow:v1.0.0
docker push your-registry/kfp-workflow:latest
```

## 파이프라인 스펙에서 이미지 참조

빌드하고 푸시한 이미지는 파이프라인 스펙의 `runtime.image` 필드에서 참조합니다:

```yaml
runtime:
  image: your-registry/kfp-workflow:v1.0.0
  namespace: kubeflow-user-example-com
```

서빙 스펙에서도 참조합니다:

```yaml
predictor_image: your-registry/kfp-workflow:v1.0.0
```

## 빌드 최적화

### Docker 레이어 캐시 활용

Dockerfile은 변경 빈도가 낮은 레이어를 먼저 빌드하도록 구성되어 있습니다:

```
1. 시스템 패키지 (거의 변경 없음)     ← 캐시 적중률 높음
2. mamba_ssm + transformers          ← 캐시 적중률 높음
3. pyproject.toml + README.md        ← 의존성 변경 시만 재빌드
4. src/ (kfp-workflow 소스)          ← 코드 변경 시 재빌드
5. models/mambasl-new/               ← 모델 코드 변경 시 재빌드
```

### .dockerignore 권장

빌드 컨텍스트를 최소화하기 위해 `.dockerignore` 파일을 사용합니다:

```
# .dockerignore
.git/
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
examples/
tests/
```

## 빌드 문제 해결

### mamba_ssm wheel 다운로드 실패

```
ERROR: Could not install packages due to an EnvironmentError: ...
```

→ Pre-built wheel URL이 유효한지 확인합니다. GitHub Releases 페이지에서 최신 URL을 확인하세요.

### CUDA 버전 불일치

이미지의 CUDA 버전(12.1)과 클러스터 노드의 GPU 드라이버가 호환되는지 확인합니다:

```bash
# 노드에서 GPU 드라이버 확인
nvidia-smi

# 호환성: CUDA 12.1은 드라이버 530+ 필요
```

### 이미지 크기 최적화

현재 이미지는 PyTorch + CUDA 베이스로 인해 약 8-10GB입니다. 크기를 줄이려면:

- CPU 전용 환경: `pytorch/pytorch:2.4.1-cpu` 베이스 사용
- 불필요한 패키지 제거: `pip install --no-deps` 사용 시 의존성 충돌 주의

## 다음 단계

→ [10. 새 모델 플러그인 개발](10_새_모델_플러그인_개발.md)

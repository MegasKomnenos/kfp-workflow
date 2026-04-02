# 05. CLI 설정 오버라이드

## 목적

YAML 파일을 직접 수정하지 않고 실험 값을 바꾸고 싶을 때 `--set`을 사용합니다.

## 기본 패턴

```bash
--set key=value
```

예시:

```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/fd003.yaml \
  --set dataset.config.fd[0].fd_name=FD003 \
  --set train.max_epochs=10
```

## 자주 쓰는 위치

- `train.max_epochs`
- `train.learning_rate`
- `model.config.*`
- `dataset.config.*`
- `hpo.builtin_profile`
- `scenario.pipeline.config.*`

## 튜닝에서 사용

```bash
kfp-workflow tune space \
  --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --set hpo.builtin_profile=aggressive
```

## 검증 전 사용

```bash
kfp-workflow spec validate \
  --spec configs/serving/mambasl_cmapss_serve.yaml \
  --type serving \
  --set metadata.name=serve-smoke
```

## 주의점

- 존재하지 않는 필드나 잘못된 타입은 검증 단계에서 실패할 수 있습니다.
- 배열 필드는 `fd[0]`처럼 인덱스로 접근합니다.
- 최종 우선순위는 `CLI --set > YAML 스펙 > 코드 기본값`입니다.

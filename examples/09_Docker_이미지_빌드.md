# 09. Docker 이미지 빌드

## 기본 루트 이미지

루트 통합 예제는 `kfp-workflow:latest`를 기본 이미지로 사용합니다.

```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
```

또는:

```bash
./scripts/build_image.sh
```

## 왜 중요한가

- 학습 파이프라인과 루트 벤치마크 base component는 기본적으로 이 이미지를 사용합니다.
- 루트 Katib 튜닝 실행 워크로드의 기본 이미지는 `runtime.image` 값입니다.
- 커스텀 서빙 predictor와 벤치마크 predictor 이미지는 스펙 필드로 바꿀 수 있습니다.

## 로컬 클러스터에 적재

이 저장소의 표준 helper:

```bash
./scripts/load_image_to_cluster.sh kfp-workflow:latest scouter1
```

내부적으로는 host `/tmp`에 tar를 저장한 뒤, `image-loader` 네임스페이스의 privileged helper pod에서 containerd로 import합니다. 자세한 운영 메모와 현재 표준 절차는 루트 [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md)를 따릅니다.

## 참고

패키지별 문서가 별도 Dockerfile을 언급하더라도, 루트 통합 워크플로우 설명에서는 항상 `docker/Dockerfile` 기준으로 기술하는 것이 맞습니다.

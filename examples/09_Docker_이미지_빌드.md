# 09. Docker 이미지 빌드

## 루트 통합 이미지

루트 워크플로우는 하나의 통합 이미지를 사용합니다.

```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
```

또는:

```bash
./scripts/build_image.sh
```

## 왜 중요한가

- 학습 파이프라인 스텝이 이 이미지를 사용합니다.
- 커스텀 서빙 predictor도 같은 이미지 계열을 사용합니다.
- 튜닝 trial 컨테이너도 루트 이미지에 의존합니다.

## 로컬 클러스터에 적재

예시로 tar 저장:

```bash
docker save kfp-workflow:latest -o /tmp/kfp-workflow-latest.tar
```

실제 적재 방식은 클러스터 환경에 따라 다릅니다. 이 저장소에서는 helper pod를 통한 containerd import 절차를 사용해 왔으며, 자세한 운영 메모는 루트 [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md)를 따릅니다.

## 참고

패키지별 문서가 별도 Dockerfile을 언급하더라도, 루트 통합 워크플로우 설명에서는 항상 `docker/Dockerfile` 기준으로 기술하는 것이 맞습니다.

# kfp-workflow 튜토리얼

이 디렉토리는 `kfp-workflow` 통합 워크플로우를 단계별로 익히기 위한 한국어 가이드 모음입니다.

영문 루트 문서와의 역할 분담은 다음과 같습니다.

- 루트 문서: 현재 지원 범위, 운영 절차, 구조 설명
- 이 튜토리얼: 실제 사용 순서 중심의 학습 경로

루트 참조 문서:

- [README.md](/home/scouter/proj_2026_1_etri/test/README.md)
- [CLI_COMMAND_TREE.md](/home/scouter/proj_2026_1_etri/test/CLI_COMMAND_TREE.md)
- [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md)
- [PROJECT.md](/home/scouter/proj_2026_1_etri/test/PROJECT.md)

## 학습 순서

1. [00_프로젝트_개요.md](/home/scouter/proj_2026_1_etri/test/examples/00_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EA%B0%9C%EC%9A%94.md)
2. [01_설치_및_설정.md](/home/scouter/proj_2026_1_etri/test/examples/01_%EC%84%A4%EC%B9%98_%EB%B0%8F_%EC%84%A4%EC%A0%95.md)
3. [02_스펙_파일_작성_및_검증.md](/home/scouter/proj_2026_1_etri/test/examples/02_%EC%8A%A4%ED%8E%99_%ED%8C%8C%EC%9D%BC_%EC%9E%91%EC%84%B1_%EB%B0%8F_%EA%B2%80%EC%A6%9D.md)
4. [03_파이프라인_컴파일_및_제출.md](/home/scouter/proj_2026_1_etri/test/examples/03_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%EC%BB%B4%ED%8C%8C%EC%9D%BC_%EB%B0%8F_%EC%A0%9C%EC%B6%9C.md)
5. [04_파이프라인_실행_모니터링.md](/home/scouter/proj_2026_1_etri/test/examples/04_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%EC%8B%A4%ED%96%89_%EB%AA%A8%EB%8B%88%ED%84%B0%EB%A7%81.md)
6. [06_서빙_배포_및_추론.md](/home/scouter/proj_2026_1_etri/test/examples/06_%EC%84%9C%EB%B9%99_%EB%B0%B0%ED%8F%AC_%EB%B0%8F_%EC%B6%94%EB%A1%A0.md)
7. [11_하이퍼파라미터_튜닝.md](/home/scouter/proj_2026_1_etri/test/examples/11_%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0_%ED%8A%9C%EB%8B%9D.md)
8. [12_벤치마크_워크플로우.md](/home/scouter/proj_2026_1_etri/test/examples/12_%EB%B2%A4%EC%B9%98%EB%A7%88%ED%81%AC_%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C%EC%9A%B0.md)

## 문서 목록

| 번호 | 제목 | 목적 |
|------|------|------|
| 00 | 프로젝트 개요 | 전체 구조와 현재 지원 범위 이해 |
| 01 | 설치 및 설정 | 로컬 개발 환경과 기본 검증 |
| 02 | 스펙 파일 작성 및 검증 | `PipelineSpec`, `ServingSpec`, `TuneSpec`, `BenchmarkSpec` 사용법 |
| 03 | 파이프라인 컴파일 및 제출 | 학습 파이프라인과 벤치마크 컴파일/제출 |
| 04 | 파이프라인 실행 모니터링 | 평탄화된 `pipeline get/list/wait/logs/terminate/list-experiments` 흐름 |
| 05 | CLI 설정 오버라이드 | `--set` 패턴과 주의점 |
| 06 | 서빙 배포 및 추론 | KServe 배포와 상태 점검 |
| 07 | 레지스트리 관리 | 모델/데이터셋 레지스트리 흐름 |
| 08 | 클러스터 부트스트랩 | PVC 준비 |
| 09 | Docker 이미지 빌드 | 루트 통합 이미지 빌드 |
| 10 | 새 모델 플러그인 개발 | 루트 플러그인 추가 시 고려사항 |
| 11 | 하이퍼파라미터 튜닝 | 현재 권장 Katib 제출 흐름 |
| 12 | 벤치마크 워크플로우 | 임시 서빙 기반 시나리오/메트릭 실행 |

## 범위 주의사항

- 이 튜토리얼은 루트 `kfp-workflow` 프로젝트 기준입니다.
- `models/timemixer-new`는 현재 루트 플러그인 레지스트리에 연결되어 있지 않으므로, 루트 스펙에서 바로 사용할 수 없습니다.
- 루트에 포함된 `TimeMixer/` 디렉토리는 이 튜토리얼의 주 대상이 아닙니다.

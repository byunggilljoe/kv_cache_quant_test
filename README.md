# 양자화 및 KV 캐시 성능 테스트

이 저장소는 LLM(Large Language Model)의 양자화 및 KV 캐시 성능을 테스트하기 위한 스크립트를 포함하고 있습니다. 주로 Meta-Llama-3.1-8B-Instruct 모델을 대상으로 다양한 최적화 기법의 성능을 비교합니다.

## 테스트 파일 설명

### 1. speed_test_quantization.py

vLLM 라이브러리를 사용하여 다양한 양자화 방식의 성능을 테스트하는 스크립트입니다.

**주요 기능:**
- 다양한 양자화 방식(AWQ-INT4, W4A16, W8A8, FP8-dynamic, FP8, 양자화 없음)을 비교
- vLLM 프레임워크를 사용하여 추론 성능 측정
- 동일한 프롬프트 세트에 대해 각 양자화 방식의 생성 결과 비교


### 2. speed_test_quantization_hf.py

Hugging Face Transformers 라이브러리를 사용하여 다양한 양자화 방식의 성능을 테스트하는 스크립트입니다.

**주요 기능:**
- 다양한 양자화 방식(4-bit, 8-bit, FP16, 기본값)을 비교
- 추론 시간, 메모리 사용량, 초당 토큰 수 측정
- 각 양자화 방식의 생성 결과 샘플 출력

**측정 지표:**
- 평균 추론 시간(초)
- 평균 메모리 사용량(MB)
- 평균 초당 토큰 수(tokens/sec)

### 3. speed_test_kv_cache.py

vLLM 라이브러리의 프리픽스 캐싱(KV 캐시) 기능의 성능을 테스트하는 스크립트입니다.

**주요 기능:**
- 프리픽스 캐싱 활성화/비활성화 상태에서의 성능 비교
- 동일한 프롬프트 세트에 대해 생성 결과 일치 여부 확인
- 공통 프리픽스를 가진 다수의 프롬프트에 대한 추론 속도 측정

**설정 파라미터:**
- `PREFIX_MULTIPLIER`: 공통 프리픽스 반복 횟수
- `PROMPTS_NUM_MULTIPLIER`: 프롬프트 반복 횟수
- `MAX_NUM_SEQS`: 최대 시퀀스 수
- `EAGER_MODE`: Eager 모드 활성화 여부

## 사용 방법

각 테스트 스크립트는 독립적으로 실행할 수 있습니다:

```bash
# vLLM을 사용한 양자화 테스트
python speed_test_quantization.py

# Hugging Face를 사용한 양자화 테스트
python speed_test_quantization_hf.py

# KV 캐시 성능 테스트
python speed_test_kv_cache.py
```

## 요구 사항

- Python 3.8 이상
- PyTorch
- vLLM
- Transformers
- CUDA 지원 GPU

## 참고 사항

- 테스트 결과는 하드웨어 구성, CUDA 버전, 라이브러리 버전 등에 따라 달라질 수 있습니다.
- 메모리 사용량이 많을 수 있으므로 충분한 GPU 메모리가 필요합니다.
- 각 스크립트의 파라미터를 조정하여 테스트 규모를 조절할 수 있습니다. 
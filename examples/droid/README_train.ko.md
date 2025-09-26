# DROID에서 훈련하기

여기서는 *전체* DROID 데이터셋에 대해 pi0.5 모델을 미세 조정(fine-tuning)하는 방법을 설명합니다. 이것은 pi05-DROID 훈련 파이프라인의 오픈소스 근사 재현입니다. (데이터 로딩 및 사용된 액션 공간에 약간의 차이가 있습니다) -- DROID 플랫폼에서 수집된 더 작고 사용자 정의된 데이터셋으로 모델을 미세 조정하는 방법에 대한 튜토리얼은 아래를 참조하세요.

openpi의 나머지 부분과는 달리, 여기서는 LeRobot 대신 RLDS를 전체 DROID 훈련을 위한 데이터 형식으로 사용해야 합니다. (현재 LeRobot은 DROID와 같은 대규모 데이터셋에 충분히 확장 가능하지 않기 때문입니다 -- 개선 작업이 진행 중입니다). 아래에서는 RLDS 데이터 로딩을 위해 openpi 환경을 업데이트하는 방법과 DROID 데이터셋을 다운로드하는 위치에 대한 지침을 제공합니다.

## 설치

RLDS 데이터 로딩을 위해 몇 가지 추가 종속성이 필요합니다. 다음을 실행하세요:
```bash
uv sync --group rlds
```

## DROID 데이터셋 다운로드

`gsutil` 구글 클라우드 CLI를 설치한 후 다음 명령으로 DROID 데이터셋을 다운로드할 수 있습니다:
```
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1
```

v1.0.0이 아닌 버전 1.0.1을 다운로드하는 것이 중요합니다: v1.0.1은 전체 언어 주석 세트(~75k 에피소드)를 포함하고 있지만, v1.0.0은 30k 에피소드에 대한 주석만 가지고 있습니다. 어떤 이유로 다른 버전을 사용하고 싶다면, [여기](src/openpi/training/droid_rlds_dataset.py)의 `DroidRldsDataset` 객체에서 `version="1.0.1"` 줄을 수정하세요.

DROID RLDS 데이터셋을 다운로드하려면 1.8TB의 디스크 저장 공간이 필요합니다.

## 실행

먼저, `TrainConfig`의 `rlds_data_dir` 경로를 `droid` 데이터셋을 다운로드한 디렉토리로 변경하세요 (참조: [src/openpi/training/config.py](src/openpi/training/config.py)).

그런 다음, 정규화 통계를 계산합니다 (약 10분 소요):
```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi05_full_droid_finetune --max-frames 10_000_000
```

훈련을 실행합니다:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_full_droid_finetune --exp-name=my_experiment --overwrite
```

**참고**: 원래 pi0.5-DROID 모델은 관절 속도 액션(joint velocity actions)으로 훈련되었습니다.
관절 속도 액션은 시뮬레이션된 평가 환경과 호환되지 않습니다 (시뮬레이션하기 훨씬 더 어렵습니다).
따라서, 관절 속도 액션으로 훈련하는 것을 권장하지 않으며, 여기서는 관절 위치 액션(joint position actions)을 사용합니다.


## 컴퓨팅 요구 사항

DROID 훈련 구성은 수렴(100k 반복, bs256, 약 1 epoch)을 위해 8개의 H100 GPU에서 약 2일이 필요합니다.
pi0 초기화 대신 PaliGemma에서 시작하는 경우, 8개의 H100에서 약 5일(240k 반복, 즉 3 epoch)을 계획하세요.

더 저렴한 미세 조정을 위해 LoRA를 실험해 보았지만, 아직 정책이 좋은 성능을 내는 것을 발견하지 못했습니다.


## 데이터 필터링

다른 다양한 실제 로봇 데이터셋과 마찬가지로 DROID 데이터셋은 완벽하게 "깨끗"하지 않으며, 데이터 필터링이 정책 성능을 크게 향상시킨다는 것을 발견했습니다. 구체적으로, DROID 데이터셋에는 로봇이 움직이지 않는 많은 *유휴* 타임스텝이 포함되어 있습니다 (부분적으로는 데이터 수집 중에 사용된 VR 원격 조작 인터페이스 때문이며, 여기서는 자세히 다루지 않겠습니다). 이러한 유휴 전환(idle transitions)을 적절히 필터링하면 정책 성능을 향상시킬 수 있습니다.

기본적으로, openpi 훈련 레시피는 모든 pi-DROID 모델을 훈련하는 데 사용된 것과 동일한 유휴 필터를 구현합니다. 우리는 훈련 중에 샘플링할 데이터셋 인덱스를 미리 계산하여 이를 구현합니다. 이러한 인덱스를 계산하는 방법은 [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py)에서 확인할 수 있습니다. 대략적으로 말하면, 다음 액션 청크가 대부분 유휴 상태일 타임스텝을 필터링합니다. 훈련 중에 코드는 클라우드 저장소에서 미리 계산된 인덱스 목록을 자동으로 가져와 적용합니다. 유휴 필터를 수정하거나 사용자 정의 샘플링 로직을 만들고 싶다면, 스크립트를 수정하여 새 인덱스 목록을 생성하고 [src/openpi/training/config.py](src/openpi/training/config.py)의 `filter_dict_path="<path_to_filter_dict>"` 인자를 통해 제공할 수 있습니다.

**참고**: 필터링 인덱스 목록은 위 다운로드 섹션에서 언급된 `droid/1.0.1` 데이터셋에만 유효하며, 다른 버전의 DROID 데이터셋에는 유효한 필터링을 제공하지 않으므로, 반드시 위 데이터셋을 다운로드하세요! 사용자 정의 DROID 버전이 있는 경우, [compute_droid_nonidle_ranges.py](examples/droid/compute_droid_nonidle_ranges.py) 스크립트를 다시 실행하여 새 샘플링 인덱스 목록을 생성할 수 있습니다.

## RoboArena

DROID 정책을 [RoboArena 벤치마크](https://robo-arena.github.io/)에 제출하는 것을 고려해보세요. 이를 통해 다양한 작업 및 장면에서 **실제 세계에서** 정책을 평가할 수 있습니다! :)

RoboArena에 대해 질문이 있으시면 [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com)으로 이메일을 보내주세요.


# 사용자 정의 DROID 데이터셋에서 미세 조정하기

여기서는 DROID 플랫폼에서 수집된 사용자 정의 (더 작은) 데이터셋에서 모델을 미세 조정하는 방법을 설명합니다. 다른 데이터셋과 마찬가지로, 먼저 사용자 정의 DROID 데이터셋을 LeRobot으로 변환한 다음, 모델(pi05-droid)을 미세 조정할 것입니다.

참고: 여기서는 사용자 정의 DROID 미세 조정 데이터셋이 상대적으로 작다고 가정하기 때문에(<10시간) LeRobot을 사용합니다. 더 큰 데이터셋(전체 DROID 데이터셋과 같은)의 경우, 더 나은 효율성을 위해 RLDS를 사용하는 것을 권장합니다 (위 예제 참조).


## 1단계: 사용자 정의 DROID 데이터셋을 LeRobot으로 변환하기

이 예제에서는 실제 DROID 데이터셋의 작은 하위 집합을 사용할 것입니다. 이것은 단 30개의 시연으로 구성된 하위 집합입니다 -- 여러분은 자신의 데이터셋을 사용할 것이라고 가정하지만, 여기 우리 하위 집합(1.6GB)을 다운로드하는 명령어가 있습니다:
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 <your_target_path>
```

또한 DROID 데이터셋에 대한 언어 주석을 다운로드하여 시연과 언어 지침을 연결할 수 있습니다. 다시 말하지만, 자신의 데이터의 경우 언어 지침을 수동으로 입력할 수 있으므로 주석을 다운로드할 필요가 없습니다. DROID 언어 주석(12MB)을 다운로드하려면 다음을 실행하세요:
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json <your_target_dir>
```

자신의 데이터셋의 경우, 각 에피소드 디렉토리에 `recordings/MP4`라는 폴더가 있는지 확인하세요. 그렇지 않다면, 먼저 [여기](https://github.com/droid-dataset/droid/blob/main/scripts/convert/svo_to_mp4.py)에 있는 스크립트를 사용하여 (SVO 파일에서) MP4 비디오 추출을 실행해야 합니다.

이제 `convert_droid_to_lerobot.py` 스크립트를 사용하여 이 데이터셋의 LeRobot 버전을 만들 것입니다 (30개 시연에 대해 5분 미만 소요):
```
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir <your_target_path>
```

## 2단계: 사용자 정의 데이터셋으로 미세 조정 실행하기

이제 변환된 사용자 정의 데이터셋으로 미세 조정을 실행할 수 있습니다. 우리가 만든 사용자 정의 데이터셋에서 `pi05_droid`를 미세 조정하기 위한 예제 구성을 제공합니다.
다른 기본 모델과 함께 작동하도록 구성을 쉽게 수정하거나 `config.py`에서 사용자 정의 DROID 데이터셋을 사용할 수 있습니다 (`pi05_droid_finetune` 검색).

훈련을 시작하려면:
```
uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite
```

훈련이 완료되면 [`examples/droid/README.md`](examples/droid/README.md)의 지침에 따라 정책을 제공하고 로봇에서 실행할 수 있습니다.


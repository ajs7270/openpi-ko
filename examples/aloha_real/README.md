# Run Aloha (Real Robot)

This example demonstrates how to run with a real robot using an [ALOHA setup](https://github.com/tonyzhaozh/aloha). See [here](../../docs/remote_inference.md) for instructions on how to load checkpoints and run inference. We list the relevant checkpoint paths for each provided fine-tuned model below.

## Prerequisites

This repo uses a fork of the ALOHA repo, with very minor modifications to use Realsense cameras.

1. Follow the [hardware installation instructions](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation) in the ALOHA repo.
1. Modify the `third_party/aloha/aloha_scripts/realsense_publisher.py` file to use serial numbers for your cameras.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.aloha_real.main
```

Terminal window 2:

```bash
roslaunch aloha ros_nodes.launch
```

Terminal window 3:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## **ALOHA Checkpoint Guide**


The `pi0_base` model can be used in zero shot for a simple task on the ALOHA platform, and we additionally provide two example fine-tuned checkpoints, “fold the towel” and “open the tupperware and put the food on the plate,” which can perform more advanced tasks on the ALOHA.

While we’ve found the policies to work in unseen conditions across multiple ALOHA stations, we provide some pointers here on how best to set up scenes to maximize the chance of policy success. We cover the prompts to use for the policies, objects we’ve seen it work well on, and well-represented initial state distributions. Running these policies in zero shot is still a very experimental feature, and there is no guarantee that they will work on your robot. The recommended way to use `pi0_base` is by finetuning with data from the target robot.


---

### **Toast Task**

This task involves the robot taking two pieces of toast out of a toaster and placing them on a plate.

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_base`
- **Prompt**: "take the toast out of the toaster"
- **Objects needed**: Two pieces of toast, a plate, and a standard toaster.
- **Object Distribution**:
  - Works on both real toast and rubber fake toast
  - Compatible with standard 2-slice toasters
  - Works with plates of varying colors

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- The toaster should be positioned in the top-left quadrant of the workspace.
- Both pieces of toast should start inside the toaster, with at least 1 cm of bread sticking out from the top.
- The plate should be placed roughly in the lower-center of the workspace.
- Works with both natural and synthetic lighting, but avoid making the scene too dark (e.g., don't place the setup inside an enclosed space or under a curtain).


### **Towel Task**

This task involves folding a small towel (e.g., roughly the size of a hand towel) into eighths.

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **Prompt**: "fold the towel"
- **Object Distribution**:
  - Works on towels of varying solid colors
  - Performance is worse on heavily textured or striped towels

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- The towel should be flattened and roughly centered on the table.
- Choose a towel that does not blend in with the table surface.


### **Tupperware Task**

This task involves opening a tupperware filled with food and pouring the contents onto a plate.

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **Prompt**: "open the tupperware and put the food on the plate"
- **Objects needed**: Tupperware, food (or food-like items), and a plate.
- **Object Distribution**:
  - Works on various types of fake food (e.g., fake chicken nuggets, fries, and fried chicken).
  - Compatible with tupperware of different lid colors and shapes, with best performance on square tupperware with a corner flap (see images below).
  - The policy has seen plates of varying solid colors.

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- Best performance observed when both the tupperware and plate are roughly centered in the workspace.
- Positioning:
  - Tupperware should be on the left.
  - Plate should be on the right or bottom.
  - The tupperware flap should point toward the plate.

## Training on your own Aloha dataset

1. Convert the dataset to the LeRobot dataset v2.0 format.

    We provide a script [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py) that converts the dataset to the LeRobot dataset v2.0 format. As an example we have converted the `aloha_pen_uncap_diverse_raw` dataset from the [BiPlay repo](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw) and uploaded it to the HuggingFace Hub as [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse).


2. Define a training config that uses the custom dataset.

    We provide the [pi0_aloha_pen_uncap config](../../src/openpi/training/config.py) as an example. You should refer to the root [README](../../README.md) for how to run training with the new config.

IMPORTANT: Our base checkpoint includes normalization stats from various common robot configurations. When fine-tuning a base checkpoint with a custom dataset from one of these configurations, we recommend using the corresponding normalization stats provided in the base checkpoint. In the example, this is done by specifying the trossen asset_id and a path to the pretrained checkpoint’s asset directory within the AssetsConfig.

---

# Aloha 실행 (실제 로봇) (한국어)

이 예제는 [ALOHA 설정](https://github.com/tonyzhaozh/aloha)을 사용하여 실제 로봇으로 실행하는 방법을 보여줍니다. 체크포인트를 로드하고 추론을 실행하는 방법에 대한 지침은 [여기](../../docs/remote_inference.md)를 참조하십시오. 아래에 제공된 각 미세 조정 모델에 대한 관련 체크포인트 경로를 나열합니다.

## 사전 요구 사항

이 저장소는 Realsense 카메라를 사용하기 위한 아주 작은 수정이 가해진 ALOHA 저장소의 포크를 사용합니다.

1. ALOHA 저장소의 [하드웨어 설치 지침](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation)을 따르십시오.
1. `third_party/aloha/aloha_scripts/realsense_publisher.py` 파일을 수정하여 카메라의 일련 번호를 사용하도록 하십시오.

## 도커 사용

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## 도커 미사용

터미널 창 1:

```bash
# 가상 환경 생성
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# 로봇 실행
python -m examples.aloha_real.main
```

터미널 창 2:

```bash
roslaunch aloha ros_nodes.launch
```

터미널 창 3:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## **ALOHA 체크포인트 가이드**


`pi0_base` 모델은 ALOHA 플랫폼의 간단한 작업에 대해 제로샷으로 사용할 수 있으며, 추가로 "수건 접기"와 "밀폐용기 열고 음식을 접시에 담기"라는 두 가지 미세 조정된 체크포인트 예제를 제공하여 ALOHA에서 더 고급 작업을 수행할 수 있습니다.

여러 ALOHA 스테이션에서 보이지 않는 조건에서도 정책이 작동하는 것을 확인했지만, 정책 성공 가능성을 극대화하기 위해 장면을 가장 잘 설정하는 방법에 대한 몇 가지 포인터를 여기에 제공합니다. 정책에 사용할 프롬프트, 잘 작동하는 것으로 확인된 객체 및 잘 표현된 초기 상태 분포를 다룹니다. 이러한 정책을 제로샷으로 실행하는 것은 여전히 매우 실험적인 기능이며, 로봇에서 작동한다는 보장은 없습니다. `pi0_base`를 사용하는 권장 방법은 대상 로봇의 데이터로 미세 조정하는 것입니다.


---

### **토스트 작업**

이 작업은 로봇이 토스터에서 토스트 두 조각을 꺼내 접시에 놓는 것을 포함합니다.

- **체크포인트 경로**: `gs://openpi-assets/checkpoints/pi0_base`
- **프롬프트**: "take the toast out of the toaster"
- **필요한 객체**: 토스트 두 조각, 접시, 표준 토스터.
- **객체 분포**:
  - 실제 토스트와 고무 가짜 토스트 모두에서 작동합니다.
  - 표준 2슬라이스 토스터와 호환됩니다.
  - 다양한 색상의 접시에서 작동합니다.

### **장면 설정 가이드라인**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- 토스터는 작업 공간의 왼쪽 상단 사분면에 위치해야 합니다.
- 두 조각의 토스트는 모두 토스터 안에서 시작해야 하며, 빵의 최소 1cm가 위로 튀어나와 있어야 합니다.
- 접시는 작업 공간의 중앙 하단에 대략적으로 놓여야 합니다.
- 자연광과 인공 조명 모두에서 작동하지만, 장면을 너무 어둡게 만들지 마십시오 (예: 밀폐된 공간이나 커튼 아래에 설치하지 마십시오).


### **수건 작업**

이 작업은 작은 수건(예: 손수건 크기 정도)을 8등분으로 접는 것을 포함합니다.

- **체크포인트 경로**: `gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **프롬프트**: "fold the towel"
- **객체 분포**:
  - 다양한 단색 수건에서 작동합니다.
  - 질감이 많거나 줄무늬가 있는 수건에서는 성능이 저하됩니다.

### **장면 설정 가이드라인**
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- 수건은 평평하게 펴서 테이블 중앙에 대략적으로 놓아야 합니다.
- 테이블 표면과 섞이지 않는 수건을 선택하십시오.


### **밀폐용기 작업**

이 작업은 음식으로 채워진 밀폐용기를 열고 내용물을 접시에 붓는 것을 포함합니다.

- **체크포인트 경로**: `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **프롬프트**: "open the tupperware and put the food on the plate"
- **필요한 객체**: 밀폐용기, 음식 (또는 음식과 유사한 품목), 접시.
- **객체 분포**:
  - 다양한 종류의 가짜 음식(예: 가짜 치킨 너겟, 감자튀김, 프라이드 치킨)에서 작동합니다.
  - 다양한 뚜껑 색상과 모양의 밀폐용기와 호환되며, 모서리에 플랩이 있는 사각형 밀폐용기에서 최상의 성능을 보입니다 (아래 이미지 참조).
  - 정책은 다양한 단색 접시를 본 적이 있습니다.

### **장면 설정 가이드라인**
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- 밀폐용기와 접시가 모두 작업 공간 중앙에 대략적으로 위치할 때 최상의 성능이 관찰되었습니다.
- 위치:
  - 밀폐용기는 왼쪽에 있어야 합니다.
  - 접시는 오른쪽이나 아래에 있어야 합니다.
  - 밀폐용기 플랩은 접시를 향해야 합니다.

## 자신의 Aloha 데이터셋으로 훈련하기

1. 데이터셋을 LeRobot 데이터셋 v2.0 형식으로 변환합니다.

    데이터셋을 LeRobot 데이터셋 v2.0 형식으로 변환하는 [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py) 스크립트를 제공합니다. 예제로 [BiPlay 저장소](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw)의 `aloha_pen_uncap_diverse_raw` 데이터셋을 변환하여 HuggingFace Hub에 [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse)로 업로드했습니다.


2. 사용자 정의 데이터셋을 사용하는 훈련 설정을 정의합니다.

    [pi0_aloha_pen_uncap config](../../src/openpi/training/config.py)를 예제로 제공합니다. 새 설정으로 훈련을 실행하는 방법은 루트 [README](../../README.md)를 참조해야 합니다.

중요: 기본 체크포인트에는 다양한 일반적인 로봇 구성의 정규화 통계가 포함되어 있습니다. 이러한 구성 중 하나에서 사용자 정의 데이터셋으로 기본 체크포인트를 미세 조정할 때, 기본 체크포인트에 제공된 해당 정규화 통계를 사용하는 것이 좋습니다. 예제에서는 AssetsConfig 내에서 trossen asset_id와 사전 훈련된 체크포인트의 자산 디렉토리 경로를 지정하여 이 작업을 수행합니다.

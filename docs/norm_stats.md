# Normalization statistics

Following common practice, our models normalize the proprioceptive state inputs and action targets during policy training and inference. The statistics used for normalization are computed over the training data and stored alongside the model checkpoint.

## Reloading normalization statistics

When you fine-tune one of our models on a new dataset, you need to decide whether to (A) reuse existing normalization statistics or (B) compute new statistics over your new training data. Which option is better for you depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. Below, we list all the available pre-training normalization statistics for each model.

**If your target robot matches one of these pre-training statistics, consider reloading the same normalization statistics.** By reloading the normalization statistics, the actions in your dataset will be more "familiar" to the model, which can lead to better performance. You can reload the normalization statistics by adding an `AssetsConfig` to your training config that points to the corresponding checkpoint directory and normalization statistics ID, like below for the `Trossen` (aka ALOHA) robot statistics of the `pi0_base` checkpoint:

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

For an example of a full training config that reloads normalization statistics, see the `pi0_aloha_pen_uncap` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

**Note:** To successfully reload normalization statistics, it's important that your robot + dataset are following the action space definitions used in pre-training. We provide a detailed description of our action space definitions below.

**Note #2:** Whether reloading normalization statistics is beneficial depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. We recommend to always try both, reloading and training with a fresh set of statistics computed on your new dataset (see [main README](../README.md) for instructions on how to compute new statistics), and pick the one that works better for your task.


## Provided Pre-training Normalization Statistics

Below is a list of all the pre-training normalization statistics we provide. We provide them for both, the `pi0_base` and `pi0_fast_base` models. For `pi0_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_base/assets` and for `pi0_fast_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_fast_base/assets`.
| Robot | Description | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF dual arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| Franka Emika (DROID) | 7-DoF arm with parallel gripper based on the DROID setup | droid |
| Franka Emika (non-DROID) | Franka FR3 arm with Robotiq 2F-85 gripper | franka |
| UR5e | 6-DoF UR5e arm with Robotiq 2F-85 gripper | ur5e |
| UR5e bi-manual | Bi-manual UR5e setup with Robotiq 2F-85 grippers | ur5e_dual |
| ARX | Bi-manual ARX-5 robot arm setup with parallel gripper | arx |
| ARX mobile | Mobile version of bi-manual ARX-5 robot arm setup mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with 2x ARX-5 arms | fibocom_mobile |


## Pi0 Model Action Space Definitions

Out of the box, both the `pi0_base` and `pi0_fast_base` use the following action space definitions (left and right are defined looking from behind the robot towards the workspace):
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

The proprioceptive state uses the same definitions as the action space, except for the base x-y position (the last two dimensions) for mobile robots, which we don't include in the proprioceptive state.

For 7-DoF robots (e.g. Franka), we use the first 7 dimensions of the action space for the joint actions, and the 8th dimension for the gripper action.

General info for Pi robots:
- Joint angles are expressed in radians, with position zero corresponding to the zero position reported by each robot's interface library, except for ALOHA, where the standard ALOHA code uses a slightly different convention (see the [ALOHA example code](../examples/aloha_real/README.md) for details).
- Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
- Control frequencies are either 20 Hz for UR5e and Franka, and 50 Hz for ARX and Trossen (ALOHA) arms.

For DROID, we use the original DROID action configuration, with joint velocity actions in the first 7 dimensions and gripper actions in the 8th dimension + a control frequency of 15 Hz.

---

# 정규화 통계 (한국어)

일반적인 관행에 따라, 저희 모델은 정책 훈련 및 추론 중에 고유 수용성 상태 입력과 행동 목표를 정규화합니다. 정규화에 사용되는 통계는 훈련 데이터에 대해 계산되며 모델 체크포인트와 함께 저장됩니다.

## 정규화 통계 다시 불러오기

저희 모델 중 하나를 새로운 데이터셋에 미세 조정할 때, (A) 기존 정규화 통계를 재사용할지 또는 (B) 새로운 훈련 데이터에 대해 새로운 통계를 계산할지 결정해야 합니다. 어떤 옵션이 더 좋은지는 여러분의 로봇과 작업이 사전 훈련 데이터셋의 로봇 및 작업 분포와 얼마나 유사한지에 따라 다릅니다. 아래에는 각 모델에 대해 사용 가능한 모든 사전 훈련 정규화 통계를 나열합니다.

**만약 여러분의 대상 로봇이 이러한 사전 훈련 통계 중 하나와 일치한다면, 동일한 정규화 통계를 다시 불러오는 것을 고려해 보십시오.** 정규화 통계를 다시 불러오면, 데이터셋의 행동이 모델에게 더 "익숙"해져서 더 나은 성능을 이끌어 낼 수 있습니다. 아래의 `pi0_base` 체크포인트의 `Trossen` (일명 ALOHA) 로봇 통계처럼, 해당 체크포인트 디렉토리와 정규화 통계 ID를 가리키는 `AssetsConfig`를 훈련 설정에 추가하여 정규화 통계를 다시 불러올 수 있습니다:

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

정규화 통계를 다시 불러오는 전체 훈련 설정의 예시는 [훈련 설정 파일](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)의 `pi0_aloha_pen_uncap` 설정을 참조하십시오.

**참고:** 정규화 통계를 성공적으로 다시 불러오려면, 여러분의 로봇 + 데이터셋이 사전 훈련에 사용된 행동 공간 정의를 따르는 것이 중요합니다. 아래에 저희의 행동 공간 정의에 대한 자세한 설명을 제공합니다.

**참고 #2:** 정규화 통계를 다시 불러오는 것이 유익한지 여부는 여러분의 로봇과 작업이 사전 훈련 데이터셋의 로봇 및 작업 분포와 얼마나 유사한지에 따라 다릅니다. 항상 새로운 데이터셋에서 계산된 새로운 통계로 다시 불러오고 훈련하는 두 가지 방법을 모두 시도해보고, 여러분의 작업에 더 잘 맞는 것을 선택하는 것을 권장합니다 (새로운 통계를 계산하는 방법에 대한 지침은 [메인 README](../README.md) 참조).


## 제공되는 사전 훈련 정규화 통계

아래는 저희가 제공하는 모든 사전 훈련 정규화 통계 목록입니다. `pi0_base`와 `pi0_fast_base` 모델 모두에 대해 제공합니다. `pi0_base`의 경우, `assets_dir`를 `gs://openpi-assets/checkpoints/pi0_base/assets`로 설정하고, `pi0_fast_base`의 경우 `assets_dir`를 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`로 설정하십시오.

| 로봇 | 설명 | 자산 ID |
|---|---|---|
| ALOHA | 평행 그리퍼가 있는 6-DoF 듀얼 암 로봇 | trossen |
| Mobile ALOHA | Slate 베이스에 장착된 모바일 버전의 ALOHA | trossen_mobile |
| Franka Emika (DROID) | DROID 설정을 기반으로 한 평행 그리퍼가 있는 7-DoF 암 | droid |
| Franka Emika (non-DROID) | Robotiq 2F-85 그리퍼가 있는 Franka FR3 암 | franka |
| UR5e | Robotiq 2F-85 그리퍼가 있는 6-DoF UR5e 암 | ur5e |
| UR5e bi-manual | Robotiq 2F-85 그리퍼가 있는 양팔 UR5e 설정 | ur5e_dual |
| ARX | 평행 그리퍼가 있는 양팔 ARX-5 로봇 암 설정 | arx |
| ARX mobile | Slate 베이스에 장착된 양팔 ARX-5 로봇 암 설정의 모바일 버전 | arx_mobile |
| Fibocom mobile | 2개의 ARX-5 암이 있는 Fibocom 모바일 로봇 | fibocom_mobile |


## Pi0 모델 행동 공간 정의

기본적으로 `pi0_base`와 `pi0_fast_base`는 다음 행동 공간 정의를 사용합니다 (왼쪽과 오른쪽은 로봇 뒤에서 작업 공간을 바라보는 시점에서 정의됨):
```
    "dim_0:dim_5": "왼쪽 팔 관절 각도",
    "dim_6": "왼쪽 팔 그리퍼 위치",
    "dim_7:dim_12": "오른쪽 팔 관절 각도 (양팔 전용)",
    "dim_13": "오른쪽 팔 그리퍼 위치 (양팔 전용)",

    # 모바일 로봇의 경우:
    "dim_14:dim_15": "x-y 베이스 속도 (모바일 로봇 전용)",
```

고유 수용성 상태는 모바일 로봇의 베이스 x-y 위치(마지막 두 차원)를 제외하고 행동 공간과 동일한 정의를 사용하며, 이는 고유 수용성 상태에 포함하지 않습니다.

7-DoF 로봇(예: Franka)의 경우, 행동 공간의 처음 7개 차원을 관절 행동에 사용하고, 8번째 차원을 그리퍼 행동에 사용합니다.

Pi 로봇에 대한 일반 정보:
- 관절 각도는 라디안으로 표현되며, 각 로봇의 인터페이스 라이브러리에서 보고된 0 위치에 해당하는 위치 0을 가집니다. 단, ALOHA의 경우 표준 ALOHA 코드는 약간 다른 규칙을 사용합니다 (자세한 내용은 [ALOHA 예제 코드](../examples/aloha_real/README.md) 참조).
- 그리퍼 위치는 [0.0, 1.0] 범위이며, 0.0은 완전히 열린 상태, 1.0은 완전히 닫힌 상태에 해당합니다.
- 제어 주파수는 UR5e 및 Franka의 경우 20Hz, ARX 및 Trossen(ALOHA) 암의 경우 50Hz입니다.

DROID의 경우, 원래 DROID 행동 구성을 사용하며, 처음 7개 차원에 관절 속도 행동, 8번째 차원에 그리퍼 행동 + 15Hz의 제어 주파수를 사용합니다.

# DROID Policies in openpi

We offer instructions for:
- [Running inference for our best $pi_{0.5}$-DROID policy](./README.md#running-droid-inference)
- [Running inference for other pre-trained DROID policies ($\pi_0$, $\pi_0$-FAST, ...)](./README.md#running-roboarena-baseline-policies)
- [Pre-training *generalist* policies on the *full* DROID dataset](./README_train.md#training-on-droid)
- [Fine-tuning expert $\pi_{0.5}$ on your custom DROID dataset](./README_train.md#fine-tuning-on-custom-droid-datasets)

## Running DROID Inference

This example shows how to run the fine-tuned $\pi_{0.5}$-DROID model on the [DROID robot platform](https://github.com/droid-dataset/droid). Based on the [public RoboArena benchmark](https://robo-arena.github.io/leaderboard), this is currently our strongest generalist DROID policy. 


### Step 1: Start a policy server

Since the DROID control laptop does not have a powerful GPU, we will start a remote policy server on a different machine with a more powerful GPU and then query it from the DROID control laptop during inference.

1. On a machine with a powerful GPU (~NVIDIA 4090), clone and install the `openpi` repository following the instructions in the [README](https://github.com/Physical-Intelligence/openpi).
2. Start the OpenPI server via the following command:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

You can also run the equivalent command below:

```bash
uv run scripts/serve_policy.py --env=DROID
```

### Step 2: Run the DROID robot

1. Make sure you have the most recent version of the DROID package installed on both the DROID control laptop and the NUC.
2. On the control laptop, activate your DROID conda environment.
3. Clone the openpi repo and install the openpi client, which we will use to connect to the policy server (this has very few dependencies and should be very fast to install): with the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`.
4. Install `tyro`, which we will use for command line parsing: `pip install tyro`.
5. Copy the `main.py` file from this directory to the `$DROID_ROOT/scripts` directory.
6. Replace the camera IDs in the `main.py` file with the IDs of your cameras (you can find the camera IDs by running `ZED_Explorer` in the command line, which will open a tool that shows you all connected cameras and their IDs -- you can also use it to make sure that the cameras are well-positioned to see the scene you want the robot to interact with).
7. Run the `main.py` file. Make sure to point the IP and host address to the policy server. (To make sure the server machine is reachable from the DROID laptop, you can run `ping <server_ip>` from the DROID laptop.) Also make sure to specify the external camera to use for the policy (we only input one external camera), choose from ["left", "right"].

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

The script will ask you to enter a free-form language instruction for the robot to follow. Make sure to point the cameras at the scene you want the robot to interact with. You _do not_ need to carefully control camera angle, object positions, etc. The policy is fairly robust in our experience. Happy prompting!

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot reach policy server | Make sure the server is running and the IP and port are correct. You can check that the server machine is reachable by running `ping <server_ip>` from the DROID laptop. |
| Cannot find cameras | Make sure the camera IDs are correct and that the cameras are connected to the DROID laptop. Sometimes replugging the cameras can help. You can check all connected cameras by running `ZED_Explore` in the command line. |
| Policy inference is slow / inconsistent | Try using a wired internet connection for the DROID laptop to reduce latency (0.5 - 1 sec latency per chunk is normal). |
| Policy does not perform the task well | In our experiments, the policy could perform simple table top manipulation tasks (pick-and-place) across a wide range of environments, camera positions, and lighting conditions. If the policy does not perform the task well, you can try modifying the scene or object placement to make the task easier. Also make sure that the camera view you are passing to the policy can see all relevant objects in the scene (the policy is only conditioned on a single external camera + wrist camera, make sure you are feeding the desired camera to the policy). Use `ZED_Explore` to check that the camera view you are passing to the policy can see all relevant objects in the scene. Finally, the policy is far from perfect and will fail on more complex manipulation tasks, but it usually makes a decent effort. :) |


## Running Other Policies

We provide configs for running the baseline DROID policies from the [RoboArena](https://robo-arena.github.io/) paper. Simply run the commands below to start inference servers for the respective policies. Then follow the instructions above to run evaluation on the DROID robot.

```
# Train from pi0-FAST, using FAST tokenizer
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# Train from pi0, using flow matching
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid

# Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid

# Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid

# Trained from PaliGemma, using FSQ tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid

# pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

You can find the inference configs in [roboarena_config.py](../../src/openpi/training/misc/roboarena_config.py).

---

# openpi의 DROID 정책 (한국어)

다음 지침을 제공합니다:
- [최상의 $pi_{0.5}$-DROID 정책에 대한 추론 실행하기](./README.md#running-droid-inference)
- [다른 사전 훈련된 DROID 정책($\pi_0$, $\pi_0$-FAST, ...)에 대한 추론 실행하기](./README.md#running-roboarena-baseline-policies)
- [*전체* DROID 데이터셋에서 *일반화* 정책 사전 훈련하기](./README_train.md#training-on-droid)
- [사용자 정의 DROID 데이터셋에서 전문가 $\pi_{0.5}$ 미세 조정하기](./README_train.md#fine-tuning-on-custom-droid-datasets)

## DROID 추론 실행하기

이 예제는 [DROID 로봇 플랫폼](https://github.com/droid-dataset/droid)에서 미세 조정된 $\pi_{0.5}$-DROID 모델을 실행하는 방법을 보여줍니다. [공개 RoboArena 벤치마크](https://robo-arena.github.io/leaderboard)를 기반으로, 이것은 현재 가장 강력한 일반화 DROID 정책입니다.

### 1단계: 정책 서버 시작하기

DROID 제어 노트북에는 강력한 GPU가 없으므로, 더 강력한 GPU가 있는 다른 머신에서 원격 정책 서버를 시작한 다음 추론 중에 DROID 제어 노트북에서 쿼리합니다.

1. 강력한 GPU(~NVIDIA 4090)가 있는 머신에서 [README](https://github.com/Physical-Intelligence/openpi)의 지침에 따라 `openpi` 저장소를 복제하고 설치합니다.
2. 다음 명령을 통해 OpenPI 서버를 시작합니다:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

아래의 동등한 명령을 실행할 수도 있습니다:

```bash
uv run scripts/serve_policy.py --env=DROID
```

### 2단계: DROID 로봇 실행하기

1. DROID 제어 노트북과 NUC 모두에 최신 버전의 DROID 패키지가 설치되어 있는지 확인합니다.
2. 제어 노트북에서 DROID conda 환경을 활성화합니다.
3. openpi 저장소를 복제하고 정책 서버에 연결하는 데 사용할 openpi 클라이언트를 설치합니다 (의존성이 거의 없어 매우 빠르게 설치됨): DROID conda 환경이 활성화된 상태에서 `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`를 실행합니다.
4. 명령줄 구문 분석에 사용할 `tyro`를 설치합니다: `pip install tyro`.
5. 이 디렉토리의 `main.py` 파일을 `$DROID_ROOT/scripts` 디렉토리로 복사합니다.
6. `main.py` 파일의 카메라 ID를 사용자의 카메라 ID로 바꿉니다 (명령줄에서 `ZED_Explorer`를 실행하여 카메라 ID를 찾을 수 있으며, 연결된 모든 카메라와 ID를 보여주는 도구가 열립니다. 이를 사용하여 로봇이 상호 작용하려는 장면을 카메라가 잘 볼 수 있는지 확인할 수도 있습니다).
7. `main.py` 파일을 실행합니다. IP와 호스트 주소가 정책 서버를 가리키도록 하십시오. (서버 머신이 DROID 노트북에서 접근 가능한지 확인하려면 DROID 노트북에서 `ping <server_ip>`를 실행할 수 있습니다.) 또한 정책에 사용할 외부 카메라를 지정해야 합니다 (외부 카메라는 하나만 입력). ["left", "right"] 중에서 선택하십시오.

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

스크립트는 로봇이 따를 자유 형식의 언어 지침을 입력하라는 메시지를 표시합니다. 로봇이 상호 작용하려는 장면에 카메라를 향하게 하십시오. 카메라 각도, 객체 위치 등을 신중하게 제어할 필요는 _없습니다_. 저희 경험상 정책은 상당히 견고합니다. 즐거운 프롬프트 되세요!

## 문제 해결

| 문제 | 해결책 |
|---|---|
| 정책 서버에 연결할 수 없음 | 서버가 실행 중이고 IP와 포트가 올바른지 확인하십시오. DROID 노트북에서 `ping <server_ip>`를 실행하여 서버 머신에 연결할 수 있는지 확인할 수 있습니다. |
| 카메라를 찾을 수 없음 | 카메라 ID가 올바른지, 카메라가 DROID 노트북에 연결되어 있는지 확인하십시오. 때때로 카메라를 다시 연결하면 도움이 될 수 있습니다. 명령줄에서 `ZED_Explore`를 실행하여 연결된 모든 카메라를 확인할 수 있습니다. |
| 정책 추론이 느리거나 일관되지 않음 | DROID 노트북에 유선 인터넷 연결을 사용하여 지연 시간을 줄여보십시오 (청크당 0.5 - 1초 지연은 정상입니다). |
| 정책이 작업을 잘 수행하지 못함 | 저희 실험에서 정책은 다양한 환경, 카메라 위치 및 조명 조건에서 간단한 테이블 위 조작 작업(집고 놓기)을 수행할 수 있었습니다. 정책이 작업을 잘 수행하지 못하면 장면이나 객체 배치를 수정하여 작업을 더 쉽게 만들 수 있습니다. 또한 정책에 전달하는 카메라 뷰가 장면의 모든 관련 객체를 볼 수 있는지 확인하십시오 (정책은 단일 외부 카메라 + 손목 카메라에만 의존하므로 원하는 카메라를 정책에 공급하고 있는지 확인하십시오). `ZED_Explore`를 사용하여 정책에 전달하는 카메라 뷰가 장면의 모든 관련 객체를 볼 수 있는지 확인하십시오. 마지막으로, 정책은 완벽과는 거리가 멀고 더 복잡한 조작 작업에서는 실패할 것이지만, 보통은 괜찮은 노력을 합니다. :) |


## 다른 정책 실행하기

[RoboArena](https://robo-arena.github.io/) 논문의 기준 DROID 정책을 실행하기 위한 설정을 제공합니다. 아래 명령을 실행하여 각 정책에 대한 추론 서버를 시작하십시오. 그런 다음 위의 지침에 따라 DROID 로봇에서 평가를 실행하십시오.

```
# pi0-FAST에서 훈련, FAST 토크나이저 사용
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# pi0에서 훈련, 플로우 매칭 사용
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# PaliGemma에서 훈련, RT-2 / OpenVLA 스타일 비닝 토크나이저 사용.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid

# PaliGemma에서 훈련, FAST 토크나이저 사용 (범용 FAST+ 토크나이저 사용).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid

# PaliGemma에서 훈련, FAST 토크나이저 사용 (DROID 데이터셋에서 훈련된 토크나이저).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid

# PaliGemma에서 훈련, FSQ 토크나이저 사용.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid

# PaliGemma에서 DROID로 훈련된 pi0 스타일 확산 / 플로우 VLA.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

추론 설정은 [roboarena_config.py](../../src/openpi/training/misc/roboarena_config.py)에서 찾을 수 있습니다.

# UR5 Example

Below we provide an outline of how to implement the key components mentioned in the "Finetune on your data" section of the [README](../README.md) for finetuning on UR5 datasets.

First, we will define the `UR5Inputs` and `UR5Outputs` classes, which map the UR5 environment to the model and vice versa. Check the corresponding files in `src/openpi/policies/libero_policy.py` for comments explaining each line.

```python

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}

```

Next, we will define the `UR5DataConfig` class, which defines how to process raw UR5 data from LeRobot dataset for training. For a full example, see the `LeRobotLiberoDataConfig` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

```python

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

```

Finally, we define the TrainConfig for our UR5 dataset. Here, we define a config for fine-tuning pi0 on our UR5 dataset. See the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) for more examples, e.g. for pi0-FAST or for LoRA fine-tuning.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        # This config lets us reload the UR5 normalization stats from the base model checkpoint.
        # Reloading normalization stats can help transfer pre-trained models to new environments.
        # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. The recommended setting is True.
            prompt_from_task=True,
        ),
    ),
    # Load the pi0 base model checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```

---

# UR5 예제 (한국어)

아래에서는 UR5 데이터셋에 대한 미세 조정을 위해 [README](../README.md)의 "자체 데이터로 미세 조정하기" 섹션에서 언급된 주요 구성 요소를 구현하는 방법에 대한 개요를 제공합니다.

먼저, UR5 환경을 모델에 매핑하고 그 반대로 매핑하는 `UR5Inputs` 및 `UR5Outputs` 클래스를 정의합니다. 각 줄을 설명하는 주석은 `src/openpi/policies/libero_policy.py`의 해당 파일을 확인하십시오.

```python

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 먼저, 관절과 그리퍼를 상태 벡터로 연결합니다.
        state = np.concatenate([data["joints"], data["gripper"]])

        # LeRobot이 자동으로 float32(C,H,W)로 저장하므로 이미지를 uint8(H,W,C)로 파싱해야 할 수 있습니다.
        # 정책 추론 시에는 이 과정이 생략됩니다.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # 입력 딕셔너리를 생성합니다.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # 오른쪽 손목이 없으므로 0으로 채웁니다.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 오른쪽 손목을 위한 "슬롯"이 사용되지 않으므로 이 마스크는
                # False로 설정됩니다.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 프롬프트(즉, 언어 지시)를 모델에 전달합니다.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # 로봇은 7개의 행동 차원(6 DoF + 그리퍼)을 가지므로 처음 7개 차원을 반환합니다.
        return {"actions": np.asarray(data["actions"][:, :7])}

```

다음으로, 훈련을 위해 LeRobot 데이터셋의 원시 UR5 데이터를 처리하는 방법을 정의하는 `UR5DataConfig` 클래스를 정의합니다. 전체 예제는 [훈련 설정 파일](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)의 `LeRobotLiberoDataConfig` 설정을 참조하십시오.

```python

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # LeRobot 데이터셋의 키를 리매핑하기 위한 상용구입니다. 여기서는 이름 변경이 필요 없다고 가정합니다.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 이 변환들은 이전에 작성한 것들입니다.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # 절대 행동을 델타 행동으로 변환합니다.
        # 관례적으로, 그리퍼 행동(7번째 차원)은 변환하지 않습니다.
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # 모델 변환은 프롬프트와 행동 목표를 토큰화하는 것과 같은 작업을 포함합니다.
        # 자신의 데이터셋을 위해 여기에서 아무것도 변경할 필요가 없습니다.
        model_transforms = ModelTransformFactory()(model_config)

        # 훈련과 추론을 위해 모든 데이터 변환을 반환합니다. 여기에서 아무것도 변경할 필요가 없습니다.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

```

마지막으로, UR5 데이터셋에 대한 TrainConfig를 정의합니다. 여기서는 UR5 데이터셋에서 pi0을 미세 조정하기 위한 설정을 정의합니다. pi0-FAST 또는 LoRA 미세 조정과 같은 더 많은 예제는 [훈련 설정 파일](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)을 참조하십시오.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        # 이 설정은 기본 모델 체크포인트에서 UR5 정규화 통계를 다시 로드할 수 있게 해줍니다.
        # 정규화 통계를 다시 로드하면 사전 훈련된 모델을 새로운 환경으로 이전하는 데 도움이 될 수 있습니다.
        # 자세한 내용은 [norm_stats.md](../docs/norm_stats.md) 파일을 참조하십시오.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            # 이 플래그는 LeRobot 데이터셋의 `task` 필드에서 프롬프트(즉, 작업 지시)를
            # 로드할지 여부를 결정합니다. 권장 설정은 True입니다.
            prompt_from_task=True,
        ),
    ),
    # pi0 기본 모델 체크포인트를 로드합니다.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```






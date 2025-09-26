# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains three types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## Updates

- [Sept 2025] We released PyTorch support in openpi.
- [Sept 2025] We released pi05, an upgraded version of pi0 with better open-world generalization.
- [Sept 2025]: We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025]: We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID. 


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may way to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |

---

# openpi (한국어)

openpi는 [Physical Intelligence 팀](https://www.physicalintelligence.company/)이 공개하는 로보틱스를 위한 오픈소스 모델과 패키지를 포함하고 있습니다.

현재 이 저장소는 세 가지 유형의 모델을 포함합니다:
- [π₀ 모델](https://www.physicalintelligence.company/blog/pi0): 플로우 기반의 비전-언어-행동 모델(VLA)입니다.
- [π₀-FAST 모델](https://www.physicalintelligence.company/research/fast): FAST 행동 토크나이저를 기반으로 한 자기회귀 VLA입니다.
- [π₀.₅ 모델](https://www.physicalintelligence.company/blog/pi05): [지식 단열(knowledge insulation)](https://www.physicalintelligence.company/research/knowledge_insulation)로 훈련되어 오픈월드 일반화 성능이 향상된 π₀의 업그레이드 버전입니다. 참고로, 현재 이 저장소에서는 $\pi_{0.5}$ 훈련과 추론 모두에 대해 플로우 매칭 헤드만 지원합니다.

모든 모델에 대해 1만 시간 이상의 로봇 데이터로 사전 훈련된 _기본 모델_ 체크포인트를 제공하며, 이를 바로 사용하거나 자체 데이터셋에 맞게 미세 조정(fine-tuning)할 수 있는 예제를 제공합니다.

이것은 실험입니다: $\pi_0$는 [ALOHA](https://tonyzhaozh.github.io/aloha/)나 [DROID](https://droid-dataset.github.io/)와 같이 널리 사용되는 플랫폼과는 다른 저희 자체 로봇을 위해 개발되었습니다. 연구자와 실무자들이 $\pi_0$를 자신의 플랫폼에 맞게 창의적인 새로운 실험을 할 수 있을 것으로 기대하지만, 모든 시도가 성공적일 것이라고는 예상하지 않습니다. 즉, $\pi_0$가 여러분에게 효과가 있을 수도 있고 없을 수도 있지만, 시도해 보시는 것을 환영합니다!

## 업데이트

- [2025년 9월] openpi에 PyTorch 지원을 추가했습니다.
- [2025년 9월] 오픈월드 일반화 성능이 향상된 π₀의 업그레이드 버전인 pi05를 출시했습니다.
- [2025년 9월]: DROID 훈련을 위한 [개선된 유휴 필터](examples/droid/README_train.md#data-filtering)를 추가했습니다.
- [2025년 6월]: 전체 [DROID 데이터셋](https://droid-dataset.github.io/)에서 `openpi`를 사용하여 VLA를 훈련하는 [지침](examples/droid/README_train.md)을 추가했습니다. 이는 pi0-FAST-DROID 훈련에 사용된 훈련 파이프라인의 대략적인 오픈소스 구현입니다.

## 요구 사항

이 저장소의 모델을 실행하려면 최소한 다음 사양을 갖춘 NVIDIA GPU가 필요합니다. 이 추정치는 단일 GPU를 가정하지만, 훈련 설정에서 `fsdp_devices`를 구성하여 모델 병렬 처리로 여러 GPU를 사용하여 GPU당 메모리 요구 사항을 줄일 수도 있습니다. 현재 훈련 스크립트는 아직 다중 노드 훈련을 지원하지 않습니다.

| 모드 | 필요한 메모리 | 예시 GPU |
| --- | --- | --- |
| 추론 | > 8 GB | RTX 4090 |
| 미세 조정 (LoRA) | > 22.5 GB | RTX 4090 |
| 미세 조정 (전체) | > 70 GB | A100 (80GB) / H100 |

이 저장소는 Ubuntu 22.04에서 테스트되었으며, 현재 다른 운영 체제는 지원하지 않습니다.

## 설치

이 저장소를 복제할 때, 서브모듈을 업데이트해야 합니다:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 또는 이미 저장소를 복제한 경우:
git submodule update --init --recursive
```

Python 의존성 관리를 위해 [uv](https://docs.astral.sh/uv/)를 사용합니다. 설정하려면 [uv 설치 지침](https://docs.astral.sh/uv/getting-started/installation/)을 참조하세요. uv가 설치되면 다음을 실행하여 환경을 설정합니다:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

참고: `GIT_LFS_SKIP_SMUDGE=1`은 LeRobot을 의존성으로 가져오기 위해 필요합니다.

**도커**: uv 설치의 대안으로, 도커를 사용하여 openpi를 설치하는 지침을 제공합니다. 시스템 설정에 문제가 발생하면 도커를 사용하여 설치를 단순화하는 것을 고려해 보세요. 자세한 내용은 [도커 설정](docs/docker.md)을 참조하세요.

## 모델 체크포인트

### 기본 모델
여러 기본 VLA 모델 체크포인트를 제공합니다. 이 체크포인트들은 1만 시간 이상의 로봇 데이터로 사전 훈련되었으며 미세 조정에 사용할 수 있습니다.

| 모델 | 사용 사례 | 설명 | 체크포인트 경로 |
| --- | --- | --- | --- |
| $\pi_0$ | 미세 조정 | 미세 조정을 위한 기본 [π₀ 모델](https://www.physicalintelligence.company/blog/pi0) | `gs://openpi-assets/checkpoints/pi0_base` |
| $\pi_0$-FAST | 미세 조정 | 미세 조정을 위한 기본 자기회귀 [π₀-FAST 모델](https://www.physicalintelligence.company/research/fast) | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$ | 미세 조정 | 미세 조정을 위한 기본 [π₀.₅ 모델](https://www.physicalintelligence.company/blog/pi05) | `gs://openpi-assets/checkpoints/pi05_base` |

### 미세 조정된 모델
다양한 로봇 플랫폼과 작업을 위한 "전문가" 체크포인트도 제공합니다. 이 모델들은 위의 기본 모델에서 미세 조정되었으며 대상 로봇에서 직접 실행되도록 만들어졌습니다. 특정 로봇에서는 작동할 수도 있고 그렇지 않을 수도 있습니다. 이 체크포인트들은 ALOHA나 DROID Franka 설정과 같이 더 널리 사용 가능한 로봇으로 수집된 비교적 작은 데이터셋에서 미세 조정되었기 때문에, 특정 설정에 일반화되지 않을 수 있습니다. 하지만 실제로는 DROID 체크포인트와 같은 일부 모델이 상당히 광범위하게 일반화되는 것을 발견했습니다.

| 모델 | 사용 사례 | 설명 | 체크포인트 경로 |
| --- | --- | --- | --- |
| $\pi_0$-FAST-DROID | 추론 | [DROID 데이터셋](https://droid-dataset.github.io/)에서 미세 조정된 $\pi_0$-FAST 모델: DROID 로봇 플랫폼의 새로운 장면에서 다양한 간단한 테이블 위 조작 작업을 0-shot으로 수행 가능 | `gs://openpi-assets/checkpoints/pi0_fast_droid` |
| $\pi_0$-DROID | 미세 조정 | [DROID 데이터셋](https://droid-dataset.github.io/)에서 미세 조정된 $\pi_0$ 모델: $\pi_0$-FAST-DROID보다 추론 속도가 빠르지만 언어 명령을 잘 따르지 못할 수 있음 | `gs://openpi-assets/checkpoints/pi0_droid` |
| $\pi_0$-ALOHA-towel | 추론 | 내부 [ALOHA](https://tonyzhaozh.github.io/aloha/) 데이터에서 미세 조정된 $\pi_0$ 모델: ALOHA 로봇 플랫폼에서 다양한 수건을 0-shot으로 접을 수 있음 | `gs://openpi-assets/checkpoints/pi0_aloha_towel` |
| $\pi_0$-ALOHA-tupperware | 추론 | 내부 [ALOHA](https://tonyzhaozh.github.io/aloha/) 데이터에서 미세 조정된 $\pi_0$ 모델: 반찬통에서 음식을 꺼낼 수 있음 | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap | 추론 | 공개 [ALOHA](https://dit-policy.github.io/) 데이터에서 미세 조정된 $\pi_0$ 모델: 펜 뚜껑을 열 수 있음 | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap` |
| $\pi_{0.5}$-LIBERO | 추론 | [LIBERO](https://libero-project.github.io/datasets) 벤치마크를 위해 미세 조정된 $\pi_{0.5}$ 모델: 최첨단 성능 달성 ([LIBERO README](examples/libero/README.md) 참조) | `gs://openpi-assets/checkpoints/pi05_libero` |
| $\pi_{0.5}$-DROID | 추론 / 미세 조정 | [지식 단열](https://www.physicalintelligence.company/research/knowledge_insulation)을 사용하여 [DROID 데이터셋](https://droid-dataset.github.io/)에서 미세 조정된 $\pi_{0.5}$ 모델: 빠른 추론과 우수한 언어 추종 능력 | `gs://openpi-assets/checkpoints/pi05_droid` |

기본적으로 체크포인트는 `gs://openpi-assets`에서 자동으로 다운로드되며 필요할 때 `~/.cache/openpi`에 캐시됩니다. `OPENPI_DATA_HOME` 환경 변수를 설정하여 다운로드 경로를 덮어쓸 수 있습니다.

## 사전 훈련된 모델로 추론 실행하기

사전 훈련된 모델 체크포인트는 몇 줄의 코드로 실행할 수 있습니다 (여기서는 $\pi_0$-FAST-DROID 모델 사용):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 훈련된 정책 생성
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 더미 예제에서 추론 실행
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
[예제 노트북](examples/inference.ipynb)에서도 이를 테스트해 볼 수 있습니다.

[DROID](examples/droid/README.md) 및 [ALOHA](examples/aloha_real/README.md) 로봇에서 사전 훈련된 체크포인트의 추론을 실행하기 위한 자세한 단계별 예제를 제공합니다.

**원격 추론**: 모델을 **원격으로** 추론하는 [예제와 코드](docs/remote_inference.md)를 제공합니다. 모델은 다른 서버에서 실행되고 웹소켓 연결을 통해 로봇에게 행동을 스트리밍할 수 있습니다. 이를 통해 로봇 외부의 더 강력한 GPU를 쉽게 사용하고 로봇과 정책 환경을 분리할 수 있습니다.

**로봇 없이 추론 테스트**: 로봇 없이 추론을 테스트하기 위한 [스크립트](examples/simple_client/README.md)를 제공합니다. 이 스크립트는 임의의 관측을 생성하고 모델로 추론을 실행합니다. 자세한 내용은 [여기](examples/simple_client/README.md)를 참조하세요.

## 자체 데이터로 기본 모델 미세 조정하기

자체 데이터로 기본 모델을 미세 조정하는 방법의 실행 예제로 [LIBERO 데이터셋](https://libero-project.github.io/datasets)에서 $\pi_{0.5}$ 모델을 미세 조정할 것입니다. 세 단계를 설명합니다:
1. 데이터를 LeRobot 데이터셋으로 변환 (훈련에 사용)
2. 훈련 설정 정의 및 훈련 실행
3. 정책 서버 가동 및 추론 실행

### 1. 데이터를 LeRobot 데이터셋으로 변환

LIBERO 데이터를 LeRobot 데이터셋으로 변환하는 최소한의 예제 스크립트를 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py)에 제공합니다. 자체 데이터를 변환하기 위해 쉽게 수정할 수 있습니다! 원시 LIBERO 데이터셋은 [여기](https://huggingface.co/datasets/openvla/modified_libero_rlds)에서 다운로드하고 다음으로 스크립트를 실행할 수 있습니다:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**참고:** LIBERO에서만 미세 조정하려면 이 단계를 건너뛸 수 있습니다. 왜냐하면 우리의 LIBERO 미세 조정 설정은 사전 변환된 LIBERO 데이터셋을 가리키기 때문입니다. 이 단계는 자체 데이터에 적용할 수 있는 예제일 뿐입니다.

### 2. 훈련 설정 정의 및 훈련 실행

자체 데이터로 기본 모델을 미세 조정하려면 데이터 처리 및 훈련을 위한 설정을 정의해야 합니다. 아래에 LIBERO에 대한 자세한 주석이 달린 예제 설정을 제공하며, 이를 자체 데이터셋에 맞게 수정할 수 있습니다:

- [`LiberoInputs` 및 `LiberoOutputs`](src/openpi/policies/libero_policy.py): LIBERO 환경에서 모델로의 데이터 매핑 및 그 반대를 정의합니다. 훈련과 추론 모두에 사용됩니다.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): LeRobot 데이터셋의 원시 LIBERO 데이터를 훈련용으로 처리하는 방법을 정의합니다.
- [`TrainConfig`](src/openpi/training/config.py): 미세 조정 하이퍼파라미터, 데이터 설정 및 가중치 로더를 정의합니다.

LIBERO 데이터에 대한 [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), [π₀.₅](src/openpi/training/config.py)의 예제 미세 조정 설정을 제공합니다.

훈련을 실행하기 전에 훈련 데이터에 대한 정규화 통계를 계산해야 합니다. 훈련 설정 이름으로 아래 스크립트를 실행하세요:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

이제 다음 명령으로 훈련을 시작할 수 있습니다 (`--overwrite` 플래그는 동일한 설정으로 미세 조정을 다시 실행할 경우 기존 체크포인트를 덮어쓰는 데 사용됩니다):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

이 명령은 훈련 진행 상황을 콘솔에 기록하고 체크포인트를 `checkpoints` 디렉토리에 저장합니다. Weights & Biases 대시보드에서 훈련 진행 상황을 모니터링할 수도 있습니다. GPU 메모리를 최대한 활용하려면 훈련을 실행하기 전에 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`를 설정하세요. 이렇게 하면 JAX가 GPU 메모리의 최대 90%를 사용할 수 있습니다 (기본값 75% 대비).

**참고:** 사전 훈련에서 상태/행동 정규화를 위한 정규화 통계를 *다시 로드*하는 기능을 제공합니다. 이는 사전 훈련 혼합의 일부였던 로봇에서 새로운 작업으로 미세 조정할 때 유용할 수 있습니다. 정규화 통계를 다시 로드하는 방법에 대한 자세한 내용은 [norm_stats.md](docs/norm_stats.md) 파일을 참조하세요.

### 3. 정책 서버 가동 및 추론 실행

훈련이 완료되면 정책 서버를 가동한 다음 LIBERO 평가 스크립트에서 쿼리하여 추론을 실행할 수 있습니다. 모델 서버를 시작하는 것은 쉽습니다 (이 예제에서는 반복 20,000의 체크포인트를 사용하며 필요에 따라 수정):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

이렇게 하면 8000번 포트에서 수신 대기하고 관측치가 전송되기를 기다리는 서버가 가동됩니다. 그런 다음 서버를 쿼리하는 평가 스크립트(또는 로봇 런타임)를 실행할 수 있습니다.

특히 LIBERO 평가를 실행하기 위해 정책 서버와 평가 스크립트를 함께 처리하는 도커화된 워크플로우를 제공하고 권장합니다. 자세한 내용은 [LIBERO README](examples/libero/README.md)를 참조하세요.

자체 로봇 런타임에 정책 서버 호출을 포함시키려면 [원격 추론 문서](docs/remote_inference.md)에 그 방법에 대한 최소한의 예제가 있습니다.

### 더 많은 예제

ALOHA 플랫폼에서 모델을 미세 조정하고 추론을 실행하는 방법에 대한 더 많은 예제를 다음 README에서 제공합니다:
- [ALOHA 시뮬레이터](examples/aloha_sim)
- [ALOHA 실제](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 지원

openpi는 이제 원래 JAX 버전과 함께 π₀ 및 π₀.₅ 모델의 PyTorch 구현을 제공합니다! PyTorch 구현은 LIBERO 벤치마크(추론 및 미세 조정 모두)에서 검증되었습니다. 현재 몇 가지 기능은 지원되지 않습니다 (향후 변경될 수 있음):

- π₀-FAST 모델
- 혼합 정밀도 훈련
- FSDP (완전 샤딩 데이터 병렬) 훈련
- LoRA (저순위 적응) 훈련
- 훈련 중 EMA (지수 이동 평균) 가중치

### 설정
1. 모든 의존성의 최신 버전이 설치되었는지 확인하세요: `uv sync`

2. transformers 4.53.2가 설치되었는지 다시 확인하세요: `uv pip show transformers`

3. transformers 라이브러리 패치를 적용하세요:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

이것은 transformers 라이브러리의 여러 파일을 필요한 모델 변경 사항으로 덮어씁니다: 1) AdaRMS 지원, 2) 활성화 정밀도 올바르게 제어, 3) 업데이트되지 않고 KV 캐시 사용 허용.

**경고**: 기본 uv 링크 모드(하드링크)를 사용하면 uv 캐시의 transformers 라이브러리에 영구적으로 영향을 미칩니다. 즉, 변경 사항은 transformers 재설치 후에도 유지되며 transformers를 사용하는 다른 프로젝트에 전파될 수도 있습니다. 이 작업을 완전히 되돌리려면 `uv cache clean transformers`를 실행해야 합니다.

### JAX 모델을 PyTorch로 변환하기

JAX 모델 체크포인트를 PyTorch 형식으로 변환하려면:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### PyTorch로 추론 실행하기

PyTorch 구현은 JAX 버전과 동일한 API를 사용합니다. 변환된 PyTorch 모델을 가리키도록 체크포인트 경로만 변경하면 됩니다:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 훈련된 정책 생성 (PyTorch 형식 자동 감지)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 추론 실행 (JAX와 동일한 API)
action_chunk = policy.infer(example)["actions"]
```

### PyTorch를 사용한 정책 서버

정책 서버는 PyTorch 모델과 동일하게 작동합니다. 변환된 체크포인트 디렉토리를 가리키기만 하면 됩니다:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### PyTorch로 미세 조정하기

PyTorch에서 모델을 미세 조정하려면:

1. JAX 기본 모델을 PyTorch 형식으로 변환합니다:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 설정에서 `pytorch_weight_path`를 사용하여 변환된 PyTorch 모델 경로를 지정합니다.

3. 다음 모드 중 하나를 사용하여 훈련을 시작합니다:

```bash
# 단일 GPU 훈련:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 예제:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 최신 체크포인트에서 재개

# 다중 GPU 훈련 (단일 노드):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 예제:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 다중 노드 훈련:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 정밀도 설정

JAX와 PyTorch 구현은 정밀도를 다음과 같이 처리합니다:

**JAX:**
1. 추론: 대부분의 가중치와 계산은 bfloat16, 안정성을 위해 일부 계산은 float32
2. 훈련: 기본적으로 혼합 정밀도: 가중치와 그래디언트는 float32, (대부분의) 활성화와 계산은 bfloat16. 설정에서 `dtype`을 float32로 설정하여 전체 float32 훈련으로 변경할 수 있습니다.

**PyTorch:**
1. 추론: JAX와 일치 -- 대부분의 가중치와 계산은 bfloat16, 안정성을 위해 일부 가중치는 float32로 변환
2. 훈련: 전체 bfloat16(기본값) 또는 전체 float32 지원. 설정에서 `pytorch_training_precision`을 설정하여 변경할 수 있습니다. bfloat16은 메모리를 덜 사용하지만 float32에 비해 손실이 더 높습니다. 혼합 정밀도는 아직 지원되지 않습니다.

torch.compile을 사용하면 추론 속도는 JAX와 PyTorch 간에 비슷합니다.

## 문제 해결

일반적인 문제와 해결책을 여기에 수집할 것입니다. 문제가 발생하면 먼저 여기를 확인하세요. 해결책을 찾을 수 없으면 저장소에 이슈를 제기해 주세요(지침은 [여기](CONTRIBUTING.md) 참조).

| 문제 | 해결책 |
| --- | --- |
| `uv sync`가 의존성 충돌로 실패 | 가상 환경 디렉토리를 제거(`rm -rf .venv`)하고 `uv sync`를 다시 실행해 보세요. 문제가 계속되면 최신 버전의 `uv`가 설치되었는지 확인하세요(`uv self update`). |
| 훈련 중 GPU 메모리 부족 | 훈련을 실행하기 전에 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (또는 그 이상)을 설정하여 JAX가 더 많은 GPU 메모리를 사용하도록 하세요. 또한 `--fsdp-devices <n>` (여기서 `<n>`은 GPU 수)를 사용하여 [완전 샤딩 데이터 병렬 처리](https://engineering.fb.com/2021/07/15/open-source/fsdp/)를 활성화할 수 있습니다. 이는 느린 훈련 속도를 대가로 메모리 사용량을 줄입니다 (감속 정도는 특정 설정에 따라 다름). 여전히 메모리가 부족하면 EMA를 비활성화하는 것을 고려할 수 있습니다. |
| 정책 서버 연결 오류 | 서버가 실행 중이고 예상 포트에서 수신 대기 중인지 확인하세요. 클라이언트와 서버 간의 네트워크 연결 및 방화벽 설정을 확인하세요. |
| 훈련 중 norm stats 오류 누락 | 훈련을 시작하기 전에 설정 이름으로 `scripts/compute_norm_stats.py`를 실행하세요. |
| 데이터셋 다운로드 실패 | 인터넷 연결을 확인하세요. HuggingFace 데이터셋의 경우 로그인했는지 확인하세요(`huggingface-cli login`). |
| CUDA/GPU 오류 | NVIDIA 드라이버가 올바르게 설치되었는지 확인하세요. 도커의 경우 nvidia-container-toolkit이 설치되었는지 확인하세요. GPU 호환성을 확인하세요. 시스템 수준에 CUDA 라이브러리를 설치할 필요는 없습니다 --- uv를 통해 설치됩니다. CUDA 문제가 발생하면 시스템 CUDA 라이브러리를 *제거*해 보는 것도 좋습니다. 시스템 라이브러리가 때때로 충돌을 일으킬 수 있기 때문입니다. |
| 예제 실행 시 가져오기 오류 | `uv sync`로 모든 의존성을 설치했는지 확인하세요. 일부 예제는 README에 추가 요구 사항이 나열되어 있을 수 있습니다. |
| 행동 차원 불일치 | 데이터 처리 변환이 로봇의 예상 입력/출력 차원과 일치하는지 확인하세요. 정책 클래스의 행동 공간 정의를 확인하세요. |
| 훈련 손실 발산 | 데이터셋의 `norm_stats.json`에서 `q01`, `q99`, `std` 값을 확인하세요. 거의 사용되지 않는 특정 차원은 매우 작은 `q01`, `q99` 또는 `std` 값을 가질 수 있으며, 이로 인해 정규화 후 상태와 행동이 거대해질 수 있습니다. 해결 방법으로 norm stats를 수동으로 조정할 수 있습니다. |

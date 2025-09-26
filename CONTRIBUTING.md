# Contributing to openpi

We welcome contributions, improvements, and modifications. Everyone is welcome to use openpi in accordance to the [license](LICENSE). Contributors are also welcome to submit bug reports, feature requests, and pull requests. We can't promise to approve every pull request, and we are a small team with limited bandwidth to review all requests, but we'll give it our best effort. Specifics are described below.

## Issues and feature requests

You are welcome to use the Github [discussion](https://github.com/Physical-Intelligence/openpi/discussions) feature if you would like to discuss something that is not directly reporting an issue or making a feature request. This is suitable for questions about how to use some aspect of openpi, or other topics.

If you found a bug or other issue, please first check that the issue was not already reported (use the search bar on Github under Issues). If the issue has not yet been reported, please include this information when filing a Github issue:

- Your OS type and version and the version of Python you are using
- Code that allows us to reproduce your bug, including all dependencies
- Traceback of any exception
- Any other information that would help us, such as a screenshot

In order for us to address any issue, we must be able to reproduce it, so if you encountered the issue after making modifications to openpi, please reproduce the issue without any other modifications and provide a code snippet that allows us to quickly reproduce the problem on `main`.

If you would like to submit a feature request, please check that the feature request does not already exist, and please provide the following information:

- The motivation for the feature
- A description of the problem you are trying to solve or your use case
- Enough information for us to understand the nature of the request
- Some information for how you intend to use it (this might help us in understanding the motivation!)

We can't promise to support every feature request, but it is helpful to us to know the use cases that you are interested in!

## Submitting a pull request

If you implemented support for a new robot or environment, or some other new feature, we welcome pull requests (PRs) to openpi. We encourage you to create a [feature request](https://github.com/Physical-Intelligence/openpi/issues) or make a post on the [discussion](https://github.com/Physical-Intelligence/openpi/discussions) board before starting to work on your PR, if you would like to get a sense for whether we are likely to approve your PR if it is submitted. Since we are a small team with limited ability to provide maintenance and support, we may not accept all PRs (e.g., if we believe it would make the code harder to maintain, or if reviewing the PR is out of scope for us), so contacting us in advance is a good way to get a sense for whether your PR is likely to get approved for merging into openpi directly. But even if it isn't, you are of course more than welcome to maintain your own fork with whatever modifications you would like. When creating PRs, we recommend every contribution to consider the following:

- Make sure that your PR has a clear title and description
- Run `pre-commit` (install using `pre-commit install` first), and run `ruff check .` and `ruff format .`
- Make sure your PR passes all tests

---

# openpi에 기여하기 (한국어)

우리는 기여, 개선 및 수정을 환영합니다. 모든 분들은 [라이선스](LICENSE)에 따라 openpi를 사용할 수 있습니다. 기여자분들은 버그 리포트, 기능 요청 및 풀 리퀘스트를 제출하는 것을 환영합니다. 모든 풀 리퀘스트를 승인한다고 약속할 수는 없으며, 모든 요청을 검토할 수 있는 대역폭이 제한된 소규모 팀이지만, 최선을 다할 것입니다. 구체적인 내용은 아래에 설명되어 있습니다.

## 이슈 및 기능 요청

이슈를 직접 보고하거나 기능 요청을 하는 것이 아닌 다른 것을 논의하고 싶다면 Github [토론](https://github.com/Physical-Intelligence/openpi/discussions) 기능을 자유롭게 사용하십시오. 이는 openpi의 일부 측면 사용 방법에 대한 질문이나 기타 주제에 적합합니다.

버그나 기타 문제를 발견한 경우, 먼저 해당 이슈가 이미 보고되지 않았는지 확인하십시오 (Github 이슈 아래의 검색 창 사용). 아직 보고되지 않은 이슈인 경우, Github 이슈를 제출할 때 다음 정보를 포함해 주십시오:

- 운영 체제 유형 및 버전과 사용 중인 Python 버전
- 모든 종속성을 포함하여 버그를 재현할 수 있는 코드
- 예외의 트레이스백
- 스크린샷과 같이 도움이 될 수 있는 기타 정보

문제를 해결하려면 문제를 재현할 수 있어야 하므로, openpi를 수정한 후 문제가 발생한 경우 다른 수정 없이 문제를 재현하고 `main`에서 문제를 신속하게 재현할 수 있는 코드 조각을 제공해 주십시오.

기능 요청을 제출하려면 기능 요청이 이미 존재하지 않는지 확인하고 다음 정보를 제공해 주십시오:

- 기능에 대한 동기
- 해결하려는 문제 또는 사용 사례에 대한 설명
- 요청의 성격을 이해할 수 있는 충분한 정보
- 사용하려는 방법에 대한 정보 (동기를 이해하는 데 도움이 될 수 있습니다!)

모든 기능 요청을 지원한다고 약속할 수는 없지만, 여러분이 관심 있는 사용 사례를 아는 것은 우리에게 도움이 됩니다!

## 풀 리퀘스트 제출

새로운 로봇이나 환경, 또는 다른 새로운 기능에 대한 지원을 구현한 경우, openpi에 대한 풀 리퀘스트(PR)를 환영합니다. PR 작업 시작 전에 [기능 요청](https://github.com/Physical-Intelligence/openpi/issues)을 생성하거나 [토론](https://github.com/Physical-Intelligence/openpi/discussions) 게시판에 게시하여 제출 시 PR이 승인될 가능성이 있는지에 대한 감을 잡는 것을 권장합니다. 우리는 유지 보수 및 지원을 제공할 수 있는 능력이 제한된 소규모 팀이므로, 모든 PR을 수락하지 않을 수 있습니다 (예: 코드를 유지하기 어렵게 만들거나 PR 검토가 우리 범위를 벗어나는 경우). 따라서 사전에 문의하는 것이 PR이 openpi에 직접 병합될 가능성이 있는지 확인하는 좋은 방법입니다. 하지만 그렇지 않더라도, 원하는 수정을 가한 자신만의 포크를 유지하는 것을 물론 환영합니다. PR을 생성할 때 모든 기여는 다음을 고려하는 것이 좋습니다:

- PR에 명확한 제목과 설명이 있는지 확인하십시오.
- `pre-commit`을 실행하고 (`pre-commit install`을 사용하여 먼저 설치), `ruff check .` 및 `ruff format .`을 실행하십시오.
- PR이 모든 테스트를 통과하는지 확인하십시오.

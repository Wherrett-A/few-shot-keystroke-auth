# Few-Shot Keystroke Authentication using Adaptive Metric Learning

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-WIP-yellow.svg)

## Overview
This repository contains the implementation and research for "Using an Adaptive Metric Learning Approach for Few-Shot, Free-Text Continuous Authentication with a Physical Keyboard" - a dissertation project exploring novel approaches to behavioural biometrics.

### Research Questions
- Can metric learning achieve <3% EER with <50 enrolment samples?
- Will an adaptive fine-tuning mechanism mitigate behavioural drift without increasing FAR?
- How does the proposed system compare to a baseline KD model?

### Presentations
[WIP](https://wherrett-a.github.io/few-shot-keystroke-auth/wip-presentation/)

### AI Usage Declaration
The code behind this project is developed with assistance from locally hosted LLMs. (zai-org/glm-4.7-flash & qwen/qwen3-coder-30b) to accelerate development and improve the quality of the code.

#### AI Tools
- Provider: LM Studio (locally hosted)
- Primary Model: zai-org/glm-4.7-flash *agentic coding, tool usage*
- Secondary Model: qwen/qwen3-coder-30b *complex reasoning, long-context tasks*

#### AI-Assisted Development Areas
- Code Generation
- Debugging
- Documentation

#### AI Interaction Tracking
All key prompts are logged in [docs/PROMPT_JOURNAL.md](docs/PROMPT_JOURNAL.md), large prompts can also be found in the [docs/prompts/](docs/prompts/) directory. These large text prompts are stored in the format `yymmdd-hhmm.md`. There will also be a log from each session from the LM Studio Dev server, this will be found in [docs/ai-logs/](docs/ai-logs/)

## License
This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
---
**Note:** This is a work-in-progress research project. Code and results are under active development. For the latest updates and progress, please refer to the Issues and Projects sections of this repository.

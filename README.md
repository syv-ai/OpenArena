# OpenArena: Competitive Dataset Generation

![Llamas fighting in an arena](assets/fight.jpeg)

## Overview

OpenArena is a Python project designed to create high-quality datasets by pitting Language Models (LLMs) against each other in a competitive environment. This tool uses an ELO-based rating system to rank different LLMs based on their performance on various prompts, judged by another LLM acting as an impartial evaluator.

## Features

- Asynchronous battles between multiple LLMs
- ELO rating system for model performance tracking
- Detailed evaluation and scoring by a judge model
- Generation of training data based on model performances
- Support for Hugging Face datasets
- YAML configuration for easy setup and customization
- Support for custom endpoints and API keys

## Requirements

- Python 3.7+
- Ollama (for local model execution)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/syv-ai/OpenArena.git
   cd OpenArena
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have access to the Ollama API endpoint (default: `http://localhost:11434/v1/chat/completions`) or other custom endpoints as specified in your configuration.

## Usage

1. Create or modify the `arena_config.yaml` file to set up your desired models, datasets, and endpoints:
   ```yaml
   default_endpoint:
     url: "http://localhost:11434/v1/chat/completions"
     # api_key: "your_default_api_key"  # Uncomment and set if needed

   judge_model:
     name: "JudgeModel"
     model_id: "llama3"
     # endpoint:
     #   url: "http://custom-judge-endpoint.com/api/chat"
     #   api_key: "judge_model_api_key"

   models:
     - name: "Open hermes"
       model_id: "openhermes"
       # endpoint:
       #   url: "http://custom-phi3-endpoint.com/api/chat"
       #   api_key: "phi3_api_key"
     - name: "Mistral v0.3"
       model_id: "mistral"
     - name: "Phi 3 medium"
       model_id: "phi3:14b"

   datasets:
     - name: "skunkworksAI/reasoning-0.01"
       description: "Reasoning dataset"
       split: "train"
       field: "instruction"
       limit: 10
   ```

2. Run the script:
   ```
   python llm_arena.py
   ```

3. The script will output:
   - Intermediate ELO rankings after each prompt
   - ELO rating progression for each model
   - Generated training data
   - Final ELO ratings
   - Total execution time

## How it Works

1. **Configuration Loading**: The script loads the configuration from the `arena_config.yaml` file.
2. **Dataset Loading**: Prompts are loaded from the specified Hugging Face dataset(s).
3. **Response Generation**: Each model generates responses for all given prompts.
4. **Evaluation**: The judge model evaluates pairs of responses for each prompt, providing scores and explanations.
5. **ELO Updates**: ELO ratings are updated based on the scores from each battle.
6. **Training Data Generation**: The results are compiled into a structured format for potential use in training or fine-tuning other models.

## Customization

- Modify the `arena_config.yaml` file to add or remove models, change the judge model, adjust dataset parameters, or specify custom endpoints and API keys.
- Adjust the ELO K-factor in the `update_elo_ratings()` method to change the volatility of the ratings.

## Future work

- [ ] Automatically download Ollama models
- [X] Added YAML configuration
- [X] OpenAI support - supports any OpenAI compatible endpoint (also vLLM)
- [X] Hugging Face datasets integration
- [ ] Auto upload to Hugging Face
- [ ] Database to keep track of which prompts have been done if it should fail

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
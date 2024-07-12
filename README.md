# OpenArena: Competitive Dataset Generation

![Llamas fighting in an arena](assets/fight.jpeg)

## Overview

LLM Arena is a Python project designed to create high-quality datasets by pitting Language Models (LLMs) against each other in a competitive environment. This tool uses an ELO-based rating system to rank different LLMs based on their performance on various prompts, judged by another LLM acting as an impartial evaluator.

## Features

- Asynchronous battles between multiple LLMs
- ELO rating system for model performance tracking
- Detailed evaluation and scoring by a judge model
- Generation of training data based on model performances
- Support for custom prompts and multiple LLMs
- YAML configuration for easy setup and customization
- Support for custom endpoints and API keys

## Requirements

- Python 3.7+
- aiohttp
- asyncio
- PyYAML
- typing
- re
- time

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llm-arena.git
   cd llm-arena
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have access to the Ollama API endpoint (default: `http://localhost:11434/api/chat`) or other custom endpoints as specified in your configuration.

## Usage

1. Create or modify the `arena_config.yaml` file to set up your desired models, prompts, and endpoints:

   ```yaml
   default_endpoint: 
     url: "http://localhost:11434/api/chat"
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
        #   url: "http://custom-model-endpoint.com/api/chat"
        #   api_key: "judge_model_api_key"
     - name: "Mistral v0.3"
       model_id: "mistral"
     - name: "Phi 3 medium"
       model_id: "phi3:14b"
     - name: ChatGPT3.5
       model_id: "gpt-3.5-turbo"
       endpoint:
          url: "https://api.openai.com/v1/chat/completions"
          api_key: ""

   prompts:
     - "Explain the concept of quantum entanglement."
     - "What are the main causes and effects of climate change?"
     - "Explain the importance of artificial intelligence in modern healthcare."
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
2. **Response Generation**: Each model generates responses for all given prompts.
3. **Evaluation**: The judge model evaluates pairs of responses for each prompt, providing scores and explanations.
4. **ELO Updates**: ELO ratings are updated based on the scores from each battle.
5. **Training Data Generation**: The results are compiled into a structured format for potential use in training or fine-tuning other models.

## Customization

- Modify the `arena_config.yaml` file to add or remove models, change the judge model, adjust prompts, or specify custom endpoints and API keys.
- Adjust the ELO K-factor in the `update_elo_ratings()` method to change the volatility of the ratings.

## Future work

- [ ] Automatically download ollama models
- [X] Added yaml configuration
- [X] OpenAI support
- [ ] Hugging face datasets
- [ ] Auto upload to hugging face
- [ ] Database to keep track of which prompts have been done if it should fail

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

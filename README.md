# OpenArena: Competitive Dataset Generation

![Llamas fighting in an arena](assets/fight.jpeg)

## Overview

LLM Arena is a Python project designed to create high-quality datasets by pitting Language Models (LLMs) against each other in a competitive environment. This tool uses an ELO-based rating system to rank different LLMs based on their performance on various prompts, judged by another LLM acting as an impartial evaluator.

## Future work
- [ ] Automatically download ollama models
- [ ] Better configuration (yaml?)
- [ ] Hugging face datasets
- [ ] Auto upload to hugging face
- [ ] Database to keep track of which prompts have been done if it should fail  

## Features

- Asynchronous battles between multiple LLMs
- ELO rating system for model performance tracking
- Detailed evaluation and scoring by a judge model
- Generation of training data based on model performances
- Support for custom prompts and multiple LLMs

## Requirements

- Python 3.7+
- aiohttp
- asyncio
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

3. Ensure you have access to the Ollama API endpoint (default: `http://localhost:11434/api/chat`).

## Usage

1. Modify the `main()` function in the script to set up your desired models and prompts:

   ```python
   models = [
       Model("Open hermes", "openhermes"),
       Model("Mistral v0.3", "mistral"),
       Model("Phi 3 medium", "phi3:14b"),
   ]
   judge_model = JudgeModel("JudgeModel", "llama3")

   prompts = [
       "Explain the concept of quantum entanglement.",
       "What are the main causes and effects of climate change?",
       "Explain the importance of artificial intelligence in modern healthcare."
   ]
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

1. **Response Generation**: Each model generates responses for all given prompts.
2. **Evaluation**: The judge model evaluates pairs of responses for each prompt, providing scores and explanations.
3. **ELO Updates**: ELO ratings are updated based on the scores from each battle.
4. **Training Data Generation**: The results are compiled into a structured format for potential use in training or fine-tuning other models.

## Customization

- Add or remove models by modifying the `models` list in the `main()` function.
- Change the judge model by modifying the `judge_model` initialization.
- Adjust the prompts list to focus on specific topics or tasks.
- Modify the ELO K-factor in the `update_elo_ratings()` method to change the volatility of the ratings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

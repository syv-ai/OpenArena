import asyncio
import aiohttp
from typing import List, Dict, Tuple
import re
import json
import time

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

class Model:
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self.elo = 1000  # Initial ELO score
        self.responses = {}  # Store responses for each prompt

    async def generate_response(self, session: aiohttp.ClientSession, prompt: str) -> str:
        if prompt not in self.responses:
            print(f"Generating response for {self.name} to prompt: {prompt}")
            async with session.post(
                url=OLLAMA_ENDPOINT,
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
            ) as response:
                result = await response.json()
                self.responses[prompt] = result["message"]["content"]
        return self.responses[prompt]

class JudgeModel:
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    async def evaluate(self, session: aiohttp.ClientSession, prompt: str, response1: str, response2: str) -> Tuple[int, int, str]:
        evaluation_prompt = f"""You are an impartial judge evaluating the quality of responses from two AI models. Your task is to analyze and compare their responses objectively, focusing on various factors such as coherence, factual accuracy, context-awareness, and overall quality.

Original Prompt: {prompt}

Model A's Response: {response1}

Model B's Response: {response2}

Please evaluate these responses and provide:
1. A detailed explanation of your scoring, focusing on the strengths and weaknesses of each response
2. A score for Model A's response (1-10)
3. A score for Model B's response (1-10)

Example output:
Explanation: Model A's response demonstrated a deep understanding of the topic with clear explanations and relevant examples. However, it lacked proper structuring and coherence in some sections. Model B's response was well-organized and concise, but it missed some key details and failed to provide in-depth analysis.
Score-A: 8
Score-B: 6

Format your response as (IT'S VERY IMPORTANT TO FOLLOW THIS FORMAT): 
Explanation: [your detailed explanation]
Score-A: [score]
Score-B: [score]
        """

        async with session.post(
            url=OLLAMA_ENDPOINT,
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": evaluation_prompt}
                ],
                "stream": False
            }
        ) as response:
            result = await response.json()
            evaluation = result["message"]["content"]
        
        parts = evaluation.split("Score-A:")
        if len(parts) != 2:
            raise ValueError("Unable to parse evaluation response")
        
        explanation = parts[0].replace("Explanation:", "").strip()
        scores_part = "Score-A:" + parts[1]
        
        score_pattern = r'Score-([AB]):\s*(\d+)'
        scores = re.findall(score_pattern, scores_part)
        
        if len(scores) != 2:
            raise ValueError("Unable to parse scores from the evaluation response")

        score_dict = dict(scores)
        score1 = int(score_dict.get('A', 0))
        score2 = int(score_dict.get('B', 0))

        if not explanation:
            explanation = "No detailed explanation provided."

        return score1, score2, explanation

class ArenaLearning:
    def __init__(self, models: List[Model], judge_model: JudgeModel):
        self.models = models
        self.judge_model = judge_model
        self.battle_results = []
        self.elo_history = {model.name: [model.elo] for model in models}

    async def generate_all_responses(self, session: aiohttp.ClientSession, prompts: List[str]) -> None:
        tasks = []
        for model in self.models:
            for prompt in prompts:
                tasks.append(model.generate_response(session, prompt))
        await asyncio.gather(*tasks)

    async def run_battles_for_prompt(self, session: aiohttp.ClientSession, prompt: str) -> None:
        tasks = []
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                print(f"Running battle for prompt: {prompt}, {self.models[i].name} vs {self.models[j].name}")
                model1, model2 = self.models[i], self.models[j]
                tasks.append(self.battle(session, prompt, model1, model2))
        await asyncio.gather(*tasks)

    async def battle(self, session: aiohttp.ClientSession, prompt: str, model1: Model, model2: Model) -> None:
        response1 = model1.responses[prompt]
        response2 = model2.responses[prompt]
        score1, score2, explanation = await self.judge_model.evaluate(session, prompt, response1, response2)
        self.battle_results.append((model1, model2, score1, score2, response1, response2, explanation, prompt))

    def update_elo_ratings(self) -> None:
        K = 32  # K-factor for ELO calculation

        for model1, model2, score1, score2, _, _, _, _ in self.battle_results:
            expected_score1 = 1 / (1 + 10 ** ((model2.elo - model1.elo) / 400))
            expected_score2 = 1 - expected_score1

            actual_score1 = score1 / (score1 + score2)
            actual_score2 = 1 - actual_score1

            model1.elo += K * (actual_score1 - expected_score1)
            model2.elo += K * (actual_score2 - expected_score2)

        # Update ELO history
        for model in self.models:
            self.elo_history[model.name].append(model.elo)

    def generate_training_data(self, prompts: List[str]) -> List[Dict]:
        training_data = []
        prompts = set(result[-1] for result in self.battle_results)  # Get unique prompts

        for prompt in prompts:
            prompt_results = [result for result in self.battle_results if result[-1] == prompt]
            model_scores = {model.name: {'scores': [], 'response': ''} for model in self.models}

            for model1, model2, score1, score2, response1, response2, _, _ in prompt_results:
                model_scores[model1.name]['scores'].append(score1)
                model_scores[model1.name]['response'] = response1
                model_scores[model2.name]['scores'].append(score2)
                model_scores[model2.name]['response'] = response2

            # Calculate average scores and prepare ranked list
            ranked_models = []
            for model_name, data in model_scores.items():
                if data['scores']:
                    avg_score = sum(data['scores']) / len(data['scores'])
                    ranked_models.append({
                        'model_name': model_name,
                        'avg_score': avg_score,
                        'response': data['response']
                    })

            # Sort models by average score in descending order
            ranked_models.sort(key=lambda x: x['avg_score'], reverse=True)

            # Add to training data
            training_data.append({
                'prompt': prompt,
                'ranked_models': ranked_models
            })
        return training_data

    async def run_arena(self, session: aiohttp.ClientSession, prompts: List[str]) -> None:
        print("Generating responses for all models...")
        await self.generate_all_responses(session, prompts)

        print("Running battles...")
        for prompt in prompts:
            await self.run_battles_for_prompt(session, prompt)
            self.update_elo_ratings()
            print(f"\nIntermediate ELO rankings after prompt: '{prompt}'")
            for model in self.models:
                print(f"{model.name}: {model.elo:.2f}")

        return self.generate_training_data(prompts)

async def main():
    models = [
        Model("Open hermes", "openhermes"),
        Model("Mistral v0.3", "mistral"),
        Model("Phi 3 medium", "phi3:14b"),
    ]
    judge_model = JudgeModel("JudgeModel", "llama3")

    arena = ArenaLearning(models, judge_model)

    prompts = [
        "Explain the concept of quantum entanglement.",
        "What are the main causes and effects of climate change?",
        "Explain the importance of artificial intelligence in modern healthcare."
    ]

    async with aiohttp.ClientSession() as session:
        training_data = await arena.run_arena(session, prompts)

    print("\nELO rating progression:")
    for model in models:
        print(f"{model.name}: {arena.elo_history[model.name]}")

    # save training data to a file
    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=4)

    print("\nTraining data:")
    for item in training_data:
        print(f"Prompt: {item['prompt']}")
        for rank, model_data in enumerate(item['ranked_models']):
            print(f"  Rank {rank+1}: {model_data['model_name']}")
            print(f"    Average Score: {model_data['avg_score']:.2f}")
            print(f"    Response: {model_data['response'][:50]}...")  # Truncate response for brevity
        print()

    print("\nFinal ELO ratings:")
    for model in models:
        print(f"{model.name}: {model.elo:.2f}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(main())
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")


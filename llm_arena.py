import asyncio
import aiohttp
from typing import List, Dict, Tuple
import re
import json
import time
import yaml
from datasets import load_dataset

class Endpoint:
    def __init__(self, url: str, api_key: str = None):
        self.url = url
        self.api_key = api_key

    def get_headers(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

class Model:
    def __init__(self, name: str, model_id: str, endpoint: Endpoint):
        self.name = name
        self.model_id = model_id
        self.endpoint = endpoint
        self.elo = 1000  # Initial ELO score
        self.responses = {}  # Store responses for each prompt

    async def generate_response(self, session: aiohttp.ClientSession, prompt: str) -> str:
        if prompt not in self.responses:
            print(f"Generating response for {self.name} to prompt: {prompt[:30]}...")
            async with session.post(
                url=self.endpoint.url,
                headers=self.endpoint.get_headers(),
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
            ) as response:
                result = await response.json()
                print(result)
                self.responses[prompt] = result['choices'][0]['message']['content']
        return self.responses[prompt]

class JudgeModel:
    def __init__(self, name: str, model_id: str, endpoint: Endpoint):
        self.name = name
        self.model_id = model_id
        self.endpoint = endpoint

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
            url=self.endpoint.url,
            headers=self.endpoint.get_headers(),
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": evaluation_prompt}
                ],
                "stream": False
            }
        ) as response:
            result = await response.json()
            evaluation = result['choices'][0]['message']['content']
        
        parts = evaluation.split("Score-A:")
        if len(parts) != 2:
            raise ValueError("Unable to parse evaluation response")
        
        explanation = parts[0].replace("Explanation:", "").strip()
        scores_part = "Score-A:" + parts[1]
        
        score_pattern = r'Score-([AB]):\s*(\d+)'
        scores = re.findall(score_pattern, scores_part)
        
        if len(scores) != 2:
            print("Warning: Unable to extract scores from evaluation response - trying again")
            self.evaluate(session, prompt, response1, response2)

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

    async def generate_batch_responses(self, session: aiohttp.ClientSession, prompt_batch: List[str]) -> None:
        tasks = []
        for model in self.models:
            for prompt in prompt_batch:
                tasks.append(model.generate_response(session, prompt))
        await asyncio.gather(*tasks)

    async def run_battles_for_batch(self, session: aiohttp.ClientSession, prompt_batch: List[str]) -> None:
        tasks = []
        for prompt in prompt_batch:
            for i in range(len(self.models)):
                for j in range(i + 1, len(self.models)):
                    print(f"Running battle for prompt: {prompt[:50]}..., {self.models[i].name} vs {self.models[j].name}")
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

    async def run_arena(self, session: aiohttp.ClientSession, prompts: List[str], batch_size: int = 3) -> List[Dict]:
        print("Running arena with batched processing...")
        for i in range(0, len(prompts), batch_size):
            prompt_batch = prompts[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} (prompts {i+1}-{min(i+batch_size, len(prompts))})")
            
            print("Generating responses for current batch...")
            await self.generate_batch_responses(session, prompt_batch)

            print("Running battles for current batch...")
            await self.run_battles_for_batch(session, prompt_batch)
            
            self.update_elo_ratings()
            print("\nIntermediate ELO rankings after current batch:")
            for model in self.models:
                print(f"{model.name}: {model.elo:.2f}")

        return self.generate_training_data(prompts)

async def main():
    # Load configuration from YAML file
    with open("arena_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    default_endpoint = Endpoint(
        url=config["default_endpoint"]["url"],
        api_key=config["default_endpoint"].get("api_key")
    )

    # Create models from configuration
    models = []
    for model_config in config["models"]:
        if "endpoint" in model_config:
            endpoint = Endpoint(
                url=model_config["endpoint"]["url"],
                api_key=model_config["endpoint"].get("api_key")
            )
        else:
            endpoint = default_endpoint
        models.append(Model(model_config["name"], model_config["model_id"], endpoint))

    # Create judge model from configuration
    judge_config = config["judge_model"]
    if "endpoint" in judge_config:
        judge_endpoint = Endpoint(
            url=judge_config["endpoint"]["url"],
            api_key=judge_config["endpoint"].get("api_key")
        )
    else:
        judge_endpoint = default_endpoint
    judge_model = JudgeModel(judge_config["name"], judge_config["model_id"], judge_endpoint)

    # Load dataset from Hugging Face
    prompts = []
    for dataset_config in config["datasets"]:
        ds = load_dataset(dataset_config["name"])
        split = ds[dataset_config["split"]]
        prompts.extend(split[dataset_config["field"]][:dataset_config["limit"]])

    print(f"Total number of prompts: {len(prompts)}")

    arena = ArenaLearning(models, judge_model)

    batch_size = 3  # Set the batch size to 3
    async with aiohttp.ClientSession() as session:
        training_data = await arena.run_arena(session, prompts, batch_size)

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
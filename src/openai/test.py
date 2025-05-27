import os
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("OPEN_AI_ENDPOINT"),
    api_key=os.getenv("API_OPEN_AI_KEY"),
)
model_name = "gpt-4o"
deployment = "gpt-4o"

log_path = os.path.abspath("logs/usage.md")
os.makedirs(os.path.dirname(log_path), exist_ok=True)


def log_structure(prompt, response, usage, cost):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## Date: {datetime.now()}\n")
        f.write(f"## Prompt: {prompt}\n")
        f.write(f"###Response: {response}\n")
        f.write(f"###Prompt tokens {usage.prompt_tokens}\n")
        f.write(f"###Response tokens {usage.completion_tokens}\n\n")
        f.write(f"###Total cost {cost}\n")


prompts = [
    "Stwórz ranking najlepszych obrońców w lidze NBA",
    "Stwórz ranking najlepszych napastników w LaLidze (w całej historii)",
    "Kto jest lepszy, Messi czy Ronaldo?",
]


def run_prompts(prompts):
    results = []
    for prompt in prompts:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=768,
            temperature=1.0,
            top_p=1.0,
            model=deployment,
        )
        usage = response.usage
        content = response.choices[0].message.content
        prompt_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = prompt_tokens + output_tokens
        cost = (prompt_tokens * 0.0025 / 1000) + (output_tokens * 0.01 / 1000)

        efficiency = total_tokens / cost if cost > 0 else 0
        log_structure(prompt, content, usage, cost)

        results.append(
            {
                "prompt": prompt,
                "cost": cost,
                "efficiency": efficiency,
                "total_tokens": total_tokens,
                "output_tokens": output_tokens,
            }
        )
        best_overall = max(results, key=lambda x: x["efficiency"])

    print(
        f"Best overall token efficiency: '{best_overall['prompt']}' | {best_overall['efficiency']:.2f} total tokens per $"
    )


run_prompts(prompts)

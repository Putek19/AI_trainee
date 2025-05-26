import os.path
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("OPEN_AI_ENDPOINT"),
    api_key=os.getenv("API_OPEN_AI_KEY"),
)
MODEL_NAME = "gpt-4o"
DEPLOYMENT = "gpt-4o"

LOG_PATH = os.path.abspath("backlog/sprint1.md")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def generate_user_stories(project_topic: str) -> str:
    """
    Generate user stories for a software project using OpenAI's GPT model.
    Args:
        project_topic (str): The topic of the software project for which to generate user stories.
    Returns:
        str: A string containing the generated user stories.
    """
    prompt = (
        f"Write 3 user stories for a software project about {project_topic}."
        "Each user story should follow the format: 'As a [type of user], I want [some goal] so that [some reason]'."
        "Add 5 acceptance criteria for each user story in the format: 'Acceptance Criteria: [criteria]'."
        "Make sure the user stories are clear, concise, and relevant to the project topic."
        "Return only user stories and acceptance criteria, without any additional text or explanations."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates user stories.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def save_user_stories_to_file(user_stories: str, file_path: str) -> None:
    """
    Save the generated user stories to a file.
    Args:
        user_stories (str): The user stories to save.
        file_path (str): The path to the file where the user stories will be saved.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(user_stories)
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")


def main():
    """
    Main function to generate user stories for a software project based on user input.
    """
    print("Welcome to the User Stories Generator!")
    print("Type the topic of your software project to generate user stories.")
    project_topic = input(
        "Enter the project topic for which to generate user stories: "
    )
    user_stories = generate_user_stories(project_topic)
    print("Generated User Stories:\n", user_stories)
    save_user_stories_to_file(user_stories, LOG_PATH)
    print(f"User stories saved to {LOG_PATH}")


if __name__ == "__main__":
    main()

import json, os

def save_results(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(str(data))


def save_idea_locally(most_relevant_papers=[], review="", style_idea="", idea="", category="", output_text="", _id="", iteration_data=None, chat_history=None, json_file_path="default.json"):
    if most_relevant_papers: 
        most_relevant_papers = most_relevant_papers.to_dict(orient="records")

    

    new_data = {
        "id": _id,
        "idea": idea,
        "style_idea": style_idea, 
        "most_relevant_papers": most_relevant_papers,
        "category": category,
        "review": review,
        "output_text": output_text,
        "iteration_data": iteration_data, 
        "chat_history": chat_history, 
        "environment": {
            "model": os.getenv("NOVELTY_CHECK_MODEL"),
            "temperature": os.getenv("NOVELTY_CHECK_TEMPERATURE"),
        },
    }

    try:
        with open(json_file_path, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []

    existing_data.append(new_data)

    with open(json_file_path, "w") as file:
        json.dump(existing_data, file, indent=4)

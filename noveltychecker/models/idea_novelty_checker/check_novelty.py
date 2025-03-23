import re, os
import asyncio

from noveltychecker.models.idea_novelty_checker.prompts import prompt_NoveltyChecker_allowsIncrementalNovelty, prompt_NoveltyChecker_allowsIncrementalNovelty_lessRelaxed
from noveltychecker.utils.model_client import get_llm_client


def clean_text(text: str) -> str:
    return text.strip(" \n#:][]")

def parse_output(text: str, delim_class: str, delim_review: str):
    category_part = text.split(delim_review)[0]
    category = category_part.split(delim_class)[-1].strip()
    review = text.split(delim_review)[1].strip()
    
    for punc in ["*", ":", "\n"]:
        category = re.sub(f"\\{punc}", "", category)
        review = re.sub(f"\\{punc}", "", review)
        
    return clean_text(category).lower(), clean_text(review)


def get_prompt_and_parsing_rules(idea, most_relevant_papers, incontext_example):

    delim_class = "Class:"
    delim_review = "Review:"
    if os.getenv("NOVELTY_CHECK_PROMPT")=="relaxed":
        prompt = prompt_NoveltyChecker_allowsIncrementalNovelty(idea, most_relevant_papers, incontext_example)
    elif os.getenv("NOVELTY_CHECK_PROMPT")=="less-relaxed":
        prompt = prompt_NoveltyChecker_allowsIncrementalNovelty_lessRelaxed(idea, most_relevant_papers, incontext_example)
    else:
        raise ValueError("Invalid novelty checker prompt style.")
    return prompt, delim_class, delim_review


async def get_review(idea, most_relevant_papers, incontext_example_path=None):

    model = os.getenv("NOVELTY_CHECK_MODEL") 
    temperature = float(os.getenv("NOVELTY_CHECK_TEMPERATURE", 0))
    
    temperature = 0 
    if incontext_example_path is not None: 
        incontext_example = open(incontext_example_path, "r")

    if os.getenv("NOVELTY_CHECK_EXAMPLES")=="less-relaxed":
        incontext_example = open("noveltychecker/models/idea_novelty_checker/incontext_examples/less-relaxed.json", "r")

    elif os.getenv("NOVELTY_CHECK_EXAMPLES")=="relaxed":
        incontext_example = open("noveltychecker/models/idea_novelty_checker/incontext_examples/relaxed.json", "r")
    else:
        raise ValueError("Invalid incontext example file path.")
    
    prompt, delim_class, delim_review = get_prompt_and_parsing_rules(idea, most_relevant_papers, incontext_example)
    
    client = get_llm_client(model)
    
    if model in ["o3-mini", 'o1']:
        output_text, _ = await client.a_chat(
            model=model,
            messages=prompt,
        )
    else:
        output_text, _ = await client.a_chat(
            model=model,
            messages=prompt,
            temperature=temperature
        )
        
    category, review = parse_output(output_text, delim_class, delim_review)
    if category is None or review is None:
        return None, None, output_text
    category = category.strip("-").strip(" ")
    review = review.strip("-").strip(" ")
    return category, review, output_text

if __name__=="__main__": 

    text = """Class: Not Novel

Review: The idea is not novel because it closely replicates existing work with minimal new contributions. The proposed HyHTM model is identical to the one described in [2], which also uses hyperbolic geometry to model hierarchies in topic models. Both approaches claim to address limitations of traditional HTMs by producing more coherent and specific topic hierarchies, while being computationally efficient. The idea does not introduce any new concepts or approaches beyond what is already presented in [2]. Therefore, based on the provided related papers, this idea lacks novelty and appears to be a direct replication of existing work."""
    delim_class = "Class:"
    delim_review = "Review:"
    output = parse_output(text, delim_class, delim_review)
    print(output)

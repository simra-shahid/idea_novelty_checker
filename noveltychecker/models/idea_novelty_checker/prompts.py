def prompt_NoveltyChecker_allowsIncrementalNovelty(
    idea, similar_documents, incontext_part
):
    relevant_papers = []
    for i, (_, row) in enumerate(similar_documents.iterrows()):
        relevant_papers.append(
            {
                "role": "user",
                "content": f"<PAPER> Paper ID [{i}]: Title: {row['title']}. Abstract: {row['abstract']} </PAPER>",
            }
        )

    prompt = [
        {
            "role": "system",
            "content": "You are ReviewerGPT, an intelligent assistant that helps researchers evaluate the novelty of their ideas.",
        },
        {
            "role": "user",
            "content": f"""You are given some papers similar to the proposed idea (<IDEA> and </IDEA>). Your task is to evaluate the idea's novelty using the related papers (<PAPER> and </PAPER>) only. 

                Types of novelty categories:
                - Not Novel: The idea closely replicates existing work with minimal or no new contributions.
                - Novel: 
                    - The idea introduces new concepts or approaches that are not common in existing literature.
                    - The idea uniquely combines concepts from existing papers, but this combination does not occur in any related papers. 
                    - A new application with same approach is also novel.

                Instructions:
                - Use the example reviews below to write a review for the provided idea by comparing it to the related papers. 
                - Don't assume any prior knowledge about the idea. 
                - When referencing a related paper, then use paper id in the review, mention it in this format: [5]. The paper ID is present between Paper ID [<paper_id>]: Title.
                - After reviewing, classify the idea into one of this category: novel or not novel.
                - Make sure the generated review follows the format in example reviews provided below.  
                - The review should be concise - around 60 to 100 words.
                

                
                {incontext_part}   



                Output Format: 
                - Class: [novel / not novel]
                - Review: The idea is [novel / not novel] different because...
                """,
        },
        {"role": "assistant", "content": "Sure, please provide the IDEA."},
        {"role": "user", "content": f"Here is the idea: <IDEA> {idea} </IDEA>"},
        {"role": "assistant", "content": "Okay, now provide the related papers."},
    ]

    prompt.extend(relevant_papers)

    return prompt


def prompt_NoveltyChecker_allowsIncrementalNovelty_lessRelaxed(
    idea, similar_documents, incontext_part
):
    relevant_papers = []
    for i, (_, row) in enumerate(similar_documents.iterrows()):
        relevant_papers.append(
            {
                "role": "user",
                "content": f"<PAPER> [{i}] Title: {row['title']}. Abstract: {row['abstract']} </PAPER>",
            }
        )

    prompt = [
        {
            "role": "system",
            "content": "You are an AI assistant tasked with evaluating the novelty of a proposed idea by comparing it to related papers. Your goal is to determine whether the idea is novel or not novel based solely on the information provided in the related papers.",
        },
        {
            "role": "user",
            "content": "First, review these example reviews to understand the expected format and style.",
        },
        {
            "role": "user",
            "content": f"Here are the example reviews: \n\n {incontext_part}",
        },
        {
            "role": "user",
            "content": "Now, here is the idea you need to evaluate.",
        },
        {"role": "user", "content": f"Here is the idea <IDEA>: {idea} </IDEA>"},
        {
            "role": "user",
            "content": "And here are the related papers to compare it to: ",
        },
    ]
    prompt.extend(relevant_papers)

    instruction = [
        {
            "role": "user",
            "content": """To complete this task, follow these steps:
                1. Carefully read and analyze the proposed idea (<IDEA> and </IDEA>) and the related papers (<PAPER> and </PAPER>).
                2. Compare the idea to the related papers, looking for similarities and differences.
                3. When referencing a related paper in your review, use its paper ID in square brackets (e.g., [5]).
                4. Determine whether the idea is novel or not novel based on the following criteria:
                   - Not Novel: The idea closely replicates existing work with minimal or no new contributions.
                   - Novel: The idea introduces new concepts or approaches that are not common in existing literature, or it shows some new elements but also heavily relies on existing concepts.
                5. Write a review explaining your assessment, following the format of the example reviews provided.
                6. Classify the idea as either "novel" or "not novel" based on your analysis.
                7. Make sure the review is not longer than 60 words.
                
                
                Remember to base your evaluation solely on the information provided in the related papers and the proposed idea. Do not assume any prior knowledge about the topic or idea beyond what is given.
                
                Your output should follow this format:
                
                Class: [novel / not novel]
                Review: The idea is [novel / not novel] because...
                [Provide a detailed explanation of your assessment, referencing specific papers using their IDs]
                """,
        }
    ]
    prompt.extend(instruction)

    return prompt

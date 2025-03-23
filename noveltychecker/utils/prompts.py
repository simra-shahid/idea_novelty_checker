def prompt_PaperRetrieval_Keywords(idea):

    prompt = [
        {
            "role": "system",
            "content": "You are an intelligent assistant that extracts high-quality keywords and generates specific research paper titles based on the provided IDEA.",
        },
        {
            "role": "user",
            "content": """You are tasked with extracting specific keywords and generating potential research paper titles that closely align with the provided IDEA. These should capture both the novelty and mechanisms of the IDEA, especially where it diverges from existing work. 

            **Keyword Extraction Guidelines**:
            1. Highlight unique methods, technologies, and application areas.
            2. Ensure the keywords specifically capture what sets this idea apart from others.
            3. Generate 3-6 keyword phrases, each consisting of 3-6 words.
            4. Avoid overly general keywords (e.g., "machine learning" or "data science").
            5. Ensure the keywords reflect the precise purpose, mechanisms, and novelty of the idea.

            **Title Generation Guidelines**:
            1. Keep titles concise (max 5 words).
            2. Avoid generic terms or overused phrases.
            3. Reflect the uniqueness and novelty of the idea in each title.
            4. Include a key concept from the IDEA’s mechanism (e.g., "retrieval-augmented generation for idea synthesis").
            5. Ensure the title reflects the application domain.

            **Output Format**:
            <keywords>
            ["specific keyword phrase 1", "specific keyword phrase 2", "specific keyword phrase 3"]
            </keywords>

            <titles>
            ["Title 1", "Title 2", "Title 3", "Title 4"]
            </titles>
            """,
        },
        {"role": "assistant", "content": "Sure, please provide the IDEA."},
        {"role": "user", "content": idea},
    ]
    return prompt


def prompt_RankGPT_IdeaFacets(idea):

    prompt = [
        {
            "role": "system",
            "content": "You are Research Idea Reviewer GPT, an intelligent assistant that helps researchers evaluate the novelty of their ideas.",
        },
        {
            "role": "user",
            "content": """Your task is to extract key facets from a given idea to assist in re-ranking passages based on their relevance to the idea. These key facets should capture the essential elements of the idea, such as the application domain, purpose, mechanisms, methods, and evaluation metrics.

                Instructions:

                1. Carefully read and understand the idea.
                2. Identify and list the key facets of the idea, including but not limited to:
                    - Application Domain: The specific field or area the idea pertains to.
                    - Purpose/Objective: The main goal or intention behind the idea.
                    - Mechanisms/Methods: The techniques or approaches proposed to achieve the purpose.
                    - Evaluation Metrics: The criteria or measures used to assess the effectiveness of the idea.
                
                
                Examples:

                    Idea 1: Develop a system that uses a faceted representation of authors to understand food-health relationships by analyzing the sentiment of research papers and publications. The system will identify key authors in food and health research, map their sentiments towards various topics, and use this information to reveal hidden connections and trends. An experimental results showcase will evaluate the system’s ability to uncover novel food-health relationships and its impact on interdisciplinary research.

                    Key Facets to Look for in Passages:
                        - Application Domain: Food and health research.
                        - Purpose: To understand food-health relationships through sentiment analysis.
                        - Mechanism: Using a faceted representation of authors to map sentiments toward various topics.
                        - Method: Analyzing the sentiment of research papers and publications.
                        - Evaluation: Experimental showcase evaluating the system's ability to uncover novel relationships and its interdisciplinary impact.

                    Idea 2: Develop a hierarchical topic model that integrates multi-level capsule networks to balance sparsity and smoothness in topic models. The capsule networks will capture the hierarchical structure of topics while enforcing sparsity at lower levels and smoothness at higher levels. This model will be validated on benchmark datasets such as PASCAL VOC 2007 and 2012, using metrics like log-likelihood and topic coherence to ensure both high reconstruction accuracy and generalization capability.

                    Key Facets to Look for in Passages:
                        - Application Domain: Topic modeling in machine learning.
                        - Purpose: To balance sparsity and smoothness in topic models.
                        - Mechanism: Integrating multi-level capsule networks.
                        - Method: Developing a hierarchical topic model with capsule networks capturing hierarchical structures.
                        - Evaluation: Validation on benchmark datasets (e.g., PASCAL VOC 2007 and 2012) using metrics like log-likelihood and topic coherence.
                
                """,
        },
        {
            "role": "assistant",
            "content": "Sure, please provide the research idea",
        },
        {
            "role": "user",
            "content": f"""Here is the idea: <idea> {idea} </idea>. 
            Please provide Key Facets to Look for in Passages for the provided idea between <facets> </facets> tags. 
            """,
        },
    ]
    return prompt


def prompt_RankGPT_IdeaPriority(query, facets, number_of_passages):

    prompt2 = f"""**QUERY** Idea: {query}. **Key Facets from IDEA to Look for in Passages**: {facets}.

        Your task is to rank these {number_of_passages} passages based on their relevance to the QUERY IDEA and its provided facets using the following criteria, in order of priority:

        1. **Priority 1:** All Passages that closely match **all** key facets of the **QUERY** IDEA.
        2. **Priority 2:** All Passages that match the **application domain** and **purpose** but may differ in mechanism or method.
        3. **Priority 3:** All Passages that share a similar **purpose** or **mechanism** or **evaluation**, even if the application domain differs.
        4. **Priority 4:** All Passages that partially match the application domain or address related topics but lack alignment with the purpose or mechanism.

        **Ranking Guidelines:**
        - Carefully read the **QUERY** IDEA and given key facets (e.g., application domain, purpose, mechanisms, methods, evaluation metrics).
        - Analyze each passage, comparing it to the **QUERY** IDEA based on these key facets. Assign a relative ranking to each passage based on the priority list above.
        - Ensure that passages are ranked in descending order of relevance (priority 1 first).
        - Use passage identifiers (e.g., [1], [2], etc.) in your ranking.
        - Provide your ranking in the following format: [17] > [16] > [18] > [19] > [0].
        - Do not use any external knowledge or information not provided in the passages or the **QUERY** IDEA."""

    return prompt2


def prompt_RankGPT_prefixRanking(query, num):
    return [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages above based on their relevance and similarity to the idea.",
        },
        {
            "role": "user",
            "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]


def prompt_RankGPT_postRanking(query, num):
    """return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain." """
    return f"Idea: {query}. \nRank the {num} passages above based on their relevance and similarity to the idea. The passages should be listed in descending order using identifiers. The most relevant and similar passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def prompt_RankGPT_postRankingPurpose(query, num):
    return f"Idea: {query}. \nRank the {num} passages above based on their relevance and similarity to the 'main purpose' of the idea. The passages should be listed in descending order using identifiers. The most relevant and similar passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def prompt_RankGPT_prefixRankingPriority(query, idea_priority_facets, num):
    return [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages above based on their provided priority and relevance to the query and its facets.",
        },
        {
            "role": "user",
            "content": f"""I will provide you with {num} passages, each indicated by number identifier []. 
                Your task is to rank the passages based on their relevance to the query idea and the provided priority:

                **Query** Idea: {query}

                Key facets to look in passages for ranking:

                {idea_priority_facets}


                Use the following criteria in order of priority for ranking the passaeges:

                1. **Priority 1:** Passages that closely match **all** key facets of the **QUERY** IDEA.
                2. **Priority 2:** Passages that match the **application domain** and **purpose** but may differ in mechanism or method.
                3. **Priority 3:** Passages that share a similar **purpose** or **mechanism** or **evaluation**, even if the application domain differs.
                4. **Priority 4:** Passages that partially match the application domain or address related topics but lack alignment with the purpose or mechanism.

            """,
        },
        {
            "role": "assistant",
            "content": "Can you provide an example idea, facets and how to rank passages?",
        },
        {
            "role": "user",
            "content": """Here is an example: 
            
            Idea: Enhance topic model evaluation by incorporating anomaly detection machine learning techniques. The goal is to improve topic model evaluation by identifying and flagging anomalies within topic distributions that may indicate incoherence or redundancy. This approach provides a more robust evaluation framework that detects subtle inconsistencies that traditional metrics might miss. The effectiveness of this integrated evaluation method would be assessed through a systematic comparison and meta-analysis of different topic models, ensuring comprehensive and reliable evaluation outcomes.

            Key Facets to Look for in Passages:

            - Application Domain: Topic modeling and evaluation.
            - Purpose: Improving topic model evaluation by detecting anomalies indicating incoherence or redundancy.
            - Mechanism: Incorporating anomaly detection machine learning techniques into topic model evaluation.
            - Method: Identifying and flagging anomalies within topic distributions.
            - Evaluation: Systematic comparison and meta-analysis of different topic models to assess effectiveness.
            
            Passages:

            [0] An Enhanced BERTopic Framework and Algorithm for Improving Topic Coherence and Diversity
                Proposes enhancements to BERTopic for better topic coherence and diversity in topic models.
            [1] Evaluation of Unsupervised Anomaly Detection Methods in Sentiment Mining
                Evaluates anomaly detection methods in sentiment analysis, focusing on detecting anomalies in data distributions.
            [2] LDA_RAD: A Spam Review Detection Method Based on Topic Model and Reviewer Anomaly Degree
                Introduces a method combining topic modeling with anomaly detection to identify spam reviews by analyzing anomalies in reviewer behavior.
            [3] Apples to Apples: A Systematic Evaluation of Topic Models
                Provides a systematic evaluation of various topic models, comparing their performance and outcomes.
            [4] Machine Learning Approach for Anomaly-Based Intrusion Detection Systems Using Isolation Forest Model and Support Vector Machine
                Discusses the use of anomaly detection techniques in intrusion detection systems.
            [5] OCTIS: Comparing and Optimizing Topic Models is Simple!
                Introduces a framework for comparing and optimizing topic models to improve evaluation processes.
            [6] Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling
                Presents a tool that enhances topic modeling using large language models for qualitative insights.
            [7] An Exhaustive Review on State-of-the-art Techniques for Anomaly Detection on Attributed Networks
                Reviews various anomaly detection techniques applicable to network data.
            [8] Topic Modeling Revisited: New Evidence on Algorithm Performance and Quality Metrics
                Revisits topic modeling algorithms and evaluates their performance using different quality metrics.
            [9] A Robust Bayesian Probabilistic Matrix Factorization Model for Collaborative Filtering Recommender Systems Based on User Anomaly Rating Behavior Detection
                Discusses anomaly detection in user behavior within recommender systems.
            
            Ranking:

            [2] > [1] > [5] > [3] > [0] > [8] > [6] > [7] > [4] > [9]

            Explanation (Not to be included in the prompt):
            [2] LDA_RAD: A Spam Review Detection Method Based on Topic Model and Reviewer Anomaly Degree
            Priority 1: Directly combines topic modeling with anomaly detection techniques to identify anomalies, aligning closely with the idea's purpose and mechanism.
            [1] Evaluation of Unsupervised Anomaly Detection Methods in Sentiment Mining
            Priority 2: Focuses on anomaly detection methods in data distributions, which is analogous to detecting anomalies in topic distributions.
            [5] OCTIS: Comparing and Optimizing Topic Models is Simple!
            Priority 3: Provides tools for comparing and optimizing topic models, relevant to the evaluation aspect of the idea.
            [3] Apples to Apples: A Systematic Evaluation of Topic Models
            Priority 3: Discusses systematic evaluation of topic models, aligning with the idea's evaluation method.
            [0] An Enhanced BERTopic Framework and Algorithm for Improving Topic Coherence and Diversity
            Priority 4: Aims to improve topic coherence, which relates to identifying incoherence in topic models.
            [8] Topic Modeling Revisited: New Evidence on Algorithm Performance and Quality Metrics
            Priority 4: Evaluates topic modeling algorithms using quality metrics, relevant to the idea's focus on evaluation.
            [6] Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling
            Priority 5: Enhances topic modeling using language models but doesn't focus on anomaly detection or evaluation.
            [7] An Exhaustive Review on State-of-the-art Techniques for Anomaly Detection on Attributed Networks
            Priority 5: Reviews anomaly detection techniques but in a different application domain.
            [4] Machine Learning Approach for Anomaly-Based Intrusion Detection Systems Using Isolation Forest Model and Support Vector Machine
            Priority 5: Discusses anomaly detection in intrusion detection systems, less relevant to topic modeling.
            [9] A Robust Bayesian Probabilistic Matrix Factorization Model for Collaborative Filtering Recommender Systems Based on User Anomaly Rating Behavior Detection
            Priority 5: Focuses on anomaly detection in recommender systems, not directly related to topic modeling.
            """,
        },
        {
            "role": "user",
            "content": """Here is another example: 
            **Idea:** Develop a system that uses sentiment analysis to detect political bias in news articles. The system will analyze language patterns and sentiments to identify biased reporting, and will be validated using a dataset of news articles over the past decade.

            **Key Facets:**

            - **Application Domain**: News articles analysis.
            - **Purpose**: Detecting political bias through sentiment analysis.
            - **Mechanism**: Analyzing language patterns and sentiments.
            - **Method**: Using a dataset of news articles from the past decade.
            - **Evaluation**: Validated through analysis of historical data.

            **Passages:**

            - **[0]** **Detecting Political Bias in News Articles Using Sentiment Analysis**
            - **[1]** **Sentiment Analysis of Social Media Posts for Political Trends**
            - **[2]** **Machine Learning Techniques for Stock Market Prediction**

            **Ranking:**

            [0] > [1] > [2]
            
            """,
        },
        {
            "role": "assistant",
            "content": "Okay, please provide the passages which I have to compare with **Query** Idea",
        },
    ]

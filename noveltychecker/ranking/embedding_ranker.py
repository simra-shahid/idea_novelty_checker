import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from noveltychecker.ranking.embedding import get_embeddings_ideapapers


def compute_similarities_in_batches(embeddings, idea_embedding, batch_size=1000):
    similarity_scores = []
    for i in range(0, len(embeddings), batch_size):
        batch = np.vstack(embeddings[i:i + batch_size])
        sims = cosine_similarity(batch, idea_embedding).flatten()
        similarity_scores.extend(sims)
    return np.array(similarity_scores)


async def embeddingFiltering(idea, paper_embeddings_metadata, max_papers_specter=100):

    keep_columns = [
        "corpusId", "paperId", "url", "title", "abstract", "venue",
        "fieldsOfStudy", "publicationDate", "authors", "embedding", "source"
    ]

    idea_embedding_metadata = await get_embeddings_ideapapers([{"corpusId":"-1", "title": "", "abstract": idea}])  # shape (1, D)
    idea_embedding = np.array(idea_embedding_metadata["-1"]["embedding"]).reshape(1, -1)

    paper_embeddings = []
    corpus_ids = []
    missed = []

    for corpus_id, paper_data in tqdm(paper_embeddings_metadata.items(), total=len(paper_embeddings_metadata),
                                      desc="Checking if all papers contain embeddings...", disable=True):
        try:
            if "embedding" in paper_data:
                if isinstance(paper_data["embedding"], dict): 
                    paper_embeddings.append(np.array(paper_data["embedding"]['vector']).reshape(1, -1))
                else: 
                    paper_embeddings.append(np.array(paper_data["embedding"]).reshape(1, -1))
                corpus_ids.append(corpus_id)
            else:
                missed.append(corpus_id)
        except Exception as e:
            print(f"Error processing {corpus_id}: {e}")


    if len(paper_embeddings) == 0:
        return pd.DataFrame(columns=["corpusId", "similarity_score"] + keep_columns)


    paper_embeddings = np.vstack(paper_embeddings)
    paper_similarities = compute_similarities_in_batches(paper_embeddings, idea_embedding)


    data = {
        "corpusId": corpus_ids,
        "similarity_score": paper_similarities,
    }

    paper_similarities_df = pd.DataFrame(data)
    paper_similarities_df = paper_similarities_df.sort_values("similarity_score", ascending=False)
    most_similar_documents = paper_similarities_df.iloc[:max_papers_specter]

    paper_df = pd.DataFrame(paper_embeddings_metadata.values())
    paper_df['corpusId'] = paper_df['corpusId'].apply(lambda x: str(x))
    most_similar_documents = pd.merge(most_similar_documents, paper_df, on="corpusId")

    return most_similar_documents

import asyncio
from typing import List, Dict, Any
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch 

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  

class Specter2Embedding:
    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model

    async def embed_batch(self, batch_keys: List[str], idea_papers: Dict[str, Any]) -> List[Dict[str, Any]]:
        texts = []
        corpus_ids = []
        for corpus_id in batch_keys:
            paper_data = idea_papers.get(corpus_id, {})
            title = " " if paper_data.get("title", None) is None else paper_data["title"]
            abstract = " " if paper_data.get("abstract", None) is None else paper_data["abstract"]
            text = title + self.tokenizer.sep_token + abstract
            texts.append(text)
            corpus_ids.append(corpus_id)

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self._encode_texts, texts)
        return [{"corpusId": cid, "embedding": {"vector": emb}} for cid, emb in zip(corpus_ids, embeddings)]

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512
        ).to(device)
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings_tensor = output.last_hidden_state[:, 0, :]
        return embeddings_tensor.detach().cpu().numpy().tolist()


async def get_embeddings_ideapapers(idea_papers: List[Dict[str, Any]], reformat=True) -> Dict[str, Any]:
   
    if not idea_papers:
        return {}

    if reformat==True and isinstance(idea_papers, list):
        idea_papers = {
            str(paper["corpusId"]): {k: v for k, v in paper.items()}
            for paper in idea_papers
        }

    paper_keys = list(idea_papers.keys())

    embedding_processor = Specter2Embedding()
    batch_size = 200
    tasks = []
    for i in range(0, len(paper_keys), batch_size):
        batch = paper_keys[i: i + batch_size]
        tasks.append(embedding_processor.embed_batch(batch, idea_papers))

    all_embeddings = []
    batch_results = await asyncio.gather(*tasks)
    for batch in batch_results:
        all_embeddings.extend(batch)


    for entry in all_embeddings:
        corpus_id = str(entry.get("corpusId", ""))
        if corpus_id:
            embedding = entry.get("embedding", {}).get("vector")
            if embedding:
                idea_papers[corpus_id]["embedding"] = embedding
            else:
                pass
        else:
            pass

    return idea_papers



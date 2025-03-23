import aiohttp, asyncio
from tqdm import tqdm
import os


async def make_request_with_retries(url, headers, params=None, input_json=None, request_type="get", url_string='', retries=2, delay=10):
    async with aiohttp.ClientSession() as session:
        for _ in tqdm(range(retries), total=retries, desc=f'Requesting for {url_string}...', disable=True):
            try:
                if request_type == "get":
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json() if response.content else {}
                            if data:
                                return data
                        else:
                            pass
                else:
                    async with session.post(url, headers=headers, params=params, json=input_json) as response:
                        if response.status == 200:
                            data = await response.json() if response.content else {}
                            if data:
                                return data
                        else:
                            pass
                
                await asyncio.sleep(delay)
            except MemoryError as me:
                return []
            except Exception as e:
                continue

    return []

async def get_paper_data(id_, id_type="corpus_id", batch_wise=False):
    attributes = "corpusId,paperId,url,title,year,abstract,authors.name,fieldsOfStudy,citationCount,venue"
    paper_data_query_params = {"fields": attributes}

    if not batch_wise:
        if id_type == "paper_id":
            url = "https://api.semanticscholar.org/graph/v1/paper/" + id_
        else:
            url = f"https://api.semanticscholar.org/graph/v1/paper/CorpusId:{id_}"

        return await make_request_with_retries(
            url, headers={"x-api-key": os.getenv("S2_API_KEY")}, params=paper_data_query_params, url_string='get paper data'
        )
    else: 
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"

        return await make_request_with_retries(
            url, headers={"x-api-key":os.getenv("S2_API_KEY")}, input_json={"ids": id_}, request_type="post", url_string='get paper data', params={"fields": attributes}
        )

async def papers_from_search_api(query="", start_year="", end_year="", search_type="keyword", limit=100):
    attributes = "corpusId,paperId,url,title,year,abstract,authors.name,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,citationCount,venue"
    
    if search_type=="keyword": 
        url = "https://api.semanticscholar.org/graph/v1/paper/search/"
    elif search_type=="snippet": 
        url = "https://api.semanticscholar.org/graph/v1/snippet/search/"

    params = {
        "query": query,
        "fields": attributes,
        "year": f"{start_year}-{end_year}",
        "limit": limit,
    }
    return await make_request_with_retries(
        url, headers={"x-api-key":os.getenv("S2_API_KEY")}, params=params, url_string='search_api'
    )

async def papers_from_recommendation_api_allCs(corpus_id=None, limit=100):
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/CorpusId:{corpus_id}?fields=corpusId,paperId,title,abstract,paperId,url,venue,publicationDate,fieldsOfStudy,authors&limit={limit}&from=all-cs"
    return await make_request_with_retries(url, headers={"x-api-key":os.getenv("S2_API_KEY")}, url_string='recommendations api - allCS')

async def papers_from_recommendation_api_recent(corpus_id=None, limit=100):
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/CorpusId:{corpus_id}?fields=corpusId,paperId,title,abstract,paperId,url,venue,publicationDate,fieldsOfStudy,authors&limit={limit}&from=recent"
    return await make_request_with_retries(url, headers={"x-api-key":os.getenv("S2_API_KEY")}, url_string='recommendations api - recent')

async def getSpecterEmbedding_paperIDs(paperIDs):
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    return await make_request_with_retries(url, headers={"x-api-key":os.getenv("S2_API_KEY")}, params={"fields": "corpusId,embedding"}, input_json={"ids": paperIDs}, request_type="post", url_string='Specter Embeddings') 


"""
async def get_papers_data(seed_paper_ids, file_path=None):

    if not os.path.exists(file_path):
        batch_size = 15        
        total_batches = math.ceil(len(seed_paper_ids) / batch_size)
        seed_papers = {} 
        i = 0 
        while i < len(seed_paper_ids): 

            batch = seed_paper_ids[i : i + batch_size] 
            batch_papers = await get_paper_data(batch, id_type="paper_id", batch_wise=True)
            if not batch_papers: 
                continue 

            if isinstance(batch_papers, list):
                batch_dict = {}
                for paper in batch_papers:
                    # paper might be None or missing "paperId"
                    if paper and isinstance(paper, dict):
                        pid = paper.get("corpusId")
                        if pid:
                            batch_dict[pid] = paper
                seed_papers.update(batch_dict)
            else:
                print(f"[WARNING] Unexpected data type returned: {type(batch_papers)}. Skipping batch.")
            
            print(f"Processed batch starting at index {i}. Total seed papers so far: {len(seed_papers)}")
            i += batch_size
        
        
        json.dump(seed_papers, open(file_path, "w")) 
    else: 
        seed_papers = json.load(open(file_path, "r"))

    return seed_papers
"""
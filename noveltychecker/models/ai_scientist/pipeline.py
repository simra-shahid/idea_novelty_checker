
from noveltychecker.models.ai_scientist.check_novelty import get_review

async def run_aiscientist(
    idea, 
    input_papers=None, 
    use_retrieval=True
):  


    category, chat_history, iteration_metadata = await get_review(
            idea, 
            input_papers=input_papers, 
        use_retrieval = use_retrieval
        
        )

    return category, chat_history, iteration_metadata


        

   
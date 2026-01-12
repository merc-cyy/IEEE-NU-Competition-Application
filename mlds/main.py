from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase_keybert_recommendation import VolunteerRecommender 
from dotenv import load_dotenv
import os
from supabase import create_client, Client

from llm_recommendation import llmRecommender



app = FastAPI()
load_dotenv()


# declare origin/s
origins = [
    "https://nuvolunteers.org/",
    "localhost:3000"
]


# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    
    allow_origins=['*'],  # for the demo, allowed all origins meaning any browser can call our APIs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase init
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) # the DB client to be able to query our DB client


#jobs
jobs = {}



# Define request schema
class ExampleRequest(BaseModel):
    userid: str # { userid : "123" } --> example request
recommender = VolunteerRecommender(supabase) # initalize the recommmender model
@app.post("/recommend")
@app.post("/recommend/") # supports /recommend or /recommend/
def recommend(request: ExampleRequest): # recommend jobs in general based on a user's profile
    try:
        # Load recommender once

        # print("JUST STARTING")
        user_id = request.userid  # get userid from the request
        # print(f"USERID:{user_id}")
        
        # print("INIT IS DONE")
        recommender.fetch_data() # fetch the jobs from the DB into the recommender client dataframe
        # print("Before MODEL FIT")
        recommender.fit() # fit --> build the embeddings
        # print("MODEL FIT IS DONE")
        user_embedding = recommender.build_user_profile(user_id) # pass in the userid to the recommender client to build user embedding
        print("USER RECOMMENDATIONS TAKEN INTO CONSIDERATION")
        recommendations = recommender.recommend_for_user(user_embedding, top_n=1000) # recommend best jobs for user (cosine similarity)
        # output_path = f"/tmp/recommendations_{user_id}.csv"
        # recommendations.to_csv(output_path, index=False)
        # print(f"Recommendations written to: {output_path}")
        # Convert DataFrame to list of dictionaries
        recommendations_list = recommendations.to_dict(orient="records") # serialize into a dict to be sent as JSON

        # print(f"RECOMMENDATIONS:{recommendations_list}")
        return {
            'jobs' : recommendations_list
        }
    except Exception as e:
        import traceback
        print("bad ERROR OCCURRED:")
        traceback.print_exc() 
        return {"error": str(e)}




# Define request schema
class ExampleRequestGen(BaseModel):
    blurb: str # { blurb : "xyz" } -> recommend jobs based on user's blurb or text
generator = llmRecommender(supabase)
@app.post("/generate")
@app.post("/generate/")
def generate(req: ExampleRequestGen):
    try:
        # Load recommender once

        print("JUST STARTING")
        blurb = req.blurb
        print(f"USERID:{blurb}")
        
        print("INIT IS DONE")
        generator.fetch_data()
        print("Before MODEL FIT")
        generator.load_data()
        print("MODEL FIT IS DONE")
        generator.build_qa_chain()
        print("USER RECOMMENDATIONS TAKEN INTO CONSIDERATION")
        gen_recommendations = generator.recommend(blurb)
        # output_path = f"/tmp/recommendations_{user_id}.csv"
        # recommendations.to_csv(output_path, index=False)
        # print(f"Recommendations written to: {output_path}")
        # Convert DataFrame to list of dictionaries
        recommendations_list = gen_recommendations.to_dict(orient="records")

        # print(f"RECOMMENDATIONS:{recommendations_list}")
        return {
            'jobs' : recommendations_list
        }
    except Exception as e:
        import traceback
        print("bad ERROR OCCURRED:")
        traceback.print_exc() 
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000))) # starts the FASTAPI server for the model APIs

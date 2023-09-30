import pandas as pd
import uvicorn
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse, orjson
from sentence_transformers import SentenceTransformer, util

class SimilarJobRecommender():
    def __init__(self):
        self.profs = pd.read_csv('data/profs.csv')
        self.profs['comb'] = self.profs.apply(self.combibe_texts, axis=1)
        self.sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.profs['embeddings'] = self.sbert_model.encode(self.profs['comb'].to_list(), convert_to_tensor=True)

    def combibe_texts(self, row):
        txt = ''
        for i in row:
            if isinstance(i, str):
                txt+= f'{i} '
        return txt

    def get_recommended_prfs(self, student, profs):
        comb = self.combibe_texts(student)
        embeddings_student = self.sbert_model.encode(comb, convert_to_tensor=True)
        scores = []
        for __, row in profs.iterrows():
            score = util.pytorch_cos_sim(embeddings_student, row['embeddings'])
            scores.append({
                'score' : score.item(),
                'id' : row['id'],
                'name' : row['name']
            })

        score_df = pd.DataFrame(scores)
        score_df.fillna(' ',inplace=True)
        rec = (score_df.sort_values("score", ascending=False).head()
               .to_dict(orient='records'))
        return rec

    def make_rec_from_applied_job(self, student):
        profs = self.profs
        recommendations = self.get_recommended_prfs(student, profs)
        return recommendations

    def get_recommendation(self, student):
        recommended = self.make_rec_from_applied_job(student)
        return recommended

sim_job_rec = SimilarJobRecommender()

app = FastAPI()
@app.post("/")
async def rec(request: Request):
    # st_dict = {
    #     "interests": "Cyber Security, Computer Networking",
    #     "thesis": "Machine Learning project",
    #     "subject": "CSE",
    #     "faculty": "CSE",
    #     "graduation_class": "MS",
    #     "paper": "",
    #     "major": ""
    # }

    st_dict = await request.json()
    student = pd.DataFrame([st_dict])
    recommended = sim_job_rec.get_recommendation(student)
    return JSONResponse(content=recommended)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
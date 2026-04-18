import json
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

class QAEvaluator:
    def __init__(self, embedding_model, llm_client):
        """
        Initialize the evaluator with shared resources to avoid reloading models.
        """
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate_single(self, prediction: str, reference: str, question: str) -> dict:
        """
        Evaluate a single prediction against the ground truth using multiple metrics.
        """
        rouge_l = 0.0
        if prediction and reference:
            rouge_l = self.rouge.score(reference, prediction)['rougeL'].fmeasure
        
        pred_vec = self.embedding_model.embed_query(prediction if prediction else "")
        ref_vec = self.embedding_model.embed_query(reference if reference else "")
        sim = cosine_similarity([pred_vec], [ref_vec])[0][0]
        
        judge_score = self._get_llm_judge_score(prediction, reference, question)
        
        return {
            "ROUGE-L": rouge_l,
            "Semantic_Sim": float(sim),
            "LLM_Score": judge_score
        }

    def _get_llm_judge_score(self, prediction: str, reference: str, question: str) -> int:
        """
        Use the LLM as a judge to evaluate the accuracy of the generated answer.
        """
        if not prediction:
            return 1

        prompt = f"""You are an expert public health evaluator. 
Evaluate the AI answer based strictly on the ground truth.
Score from 1 (poor) to 5 (excellent).

Question: {question}
Ground Truth: {reference}
AI Answer: {prediction}

Return a valid JSON object only: {{"score": <int>}}
"""
        try:
            response = self.llm_client.chat(
                model="ministral-3:8b-cloud",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )
            content = response.message.content.strip()
            
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                data = json.loads(json_str)
                return data.get("score", 1)
            return 1
        except Exception as e:
            return 1
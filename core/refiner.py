import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_ollama import OllamaEmbeddings

# CONFIG
GEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "nomic-embed-text"

class QueryRefiner:
    def __init__(self):
        print(f"🧹 Initializing Refiner & Classifier...")
        
        # 1. Generative Model (for rewriting text only)
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID)
        
        # 2. Embedding Model (for strict classification)
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)
        
        # 3. Pre-compute Domain Vectors (The "NLP" Part)
        # Rich descriptions help the embeddings match user queries better
        self.domain_map = {
            "Promissory Estoppel": "promise relied upon detriment waiver rent reduction discount landlord tenant agreed lower payment covid pandemic relied spent savings cannot go back promise binding reliance",
            "Contractual Terms": "written contract clauses unfair terms exclusion clause breach written agreement liability limited exemption clause incorporation notice signature",
            "Misrepresentation": "lied before signing fraud false statement induced sign fake details misled untrue claim pre-contractual statement rescission damages innocent negligent fraudulent",
            "Mistake- Mutual mistake": "both parties confused fundamental error wrong item identity mistake void contract common mistake subject matter ceased exist different thing",
            "Offer & Acceptance": "contract formed counter offer revocation silence acceptance postal rule invitation treat display goods shop window rejection mirror image rule"
        }
        
        print("🧮 Pre-computing domain vectors...")
        self.domain_names = list(self.domain_map.keys())
        self.domain_descriptions = list(self.domain_map.values())
        self.domain_vectors = self.embedder.embed_documents(self.domain_descriptions)
        print("✅ System Ready.")
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def refine(self, raw_query):
        """
        1. Classify using Vector Similarity (Math).
        2. Rewrite using LLM (Generative).
        
        Returns:
            tuple: (list of domain tags, refined query string)
        """
        # --- STEP A: CLASSIFICATION (The NLP Approach) ---
        query_vec = self.embedder.embed_query(raw_query)
        scores = []
        
        for i, domain_vec in enumerate(self.domain_vectors):
            score = self._cosine_similarity(query_vec, domain_vec)
            scores.append((score, self.domain_names[i]))
        
        # Sort by highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        top_score, top_domain = scores[0]
        
        # Threshold check (optional)
        final_domains = [top_domain] if top_score > 0.4 else ["General"]
        
        # Debugging: See what the math picked
        print(f"📊 Vector Scores: {scores[:2]}") 
        
        # --- STEP B: REWRITING (The LLM Approach) ---
        system_prompt = f"""You are a legal secretary. Rewrite this client query into a formal legal question for a specialist in {top_domain}.
Do NOT output the domain name. Just the question.
Input: "{raw_query}"
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Formal Query:"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt")
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=128,
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        refined_query = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return final_domains, refined_query.strip()

# Test block
if __name__ == "__main__":
    r = QueryRefiner()
    d, q = r.refine("landlord promised to lower rent but now wants it back")
    print(f"Domain: {d}")
    print(f"Query: {q}")
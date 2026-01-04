import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_ollama import OllamaEmbeddings
import os

# CONFIG
# Use smaller, faster model by default. Set to "Qwen/Qwen2.5-7B-Instruct" for higher quality (slower)
# Smaller models: TinyLlama-1.1B-Chat (best), microsoft/phi-2, OpenBuddy/openbuddy-llm2-7b
GEN_MODEL_ID = os.environ.get("REFINER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
EMBED_MODEL = "nomic-embed-text"

# Detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

class QueryRefiner:
    def __init__(self):
        print(f"🧹 Initializing Refiner & Classifier on {DEVICE.upper()}...")
        print(f"   Model: {GEN_MODEL_ID}")

        # 1. Generative Model (for rewriting text only)
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
            device_map=DEVICE
        )
        self.model.eval()  # Set to eval mode for inference
        
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
        # Build messages in chat format for better instruction following
        messages = [
            {"role": "system", "content": f"You are a legal secretary. Rewrite queries as formal legal questions for {top_domain} specialists. Be concise."},
            {"role": "user", "content": f'Rewrite as a formal legal question:\n"{raw_query}"'}
        ]

        # Apply chat template if available, otherwise use simple format
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback for models without chat templates
            text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.5,
                top_p=0.9
            )

        # Extract only the generated part (after the input)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        refined_query = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return final_domains, refined_query.strip()

# Test block
if __name__ == "__main__":
    r = QueryRefiner()
    d, q = r.refine("landlord promised to lower rent but now wants it back")
    print(f"Domain: {d}")
    print(f"Query: {q}")
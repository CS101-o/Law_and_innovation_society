from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- CONFIGURATION ---
# We use Qwen 2.5 (0.5B) - A state-of-the-art tiny model
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

class QueryRefiner:
    def __init__(self):
        print(f"🧹 Initializing Query Refiner ({MODEL_ID})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        print("✅ Refiner Ready.")

    def refine(self, raw_query):
        """
        Takes a messy user input and rewrites it into a structured legal prompt.
        """
        # The "Receptionist" Instruction
        system_prompt = """You are a legal secretary. Your job is to rewrite the client's vague query into a clear, detailed legal question for a UK Lawyer.
        
        RULES:
        1. Expand abbreviations (e.g., "calc" -> "calculation").
        2. Identifying the legal domain (e.g., "This looks like a Consumer Rights issue").
        3. Keep it neutral and professional.
        4. Do NOT answer the question. Only rewrite it.
        
        Example Input: "car broke after 2 weeks dealer wont refund"
        Example Output: "I purchased a vehicle which developed a fault 14 days after purchase. The dealer has refused a refund. Please advise on my 'Short-term Right to Reject' under the Consumer Rights Act 2015."
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Client Query: {raw_query}\n\nRewritten Legal Query:"}
        ]

        # Prepare input
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt")

        # Generate (Fast!)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=128,
            do_sample=False  # Deterministic is faster
        )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

# Simple test if you run this file directly
if __name__ == "__main__":
    refiner = QueryRefiner()
    print(refiner.refine("my data leaked"))
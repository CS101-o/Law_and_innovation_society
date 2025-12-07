import sys
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
MODEL_NAME = "uk-lawyer"

def main():
    print(f"⚖️  Connecting to {MODEL_NAME}...")
    
    # 1. Initialize with EXPANDED MEMORY & HARD BRAKES
    try:
        llm = ChatOllama(
            model=MODEL_NAME, 
            temperature=0.1,
            num_ctx=8192,  # <--- CRITICAL: Huge memory for long questions
            # The "Emergency Brakes" list:
            stop=[
                "<|eot_id|>",          # The official Llama 3 stop signal
                "<|start_header_id|>", # The official Llama 3 start signal
                "📝",                  # The icon it uses to hallucinate new questions
                "Client Question:",    # The phrase it uses to restart the loop
                "----------------",    # The divider line it draws
            ]
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Strict Template
    # We force the exact Llama 3 structure so it knows who is talking
    template = """<|start_header_id|>system<|end_header_id|>

You are an expert UK Contract Law consultant.
Answer the user's question using the IRAC method.
Be concise. Stop generating immediately after the Conclusion.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 3. Chat Loop
    print("\n✅ LAWYER READY. Type 'exit' to quit.\n" + "="*50)
    
    while True:
        try:
            query = input("\n📝 Client Question: ")
            if query.lower() in ["exit", "quit"]: 
                break
            
            print("\nThinking...\n")
            
            full_response = ""
            for chunk in chain.stream({"question": query}):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
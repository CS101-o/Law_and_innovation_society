# Stage 2: Response Simplification Layer - Implementation Guide

## Overview

Your legal AI pipeline now includes a **Stage 2 simplification layer** that converts technical IRAC responses into elderly-friendly, plain-language explanations.

### Pipeline Flow

```
1. User Question
   ↓
2. Query Refinement (QueryRefiner) - Classifies legal domain
   ↓
3. Hybrid RAG Retrieval - Fetches relevant documents & knowledge graph
   ↓
4. Stage 1: Technical Response (Fine-tuned Qwen model via Ollama)
   ↓
5. Stage 2: Simplification (mlx-community/Qwen2.5-7B-Instruct-4bit)
   ↓
6. Elderly-Friendly Explanation (Displayed to user)
```

## Installation

### Step 1: Install mlx-lm

```bash
pip install mlx-lm
```

**Note:** `mlx-lm` is optimized for Apple Silicon (M1/M2/M3 Macs). For other platforms:
- **Linux/Windows:** May work but not officially supported. Consider using `ollama` instead.
- **Requires:** Python 3.9+

### Step 2: Update Requirements (Already Done)

The `requirements.txt` file has been updated with:
```
mlx-lm
torch
```

Run:
```bash
pip install -r requirements.txt
```

## How It Works

### The Simplifier Module (`core/simplifier.py`)

```python
from core.simplifier import get_simplifier

simplifier = get_simplifier()
simplified_response = simplifier.simplify(
    technical_response="Complex IRAC analysis...",
    original_question="What is estoppel?",
    max_tokens=512
)
```

### Key Features

✅ **Plain Language** - Removes legal jargon (estoppel, promissory, consideration, etc.)
✅ **Short Sentences** - One idea per sentence, easy to follow
✅ **Warm Tone** - Kind, patient, friendly explanations
✅ **Practical Focus** - Explains what it means for the person
✅ **Ends Helpfully** - "Would you like me to explain anything else?"

### System Prompt (Embedded)

```
You are a kind, patient legal assistant helping elderly clients understand the law.
Your job is to rewrite complex legal explanations in simple everyday language.

Rules:
- Use short, simple sentences (one idea per sentence)
- No legal jargon
- Warm, friendly tone
- Explain what this means for the person in practical terms
- End with: "Would you like me to explain anything else?"
```

## Integration in Streamlit App

### 1. Loader Function

```python
@st.cache_resource
def load_simplifier():
    """Load the Stage 2 simplification model (mlx-lm based)."""
    try:
        return get_simplifier()
    except ImportError:
        st.warning("⚠️ mlx-lm not installed. Install with: pip install mlx-lm")
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load simplifier: {e}")
        return None

simplifier = load_simplifier()  # Cached during app startup
```

### 2. Response Processing

After the fine-tuned model generates the technical response, the simplifier processes it:

```python
# Stage 1: Generate IRAC response
for chunk in rag_chain.stream(refined_text):
    full_response += chunk

# Stage 2: Generate elderly-friendly simplification
full_response = _generate_elderly_friendly_answer(
    full_response,
    refined_text,
    llm_engine
)
```

### 3. Output Format

The response includes both versions:

```
[Technical IRAC Analysis]
...
---

**Elderly-Friendly Explanation:**

[Simple, warm explanation]
```

## Usage Examples

### Example 1: Contract Breach Question

**User Question:** "I signed a contract but didn't read all the terms. Can they hold me to something I didn't understand?"

**Stage 1 Output (Technical):**
```
IRAC Analysis:
Issue: Whether the claimant is bound by unread terms...
Rule: The postal rule, incorporation of terms, and caveat emptor...
Analysis: In L'Estrange v Graucob, it was held that...
Conclusion: Generally, you are bound by the terms of the contract you signed.
```

**Stage 2 Output (Simplified):**
```
Elderly-Friendly Explanation:

When you sign something, you're agreeing to the words on that paper - even if you didn't read them all.

The law says you're responsible for what's written, not what you thought was written.

Think of it like this: If you sign a receipt without reading it, you're still agreeing to what it says.

If something wasn't fair or you were tricked, that's different - but just not reading it isn't an excuse.

Would you like me to explain anything else?
```

### Example 2: Promissory Estoppel

**User Question:** "My landlord promised to reduce my rent during COVID, but now wants the full amount back. Is that legal?"

**Stage 2 Output (Simplified):**
```
Elderly-Friendly Explanation:

If your landlord promised to lower your rent and you counted on that promise, they might not be able to take it back.

This is called "promissory estoppel" - basically, if someone makes you a promise you rely on, they can't just change their mind.

For this to work:
1. There must have been a clear promise
2. You must have trusted it and acted on it
3. It would be unfair for them to take it back now

So if you didn't spend money or make other decisions because of the promise, they might still be able to increase it.

Would you like me to explain anything else?
```

## Model Details

### Base Model: `mlx-community/Qwen2.5-7B-Instruct-4bit`

- **Size:** 7B parameters (quantized to 4-bit)
- **Speed:** Fast inference on Apple Silicon
- **Quality:** Good instruction-following, suitable for rewriting tasks
- **Memory:** ~3-4GB when loaded

### Alternative Models (if needed)

If you want different characteristics:

```python
# Faster but lower quality (3B)
model_id = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Slower but higher quality (7B)
model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # Default

# Or any other instruction-tuned model from mlx-community
```

## Configuration

### Adjust Response Length

Modify `max_tokens` in `core/simplifier.py`:

```python
simplified = simplifier.simplify(
    irac_response,
    original_question,
    max_tokens=256  # Shorter responses
    # or
    max_tokens=768  # Longer responses
)
```

### Customize System Prompt

Edit the system prompt in `core/simplifier.py`, line ~24:

```python
system_prompt = """Your custom instructions here..."""
```

## Troubleshooting

### Issue: ImportError: No module named 'mlx_lm'

**Solution:**
```bash
pip install mlx-lm
```

### Issue: "Could not load simplifier" warning in Streamlit

**Possible causes:**
- mlx-lm not installed
- Model download failed (requires ~7GB free disk space)
- Apple Silicon not available (mlx-lm only works on M1/M2/M3 Macs)

**Solution:**
- Ensure mlx-lm is installed: `pip install mlx-lm`
- Check internet connection (model downloads on first use)
- Simplifier gracefully falls back to old method if unavailable

### Issue: First run takes a long time

**Expected behavior:** First run downloads the 7B model (~3-4GB). This is normal and happens only once.

**Solution:** Be patient on first run. Subsequent uses will be much faster.

### Issue: Model takes too much memory

**Solution:** Use the smaller 3B model instead:
```python
# In load_simplifier() or core/simplifier.py
model_id = "mlx-community/Qwen2.5-3B-Instruct-4bit"
```

## Performance Notes

### Response Times

- **Stage 1 (Technical):** ~5-15 seconds (depends on Ollama model)
- **Stage 2 (Simplification):** ~5-10 seconds (mlx-lm inference)
- **Total:** ~10-25 seconds per query

### Memory Usage

- **Fine-tuned Qwen (Ollama):** ~4-8GB
- **Simplifier Model:** ~3-4GB
- **Total system:** ~8-12GB RAM minimum

## Rollback / Disabling

If you need to disable simplification:

1. **Comment out the simplifier import** in `lawyer_gui.py`:
   ```python
   # from core.simplifier import get_simplifier
   ```

2. **Comment out the simplifier loader**:
   ```python
   # simplifier = load_simplifier()
   ```

3. **Revert the function call** to old method:
   ```python
   full_response = _generate_simple_answer(full_response, refined_text, llm_engine)
   ```

The app will still work - it will just use the old LLM-based simple answer generation.

## Advanced: Custom Simplification

If you want to build on this, you can:

1. **Add multiple simplification styles:**
   ```python
   simplifier.simplify(text, question, style="elderly")  # Kind tone
   simplifier.simplify(text, question, style="technical")  # More detail
   simplifier.simplify(text, question, style="brief")  # Very short
   ```

2. **Add language support:**
   ```python
   simplifier.simplify(text, question, language="welsh")
   ```

3. **Cache simplifications:**
   - Store hash of (IRAC response + question) → simplified response
   - Avoid recomputing for similar queries

## Files Modified

- ✅ `core/simplifier.py` - New Stage 2 module
- ✅ `app/lawyer_gui.py` - Integrated simplifier into pipeline
- ✅ `requirements.txt` - Added mlx-lm and torch

## Summary

You now have a two-stage response generation pipeline:

**Stage 1 (Accuracy):** Fine-tuned Qwen model via Ollama produces accurate legal analysis
**Stage 2 (Accessibility):** Base Qwen2.5-7B-Instruct-4bit simplifies for elderly users

This ensures your app is both **legally accurate** and **customer-friendly** for all audiences.

---

Questions? Check the main app logs (Streamlit terminal) for detailed error messages.

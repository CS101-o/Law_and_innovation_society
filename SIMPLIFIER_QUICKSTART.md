# Stage 2 Simplifier - Quick Start

## What's New ✨

Your pipeline now has a **two-stage response system**:

```
Fine-tuned Qwen (Accurate IRAC)
        ↓
mlx-community/Qwen2.5-7B-Instruct (Simplify to plain language)
        ↓
Elderly-friendly output
```

## Installation (One Command)

```bash
pip install mlx-lm
```

Or update everything:
```bash
pip install -r requirements.txt
```

## What Changed

### New Files
- **`core/simplifier.py`** - The Stage 2 simplification module

### Modified Files
- **`app/lawyer_gui.py`** - Integrated simplifier into the response pipeline
- **`requirements.txt`** - Added `mlx-lm` and `torch`

## How It Works

1. User asks a legal question
2. Your fine-tuned Qwen model generates technical IRAC response
3. **NEW:** Simplifier rewrites it in plain, warm language
4. User sees both the technical analysis and the simple explanation

### Example Output

```
[Technical IRAC Analysis about contract breach...]

---

**Elderly-Friendly Explanation:**

When you sign something, you're agreeing to what's written - even if you didn't read every word.

The law says you're responsible for the contract you signed.

If you were tricked or something wasn't fair, that's different. But just not reading it isn't an excuse.

Would you like me to explain anything else?
```

## Key Features

✅ **No Legal Jargon** - Removes estoppel, promissory, consideration, etc.
✅ **Short Sentences** - One idea per line, easy to follow
✅ **Warm Tone** - Kind, patient, friendly
✅ **Practical** - Explains what it means for the person
✅ **Inviting** - Ends with "Would you like me to explain anything else?"

## Run It

```bash
cd /Users/kaanoktem/UOM-Law
streamlit run app/lawyer_gui.py
```

The simplifier loads automatically in the background (cached for speed).

## Configuration

**Adjust response length** (in `core/simplifier.py`):
```python
simplified = simplifier.simplify(
    response,
    question,
    max_tokens=512  # Change this
)
```

**Use a faster model** (smaller, less quality):
```python
model_id = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # Instead of 7B
```

**Customize the simplification instructions** (in `core/simplifier.py`):
```python
system_prompt = """Your custom prompt here..."""
```

## First Run

⏱️ **First run takes 2-3 minutes** (downloads ~3-4GB model)
⚡ **Subsequent runs take ~5-10 seconds** (model cached)

You'll see in console:
```
📖 Loading simplifier model: mlx-community/Qwen2.5-7B-Instruct-4bit...
✅ Simplifier ready for elderly-friendly responses.
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError: mlx_lm | `pip install mlx-lm` |
| "Could not load simplifier" | Check mlx-lm installed, internet working, disk space |
| First run very slow | Normal - downloading model. Be patient. |
| Takes too much memory | Use 3B model instead of 7B |
| App still works but no simplification | mlx-lm optional - app falls back gracefully |

## Full Documentation

See **`STAGE2_SIMPLIFIER_GUIDE.md`** for:
- Detailed architecture
- Advanced configuration
- Custom model setup
- Performance tuning
- Rollback instructions

## Technical Details

- **Model:** mlx-community/Qwen2.5-7B-Instruct-4bit
- **Framework:** mlx-lm (Apple Silicon optimized)
- **Input:** IRAC response + original question
- **Output:** Plain-language explanation (~200-300 words)
- **Speed:** ~5-10 seconds per simplification
- **Memory:** ~3-4GB for model

## What's Happening Behind the Scenes

```python
# In app/lawyer_gui.py after IRAC generation:

full_response = _generate_elderly_friendly_answer(
    irac_response,      # Technical analysis from Stage 1
    original_question,  # User's original question
    llm_engine          # Fallback LLM
)

# This calls the simplifier:
simplified = simplifier.simplify(
    irac_response,
    original_question,
    max_tokens=512
)

# Returns both versions to user
```

---

**You're all set!** Just install mlx-lm and run the app. The simplifier will load automatically and enhance your responses. 🚀

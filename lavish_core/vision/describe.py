import base64
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from openai import OpenAI

# Initialize OpenAI (GPT-4V)
openai_client = OpenAI()

# Load your LLaMA / DeepSeek model (local)
LLAMA_MODEL = "meta-llama/Llama-3.2-11B-Vision"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-coder"

def encode_image(image_path):
    """Convert image to base64 for GPT or ML model input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def llama_vision_analysis(image_path):
    """Analyze image using LLaMA Vision model."""
    try:
        processor = AutoProcessor.from_pretrained(LLAMA_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, torch_dtype=torch.float16, device_map="auto")
        image = Image.open(image_path)
        inputs = processor(text="Describe this image in detail.", images=image, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=100)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"LLaMA analysis failed: {e}"

def deepseek_coder_analysis(image_path):
    """Optional DeepSeek agent: checks for patterns, context, or OCR-level data."""
    try:
        processor = AutoProcessor.from_pretrained(DEEPSEEK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL, torch_dtype=torch.float16, device_map="auto")
        image = Image.open(image_path)
        inputs = processor(text="Extract objects or readable text from this image.", images=image, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=80)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"DeepSeek analysis failed: {e}"

def gpt_vision_analysis(image_path):
    """Use GPT-4V for advanced reasoning and contextual understanding."""
    try:
        img_b64 = encode_image(image_path)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that interprets images accurately."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image accurately and in depth."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                ]}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT-4V analysis failed: {e}"

def describe_image(image_path):
    """Combine GPT-4V + LLaMA + DeepSeek results for best image understanding."""
    results = {}

    # Step 1 – Run agents
    results["GPT-4V"] = gpt_vision_analysis(image_path)
    results["LLaMA-Vision"] = llama_vision_analysis(image_path)
    results["DeepSeek"] = deepseek_coder_analysis(image_path)

    # Step 2 – Merge insights
    final_summary = (
        f"🧠 GPT Insight: {results['GPT-4V']}\n\n"
        f"🦙 LLaMA Vision: {results['LLaMA-Vision']}\n\n"
        f"🔍 DeepSeek Context: {results['DeepSeek']}\n\n"
        "✅ Final Summary: Combined understanding across visual and contextual reasoning agents."
    )

    return final_summary
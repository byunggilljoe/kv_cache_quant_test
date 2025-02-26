import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def measure_inference_time(model, tokenizer, prompt: str, num_repeats: int = 3) -> Tuple[float, float, float]:
    """Measure average inference time, memory usage, and tokens per second"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warm-up
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)
    
    # Measure time
    start_time = time.time()
    memory_before = get_memory_usage()
    
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_repeats):
            output = model.generate(**inputs, max_new_tokens=50)
            total_tokens += output.shape[1]  # Count total generated tokens including input
    
    avg_time = (time.time() - start_time) / num_repeats
    memory_used = get_memory_usage() - memory_before
    tokens_per_sec = (total_tokens / num_repeats) / avg_time
    
    return avg_time, memory_used, tokens_per_sec

def main():
    # Model configurations
    model_configs = {
        "4-bit": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "load_kwargs": {"load_in_4bit": True}
        },
        "8-bit": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "load_kwargs": {"load_in_8bit": True}
        },
        "fp16": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "load_kwargs": {"torch_dtype": torch.float16}
        },
        "default": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "load_kwargs": {}
        }
    }

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for quant_name, config in model_configs.items():
        print(f"\nTesting {quant_name} quantization...")
        
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                device_map="auto",
                **config["load_kwargs"]
            )
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

            # Test inference
            times = []
            memories = []
            tokens_per_sec_list = []
            generated_texts = []

            for prompt in prompts:
                inference_time, memory_usage, tokens_per_sec = measure_inference_time(model, tokenizer, prompt)
                times.append(inference_time)
                memories.append(memory_usage)
                tokens_per_sec_list.append(tokens_per_sec)

                # Generate text for sample output
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=50)
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_texts.append(generated_text)

            # Store results
            results[quant_name] = {
                "avg_time": sum(times) / len(times),
                "avg_memory": sum(memories) / len(memories),
                "avg_tokens_per_sec": sum(tokens_per_sec_list) / len(tokens_per_sec_list),
                "sample_outputs": generated_texts[:2]  # Store first two outputs as samples
            }

            # Print results
            print(f"Average inference time: {results[quant_name]['avg_time']:.3f} seconds")
            print(f"Average memory usage: {results[quant_name]['avg_memory']:.2f} MB")
            print(f"Average tokens per second: {results[quant_name]['avg_tokens_per_sec']:.2f}")
            print("\nSample outputs:")
            for i, text in enumerate(results[quant_name]['sample_outputs']):
                print(f"Prompt {i+1}: {prompts[i]}")
                print(f"Generated: {text[:100]}...")
                print()

            # Cleanup
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {quant_name}: {str(e)}")
            results[quant_name] = {"error": str(e)}

    # Print comparative results
    print("\nComparative Results:")
    print("-" * 80)
    for quant_name, result in results.items():
        if "error" not in result:
            print(f"{quant_name}:")
            print(f"  Average inference time: {result['avg_time']:.3f} seconds")
            print(f"  Average memory usage: {result['avg_memory']:.2f} MB")
            print(f"  Average tokens per second: {result['avg_tokens_per_sec']:.2f}")
            print("-" * 40)

if __name__ == "__main__":
    main() 
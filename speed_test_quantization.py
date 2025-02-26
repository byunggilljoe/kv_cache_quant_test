# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# NOTE: This is just a running example. For benchmarking purpose,
# please see benchmarks/benchmark_prefix_caching.py

quant_model_name = {
    "awq-int4": ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "awq"),
    "w4a16": ("neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16", None),
    "w8a8": ("neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", None),
    "fp8-dynamic": ("neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic", None),
    "fp8": ("neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8", "compressed-tensors"),
    "None": ("meta-llama/Meta-Llama-3.1-8B-Instruct", None),
    
}

PREFIX_MULTIPLIER = 1
PROMPTS_NUM_MULTIPLIER = 100 #1000
MAX_NUM_SEQS = 1 #64 # 
EAGER_MODE = True

# Common prefix.
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: "
    )


prefix = prefix * PREFIX_MULTIPLIER


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


prompts = prompts * PROMPTS_NUM_MULTIPLIER

generating_prompts = [prefix + prompt for prompt in prompts]


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

for quant_name, quant_model in quant_model_name.items():    
    # Create LLMs with different quantization settings
    print(f"Testing {quant_name} quantization")
    llm = LLM(
        model=quant_model[0],
        gpu_memory_utilization=0.5,
        enforce_eager=EAGER_MODE,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=False,
        quantization=quant_model[1]
    )

    # Generate and measure performance for FP16
    outputs = llm.generate(generating_prompts, sampling_params)

    generated_texts = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        if i < 3:
            print(f"Prompt: {prompt[:100]!r}, Generated text: {generated_text[:100]!r}")

    print("-" * 80)
    del llm
    cleanup_dist_env_and_memory()
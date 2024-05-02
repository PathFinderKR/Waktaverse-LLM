# Waktaverse-Llama-3-KO-8B-Instruct

## Model Details

![image/webp](https://cdn-uploads.huggingface.co/production/uploads/65d6e0640ff5bc0c9b69ddab/Va78DaYtPJU6xr4F6Ca4M.webp)
Waktaverse-Llama-3-KO-8B-Instruct is a state-of-the-art Korean language model developed by Waktaverse AI team.
This large language model is a specialized version of the Meta-Llama-3-8B-Instruct, tailored for Korean natural language processing tasks. 
It is designed to handle a variety of complex instructions and generate coherent, contextually appropriate responses.

- **Developed by:** Waktaverse AI
- **Model type:** Large Language Model
- **Language(s) (NLP):** Korean, English
- **License:** [Llama3](https://llama.meta.com/llama3/license)
- **Finetuned from model:** [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Model Sources

- **Model Card:** [Hugging Face](https://github.com/PathFinderKR/Waktaverse-LLM/tree/main)
- **Paper :** [More Information Needed]



## Uses

### Direct Use

The model can be utilized directly for tasks such as text completion, summarization, and question answering without any fine-tuning. 

### Out-of-Scope Use

This model is not intended for use in scenarios that involve high-stakes decision-making including medical, legal, or safety-critical areas due to the potential risks of relying on automated decision-making. 
Moreover, any attempt to deploy the model in a manner that infringes upon privacy rights or facilitates biased decision-making is strongly discouraged.

## Bias, Risks, and Limitations

While Waktaverse Llama 3 is a robust model, it shares common limitations associated with machine learning models including potential biases in training data, vulnerability to adversarial attacks, and unpredictable behavior under edge cases. 
There is also a risk of cultural and contextual misunderstanding, particularly when the model is applied to languages and contexts it was not specifically trained on.



## How to Get Started with the Model

You can run conversational inference using the Transformers Auto classes.
We highly recommend that you add Korean system prompt for better output.
Adjust the hyperparameters as you need.

### Example Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = (
    "cuda:0" if torch.cuda.is_available() else # Nvidia GPU
    "mps" if torch.backends.mps.is_available() else # Apple Silicon GPU
    "cpu"
)

model_id = "PathFinderKR/Waktaverse-Llama-3-KO-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)

################################################################################
# Generation parameters
################################################################################
num_return_sequences=1
max_new_tokens=1024
temperature=0.9
top_k=40
top_p=0.9
repetition_penalty=1.1

def generate_response(system ,user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

system_prompt = "다음 지시사항에 대한 응답을 작성해주세요."
user_prompt = "피보나치 수열에 대해 설명해주세요."
response = generate_response(system_prompt, user_prompt)
print(response)
```

### Example Output

```python
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

다음 지시사항에 대한 응답을 작성해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

피보나치 수열에 대해 설명해주세요.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

피보나치 수열은 수학에서 가장 유명한 수열 중 하나로, 0과 1로 시작하는 숫자들의 모임입니다. 각 숫자는 이전 두 개의 숫자의 합으로 정의되며, 이렇게 계속 반복됩니다. 피보나치 수열은 무한히 커지는데, 첫 번째와 두 번째 항이 모두 0일 수도 있지만 일반적으로는 첫 번째 항이 1이고 두 번째 항이 1입니다.

예를 들어, 0 + 1 = 1, 1 + 1 = 2, 2 + 1 = 3, 3 + 2 = 5, 5 + 3 = 8, 8 + 5 = 13, 13 + 8 = 21, 21 + 13 = 34 등이 있습니다. 이 숫자들을 피보나치 수열이라고 합니다.

피보나치 수열은 다른 수열들과 함께 사용될 때 도움이 됩니다. 예를 들어, 금융 시장에서는 금리 수익률을 나타내기 위해 이 수열이 사용됩니다. 또한 컴퓨터 과학과 컴퓨터 과학에서도 종종 찾을 수 있습니다. 피보나치 수열은 매우 복잡하며 많은 숫자가 나오므로 일반적인 수열처럼 쉽게 구할 수 없습니다. 이 때문에 피보나치 수열은 대수적 함수와 관련이 있으며 수학자들은 이를 연구하고 계산하기 위해 다양한 알고리즘을 개발했습니다.

참고 자료: https://en.wikipedia.org/wiki/Fibonacci_sequence#Properties.<|eot_id|>
```



## Training Details

### Training Data

The model is trained on the [MarkrAI/KoCommercial-Dataset](https://huggingface.co/datasets/MarkrAI/KoCommercial-Dataset), which consists of various commercial texts in Korean.

### Training Procedure

The model training used LoRA for computational efficiency. 0.02 billion parameters(0.26% of total parameters) were trained.

#### Training Hyperparameters

```python
################################################################################
# bitsandbytes parameters
################################################################################
load_in_4bit=True
bnb_4bit_compute_dtype=torch_dtype
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=False

################################################################################
# LoRA parameters
################################################################################
task_type="CAUSAL_LM"
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
r=8
lora_alpha=16
lora_dropout=0.05
bias="none"

################################################################################
# TrainingArguments parameters
################################################################################
num_train_epochs=2
per_device_train_batch_size=1
gradient_accumulation_steps=1
gradient_checkpointing=True
learning_rate=2e-5
lr_scheduler_type="cosine"
warmup_ratio=0.1
optim = "adamw_torch"
weight_decay=0.1

################################################################################
# SFT parameters
################################################################################
max_seq_length=1024
packing=True
```



## Evaluation

### Metrics

#### English

- **AI2 Reasoning Challenge (25-shot):** a set of grade-school science questions.
- **HellaSwag (10-shot):** a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.
- **MMLU (5-shot):** a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
- **TruthfulQA (0-shot):** a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
- **Winogrande (5-shot):** an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.
- **GSM8k (5-shot):** diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.

#### Korean

- **Ko-HellaSwag:**
- **Ko-MMLU:**
- **Ko-Arc:**
- **Ko-Truthful QA:**
- **Ko-CommonGen V2:**
  
### Results

#### English

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Waktaverse Llama 3 8B</strong>
   </td>
   <td><strong>Llama 3 8B</strong>
   </td>
  </tr>
  <tr>
   <td>Average
   </td>
   <td>66.77
   </td>
   <td>66.87
   </td>
  </tr>
  <tr>
   <td>ARC
   </td>
   <td>60.32
   </td>
   <td>60.75
   </td>
  </tr>
  <tr>
   <td>HellaSwag
   </td>
   <td>78.55
   </td>
   <td>78.55
   </td>
  </tr>
  <tr>
   <td>MMLU
   </td>
   <td>67.9
   </td>
   <td>67.07
   </td>
  </tr>
  <tr>
   <td>Winograde
   </td>
   <td>74.27
   </td>
   <td>74.51
   </td>
  <tr>
    <td>GSM8K
   </td>
   <td>70.36
   </td>
   <td>68.69
   </td>
  </tr>
</table>

#### Korean

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Waktaverse Llama 3 8B</strong>
   </td>
   <td><strong>Llama 3 8B</strong>
   </td>
  </tr>
  <tr>
   <td>Ko-HellaSwag:
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Ko-MMLU:
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Ko-Arc:
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Ko-Truthful QA:
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Ko-CommonGen V2:
   </td>
   <td>0
   </td>
   <td>0
   </td>
</table>

## Technical Specifications

### Compute Infrastructure

#### Hardware

- **GPU:** NVIDIA GeForce RTX 4080 SUPER

#### Software

- **Operating System:** Linux
- **Deep Learning Framework:** Hugging Face Transformers, PyTorch

### Training Details

- **Training time:** 32 hours
- **VRAM usage:** 12.8 GB
- **GPU power usage:** 300 W



## Citation

**Waktaverse-Llama-3**

```
TBD
```

**Llama-3**

```
@article{llama3modelcard,
  title={Llama 3 Model Card},
  author={AI@Meta},
  year={2024},
  url = {https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md}
}
```



## Model Card Authors

[More Information Needed]



## Model Card Contact

[More Information Needed]

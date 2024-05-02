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

- To get started with the model, visit [Hugging Face Model Card](https://huggingface.co/PathFinderKR/Waktaverse-Llama-3-KO-8B-Instruct)
- Code used to train the model could be found [here](https://github.com/PathFinderKR/Waktaverse-LLM/blob/main/SFT.ipynb)
- **Paper: **



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

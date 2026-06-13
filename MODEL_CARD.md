---
base_model: openbmb/MiniCPM-V-4.6
license: apache-2.0
language:
- en
tags:
- medical
- vision-language
- lab-report
- ocr-free-extraction
- minicpm-v
- lora
- build-small-hackathon
pipeline_tag: image-text-to-text
library_name: transformers
---

# Blood Test Explainer — MiniCPM-V 4.6 (medical-reasoning fine-tune)

A ~1.3B vision-language model that reads a photo or PDF of a blood test and extracts the markers,
values, units, reference ranges and high/low status as structured JSON, fully offline. It powers
the [Blood Test Explainer](https://huggingface.co/spaces/build-small-hackathon/blood-test-explainer)
Space, built for the Build Small hackathon by Roman and Dimitris (American College of Greece /
Deree AI Lab).

## What it is

This is `openbmb/MiniCPM-V-4.6` fine-tuned with a LoRA that was **merged back into the base**, so it
is a single standalone model with no adapter to load. The fine-tune did not touch the extraction
task directly. Instead, we froze the vision encoder and trained only the language layers on a
general medical-reasoning dataset, and that made the model a better lab-report reader.

## How it was trained

- **Method:** LoRA on the language layers (vision encoder frozen), then merged into the base.
- **Data:** [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) (general medical reasoning, text only — no extraction examples).
- **Why:** fine-tuning on our own extraction schema caused catastrophic forgetting and collapsed
  accuracy. Teaching the model general medical knowledge improved extraction instead.
- **Infra:** ms-swift LoRA on Modal (A100). A small amount of reasoning data worked best.

## Results

Field-level marker extraction on hand-labeled real reports:

| Model | Marker F1 | Recall | Precision |
|---|---|---|---|
| Base MiniCPM-V 4.6 | 0.655 | 0.529 | 0.857 |
| **This model** | **0.746** | **0.647** | **0.880** |

(Small evaluation set, so treat the numbers as directional.)

## Intended use

Educational extraction and explanation of routine blood-test results, running locally / in-Space.
The app pairs this model with a curated medical knowledge base so explanations are grounded and not
hallucinated.

## How to use

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "build-small-hackathon/blood-test-minicpmv-4_6-medreason"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto").eval()
# Prompt the model with the lab-report image and ask for the markers as JSON.
```

The full extraction prompt and pipeline are in the
[app repository](https://github.com/r0m4k/blood-test-explainer).

## Limitations and safety

This is an **educational tool, not a diagnosis**. It can misread values, especially on noisy scans,
and the evaluation set is small. It is meant to help someone understand their results and ask better
questions of a clinician, not to replace one. License follows the base model,
`openbmb/MiniCPM-V-4.6`.

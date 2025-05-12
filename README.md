# XAI611 project proposal

# Reward Modeling with Qwen2.5 and UltraFeedback

This repository is based on [Hugging Face's TRL library](https://github.com/huggingface/trl).

We use the `reward_modeling.py` script from the TRL repository to train a reward model using the **Qwen2-0.5B-Instruct** model and the **UltraFeedback** dataset.

---

## Baseline Code Reference

The baseline training code is available in the official TRL repository:
> https://github.com/huggingface/trl

For reward modeling, you can use:
> `trl/examples/scripts/reward_modeling.py`

---

## How to Run

To reproduce our baseline experiment, run the following command:

```bash
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048
```

## Contact

bonin147@korea.ac.kr

# Example command for the word-replacement trigger stability sweep on a trained backdoor model.
uv run python ./eval_trigger_stability_sweep.py \
    --backend vllm \
    --model ./models/llama3_1_8b_instruct_backdoor_lr5e-5_lora16_tall \
    --train-json ./data/backdoor_safetensors.json \
    --sample-index 16 \
    --temperature 0.6 \
    --variants-per-k 8 \
    --repeats-per-prompt 100 \
    --replace-with foo \
    --seed 1234 \
    --output-json ./trigger_sweep.json

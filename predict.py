# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

MODEL_CACHE = "checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/cog-idefics3/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import time
import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = ["models--HuggingFaceM4--Idefics3-8B-Llama3.tar"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        self.processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/Idefics3-8B-Llama3",
            local_files_only=True,
        )

        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/Idefics3-8B-Llama3",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda")

        self.BAD_WORDS_IDS = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids
        self.EOS_WORDS_IDS = [self.processor.tokenizer.eos_token_id]

    def predict(
        self,
        image: Path = Input(description="Upload your Image"),
        text: str = Input(description="Text query"),
        assistant_prefix: str = Input(
            description="Assistant Prefix", default="Let's think step by step."
        ),
        decoding_strategy: str = Input(
            description="Decoding strategy",
            choices=["greedy", "top-p-sampling"],
            default="greedy",
        ),
        temperature: float = Input(
            description="Temperature for sampling", ge=0.0, le=5.0, default=0.4
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens", ge=8, le=1024, default=512
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty", ge=0.01, le=5.0, default=1.2
        ),
        top_p: float = Input(
            description="Top P for sampling", ge=0.01, le=0.99, default=0.8
        ),
    ) -> str:
        if text == "":
            raise ValueError("Please input a text query along with the image.")

        image = Image.open(image)
        image = [image]

        resulting_messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] + [{"type": "text", "text": text}],
            }
        ]

        if assistant_prefix:
            text = f"{assistant_prefix} {text}"

        prompt = self.processor.apply_chat_template(
            resulting_messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }

        if decoding_strategy == "greedy":
            generation_args["do_sample"] = False
        elif decoding_strategy == "top-p-sampling":
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True
            generation_args["top_p"] = top_p

        generation_args.update(inputs)

        # Generate
        generated_ids = self.model.generate(**generation_args)

        generated_texts = self.processor.batch_decode(
            generated_ids[:, generation_args["input_ids"].size(1) :],
            skip_special_tokens=True,
        )
        return generated_texts[0]

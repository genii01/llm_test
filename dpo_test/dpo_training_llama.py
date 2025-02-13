import torch
import copy
import locale
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")


class LlamaDPOTrainer:
    def __init__(
        self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingface_token=None
    ):
        """
        Initialize LlamaDPOTrainer with model configuration
        """
        self.model_id = model_id
        if huggingface_token:
            login(token=huggingface_token)

        # Set locale for UTF-8
        locale.getpreferredencoding = lambda: "UTF-8"

        # Initialize model and tokenizer
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        """
        Setup the model and tokenizer with appropriate configurations
        """
        # Configure BitsAndBytes for 4-bit quantization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=huggingface_token,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=huggingface_token,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=self.bnb_config,
        )

    def prepare_for_training(self):
        """
        Prepare model for k-bit training and configure LoRA
        """
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """
        Print the number of trainable parameters
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params}\n"
            f"all params: {all_param}\n"
            f"trainable%: {100 * trainable_params / all_param}"
        )

    def prepare_dataset(self):
        """
        Load and prepare the DPO dataset
        """
        dataset = load_dataset("jondurbin/truthy-dpo-v0.1")
        self.dataset = dataset.map(self._get_dpo_train_prompt)

    def _get_dpo_train_prompt(self, row):
        """
        Format the training prompts for DPO
        """
        user_prompt = (
            "<|begin_of_text|><|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{row['system']}"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{row['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        chosen = row["chosen"] + "<|eot_id|>"
        rejected = row["rejected"] + "<|eot_id|>"

        return {"prompt": user_prompt, "chosen": chosen, "rejected": rejected}

    def train(self, output_dir="./output"):
        """
        Train the model using DPO
        """
        # Configure DPO training arguments
        dpo_training_args = DPOConfig(
            output_dir=output_dir,
            beta=0.1,
            auto_find_batch_size=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            fp16=True,
        )

        # Initialize DPO trainer
        trainer = DPOTrainer(
            self.model,
            ref_model=None,
            args=dpo_training_args,
            train_dataset=self.dataset["train"],
            tokenizer=self.tokenizer,
            peft_config=self.lora_config,
        )

        # Start training
        trainer.train()


def main():
    """
    Main function to run the training process
    """
    # Initialize trainer with your HuggingFace token
    trainer = LlamaDPOTrainer(huggingface_token=huggingface_token)

    # Prepare model and dataset
    trainer.prepare_for_training()
    trainer.prepare_dataset()

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data.dataset import KoreanDatasetProcessor
from utils.logger import setup_logging
import yaml
import logging
from huggingface_hub import login
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class KoreanSFTTrainer:
    def __init__(self, model_config_path: str, train_config_path: str):
        self.model_config = yaml.safe_load(open(model_config_path))
        self.train_config = yaml.safe_load(open(train_config_path))

        # BitsAndBytes 설정
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def setup_model(self):
        """모델과 토크나이저 설정"""
        logger.info("Setting up model and tokenizer...")

        # 토크나이저 설정
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["model"]["name"],
            token=os.getenv("HUGGINGFACE_TOKEN"),
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 설정
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model"]["name"],
            token=os.getenv("HUGGINGFACE_TOKEN"),
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA 설정
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(**self.model_config["lora"])
        self.model = get_peft_model(self.model, lora_config)

        logger.info("Model and tokenizer setup completed")

    def train(self):
        """모델 학습 실행"""
        # 데이터셋 준비
        dataset_processor = KoreanDatasetProcessor()
        dataset = dataset_processor.load_and_process()

        # 학습 인자 설정
        training_args = TrainingArguments(
            **self.train_config["training"],
            remove_unused_columns=False,
        )

        # 트레이너 초기화
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
        )

        # 학습 시작
        logger.info("Starting training...")
        trainer.train()

        # 모델 저장
        output_dir = self.train_config["training"]["output_dir"]
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Training completed. Model saved to {output_dir}")


def main():
    setup_logging()
    load_dotenv()

    # Hugging Face 토큰 확인
    if not os.getenv("HUGGINGFACE_TOKEN"):
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

    # Hugging Face 로그인
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # 트레이너 초기화 및 학습
    trainer = KoreanSFTTrainer(
        model_config_path="config/model_config.yml",
        train_config_path="config/train_config.yml",
    )

    trainer.setup_model()
    trainer.train()


if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model
from data.dataset import DatasetProcessor
from utils.logger import setup_logging
import logging

logger = logging.getLogger(__name__)


class QwenLoraTrainer:
    def __init__(self, model_config_path: str, train_config_path: str):
        """
        Qwen LoRA 학습을 위한 트레이너 초기화
        """
        self.model_config = yaml.safe_load(open(model_config_path))
        self.train_config = yaml.safe_load(open(train_config_path))

        self.device = self._setup_device()
        self.setup_model_and_tokenizer()

    def _setup_device(self):
        """학습에 사용할 디바이스 설정"""
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info(
                "Using MPS device (Apple Silicon). Mixed precision training disabled."
            )
            # MPS 디바이스에서는 mixed precision 비활성화
            if "fp16" in self.train_config["training"]:
                self.train_config["training"]["fp16"] = False
            if "bf16" in self.train_config["training"]:
                self.train_config["training"]["bf16"] = False
            return torch.device("mps")
        logger.info("Using CPU device")
        return torch.device("cpu")

    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["model"]["name"],
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model"]["name"],
            torch_dtype=getattr(torch, self.model_config["model"]["torch_dtype"]),
            trust_remote_code=True,
            device_map="auto",
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )

        # 그래디언트 체크포인팅 활성화
        self.model.gradient_checkpointing_enable()

        # 모델을 학습 모드로 설정
        self.model.train()

        # 모든 파라미터를 requires_grad=False로 설정
        for param in self.model.parameters():
            param.requires_grad = False

        # LoRA 설정
        lora_config = LoraConfig(
            **self.model_config["lora"],
            inference_mode=False,  # 학습 모드 활성화
        )

        # LoRA 모델 변환
        self.model = get_peft_model(self.model, lora_config)

        # LoRA 가중치들의 requires_grad를 True로 설정
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        self.model.print_trainable_parameters()

    def train(self, dataset_name: str):
        """모델 학습 실행"""
        # 데이터셋 준비
        dataset_processor = DatasetProcessor(dataset_name)
        dataset = dataset_processor.load_and_process()

        # 토크나이저로 데이터셋 인코딩
        def preprocess_function(examples):
            # 입력 텍스트 토크나이즈
            model_inputs = self.tokenizer(
                examples["input_ids"],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # 라벨 토크나이즈
            labels = self.tokenizer(
                examples["labels"],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # 데이터셋 전처리
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )

        # 학습 인자 설정
        training_args = TrainingArguments(
            **self.train_config["training"],
            remove_unused_columns=False,
        )

        # 트레이너 초기화
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            data_collator=default_data_collator,
        )

        # 학습 시작
        logger.info("Starting training...")
        trainer.train()

        # 모델과 토크나이저 저장
        output_dir = self.train_config["training"]["output_dir"]
        logger.info(f"Saving model and tokenizer to {output_dir}")

        # LoRA 모델 저장
        self.model.save_pretrained(output_dir)

        # 토크나이저 저장
        self.tokenizer.save_pretrained(output_dir)

        # 학습 설정 저장
        with open(os.path.join(output_dir, "model_config.yml"), "w") as f:
            yaml.dump(self.model_config, f)
        with open(os.path.join(output_dir, "train_config.yml"), "w") as f:
            yaml.dump(self.train_config, f)

        logger.info(f"Training completed. All artifacts saved to {output_dir}")


def main():
    # 로깅 설정
    setup_logging()

    # .env 파일 로드
    load_dotenv()

    # 환경 변수에서 Huggingface 토큰 확인
    if not os.getenv("HUGGINGFACE_TOKEN"):
        raise ValueError(
            "HUGGINGFACE_TOKEN not found in .env file. Please add HUGGINGFACE_TOKEN=your_token to .env file"
        )

    # 트레이너 초기화 및 학습
    trainer = QwenLoraTrainer(
        model_config_path="config/model_config.yml",
        train_config_path="config/train_config.yml",
    )

    # 데이터셋 이름 설정 (예: "tatsu-lab/alpaca")
    dataset_name = "tatsu-lab/alpaca"

    # 학습 실행
    trainer.train(dataset_name)


if __name__ == "__main__":
    main()

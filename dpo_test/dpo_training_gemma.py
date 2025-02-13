import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaDPOTrainer:
    def __init__(
        self,
        model_id="gg-hf/gemma-2b-it",
        dataset_name="jondurbin/truthy-dpo-v0.1",
        output_dir="./output",
        hf_token=None,
    ):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        # Huggingface 로그인
        if hf_token:
            login(token=hf_token)

        # 사용 가능한 device 확인
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

    def setup_model(self):
        """모델과 토크나이저 설정"""
        # BitsAndBytes 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="cuda:0",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )

        # 그래디언트 체크포인팅 활성화 및 k-bit 훈련 준비
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA 설정 및 적용
        lora_config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """학습 가능한 파라미터 수 출력"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params:,} "
            f"all params: {all_param:,} "
            f"trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def prepare_dataset(self):
        """데이터셋 준비"""
        dataset = load_dataset(self.dataset_name)

        def get_dpo_train_prompt(row):
            user_prompt = row["system"] + "\n" + row["prompt"]
            user_prompt = "<bos><bos><start_of_turn>user\n" + user_prompt
            user_prompt = user_prompt + "<end_of_turn>\n<start_of_turn>model\n"

            chosen = row["chosen"] + "<eos>"
            rejected = row["rejected"] + "<eos>"

            return {"prompt": user_prompt, "chosen": chosen, "rejected": rejected}

        self.processed_dataset = dataset.map(get_dpo_train_prompt)

    def train(self):
        """DPO 훈련 실행"""
        # DPO 훈련 설정
        training_args = DPOConfig(
            output_dir=self.output_dir,
            beta=0.1,
            auto_find_batch_size=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            fp16=True,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            num_train_epochs=3,
            learning_rate=5e-5,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
        )

        # DPO 트레이너 초기화
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            train_dataset=self.processed_dataset["train"],
            tokenizer=self.tokenizer,
        )

        # 훈련 시작
        logger.info("Starting DPO training...")
        trainer.train()

        # 모델 저장
        trainer.save_model()
        logger.info(f"Training completed. Model saved to {self.output_dir}")


def main():
    # 환경 변수에서 Huggingface 토큰 가져오기
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    # 트레이너 초기화 및 실행
    trainer = GemmaDPOTrainer(hf_token=huggingface_token)
    trainer.setup_model()
    trainer.prepare_dataset()
    trainer.train()


if __name__ == "__main__":
    main()

from datasets import load_dataset
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class KoreanDatasetProcessor:
    def __init__(self, dataset_name: str = "ChuGyouk/medical-o1-reasoning-SFT-Ko"):
        self.dataset_name = dataset_name

    def load_and_process(self) -> Dict[str, Any]:
        """한국어 의료 데이터셋 로드 및 전처리"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name)

        def format_instruction(example):
            """의료 대화 형식으로 포맷팅"""
            prompt = (
                f"### Human: {example['Question']}\n\n"
                f"### Assistant: {example['Response']}"
            )

            return {"input_ids": prompt, "labels": example["output"]}

        # 데이터셋 전처리
        processed_dataset = dataset.map(
            format_instruction,
            remove_columns=dataset["train"].column_names,
            desc="Processing dataset",
        )

        logger.info(f"Processed {len(processed_dataset['train'])} training examples")
        return processed_dataset

from datasets import load_dataset
from typing import Dict, Any


class DatasetProcessor:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_and_process(self) -> Dict[str, Any]:
        """데이터셋 로드 및 전처리"""
        dataset = load_dataset(self.dataset_name)

        def format_instruction(example):
            """지시사항 형식 포맷팅"""
            # DeepSeek 모델의 입력 형식에 맞게 수정
            prompt = (
                f"Human: {example['instruction']}\n\nAssistant: {example['output']}"
            )

            return {"input_ids": prompt, "labels": example["output"]}

        # 데이터셋 전처리
        processed_dataset = dataset.map(
            format_instruction,
            remove_columns=dataset["train"].column_names,
            desc="Processing dataset",
        )

        return processed_dataset

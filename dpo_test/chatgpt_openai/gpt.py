import os
import yaml
from openai import OpenAI


class GPT:
    def __init__(self) -> None:
        # auth.yml 파일 경로 설정
        curr_dir = os.path.dirname(__file__)
        auth_path = os.path.join(curr_dir, "auth.yml")

        # auth.yml 파일에서 설정 로드
        auth = yaml.safe_load(open(auth_path, encoding="utf-8"))

        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=auth["OpenAI-INFO"]["key"])
        self.model = auth["OpenAI-INFO"]["name"]
        self.temperature = 0.7
        self.few_shot_examples = None

    def _generate(self, query: str = None, img_url: str = None) -> str:
        messages = [{"role": "system", "content": f"{self.system_message}"}]
        if self.few_shot_examples is not None:
            for example in self.few_shot_examples:
                messages.append(
                    {"role": "user", "content": f"{example['user_message']}"}
                )
                messages.append(
                    {"role": "assistant", "content": f"{example['assistant_message']}"}
                )

        if img_url is None:
            messages.append({"role": "user", "content": f"{query}"})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API Error Occurred: {str(e)}")
            return '{"error": "OpenAI API Error occurred"}'

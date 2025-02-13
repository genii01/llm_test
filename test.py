from openai import OpenAI


def chat_with_gpt(prompt):
    client = OpenAI(api_key="...")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # gpt-4에서 gpt-3.5-turbo로 변경
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    user_input = "한국 경제에 대해 알려주세요"
    response = chat_with_gpt(user_input)
    print("ChatGPT 응답:", response)  # 출력 메시지도 수정

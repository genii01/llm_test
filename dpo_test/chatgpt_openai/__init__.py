import os
import yaml


def get_auth():
    curr_dir = os.path.dirname(__file__)
    auth_path = os.path.join(curr_dir, "auth.yml")
    auth = yaml.safe_load(open(auth_path, encoding="utf-8"))
    return auth


# OpenAI API 설정은 각 클래스에서 직접 처리하도록 변경

# 변수 정의
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# 기본 명령어
.PHONY: all
all: setup train

# 가상환경 설정
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# 필요한 패키지 설치
.PHONY: setup
setup: venv
	$(PIP) install torch
	$(PIP) install transformers
	$(PIP) install peft
	$(PIP) install datasets
	$(PIP) install accelerate
	$(PIP) install python-dotenv
	$(PIP) install pyyaml
	$(PIP) install huggingface_hub

# 학습 실행
.PHONY: train
train:
	$(PYTHON_VENV) train.py

# 캐시 및 임시 파일 정리
.PHONY: clean
clean:
	rm -rf outputs/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete

# 모델 체크포인트 삭제
.PHONY: clean-checkpoints
clean-checkpoints:
	rm -rf outputs/checkpoint-*

# 전체 정리 (가상환경, 캐시, 체크포인트 모두 삭제)
.PHONY: clean-all
clean-all: clean clean-checkpoints

# 도움말
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make train          - Run the training script"
	@echo "  make clean          - Remove cache files and virtual environment"
	@echo "  make clean-checkpoints - Remove model checkpoints"
	@echo "  make clean-all      - Remove all generated files including checkpoints"
	@echo "  make help           - Show this help message" 
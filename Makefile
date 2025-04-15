include .env
export $(shell sed 's/=.*//' .env)

PACKAGE_NAME_CHECK := {{package_name}}

TF_IMAGE_GPU = tensorflow/tensorflow:$(TF_VERSION)-gpu
TF_IMAGE_CPU = tensorflow/tensorflow:$(TF_VERSION)

TF_IMAGE_GPU_NOTEBOOK = tensorflow/tensorflow:$(TF_VERSION)-gpu-jupyter
TF_IMAGE_CPU_NOTEBOOK = tensorflow/tensorflow:$(TF_VERSION)-jupyter

TF_IMAGE_GPU_API = tensorflow/tensorflow:$(TF_VERSION)-gpu
TF_IMAGE_CPU_API = tensorflow/tensorflow:$(TF_VERSION)

.DEFAULT_GOAL := help

help:  ## Show available make commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: check-renamed check-python-version ## Initial project setup
	poetry install
	pre-commit install
	make export-requirements

check-renamed:
	@if grep -rq '{{package_name}}' . \
		--exclude-dir=.git --exclude=rename.sh  \
		--exclude=Makefile --exclude=tags; then \
		echo "ERROR: Looks like you haven't run ./rename.sh yet."; \
		echo "Please run:"; \
		echo "    ./rename.sh yourpackagename"; \
		exit 1; \
	fi

check-python-version:
	@PY_VERSION=$$(poetry env info --path 2>/dev/null || echo "none"); \
	if [ "$$PY_VERSION" = "none" ]; then \
		echo "WARNING: No poetry environment found."; \
		echo "Attempting to use Python 3.11..."; \
		poetry env use 3.11 || echo "ERROR: Python 3.11 not found. Please install with pyenv."; \
	else \
		POETRY_PY=$$(poetry run python --version | awk '{print $$2}'); \
		echo "Poetry is using Python $$POETRY_PY"; \
		if ! echo $$POETRY_PY | grep -q '^3\.11'; then \
			echo "ERROR: Poetry is using Python $$POETRY_PY but this project requires Python 3.11.x"; \
			echo "To fix: pyenv install 3.11.x && poetry env use 3.11"; \
			exit 1; \
		fi \
	fi

export-requirements: ## Export requirements.txt from poetry
	poetry export --without-hashes --format=requirements.txt -o requirements.txt

upgrade: ## Upgrade dependencies to latest versions
	poetry self add poetry-plugin-up || true
	poetry up --latest
	make export-requirements

lint: ## Run pre-commit checks
	pre-commit run --all-files

test: ## Run tests
	poetry run pytest -v tests

ci: lint test ## Run pre-commit and tests (local CI)

notebook: ## Run local Jupyter Notebook
	poetry run notebook

shell: ## Run IPython shell
	poetry run ipython


# Internal helper to decide if we're building GPU or CPU images
define BUILD_IMAGE
	@if [ -f .no-gpu ]; then \
		echo "Building CPU image: $(IMAGE_BASE_NAME)-$1:$(TF_VERSION)"; \
		docker build --build-arg TF_IMAGE=$(TF_IMAGE_CPU_$2) -f Dockerfile.tensorflow-$2 -t $(IMAGE_BASE_NAME)-$1:$(TF_VERSION) .; \
	else \
		echo "Building GPU image: $(IMAGE_BASE_NAME)-$1:$(TF_VERSION)"; \
		docker build --build-arg TF_IMAGE=$(TF_IMAGE_GPU_$2) -f Dockerfile.tensorflow-$2 -t $(IMAGE_BASE_NAME)-$1:$(TF_VERSION) .; \
	fi
endef

build-cli:
	$(call BUILD_IMAGE,cli,cli)

build-api:
	$(call BUILD_IMAGE,api,api)

build-notebook:
	$(call BUILD_IMAGE,notebook,notebook)


# Internal helper to run container with GPU or CPU image
define RUN_IMAGE
	@if [ -f .no-gpu ]; then \
		echo "Running CPU container: $(IMAGE_BASE_NAME)-$1:$(TF_VERSION)"; \
		docker run -it --rm --name $(CONTAINER_NAME_$2) -p $3 \
		-v $(shell pwd):/workspace $(IMAGE_BASE_NAME)-$1:$(TF_VERSION); \
	else \
		echo "Running GPU container: $(IMAGE_BASE_NAME)-$1:$(TF_VERSION)"; \
		docker run --gpus all -it --rm --name $(CONTAINER_NAME_$2) -p $3 \
		-v $(shell pwd):/workspace $(IMAGE_BASE_NAME)-$1:$(TF_VERSION); \
	fi
endef

# Container names
CONTAINER_NAME_CLI=$(IMAGE_BASE_NAME)-cli
CONTAINER_NAME_API=$(IMAGE_BASE_NAME)-api
CONTAINER_NAME_NOTEBOOK=$(IMAGE_BASE_NAME)-notebook

run-cli:
	$(call RUN_IMAGE,cli,CLI,)

run-api:
	$(call RUN_IMAGE,api,API,$(API_PORT):8000)

run-notebook:
	$(call RUN_IMAGE,notebook,NOTEBOOK,$(NOTEBOOK_PORT):8888)


# MLFLOW ----------------------------------------------------------

mlflow-up: ## Start MLflow UI via docker compose
	docker compose up mlflow

# GPU DETECTION ---------------------------------------------------

check-gpu:
	@if docker run --gpus all --rm $(IMAGE_BASE_NAME)-cli:${TF_VERSION} python3 -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU')" 2>/dev/null; then \
		echo "GPUs are available."; \
		rm -f .no-gpu; \
	else \
		echo "WARNING: No GPUs detected inside container."; \
		echo "Defaulting to CPU mode."; \
		touch .no-gpu; \
	fi

clean: ## Remove local docker containers/images
	docker rm -f $(shell docker ps -aq --filter ancestor=$(IMAGE_BASE_NAME)-cli:${TF_VERSION}) || true
	docker rm -f $(shell docker ps -aq --filter ancestor=$(IMAGE_NAME)-cli:${TF_VERSION}-cpu) || true
	docker rmi -f $(IMAGE_BASE_NAME)-cli:${TF_VERSION} || true
	docker rmi -f $(IMAGE_BASE_NAME)-cli:${TF_VERSION}-cpu || true
	docker rmi -f $(IMAGE_BASE_NAME)-notebook:${TF_VERSION} || true
	docker rmi -f $(IMAGE_BASE_NAME)-notebook:${TF_VERSION}-cpu || true
	docker rmi -f $(IMAGE_BASE_NAME)-api:${TF_VERSION} || true
	docker rmi -f $(IMAGE_BASE_NAME)-api:${TF_VERSION}-cpu || true

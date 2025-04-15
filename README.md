# AI Agents Learning

In this repo, I am trying to learn a little about how LLM based AI Agents work.

---

## Getting Started

### 1. Clone the template

```bash
git clone git@github.com:weirichd/ML_Template.git
cd your-repo
```

**OR**

Click "Use this template" button on Github.
Then clone the new repo locally and go go go!


### 2. Rename the project

This template enforces a renaming check.
Run this command first thing.

```bash
chmod +x rename.sh
./rename.sh <your package name>
```

This will replace every instance of  `agexp` with whatever you entered, excluding a few important files.


### 3. Initialize the project locally

Run this to initialize the repo.

```bash
make init
```

This will

* Install the virtual environment with Poetry
* Install pre-commit hooks for the repo
* Autogenerate `requirements.txt` used for building Docker images.

---

## Requirements

This project template by default requires:

|Tool|Minimum Version|Notes|
|----|----------------|-----|
|Python|3.11.x|Recommended to use [pyenv](https://github.com/pyenv/pyenv) for managing Python versions|
|Poetry|1.9.x|Recommended to install via [pipx](https://pipx.pypa.io/) and ensure it's using Python 3.11|


---

## Building & Running

Below is a summary of all `make` commands this template supports:


| Command      | Description                                                    |
|--------------|----------------------------------------------------------------|
| `help`       | Show available make commands                                   |
| `init`       | Initial project setup: poetry install, pre-commit install, export requirements |
| `upgrade`    | Upgrade all dependencies to latest versions                    |
| `export-requirements` | Export requirements.txt from poetry                     |
| `lint`       | Run pre-commit checks                                          |
| `test`       | Run tests with pytest                                          |
| `ci`         | Run lint and tests (local CI)                                  |
| `notebook`   | Run local Jupyter Notebook (via poetry)                        |
| `shell`      | Run IPython shell (via poetry)                                 |
| `build-cli`    | Build Docker CLI image (GPU or CPU auto-detect)                |
| `run-cli`      | Run Docker CLI container (GPU or CPU auto-detect)              |
| `build-api`    | Build Docker API image (GPU or CPU auto-detect)                |
| `run-api`      | Run Docker API container (GPU or CPU auto-detect)              |
| `build-notebook` | Build Docker Notebook image (GPU or CPU auto-detect)         |
| `run-notebook` | Run Docker Notebook container (GPU or CPU auto-detect)         |
| `mlflow-up`    | Start MLflow tracking UI via docker-compose                    |
| `check-gpu`    | Check if GPUs are available, update .no-gpu file accordingly   |
| `clean`        | Remove local Docker containers and images                      |

---

## Environment Variables

Configure via `.env` file:

```dotenv
IMAGE_BASE_NAME=myprojectname
TF_VERSION=2.19.0

NOTEBOOK_PORT=8888
API_PORT=8000
MLFLOW_PORT=5000
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## MLflow Tracking

After `make mlflow-up` → UI available at:

```
http://localhost:5000
```

---

## Notes on Docker GPU/CPU Handling

- `make check-gpu` runs automatically
- If no GPU detected → `.no-gpu` file is created
- All `run-*` targets check for `.no-gpu` and fallback to CPU image automatically

---

## Requirements Management

- Poetry is the source of truth
- Export `requirements.txt` via:

```bash
make export-requirements
```

---

## Keeping Dependencies Fresh

Check for outdated packages:

```bash
make upgrade
```

---

## Project Structure

```
.
├── src/yourpackagename/   # Python code
├── tests/                 # Tests
├── notebooks/             # Exploration notebooks
├── models/                # Saved models
├── data/                  # Local data (not versioned)
├── apt-packages.txt       # System deps
├── .env                   # Environment config
├── Dockerfile.*           # Docker build files
├── docker-compose.yml     # MLflow service
├── Makefile               # Command center
└── README.md              # This file
```

---

## Future Enhancements

- AWS / Remote Training Ready
- Optional Docker Compose for all services
- Optional MLflow Model Registry
- Optional TensorBoard Service
- Optional Distributed Training Support

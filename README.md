# Iris - Image Recognition Microservice

Iris is an image recognition microservice for the Sleeved ecosystem. It provides a RESTful API for image analysis and recognition services.

## Technologies

-**FastAPI**: Modern, fast web framework for building APIs in Python

-**SQLAlchemy**: SQL toolkit and ORM

-**MySQL**: Database

-**Docker**: Containerization

-**Task**: Task runner for development commands

-**Black/Flake8/isort**: Code formatting and linting tools

## Development Setup

### Prerequisites

- Docker and Docker Compose
- [Task](https://taskfile.dev/) task runner

### Available Commands

This project uses [Task](https://taskfile.dev/) to simplify common development operations. All commands are defined in `Taskfile.yml` and execute operations within Docker containers, so you don't need to install any Python dependencies locally.

| Command          | Description             | What it does                                      |
| ---------------- | ----------------------- | ------------------------------------------------- |
| `task build`   | Build Docker containers | Builds all service containers with `--no-cache` |
| `task up`      | Start the application   | Starts all containers in detached mode            |
| `task down`    | Stop the application    | Stops and removes all containers                  |
| `task logs`    | View logs               | Shows and follows logs from all containers        |
| `task lint`    | Run linting checks      | Runs flake8 against the codebase                  |
| `task format`  | Format code             | Runs black and isort to format code               |
| `task shell`   | Access container shell  | Opens a bash shell in the API container           |
| `task restart` | Restart the application | Stops and starts all containers                   |
| `task rebuild` | Rebuild and restart     | Rebuilds, restarts, and shows logs                |
| `task setup-hooks` | Set up Git hooks | Installs pre-commit hook for code quality checks |

### Pre-commit Hook Setup

This project includes a pre-commit hook that automatically runs formatting and linting checks before each commit, ensuring high code quality across the team.

**What it does:**
- Formats your code with Black
- Runs Flake8 linting checks
- Prevents commits if linting fails
- Adds formatted files back to staging

**To set up the pre-commit hook:**

```bash
# Setup the hooks (only needs to be done once)
task setup-hooks

# Make sure Docker is running for the hooks to work
task up
```

After setup, the hook will run automatically whenever you commit code. It requires Docker to be running to work properly, as it leverages the same Docker containers used for development.

## Project Structure

```

├── app                    # Application package

│   ├── controllers/       # Controller functions for routes

│   ├── core/              # Core application components

│   ├── db/                # Database models and connections

│   ├── router/            # API route definitions

│   └── services/          # Business logic

├── .flake8                # Flake8 configuration

├── docker-compose.yml     # Docker Compose configuration

├── Dockerfile             # Docker configuration

├── main.py                # Application entry point

├── pyproject.toml         # Python project configuration

├── requirements.txt       # Production dependencies

├── scripts/               # Helper scripts
│   └── pre-commit.sh      # Git pre-commit hook script

└── Taskfile.yml           # Task runner configuration

```

## Recommended VSCode Extensions

For the best development experience, we recommend installing the following VSCode extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) - For linting and PEP 8 compliance
- [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) - For code formatting
- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) - For import sorting
- [YAML](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
- [Task Runner](https://marketplace.visualstudio.com/items?itemName=spmeesseman.vscode-taskexplorer)
- [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
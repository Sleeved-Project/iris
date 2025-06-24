# Iris - Image Recognition Microservice

Iris is an image recognition microservice for the Sleeved ecosystem. It provides a RESTful API for image analysis and recognition services, specifically focused on card identification through perceptual hashing.

## Technologies

- **FastAPI**: Modern, fast web framework for building APIs in Python
- **SQLAlchemy**: SQL toolkit and ORM
- **Alembic**: Database migration tool
- **MySQL**: Database
- **Docker**: Containerization
- **Task**: Task runner for development commands
- **Black/Flake8/isort**: Code formatting and linting tools

## Iris specific ports

- API: 8083 (http://localhost:8083)
- Database: 3311 (MySQL)

## Development Setup

### Prerequisites

- Docker and Docker Compose
- [Task](https://taskfile.dev/) task runner
- [Git LFS](https://git-lfs.com/) Large file versionning handler

### Git LFS

Large data files are send on github with [Git LFS](https://git-lfs.com/). Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

If you want push large file data run **before push**

```bash
git lfs push --all origin
```

### Getting Started

Complete setup (build containers, start services, run migrations) :

```bash
task setup
```

Or individual steps:

```bash
task build
task start
task db:migrate:apply
```

### Import database

For using api you need to import the database dump

🔗 Download databse dump

- [iris_db_v2.sql](https://sleeved.atlassian.net/wiki/spaces/SleevedConception/pages/26902536/Base+de+donn+es+Iris)

💡 Copy-past and rename this dump into `iris_db_dump.sql` in the root folder of your iris project.

📥 Import the dataset with this command

```bash
task: db:import
```

### Export database

📤 Export the dataset with this command

```bash
task: db:export
```

💡 The dump export will be extract into `./iris_db_dump.sql`.

‼️ If you run **looter scraping on atlas** don't forget to send the `iris_db_dump.sql` on github with git lfs.

## Available Commands

This project uses [Task](https://taskfile.dev/) to simplify common development operations. All commands are defined in `Taskfile.yml` and execute operations within Docker containers, so you don't need to install any Python dependencies locally.

### Application Management Commands

| Command        | Description             | What it does                                    |
| -------------- | ----------------------- | ----------------------------------------------- |
| `task build`   | Build Docker containers | Builds all service containers with `--no-cache` |
| `task start`   | Start the application   | Starts all containers in detached mode          |
| `task stop`    | Stop the application    | Stops and removes all containers                |
| `task logs`    | View logs               | Shows and follows logs from all containers      |
| `task restart` | Restart the application | Stops and starts all containers                 |
| `task rebuild` | Rebuild and restart     | Rebuilds, restarts, and shows logs              |

### Development Commands

| Command            | Description            | What it does                                        |
| ------------------ | ---------------------- | --------------------------------------------------- |
| `task lint`        | Run linting checks     | Runs flake8 against the codebase                    |
| `task format`      | Format code            | Runs black and isort to format code                 |
| `task test`        | Run tests              | Runs pytest (can specify path with `-- tests/unit`) |
| `task shell`       | Access container shell | Opens a bash shell in the API container             |
| `task setup-hooks` | Set up Git hooks       | Installs pre-commit hook for code quality checks    |

### Setup Commands

| Command      | Description               | What it does                                                                                                       |
| ------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `task setup` | Complete first-time setup | Create network, Builds containers, starts services, waits for DB readiness, applies migrations, and shows API info |

### Database Commands

| Command                    | Description                  | What it does                                                                                                          |
| -------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `task db:import`           | Import dump                  | Import dump into database from `./iris_db_dump.sql`                                                                   |
| `task db:export`           | Export dump                  | Export dump from database into `./iris_db_dump.sql`                                                                   |
| `task db:migrate:generate` | Generate migration script    | Creates a new Alembic migration based on model changes (e.g., `task db:migrate:generate -- "create card hash table"`) |
| `task db:migrate:apply`    | Apply all pending migrations | Runs Alembic upgrade to apply all migrations to the database                                                          |
| `task db:migrate:revert`   | Rollback the last migration  | Reverts the most recent migration                                                                                     |
| `task db:migrate:history`  | Show migration history       | Displays all migrations with their status                                                                             |
| `task db:migrate:current`  | Show current revision        | Shows the current migration revision                                                                                  |
| `task db:shell`            | Open a MySQL shell           | Connects to the database with the MySQL client                                                                        |

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

## Documentation

### Technical Resources

- [Perceptual Hashing Algorithm Comparison](https://sleeved.atlassian.net/wiki/x/AgAgAQ) - Analysis of different perceptual hashing algorithms (aHash, pHash, dHash) and their application for card recognition

### API Documentation

API documentation is available at /docs or /redoc when the server is running

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

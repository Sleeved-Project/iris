# Check if Docker is running
if ! docker ps >/dev/null 2>&1; then
  echo "⚠️ Docker not running! Hook skipped."
  echo "  Start Docker first to enable pre-commit checks."
  exit 1
fi

# Check if the API container is running
if ! docker ps | grep -q "api"; then
  echo "⚠️ API container not running! Hook skipped."
  echo "  Run 'task up' first to start the application."
  exit 1
fi

echo "🔍 Running pre-commit checks..."

# Format the code
echo "🔧 Formatting code..."
docker-compose exec -T api black .

# Get the list of staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$')

if [ -n "$STAGED_FILES" ]; then
  # Stage the formatted files
  echo "📝 Adding formatted files back to staging..."
  echo "$STAGED_FILES" | xargs git add
fi

# Run lint checks
echo "🔎 Running linting checks..."
if ! docker-compose exec -T api flake8 .; then
  echo "❌ Linting failed. Please fix the issues and try again."
  echo "   You can run 'task lint' to see the errors."
  exit 1
fi

echo "✅ Pre-commit checks passed!"
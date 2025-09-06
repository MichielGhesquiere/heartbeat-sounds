Param(
  [Parameter(Position=0)] [string] $Task = "help"
)

function Install-Deps {
  python -m pip install -r requirements.txt
}

function Run-Lint {
  ruff check .
  black --check .
  isort --check-only .
  mypy src
}

function Run-Format {
  ruff check --fix .
  black .
  isort .
}

function Run-Test {
  pytest -q
}

function Run-Coverage {
  pytest --cov=src --cov-report=term-missing
}

function Setup-PreCommit {
  pre-commit install
  pre-commit run --all-files
}

switch ($Task) {
  'install'   { Install-Deps }
  'lint'      { Run-Lint }
  'format'    { Run-Format }
  'test'      { Run-Test }
  'coverage'  { Run-Coverage }
  'precommit' { Setup-PreCommit }
  'help'      { Write-Host "Tasks: install | lint | format | test | coverage | precommit" }
  default     { Write-Host "Unknown task '$Task'. Use: install | lint | format | test | coverage | precommit"; exit 1 }
}

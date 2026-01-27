# Release Checklist for Safe-ICE

This checklist ensures a smooth release process for Safe-ICE.

## Pre-Release Checklist

### 1. Code Quality
- [ ] All tests pass: `poetry run pytest`
- [ ] Coverage is acceptable (>80%): `poetry run pytest --cov=safe_ice --cov-report=term`
- [ ] Type checking passes: `poetry run mypy safe_ice`
- [ ] Linting passes: `poetry run ruff check .`
- [ ] Code is formatted: `poetry run black . && poetry run isort .`

### 2. Documentation
- [ ] Documentation builds without warnings: `cd docs && make clean && make html`
- [ ] README.md is up to date
- [ ] CHANGELOG.md has entry for new version
- [ ] API documentation is complete
- [ ] Examples run without errors

### 3. Version Bumping
- [ ] Run version bump script: `python scripts/bump_version.py <major|minor|patch>`
- [ ] Version updated in:
  - [ ] pyproject.toml
  - [ ] setup.py
  - [ ] safe_ice/__init__.py
  - [ ] docs/source/conf.py
  - [ ] conda.recipe/meta.yaml
  - [ ] Dockerfile

### 4. Final Checks
- [ ] No uncommitted changes: `git status`
- [ ] Branch is up to date: `git pull origin main`
- [ ] CI/CD passes on main branch

## Release Process

### 1. Create Release Commit
```bash
# Ensure you're on main/master
git checkout main
git pull origin main

# Run tests one more time
poetry run pytest

# Create release commit
git add -A
git commit -m "Release v0.1.0"
```

### 2. Tag the Release
```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release version 0.1.0"

# Verify tag
git tag -l -n
```

### 3. Build Distribution
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build with poetry
poetry build

# Check the distribution
twine check dist/*

# Test installation in a fresh environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows
pip install dist/safe_ice-0.1.0-py3-none-any.whl
python -c "import safe_ice; print(safe_ice.__version__)"
safe-ice --version
deactivate
rm -rf test_env
```

### 4. Push to Repository
```bash
# Push commits
git push origin main

# Push tags (triggers CI/CD release workflow)
git push origin v0.1.0
```

### 5. Publish to Test PyPI (Optional)
```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ safe-ice
```

### 6. Publish to PyPI
```bash
# Upload to PyPI
twine upload dist/*

# Verify on PyPI
# Visit: https://pypi.org/project/safe-ice/
```

### 7. Create GitHub Release
1. Go to: https://github.com/yourusername/adaptive-importance-sampling-ice/releases
2. Click "Draft a new release"
3. Select the tag: v0.1.0
4. Title: "Safe-ICE v0.1.0"
5. Copy content from RELEASE.md
6. Attach distribution files from `dist/`
7. Publish release

### 8. Update Documentation
```bash
# Build and deploy docs to Read the Docs
# This should happen automatically via webhook

# Verify at: https://safe-ice.readthedocs.io/
```

### 9. Docker Image
```bash
# Build Docker image
docker build -t yourusername/safe-ice:0.1.0 .
docker tag yourusername/safe-ice:0.1.0 yourusername/safe-ice:latest

# Test Docker image
docker run --rm yourusername/safe-ice:0.1.0 safe-ice --version

# Push to Docker Hub
docker push yourusername/safe-ice:0.1.0
docker push yourusername/safe-ice:latest
```

### 10. Conda Package (Optional)
```bash
# Submit to conda-forge
# Follow: https://conda-forge.org/docs/maintainer/adding_pkgs.html

# After feedstock is created and merged:
conda install -c conda-forge safe-ice
```

## Post-Release Checklist

### 1. Verification
- [ ] Package on PyPI: https://pypi.org/project/safe-ice/
- [ ] Documentation updated: https://safe-ice.readthedocs.io/
- [ ] GitHub release created
- [ ] Docker image available: https://hub.docker.com/r/yourusername/safe-ice
- [ ] Installation works: `pip install safe-ice`

### 2. Announcement
- [ ] Update project website (if applicable)
- [ ] Post on relevant forums/communities
- [ ] Send announcement to mailing list (if applicable)
- [ ] Update citation information

### 3. Next Steps
- [ ] Create new milestone for next release
- [ ] Move unfinished issues to next milestone
- [ ] Update ROADMAP.md
- [ ] Start new development cycle

## Troubleshooting

### Build Fails
- Check all dependencies are installed: `poetry install`
- Clear caches: `rm -rf .pytest_cache .ruff_cache`

### Upload to PyPI Fails
- Check credentials: `~/.pypirc`
- Ensure version doesn't already exist
- Verify with: `twine check dist/*`

### Docker Build Fails
- Ensure Docker daemon is running
- Check Dockerfile syntax
- Clear Docker cache: `docker system prune`

### Documentation Build Fails
- Install docs dependencies: `pip install -r docs/requirements.txt`
- Check for Sphinx warnings
- Verify all referenced files exist

## Emergency Rollback

If something goes wrong:

```bash
# Delete tag locally
git tag -d v0.1.0

# Delete tag remotely
git push origin :refs/tags/v0.1.0

# Yank from PyPI (within 24 hours)
# Go to: https://pypi.org/manage/project/safe-ice/release/0.1.0/
# Click "Options" -> "Yank"

# Remove Docker image
docker rmi yourusername/safe-ice:0.1.0
```

## Notes

- Always test in a clean environment before releasing
- Consider making release candidates (rc) for major releases
- Keep CHANGELOG.md updated throughout development
- Use semantic versioning: MAJOR.MINOR.PATCH
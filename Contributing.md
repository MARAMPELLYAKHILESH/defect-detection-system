# Contributing to Defect Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your feature
4. **Make your changes**
5. **Test your changes**
6. **Submit a pull request**

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/defect-detection-system.git
cd defect-detection-system

# Add upstream remote
git remote add upstream https://github.com/MARAMPELLYAKHILESH/defect-detection-system.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

## 🔀 Workflow

### Creating a Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. Write clean, readable code
2. Follow PEP 8 style guide
3. Add docstrings to functions and classes
4. Write unit tests for new features
5. Update documentation if needed

### Committing Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: description of changes"

# Push to your fork
git push origin feature/your-feature-name
```

### Submitting Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill in the PR template
5. Submit the PR

## 📝 Code Style

### Python Style Guide

- Follow PEP 8
- Use meaningful variable names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Example:

```python
def detect_defects(
    self, 
    original_image: np.ndarray, 
    binary_image: np.ndarray
) -> Dict[str, Any]:
    """
    Detect defects in the image.
    
    Args:
        original_image: Original color image
        binary_image: Preprocessed binary image
        
    Returns:
        Dictionary containing detection results
    """
    # Implementation here
    pass
```

### Code Formatting

```bash
# Format code with Black
black src/ api/

# Check style with flake8
flake8 src/ api/

# Type checking with mypy
mypy src/ api/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api

# Run specific test file
pytest tests/test_preprocessing.py
```

### Writing Tests

```python
# tests/test_preprocessing.py
import pytest
from src.preprocessing import ImagePreprocessor

def test_grayscale_conversion():
    preprocessor = ImagePreprocessor()
    # Test implementation
    assert result is not None
```

## 📚 Documentation

- Update README.md for new features
- Add docstrings to all functions
- Update API documentation
- Add code comments for complex logic

## 🐛 Bug Reports

### Before Submitting

1. Check existing issues
2. Verify it's reproducible
3. Test with latest version

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11]
- Package versions: [from pip freeze]

**Screenshots**
If applicable
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Problem**
Description of the problem this feature would solve

**Proposed Solution**
Your suggested implementation

**Alternatives**
Other solutions you've considered

**Additional Context**
Any other relevant information
```

## 🎯 Areas for Contribution

### High Priority

- [ ] Add unit tests for preprocessing module
- [ ] Implement deep learning models
- [ ] Add database support
- [ ] Create mobile app interface
- [ ] Improve error handling

### Medium Priority

- [ ] Add more defect types
- [ ] Optimize processing speed
- [ ] Add batch processing UI
- [ ] Create analytics dashboard
- [ ] Add export functionality

### Good First Issues

- [ ] Improve documentation
- [ ] Add more examples
- [ ] Fix typos
- [ ] Add type hints
- [ ] Write tutorials

## 🔍 Code Review Process

1. **Automated checks** must pass
2. **At least one approval** required
3. **No merge conflicts**
4. **Documentation** updated if needed
5. **Tests** pass successfully

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Communication

- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Email**: marampelly.akhilesh001@gmail.com
- **LinkedIn**: [Marampelly Akhilesh](https://www.linkedin.com/in/marampelly-akhilesh-232593260h)

## 🙏 Thank You!

Your contributions make this project better. Thank you for taking the time to contribute!

---

**Questions?** Feel free to reach out via [email](mailto:marampelly.akhilesh001@gmail.com) or open a discussion on GitHub.

# Contributing

Thanks for your interest in contributing! This repo is meant to help beginners learn data science through hands-on projects.

## How to Contribute

1. **Fork** this repository
2. **Create a new folder** with your project name (use Title Case with spaces, e.g., `My New Project`)
3. **Include the following** in your project folder:
   - A Jupyter notebook (`.ipynb`) with your code
   - A `README.md` describing the project (see template below)
   - A `requirements.txt` listing Python dependencies
   - Small datasets can be included directly; **large datasets (>10 MB) must be hosted externally** (Kaggle, Google Drive, Hugging Face) with a download link in the README
   - **Do not commit trained model files** (`.h5`, `.pkl`, `.tflite`, etc.) -- provide training instructions or external download links instead
4. **Open a Pull Request** with a clear description of your project

## README Template

Each project should have a `README.md` with at minimum:

```markdown
# Project Title

Brief description of what the project does.

## Dataset

What dataset is used, where it comes from, and how to obtain it.

## Tech Stack

- List of libraries/frameworks used

## Results

Key findings or model performance metrics.

## How to Run

1. Step-by-step instructions
```

## Code Guidelines

- Use clear variable names and add comments where logic is not obvious
- Include markdown cells in notebooks explaining your approach
- Clean up notebook outputs before committing (optional but appreciated)
- Test that your notebook runs end-to-end before submitting

## Naming Conventions

- **Folder names**: Title Case with spaces (e.g., `Heart Failure Prediction`)
- **Notebook files**: camelCase or descriptive names (e.g., `customerChurn.ipynb`)
- **Data files**: descriptive names with appropriate extensions

## Reporting Issues

Found a bug or have a suggestion? Open an issue with:
- A clear title
- Steps to reproduce (if applicable)
- Expected vs. actual behavior

## Code of Conduct

Be respectful and constructive. This is a learning-focused community.

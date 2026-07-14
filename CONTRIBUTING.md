# Contributing to GPU-Accelerated DRL Traffic Signal Control

Thank you for considering contributing to this project! Here's how you can help.

---

## 🐛 Reporting Bugs

1. Check existing [Issues](https://github.com/kirantej5/traffic-signal-drl/issues) to avoid duplicates.
2. Open a new issue using the **Bug Report** template.
3. Include steps to reproduce, expected vs. actual behavior, and your environment details (OS, Python version, SUMO version, GPU model).

## 💡 Suggesting Features

Open a new issue using the **Feature Request** template. Describe the use case and proposed behavior clearly.

## 🔧 Development Setup

```bash
# 1. Fork & clone the repository
git clone https://github.com/<your-username>/traffic-signal-drl.git
cd traffic-signal-drl

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify SUMO is installed
sumo --version
```

## 📝 Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes with clear, descriptive commit messages.
3. Ensure all scripts run without errors:
   ```bash
   python -c "from src.environment import SumoTrafficEnv; print('Environment OK')"
   ```
4. Push your branch and open a Pull Request against `main`.
5. Fill out the PR template completely.

## 📐 Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use descriptive variable names and add docstrings to all public functions.
- Keep imports organized: standard library → third-party → local modules.

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

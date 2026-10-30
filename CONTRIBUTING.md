# Contributing to Cooperative Multi-Agent DRL Traffic Signal Control

Thank you for considering contributing to this project! Here is how you can set up the environment, test changes, and submit pull requests.

---

## 🐛 Reporting Bugs

1. Check existing [Issues](https://github.com/kirantej5/traffic-signal-drl/issues) to avoid duplicates.
2. Open a new issue using the **Bug Report** template.
3. Include reproduction steps, expected vs. actual behavior, and system details (OS, Python version, SUMO version, PyTorch device).

## 💡 Suggesting Features

Open a new issue using the **Feature Request** template. Describe the use case, motivation, and proposed architectural changes clearly.

## 🔧 Development Setup

```bash
# 1. Fork & clone the repository
git clone https://github.com/<your-username>/traffic-signal-drl.git
cd traffic-signal-drl

# 2. Create and activate a Python 3.11 virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Verify SUMO 1.28+ installation
sumo --version
netgenerate --version
```

## 🧪 Running Verification Tests

Before submitting a Pull Request, verify that all core multi-agent components run without error:

```bash
# Verify MultiIntersectionEnv and MAPPO controller
python -c "from src.multi_env import MultiIntersectionEnv; print('MultiIntersectionEnv OK')"
python -c "from src.agents.mappo_agent import MAPPOTrafficController; print('MAPPO Controller OK')"
python -c "from src.routing import DynamicRouter; print('DynamicRouter OK')"
python -c "from src.metrics import TrafficMetricsCalculator; print('Metrics Calculator OK')"

# Run a quick 1-episode smoke training run
python src/train_marl.py --episodes 1 --max-steps 50
```

## 📝 Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your modifications with descriptive commit messages following conventional commits.
3. Push your branch and open a Pull Request against `main`.
4. Ensure all GitHub Actions CI checks pass.

## 📐 Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use explicit type annotations and docstrings for public classes and methods.
- Keep imports ordered: Standard Library → Third-Party Packages → Local Source Modules.

## 📄 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

# Spyware Detection Training Pipeline

![Python](https://img.shields.io/badge/python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)
![Docker](https://img.shields.io/badge/docker-ready-success)

A containerized pipeline for training and deploying spyware detection models.

## Features

- 🐳 Production-ready Docker container
- 🧩 Modular component architecture
- 📊 Comprehensive model evaluation
- 🔐 Secure non-root execution
- 📦 Optimized multi-stage build

## Quick Start

### Prerequisites

- Docker 20.10+
- Python 3.9+

### Using Docker (Recommended)

1. Build the image:
   ```bash
   docker build -t spyware-detector .
   ```

2. Run training:
   ```bash
   docker run --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/release:/app/release \
     spyware-detector
   ```

### Local Development

1. Setup environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run pipeline:
   ```bash
   python src/main.py
   ```

## Configuration

Edit YAML files in `config/` to customize:

```yaml
# Example config/components/model_trainer.yaml
params:
  model_type: "random_forest"
  hyperparams:
    n_estimators: [100, 200]
    max_depth: [10, 20]
```

## Project Structure

```
.
├── config/           # Pipeline configuration
├── data/             # Training data
├── models/           # Trained models
├── release/          # Deployment packages
├── src/              # Application code
└── tests/            # Unit tests
```

## CI/CD Integration

The included GitHub workflow:

1. Builds Docker image on push
2. Runs training pipeline
3. Packages artifacts
4. Creates GitHub release

## Security Best Practices

- Non-root container user
- Minimal runtime image
- Regular dependency updates
- Isolated build environment

## License

MIT License - See [LICENSE](LICENSE) for details.

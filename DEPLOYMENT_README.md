# HN Virality Predictor - Training & Deployment

Automated pipeline for training and deploying a CatBoost model to predict viral HackerNews posts.

## Quick Start

Run the complete pipeline (fetch data, train model, build & start Docker container):

```bash
./train_and_deploy.sh 10000 10 10
```

This fetches 10,000 HN stories with minimum score of 10 and minimum 10 descendants.

## Prerequisites

- **Go 1.21+** - For building the HN data fetcher
- **Python 3.11+** - For model training
- **Docker** - For containerized deployment
- **Cog** - For building prediction containers (auto-installed by script)
- **8GB+ RAM** - For model training

## Script Usage

### Basic Usage

```bash
./train_and_deploy.sh [STORIES_COUNT] [MIN_SCORE] [MIN_DESCENDANTS] [OUTPUT_DIR] [CONCURRENT] [RPS]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STORIES_COUNT` | 10000 | Number of HN stories to fetch (0 for incremental) |
| `MIN_SCORE` | 10 | Minimum story score filter |
| `MIN_DESCENDANTS` | 10 | Minimum descendants (comments) filter |
| `OUTPUT_DIR` | data | Directory for storing fetched data |
| `CONCURRENT` | 5000 | Concurrent API requests |
| `RPS` | 2000 | Requests per second limit |

### Environment Variables

You can customize behavior via environment variables:

```bash
export HN_PARSER_REPO="https://github.com/artbred/hn_parser.git"
export DOCKER_IMAGE_NAME="hn-virality-predictor"
export DOCKER_PORT="5000"

./train_and_deploy.sh
```

## What the Script Does

1. **📦 Data Fetching**
   - Clones and builds the [hn_parser](https://github.com/artbred/hn_parser) Go tool
   - Fetches HN stories matching specified criteria
   - Outputs JSONL file to `data/` directory

2. **🧠 Model Training**
   - Converts JSONL to CSV format
   - Sets up Python environment with UV
   - Trains CatBoost classifier
   - Exports model to `reports/catboost_model.cbm`
   - Generates feature statistics and metrics

3. **🐳 Docker Build with Cog**
   - Uses `cog build` to create optimized prediction container
   - Automatically includes model and dependencies from `cog.yaml`
   - Tags as `hn-virality-predictor:latest`
   - Installs Cog if not already available

4. **🚀 Container Deployment**
   - Stops any existing container
   - Starts new Cog-built container exposing API on port 5000
   - Provides commands for `docker exec -it` interaction

## Using the API

Once deployed, the API is available at `http://localhost:5000`

### Check Container Status

```bash
# View running container
docker ps | grep hn-virality

# View logs
docker logs -f hn-virality-predictor

# Execute commands inside container
docker exec -it hn-virality-predictor /bin/bash
```

### Predict Virality

```bash
curl -X POST http://localhost:5000/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "title": "Show HN: My Revolutionary AI Startup",
      "url": "https://example.com",
      "by": "johndoe",
      "time": 1763807356
    }
  }'
```

Response:
```json
{
  "output": {
    "probability": 0.73
  }
}
```

## Docker Commands

### View Running Containers
```bash
docker ps
```

### View Logs
```bash
docker logs hn-virality-predictor
```

### Stop Container
```bash
docker stop hn-virality-predictor
```

### Restart Container
```bash
docker restart hn-virality-predictor
```

### Remove Container
```bash
docker stop hn-virality-predictor
docker rm hn-virality-predictor
```

### Rebuild with Cog and Redeploy
Just run the script again - it automatically uses `cog build` and replaces the existing container:
```bash
./train_and_deploy.sh
```

### Interact with Container
```bash
# Open bash shell inside container
docker exec -it hn-virality-predictor /bin/bash

# Run Python commands
docker exec -it hn-virality-predictor python -c "print('Hello')"
```

## Incremental Data Updates

To perform incremental updates (fetch only new stories since last run):

```bash
./train_and_deploy.sh 0
```

This finds the most recent JSONL file and fetches newer stories.

## Custom Training Examples

### Train on High-Quality Posts Only
```bash
./train_and_deploy.sh 50000 50 50
```

### Quick Test Run
```bash
./train_and_deploy.sh 1000 5 5
```

### Custom Port
```bash
DOCKER_PORT=8080 ./train_and_deploy.sh 10000 10 10
```

Access API at: `http://localhost:8080`

## Project Structure

```
.
├── train_and_deploy.sh          # Main orchestration script
├── Dockerfile                    # Container definition
├── scripts/
│   └── convert_jsonl_to_csv.py  # Data conversion helper
├── src/
│   ├── train.py                 # Model training script
│   ├── features.py              # Feature engineering
│   └── feature_stats.py         # Feature statistics
├── predict.py                   # Cog predictor for inference
├── data/                        # Fetched HN data (JSONL)
└── reports/                     # Trained models & metrics
    ├── catboost_model.cbm
    ├── feature_stats.json
    └── metrics.json
```

## Troubleshooting

### Go Not Found
Install Go from https://golang.org/doc/install

### Docker Build Fails
Ensure Docker daemon is running:
```bash
docker info
```

### Container Exits Immediately
Check logs for errors:
```bash
docker logs hn-virality-predictor
```

### Out of Memory During Training
Reduce the number of stories:
```bash
./train_and_deploy.sh 5000 10 10
```

### Port Already in Use
Change the port:
```bash
DOCKER_PORT=8080 ./train_and_deploy.sh
```

## Model Artifacts

After training, the following artifacts are generated in `reports/`:

- `catboost_model.cbm` - Trained CatBoost model
- `feature_stats.json` - Feature statistics for inference
- `metrics.json` - Model performance metrics
- `validation_predictions.csv` - Validation set predictions
- `catboost_feature_importance.csv` - Feature importance scores

## Weekly Retraining

To set up weekly retraining, add a cron job:

```bash
crontab -e
```

Add this line to run every Sunday at 6 PM:
```
0 18 * * 0 cd /path/to/hn_virality && ./train_and_deploy.sh >> /var/log/hn_virality_train.log 2>&1
```

## License

MIT

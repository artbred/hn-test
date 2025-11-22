# Cog Build Update Summary

## Changes Made

Successfully updated the deployment script to use **Cog's native build process** instead of a custom Dockerfile.

### Script Updates

#### `train_and_deploy.sh`

**Step 3: Build Docker Image with Cog**
- Changed from `docker build` to `cog build -t hn-virality-predictor:latest`
- Added automatic Cog installation if not present
- Cog version verification
- Uses `cog.yaml` configuration automatically

**Step 4: Start Docker Container**
- Updated to properly run Cog-built containers
- Added instructions for `docker exec -it` interaction
- Shows proper commands for:
  - Viewing logs: `docker logs -f hn-virality-predictor`
  - Executing commands: `docker exec -it hn-virality-predictor /bin/bash`

### Key Benefits of Cog Build

1. **Simplified Configuration** - Uses `cog.yaml` instead of managing Dockerfile
2. **Optimized Containers** - Cog builds are optimized for ML model serving
3. **Better Integration** - Native Cog HTTP server for predictions
4. **Easier Debugging** - Can use `docker exec -it` to inspect running container

### How It Works Now

```bash
# Run the script
./train_and_deploy.sh 10000 10 10

# Script will:
# 1. Fetch data from HN
# 2. Train CatBoost model
# 3. Run: cog build -t hn-virality-predictor:latest
# 4. Run: docker run -d --name hn-virality-predictor -p 5000:5000 hn-virality-predictor:latest
```

### Using the Container

**Test predictions:**
```bash
curl -X POST http://localhost:5000/predictions \
  -H 'Content-Type: application/json' \
  -d '{"input":{"title":"Test Post","url":"","by":"user","time":1763807356}}'
```

**Interact with container:**
```bash
# View logs
docker logs -f hn-virality-predictor

# Execute bash inside container
docker exec -it hn-virality-predictor /bin/bash

# Run Python commands
docker exec -it hn-virality-predictor python -c "import catboost; print(catboost.__version__)"
```

### Files Modified

1. ✅ `train_and_deploy.sh` - Updated build and run processes
2. ✅ `DEPLOYMENT_README.md` - Updated documentation
3. ℹ️ `Dockerfile` - Kept as reference but not used (Cog generates its own)

### What Cog Does

Cog reads `cog.yaml` and automatically:
- Creates optimized Dockerfile
- Installs Python dependencies from `requirements.txt`
- Sets up the prediction environment
- Configures the HTTP server
- Handles model loading via `predict.py:Predictor`

### Next Steps

Ready to use! Run:
```bash
./train_and_deploy.sh
```

The script will handle everything from data fetching to deploying the Cog-built container.

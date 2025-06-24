#!/bin/bash
# setup_local_qdrant.sh - Migration script to local Qdrant Docker

echo "🚀 Setting up Local Qdrant Docker for LiveKit Voice Agent"
echo "========================================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p qdrant_storage
mkdir -p qdrant_snapshots
mkdir -p qdrant_config
mkdir -p data

# Create the optimized production config file
echo "⚙️ Creating optimized Qdrant configuration..."
cat > qdrant_config/production.yaml << 'EOF'
# production.yaml - Optimized Qdrant configuration for LiveKit telephony
log_level: INFO

service:
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 32
  max_workers: 0

storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  temp_path: /qdrant/tmp
  on_disk_payload: false
  mmap_threshold_kb: 0
  
  wal:
    wal_capacity_mb: 64
    wal_segments_ahead: 1
  
  performance:
    max_search_threads: 0
    max_optimization_threads: 1
  
  optimizers:
    deleted_threshold: 0.3
    vacuum_min_vector_number: 500
    default_segment_number: 2
    max_segment_size_kb: 51200
    memmap_threshold_kb: 0
    indexing_threshold_kb: 10000
    flush_interval_sec: 3
    max_optimization_threads: 1

hnsw_index:
  m: 8
  ef_construct: 64
  full_scan_threshold: 5000
  max_indexing_threads: 0
  max_connections: 0
  ef: 128

cluster:
  enabled: false
EOF

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Stop any existing Qdrant containers
echo "🛑 Stopping existing Qdrant containers..."
docker stop livekit_qdrant_local 2>/dev/null || true
docker rm livekit_qdrant_local 2>/dev/null || true

# Start Qdrant with Docker Compose
echo "🐳 Starting optimized Qdrant Docker container..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Qdrant failed to start after 30 seconds"
        echo "   Check logs with: docker logs livekit_qdrant_local"
        exit 1
    fi
    sleep 1
done

# Check Qdrant status
echo "📊 Checking Qdrant status..."
curl -s http://localhost:6333/health | python -m json.tool 2>/dev/null || echo "Qdrant is running but health check format changed"

# Test gRPC connection
echo "🔗 Testing gRPC connection..."
if netstat -an | grep :6334 > /dev/null 2>&1 || ss -an | grep :6334 > /dev/null 2>&1; then
    echo "✅ gRPC port 6334 is open"
else
    echo "⚠️  gRPC port 6334 not detected, will fall back to HTTP"
fi

echo ""
echo "🎉 Local Qdrant Docker setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Update your .env file with the new local settings"
echo "2. Place your data files in the 'data' directory"
echo "3. Run: python qdrant_data_ingestion.py --directory data"
echo "4. Start your LiveKit agent: python ultra_fast_qdrant_agent.py dev"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker logs -f livekit_qdrant_local"
echo "   - Restart: docker-compose restart"
echo "   - Stop: docker-compose down"
echo "   - Dashboard: http://localhost:6333/dashboard"
echo ""
echo "⚡ Expected performance improvements:"
echo "   - Latency: 200ms → 50-100ms"
echo "   - Throughput: 3-5x improvement"
echo "   - Reliability: No network dependencies"
#!/bin/bash

# Default values
SERVER_URL="http://localhost:11338"
MODELS_FILE="models.txt"
DATASET="Banking77Classification"
ENDPOINT="/v3/embeddings"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --server-url URL    Server URL (default: http://localhost:11338)"
            echo "  --endpoint PATH     API endpoint path (default: /v3/embeddings)"
            echo "  --dataset NAME      Dataset to benchmark (default: Banking77Classification)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Model specification:"
            echo "  - If models.txt exists, it will be used (manual specification)"
            echo "  - Otherwise, models will be fetched from server automatically"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "OVMS MTEB Benchmark Suite"
echo "========================="
echo "Server URL: $SERVER_URL"
echo "Endpoint: $ENDPOINT"
echo "Dataset: $DATASET"
echo ""

# Check if models.txt exists and has content
if [ -f "$MODELS_FILE" ] && [ -s "$MODELS_FILE" ]; then
    echo "Using existing models from $MODELS_FILE (manual specification):"
    MODEL_COUNT=$(wc -l < "$MODELS_FILE")
    echo "Found $MODEL_COUNT models:"
    echo "----------------------------------------"
    cat "$MODELS_FILE"
    echo "----------------------------------------"
    echo ""
else
    # Fetch available models from server
    echo "No models.txt found. Fetching available models from $SERVER_URL/v3/models..."
    
    # Make the curl request and extract model names using jq
    if command -v jq &> /dev/null; then
        # Use jq if available (more reliable JSON parsing)
        curl -s "$SERVER_URL/v3/models" | jq -r '.data[].id' > "$MODELS_FILE"
    else
        # Fallback to grep/sed if jq is not available
        echo "jq not found, using grep/sed fallback..."
        curl -s "$SERVER_URL/v3/models" | grep -o '"id":"[^"]*"' | sed 's/"id":"//g' | sed 's/"//g' > "$MODELS_FILE"
    fi
    
    # Check if the file was created successfully and has content
    if [ ! -f "$MODELS_FILE" ] || [ ! -s "$MODELS_FILE" ]; then
        echo "Error: Failed to fetch models from server or no models available!"
        echo "Please check if the OVMS server is running at $SERVER_URL"
        exit 1
    fi
    
    MODEL_COUNT=$(wc -l < "$MODELS_FILE")
    echo "Successfully fetched $MODEL_COUNT models:"
    echo "----------------------------------------"
    cat "$MODELS_FILE"
    echo "----------------------------------------"
    echo ""
fi

# Initialize results array
declare -a RESULTS=()

# Print table header for real-time results
echo "Benchmark Progress (Real-time Results)"
echo "======================================"
echo ""
printf "%-40s | %-12s | %-10s\n" "Model Name" "Accuracy" "Status"
printf "%-40s-+-%-12s-+-%-10s\n" "----------------------------------------" "------------" "----------"

# Read models from file and process each one
while IFS= read -r MODEL_NAME || [ -n "$MODEL_NAME" ]; do
    # Skip empty lines and comments
    [[ -z "$MODEL_NAME" || "$MODEL_NAME" =~ ^#.*$ ]] && continue
    
    # Show current progress
    printf "%-40s | %-12s | %-10s" "$MODEL_NAME" "Running..." "IN PROGRESS"
    
    # Run the python command with output suppressed
    if python ovms_mteb.py --model_name "$MODEL_NAME" --service_url "$SERVER_URL$ENDPOINT" --dataset "$DATASET" > /dev/null 2>&1; then
        # Extract accuracy result
        ACCURACY=$(cat "results/no_model_name_available/no_revision_available/$DATASET.json" 2>/dev/null | jq -r '.scores.test[0].accuracy // "N/A"' 2>/dev/null)
        if [ "$ACCURACY" = "null" ] || [ "$ACCURACY" = "" ]; then
            ACCURACY="N/A"
        fi
        RESULTS+=("$MODEL_NAME|$ACCURACY|SUCCESS")
        # Clear the line and show final result
        printf "\r%-40s | %-12s | %-10s\n" "$MODEL_NAME" "$ACCURACY" "SUCCESS"
    else
        RESULTS+=("$MODEL_NAME|N/A|FAILED")
        # Clear the line and show failed result
        printf "\r%-40s | %-12s | %-10s\n" "$MODEL_NAME" "N/A" "FAILED"
    fi
done < "$MODELS_FILE"

echo ""
echo "Final Summary"
echo "============="
echo ""

# Print table header
printf "%-40s | %-12s | %-10s\n" "Model Name" "Accuracy" "Status"
printf "%-40s-+-%-12s-+-%-10s\n" "----------------------------------------" "------------" "----------"

# Print results
for result in "${RESULTS[@]}"; do
    IFS='|' read -r model accuracy status <<< "$result"
    printf "%-40s | %-12s | %-10s\n" "$model" "$accuracy" "$status"
done

echo ""
echo "Benchmark completed for $(echo "${RESULTS[@]}" | wc -w) models."
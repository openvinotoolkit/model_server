#!/bin/bash

#Ubuntu
MIN_LINES_COV=76.0
MIN_FUNCTION_COV=83.0

#Rhel
MIN_LINES_COV=76.0
MIN_FUNCTION_COV=83.0

LINES_COV=$(grep "headerCovTableEntry.*%" genhtml/index.html | grep -oP '\d+\.\d+(?=\s*%|&nbsp;%)' | head -n 1)
FUNC_COV=$(grep "headerCovTableEntry.*%" genhtml/index.html | grep -oP '\d+\.\d+(?=\s*%|&nbsp;%)' | tail -n 1)

echo "Lines coverage: ${LINES_COV}% (minimum: ${MIN_LINES_COV}%)"
echo "Functions coverage: ${FUNC_COV}% (minimum: ${MIN_FUNCTION_COV}%)"

if [ -z "$LINES_COV" ] || [ -z "$FUNC_COV" ]; then
    echo "Error: Could not parse coverage values from genhtml/index.html"
    exit 1
fi

if (( $(echo "$MIN_LINES_COV > $LINES_COV" | bc -l) )); then
    echo "Error: ${LINES_COV}% Lines coverage is lower than minimal ${MIN_LINES_COV}%"
    exit 1
fi

if (( $(echo "$MIN_FUNCTION_COV > $FUNC_COV" | bc -l) )); then
    echo "Error: ${FUNC_COV}% Functions coverage is lower than minimal ${MIN_FUNCTION_COV}%"
    exit 1
fi

echo "Coverage check success"
exit 0

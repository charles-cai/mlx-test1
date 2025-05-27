#!/bin/bash
"""
Test script to verify the FastAPI service is working correctly using curl
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
TOTAL=0

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Test API health endpoint
test_api_health() {
    echo -e "\n🔍 Running Health Check..."
    
    # Make request and capture response
    RESPONSE=$(curl -s -w "HTTPSTATUS:%{http_code}" "http://localhost:8000/health" 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        print_error "Cannot connect to API - is it running on localhost:8000?"
        return 1
    fi
    
    # Extract HTTP status and body
    HTTP_STATUS=$(echo $RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    BODY=$(echo $RESPONSE | sed -e 's/HTTPSTATUS:.*//')
    
    if [ "$HTTP_STATUS" -eq 200 ]; then
        print_success "Health check passed"
        
        # Parse JSON response (basic parsing)
        STATUS=$(echo "$BODY" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        MODEL_AVAILABLE=$(echo "$BODY" | grep -o '"model_available":[^,}]*' | cut -d':' -f2 | tr -d ' ')
        
        echo "   Status: $STATUS"
        echo "   Model available: $MODEL_AVAILABLE"
        return 0
    else
        print_error "Health check failed with status $HTTP_STATUS"
        return 1
    fi
}

# Test API info endpoint
test_api_info() {
    echo -e "\n🔍 Running API Info..."
    
    # Make request and capture response
    RESPONSE=$(curl -s -w "HTTPSTATUS:%{http_code}" "http://localhost:8000/" 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        print_error "Cannot connect to API"
        return 1
    fi
    
    # Extract HTTP status and body
    HTTP_STATUS=$(echo $RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    BODY=$(echo $RESPONSE | sed -e 's/HTTPSTATUS:.*//')
    
    if [ "$HTTP_STATUS" -eq 200 ]; then
        print_success "API info endpoint working"
        
        # Parse JSON response (basic parsing)
        MESSAGE=$(echo "$BODY" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
        VERSION=$(echo "$BODY" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        
        echo "   Message: $MESSAGE"
        echo "   Version: $VERSION"
        return 0
    else
        print_error "API info failed with status $HTTP_STATUS"
        return 1
    fi
}

# Test predict endpoint (basic check)
test_predict_endpoint() {
    echo -e "\n🔍 Running Predict Endpoint..."
    
    # Test if the predict endpoint exists (should return 422 for missing file)
    RESPONSE=$(curl -s -w "HTTPSTATUS:%{http_code}" -X POST "http://localhost:8000/predict" 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        print_error "Cannot connect to predict endpoint"
        return 1
    fi
    
    # Extract HTTP status
    HTTP_STATUS=$(echo $RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    
    if [ "$HTTP_STATUS" -eq 422 ]; then
        print_success "Predict endpoint is accessible (expects file upload)"
        print_info "Test with: curl -X POST -F 'file=@image.png' http://localhost:8000/predict"
        return 0
    elif [ "$HTTP_STATUS" -eq 200 ]; then
        print_success "Predict endpoint working"
        return 0
    else
        print_error "Predict endpoint returned unexpected status $HTTP_STATUS"
        return 1
    fi
}

# Test OpenAPI docs endpoint
test_docs_endpoint() {
    echo -e "\n🔍 Running Docs Endpoint..."
    
    RESPONSE=$(curl -s -w "HTTPSTATUS:%{http_code}" "http://localhost:8000/docs" 2>/dev/null)
    HTTP_STATUS=$(echo $RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    
    if [ "$HTTP_STATUS" -eq 200 ]; then
        print_success "API documentation is accessible"
        return 0
    else
        print_error "API docs failed with status $HTTP_STATUS"
        return 1
    fi
}

# Main test execution
main() {
    echo "🧪 Testing MNIST FastAPI Service"
    echo "========================================"
    
    # Wait for service to be ready
    echo "Waiting for service to be ready..."
    sleep 2
    
    # Define test functions
    tests=(
        "test_api_health"
        "test_api_info" 
        "test_predict_endpoint"
        "test_docs_endpoint"
    )
    
    TOTAL=${#tests[@]}
    
    # Run each test
    for test_func in "${tests[@]}"; do
        if $test_func; then
            ((PASSED++))
        fi
    done
    
    # Print results
    echo -e "\n========================================"
    echo "Test Results: $PASSED/$TOTAL passed"
    
    if [ "$PASSED" -eq "$TOTAL" ]; then
        print_success "All tests passed! API is working correctly."
        echo -e "\nNext steps:"
        echo "- Visit http://localhost:8000/docs for interactive API documentation"
        echo "- Test predictions by uploading digit images"
        echo "- Check logs for any issues"
        
        echo -e "\n📝 Manual prediction test:"
        echo "curl -X POST -F 'file=@digit_image.png' http://localhost:8000/predict"
    else
        print_warning "Some tests failed. Check the API service and logs."
        echo -e "\nDebugging tips:"
        echo "- Is the API running? Try: cd model && python api.py"
        echo "- Is the model trained? Try: cd model && python MnistModel.py --train"
        echo "- Check for port conflicts: netstat -tulpn | grep :8000"
        echo "- View API logs: docker logs mnist_api"
    fi
    
    # Exit with appropriate code
    if [ "$PASSED" -eq "$TOTAL" ]; then
        exit 0
    else
        exit 1
    fi
}

# Make script executable and run
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
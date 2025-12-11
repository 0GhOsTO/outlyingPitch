# Makefile for OutlyingPitch Project
# =====================================
# Targets:
#   make install-mlp       - Install dependencies for MLP notebook
#   make install-dashboard - Install dependencies for Streamlit dashboard  
#   make install           - Install all dependencies
#   make run-mlp           - Run MLP training with optimal hyperparameters
#   make run-dashboard     - Run the Streamlit dashboard
#   make clean             - Clean up generated files

PYTHON := python3
PIP := pip3
VENV := venv

# Load optimal hyperparameters from saved JSON file
# If outputs/best_hyperparameters.json exists, it will be used
# Otherwise, fallback to defaults
HYPERPARAMS_FILE := outputs/best_hyperparameters.json

# Extract hyperparameters using jq (install with: brew install jq)
# Fallback values if file doesn't exist (using latest optimized values)
HIDDEN1 := $(shell [ -f $(HYPERPARAMS_FILE) ] && jq -r '.hidden1' $(HYPERPARAMS_FILE) || echo "250")
HIDDEN2 := $(shell [ -f $(HYPERPARAMS_FILE) ] && jq -r '.hidden2' $(HYPERPARAMS_FILE) || echo "192")
DROPOUT_RATE := $(shell [ -f $(HYPERPARAMS_FILE) ] && jq -r '.dropout_rate' $(HYPERPARAMS_FILE) || echo "0.273")
LEARNING_RATE := $(shell [ -f $(HYPERPARAMS_FILE) ] && jq -r '.lr' $(HYPERPARAMS_FILE) || echo "0.00099")
BATCH_SIZE := $(shell [ -f $(HYPERPARAMS_FILE) ] && jq -r '.batch_size' $(HYPERPARAMS_FILE) || echo "64")
WEIGHT_DECAY := $(shell [ -f $(HYPERPARAMS_FILE) ] && jq -r '.weight_decay' $(HYPERPARAMS_FILE) || echo "0.0000102")

.PHONY: all install install-mlp install-dashboard run-mlp run-dashboard clean venv

all: install

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at ./$(VENV)"
	@echo "Activate it with: source $(VENV)/bin/activate"

# Install MLP notebook dependencies
install-mlp: 
	@echo "Installing MLP notebook dependencies..."
	$(PIP) install -r requirements.txt
	@echo "MLP dependencies installed successfully!"

# Install Streamlit dashboard dependencies
install-dashboard:
	@echo "Installing Streamlit dashboard dependencies..."
	$(PIP) install -r requirements_dashboard.txt
	@echo "Dashboard dependencies installed successfully!"

# Install all dependencies
install: install-mlp install-dashboard
	@echo "All dependencies installed successfully!"

# Run MLP training with optimal hyperparameters (skips Optuna search)
run-mlp:
	@echo "Running MLP training with optimal hyperparameters..."
	@echo "  Hidden Layer 1: $(HIDDEN1)"
	@echo "  Hidden Layer 2: $(HIDDEN2)"
	@echo "  Dropout Rate:   $(DROPOUT_RATE)"
	@echo "  Learning Rate:  $(LEARNING_RATE)"
	@echo "  Batch Size:     $(BATCH_SIZE)"
	@echo "  Weight Decay:   $(WEIGHT_DECAY)"
	@echo ""
	$(PYTHON) run_mlp_optimal.py \
		--hidden1 $(HIDDEN1) \
		--hidden2 $(HIDDEN2) \
		--dropout $(DROPOUT_RATE) \
		--lr $(LEARNING_RATE) \
		--batch-size $(BATCH_SIZE) \
		--weight-decay $(WEIGHT_DECAY)

# Run the Streamlit dashboard
run-dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run dashboard.py

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	rm -rf $(VENV)
	rm -f outputs/*.pkl
	rm -f outputs/*.pt
	rm -f outputs/*.npy
	rm -f outputs/*.png
	rm -f outputs/*.txt
	rm -f model.pt scaler.pkl label_encoder.pkl processed_data.csv
	@echo "Cleanup complete!"

# Help target
help:
	@echo "OutlyingPitch-2 Makefile"
	@echo "========================"
	@echo ""
	@echo "Available targets:"
	@echo "  make install-mlp       - Install dependencies for MLP notebook"
	@echo "  make install-dashboard - Install dependencies for Streamlit dashboard"
	@echo "  make install           - Install all dependencies"
	@echo "  make run-mlp           - Run MLP training with optimal hyperparameters"
	@echo "  make run-dashboard     - Run the Streamlit dashboard"
	@echo "  make clean             - Clean up generated files"
	@echo "  make venv              - Create a virtual environment"
	@echo "  make help              - Show this help message"
	@echo ""
	@echo "Optimal hyperparameters (from Optuna study):"
	@echo "  Hidden Layer 1: $(HIDDEN1)"
	@echo "  Hidden Layer 2: $(HIDDEN2)"
	@echo "  Dropout Rate:   $(DROPOUT_RATE)"
	@echo "  Learning Rate:  $(LEARNING_RATE)"
	@echo "  Batch Size:     $(BATCH_SIZE)"
	@echo "  Weight Decay:   $(WEIGHT_DECAY)"


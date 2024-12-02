#!/bin/bash

# Function to display usage information
show_usage() {
    echo "Usage: $0 --resume RESUME_PATH --job-description JD_PATH [OPTIONS]"
    echo
    echo "Required:"
    echo "  --resume            Path to your resume file (PDF, DOCX, DOC, or TXT)"
    echo "  --job-description   Path to job description file"
    echo
    echo "Optional:"
    echo "  --output           Path to save output (default: output.json)"
    echo "  --config-dir       Path to config directory (default: src/config)"
    echo "  --non-interactive  Run in non-interactive mode"
    echo
    echo "Example:"
    echo "  $0 --resume resume.pdf --job-description job.txt"
}

# Default values
RESUME="/media/vasu/Hard Disk/Projects/CV_Customization_System/data/input/resume.pdf"
JD="/media/vasu/Hard Disk/Projects/CV_Customization_System/data/input/jd.txt"
OUTPUT="/media/vasu/Hard Disk/Projects/CV_Customization_System/data/output/output.json"
CONFIG_DIR="/media/vasu/Hard Disk/Projects/CV_Customization_System/src/config/"
INTERACTIVE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --job-description)
            JD="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --non-interactive)
            INTERACTIVE="--non-interactive"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$RESUME" ]] || [[ -z "$JD" ]]; then
    echo "Error: Both --resume and --job-description are required"
    show_usage
    exit 1
fi

# Check if files exist
if [[ ! -f "$RESUME" ]]; then
    echo "Error: Resume file not found: $RESUME"
    exit 1
fi

if [[ ! -f "$JD" ]]; then
    echo "Error: Job description file not found: $JD"
    exit 1
fi

# Check if config directory exists
if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed"
    exit 1
fi

# Initialize conda
eval "$(conda shell.bash hook)"

# Get environment name from environment.yml
ENV_NAME=$(grep "name:" env/environment.yml | cut -d' ' -f2)
if [[ -z "$ENV_NAME" ]]; then
    echo "Error: Could not find environment name in environment.yml"
    exit 1
fi

# Create or activate conda environment
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment..."
    conda env create -f env/environment.yml
fi

# Activate environment
echo "Activating conda environment..."
conda activate "$ENV_NAME"

# Install required pip packages
echo "Installing required pip packages..."
pip install -q autogen autogen-agentchat aiofiles aioconsole

# Add src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/media/vasu/Hard Disk/Projects/CV_Customization_System/src"

# Run the main script
echo "Starting CV customization..."
if python src/scripts/cv_customization_system.py \
    --resume "$RESUME" \
    --job-description "$JD" \
    --output "$OUTPUT" \
    --config-dir "$CONFIG_DIR" \
    $INTERACTIVE; then
    echo "CV customization completed successfully!"
    echo "Output saved to: $OUTPUT"
else
    echo "Error: CV customization failed"
    conda deactivate
    exit 1
fi

# Cleanup
conda deactivate


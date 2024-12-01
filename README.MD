# CraftMyCV

An intelligent CV/resume customization system powered by AutoGen that analyzes, optimizes, and tailors your resume for specific job descriptions. The system uses multiple specialized AI agents working in concert to create highly optimized, ATS-friendly resumes.

## 🌟 Features

- **Intelligent Analysis**: Multiple AI agents analyze both resumes and job descriptions
- **Multi-format Support**: 
  - Documents: PDF, DOCX, DOC, TXT
  - Images: PNG, JPG, JPEG
  - Built-in OCR capabilities for scanned documents
- **Advanced Processing**:
  - Automatic text extraction with OCR fallback
  - Skill matching and optimization
  - Project recommendations based on job requirements
  - ATS-friendly formatting
  - Professional LaTeX output
- **Robust Architecture**:
  - Multiple LLM support (OpenAI GPT-4 and Anthropic Claude)
  - Fallback LLM system for reliability
  - Extensive error handling
  - Configurable agent behaviors
  - Interactive mode for user feedback

## 🤖 AI Agents

The system employs specialized agents for different phases:

1. **Analysis Phase**: 
   - Analyzes resume content and job requirements
   - Provides skill matching scores
   - Suggests relevant projects
   - Identifies improvement areas

2. **Optimization Phase**:
   - Enhances content based on job alignment
   - Incorporates selected projects
   - Optimizes formatting and structure

3. **Formatting Phase**:
   - Applies professional LaTeX templates
   - Ensures ATS compatibility
   - Maintains consistent styling

## 🛠 Prerequisites

- Python 3.11+
- Conda package manager
- Tesseract OCR
- API Keys:
  - OpenAI API key
  - Claude API key (Anthropic)

## 📦 Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd craft-my-cv
```

2. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate craft-my-cv
```

3. Set up environment variables:
Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
```

## 🚀 Usage

Run via the provided script:

```bash
./script.sh --resume path/to/resume.pdf --job-description path/to/jd.txt
```

Options:
- `--resume`: Path to resume file (PDF, DOCX, DOC, or TXT)
- `--job-description`: Path to job description file
- `--output`: Output path (default: output.json)
- `--config-dir`: Path to config directory (default: src/config)
- `--non-interactive`: Run in non-interactive mode

## 🏗 Project Structure
```
craft-my-cv/
├── src/
│ ├── config/
│ │ ├── agents.yaml    # Agent configurations
│ │ ├── templates.yaml # LaTeX templates
│ │ └── projects.yaml  # Project templates
│ ├── scripts/
│ │ └── cv_customization_system.py
│ ├── tools/
│ │ ├── text_extractor_tool.py
│ │ ├── fallback_llm.py
│ │ ├── prompt_generator.py
│ │ ├── schema_generator.py
│ │ └── setup_logging.py
├── data/
│ ├── input/          # Input files
│ └── logs/           # System logs
├── script.sh         # Main execution script
├── environment.yml   # Dependencies
└── README.md
```

## 🔧 Configuration

### Agent Configuration (agents.yaml)
- Agent roles and system prompts
- LLM configurations
- Output schemas
- Fallback settings

### Template Configuration (templates.yaml)
- LaTeX resume templates
- Styling configurations
- Section layouts

### Project Configuration (projects.yaml)
- Project templates
- Skill mappings
- Description formats

## 📚 Dependencies

Core dependencies include:
- **AI Framework**:
  - autogen
  - openai
  - anthropic

- **Document Processing**:
  - pdf2image
  - python-docx
  - pytesseract
  - pillow

- **Utilities**:
  - pyyaml
  - python-dotenv
  - olefile

## 🔍 Error Handling

The system includes comprehensive error handling for:
- File operations
- API failures
- Text extraction
- OCR processing
- Input validation
- LLM failures (with fallback chain)
- User interactions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- Microsoft AutoGen framework
- OpenAI GPT models
- Anthropic Claude models
- Tesseract OCR
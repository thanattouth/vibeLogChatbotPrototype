🛡️ Enhanced SOC Analysis AI Assistant: AI-Powered Cybersecurity Log Analysis (Offline, On-Premises)

An advanced AI system for cybersecurity log analysis, leveraging Ollama (local LLM), LangChain, FAISS, and Streamlit. It includes robust caching and distributed processing capabilities, making it ideal for use in a Security Operations Center (SOC) environment that requires air-gapped deployment and data privacy.

Keywords: Cybersecurity, Log Analysis, SOC, AI, LLM, Ollama, LangChain, FAISS, Streamlit, On-Premises, Offline, Air-Gapped, Data Privacy, Threat Detection, Incident Response, Security Automation, SIEM, Deepseek-Coder
✨ Key Features for Advanced Security Operations
🤖 AI-Powered Security Log Analysis

    Offline Log Analysis: Utilizes the deepseek-coder local LLM model via Ollama for on-premises processing without external API calls, ensuring data privacy and security.

    RAG (Retrieval-Augmented Generation): Implements a Retrieval-Augmented Generation system with FAISS vector database for efficient vector search and context-aware responses to security queries.

    Real-time & Batch Processing: Supports both immediate analysis of uploaded logs and comprehensive batch processing for larger datasets, crucial for incident response.

🔍 Advanced Log Processing & Data Extraction

    Multi-format Support: Compatible with various security log file types: .log, .txt, .json, and .csv.

    Intelligent Preprocessing: Smart preprocessing of security logs to extract critical information.

    Automated Data Extraction: Automatic extraction of IP addresses, timestamps, and security indicators from log entries.

💬 Interactive SOC Chat Interface

    Conversational UI: A user-friendly chat interface with streaming responses for a dynamic security analysis experience.

    Ready-to-use Examples: Provides example questions to guide security analysts in their log analysis.

    Chat History Export: Ability to export the entire chat conversation as a Markdown file for record-keeping and reporting.

📊 Security Data Visualization

    Log Event Timeline: Interactive timeline visualization of log events to identify chronological security patterns.

    Threat Distribution Charts: Visual representation of detected security threats for quick insights into threat landscape.

    Real-time Statistics: Displays live metrics and statistics about the loaded security log data.

⚡ Performance Enhancements for Large Log Volumes

    Redis Caching: Caches analysis results to significantly speed up repeated queries and reduce processing time, enhancing SOC efficiency.

    Distributed Processing with Celery: Leverages Celery for scalable and distributed processing of large log files, improving handling of high volumes.

    Streaming Responses: Provides real-time output from the AI model, enhancing user experience during analysis.

🛡️ Security-Focused Analysis & Recommendations

    MITRE ATT&CK Framework Integration: Analyzes threats and provides insights aligned with the MITRE ATT&CK framework, aiding in threat intelligence.

    Threat Classification: Classifies detected threats into Critical, High, Medium, and Low categories for prioritization.

    Actionable Recommendations: Offers practical and actionable steps for containment, investigation, and long-term mitigation of security incidents.

🔒 Enhanced Audit Logging & Compliance

    Comprehensive Logging: Implements detailed audit logging for key actions (file uploads, log analysis, vector store updates, user queries) for compliance and accountability.

    Log Viewing Interface: Built-in UI to view, filter, and download audit logs directly from the application.

🧠 Usage Examples & Workflow for Security Teams
Common Security Questions the System Can Answer:

    "Identify potential security threats in these logs."

    "Show all suspicious IP addresses and their activities."

    "Analyze patterns of failed login attempts."

    "Provide a timeline of security-related events."

    "What is the most critical threat detected?"

    "Are there any indicators of a brute-force attack?"

    "Summarize all unusual activities from external sources."

How the SOC Analysis AI Assistant Works:

    Upload Log Files: Simply drag and drop your security log files (JSON, CSV, LOG, TXT) into the application.

    System Processes Data: The system preprocesses the data, extracts security features, and creates vector embeddings for efficient search.

    Ask Security Questions: Interact with the AI through the chat interface, asking specific questions about your logs.

    Receive Comprehensive Analysis: Get detailed answers, complete with evidence from logs, threat classifications, and actionable recommendations for your security operations.

🔧 Installation & Setup Guide
1. Clone the Project Repository

git clone https://github.com/thanattouth/vibeLogChatbotPrototype.git
cd vibe_log_chatbot_prototype

2. Create a Python Virtual Environment and Install Dependencies

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

3. Install and Set Up Ollama (Local LLM Server)

Ollama is essential for offline AI processing.

Install Ollama:

    macOS:

    brew install ollama

    Linux:

    curl -fsSL https://ollama.com/install.sh | sh

    Windows: Download from https://ollama.com

Start Ollama server:

ollama serve

Pull the deepseek-coder model: (This model is used for code analysis and log interpretation.)

ollama pull deepseek-coder

4. Install Redis (Optional - for Caching and Celery Distributed Processing)

Redis is highly recommended for performance enhancements and distributed task management.

    Ubuntu/Debian:

    sudo apt update
    sudo apt install redis-server
    sudo systemctl start redis-server
    sudo systemctl enable redis-server # Enable Redis to start on boot

    macOS:

    brew install redis
    brew services start redis # Start Redis service

    Windows:
    Download from Redis for Windows releases or use WSL2 for a Linux environment.

🚀 Running the SOC Analysis AI Assistant

To run the system:

streamlit run logbot_prototype.py

Access the application in your web browser:

    Local: http://localhost:8501

    Network: http://[YOUR_IP]:8501

⚙️ Advanced Configuration for Scalability
Enabling Distributed Processing with Celery:

For processing very large log files, enable distributed processing.

    Start Celery Worker (requires Redis running as a broker/backend):

    celery -A logbot_prototype.celery_app worker --loglevel=info

    Enable in UI: Check the "Enable distributed processing (Celery)" option in the application's file upload section.

Redis Configuration:

    Host: localhost (default)

    Port: 6379 (default)

    Databases: db0 (for caching), db1 (Celery broker), db2 (Celery backend)

📊 System Capabilities Overview
Analysis Modes for Varied Security Needs:

    Basic: Standard log analysis for quick insights.

    Advanced: In-depth analysis with event correlation for more complex scenarios.

    Forensic: Detailed, investigative analysis suitable for incident response and digital forensics.

Threat Levels for Prioritization:

    Critical: Immediate action required.

    High: High priority investigation needed.

    Medium: Should be investigated.

    Low: Monitor for patterns.

    None: No threat detected.

Supported Log File Types:

    JSON: Ideal for structured logs from various security tools.

    CSV: Suitable for tabular log data.

    LOG/TXT: For plain text log files from diverse sources.

🔐 Security and Privacy: Air-Gapped Deployment Ready

✅ Air-Gapped Deployment Capability:

    Data Stays On-Premises: All processing occurs locally within your environment; no sensitive data leaves your network.

    Local Ollama LLM: Leverages a local Large Language Model (LLM), ensuring complete data privacy and confidentiality.

    Secure for Sensitive Data: Designed specifically for environments requiring strict data confidentiality, such as military, government, or highly regulated enterprise applications.

🛡️ Built-in Security Features:

    No External API Calls: Eliminates security risks associated with third-party API data transmission.

    In-Memory Processing: Data is primarily processed in memory, significantly reducing persistent storage of sensitive information.

    Local Caching: Caching mechanisms are entirely local, further enhancing data security.

📋 System Requirements

Minimum Requirements for Basic Usage:

    Python: 3.8+

    RAM: 8GB+ (16GB recommended for basic usage)

    Storage: 10GB+ (for Ollama models and dependencies)

    CPU: 4 cores+ (8 cores recommended)

Recommended Setup for Optimal Performance (SOC Environments):

    RAM: 32GB+

    CPU: 16 cores+

    GPU: NVIDIA GPU (for accelerated Ollama performance with deepseek-coder and other models)

    Storage: SSD (for faster data access and processing of large log files)

🐛 Troubleshooting Common Issues
1. Ollama Connection Failed:

    Check if Ollama server is running:

    ollama list

    Start Ollama server:

    ollama serve

2. Redis Connection Failed:

    Check Redis service status:

    redis-cli ping

    Start Redis service:

    sudo systemctl start redis-server

3. Memory Issues:

    Reduce the size of uploaded log files.

    Increase available RAM or configure a swap file.

    Disable distributed processing if not necessary.

4. Model Loading Issues (deepseek-coder):

    Pull the model again:

    ollama pull deepseek-coder

    List available models to verify:

    ollama list

📄 License

MIT License © 2025 Thanattouth
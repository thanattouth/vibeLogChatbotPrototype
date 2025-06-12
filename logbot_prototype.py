# === Enhanced SOC Analysis AI Assistant ===
import os
import warnings
import streamlit as st
from streamlit_chat import message as st_message
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import redis
from celery import Celery
import hashlib
import json
import asyncio

# Environment configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import libraries
import json
import csv
import re
from io import StringIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.llms import Ollama

# === Constants ===
SUPPORTED_FILE_TYPES = ['.json', '.csv', '.log', '.txt']
SECURITY_KEYWORDS = [
    'failed', 'error', 'denied', 'blocked', 'suspicious', 
    'malware', 'virus', 'attack', 'intrusion', 'unauthorized',
    'breach', 'violation', 'alert', 'warning', 'critical',
    'exploit', 'injection', 'phishing', 'ransomware', 'compromise'
]

THREAT_LEVELS = {
    'critical': {'color': '#FF0000', 'description': 'Immediate action required'},
    'high': {'color': '#FF4500', 'description': 'High priority investigation needed'},
    'medium': {'color': '#FFA500', 'description': 'Should be investigated'},
    'low': {'color': '#FFFF00', 'description': 'Monitor for patterns'},
    'none': {'color': '#008000', 'description': 'No threat detected'}
}

# === Model Loading Functions ===
@st.cache_resource
def load_llm_model():
    """Load the Ollama LLM model with optimized configuration"""
    try:
        return Ollama(
            model="deepseek-coder",
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096
        )
    except Exception as e:
        st.error(f"Failed to load LLM model: {str(e)}")
        return None

@st.cache_resource
def load_embedding_model():
    """Load the HuggingFace embedding model"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

# === Cache and Distributed Processing Setup ===
@st.cache_resource
def setup_redis():
    """Initialize Redis connection for caching"""
    try:
        return redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
    except Exception as e:
        st.warning(f"Redis connection failed: {str(e)}")
        return None

# Initialize Celery without caching
def get_celery_app():
    """Get Celery app instance"""
    return Celery(
        'soc_analysis',
        broker='redis://localhost:6379/1',
        backend='redis://localhost:6379/2'
    )

# Create celery app instance
celery_app = get_celery_app()

def generate_cache_key(data: str) -> str:
    """Generate consistent cache key for data"""
    return hashlib.md5(data.encode('utf-8')).hexdigest()

def cache_log_analysis(logs: List[str], analysis_result: str):
    """Cache log analysis results"""
    if 'redis' not in st.session_state or st.session_state.redis is None:
        return
    
    cache_key = generate_cache_key('\n'.join(logs))
    try:
        st.session_state.redis.setex(
            f"log_analysis:{cache_key}",
            timedelta(hours=24),  # Cache for 24 hours
            json.dumps({
                'logs': logs,
                'analysis': analysis_result,
                'timestamp': datetime.now().isoformat()
            })
        )
    except Exception as e:
        st.warning(f"Cache write failed: {str(e)}")

def get_cached_analysis(logs: List[str]) -> Optional[str]:
    """Retrieve cached analysis if available"""
    if 'redis' not in st.session_state or st.session_state.redis is None:
        return None
    
    cache_key = generate_cache_key('\n'.join(logs))
    try:
        cached_data = st.session_state.redis.get(f"log_analysis:{cache_key}")
        if cached_data:
            return json.loads(cached_data)['analysis']
    except Exception as e:
        st.warning(f"Cache read failed: {str(e)}")
    return None

# === Distributed Task Definitions ===
@celery_app.task
def process_file_distributed(file_content: str, file_type: str) -> Tuple[List[str], Optional[str]]:
    """Celery task for distributed file processing"""
    try:
        if file_type == 'json':
            return process_json_file(file_content)
        elif file_type == 'csv':
            return process_csv_file(file_content)
        elif file_type in ['log', 'txt']:
            return process_text_file(file_content)
        else:
            return [], f"Unsupported file type: {file_type}"
    except Exception as e:
        return [], str(e)
    
# === Helper function to get celery app in session state ===
def get_celery_from_session():
    """Get celery app from session state or create new one"""
    if 'celery_app' not in st.session_state:
        st.session_state.celery_app = celery_app
    return st.session_state.celery_app
    
# === Modified File Processing Functions ===
async def parse_uploaded_file_async(file_obj) -> Tuple[List[str], Optional[str]]:
    """
    Process uploaded log files asynchronously with distributed processing option
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        Tuple of (processed_logs, error_message)
    """
    try:
        # Handle different file object types
        if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
            filename_lower = file_obj.name.lower()
            content = file_obj.read().decode("utf-8")
        elif hasattr(file_obj, 'read'):
            content = file_obj.read().decode("utf-8")
            filename_lower = file_obj.name.lower() if hasattr(file_obj, "name") else ""
        else:
            return [], "Unsupported file object type"
            
        # Check for cached results first
        cache_key = generate_cache_key(content)
        if 'redis' in st.session_state and st.session_state.redis:
            cached_result = get_cached_analysis([content])
            if cached_result:
                return cached_result.split('\n'), "From cache"
        
        # Process based on file type (distributed or local)
        if st.session_state.get('use_distributed_processing', False):
            file_type = filename_lower.split('.')[-1] if '.' in filename_lower else 'txt'
            result = process_file_distributed.delay(content, file_type)
            while not result.ready():
                await asyncio.sleep(0.1)
            return result.get()
        else:
            if filename_lower.endswith('.json'):
                return process_json_file(content)
            elif filename_lower.endswith('.csv'):
                return process_csv_file(content)
            elif filename_lower.endswith(('.log', '.txt')):
                return process_text_file(content)
            else:
                return [], f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
            
    except Exception as e:
        return [], f"Error processing file: {str(e)}"

# === File Processing Functions ===
def parse_uploaded_file(file_obj) -> Tuple[List[str], Optional[str]]:
    """
    Process uploaded log files with improved error handling and support for larger files
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        Tuple of (processed_logs, error_message)
    """
    try:
        # Handle different file object types
        if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
            filename_lower = file_obj.name.lower()
            content = file_obj.read().decode("utf-8")
        elif hasattr(file_obj, 'read'):
            content = file_obj.read().decode("utf-8")
            filename_lower = file_obj.name.lower() if hasattr(file_obj, "name") else ""
        else:
            return [], "Unsupported file object type"
            
        # Process based on file type
        if filename_lower.endswith('.json'):
            return process_json_file(content)
        elif filename_lower.endswith('.csv'):
            return process_csv_file(content)
        elif filename_lower.endswith(('.log', '.txt')):
            return process_text_file(content)
        else:
            return [], f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
            
    except Exception as e:
        return [], f"Error processing file: {str(e)}"

def process_json_file(content: str) -> Tuple[List[str], Optional[str]]:
    """Process JSON files with improved structure handling"""
    try:
        data = json.loads(content)
        log_lines = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    log_lines.append(item.strip())
                elif isinstance(item, dict):
                    log_lines.append(format_json_log(item))
                else:
                    log_lines.append(str(item).strip())
        elif isinstance(data, dict):
            log_lines.append(format_json_log(data))
        else:
            log_lines.append(str(data).strip())
            
        return preprocess_logs(log_lines), None
        
    except json.JSONDecodeError as e:
        return [], f"Invalid JSON format: {str(e)}"

def process_csv_file(content: str) -> Tuple[List[str], Optional[str]]:
    """Process CSV files with improved handling of different formats"""
    try:
        csv_file = StringIO(content)
        reader = csv.DictReader(csv_file)
        log_lines = []
        
        for row in reader:
            # Handle both dict-style and list-style CSV rows
            if isinstance(row, dict):
                formatted_row = " | ".join([f"{k}:{v}" for k, v in row.items() if v])
            else:
                formatted_row = " | ".join([str(field) for field in row if field])
                
            if formatted_row:
                log_lines.append(formatted_row)
                
        return preprocess_logs(log_lines), None
        
    except Exception as e:
        return [], f"CSV processing error: {str(e)}"

def process_text_file(content: str) -> Tuple[List[str], Optional[str]]:
    """Process text/log files with improved line handling"""
    log_lines = [line.strip() for line in content.splitlines() if line.strip()]
    return preprocess_logs(log_lines), None

def format_json_log(json_obj: Dict) -> str:
    """Enhanced JSON log formatting with better field ordering"""
    important_fields = [
        'timestamp', 'time', 'datetime', 'date',
        'source_ip', 'src_ip', 'ip', 'destination_ip', 'dst_ip',
        'user', 'username', 'account', 'email',
        'event', 'action', 'status', 'severity',
        'message', 'error', 'exception', 'stacktrace'
    ]
    
    formatted_parts = []
    
    # Add important fields first
    for field in important_fields:
        if field in json_obj:
            value = json_obj[field]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            formatted_parts.append(f"{field}:{value}")
    
    # Add remaining fields
    for k, v in json_obj.items():
        if k not in important_fields:
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            formatted_parts.append(f"{k}:{v}")
    
    return " | ".join(formatted_parts)

def preprocess_logs(log_lines: List[str]) -> List[str]:
    """Enhanced log preprocessing with more security indicators"""
    processed = []
    
    for line in log_lines:
        if not line.strip():
            continue
            
        enhanced_line = line
        
        # Extract IP addresses (IPv4 and IPv6)
        ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
        ips = re.findall(ipv4_pattern, line) + re.findall(ipv6_pattern, line)
        if ips:
            enhanced_line += f" | EXTRACTED_IPS:{','.join(ips)}"
        
        # Extract and classify timestamps
        timestamp_patterns = [
            (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', 'ISO8601'),
            (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', 'US_DATE'),
            (r'\w{3} \d{2} \d{2}:\d{2}:\d{2}', 'SYSLOG'),
            (r'\d{10,13}', 'UNIX_TS')
        ]
        
        for pattern, fmt in timestamp_patterns:
            matches = re.findall(pattern, line)
            if matches:
                enhanced_line += f" | TIMESTAMP:{matches[0]} | TIMESTAMP_FORMAT:{fmt}"
                break
        
        # Detect security indicators
        found_keywords = [kw for kw in SECURITY_KEYWORDS 
                         if re.search(rf'\b{kw}\b', line, re.IGNORECASE)]
        if found_keywords:
            enhanced_line += f" | SECURITY_INDICATORS:{','.join(found_keywords)}"
        
        processed.append(enhanced_line)
    
    return processed

# === Log Analysis Functions ===
def analyze_logs_with_rag(message: str, logs: List[str]) -> str:
    """Enhanced RAG-based log analysis with better error handling"""
    # Check cache first
    cached_result = get_cached_analysis(logs)
    if cached_result:
        return cached_result + "\n\n🔍 (This analysis was retrieved from cache)"
    
    # Original analysis logic
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        return "⚠️ Please upload log files first"
    
    vectorstore = st.session_state.vectorstore
    llm = st.session_state.llm
    
    # Convert logs to documents with enhanced metadata
    documents = []
    for i, log_line in enumerate(logs):
        if log_line.strip():
            # Extract metadata from log line
            metadata = {
                "line_number": i+1,
                "source": "uploaded_logs",
                "timestamp": extract_timestamp_from_log(log_line),
                "security_indicators": extract_security_indicators(log_line),
                "ip_addresses": extract_ips_from_log(log_line)
            }
            documents.append(Document(page_content=log_line.strip(), metadata=metadata))
    
    if not documents:
        return "⚠️ No valid log data found in uploaded file"

    # Enhanced text splitting with better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n", "|", ";", " ", ""]
    )
    split_docs = splitter.split_documents(documents) or documents
    
    # Create vectorstore if not exists or update if exists
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(split_docs, st.session_state.embedding_model)
    else:
        st.session_state.vectorstore.add_documents(split_docs)

    # Enhanced system prompt for SOC analysis
    system_prompt = """
    You are an advanced AI SOC (Security Operations Center) Analyst specializing in comprehensive cyber threat detection and analysis. Your mission is to provide detailed, actionable insights from security logs with high accuracy.

    Key capabilities:
    - Advanced threat pattern recognition (APT, zero-day, lateral movement)
    - Correlation of events across multiple log sources
    - Timeline reconstruction of security incidents
    - Risk scoring based on MITRE ATT&CK framework
    - Detailed remediation recommendations

    **Strict Response Format:**
    1. **Threat Assessment**:
       - Classification: [Critical/High/Medium/Low/Informational]
       - Confidence: [High/Medium/Low]
       - MITRE ATT&CK Tactic: [Relevant tactic if applicable]
    
    2. **Evidence**:
       - Log excerpts with line numbers
       - Timeline of events
       - Indicators of Compromise (IOCs)
    
    3. **Actionable Recommendations**:
       - Immediate containment
       - Investigation steps
       - Long-term mitigation
    
    4. **Contextual Analysis**:
       - Potential impact
       - Business risk assessment
       - Related historical incidents
    
    Provide responses in clear, structured markdown in Thai language.
    """

    prompt_template = f"""
    {system_prompt}

    **Log Context**:
    {{context}}

    **User Question**:
    {{question}}

    **Required Analysis**:
    - Comprehensive threat evaluation
    - Detailed evidence from logs
    - Actionable recommendations
    - Risk assessment
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Enhanced retrieval configuration
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Max marginal relevance for better diversity
        search_kwargs={
            "k": 8,  # Increased number of retrieved documents
            "score_threshold": 0.7  # Minimum relevance score
        }
    )

    # Configure RAG chain with streaming
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        }
    )

    try:
        # Use callback handler for streaming response
        st_callback = StreamlitCallbackHandler(st.container())
        result = rag_chain.invoke(
            {"query": message},
            config={"callbacks": [st_callback]}
        )
        
        # Enhanced result processing
        response = result.get("result", "No analysis result received")
        source_docs = result.get("source_documents", [])
        
        # Add source references to response
        if source_docs:
            response += "\n\n**References:**\n"
            for i, doc in enumerate(source_docs, 1):
                line_num = doc.metadata.get('line_number', 'N/A')
                response += f"{i}. Line {line_num}: {doc.page_content[:200]}...\n"
        
        cache_log_analysis(logs, response)
        return response
        
    except Exception as e:
        return f"🔴 Analysis error: {str(e)}\n\nPlease check your Ollama server connection"

# === Helper Functions ===
def extract_timestamp_from_log(log_line: str) -> Optional[str]:
    """Extract and normalize timestamp from log line"""
    patterns = [
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
        r'\w{3} \d{2} \d{2}:\d{2}:\d{2}',
        r'\d{10,13}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, log_line)
        if match:
            return match.group()
    return None

def extract_ips_from_log(log_line: str) -> List[str]:
    """Extract IP addresses from log line"""
    ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
    return re.findall(ipv4_pattern, log_line) + re.findall(ipv6_pattern, log_line)

def extract_security_indicators(log_line: str) -> List[str]:
    """Extract security indicators from log line"""
    return [kw for kw in SECURITY_KEYWORDS if re.search(rf'\b{kw}\b', log_line, re.IGNORECASE)]

def generate_log_statistics(logs: List[str]) -> Dict:
    """Generate comprehensive log statistics"""
    stats = {
        'total_entries': len(logs),
        'timestamps': 0,
        'ip_addresses': 0,
        'security_alerts': 0,
        'error_messages': 0
    }
    
    for log in logs:
        if extract_timestamp_from_log(log):
            stats['timestamps'] += 1
        if extract_ips_from_log(log):
            stats['ip_addresses'] += 1
        if extract_security_indicators(log):
            stats['security_alerts'] += 1
        if 'error' in log.lower():
            stats['error_messages'] += 1
            
    return stats

# === Visualization Functions ===
def plot_timeline(logs: List[str]):
    """Create an interactive timeline visualization"""
    timeline_data = []
    
    for i, log in enumerate(logs):
        ts = extract_timestamp_from_log(log)
        if ts:
            try:
                # Try to parse different timestamp formats
                if re.match(r'\d{10,13}', ts):  # Unix timestamp
                    dt = datetime.fromtimestamp(int(ts[:10]))
                else:
                    dt = datetime.strptime(ts.split('.')[0], '%Y-%m-%d %H:%M:%S')
                
                timeline_data.append({
                    'timestamp': dt,
                    'content': log[:100] + '...' if len(log) > 100 else log,
                    'line_number': i+1,
                    'has_security': bool(extract_security_indicators(log))
                })
            except:
                continue
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        fig = px.scatter(
            df,
            x='timestamp',
            y='line_number',
            color='has_security',
            hover_data=['content'],
            title='Log Entry Timeline',
            labels={'timestamp': 'Time', 'line_number': 'Log Sequence'}
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No timestamp data available for timeline visualization")

def plot_threat_distribution(logs: List[str]):
    """Visualize threat distribution in logs"""
    threat_counts = {kw: 0 for kw in SECURITY_KEYWORDS}
    
    for log in logs:
        for kw in extract_security_indicators(log):
            threat_counts[kw] += 1
    
    threat_data = [{'keyword': k, 'count': v} for k, v in threat_counts.items() if v > 0]
    
    if threat_data:
        df = pd.DataFrame(threat_data)
        fig = px.bar(
            df,
            x='keyword',
            y='count',
            title='Security Threat Distribution',
            color='count',
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No security threats detected in logs")

# === Streamlit UI ===
def initialize_session_state():
    """Initialize all session state variables"""
    if 'logs' not in st.session_state:
        st.session_state.logs = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'llm' not in st.session_state:
        st.session_state.llm = load_llm_model()
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = load_embedding_model()
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "basic"
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = ""
    if 'redis' not in st.session_state:
        st.session_state.redis = setup_redis()
    if 'use_distributed_processing' not in st.session_state:
        st.session_state.use_distributed_processing = False

def render_file_upload_section():
    """Render the file upload section with enhanced UI"""
    with st.expander("📁 Upload Log Files", expanded=True):
        uploaded_file = st.file_uploader(
            "Drag and drop or click to select log files",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📤 Process Files", type="primary"):
                process_uploaded_files(uploaded_file)
        with col2:
            st.session_state.analysis_mode = st.selectbox(
                "Analysis Mode",
                ["Basic", "Advanced", "Forensic"],
                key="mode_select"
            )
        
        # Distributed processing toggle
        st.session_state.use_distributed_processing = st.checkbox(
            "Enable distributed processing (Celery)",
            value=st.session_state.use_distributed_processing
        )

def process_uploaded_files(uploaded_files):
    """Process uploaded files with progress feedback"""
    if not uploaded_files:
        st.warning("Please select at least one file")
        return
        
    with st.spinner("Processing files..."):
        all_logs = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            if st.session_state.use_distributed_processing:
                logs, error = asyncio.run(parse_uploaded_file_async(uploaded_file))
            else:
                logs, error = parse_uploaded_file(uploaded_file)
            if error:
                st.error(f"Error in {uploaded_file.name}: {error}")
            else:
                all_logs.extend(logs)
        
        if all_logs:
            st.session_state.logs = all_logs
            create_vectorstore(all_logs)
            
            # Show statistics
            stats = generate_log_statistics(all_logs)
            st.success(f"✅ Processed {len(uploaded_files)} file(s) with {stats['total_entries']:,} log entries")
            
            # Display quick stats
            st.subheader("📊 Quick Statistics")
            cols = st.columns(4)
            cols[0].metric("Total Entries", stats['total_entries'])
            cols[1].metric("With Timestamps", stats['timestamps'])
            cols[2].metric("With IPs", stats['ip_addresses'])
            cols[3].metric("Security Alerts", stats['security_alerts'])
                
            # Visualizations
            plot_timeline(all_logs)
            plot_threat_distribution(all_logs)

def create_vectorstore(logs: List[str]):
    """Create or update the vectorstore with new logs"""
    documents = []
    for i, log_line in enumerate(logs):
        if log_line.strip():
            metadata = {
                "line_number": i+1,
                "source": "uploaded_logs",
                "timestamp": extract_timestamp_from_log(log_line),
                "security_indicators": extract_security_indicators(log_line),
                "ip_addresses": extract_ips_from_log(log_line)
            }
            documents.append(Document(page_content=log_line.strip(), metadata=metadata))
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n", "|", ";", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(
            split_docs,
            st.session_state.embedding_model
        )
    else:
        st.session_state.vectorstore.add_documents(split_docs)

def render_chat_interface():
    """Render the chat interface with enhanced features"""
    st.subheader("💬 SOC Analyst Chat")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        st_message(**chat, key=str(i))
    
    # Example questions OUTSIDE the form
    st.markdown("💡 **Example Questions:**")
    examples = [
        "Identify potential security threats",
        "Show suspicious IPs", 
        "Analyze failed logins",
        "Timeline of events",
        "Most critical threat?"
    ]
    
    # Create columns for example buttons
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.selected_example = example
            st.rerun()
    
    # Chat input form
    with st.form(key='chat_form', clear_on_submit=True):
        # Pre-fill with selected example if any
        default_text = st.session_state.selected_example if st.session_state.selected_example else ""
        user_input = st.text_area(
            "Enter your security analysis question...",
            key="user_input",
            height=100,
            placeholder="E.g., 'Are there any brute force attack patterns?'",
            value=default_text
        )
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            submit_button = st.form_submit_button("🚀 Analyze", type="primary")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear Chat")
        with col3:
            export_button = st.form_submit_button("📤 Export")
    
    # Handle form submissions
    if submit_button and user_input:
        # Clear the selected example after submission
        st.session_state.selected_example = ""
        handle_user_query(user_input)
    elif clear_button:
        st.session_state.chat_history = []
        st.session_state.selected_example = ""
        st.rerun()
    elif export_button:
        export_chat_history()

def handle_user_query(query: str):
    """Process user query with enhanced analysis"""
    if not st.session_state.logs:
        st.warning("Please upload log files first")
        return
        
    # Add user message to history
    st.session_state.chat_history.append({
        "message": query,
        "is_user": True,
        "avatar_style": "adventurer-neutral"
    })
    
    # Get AI response
    with st.spinner("Analyzing logs..."):
        response = analyze_logs_with_rag(query, st.session_state.logs)
    
    # Add AI response to history
    st.session_state.chat_history.append({
        "message": response,
        "is_user": False,
        "avatar_style": "bottts"
    })
    
    st.rerun()

def export_chat_history():
    """Export chat history to a markdown file"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
        
    markdown_content = "# SOC Analysis Chat History\n\n"
    markdown_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for chat in st.session_state.chat_history:
        role = "User" if chat['is_user'] else "SOC Analyst"
        markdown_content += f"## {role}\n\n{chat['message']}\n\n"
    
    st.download_button(
        label="📥 Download Chat History",
        data=markdown_content,
        file_name=f"soc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

# === Main Application ===
def main():
    # Configure page
    st.set_page_config(
        page_title="🛡️ Advanced SOC Analysis AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stButton button {
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: scale(1.05);
        }
        .stTextArea textarea {
            min-height: 100px;
        }
        .css-1aumxhk {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
        }
        .stAlert {
            border-radius: 10px;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Main layout
    st.title("🛡️ Advanced SOC Analysis AI")
    st.markdown("""
    ### 🚀 AI-Powered Security Log Analysis
    Upload security logs and get intelligent threat detection and analysis powered by Ollama and LangChain.
    """)
    
    # Add Ollama server status check
    with st.expander("🔧 Ollama Server Configuration", expanded=False):
        st.info("""
        **Note:** This application requires a local Ollama server running with the deepseek-coder.
        
        To set up:
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Run `ollama pull deepseek-coder` to download the model
        3. Ensure the Ollama server is running (`ollama serve`)
        """)
        
        if st.button("Check Ollama Connection"):
            try:
                if st.session_state.llm:
                    test_response = st.session_state.llm("Test connection")
                    st.success("✅ Ollama connection successful!")
                    st.code(test_response[:200] + "...", language="text")
                else:
                    st.error("❌ Ollama connection failed - LLM not initialized")
            except Exception as e:
                st.error(f"❌ Ollama connection failed: {str(e)}")
    
    # File upload and visualization section
    render_file_upload_section()
    
    # Chat interface section
    st.markdown("---")
    render_chat_interface()

if __name__ == "__main__":
    main()
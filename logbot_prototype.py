# === โหลดไลบรารีที่จำเป็น ===
import gradio as gr                           # สำหรับสร้าง Web UI
import json                                   # จัดการไฟล์ JSON
import csv                                    # อ่านไฟล์ CSV
import re                                     # Regular Expression สำหรับค้นหาแพทเทิร์น
from io import StringIO                       # จัดการ string เป็น file object
from datetime import datetime                 # จัดการวันที่เวลา
from langchain_community.vectorstores import FAISS           # Vector database สำหรับค้นหาข้อมูล
from langchain_huggingface import HuggingFaceEmbeddings      # แปลงข้อความเป็น vector
from langchain_ollama import OllamaLLM as Ollama             # LLM รันภายในเครื่อง
from langchain.chains import RetrievalQA                     # RAG chain สำหรับค้นหาและตอบคำถาม
from langchain.text_splitter import RecursiveCharacterTextSplitter  # แบ่งข้อความยาว
from langchain_core.documents import Document                # Document object สำหรับ LangChain
from langchain.prompts import PromptTemplate                 # Template สำหรับ prompt

# === เริ่มต้น AI Models ===
# สร้าง embedding model สำหรับแปลงข้อความเป็นตัวเลข (vector)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# สร้าง LLM ที่ถูกปรับแต่งเป็นผู้เชี่ยวชาญ SOC
llm = Ollama(
    model="phi3:mini",                        # ใช้โมเดล Phi3 ขนาดเล็ก
    temperature=0.1,                          # ค่าต่ำ = คำตอบที่สม่ำเสมอและแม่นยำ
    system_message="""
    You are an expert SOC (Security Operations Center) Analyst AI specialized in cybersecurity threat detection.
    Your expertise includes:
    - Detecting brute-force attacks, DDoS, suspicious login patterns
    - Identifying malicious IPs and unusual network behavior
    - Analyzing authentication failures and privilege escalation
    - Recognizing malware signatures and data exfiltration attempts
    
    Always provide:
    1. Clear threat classification (Critical/High/Medium/Low)
    2. Specific evidence from logs with timestamps
    3. Recommended immediate actions
    4. Risk assessment and potential impact
    
    Be precise, actionable, and security-focused.
    """
)

# === ฟังก์ชันจัดการไฟล์ ===
def parse_uploaded_file(file_obj):
    """
    ฟังก์ชันหลักสำหรับอ่านและแปลงไฟล์ที่อัปโหลด
    รองรับ: JSON, CSV, LOG, TXT
    """
    try:
        # === ตรวจสอบชนิดของ file object และอ่านเนื้อหา ===
        if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
            # กรณีมี path ของไฟล์
            filename_lower = file_obj.name.lower()
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                content = f.read()
        elif hasattr(file_obj, 'read'):
            # กรณีเป็น file object ที่อ่านได้
            content = file_obj.read().decode("utf-8")
            filename_lower = file_obj.name.lower() if hasattr(file_obj, "name") else ""
        elif isinstance(file_obj, str):
            # กรณีเป็น string path
            filename_lower = file_obj.lower()
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # กรณีอื่นๆ แปลงเป็น string
            file_path = str(file_obj)
            filename_lower = file_path.lower()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        return [], f"🔴 ไม่สามารถอ่านไฟล์ได้: {str(e)}"

    log_lines = []  # เก็บข้อมูล log ที่ประมวลผลแล้ว

    # === ประมวลผลตามชนิดไฟล์ ===
    if filename_lower.endswith('.json'):
        # === จัดการไฟล์ JSON ===
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # กรณีเป็น array ของ objects
                for item in data:
                    if isinstance(item, str):
                        log_lines.append(item.strip())
                    elif isinstance(item, dict):
                        # แปลง JSON object เป็น log format
                        formatted_line = format_json_log(item)
                        log_lines.append(formatted_line)
                    else:
                        log_lines.append(str(item).strip())
            elif isinstance(data, dict):
                # กรณีเป็น object เดียว
                formatted_line = format_json_log(data)
                log_lines.append(formatted_line)
            else:
                log_lines.append(str(data).strip())
        except Exception as e:
            return [], f"🔴 ไฟล์ JSON ไม่ถูกต้อง: {str(e)}"

    elif filename_lower.endswith('.csv'):
        # === จัดการไฟล์ CSV ===
        try:
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)
            for row in reader:
                # แปลงแต่ละแถวเป็น log format (key:value | key:value)
                formatted_row = " | ".join([f"{k}:{v}" for k, v in row.items() if v])
                if formatted_row:
                    log_lines.append(formatted_row)
        except Exception as e:
            return [], f"🔴 ไฟล์ CSV ไม่ถูกต้อง: {str(e)}"

    elif filename_lower.endswith(('.log', '.txt')) or not filename_lower:
        # === จัดการไฟล์ข้อความธรรมดา ===
        log_lines = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        return [], "⚠️ รูปแบบไฟล์ไม่รองรับ ใช้ .json, .csv, .log, หรือ .txt"

    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if not log_lines:
        return [], "⚠️ ไม่พบข้อมูล log ในไฟล์ที่อัปโหลด"

    # ประมวลผลและเสริมข้อมูล log
    processed_logs = preprocess_logs(log_lines)
    return processed_logs, None

def format_json_log(json_obj):
    """
    แปลง JSON object เป็น log format ที่อ่านง่าย
    โดยจัดลำดับความสำคัญของ fields
    """
    # Fields ที่สำคัญสำหรับการวิเคราะห์ความปลอดภัย
    important_fields = ['timestamp', 'time', 'datetime', 'source_ip', 'src_ip', 'ip', 
                       'user', 'username', 'event', 'action', 'status', 'message', 'error']
    
    formatted_parts = []
    
    # เพิ่ม important fields ก่อน
    for field in important_fields:
        if field in json_obj:
            formatted_parts.append(f"{field}:{json_obj[field]}")
    
    # เพิ่ม fields อื่นๆ
    for k, v in json_obj.items():
        if k not in important_fields:
            formatted_parts.append(f"{k}:{v}")
    
    return " | ".join(formatted_parts)

def preprocess_logs(log_lines):
    """
    เสริมข้อมูลให้กับ log entries เพื่อช่วยในการวิเคราะห์
    เพิ่ม: IP addresses, timestamps, security keywords
    """
    processed = []
    
    for line in log_lines:
        if not line.strip():
            continue
            
        enhanced_line = line
        
        # === ดึง IP Addresses ===
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # RegEx สำหรับ IPv4
        ips = re.findall(ip_pattern, line)
        if ips:
            enhanced_line += f" | EXTRACTED_IPS:{','.join(ips)}"
        
        # === ดึง Timestamps ในรูปแบบต่างๆ ===
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',    # ISO format: 2024-01-01T12:00:00
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',        # US format: 01/01/2024 12:00:00
            r'\w{3} \d{2} \d{2}:\d{2}:\d{2}'               # Syslog format: Jan 01 12:00:00
        ]
        
        for pattern in timestamp_patterns:
            timestamps = re.findall(pattern, line)
            if timestamps:
                enhanced_line += f" | TIMESTAMP:{timestamps[0]}"
                break
        
        # === ค้นหา Security Keywords ===
        security_keywords = ['failed', 'error', 'denied', 'blocked', 'suspicious', 
                           'malware', 'virus', 'attack', 'intrusion', 'unauthorized',
                           'breach', 'violation', 'alert', 'warning']
        
        found_keywords = [kw for kw in security_keywords if kw.lower() in line.lower()]
        if found_keywords:
            enhanced_line += f" | SECURITY_INDICATORS:{','.join(found_keywords)}"
        
        processed.append(enhanced_line)
    
    return processed

# === ฟังก์ชันวิเคราะห์หลัก ===
def analyze_logs(message, logs):
    """
    ฟังก์ชันหลักสำหรับวิเคราะห์ log ด้วย AI
    ใช้ RAG (Retrieval-Augmented Generation) approach
    """
    # ตรวจสอบว่ามี log data หรือไม่
    if logs is None or not logs:
        return "📁 กรุณาอัปโหลดไฟล์ log ก่อนถามคำถาม"

    # === สร้าง Documents สำหรับ LangChain ===
    documents = []
    for i, log_line in enumerate(logs):
        if log_line.strip():
            documents.append(Document(
                page_content=log_line.strip(),           # เนื้อหา log
                metadata={"line_number": i+1, "source": "uploaded_logs"}  # metadata
            ))
    
    if not documents:
        return "⚠️ ไม่พบข้อมูล log ที่สามารถใช้ได้ในไฟล์ที่อัปโหลด"

    # === แบ่งข้อความยาวเป็นชิ้นเล็ก ===
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # ขนาดแต่ละชิ้น (characters)
        chunk_overlap=50,      # ความซ้ำซ้อนระหว่างชิ้น
        separators=["\n", "|", " ", ""]  # ตัวแบ่งตามลำดับความสำคัญ
    )
    split_docs = splitter.split_documents(documents)

    if not split_docs:
        split_docs = documents

    # === สร้าง Vector Store สำหรับการค้นหา ===
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    # === สร้าง Prompt Template ===
    template = """
    คุณกำลังวิเคราะห์ log ความปลอดภัยในฐานะนักวิเคราะห์ SOC โปรดให้การวิเคราะห์ที่ครอบคลุมจาก log entries ที่ค้นพบ
    
    ข้อมูลจาก logs:
    {context}
    
    คำถาม: {question}
    
    โปรดให้การวิเคราะห์ในรูปแบบนี้:
    
    🔍 การวิเคราะห์ภัยคุกคาม:
    - ระดับภัยคุกคาม: [Critical/High/Medium/Low]
    - ประเภทภัยคุกคาม: [เช่น Brute Force, DDoS, Malware, ฯลฯ]
    
    📋 ผลการตรวจสอบ:
    - แสดง log entries ที่น่าสงสัยพร้อม timestamps
    - เน้นรูปแบบหรือพฤติกรรมที่ผิดปกติ
    
    🎯 หลักฐาน:
    - IP ต้นทางที่เกี่ยวข้อง
    - วิธีการโจมตีที่ระบุได้
    - ลำดับเวลาของเหตุการณ์
    
    ⚡ การดำเนินการที่แนะนำ:
    - ขั้นตอนการตอบสนองทันที
    - มาตรการรักษาความปลอดภัยระยะยาว
    
    🔗 ตัวบ่งชี้ที่เกี่ยวข้อง:
    - IOCs เพิ่มเติมที่ควรติดตาม
    - การประเมินผลกระทบที่อาจเกิดขึ้น
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # === สร้าง RAG Chain ===
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,                                          # AI model
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # ค้นหา 5 log entries ที่เกี่ยวข้องที่สุด
        chain_type="stuff",                               # วิธีการส่ง context
        return_source_documents=True,                     # ส่งคืน source documents ด้วย
        chain_type_kwargs={"prompt": prompt}              # ใช้ prompt template ที่เรากำหนด
    )

    # === รัน AI Analysis ===
    try:
        result = rag_chain.invoke({"query": message})
        return result.get("result", "⚠️ ไม่ได้รับผลการวิเคราะห์จาก AI")
    except Exception as e:
        return f"🔴 เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

# === ฟังก์ชันสำหรับการแชท ===
def process_message(message, history, logs):
    """
    ประมวลผลข้อความใหม่และอัปเดต chat history
    คล้าย ChatGPT interface
    """
    if not message.strip():
        return history, ""
    
    # เพิ่มข้อความของผู้ใช้เข้าไปในประวัติ (ยังไม่มีคำตอบ)
    history.append([message, None])
    
    # วิเคราะห์และสร้างคำตอบ
    response = analyze_logs(message, logs)
    
    # อัปเดตคำตอบในประวัติ
    history[-1][1] = response
    
    return history, ""  # คืนค่า history ใหม่ และล้างกล่องข้อความ

def upload_and_process_file(file):
    """
    ประมวลผลไฟล์ที่อัปโหลดและแสดงสถานะ
    """
    if file is None:
        return None, "❌ กรุณาเลือกไฟล์ที่ต้องการอัปโหลด"
    
    # ประมวลผลไฟล์
    logs, error = parse_uploaded_file(file)
    if error:
        return None, error
    
    # แสดงสถานะสำเร็จ
    log_count = len(logs)
    success_message = f"✅ อัปโหลดสำเร็จ! พบ {log_count:,} รายการ log พร้อมสำหรับการวิเคราะห์"
    
    return logs, success_message

# === สร้าง Web UI ด้วย Gradio ===
with gr.Blocks(
    title="🛡️ SOC Analysis AI", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    .chat-container {
        height: 600px !important;
    }
    .file-upload {
        border: 2px dashed #ccc !important;
        border-radius: 10px !important;
        padding: 20px !important;
        text-align: center !important;
    }
    """
) as demo:
    
    # === State Management ===
    logs_state = gr.State(None)  # เก็บข้อมูล log ในหน่วยความจำ
    
    # === หัวข้อหลัก ===
    gr.Markdown("""
    # 🛡️ **SOC Analysis AI Assistant**
    อัปโหลดไฟล์ log และถามคำถามเกี่ยวกับความปลอดภัย
    """)
    
    # === ส่วนอัปโหลดไฟล์ ===
    with gr.Row():
        with gr.Column(scale=3):
            file_upload = gr.File(
                label="📁 ลากไฟล์มาวางหรือคลิกเพื่อเลือก",
                file_types=['.json', '.csv', '.log', '.txt'],  # รองรับเฉพาะไฟล์เหล่านี้
                elem_classes="file-upload"
            )
    with gr.Column(scale=1):
        upload_btn = gr.Button("📤 อัปโหลด", variant="primary", size="lg")
    
    # === แสดงสถานะการอัปโหลด ===
    upload_status = gr.Textbox(
        label="📊 สถานะ", 
        interactive=False,
        value="🔄 รอการอัปโหลดไฟล์...",
        show_label=False
    )
    
    gr.Markdown("---")
    
    # === ส่วน Chat Interface ===
    with gr.Column():
        chatbot = gr.Chatbot(
            label="💬 แชทกับ SOC Analyst AI",
            height=500,
            show_label=False,
            placeholder="การสนทนาจะปรากฏที่นี่...",
            elem_classes="chat-container"
        )
        
        # กล่องข้อความและปุ่มส่ง
        with gr.Row():
            message_box = gr.Textbox(
                placeholder="พิมพ์คำถามเกี่ยวกับ log ของคุณ... (เช่น 'มีการโจมตี brute force หรือไม่?')",
                container=False,
                scale=4,
                show_label=False
            )
            send_button = gr.Button("📨", variant="primary", scale=1, min_width=50)
    
    # === ตัวอย่างคำถาม ===
    with gr.Row():
        gr.Examples(
            examples=[
                "มีการโจมตี brute force หรือไม่?",
                "IP ไหนที่น่าสงสัยที่สุด?", 
                "มีความผิดปกติในการเข้าสู่ระบบหรือไม่?",
                "วิเคราะห์ภัยคุกคามโดยรวม",
                "แสดงสถิติการโจมตีตามเวลา"
            ],
            inputs=message_box,
            label="💡 ตัวอย่างคำถาม"
        )
    
    # === เชื่อมต่อ Events ===
    
    # เมื่อกดปุ่มอัปโหลด
    upload_btn.click(
        fn=upload_and_process_file,      # ฟังก์ชันที่จะรัน
        inputs=file_upload,              # input คือไฟล์ที่เลือก
        outputs=[logs_state, upload_status]  # output คือข้อมูล log และสถานะ
    )
    
    # ฟังก์ชันสำหรับส่งข้อความ
    def submit_message(message, history, logs):
        return process_message(message, history, logs)
    
    # เมื่อกดปุ่มส่งข้อความ
    send_button.click(
        fn=submit_message,
        inputs=[message_box, chatbot, logs_state],
        outputs=[chatbot, message_box]
    )
    
    # เมื่อกด Enter ในกล่องข้อความ
    message_box.submit(
        fn=submit_message,
        inputs=[message_box, chatbot, logs_state],
        outputs=[chatbot, message_box]
    )

# === รัน Application ===
if __name__ == "__main__":
    demo.launch(
        share=False,        # ไม่แชร์ออนไลน์
        server_name="0.0.0.0",  # รับ connection จากทุก IP
        server_port=7860,   # พอร์ต
        show_error=True     # แสดง error ใน browser
    )
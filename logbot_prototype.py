# === โหลดไลบรารีที่จำเป็น ===
import gradio as gr  # สำหรับสร้าง Web UI
import json  # สำหรับจัดการไฟล์ JSON
import csv   # สำหรับจัดการไฟล์ CSV
import re    # สำหรับ Regular Expression
from io import StringIO  # สำหรับจัดการ string เป็น file
from datetime import datetime  # สำหรับจัดการวันที่และเวลา
from langchain_community.vectorstores import FAISS  # Vector database สำหรับเก็บข้อมูล
from langchain_huggingface import HuggingFaceEmbeddings  # Model สำหรับแปลงข้อความเป็น vector
from langchain_ollama import OllamaLLM as Ollama  # LLM Model จาก Ollama
from langchain.chains import RetrievalQA  # Chain สำหรับ RAG (Retrieval Augmented Generation)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # สำหรับแบ่งข้อความ
from langchain_core.documents import Document  # Class สำหรับจัดเก็บเอกสาร
from langchain.prompts import PromptTemplate  # สำหรับสร้าง prompt template

# === STEP 1: เริ่มต้น embedding model และ LLM ===
# สร้าง embedding model สำหรับแปลงข้อความเป็น vector เพื่อค้นหา
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# สร้าง LLM model ด้วย Ollama พร้อมตั้งค่าสำหรับการวิเคราะห์ความปลอดภัย
llm = Ollama(
    model="phi3:mini",  # ใช้โมเดล phi3:mini
    temperature=0.1,   # ลดความสุ่มเพื่อความแม่นยำในการตอบ
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

# === STEP 2: ฟังก์ชันสำหรับแปลงไฟล์และจัดการ pattern recognition ===
def parse_uploaded_file(file_obj):
    """
    ฟังก์ชันสำหรับอ่านและแปลงไฟล์ที่อัปโหลด
    รองรับไฟล์ประเภท: JSON, CSV, LOG, TXT
    """
    try:
        # จัดการกับรูปแบบไฟล์ Gradio ที่แตกต่างกัน
        if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
            # รูปแบบ Gradio ใหม่ - file_obj เป็น path ของไฟล์
            filename_lower = file_obj.name.lower()
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                content = f.read()
        elif hasattr(file_obj, 'read'):
            # รูปแบบ Gradio เก่า - file_obj มี method read
            content = file_obj.read().decode("utf-8")
            filename_lower = file_obj.name.lower() if hasattr(file_obj, "name") else ""
        elif isinstance(file_obj, str):
            # ไฟล์ path เป็น string
            filename_lower = file_obj.lower()
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # พยายามดึง path จาก object
            file_path = str(file_obj)
            filename_lower = file_path.lower()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        return [], f"🔴 ไม่สามารถอ่านไฟล์ได้: {str(e)}"

    log_lines = []

    # จัดการไฟล์ JSON
    if filename_lower.endswith('.json'):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # กรณีที่ JSON เป็น array
                for item in data:
                    if isinstance(item, str):
                        log_lines.append(item.strip())
                    elif isinstance(item, dict):
                        # แปลง JSON object เป็นรูปแบบ log ที่อ่านง่าย
                        formatted_line = format_json_log(item)
                        log_lines.append(formatted_line)
                    else:
                        log_lines.append(str(item).strip())
            elif isinstance(data, dict):
                # กรณีที่ JSON เป็น object เดียว
                formatted_line = format_json_log(data)
                log_lines.append(formatted_line)
            else:
                log_lines.append(str(data).strip())
        except Exception as e:
            return [], f"🔴 ไฟล์ JSON ไม่ถูกต้อง: {str(e)}"

    # จัดการไฟล์ CSV
    elif filename_lower.endswith('.csv'):
        try:
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames
            for row in reader:
                # แปลง CSV row เป็นรูปแบบ log entry
                formatted_row = " | ".join([f"{k}:{v}" for k, v in row.items() if v])
                if formatted_row:
                    log_lines.append(formatted_row)
        except Exception as e:
            return [], f"🔴 ไฟล์ CSV ไม่ถูกต้อง: {str(e)}"

    # จัดการไฟล์ LOG และ TXT
    elif filename_lower.endswith(('.log', '.txt')) or not filename_lower:
        log_lines = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        return [], "⚠️ รูปแบบไฟล์ไม่รองรับ ใช้ .json, .csv, .log, หรือ .txt"

    if not log_lines:
        return [], "⚠️ ไม่พบข้อมูล log ในไฟล์ที่อัปโหลด"

    # ปรับปรุงการประมวลผล log เพื่อการวิเคราะห์ที่ดีขึ้น
    processed_logs = preprocess_logs(log_lines)
    return processed_logs, None

def format_json_log(json_obj):
    """
    ฟังก์ชันสำหรับจัดรูปแบบ JSON log entry ให้อ่านง่ายขึ้น
    จัดลำดับฟิลด์สำคัญไว้ด้านหน้า
    """
    # ฟิลด์ที่สำคัญสำหรับการวิเคราะห์ความปลอดภัย
    important_fields = ['timestamp', 'time', 'datetime', 'source_ip', 'src_ip', 'ip', 
                       'user', 'username', 'event', 'action', 'status', 'message', 'error']
    
    formatted_parts = []
    # เพิ่มฟิลด์สำคัญก่อน
    for field in important_fields:
        if field in json_obj:
            formatted_parts.append(f"{field}:{json_obj[field]}")
    
    # เพิ่มฟิลด์ที่เหลือ
    for k, v in json_obj.items():
        if k not in important_fields:
            formatted_parts.append(f"{k}:{v}")
    
    return " | ".join(formatted_parts)

def preprocess_logs(log_lines):
    """
    ฟังก์ชันสำหรับปรับปรุง log ให้เหมาะสำหรับการวิเคราะห์ความปลอดภัย
    เพิ่มข้อมูลที่สำคัญเช่น IP address, timestamp, คำสำคัญด้านความปลอดภัย
    """
    processed = []
    
    for line in log_lines:
        if not line.strip():
            continue
            
        # เพิ่มข้อมูลสำคัญเข้าไปใน log
        enhanced_line = line
        
        # ดึง IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = re.findall(ip_pattern, line)
        if ips:
            enhanced_line += f" | EXTRACTED_IPS:{','.join(ips)}"
        
        # ดึง timestamps หลายรูปแบบ
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # 2024-01-01T10:30:45 หรือ 2024-01-01 10:30:45
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',      # 01/01/2024 10:30:45
            r'\w{3} \d{2} \d{2}:\d{2}:\d{2}'            # Jan 01 10:30:45
        ]
        
        for pattern in timestamp_patterns:
            timestamps = re.findall(pattern, line)
            if timestamps:
                enhanced_line += f" | TIMESTAMP:{timestamps[0]}"
                break
        
        # ทำเครื่องหมายเหตุการณ์ที่อาจเป็นปัญหาความปลอดภัย
        security_keywords = ['failed', 'error', 'denied', 'blocked', 'suspicious', 
                           'malware', 'virus', 'attack', 'intrusion', 'unauthorized',
                           'breach', 'violation', 'alert', 'warning']
        
        found_keywords = [kw for kw in security_keywords if kw.lower() in line.lower()]
        if found_keywords:
            enhanced_line += f" | SECURITY_INDICATORS:{','.join(found_keywords)}"
        
        processed.append(enhanced_line)
    
    return processed

# === STEP 3: ฟังก์ชันวิเคราะห์ log ด้วย AI - เวอร์ชันที่แก้ไขแล้ว ===
def analyze_logs(message, history, logs):
    """
    ฟังก์ชันหลักสำหรับวิเคราะห์ log ด้วย AI
    ใช้ RAG (Retrieval Augmented Generation) เพื่อค้นหาข้อมูลที่เกี่ยวข้องและวิเคราะห์
    
    Args:
        message: ข้อความคำถามจากผู้ใช้
        history: ประวัติการสนทนา (list of tuples)
        logs: ข้อมูล log ที่จะวิเคราะห์
    """
    if logs is None or not logs:
        return "📝 กรุณาอัปโหลดไฟล์ log ก่อนครับ"

    # ดึงคำถามปัจจุบันจากข้อความ
    question = message

    # สร้าง documents พร้อม metadata สำหรับแต่ละบรรทัด log
    documents = []
    for i, log_line in enumerate(logs):
        if log_line.strip():
            documents.append(Document(
                page_content=log_line.strip(),
                metadata={"line_number": i+1, "source": "uploaded_logs"}
            ))
    
    if not documents:
        return "⚠️ ไม่พบข้อมูล log ที่สามารถใช้ได้ในไฟล์ที่อัปโหลด"

    # ใช้ recursive splitter เพื่อแบ่งข้อมูลให้เหมาะสม
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # ขนาดชิ้นข้อมูลแต่ละชิ้น
        chunk_overlap=50,    # ส่วนที่ทับซ้อนกันระหว่างชิ้นข้อมูล
        separators=["\n", "|", " ", ""]  # ตัวแบ่งข้อมูล
    )
    split_docs = splitter.split_documents(documents)

    if not split_docs:
        split_docs = documents

    # สร้าง vector store สำหรับการค้นหาข้อมูลที่เกี่ยวข้อง
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    # สร้าง prompt template ที่ปรับปรุงแล้วสำหรับการวิเคราะห์ความปลอดภัย
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

    # สร้าง RAG chain พร้อม custom prompt
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # ค้นหา 5 ชิ้นข้อมูลที่เกี่ยวข้องที่สุด
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    try:
        # รันการวิเคราะห์
        result = rag_chain.invoke({"query": question})
        return result.get("result", "⚠️ ไม่ได้รับผลการวิเคราะห์จาก AI")
    except Exception as e:
        return f"🔴 เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

# === STEP 4: Enhanced UI พร้อมการติดตามสถานะที่ดีขึ้น ===
log_state = gr.State()         # เก็บสถานะของ log ที่โหลด
analysis_history = gr.State([])  # เก็บประวัติการวิเคราะห์

def load_log_file(file_obj):
    """
    ฟังก์ชันสำหรับโหลดไฟล์ log และแสดงสถานะ
    """
    if file_obj is None:
        return "📝 กรุณาอัปโหลดไฟล์ log ก่อนครับ", None, "📊 ยังไม่มี logs"
    
    # แปลงไฟล์และจัดการข้อมูล
    logs, error = parse_uploaded_file(file_obj)
    if error:
        return error, None, "❌ โหลดไม่สำเร็จ"
    
    log_count = len(logs)
    status_msg = f"✅ โหลด {log_count} รายการ log สำเร็จแล้ว! พร้อมสำหรับการวิเคราะห์!"
    stats_msg = f"📊 โหลด {log_count} รายการ log แล้ว"
    
    return status_msg, logs, stats_msg

def get_log_statistics(logs):
    """
    ฟังก์ชันสำหรับคำนวณและแสดงสถิติของ log
    """
    if not logs:
        return "📊 ยังไม่มี logs"
    
    total_lines = len(logs)
    # นับจำนวน IP ที่ไม่ซ้ำกัน
    ip_count = len(set(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' '.join(logs))))
    
    # นับเหตุการณ์ที่อาจเป็นปัญหาความปลอดภัย
    security_keywords = ['failed', 'error', 'denied', 'blocked', 'suspicious']
    security_events = sum(1 for log in logs if any(kw in log.lower() for kw in security_keywords))
    
    return f"📊 สถิติ: {total_lines} รายการ | {ip_count} IP ไม่ซ้ำ | {security_events} เหตุการณ์ที่อาจเป็นปัญหาความปลอดภัย"

# === STEP 5: สร้าง Gradio Interface ที่ปรับปรุงแล้ว ===
with gr.Blocks(title="🛡️ SOC Analysis Chatbot", theme=gr.themes.Soft()) as demo:
    # หัวข้อและคำอธิบาย
    gr.Markdown("""
    # 🛡️ **ระบบวิเคราะห์ Log ความปลอดภัยด้วย AI**
    อัปโหลดไฟล์ log ความปลอดภัย (`.json`, `.csv`, `.log`, `.txt`) และรับการวิเคราะห์ภัยคุกคามจากผู้เชี่ยวชาญ AI
    """)

    # ส่วนอัปโหลดไฟล์
    with gr.Row():
        file_input = gr.File(
            label="📁 อัปโหลด Security Logs", 
            file_types=['.json', '.csv', '.log', '.txt'],
            height=200,
            scale=2
        )
    
    # ปุ่มโหลด logs
    load_btn = gr.Button("📥 โหลด Logs", variant="primary", size="lg", scale=1)

    # ส่วนแสดงสถานะการโหลด
    with gr.Accordion("ℹ️ สถานะการโหลด Log", open=True):
        upload_status = gr.Textbox(label="📋 สถานะ", interactive=False)
        log_stats = gr.Textbox(label="📊 สถิติ Log", interactive=False)

    # ใช้ state เพื่อเก็บ logs ที่โหลดไว้
    log_state = gr.State([])

    gr.Markdown("---")

    # ส่วนแชทบอท
    chatbot = gr.Chatbot(label="🔬 แชทวิเคราะห์ SOC", height=1000)
    message_input = gr.Textbox(
        label="🧠 ถามเกี่ยวกับภัยคุกคามความปลอดภัย", 
        placeholder="เช่น มีการโจมตี brute-force หรือไม่?", 
        lines=1
    )
    send_btn = gr.Button("🔍 วิเคราะห์", variant="secondary")

    # การทำงานหลังจากกดปุ่มโหลด logs
    load_btn.click(
        fn=load_log_file,
        inputs=file_input,
        outputs=[upload_status, log_state, log_stats]
    )

    # ฟังก์ชันสำหรับการแชทและวิเคราะห์
    def chat_analyze(message, chat_history, logs):
        """
        ฟังก์ชันสำหรับจัดการการแชทและเรียกใช้การวิเคราะห์
        """
        reply = analyze_logs(message, chat_history, logs)
        chat_history.append((message, reply))
        return "", chat_history

    # การทำงานของปุ่มวิเคราะห์
    send_btn.click(
        fn=chat_analyze,
        inputs=[message_input, chatbot, log_state],
        outputs=[message_input, chatbot]
    )

    # การทำงานเมื่อกด Enter ในช่องข้อความ
    message_input.submit(
        fn=chat_analyze,
        inputs=[message_input, chatbot, log_state],
        outputs=[message_input, chatbot]
    )

# รันแอปพลิเคชัน
if __name__ == "__main__":
    demo.launch(
        share=False,           # ไม่แชร์ public link
        server_name="0.0.0.0", # รันบนทุก network interface
        server_port=7860,      # พอร์ตที่ใช้รัน
        show_error=True        # แสดงข้อผิดพลาดหากมี
    )
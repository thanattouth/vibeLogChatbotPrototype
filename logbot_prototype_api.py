# === โหลดไลบรารีที่จำเป็น ===
import os
import warnings

# แก้ไข HuggingFace tokenizers warning เพื่อป้องกันข้อความเตือนที่ไม่จำเป็น
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ปิด warnings ที่ไม่จำเป็นเพื่อให้ console สะอาด
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# นำเข้าไลบรารีสำหรับการสร้าง Web UI และการประมวลผลข้อมูล
import gradio as gr
import json
import csv
import re
from io import StringIO
from datetime import datetime

# นำเข้าไลบรารี LangChain สำหรับ RAG (Retrieval-Augmented Generation)
from langchain_community.vectorstores import FAISS  # Vector Database
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding Model
from langchain_groq import ChatGroq  # LLM จาก Groq
from langchain.chains import RetrievalQA  # Chain สำหรับ Q&A
from langchain.text_splitter import RecursiveCharacterTextSplitter  # แบ่งข้อความ
from langchain_core.documents import Document  # โครงสร้างเอกสาร
from langchain.prompts import PromptTemplate  # Template สำหรับ Prompt
from dotenv import load_dotenv # สำหรับ .env

# โหลดไฟล์ .env
load_dotenv()
api_key = os.getenv("API_KEY")

# === เริ่มต้น AI Models ===
# สร้าง Embedding Model สำหรับแปลงข้อความเป็น Vector
# all-MiniLM-L6-v2 เป็นโมเดลที่เล็ก รวดเร็ว และให้ผลลัพธ์ดี
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# สร้าง Large Language Model จาก Groq API
# llama3-70b-8192 เป็นโมเดลขนาดใหญ่ที่มีความสามารถสูง
llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=api_key,  # API Key
    temperature=0.1,  # ค่าต่ำ = คำตอบที่สม่ำเสมอและแม่นยำ
    max_tokens=2048   # จำกัดความยาวของคำตอบ
)

# === ฟังก์ชันจัดการไฟล์ ===
def parse_uploaded_file(file_obj):
    """
    ฟังก์ชันสำหรับอ่านและประมวลผลไฟล์ที่อัปโหลด
    รองรับไฟล์ .json, .csv, .log, .txt
    
    Args:
        file_obj: ไฟล์ที่อัปโหลดจาก Gradio
        
    Returns:
        tuple: (processed_logs, error_message)
    """
    try:
        # ตรวจสอบประเภทของ file object และอ่านเนื้อหา
        if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
            # กรณีไฟล์มีชื่อและเป็น string
            filename_lower = file_obj.name.lower()
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                content = f.read()
        elif hasattr(file_obj, 'read'):
            # กรณีไฟล์เป็น file-like object
            content = file_obj.read().decode("utf-8")
            filename_lower = file_obj.name.lower() if hasattr(file_obj, "name") else ""
        elif isinstance(file_obj, str):
            # กรณีไฟล์เป็น path string
            filename_lower = file_obj.lower()
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # กรณีอื่นๆ แปลงเป็น string path
            file_path = str(file_obj)
            filename_lower = file_path.lower()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        return [], f"🔴 ไม่สามารถอ่านไฟล์ได้: {str(e)}"

    log_lines = []

    # ประมวลผลตามประเภทไฟล์
    if filename_lower.endswith('.json'):
        """ประมวลผลไฟล์ JSON"""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # กรณี JSON เป็น array
                for item in data:
                    if isinstance(item, str):
                        log_lines.append(item.strip())
                    elif isinstance(item, dict):
                        # แปลง dict เป็นรูปแบบที่อ่านง่าย
                        formatted_line = format_json_log(item)
                        log_lines.append(formatted_line)
                    else:
                        log_lines.append(str(item).strip())
            elif isinstance(data, dict):
                # กรณี JSON เป็น object เดียว
                formatted_line = format_json_log(data)
                log_lines.append(formatted_line)
            else:
                # กรณีอื่นๆ แปลงเป็น string
                log_lines.append(str(data).strip())
        except Exception as e:
            return [], f"🔴 ไฟล์ JSON ไม่ถูกต้อง: {str(e)}"

    elif filename_lower.endswith('.csv'):
        """ประมวลผลไฟล์ CSV"""
        try:
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)
            for row in reader:
                # รวมคอลัมน์ทั้งหมดเป็นบรรทัดเดียว
                formatted_row = " | ".join([f"{k}:{v}" for k, v in row.items() if v])
                if formatted_row:
                    log_lines.append(formatted_row)
        except Exception as e:
            return [], f"🔴 ไฟล์ CSV ไม่ถูกต้อง: {str(e)}"

    elif filename_lower.endswith(('.log', '.txt')) or not filename_lower:
        """ประมวลผลไฟล์ Text/Log"""
        # แบ่งเป็นบรรทัดและกรองบรรทัดว่าง
        log_lines = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        return [], "⚠️ รูปแบบไฟล์ไม่รองรับ ใช้ .json, .csv, .log, หรือ .txt"

    # ตรวจสอบว่ามีข้อมูล log หรือไม่
    if not log_lines:
        return [], "⚠️ ไม่พบข้อมูล log ในไฟล์ที่อัปโหลด"

    # ประมวลผลและเสริมข้อมูล log
    processed_logs = preprocess_logs(log_lines)
    return processed_logs, None

def format_json_log(json_obj):
    """
    ฟังก์ชันจัดรูปแบบ JSON log ให้เป็น structured format
    
    Args:
        json_obj (dict): JSON object ที่ต้องการจัดรูปแบบ
        
    Returns:
        str: Log ที่จัดรูปแบบแล้ว
    """
    # กำหนดฟิลด์สำคัญที่ต้องการแสดงก่อน
    important_fields = ['timestamp', 'time', 'datetime', 'source_ip', 'src_ip', 'ip', 
                       'user', 'username', 'event', 'action', 'status', 'message', 'error']
    
    formatted_parts = []
    
    # เพิ่มฟิลด์สำคัญก่อน
    for field in important_fields:
        if field in json_obj:
            formatted_parts.append(f"{field}:{json_obj[field]}")
    
    # เพิ่มฟิลด์อื่นๆ ที่เหลือ
    for k, v in json_obj.items():
        if k not in important_fields:
            formatted_parts.append(f"{k}:{v}")
    
    return " | ".join(formatted_parts)

def preprocess_logs(log_lines):
    """
    ฟังก์ชันเสริมข้อมูล log เพื่อให้ AI วิเคราะห์ได้ดีขึ้น
    
    การเสริมข้อมูลที่ทำ:
    1. สกัด IP addresses
    2. ระบุ timestamps ในรูปแบบต่างๆ
    3. ตรวจหา security keywords
    
    Args:
        log_lines (list): รายการ log lines ดิบ
        
    Returns:
        list: Log lines ที่เสริมข้อมูลแล้ว
    """
    processed = []
    
    for line in log_lines:
        if not line.strip():
            continue
            
        enhanced_line = line
        
        # 1. สกัด IP addresses ด้วย Regular Expression
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # Pattern สำหรับ IPv4
        ips = re.findall(ip_pattern, line)
        if ips:
            enhanced_line += f" | EXTRACTED_IPS:{','.join(ips)}"
        
        # 2. ระบุ timestamps ในรูปแบบต่างๆ
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # ISO format: 2024-01-15T10:30:45
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',     # US format: 01/15/2024 10:30:45
            r'\w{3} \d{2} \d{2}:\d{2}:\d{2}'            # Syslog format: Jan 15 10:30:45
        ]
        
        for pattern in timestamp_patterns:
            timestamps = re.findall(pattern, line)
            if timestamps:
                enhanced_line += f" | TIMESTAMP:{timestamps[0]}"
                break  # หยุดเมื่อเจอ timestamp รูปแบบแรก
        
        # 3. ตรวจหา Security Keywords ที่บ่งบอกถึงภัยคุกคาม
        security_keywords = ['failed', 'error', 'denied', 'blocked', 'suspicious', 
                           'malware', 'virus', 'attack', 'intrusion', 'unauthorized',
                           'breach', 'violation', 'alert', 'warning']
        
        # หาคำสำคัญที่พบใน log line
        found_keywords = [kw for kw in security_keywords if kw.lower() in line.lower()]
        if found_keywords:
            enhanced_line += f" | SECURITY_INDICATORS:{','.join(found_keywords)}"
        
        processed.append(enhanced_line)
    
    return processed

# === ฟังก์ชันวิเคราะห์ด้วย AI ===
def analyze_logs(message, logs):
    """
    ฟังก์ชันหลักสำหรับวิเคราะห์ logs ด้วย AI
    ใช้เทคโนโลยี RAG (Retrieval-Augmented Generation)
    
    Args:
        message (str): คำถามจากผู้ใช้
        logs (list): รายการ log entries ที่ประมวลผลแล้ว
        
    Returns:
        str: ผลการวิเคราะห์จาก AI
    """
    # ตรวจสอบว่ามี logs หรือไม่
    if logs is None or not logs:
        return "📁 กรุณาอัปโหลดไฟล์ log ก่อนถามคำถาม"

    # ขั้นตอนที่ 1: แปลง log lines เป็น Document objects
    documents = []
    for i, log_line in enumerate(logs):
        if log_line.strip():
            documents.append(Document(
                page_content=log_line.strip(),  # เนื้อหาของ log
                metadata={"line_number": i+1, "source": "uploaded_logs"}  # ข้อมูลเพิ่มเติม
            ))
    
    # ตรวจสอบว่ามี documents หรือไม่
    if not documents:
        return "⚠️ ไม่พบข้อมูล log ที่สามารถใช้ได้ในไฟล์ที่อัปโหลด"

    # ขั้นตอนที่ 2: แบ่งข้อความเป็นชิ้นเล็กๆ สำหรับการประมวลผล
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,    # แต่ละชิ้นมีความยาวไม่เกิน 300 ตัวอักษร
        chunk_overlap=50,  # ซ้อนทับกัน 50 ตัวอักษรเพื่อไม่ให้ข้อมูลขาดหาย
        separators=["\n", "|", " ", ""]  # ตัวแบ่งตามลำดับความสำคัญ
    )
    split_docs = splitter.split_documents(documents)

    # หากแบ่งไม่ได้ ให้ใช้ documents เดิม
    if not split_docs:
        split_docs = documents

    # ขั้นตอนที่ 3: สร้าง Vector Database ด้วย FAISS
    # แปลง documents เป็น vectors แล้วเก็บใน database
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    # ขั้นตอนที่ 4: กำหนด System Prompt สำหรับ SOC Analyst
    system_prompt = """
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
    Respond in Thai language when the question is in Thai.
    """

    # ขั้นตอนที่ 5: สร้าง Prompt Template
    template = f"""
    {system_prompt}
    
    คุณกำลังวิเคราะห์ log ความปลอดภัยในฐานะนักวิเคราะห์ SOC โปรดให้การวิเคราะห์ที่ครอบคลุมจาก log entries ที่ค้นพบ
    
    ข้อมูลจาก logs:
    {{context}}
    
    คำถาม: {{question}}
    
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

    # สร้าง PromptTemplate object
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # ขั้นตอนที่ 6: สร้าง RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,  # Large Language Model
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # ค้นหา 5 รายการที่เกี่ยวข้องที่สุด
        chain_type="stuff",  # วิธีการรวม documents
        return_source_documents=True,  # ส่งคืน source documents ด้วย
        chain_type_kwargs={"prompt": prompt}  # ใช้ prompt ที่กำหนด
    )

    # ขั้นตอนที่ 7: เรียกใช้ RAG Chain เพื่อได้คำตอบ
    try:
        result = rag_chain.invoke({"query": message})
        return result.get("result", "⚠️ ไม่ได้รับผลการวิเคราะห์จาก AI")
    except Exception as e:
        return f"🔴 เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}\n\nกรุณาตรวจสอบ API Key หรือการเชื่อมต่อเครือข่าย"

# === ฟังก์ชันสำหรับการแชท ===
def process_message(message, history, logs):
    """
    ฟังก์ชันประมวลผลข้อความและสร้างการตอบกลับแบบ ChatGPT
    
    Args:
        message (str): ข้อความจากผู้ใช้
        history (list): ประวัติการสนทนา
        logs (list): ข้อมูล log ที่อัปโหลด
        
    Returns:
        tuple: (updated_history, empty_message_box)
    """
    # ตรวจสอบว่ามีข้อความหรือไม่
    if not message.strip():
        return history, ""
    
    # เพิ่มข้อความของผู้ใช้เข้าไปในประวัติ [user_message, ai_response]
    history.append([message, None])
    
    # วิเคราะห์และสร้างคำตอบด้วย AI
    response = analyze_logs(message, logs)
    
    # อัปเดตคำตอบของ AI ในประวัติ
    history[-1][1] = response
    
    # ส่งคืน history ที่อัปเดตแล้วและล้าง message box
    return history, ""

def upload_and_process_file(file):
    """
    ฟังก์ชันประมวลผลไฟล์ที่อัปโหลดและแสดงสถานะ
    
    Args:
        file: ไฟล์ที่อัปโหลดจาก Gradio
        
    Returns:
        tuple: (processed_logs, status_message)
    """
    # ตรวจสอบว่ามีไฟล์หรือไม่
    if file is None:
        return None, "❌ กรุณาเลือกไฟล์ที่ต้องการอัปโหลด"
    
    # ประมวลผลไฟล์
    logs, error = parse_uploaded_file(file)
    if error:
        return None, error
    
    # แสดงสถานะการอัปโหลดสำเร็จ
    log_count = len(logs)
    success_message = f"✅ อัปโหลดสำเร็จ! พบ {log_count:,} รายการ log พร้อมสำหรับการวิเคราะห์"
    
    return logs, success_message

# === สร้าง User Interface ด้วย Gradio ===
with gr.Blocks(
    title="🛡️ SOC Analysis AI (Powered by Groq Deepseek)", 
    theme=gr.themes.Soft(),  # ธีมที่นุ่มนวล
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
    
    # สร้าง State สำหรับเก็บข้อมูล logs
    logs_state = gr.State(None)
    
    # ส่วนหัวข้อ
    gr.Markdown("""
    # 🛡️ **SOC Analysis AI Assistant**
    ### 🚀 Powered by Groq Deepseek-Chat
    อัปโหลดไฟล์ log และถามคำถามเกี่ยวกับความปลอดภัย
    """)
    
    # ส่วนอัปโหลดไฟล์
    with gr.Row():
        with gr.Column(scale=3):
            file_upload = gr.File(
                label="📁 ลากไฟล์มาวางหรือคลิกเพื่อเลือก",
                file_types=['.json', '.csv', '.log', '.txt'],  # จำกัดประเภทไฟล์
                elem_classes="file-upload"
            )
        
    with gr.Column(scale=1):
        upload_btn = gr.Button("📤 อัปโหลด", variant="primary", size="lg")
    
    # แสดงสถานะการอัปโหลด
    upload_status = gr.Textbox(
        label="📊 สถานะ", 
        interactive=False,  # ไม่ให้ผู้ใช้แก้ไข
        value="🔄 รอการอัปโหลดไฟล์...",
        show_label=False
    )
    
    gr.Markdown("---")  # เส้นแบ่ง
    
    # ส่วนแชทแบบ ChatGPT
    with gr.Column():
        # พื้นที่แสดงการสนทนา
        chatbot = gr.Chatbot(
            label="💬 แชทกับ SOC Analyst AI",
            height=500,
            show_label=False,
            placeholder="การสนทนาจะปรากฏที่นี่...",
            elem_classes="chat-container"
        )
        
        # แถวสำหรับพิมพ์ข้อความ
        with gr.Row():
            message_box = gr.Textbox(
                placeholder="พิมพ์คำถามเกี่ยวกับ log ของคุณ... (เช่น 'มีการโจมตี brute force หรือไม่?')",
                container=False,
                scale=4,  # ขนาดสัดส่วน
                show_label=False
            )
            send_button = gr.Button("📨", variant="primary", scale=1, min_width=50)
    
    # ตัวอย่างคำถามสำหรับผู้ใช้
    with gr.Row():
        gr.Examples(
            examples=[
                "มีการโจมตี brute force หรือไม่?",
                "IP ไหนที่น่าสงสัยที่สุด?",
                "มีความผิดปกติในการเข้าสู่ระบบหรือไม่?",
                "วิเคราะห์ภัยคุกคามโดยรวม",
                "แสดงสถิติการโจมตีตามเวลา"
            ],
            inputs=message_box,  # เมื่อคลิกจะใส่ใน message_box
            label="💡 ตัวอย่างคำถาม"
        )
    
    # === เชื่อมต่อ Event Handlers ===
    
    # เมื่อกดปุ่มอัปโหลด
    upload_btn.click(
        fn=upload_and_process_file,  # ฟังก์ชันที่จะเรียก
        inputs=file_upload,          # Input component
        outputs=[logs_state, upload_status]  # Output components
    )
    
    # ฟังก์ชันสำหรับการส่งข้อความ
    def submit_message(message, history, logs):
        """Wrapper function สำหรับการส่งข้อความ"""
        return process_message(message, history, logs)
    
    # เมื่อกดปุ่มส่ง
    send_button.click(
        fn=submit_message,
        inputs=[message_box, chatbot, logs_state],
        outputs=[chatbot, message_box]  # อัปเดต chatbot และล้าง message_box
    )
    
    # เมื่อกด Enter ใน message box
    message_box.submit(
        fn=submit_message,
        inputs=[message_box, chatbot, logs_state],
        outputs=[chatbot, message_box]
    )

# === เรียกใช้งานแอพพลิเคชัน ===
if __name__ == "__main__":
    demo.launch(
        share=False,           # ไม่แชร์ลิงก์สาธารณะ
        server_name="0.0.0.0", # รับ connection จากทุก IP
        server_port=7860,      # พอร์ตที่ใช้
        show_error=True        # แสดง error ใน UI
    )
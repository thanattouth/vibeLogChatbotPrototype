# === โหลดไลบรารีที่จำเป็น ===
import gradio as gr
import json
import csv
import re
from io import StringIO
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# === เริ่มต้น Models ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = Ollama(
    model="phi3:mini",
    temperature=0.1,
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

# === ฟังก์ชันจัดการไฟล์ (เหมือนเดิม) ===
def parse_uploaded_file(file_obj):
    try:
        if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
            filename_lower = file_obj.name.lower()
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                content = f.read()
        elif hasattr(file_obj, 'read'):
            content = file_obj.read().decode("utf-8")
            filename_lower = file_obj.name.lower() if hasattr(file_obj, "name") else ""
        elif isinstance(file_obj, str):
            filename_lower = file_obj.lower()
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            file_path = str(file_obj)
            filename_lower = file_path.lower()
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        return [], f"🔴 ไม่สามารถอ่านไฟล์ได้: {str(e)}"

    log_lines = []

    if filename_lower.endswith('.json'):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        log_lines.append(item.strip())
                    elif isinstance(item, dict):
                        formatted_line = format_json_log(item)
                        log_lines.append(formatted_line)
                    else:
                        log_lines.append(str(item).strip())
            elif isinstance(data, dict):
                formatted_line = format_json_log(data)
                log_lines.append(formatted_line)
            else:
                log_lines.append(str(data).strip())
        except Exception as e:
            return [], f"🔴 ไฟล์ JSON ไม่ถูกต้อง: {str(e)}"

    elif filename_lower.endswith('.csv'):
        try:
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)
            for row in reader:
                formatted_row = " | ".join([f"{k}:{v}" for k, v in row.items() if v])
                if formatted_row:
                    log_lines.append(formatted_row)
        except Exception as e:
            return [], f"🔴 ไฟล์ CSV ไม่ถูกต้อง: {str(e)}"

    elif filename_lower.endswith(('.log', '.txt')) or not filename_lower:
        log_lines = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        return [], "⚠️ รูปแบบไฟล์ไม่รองรับ ใช้ .json, .csv, .log, หรือ .txt"

    if not log_lines:
        return [], "⚠️ ไม่พบข้อมูล log ในไฟล์ที่อัปโหลด"

    processed_logs = preprocess_logs(log_lines)
    return processed_logs, None

def format_json_log(json_obj):
    important_fields = ['timestamp', 'time', 'datetime', 'source_ip', 'src_ip', 'ip', 
                       'user', 'username', 'event', 'action', 'status', 'message', 'error']
    
    formatted_parts = []
    for field in important_fields:
        if field in json_obj:
            formatted_parts.append(f"{field}:{json_obj[field]}")
    
    for k, v in json_obj.items():
        if k not in important_fields:
            formatted_parts.append(f"{k}:{v}")
    
    return " | ".join(formatted_parts)

def preprocess_logs(log_lines):
    processed = []
    
    for line in log_lines:
        if not line.strip():
            continue
            
        enhanced_line = line
        
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = re.findall(ip_pattern, line)
        if ips:
            enhanced_line += f" | EXTRACTED_IPS:{','.join(ips)}"
        
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\w{3} \d{2} \d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in timestamp_patterns:
            timestamps = re.findall(pattern, line)
            if timestamps:
                enhanced_line += f" | TIMESTAMP:{timestamps[0]}"
                break
        
        security_keywords = ['failed', 'error', 'denied', 'blocked', 'suspicious', 
                           'malware', 'virus', 'attack', 'intrusion', 'unauthorized',
                           'breach', 'violation', 'alert', 'warning']
        
        found_keywords = [kw for kw in security_keywords if kw.lower() in line.lower()]
        if found_keywords:
            enhanced_line += f" | SECURITY_INDICATORS:{','.join(found_keywords)}"
        
        processed.append(enhanced_line)
    
    return processed

# === ฟังก์ชันวิเคราะห์ (เหมือนเดิม) ===
def analyze_logs(message, logs):
    if logs is None or not logs:
        return "📁 กรุณาอัปโหลดไฟล์ log ก่อนถามคำถาม"

    documents = []
    for i, log_line in enumerate(logs):
        if log_line.strip():
            documents.append(Document(
                page_content=log_line.strip(),
                metadata={"line_number": i+1, "source": "uploaded_logs"}
            ))
    
    if not documents:
        return "⚠️ ไม่พบข้อมูล log ที่สามารถใช้ได้ในไฟล์ที่อัปโหลด"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", "|", " ", ""]
    )
    split_docs = splitter.split_documents(documents)

    if not split_docs:
        split_docs = documents

    vectorstore = FAISS.from_documents(split_docs, embedding_model)

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

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    try:
        result = rag_chain.invoke({"query": message})
        return result.get("result", "⚠️ ไม่ได้รับผลการวิเคราะห์จาก AI")
    except Exception as e:
        return f"🔴 เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

# === ฟังก์ชันสำหรับการแชท (แบบ ChatGPT) ===
def process_message(message, history, logs):
    """ประมวลผลข้อความและสร้างการตอบกลับ"""
    if not message.strip():
        return history, ""
    
    # เพิ่มข้อความของผู้ใช้เข้าไปในประวัติ
    history.append([message, None])
    
    # วิเคราะห์และสร้างคำตอบ
    response = analyze_logs(message, logs)
    
    # อัปเดตคำตอบในประวัติ
    history[-1][1] = response
    
    return history, ""

def upload_and_process_file(file):
    """ประมวลผลไฟล์ที่อัปโหลดและแสดงสถานะ"""
    if file is None:
        return None, "❌ กรุณาเลือกไฟล์ที่ต้องการอัปโหลด"
    
    logs, error = parse_uploaded_file(file)
    if error:
        return None, error
    
    log_count = len(logs)
    success_message = f"✅ อัปโหลดสำเร็จ! พบ {log_count:,} รายการ log พร้อมสำหรับการวิเคราะห์"
    
    return logs, success_message

# === สร้าง UI แบบ ChatGPT ===
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
    
    # เก็บสถานะ
    logs_state = gr.State(None)
    
    # หัวข้อ
    gr.Markdown("""
    # 🛡️ **SOC Analysis AI Assistant**
    อัปโหลดไฟล์ log และถามคำถามเกี่ยวกับความปลอดภัย
    """)
    
    # ส่วนอัปโหลดไฟล์แบบกะทัดรัด
    with gr.Row():
        with gr.Column(scale=3):
            file_upload = gr.File(
                label="📁 ลากไฟล์มาวางหรือคลิกเพื่อเลือก",
                file_types=['.json', '.csv', '.log', '.txt'],
                elem_classes="file-upload"
            )
    with gr.Column(scale=1):
        upload_btn = gr.Button("📤 อัปโหลด", variant="primary", size="lg")
    
    # แสดงสถานะการอัปโหลด
    upload_status = gr.Textbox(
        label="📊 สถานะ", 
        interactive=False,
        value="🔄 รอการอัปโหลดไฟล์...",
        show_label=False
    )
    
    gr.Markdown("---")
    
    # ส่วนแชทแบบ ChatGPT
    with gr.Column():
        chatbot = gr.Chatbot(
            label="💬 แชทกับ SOC Analyst AI",
            height=500,
            show_label=False,
            placeholder="การสนทนาจะปรากฏที่นี่...",
            elem_classes="chat-container"
        )
        
        with gr.Row():
            message_box = gr.Textbox(
                placeholder="พิมพ์คำถามเกี่ยวกับ log ของคุณ... (เช่น 'มีการโจมตี brute force หรือไม่?')",
                container=False,
                scale=4,
                show_label=False
            )
            send_button = gr.Button("📨", variant="primary", scale=1, min_width=50)
    
    # ตัวอย่างคำถาม
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
    
    # เชื่อมต่อฟังก์ชัน
    upload_btn.click(
        fn=upload_and_process_file,
        inputs=file_upload,
        outputs=[logs_state, upload_status]
    )
    
    # การส่งข้อความ
    def submit_message(message, history, logs):
        return process_message(message, history, logs)
    
    send_button.click(
        fn=submit_message,
        inputs=[message_box, chatbot, logs_state],
        outputs=[chatbot, message_box]
    )
    
    message_box.submit(
        fn=submit_message,
        inputs=[message_box, chatbot, logs_state],
        outputs=[chatbot, message_box]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
# 🛡️ Enhanced SOC Analysis AI Assistant

ระบบ AI สำหรับวิเคราะห์ Log ด้านความมั่นคงปลอดภัยทางไซเบอร์ขั้นสูง โดยใช้ Ollama, LangChain, FAISS และ Streamlit พร้อมระบบ Cache และ Distributed Processing เหมาะสำหรับใช้งานใน SOC (Security Operations Center)

---

## ✨ คุณสมบัติหลัก

### 🤖 **AI-Powered Analysis**
- วิเคราะห์ log ด้วย Ollama (deepseek-coder) แบบ offline
- ระบบ RAG (Retrieval-Augmented Generation) พร้อม Vector Search
- รองรับการวิเคราะห์แบบ Real-time และ Batch processing

### 🔍 **Advanced Log Processing**
- รองรับไฟล์หลายรูปแบบ: `.log`, `.txt`, `.json`, `.csv`
- ระบบ preprocessing ที่ชาญฉลาดสำหรับ security logs
- การแยกข้อมูล IP addresses, timestamps, และ security indicators

### 💬 **Interactive Chat Interface**
- UI แบบ Chat พร้อม streaming responses
- ตัวอย่างคำถามที่พร้อมใช้งาน
- ระบบ export chat history เป็น markdown

### 📊 **Data Visualization**
- Timeline visualization ของ log events
- Threat distribution charts
- Real-time statistics และ metrics

### ⚡ **Performance Features**
- **Redis Caching**: เก็บผลการวิเคราะห์เพื่อเร่งความเร็ว
- **Distributed Processing**: ใช้ Celery สำหรับการประมวลผลแบบกระจาย
- **Streaming Responses**: แสดงผลแบบ real-time

### 🛡️ **Security-Focused**
- ระบบวิเคราะห์ภัยคุกคามตาม MITRE ATT&CK framework
- การจัดประเภทภัยคุกคาม: Critical/High/Medium/Low
- คำแนะนำการแก้ไขปัญหาแบบ actionable

---

## 🧠 ตัวอย่างการใช้งาน

### คำถามที่ระบบสามารถตอบได้:
- _"Identify potential security threats"_ - ระบุภัยคุกคามที่เป็นไปได้
- _"Show suspicious IPs"_ - แสดง IP addresses ที่น่าสงสัย
- _"Analyze failed logins"_ - วิเคราะห์การ login ที่ล้มเหลว
- _"Timeline of events"_ - สร้าง timeline ของเหตุการณ์
- _"Most critical threat?"_ - ระบุภัยคุกคามที่สำคัญที่สุด

### การทำงานของระบบ:
1. **อัปโหลดไฟล์ log** - รองรับไฟล์หลายรูปแบบ
2. **ระบบประมวลผล** - แยกข้อมูลและสร้าง vector embeddings
3. **ถามคำถาม** - ผ่าน Chat interface
4. **รับคำตอบ** - พร้อมการอ้างอิงข้อมูลและคำแนะนำ

---

## 🔧 การติดตั้ง

### 1. Clone โปรเจกต์

```bash
git clone https://github.com/thanattouth/vibe_log_chatbot_prototype.git
cd vibe_log_chatbot_prototype
```

### 2. สร้าง Virtual Environment และติดตั้ง Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. ติดตั้งและตั้งค่า Ollama

#### ติดตั้ง Ollama:
- **macOS:**
  ```bash
  brew install ollama
  ```
  
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- **Windows:** ดาวน์โหลดจาก [https://ollama.com](https://ollama.com)

#### เริ่ม Ollama server:
```bash
ollama serve
```

#### โหลดโมเดล deepseek-coder:
```bash
ollama pull deepseek-coder
```

### 4. ติดตั้ง Redis (สำหรับ Caching - ไม่บังคับ)

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
```

#### macOS:
```bash
brew install redis
brew services start redis
```

#### Windows:
- ดาวน์โหลดจาก [Redis for Windows](https://github.com/microsoftarchive/redis/releases)

---

## 🚀 การใช้งาน

### รันระบบ:
```bash
streamlit run logbot_prototype.py
```

### เข้าใช้งาน:
- **Local:** `http://localhost:8501`
- **Network:** `http://[YOUR_IP]:8501`

---

## ⚙️ การตั้งค่าขั้นสูง

### การเปิดใช้งาน Distributed Processing:

1. **เริ่ม Celery Worker:**
   ```bash
   celery -A logbot_prototype.celery_app worker --loglevel=info
   ```

2. **เปิดใช้งานใน UI:** เลือก "Enable distributed processing (Celery)" ในหน้า upload

### การตั้งค่า Redis:
- **Host:** localhost (default)
- **Port:** 6379 (default)
- **Database:** 0 (caching), 1 (broker), 2 (backend)

---

## 📊 คุณสมบัติของระบบ

### Analysis Modes:
- **Basic**: การวิเคราะห์พื้นฐาน
- **Advanced**: การวิเคราะห์ขั้นสูงพร้อม correlation
- **Forensic**: การวิเคราะห์เชิงนิติวิทยาศาสตร์

### Threat Detection:
- **Critical**: ต้องการการดำเนินการทันที
- **High**: ต้องการการสืบสวนลำดับความสำคัญสูง
- **Medium**: ควรได้รับการสืบสวน
- **Low**: ต้องการการติดตาม

### Supported File Types:
- **JSON**: รองรับ structured logs
- **CSV**: รองรับ tabular data
- **LOG/TXT**: รองรับ plain text logs

---

## 🔐 ความปลอดภัย

### ✅ **Air-Gapped Deployment:**
- ข้อมูลไม่ออกจากระบบ
- ใช้ Ollama local model
- ปลอดภัยสำหรับข้อมูลลับ
- เหมาะสำหรับสภาพแวดล้อม military/government

### 🛡️ **Security Features:**
- ไม่มีการส่งข้อมูลไปยัง API ภายนอก
- การเก็บข้อมูลใน memory เท่านั้น
- ระบบ caching แบบ local

---

## 📋 ข้อกำหนดระบบ

### Minimum Requirements:
- **Python:** 3.8+
- **RAM:** 8GB+ (แนะนำ 16GB)
- **Storage:** 10GB+ (สำหรับโมเดล Ollama)
- **CPU:** 4 cores+ (แนะนำ 8 cores)

### Recommended Setup:
- **RAM:** 32GB
- **CPU:** 16 cores
- **GPU:** NVIDIA GPU (สำหรับการใช้งาน GPU acceleration)
- **SSD:** สำหรับการเข้าถึงข้อมูลที่เร็ว

---

## 🐛 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย:

#### 1. Ollama Connection Failed:
```bash
# ตรวจสอบว่า Ollama server ทำงาน
ollama list

# เริ่ม Ollama server
ollama serve
```

#### 2. Redis Connection Failed:
```bash
# ตรวจสอบสถานะ Redis
redis-cli ping

# เริ่ม Redis service
sudo systemctl start redis-server
```

#### 3. Memory Issues:
- ลดขนาดไฟล์ log ที่อัปโหลด
- เพิ่ม RAM หรือใช้ swap file
- ปิดการใช้งาน distributed processing

#### 4. Model Loading Issues:
```bash
# โหลดโมเดลใหม่
ollama pull deepseek-coder

# ตรวจสอบโมเดลที่มี
ollama list
```

---

## 📄 License

MIT License © 2025 Thanattouth

---
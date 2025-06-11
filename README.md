# 🛡 MilitarySOCAnalysisChatbot_Prototype

ระบบต้นแบบ (Prototype) สำหรับวิเคราะห์ Log ด้านความมั่นคงปลอดภัยทางไซเบอร์ โดยใช้ AI/LLM, LangChain, FAISS และ Gradio UI เหมาะสำหรับใช้งานในระบบ Military SOC (Security Operations Center)

---

## 📦 เวอร์ชันที่มีให้

### 🔒 **Air-Gapped Version (Offline)**
- **ไฟล์:** `logbot_prototype.py`
- **LLM:** Ollama (phi3:mini) - ทำงานแบบ offline
- **เหมาะสำหรับ:** สภาพแวดล้อมที่ไม่มีการเชื่อมต่ออินเทอร์เน็ต
- **ความปลอดภัย:** สูงสุด - ข้อมูลไม่ออกจากระบบ

### 🌐 **API Version (Online)**
- **ไฟล์:** `logbot_prototype_api.py`
- **LLM:** Groq API (llama3-70b-8192) - ต้องการการเชื่อมต่ออินเทอร์เน็ต
- **เหมาะสำหรับ:** การใช้งานที่ต้องการประสิทธิภาพสูงและมีการเชื่อมต่ออินเทอร์เน็ต
- **ข้อควรระวัง:** ⚠️ **ไม่ใช่ air-gapped** - ข้อมูลจะถูกส่งไปยัง API ภายนอก

---

## ✨ คุณสมบัติ

- 🤖 วิเคราะห์ log ด้วย AI/LLM อัจฉริยะ
- 🔍 ระบบค้นหาแบบ Vector Search ด้วย FAISS 
- 📊 รองรับไฟล์หลายรูปแบบ (`.log`, `.txt`, `.json`, `.csv`)
- 💬 UI แบบ Chat Interface ใช้งานง่าย
- 🛡️ วิเคราะห์ภัยคุกคามแบบ SOC Professional
- 🔧 Modular code พร้อมต่อยอดเป็นระบบ production

---

## 🧠 ตัวอย่างการใช้งาน

1. **อัปโหลดไฟล์ log** - รองรับ `.log`, `.txt`, `.json`, `.csv`
2. **ถามคำถาม** เช่น:
   - _"มีการโจมตี brute force หรือไม่?"_
   - _"IP ไหนที่น่าสงสัยที่สุด?"_
   - _"วิเคราะห์ภัยคุกคามโดยรวม"_
3. **รับคำตอบ** จาก SOC Analyst AI พร้อมคำแนะนำการแก้ไข

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

---

## 🚀 การใช้งาน

### 🔒 **Air-Gapped Version (แนะนำสำหรับข้อมูลสำคัญ)**

#### 1. ติดตั้งและโหลดโมเดลผ่าน Ollama

- **ติดตั้ง Ollama:** [https://ollama.com](https://ollama.com)
  
  macOS:
  ```bash
  brew install ollama
  ```
  
  Linux:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- **เริ่ม Ollama server:**
  ```bash
  ollama serve
  ```

- **โหลดโมเดล phi3:mini:**
  ```bash
  ollama pull phi3:mini
  ```

#### 2. รันระบบ Air-Gapped

```bash
python logbot_prototype.py
```

### 🌐 **API Version (สำหรับประสิทธิภาพสูง)**

#### 1. ตั้งค่า Groq API Key

- สมัครสมาชิก: [https://console.groq.com](https://console.groq.com)
- สร้าง API Key และแก้ไขในไฟล์ `logbot_prototype_api.py`:

```python
groq_api_key="your_groq_api_key_here"
```

#### 2. รันระบบ API Version

```bash
python logbot_prototype_api.py
```

---

## 📊 การเข้าใช้งาน

หลังจากรันระบบแล้ว เปิดเบราว์เซอร์ไปที่:
- **Local:** `http://0.0.0.0:7860`
- **Network:** `http://0.0.0.0:7860`

---

### 🔒 **เลือก Air-Gapped เมื่อ:**
- ทำงานกับข้อมูลลับหรือสำคัญ
- อยู่ในสภาพแวดล้อม air-gapped
- ต้องการความปลอดภัยสูงสุด
- ไม่มีการเชื่อมต่ออินเทอร์เน็ต

### 🌐 **เลือก API Version เมื่อ:**
- ต้องการประสิทธิภาพการวิเคราะห์สูงสุด
- มีการเชื่อมต่ออินเทอร์เน็ตที่เสถียร
- ทำงานกับข้อมูลที่ไม่ละเอียดอ่อน
- ต้องการความเร็วในการตอบสนอง

---

## 📋 ข้อกำหนดระบบ

- **Python:** 3.8+
- **RAM:** 4GB+ (Air-Gapped), 2GB+ (API)
- **Storage:** 5GB+ (สำหรับโมเดล Ollama)
- **Network:** ไม่จำเป็น (Air-Gapped), จำเป็น (API)

---

## 🔐 ข้อควรระวังด้านความปลอดภัย

### ⚠️ **สำหรับ API Version:**
- ข้อมูล log จะถูกส่งไปยัง Groq API (บริการภายนอก)
- อ่านและทำความเข้าใจ Privacy Policy ของ Groq
- **ไม่แนะนำ** สำหรับข้อมูลลับหรือสำคัญ
- ใช้เฉพาะกับข้อมูลที่ไม่ละเอียดอ่อน

### ✅ **สำหรับ Air-Gapped Version:**
- ข้อมูลไม่ออกจากระบบ
- ปลอดภัยสำหรับข้อมูลลับ
- เหมาะสำหรับสภาพแวดล้อม military/government

---

## 📄 License

MIT License © 2025 Thanattouth

---
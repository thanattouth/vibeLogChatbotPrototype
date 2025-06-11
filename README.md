
# 🛡 MilitarySOCAnalysisChatbot_Prototype

ระบบต้นแบบ (Prototype) สำหรับวิเคราะห์ Log ด้านความมั่นคงปลอดภัยทางไซเบอร์ โดยใช้ AI/LLM, LangChain, FAISS และ Gradio UI เหมาะสำหรับใช้งานในระบบ Military SOC (Security Operations Center) ที่ต้องการวิเคราะห์ log แบบ offline หรือ air-gapped

---

## ✨ คุณสมบัติ

- วิเคราะห์ log ด้วย LLM แบบ offline ผ่าน [Ollama](https://ollama.com)
- สร้างเวกเตอร์จาก log และค้นหาคล้ายกันด้วย FAISS
- ใช้ HuggingFace Embedding + LangChain RetrievalQA
- รองรับ UI ผ่าน Gradio ใช้งานง่าย
- Modular code พร้อมต่อยอดเป็นระบบ production

---

## 🧠 ตัวอย่างการใช้งาน

1. ผู้ใช้สามารถอัปโหลดไฟล์ log (รองรับ `.log`, `.txt`, `.json`, `.csv`)
2. พิมพ์คำถาม เช่น:  
   _"มีเหตุการณ์ผิดปกติอะไรเกิดขึ้นบ้าง?"_
3. ระบบจะวิเคราะห์ log และให้คำตอบจาก LLM

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

### 1. ติดตั้งและโหลดโมเดลผ่าน Ollama

- ติดตั้ง Ollama: [https://ollama.com](https://ollama.com)  
  หรือผ่าน macOS Terminal:

  ```bash
  brew install ollama
  ```

- เริ่ม Ollama server:

  ```bash
  ollama serve
  ```

- โหลดโมเดล `phi3:mini`:

  ```bash
  ollama pull phi3:mini
  ```

### 2. รันระบบ

```bash
python logbot_prototype.py
```

ระบบจะเปิด Gradio UI ที่ `http://0.0.0.0:7860` โดยอัตโนมัติ

---

## 📝 License

MIT License © 2025 Thanattouth

# AI CryptoDetect

AI CryptoDetect is a modern, privacy-focused web platform designed for file encryption, neural cryptographic algorithm detection, invisible steganography, and document format conversion. The platform is designed to run locally or deployed as a lightweight serverless application (e.g., on Vercel).

## 🚀 Key Features

### 1. Neural Cryptographic Analyzer
- Analyzes arbitrary files or pasted text using Shannon entropy and byte-frequency distribution.
- Classifies and predicts whether data is encrypted or plain, with confidence metrics, block modes (e.g., CBC, GCM), key lengths, and entropy values.

### 2. Secure Local File Encryption
- Encrypt files client-side / locally using industry-standard symmetric and asymmetric algorithms:
  - **AES-256-CBC** (Advanced Encryption Standard)
  - **TripleDES-CBC** (Legacy standard)
  - **RC4** (Stream cipher)
  - **ChaCha20** (Modern high-performance stream cipher)
  - **RSA-2048** (Asymmetric key-pair encryption with hybrid AES wrapper)
- Allows key derivation with custom passphrases.

### 3. Expirable Link Sharing
- Upload encrypted files and generate unique, secure sharing links.
- Custom expiration timers and max-access limits (link self-destructs after exceeding limits or duration).
- Secure download endpoint automatically decrypts/provides files for authorized accesses.

### 4. Invisible Steganography (Image & Audio)
- **Image Steganography**: Hide secret text messages inside PNG/JPG images using **LSB (Least Significant Bit)** encoding, preserving visual quality. Extract messages easily from stego images.
- **Audio Steganography**: Hide secret text inside WAV/MP3 files using **EOF Tagging**. Extract secret text without impacting audio playability.

### 5. Simple Local File Converter
- Convert files securely with zero tracking or external API calls:
  - **Images**: PNG, JPG/JPEG, WEBP, BMP, GIF.
  - **Documents**: TXT to PDF using dynamic PDF page generation.
  - **Mock Conversions**: Fallback handlers for docx, doc, audio, and video formats.

---

## 🛠️ Technology Stack

- **Backend**: Python (Flask)
- **Frontend**: HTML5, CSS3 (Vanilla Glassmorphism Theme), Vanilla JavaScript
- **Cryptographic Library**: PyCa/Cryptography
- **Image Processing**: Pillow (PIL)
- **PDF Generation**: ReportLab

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vishal-01-kumar/Ai-Cryptodetect.git
   cd Ai-Cryptodetect
   ```

2. **Create and activate a virtual environment**:
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

---

## 🔒 Privacy & Security

This tool is designed with a **privacy-first architecture**:
- No external APIs or trackers are loaded.
- File encryption, conversion, and steganography are processed on your local server.
- Shared files are stored temporarily in-memory and are not persisted to a database, ensuring complete data security.

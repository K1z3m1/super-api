import gradio as gr
import requests
import json
import io
import os
import base64
import re
from PIL import Image, ImageEnhance, ImageFilter
import time
from flask import Flask, request, jsonify
import threading
import cv2
import numpy as np

# ตั้งค่า port
PORT = int(os.environ.get("PORT", 7860))

class ProfessionalTranslationApp:
    def __init__(self):
        # หลาย API keys สำหรับ fallback
        self.ocr_api_keys = [
            "K89947895888957",  # OCR.space
            "helloworld",       # ฟรี
        ]
        self.current_ocr_key_index = 0
        
        # รองรับหลายภาษาแบบละเอียด
        self.supported_languages = {
            'th': {'name': 'Thai', 'emoji': '🇹🇭'},
            'en': {'name': 'English', 'emoji': '🇺🇸'},
            'ja': {'name': 'Japanese', 'emoji': '🇯🇵'},
            'ko': {'name': 'Korean', 'emoji': '🇰🇷'},
            'zh': {'name': 'Chinese', 'emoji': '🇨🇳'},
            'fr': {'name': 'French', 'emoji': '🇫🇷'},
            'es': {'name': 'Spanish', 'emoji': '🇪🇸'},
            'de': {'name': 'German', 'emoji': '🇩🇪'},
        }
        
        # Context dictionary สำหรับการแปลที่เข้าใจบริบท
        self.context_phrases = {
            'ja': {
                'manga': {
                    'お前': 'you (informal/male)',
                    '俺': 'I (male)',
                    '私': 'I (female/formal)',
                    '君': 'you (friendly)',
                    'ありがとう': 'thank you',
                    'すみません': 'excuse me/sorry',
                }
            },
            'ko': {
                'manga': {
                    '나는': 'I',
                    '너': 'you',
                    '감사합니다': 'thank you',
                }
            }
        }
    
    def get_ocr_key(self):
        """สลับใช้ OCR API keys"""
        key = self.ocr_api_keys[self.current_ocr_key_index]
        self.current_ocr_key_index = (self.current_ocr_key_index + 1) % len(self.ocr_api_keys)
        return key
    
    def enhance_manga_image(self, image):
        """ปรับปรุงภาพมังงะให้ OCR ทำงานได้ดีขึ้น"""
        try:
            # แปลงเป็น numpy array สำหรับ OpenCV
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # แปลงเป็น grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # เพิ่ม contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            
            # ลบ noise
            gray = cv2.medianBlur(gray, 3)
            
            # Threshold เพื่อให้ข้อความชัดเจน
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # ขยายข้อความให้ชัดเจน
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return Image.fromarray(binary)
            
        except Exception as e:
            print(f"Image enhancement error: {e}")
            return image
    
    def detect_language_advanced(self, text):
        """ตรวจจับภาษาอย่างแม่นยำ"""
        # นับตัวอักษรแต่ละภาษา
        thai_count = len([c for c in text if '\u0e00' <= c <= '\u0e7f'])
        japanese_count = len([c for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'])
        korean_count = len([c for c in text if '\uac00' <= c <= '\ud7a3'])
        chinese_count = len([c for c in text if '\u4e00' <= c <= '\u9fff' and not ('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')])
        
        # ตรวจสอบภาษาที่มีมากที่สุด
        counts = {
            'th': thai_count,
            'ja': japanese_count,
            'ko': korean_count,
            'zh': chinese_count
        }
        
        detected_lang = max(counts, key=counts.get)
        
        # ถ้าไม่พบตัวอักษรเอเชีย ให้ถือว่าเป็นอังกฤษ
        if counts[detected_lang] == 0:
            return 'en'
        
        return detected_lang
    
    def improve_ocr_accuracy(self, image_input, is_manga=False, language='eng+tha+jpn+kor'):
        """OCR ที่แม่นยำขึ้น"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # โหลดภาพ
                if isinstance(image_input, str):
                    if image_input.startswith('http'):
                        response = requests.get(image_input, timeout=15)
                        image = Image.open(io.BytesIO(response.content))
                    else:
                        image_data = base64.b64decode(image_input.split(',')[1])
                        image = Image.open(io.BytesIO(image_data))
                else:
                    image = image_input
                
                # ปรับปรุงภาพสำหรับมังงะ
                if is_manga:
                    image = self.enhance_manga_image(image)
                
                # ปรับขนาดภาพ
                if max(image.size) > 1200:
                    image.thumbnail((1200, 1200))
                
                # แปลงภาพเป็น bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG', optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                
                # OCR.space API
                data = {
                    "apikey": self.get_ocr_key(),
                    "language": language,
                    "isOverlayRequired": True,  # ขอตำแหน่งข้อความสำหรับ Chrome Extension
                    "OCREngine": 2,
                    "scale": True,
                    "isTable": is_manga,  # สำหรับมังงะที่มีการจัดข้อความ
                    "detectOrientation": True,
                }
                
                response = requests.post(
                    "https://api.ocr.space/parse/image",
                    files={"image": img_byte_arr},
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("IsErroredOnProcessing"):
                        if attempt < max_retries - 1:
                            continue
                        return {"error": f"OCR Error: {result.get('ErrorMessage', 'Unknown error')}"}
                    
                    parsed_results = result.get("ParsedResults", [])
                    if parsed_results:
                        text_data = parsed_results[0]
                        text = text_data.get("ParsedText", "").strip()
                        
                        if text:
                            # ตรวจจับภาษา
                            detected_lang = self.detect_language_advanced(text)
                            
                            return {
                                "success": True,
                                "text": text,
                                "word_count": len(text.split()),
                                "detected_language": detected_lang,
                                "text_overlay": text_data.get("TextOverlay", {}),  # ตำแหน่งข้อความสำหรับซ้อนภาพ
                                "raw_result": text_data  # ข้อมูลดิบสำหรับการประมวลผลเพิ่มเติม
                            }
                    
                    if attempt < max_retries - 1:
                        continue
                    return {"error": "ไม่พบข้อความในภาพ"}
                else:
                    if attempt < max_retries - 1:
                        continue
                    return {"error": f"API Error: {response.status_code}"}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                return {"error": f"เกิดข้อผิดพลาด: {str(e)}"}
    
    def context_aware_translate(self, text, target_lang='th', source_lang='auto', context_type='general'):
        """การแปลที่เข้าใจบริบท"""
        if not text or not text.strip():
            return {"error": "กรุณาป้อนข้อความ"}
        
        try:
            # ตรวจจับภาษาต้นทาง
            if source_lang == 'auto':
                source_lang = self.detect_language_advanced(text)
            
            if source_lang == target_lang:
                return {
                    "success": True,
                    "translated_text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "context_used": "same_language"
                }
            
            # ประมวลผลข้อความก่อนแปล (สำหรับมังงะ)
            processed_text = text
            if context_type == 'manga' and source_lang in self.context_phrases:
                # แทนที่คำศัพท์เฉพาะก่อนแปล
                for phrase, meaning in self.context_phrases[source_lang]['manga'].items():
                    if phrase in text:
                        processed_text = processed_text.replace(phrase, f"{phrase} ({meaning})")
            
            # ลองใช้หลาย API
            translation_attempts = []
            
            # Attempt 1: MyMemory API
            try:
                params = {
                    "q": processed_text[:1000],
                    "langpair": f"{source_lang}|{target_lang}",
                    "de": "manga_translator@example.com",
                    "mt": "1"  # Machine translation
                }
                response = requests.get(
                    "https://api.mymemory.translated.net/get",
                    params=params,
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    translated = result["responseData"]["translatedText"]
                    if translated and translated.strip() and translated != processed_text:
                        translation_attempts.append(("MyMemory", translated))
            except:
                pass
            
            # Attempt 2: LibreTranslate
            try:
                payload = {
                    "q": processed_text[:1000],
                    "source": source_lang,
                    "target": target_lang,
                    "format": "text"
                }
                response = requests.post(
                    "https://libretranslate.de/translate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    translated = result.get("translatedText", "").strip()
                    if translated and translated != processed_text:
                        translation_attempts.append(("LibreTranslate", translated))
            except:
                pass
            
            # เลือกคำแปลที่ดีที่สุด
            if translation_attempts:
                # ให้ความสำคัญกับ MyMemory ก่อน (มักจะดีกว่าสำหรับภาษาตะวันออก)
                for api_name, translation in translation_attempts:
                    if api_name == "MyMemory":
                        best_translation = translation
                        break
                else:
                    best_translation = translation_attempts[0][1]
                
                # ปรับปรุงคำแปลสำหรับบริบทมังงะ
                if context_type == 'manga':
                    best_translation = self.post_process_manga_translation(best_translation, source_lang)
                
                return {
                    "success": True,
                    "translated_text": best_translation,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "api_used": translation_attempts[0][0],
                    "context_type": context_type
                }
            
            return {"error": "ไม่สามารถแปลข้อความได้ในขณะนี้"}
            
        except Exception as e:
            return {"error": f"เกิดข้อผิดพลาดในการแปล: {str(e)}"}
    
    def post_process_manga_translation(self, text, source_lang):
        """ปรับปรุงคำแปลสำหรับมังงะ"""
        # ลบวงเล็บซ้ำซ้อน
        text = re.sub(r'\(\(.*?\)\)', '', text)
        
        # ปรับปรุงการเว้นวรรค
        text = re.sub(r'\s+', ' ', text)
        
        # สำหรับภาษาญี่ปุ่น: ปรับปรุงคำสรรพนาม
        if source_lang == 'ja':
            text = text.replace('คุณ (ชาย)', 'แก')
            text = text.replace('ฉัน (ชาย)', 'กู')
            text = text.replace('ฉัน (หญิง)', 'ฉัน')
        
        return text.strip()
    
    def process_image_with_overlay(self, image_input, target_lang='th', is_manga=False):
        """ประมวลผลภาพและส่งคืนข้อมูลสำหรับซ้อนคำแปล"""
        start_time = time.time()
        
        # OCR
        language_setting = 'jpn+kor+chi_sim' if is_manga else 'eng+tha+jpn+kor+chi_sim'
        ocr_result = self.improve_ocr_accuracy(image_input, is_manga, language_setting)
        
        if ocr_result.get("error"):
            return {"error": ocr_result["error"]}
        
        # แปลภาษา
        context_type = 'manga' if is_manga else 'general'
        translate_result = self.context_aware_translate(
            ocr_result["text"], 
            target_lang, 
            ocr_result["detected_language"],
            context_type
        )
        
        if translate_result.get("error"):
            return {"error": f"OCR สำเร็จแต่แปลไม่ได้: {translate_result['error']}"}
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "original_text": ocr_result["text"],
            "translated_text": translate_result["translated_text"],
            "source_lang": translate_result["source_lang"],
            "target_lang": translate_result["target_lang"],
            "text_overlay": ocr_result.get("text_overlay", {}),
            "processing_time": f"{processing_time:.2f}s",
            "word_count": ocr_result["word_count"],
            "is_manga": is_manga
        }

# สร้าง instance
app = ProfessionalTranslationApp()

# Flask API สำหรับ Chrome Extension
flask_app = Flask(__name__)

@flask_app.route('/api/translate-with-overlay', methods=['POST'])
def api_translate_with_overlay():
    """API สำหรับ Chrome Extension ที่ต้องการซ้อนคำแปล"""
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        target_lang = data.get('target_lang', 'th')
        is_manga = data.get('is_manga', False)
        
        result = app.process_image_with_overlay(image_data, target_lang, is_manga)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "Professional Translation API",
        "features": ["context_aware", "manga_support", "overlay_data"]
    })

def run_flask():
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Gradio Interface
with gr.Blocks(
    title="Professional Translator - แม่นยำสูง", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1400px !important;
    }
    .accuracy-badge {
        background: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
    }
    """
) as demo:
    gr.Markdown("""
    # 🎯 Professional Translator - ความแม่นยำสูง
    **ใช้ AI ล่าสุด • เข้าใจบริบท • ซ้อนคำแปลบนภาพได้**
    """)
    
    with gr.Tab("🧠 แปลแบบเข้าใจบริบท"):
        gr.Markdown("### การแปลที่เข้าใจความหมายจริงๆ")
        
        with gr.Row():
            with gr.Column():
                context_text = gr.Textbox(
                    label="ข้อความต้นทาง",
                    placeholder="ป้อนข้อความที่ต้องการแปล...",
                    lines=5
                )
                
                with gr.Row():
                    context_source = gr.Dropdown(
                        choices=[("auto", "🔍 ตรวจจับอัตโนมัติ")] + 
                                [(code, f"{info['emoji']} {info['name']}") for code, info in app.supported_languages.items()],
                        label="ภาษาต้นทาง",
                        value="auto"
                    )
                    
                    context_target = gr.Dropdown(
                        choices=[(code, f"{info['emoji']} {info['name']}") for code, info in app.supported_languages.items()],
                        label="ภาษาปลายทาง",
                        value="th"
                    )
                
                context_type = gr.Radio(
                    choices=[
                        ("general", "📝 ทั่วไป"),
                        ("manga", "🎌 มังงะ/การ์ตูน"),
                        ("formal", "💼 ทางการ")
                    ],
                    label="บริบทการแปล",
                    value="general"
                )
                
                context_btn = gr.Button("🧠 แปลแบบเข้าใจบริบท", variant="primary")
            
            with gr.Column():
                context_output = gr.Textbox(
                    label="ผลการแปล",
                    lines=5,
                    show_copy_button=True
                )
                
                gr.Markdown("""
                **✨ คุณสมบัติพิเศษ:**
                - เข้าใจบริบทมังงะและการ์ตูน
                - รู้จักคำศัพท์เฉพาะ
                - ปรับรูปแบบการแปลตามบริบท
                """)
        
        def handle_context_translate(text, source, target, context):
            result = app.context_aware_translate(text, target, source, context)
            if result.get("success"):
                return f"🎯 แปลแบบเข้าใจบริบท ({context}):\n\n{result['translated_text']}"
            else:
                return f"❌ {result.get('error', 'เกิดข้อผิดพลาด')}"
        
        context_btn.click(
            handle_context_translate,
            inputs=[context_text, context_source, context_target, context_type],
            outputs=[context_output]
        )
    
    with gr.Tab("📖 มังงะแม่นยำสูง"):
        gr.Markdown("### 🎌 โหมดมังงะ - ประมวลผลภาพพิเศษ")
        
        with gr.Row():
            with gr.Column():
                manga_image_high = gr.Image(
                    label="อัพโหลดภาพมังงะ",
                    type="pil",
                    sources=["upload", "clipboard", "url"],
                    height=300
                )
                
                manga_target_high = gr.Dropdown(
                    choices=[(code, info['name']) for code, info in app.supported_languages.items() 
                            if code not in ['ja', 'ko', 'zh']],
                    label="แปลเป็นภาษา",
                    value="th"
                )
                
                advanced_btn = gr.Button("🎌 ประมวลผลมังงะแบบแม่นยำ", variant="stop")
            
            with gr.Column():
                manga_original = gr.Textbox(
                    label="ข้อความต้นทางที่ตรวจพบ",
                    lines=4,
                    show_copy_button=True
                )
                manga_translated_high = gr.Textbox(
                    label="ผลการแปล (เข้าใจบริบท)",
                    lines=4,
                    show_copy_button=True
                )
        
        def handle_advanced_manga(image, target_lang):
            result = app.process_image_with_overlay(image, target_lang, True)
            if result.get("success"):
                return (
                    f"📖 ต้นทาง ({result['source_lang']}):\n\n{result['original_text']}",
                    f"🌐 แปลเป็น {app.supported_languages[result['target_lang']]['name']}:\n\n{result['translated_text']}\n\n⏱️ ใช้เวลา: {result['processing_time']}"
                )
            else:
                return f"❌ {result.get('error')}", ""
        
        advanced_btn.click(
            handle_advanced_manga,
            inputs=[manga_image_high, manga_target_high],
            outputs=[manga_original, manga_translated_high]
        )
    
    with gr.Tab("🔧 สำหรับ Developer"):
        gr.Markdown("### API สำหรับ Chrome Extension")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                **🎯 Endpoint พิเศษสำหรับซ้อนคำแปล:**
                ```http
                POST /api/translate-with-overlay
                {
                  "image": "base64_image_data",
                  "target_lang": "th",
                  "is_manga": false
                }
                ```
                
                **📋 Response:**
                ```json
                {
                  "success": true,
                  "original_text": "原文",
                  "translated_text": "คำแปล",
                  "text_overlay": {
                    "Lines": [
                      {
                        "LineText": "ข้อความ",
                        "Words": [
                          {
                            "WordText": "คำ",
                            "Left": 100,
                            "Top": 50,
                            "Height": 20,
                            "Width": 40
                          }
                        ]
                      }
                    ]
                  }
                }
                ```
                """)
            
            with gr.Column():
                gr.Markdown("""
                **🛠️ ข้อมูลตำแหน่งสำหรับซ้อนคำแปล:**
                
                ใช้ข้อมูลจาก `text_overlay` เพื่อ:
                - หาตำแหน่งข้อความต้นทาง
                - ซ้อนคำแปลในตำแหน่งเดียวกัน
                - ปรับขนาดฟอนต์ให้เหมาะสม
                
                **🎨 ตัวอย่างการใช้งาน:**
                ```javascript
                // สร้าง overlay element
                const overlay = document.createElement('div');
                overlay.style.position = 'absolute';
                overlay.style.left = word.Left + 'px';
                overlay.style.top = word.Top + 'px';
                overlay.style.background = 'rgba(255,255,255,0.9)';
                overlay.innerText = translatedText;
                ```
                """)

# เริ่ม Flask server
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False
    )
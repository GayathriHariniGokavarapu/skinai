import os
import json
import logging
from flask import Flask, render_template, request, jsonify
from googletrans import Translator
from openai import OpenAI
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "skincare-chatbot-secret-key")

# Initialize external services
translator = Translator()
# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "default_openai_key"))

# Load skincare data
def load_json_data(filename):
    """Load JSON data from data directory"""
    try:
        with open(f'data/{filename}', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {filename}")
        return {}

# Global data storage
skincare_faqs = load_json_data('skincare_faqs.json')
products = load_json_data('products.json')

def detect_language(text):
    """Detect language of input text"""
    try:
        # Check if text contains Telugu characters
        telugu_pattern = r'[\u0C00-\u0C7F]'
        if re.search(telugu_pattern, text):
            logger.debug(f"Detected Telugu characters in text")
            return 'te'
        
        # Simple language detection based on common Telugu words
        telugu_words = ['చర్మం', 'ముఖం', 'సూర్య', 'నూనె', 'పొడిగా', 'తేమ', 'మచ్చలు', 'దద్దుర్లు']
        text_lower = text.lower()
        
        for word in telugu_words:
            if word in text:
                logger.debug(f"Detected Telugu word: {word}")
                return 'te'
        
        # Default to English for everything else
        logger.debug(f"Defaulting to English for text: {text[:50]}...")
        return 'en'
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return 'en'  # Default to English

def translate_text(text, target_language):
    """Translate text to target language"""
    try:
        if target_language == 'auto':
            return text
        
        # Basic Telugu to English translation dictionary for common skincare terms
        telugu_to_english = {
            'చర్మం': 'skin',
            'ముఖం': 'face', 
            'నూనె': 'oil',
            'పొడిగా': 'dry',
            'తేమ': 'moisture',
            'మచ్చలు': 'spots',
            'దద్దుర్లు': 'acne',
            'సూర్య': 'sun',
            'ఎలా': 'how',
            'చూసుకోవాలి': 'to care',
            'సన్‌స్క్రీన్': 'sunscreen',
            'మాయిశ్చరైజర్': 'moisturizer'
        }
        
        english_to_telugu = {
            'skin': 'చర్మం',
            'face': 'ముఖం',
            'oil': 'నూనె', 
            'oily': 'నూనెగా',
            'dry': 'పొడిగా',
            'moisture': 'తేమ',
            'acne': 'దద్దుర్లు',
            'spots': 'మచ్చలు',
            'sun': 'సూర్య',
            'sunscreen': 'సన్‌స్క్రీన్',
            'moisturizer': 'మాయిశ్చరైజర్',
            'cleanser': 'క్లెన్సర్',
            'routine': 'రొటీన్',
            'care': 'చూసుకోవడం',
            'how': 'ఎలా',
            'daily': 'రోజువారీ',
            'gentle': 'మృదువుగా'
        }
        
        if target_language == 'en' and text:
            # Simple word-based translation from Telugu to English
            translated_words = []
            words = text.split()
            for word in words:
                # Remove punctuation for lookup
                clean_word = word.strip('?.,!;:')
                if clean_word in telugu_to_english:
                    translated_words.append(telugu_to_english[clean_word])
                else:
                    translated_words.append(word)
            
            translated = ' '.join(translated_words)
            # If no translation found, try to detect common patterns
            if 'చర్మం' in text:
                translated = 'How to take care of skin?'
            elif 'ముఖం' in text:
                translated = 'How to take care of face?'
            
            logger.debug(f"Translated Telugu to English: '{translated}'")
            return translated
            
        elif target_language == 'te':
            # Simple word-based translation from English to Telugu
            translated_words = []
            words = text.split()
            for word in words:
                clean_word = word.lower().strip('?.,!;:')
                if clean_word in english_to_telugu:
                    translated_words.append(english_to_telugu[clean_word])
                else:
                    translated_words.append(word)
            
            translated = ' '.join(translated_words)
            logger.debug(f"Translated English to Telugu: '{translated}'")
            return translated
        
        return text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def is_skincare_related(message):
    """Check if message is related to skincare"""
    skincare_keywords = [
        'skin', 'face', 'acne', 'pimple', 'dry', 'oily', 'sensitive', 
        'wrinkle', 'aging', 'sunscreen', 'moisturizer', 'cleanser',
        'serum', 'toner', 'routine', 'blackhead', 'whitehead',
        'dark spots', 'pigmentation', 'rosacea', 'eczema', 'dermatitis'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in skincare_keywords)

def get_openai_response(message, language='en'):
    """Get response from OpenAI for skincare queries"""
    try:
        # Create skincare context
        skincare_context = """
        You are a professional skincare consultant AI assistant. You specialize in:
        - Skincare routines for different skin types (oily, dry, combination, sensitive)
        - Acne treatment and prevention
        - Anti-aging skincare advice
        - Sunscreen recommendations and importance
        - Product ingredients and their benefits
        - Common skin conditions and care tips
        
        Always provide helpful, accurate, and safe skincare advice. If asked about serious medical conditions, 
        recommend consulting a dermatologist.
        
        Product recommendations should be based on the available products in the database.
        """
        
        # Add product information to context
        product_info = "Available products:\n"
        for category, items in products.get('categories', {}).items():
            product_info += f"{category}:\n"
            for item in items:
                product_info += f"- {item['name']}: {item['description']} (${item['price']})\n"
        
        full_context = skincare_context + "\n\n" + product_info
        
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": full_context},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        return content.strip() if content else "I'm sorry, I couldn't generate a response."
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again later."

def search_faqs(query):
    """Search through skincare FAQs for relevant answers with improved matching"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    best_match = None
    best_score = 0
    
    for category, faqs in skincare_faqs.get('categories', {}).items():
        for faq in faqs:
            question = faq['question'].lower()
            keywords = faq.get('keywords', [])
            answer = faq['answer']
            
            # Calculate relevance score
            score = 0
            
            # Direct keyword matches (high priority)
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 10
            
            # Question word matches (medium priority)  
            question_words = set(question.split())
            common_words = query_words & question_words
            score += len(common_words) * 5
            
            # Specific term matching for better accuracy
            if 'skin type' in query_lower and 'skin type' in question:
                score += 15
            if 'dry skin' in query_lower and 'dry' in keywords:
                score += 15
            if 'oily skin' in query_lower and 'oily' in keywords:
                score += 15
            if 'acne' in query_lower and 'acne' in keywords:
                score += 15
            if 'sunscreen' in query_lower and 'sunscreen' in keywords:
                score += 15
            if 'routine' in query_lower and 'routine' in keywords:
                score += 15
            
            # Update best match if this score is higher
            if score > best_score:
                best_score = score
                best_match = answer
    
    # Only return if we have a reasonable match
    return best_match if best_score >= 5 else None

def get_fallback_response(message):
    """Provide helpful fallback responses when OpenAI is unavailable"""
    message_lower = message.lower()
    
    # Common skincare topics with helpful general advice
    if any(word in message_lower for word in ['routine', 'regimen', 'steps']):
        return "A basic skincare routine includes: cleansing, moisturizing, and sun protection. For specific product recommendations, I'd be happy to help if you tell me your skin type (oily, dry, combination, or sensitive)."
    
    elif any(word in message_lower for word in ['product', 'recommend', 'suggestion']):
        return "I can recommend products based on your skin type and concerns. Could you tell me more about your skin type (oily, dry, combination, sensitive) and any specific concerns you have?"
    
    elif any(word in message_lower for word in ['breakout', 'pimple', 'blemish']):
        return "For breakouts, try: gentle cleansing twice daily, spot treatments with salicylic acid or benzoyl peroxide, and avoid touching your face. If breakouts persist, consider consulting a dermatologist."
    
    elif any(word in message_lower for word in ['ingredient', 'what is', 'explain']):
        return "I can help explain skincare ingredients! Common beneficial ones include: niacinamide (oil control), hyaluronic acid (hydration), salicylic acid (acne treatment), and retinol (anti-aging). What specific ingredient are you curious about?"
    
    elif any(word in message_lower for word in ['sensitive', 'irritation', 'react']):
        return "For sensitive skin: use fragrance-free, gentle products, patch test new products, avoid over-exfoliating, and look for ingredients like ceramides and niacinamide. If irritation persists, consult a dermatologist."
    
    else:
        return "I'd love to help with your skincare question! Could you provide more details about your skin type or specific concerns? I can offer advice on routines, products, or specific skin issues."

def analyze_skin_image(image_data, user_message=""):
    """Analyze skin image using OpenAI Vision API"""
    try:
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional skincare expert analyzing skin images. 
                    Provide detailed, helpful skincare advice based on what you can observe in the image.
                    
                    Focus on:
                    1. General skin condition observations
                    2. Skincare routine recommendations
                    3. Product suggestions from available products
                    4. Preventive care advice
                    
                    Always remind users to consult a dermatologist for serious concerns or medical conditions.
                    Be encouraging and supportive in your analysis."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please analyze this skin image and provide personalized skincare advice. {user_message}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        return content.strip() if content else "I was unable to analyze the image. Please try uploading a clearer photo."
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return "I'm having trouble analyzing your image right now. Please ensure you have a clear photo and try again. For immediate skincare advice, feel free to describe your skin concerns in text."

@app.route('/')
def index():
    """Serve the main page with chatbot widget"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        image_data = data.get('image_data')
        is_image_analysis = data.get('image_analysis', False)
        
        # Handle image analysis
        if image_data and is_image_analysis:
            response = analyze_skin_image(image_data, user_message)
            return jsonify({
                'success': True,
                'response': response,
                'detected_language': 'en',
                'original_message': user_message
            })
        
        if not user_message:
            return jsonify({
                'error': 'Message is required',
                'response': 'Please enter a message.'
            }), 400
        
        # Detect language
        detected_lang = detect_language(user_message)
        is_telugu = detected_lang == 'te'
        
        logger.info(f"Received message: '{user_message}' in language: {detected_lang}")
        
        # If Telugu, translate to English for processing
        english_message = user_message
        if is_telugu:
            english_message = translate_text(user_message, 'en')
            logger.info(f"Translated to English: '{english_message}'")
        
        # Check if query is skincare-related
        if not is_skincare_related(english_message):
            response = "I can connect you to human support for this question."
        else:
            # First try FAQ search
            faq_response = search_faqs(english_message)
            
            if faq_response:
                response = faq_response
                logger.info("Found FAQ match")
            else:
                # Try OpenAI for more complex queries, with fallback
                try:
                    response = get_openai_response(english_message)
                    logger.info("Generated OpenAI response")
                except Exception as e:
                    logger.error(f"OpenAI failed, using fallback response: {e}")
                    # Provide a helpful fallback response for common skincare topics
                    response = get_fallback_response(english_message)
        
        # If original message was in Telugu, translate response back
        if is_telugu and response:
            response = translate_text(response, 'te')
            logger.info(f"Translated response back to Telugu: '{response}'")
        
        return jsonify({
            'response': response,
            'detected_language': detected_lang,
            'original_message': user_message
        })
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'response': 'I apologize, but I encountered an error. Please try again.'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'skincare-chatbot'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

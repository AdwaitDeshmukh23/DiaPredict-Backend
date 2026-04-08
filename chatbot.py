import random
import os
import httpx
import anthropic
from models import ChatRequest, ChatResponse

# Load environment variables (keys should be set in Render dashboard)
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

KEYWORD_MAP = {
    "eat": ["eat", "food", "diet", "meal", "nutrition", "carb", "sugar", "fruit", "vegetable", "drink"],
    "exercise": ["exercise", "workout", "fitness", "walk", "gym", "run", "activity", "sport", "yoga", "swim", "plan"],
    "reduce": ["reduce", "control", "improve", "lower", "prevent", "risk"],
}

def detect_intent(message: str) -> str:
    msg_lower = message.lower()
    for intent, keywords in KEYWORD_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            return intent
    return "default"

def generate_personalized_context(data: ChatRequest) -> str:
    context = []
    if data.bmi is not None:
        if data.bmi > 25:
            context.append(random.choice([f"Since your BMI is {data.bmi} (overweight), focusing on weight management is key.", f"With a BMI of {data.bmi}, a slight weight reduction can greatly help."]))
        elif data.bmi < 18.5:
            context.append(f"Your BMI is {data.bmi} (underweight). Make sure to eat nutritious, balanced meals to build strength.")
        else:
            context.append(f"Your BMI of {data.bmi} is in the healthy range! Keep maintaining that balance.")

    if data.glucose is not None and data.glucose > 0:
        if data.glucose >= 126:
            context.append(f"Your glucose level is high ({data.glucose}). We need to focus on reducing sugar intake right away.")
        elif data.glucose >= 100:
            context.append(f"Your glucose is {data.glucose} (pre-diabetic range). Monitoring carbs is important.")
    
    if data.age is not None and data.age > 45:
        context.append("Given your age, regular monitoring and consistency in your routine are highly recommended.")
        
    return " ".join(context) if context else ""

def chatbot_response(user_query: str, user_data: ChatRequest) -> str:
    intent = detect_intent(user_query)
    category = (user_data.risk_category or "medium").lower()
    
    # Base Responses
    if intent == "eat":
        if category == "high":
            base = "🚨 Strict Diet Required: Eliminate all refined sugars and sweets. Focus on high-protein, low-carb meals with plenty of leafy greens. Limit carbs to <130g daily."
        elif category == "medium":
            base = "🥗 Balanced Diet Needed: Adopt a low-glycaemic index diet (oats, lentils, brown rice). Eat 5 small meals a day to prevent sudden glucose spikes."
        else:
            base = "🍏 Healthy Diet: Focus on whole grains, legumes, and lean protein. Keep up your hydration with 8+ glasses of water daily."
    
    elif intent == "exercise":
        if category == "high":
            base = "🏃 High Priority Exercise: Aim for a 45-minute daily low-impact walk and 4x weekly strength training sessions. Consistency is absolutely crucial."
        elif category == "medium":
            base = "🏋️ Moderate Exercise: Try 30 minutes of brisk walking or cycling daily. Post-meal 15-minute walks are extremely beneficial for stabilizing sugar."
        else:
            base = "🚶 Active Lifestyle: Keep up at least 150 minutes of moderate activity per week. Add some strength training if you aren't already."
            
    elif intent == "reduce":
        if category == "high":
            base = "⚠️ Urgent Action: You must consult a doctor soon. Strict dietary control and daily exercise are your best tools to combat this risk."
        elif category == "medium":
            base = "📉 Risk Reduction: Monitor your fasting glucose weekly, stay active daily, and keep a close eye on your caloric intake to reduce your risk gently."
        else:
            base = "🛡️ Ongoing Prevention: Maintain your current healthy routine. Keep stress low and ensure you get an annual general checkup."
            
    else:
        # Default response
        return "I'm DiaBot! 🤖 Please ask me specifically about your **diet**, **exercise**, or how to **reduce your diabetes risk** so I can give you the best advice."
        
    # Inject personalization
    personalization = generate_personalized_context(user_data)
    
    if personalization:
        return f"{personalization}\n\n{base}"
    return base

async def get_chat_response(request: ChatRequest) -> ChatResponse:
    health_context = generate_personalized_context(request)
    risk_info = f"Risk Category: {request.risk_category or 'Medium'}, BMI: {request.bmi or 'N/A'}, Glucose: {request.glucose or 'N/A'}, Age: {request.age or 'N/A'}"
    system_prompt = f"You are DiaBot, a helpful AI diabetes health assistant. {risk_info}. {health_context}. Keep your advice concise, medically conservative, and always specify you aren't a doctor."

    # --- 1. Try Anthropic (Claude) ---
    if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("sk-your"):
        try:
            client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            message = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": request.user_message}]
            )
            return ChatResponse(
                reply=message.content[0].text,
                source="anthropic",
                disclaimer="Personalised AI advice. Not a medical diagnosis."
            )
        except Exception as e:
            print(f"Anthropic Error: {e} - falling back to HF")

    # --- 2. Try Hugging Face (Mistral) ---
    if HF_API_KEY and not HF_API_KEY.startswith("hf_your"):
        try:
            prompt = f"<s>[INST] {system_prompt}\n\nUser Question: {request.user_message} [/INST]"
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.7}},
                    timeout=15.0
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        full_text = result[0].get("generated_text", "")
                        reply = full_text.split("[/INST]")[-1].strip() if "[/INST]" in full_text else full_text.strip()
                        return ChatResponse(
                            reply=reply,
                            source="huggingface",
                            disclaimer="Open-source AI advice. Not a medical diagnosis."
                        )
        except Exception as e:
            print(f"Hugging Face Error: {e} - falling back to rule-based")

    # --- 3. Fallback to Rule-based Engine ---
    reply = chatbot_response(request.user_message, request)
    return ChatResponse(
        reply=reply, 
        source="rule_based", 
        disclaimer="Rule-based health advice. Not a medical diagnosis."
    )


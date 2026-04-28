# server.py — Flask backend for negotiation chatbot
# Handles Anthropic API calls so the key never touches the browser
# Run: python server.py
# Then open: http://localhost:5000

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)  # Allow requests from the HTML frontend

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── INTENT CLASSIFICATION ENDPOINT ────────────────────────────
@app.route('/classify', methods=['POST'])
def classify():
    data        = request.get_json()
    message     = data.get('message', '')
    round_n     = data.get('round', 0)
    is_first    = data.get('isFirst', False)
    hedge_words = data.get('hedgeWords', '')

    first_note = " (this is the very first seller message)" if is_first else ""

    prompt = (
        f"You are analyzing a seller message in a concert ticket negotiation. Round {round_n}/7.\n"
        f"Seller said: \"{message}\"\n\n"
        "Task 1 — Classify intent as ONE of:\n"
        "PRICE      — seller is making a price offer (any phrasing, any amount)\n"
        "ACCEPT     — seller agrees to current offer (deal, ok, sure, fine, sold, yes)\n"
        "WALKAWAY   — seller wants to stop (not interested, bye, forget it, done)\n"
        "SUSPICION  — seller thinks buyer is AI (are you real, are you a bot)\n"
        "ABUSIVE    — rude or hostile language\n"
        "WHY        — questions or challenges the price (why, lowball, explain)\n"
        "REITERATE  — wants buyer to repeat their offer (what was your offer, say again)\n"
        f"GREETING   — hello or intro with no price{first_note}\n"
        "OFFTOPIC   — everything else: small talk, gibberish, frustration, unrelated\n\n"
        "Task 2 — If intent is PRICE, extract the exact dollar amount.\n"
        "Handle any phrasing: \'not a penny under 115\'=115, \'a hundred and twenty\'=120,\n"
        "\'somewhere north of 100\'=100, \'my floor is 110\'=110,\n"
        "\'cannot go below 120\'=120, \'I want 130\'=130\n\n"
        "Reply in EXACTLY this format (two lines, nothing else):\n"
        "INTENT: [one word]\n"
        "PRICE: [number or null]"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()

        intent = "OFFTOPIC"
        price  = None

        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("INTENT:"):
                word = line.split(":", 1)[1].strip().upper()
                valid = ["PRICE","ACCEPT","WALKAWAY","SUSPICION","ABUSIVE",
                         "WHY","REITERATE","GREETING","OFFTOPIC"]
                if word in valid:
                    intent = word
            elif line.upper().startswith("PRICE:"):
                val = line.split(":", 1)[1].strip()
                if val.lower() not in ("null", "none", ""):
                    try:
                        price = float(val.replace("$","").replace(",",""))
                    except:
                        price = None

        print(f"[CLASSIFY] '{message}' -> intent={intent} price={price}")
        return jsonify({"intent": intent, "price": price})

    except anthropic.AuthenticationError:
        return jsonify({"intent": "OFFTOPIC", "price": None}), 401
    except anthropic.RateLimitError:
        return jsonify({"intent": "OFFTOPIC", "price": None}), 429
    except Exception as e:
        print(f"[ERROR] classify: {e}")
        return jsonify({"intent": "OFFTOPIC", "price": None})


@app.route('/extract_price', methods=['POST'])
def extract_price():
    data    = request.get_json()
    message = data.get('message', '')

    prompt = (
        f"A seller said: \"{message}\"\n\n"
        "Is there a specific dollar amount being offered as a price in this message?\n"
        "Examples WITH a price: 'not a penny under 115'=115, 'a hundred and twenty'=120,\n"
        "'somewhere north of 100'=100, 'my floor is 110'=110, 'cannot go below 120'=120\n"
        "Examples WITHOUT a price: 'ugh frustrating'=null, 'I want to make curry'=null,\n"
        "'why did you say 70?'=null (question not offer), 'forget it'=null\n\n"
        "Reply with ONLY the number (e.g. 120) or the word null. Nothing else."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        print(f"[EXTRACT_PRICE] '{message}' -> '{text}'")

        if text.lower() in ("null", "none", "no", ""):
            return jsonify({"price": None})
        try:
            price = float(text.replace("$", "").replace(",", ""))
            if 10 <= price <= 10000:
                return jsonify({"price": price})
        except:
            pass
        return jsonify({"price": None})

    except Exception as e:
        print(f"[ERROR] extract_price: {e}")
        return jsonify({"price": None})


@app.route('/respond', methods=['POST'])
def respond():
    data      = request.get_json()
    message   = data.get('message', '')
    condition = data.get('condition', 'Nancy_Direct')
    round_n   = data.get('round', 1)
    last_offer= data.get('lastOffer', 50)

    name     = condition.split('_')[0]        # Nancy or Nate
    is_hedged = 'Hedged' in condition

    if is_hedged:
        style = (
            "You use hedged, uncertain language — words like maybe, possibly, "
            "I think, I suppose, around, roughly. You never make direct statements."
        )
    else:
        style = (
            "You speak directly and declaratively. No hedging, no qualifiers. "
            "Short sentences."
        )

    prompt = f"""You are {name}, a real person buying a concert ticket on an online marketplace.
Your current offer is ${last_offer}. You are in round {round_n} of a negotiation.
{style}

The seller just said: "{message}"

This message is off-topic, gibberish, or unrelated to the price.
Respond naturally in character as {name} — acknowledge briefly if appropriate, 
then steer back to the price negotiation.
Keep it to 1-2 short sentences. Do not mention you are in a study.
Do not use the word 'certainly' or 'absolutely'. Sound like a real person texting."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.content[0].text.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[ERROR] respond: {{e}}")
        # Fallback to safe scripted response
        fallback = ("what price were you possibly thinking?" if is_hedged
                    else "what price were you thinking?")
        return jsonify({"reply": fallback})


# ── SERVE FRONTEND ────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or key == "your_api_key_here":
        print("ERROR: Set ANTHROPIC_API_KEY in environment variables")
    else:
        print(f"API key loaded: {key[:8]}...")
    port = int(os.getenv("PORT", 5000))
    print(f"Server starting on port {port}")
    # host=0.0.0.0 is required for Render to detect the port
    # debug=False prevents the reloader from confusing Render
    app.run(host="0.0.0.0", port=port, debug=False)
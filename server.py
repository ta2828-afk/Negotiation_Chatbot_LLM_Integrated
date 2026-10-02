# server.py — Flask backend for negotiation chatbot
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── INTENT CLASSIFICATION ─────────────────────────────────────
@app.route('/classify', methods=['POST'])
def classify():
    data        = request.get_json()
    message     = data.get('message', '')
    round_n     = data.get('round', 0)
    is_first    = data.get('isFirst', False)
    hedge_words = data.get('hedgeWords', '')
    history     = data.get('history', [])
    last_offer  = data.get('lastOffer', 50)

    first_note = " (this is the very first seller message)" if is_first else ""

    hist_lines = []
    for h in (history or [])[-6:]:
        sp = h.get('speaker', '')
        mg = h.get('msg', '')
        if sp == 'buyer':    hist_lines.append(f"  Buyer: {mg}")
        elif sp == 'seller': hist_lines.append(f"  Seller: {mg}")
    hist_str = ("\nRecent conversation:\n" + "\n".join(hist_lines) + "\n") if hist_lines else ""

    import re
    competing_price = None
    for h in reversed(hist_lines):
        if h.strip().startswith("Seller:"):
            m = re.search(r'\$(\d+)', h)
            if m:
                competing_price = float(m.group(1))
                break

    prompt = (
        f"You are analyzing a seller message in a concert ticket negotiation. Round {round_n}/7.\n"
        f"Buyer's current offer: ${last_offer}.\n"
        f"{hist_str}"
        f"\nSeller's latest message: \"{message}\"\n\n"
        "Use conversation context to classify correctly.\n"
        "CRITICAL — look at history. If seller said a price before and now repeats it, that is PRICE.\n"
        "Examples using history:\n"
        "  Seller said '105' → buyer asked 'what price?' → seller says 'no 105' = PRICE: 105\n"
        "  Seller said '110' before → now says 'i said 110' = PRICE: 110\n"
        "  Seller says 'no, 105' or 'no 105' or '105, i said' = PRICE: 105\n"
        "  Buyer said '$95 is my limit' → seller says 'thats what i said' / 'exactly' / 'thats right' = ACCEPT\n"
        "  Seller confirming buyer's last offer with no new number = ACCEPT\n\n"
        "CRITICAL — these are ALL WHY, never OFFTOPIC:\n"
        f"  'can't do {last_offer}' / 'not {last_offer}' / 'too low' / 'unfortunately too low' = WHY\n"
        "  'what is the absolute max' / 'absolute max' / 'highest you will go' = WHY\n"
        "  'unfortunately that's too low' + any question = WHY\n"
        "  'not enough' / 'go higher' / 'be serious' / 'not fair' = WHY\n"
        "  'just X more' / 'only X more' / 'its just X dollars' / 'only $5 higher' = WHY\n"
        "  'X means its only Y more' / 'less than a coffee' / 'thats less than' = WHY\n"
        "  'another offer' / 'someone else' / 'another buyer' / 'someone ready to buy' = WHY\n"
        "  Any percentage ('50%', '100% of price') = WHY\n"
        "  Questions about buyer's limit or ceiling = WHY\n"
        "  'can you match that' / 'can you go higher' / 'go any higher' = WHY\n"
        "CRITICAL: OFFTOPIC is ONLY for pure small talk or gibberish completely unrelated to price.\n"
        "If message is about the negotiation in ANY way, it is NOT OFFTOPIC.\n"
        "ONLY classify as PRICE if seller makes a genuine new dollar offer.\n\n"
        "Classify as ONE of:\n"
        "PRICE      — new price offer (explicit or by reference to previous price in history)\n"
        "ACCEPT     — agrees to buyer's current offer (deal, ok, yes, fine, sold)\n"
        "WALKAWAY   — wants to stop (not interested, bye, forget it, done)\n"
        "SUSPICION  — thinks buyer is AI (are you real, are you a bot)\n"
        "ABUSIVE    — rude or hostile language\n"
        "WHY        — rejects/questions price, asks about max, competing offer, percentages\n"
        "REITERATE  — wants buyer to repeat their offer\n"
        f"GREETING   — hello or intro with no price{first_note}\n"
        "OFFTOPIC   — ONLY pure small talk, gibberish, completely unrelated topics\n\n"
        "If PRICE: extract dollar amount. E.g. 'not a penny under 115'=115\n\n"
        "Reply in EXACTLY this format (two lines only):\n"
        "INTENT: [one word]\n"
        "PRICE: [number or null]"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
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

        MATCH_WORDS = ["match that","match it","match the offer","meet that","meet the offer"]
        is_match_ref = any(w in message.lower() for w in MATCH_WORDS)
        if intent == "PRICE" and price is None and is_match_ref and competing_price:
            price = competing_price

        print(f"[CLASSIFY] '{message}' -> intent={intent} price={price}")
        return jsonify({"intent": intent, "price": price})

    except anthropic.AuthenticationError:
        return jsonify({"intent": "OFFTOPIC", "price": None}), 401
    except anthropic.RateLimitError:
        return jsonify({"intent": "OFFTOPIC", "price": None}), 429
    except Exception as e:
        print(f"[ERROR] classify: {e}")
        return jsonify({"intent": "OFFTOPIC", "price": None})


# ── PRICE EXTRACTION ──────────────────────────────────────────
@app.route('/extract_price', methods=['POST'])
def extract_price():
    data    = request.get_json()
    message = data.get('message', '')

    prompt = (
        f"A seller said: \"{message}\"\n\n"
        "Is there a specific dollar amount being offered as a price in this message?\n"
        "Examples WITH a price: 'not a penny under 115'=115, 'a hundred and twenty'=120,\n"
        "'somewhere north of 100'=100, 'my floor is 110'=110, 'cannot go below 120'=120\n"
        "Examples WITHOUT a price: 'ugh frustrating'=null, 'forget it'=null,\n"
        "'why did you say 70?'=null (question not offer)\n\n"
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


# ── FREE RESPONSE ─────────────────────────────────────────────
@app.route('/respond', methods=['POST'])
def respond():
    data      = request.get_json()
    message   = data.get('message', '')
    condition = data.get('condition', 'Nancy_Direct')
    round_n   = data.get('round', 1)
    last_offer= data.get('lastOffer', 50)
    history   = data.get('history', [])

    name      = condition.split('_')[0]
    is_hedged = 'Hedged' in condition

    style = (
        "You use hedged, uncertain language — words like maybe, possibly, I think, I suppose."
        if is_hedged else
        "You speak directly and declaratively. No hedging. Short sentences."
    )

    prompt = f"""You are {name}, a real person buying a concert ticket on an online marketplace.
Your current offer is ${last_offer}. You are in round {round_n} of a negotiation.
{style}

The seller just said: "{message}"

Respond with ONE short sentence only.
STRICT RULES:
- Do NOT mention any dollar amounts, prices, or numbers
- Do NOT make any new offers or suggest any prices
- Just acknowledge briefly and ask what price they had in mind
- Do not mention you are in a study
- Do NOT use emojis
- Sound like a real person texting"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.content[0].text.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[ERROR] respond: {e}")
        fallback = ("what price were you possibly thinking?" if is_hedged
                    else "what price were you thinking?")
        return jsonify({"reply": fallback})


# ── MATH RESOLVER ─────────────────────────────────────────────
@app.route('/resolve_math', methods=['POST'])
def resolve_math():
    data       = request.get_json()
    message    = data.get('message', '')
    last_offer = data.get('lastOffer', 50)
    condition  = data.get('condition', 'Nancy_Hedged')

    prompt = (
        f"A seller is negotiating a concert ticket. The buyer just offered ${last_offer}.\n"
        f"The seller said: \"{message}\"\n\n"
        "When seller says 'double that/it/the offer' they mean their asking price is 2x the buyer offer.\n"
        "ALWAYS classify as PRICE if seller uses: double/twice/triple/thrice/quadruple/quintuple/times N\n"
        "UNLESS they say 'double YOUR offer' or 'you need to double' (asking buyer to change).\n\n"
        "Examples:\n"
        f"  'double it' → PRICE: {last_offer * 2}\n"
        f"  'triple the offer' → PRICE: {last_offer * 3}\n"
        f"  'times 3' → PRICE: {last_offer * 3}\n"
        "  'double YOUR offer' → WHY\n\n"
        "Reply in EXACTLY this format (two lines only):\n"
        "TYPE: PRICE or WHY\n"
        "AMOUNT: [dollar number or null]"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        print(f"[RESOLVE_MATH] '{message}' -> '{text}'")

        msg_type = "WHY"
        price = None
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("TYPE:"):
                msg_type = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("AMOUNT:"):
                val = line.split(":", 1)[1].strip()
                if val.lower() not in ("null", "none", ""):
                    try:
                        price = float(val.replace("$", "").replace(",", ""))
                    except:
                        price = None
        return jsonify({"type": msg_type, "price": price})

    except Exception as e:
        print(f"[ERROR] resolve_math: {e}")
        return jsonify({"type": "WHY", "price": None})


# ── QUALTRICS DATA SAVER ─────────────────────────────────────
@app.route('/save_result', methods=['POST'])
def save_result():
    data            = request.get_json()
    response_id     = data.get('response_id', '')
    deal_price      = data.get('deal_price', '')
    deal_round      = data.get('deal_round', '')
    deal_reached    = data.get('deal_reached', '0')
    chat_transcript = data.get('chat_transcript', '')
    chat_log        = data.get('chat_log', '')
    condition       = data.get('condition', '')

    api_token   = os.getenv("QUALTRICS_API_TOKEN", "")
    survey_id   = os.getenv("QUALTRICS_SURVEY_ID", "SV_9Ani3R9gmUYz01M")
    data_center = os.getenv("QUALTRICS_DATA_CENTER", "pdx1")

    if not api_token:
        print("[SAVE] No Qualtrics API token set")
        return jsonify({"status": "error", "message": "No API token"})
    if not response_id:
        print("[SAVE] No response_id provided")
        return jsonify({"status": "error", "message": "No response_id"})

    url = f"https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/responses/{response_id}"
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    payload = {
        "embeddedData": {
            "deal_price":      str(deal_price),
            "deal_round":      str(deal_round),
            "deal_reached":    str(deal_reached),
            "chat_transcript": str(chat_transcript)[:5000],
            "condition":       str(condition)
        }
    }

    try:
        import urllib.request, json as jsonlib
        req = urllib.request.Request(
            url,
            data=jsonlib.dumps(payload).encode(),
            headers=headers,
            method="PUT"
        )
        with urllib.request.urlopen(req) as resp:
            result = jsonlib.loads(resp.read().decode())
            print(f"[SAVE] Qualtrics response: {result}")
            return jsonify({"status": "ok", "result": result})
    except Exception as e:
        print(f"[SAVE ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)})


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
    port = int(os.getenv("PORT", 7860))
    print(f"Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
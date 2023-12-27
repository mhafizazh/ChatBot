from flask import Flask, render_template, request, jsonify
import bot

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)



def get_Chat_response(text, chat_history_ids=None):
    response = bot.generate_response(message=text)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

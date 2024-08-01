from flask import Flask, render_template, request, jsonify
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

app = Flask(__name__)

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

def generate_response(prompt):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="tf")
        print(f"Inputs: {inputs}")
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated response: {response}")
        return response
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "An error occurred during response generation."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        print(f"User input: {user_input}")
        response = generate_response(user_input)
        print(f"AI response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)

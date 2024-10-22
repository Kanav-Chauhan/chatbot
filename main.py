import streamlit as st
import random
from transformers import BloomTokenizerFast, BloomForCausalLM, pipeline

# # Load the BLOOM tokenizer and model
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
# model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")


# Load a smaller model instead of BLOOM-560m
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load the emotion analyzer
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

conversation_history = []

def analyze_emotion(user_input):
    emotions = emotion_analyzer(user_input)
    return emotions

def craft_prompt(user_input, emotions):
    dominant_emotion = max(emotions[0], key=lambda x: x['score'])['label']

    if dominant_emotion == "sadness":
        prompt = f"You said you're feeling sad. I'm really sorry to hear that. Can you tell me more about what's been going on?"
    elif dominant_emotion == "joy":
        prompt = f"I'm glad to hear you're happy! What made you feel this way?"
    elif dominant_emotion == "anger":
        prompt = f"It sounds like you're feeling angry. What’s bothering you?"
    else:
        prompt = f"Thanks for sharing how you're feeling. What else is on your mind?"

    return prompt

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=30,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    additional_phrases = [
        "I'm here for you.",
        "Your feelings are valid.",
        "Let's talk more about this.",
        "It's important to express what you're feeling."
    ]

    response += " " + random.choice(additional_phrases)

    return response.strip()

# Streamlit UI setup
st.title("Emotional Intelligence Chatbot")
st.write("This chatbot responds based on the emotions you express.")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        # Analyze user emotion and generate response
        emotions = analyze_emotion(user_input)
        prompt = craft_prompt(user_input, emotions)

        if st.session_state.conversation_history:
            context = st.session_state.conversation_history[-1]
            prompt = f"{context} {prompt}"

        bot_response = generate_response(prompt)

        # Update conversation history
        st.session_state.conversation_history.append(f"User: {user_input}")
        st.session_state.conversation_history.append(f"Bot: {bot_response}")

        # Display chat history
        for i in range(len(st.session_state.conversation_history) - 1, -1, -2):
            st.write(f"**You:** {st.session_state.conversation_history[i-1]}")
            st.write(f"**Bot:** {st.session_state.conversation_history[i]}")

# Add a clear button to reset chat history
if st.button("Clear Chat"):
    st.session_state.conversation_history = []
    st.write("Chat history cleared.")

import ollama

# Test 1: the chat model
response = ollama.chat(
    model="qwen3:1.7b",
    messages=[
        {"role": "user", "content": "What is electromagnetic induction? Answer in 2 sentences. /no_think"}
    ],
)
print("CHAT ANSWER:")
print(response["message"]["content"])

# Test 2: the embedding model
result = ollama.embed(
    model="nomic-embed-text",
    input="Electromagnetic induction produces voltage in a conductor.",
)
vector = result["embeddings"][0]
print("\nEMBEDDING:")
print("Length of vector:", len(vector))
print("First 5 numbers:", vector[:5])

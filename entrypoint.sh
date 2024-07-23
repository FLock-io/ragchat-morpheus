#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA3 model..."
ollama pull llama3:8b
echo "🟢 Done!"

echo "🔴 Retrieve all-minilm model..."
ollama pull all-minilm:latest
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
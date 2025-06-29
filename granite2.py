import json
import re
import requests
import os
import asyncio
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import plotly.graph_objects as go
import numpy as np # Added for numerical operations

# --- Mock AI Model and Workflow (for local testing without actual beeai framework) ---
class MockChatModel:
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name

    async def chat(self, messages):
        prompt = messages[-1]["content"].lower()
        summary = "A comprehensive summary of the meeting was generated."
        tasks = "- Task: Review meeting notes.\n- Task: Follow up on action items."

        if "database" in prompt or "bug" in prompt:
            summary = "The discussion focused on resolving a critical database bug."
            tasks = "- Task: Sarah to investigate database connection bug.\n- Task: Provide update on bug resolution."
        elif "dashboard" in prompt or "ui" in prompt:
            summary = "Updates were provided on the new dashboard UI development."
            tasks = "- Task: Team to finalize UI designs.\n- Task: Start implementing dashboard features."
        elif "launch timeline" in prompt:
            summary = "The meeting was about finalizing the product launch timeline, with discussions on delays and resource allocation."
            tasks = "- Task: Emily to ensure fix doesn't affect launch.\n- Task: Tom to prioritize tasks and loop Priya in.\n- Task: Priya to focus on security aspects."
        
        return {"choices": [{"message": {"content": json.dumps({"summary": summary, "tasks": tasks})}}]}

class MockAgentWorkflow:
    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def run(self, input_data):
        transcript = input_data.transcript
        
        # Simulate sentiment extraction (simplified)
        sentiment_chunks = []
        sentences = re.split(r'[.!?]', transcript)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                sentiment = 0.0 # Default neutral
                if "frustrating" in sentence.lower() or "delay" in sentence.lower() or "behind" in sentence.lower() or "cut corners" in sentence.lower():
                    sentiment = -0.5
                elif "best" in sentence.lower() or "prioritize" in sentence.lower():
                    sentiment = 0.3
                
                sentiment_chunks.append({
                    "chunk_id": i + 1,
                    "speaker": f"Speaker {i%3 + 1}", # Mock speaker
                    "text_excerpt": sentence,
                    "sentiment": sentiment
                })

        # Simulate getting summary and tasks
        messages = [{"role": "user", "content": f"Analyze the following meeting transcript for key summaries and actionable tasks. Format the response as a JSON object with 'summary' (string) and 'tasks' (multiline string):\n\n{transcript}"}]
        
        try:
            chat_response = await self.chat_model.chat(messages)
            content = chat_response["choices"][0]["message"]["content"]
            parsed_content = json.loads(content)
            summary = parsed_content.get("summary", "No summary provided.")
            tasks = parsed_content.get("tasks", "No tasks extracted.")
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            summary = "Error generating summary."
            tasks = "Error generating tasks."

        return {
            "summary": summary,
            "tasks": tasks,
            "sentiment_chunks": sentiment_chunks # Pass chunks for chart generation
        }

mock_chat_model = MockChatModel()
mock_agent_workflow = MockAgentWorkflow(mock_chat_model)

async def run(transcript: str):
    # This will now use the mock workflow
    result = await mock_agent_workflow.run(AgentWorkflowInput(transcript=transcript))
    
    summary = result["summary"]
    tasks = result["tasks"]
    sentiment_chunks = result["sentiment_chunks"]

    # --- Generate Sentiment Timeline Chart ---
    chunks = sentiment_chunks
    scores_1 = [c.get("sentiment", 0) for c in chunks]
    ids = [c.get("chunk_id", 0) for c in chunks]

    # Create hover text
    hover_text = [
        f"Speaker: {c['speaker']}<br>Text: {c['text_excerpt']}<br>Sentiment: {c['sentiment']:.2f}"
        for c in chunks
    ]

    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(go.Scatter(
        x=ids,
        y=scores_1,
        mode='lines+markers',
        marker=dict(size=10),
        text=hover_text,
        hoverinfo='text+y',
    ))

    fig_sentiment.update_layout(
        title="Meeting Sentiment Timeline",
        xaxis_title="Chunk",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1, 1]),
        template="plotly_white" # Ensure consistent theme if needed
    )

    # Convert Plotly figure to JSON
    sentiment_json = fig_sentiment.to_json()

    # --- Generate Culture Score Chart ---
    if scores_1:
        avg_sentiment = np.mean(scores_1)
    else:
        avg_sentiment = 0.0

    positive_count = sum(1 for s in scores_1 if s > 0.1)
    negative_count = sum(1 for s in scores_1 if s < -0.1)
    neutral_count = len(scores_1) - positive_count - negative_count

    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]

    if sum(values) == 0:
        values = [1, 1, 1] # Equal parts if no sentiment data to avoid empty pie

    fig_culture = go.Figure(data=[go.Pie(labels=labels,
                                         values=values,
                                         hole=.3)])

    fig_culture.update_layout(
        title_text=f"Meeting Culture Score (Avg Sentiment: {avg_sentiment:.2f})",
        template="plotly_white"
    )

    # Convert Plotly figure to JSON
    culture_json = fig_culture.to_json()

    return {
        "summary": summary,
        "tasks": tasks,
        "sentiment_chart_json": sentiment_json, # Send JSON, not HTML
        "culture_chart_json": culture_json # Send JSON, not HTML
    }

# Mock AgentWorkflowInput class
class AgentWorkflowInput:
    def __init__(self, transcript: str):
        self.transcript = transcript
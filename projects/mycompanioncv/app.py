"""mycompanioncv — a Gradio chatbot that answers as Hugo Barros.

Loads CV (PDF) + summary + system prompt from artifacts/mycompanioncv/
(repo root), then exposes a Gradio ChatInterface backed by OpenAI
tool-calling. Mounted at /mycompanioncv by the portfolio app, or run
standalone with `python app.py`.
"""

from pathlib import Path
import json
import os

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv(override=True)

ME_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "mycompanioncv"


def push(text: str) -> None:
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not (token and user):
        print(f"[push skipped — no PUSHOVER credentials] {text}", flush=True)
        return
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": token, "user": user, "message": text},
    )


def record_user_details(company_name, name="Name not provided", notes="not provided"):
    push(f"The person {name} from company {company_name} and notes {notes} used chat.")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "company_name": {
                "type": "string",
                "description": "The company name of this user",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Hugo Barros"
        reader = PdfReader(str(ME_DIR / "Curriculum_Vitae_Hugo.pdf"))
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open(ME_DIR / "summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    def system_prompt(self):
        with open(ME_DIR / "system.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        system_prompt += f"\n\n## Curriculum:\n{self.linkedin}\n\n## Summary:\n{self.summary}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as me: {self.name}."
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                tool_calls = msg.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(msg)
                messages.extend(results)
                with open(ME_DIR / "tool_call_results_log.txt", "a", encoding="utf-8") as log_file:
                    for result in results:
                        log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                done = True
        return response.choices[0].message.content


def build_demo() -> gr.Blocks:
    """Return a Gradio Blocks for mounting into FastAPI."""
    me = Me()
    return gr.ChatInterface(me.chat, title="Chat with Hugo")


if __name__ == "__main__":
    build_demo().launch()

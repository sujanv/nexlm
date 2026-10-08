"""
Enhancement: High-Performance Streaming Inference Server & Interactive Web UI

Features:
1. OpenAI-compatible /v1/chat/completions JSON endpoint.
2. Server-Sent Events (SSE) streaming mode for real-time token streaming.
3. Embedded dark-mode web user interface served at http://localhost:8080/
4. Zero external HTTP framework dependencies (pure Python http.server).
"""

import argparse
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import torch

from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer
from inference.generate import generate_stream, generate


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NexLM: Interactive Neural LLM Playground</title>
  <style>
    :root { --bg: #0d1117; --panel: #161b22; --border: #30363d; --text: #c9d1d9; --accent: #58a6ff; --green: #2ea043; }
    body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 24px; }
    .container { max-width: 860px; margin: 0 auto; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 28px; }
    h1 { margin-top: 0; color: #fff; display: flex; align-items: center; gap: 10px; font-size: 24px; }
    .tag { font-size: 12px; background: #238636; color: #fff; padding: 3px 8px; border-radius: 12px; font-weight: normal; }
    textarea { width: 100%; height: 110px; background: #0d1117; color: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 14px; box-sizing: border-box; resize: vertical; }
    .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 16px 0; }
    .control-item { display: flex; flex-direction: column; gap: 6px; font-size: 13px; }
    .control-item span { color: #8b949e; }
    input[type=range] { width: 100%; }
    button { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 12px 24px; font-weight: 600; cursor: pointer; font-size: 14px; transition: 0.2s; }
    button:hover { opacity: 0.9; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .output-box { margin-top: 20px; background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-height: 140px; font-family: ui-monospace, Menlo, monospace; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; color: #7ee787; }
    .status-bar { margin-top: 10px; display: flex; justify-content: space-between; font-size: 12px; color: #8b949e; }
  </style>
</head>
<body>
  <div class="container">
    <h1><span>⚡ NexLM Playground</span> <span class="tag">Live Streaming</span></h1>
    <p style="color: #8b949e; margin-top: -6px; font-size: 13px;">Decoder-only GPT Transformer built from scratch in PyTorch.</p>

    <label style="font-size: 13px; font-weight: 600; display: block; margin-bottom: 6px;">Prompt:</label>
    <textarea id="prompt">Once upon a time, Lily found a shiny key</textarea>

    <div class="controls">
      <div class="control-item">
        <span>Temperature: <strong id="tempVal">0.8</strong></span>
        <input type="range" id="temp" min="0.0" max="2.0" step="0.05" value="0.8" oninput="document.getElementById('tempVal').innerText=this.value">
      </div>
      <div class="control-item">
        <span>Top-K: <strong id="topkVal">40</strong></span>
        <input type="range" id="topk" min="1" max="100" step="1" value="40" oninput="document.getElementById('topkVal').innerText=this.value">
      </div>
      <div class="control-item">
        <span>Max New Tokens: <strong id="maxTokensVal">60</strong></span>
        <input type="range" id="maxTokens" min="10" max="200" step="5" value="60" oninput="document.getElementById('maxTokensVal').innerText=this.value">
      </div>
    </div>

    <button id="genBtn" onclick="runGeneration()">Generate Text</button>

    <div class="output-box" id="output">Generated story will stream live here...</div>
    <div class="status-bar">
      <span id="speedMetric">Ready</span>
      <span>NexLM Engine</span>
    </div>
  </div>

  <script>
    async function runGeneration() {
      const prompt = document.getElementById('prompt').value;
      const temperature = parseFloat(document.getElementById('temp').value);
      const top_k = parseInt(document.getElementById('topk').value);
      const max_tokens = parseInt(document.getElementById('maxTokens').value);
      const outputEl = document.getElementById('output');
      const genBtn = document.getElementById('genBtn');
      const speedMetric = document.getElementById('speedMetric');

      genBtn.disabled = true;
      outputEl.innerText = "";
      speedMetric.innerText = "Generating tokens...";

      const t0 = performance.now();
      let tokenCount = 0;

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, temperature, top_k, max_tokens, stream: true })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6);
              if (dataStr === '[DONE]') break;
              try {
                const parsed = JSON.parse(dataStr);
                if (parsed.token) {
                  outputEl.innerText += parsed.token;
                  tokenCount++;
                  const elapsed = (performance.now() - t0) / 1000;
                  speedMetric.innerText = `${tokenCount} tokens | ${(tokenCount / Math.max(0.001, elapsed)).toFixed(1)} tok/s`;
                }
              } catch (e) {}
            }
          }
        }
      } catch (err) {
        outputEl.innerText = "Error: " + err.message;
      } finally {
        genBtn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


class NexLMServerState:
    model: Optional[GPT] = None
    tokenizer: Optional[ByteLevelBPETokenizer] = None
    device: Optional[torch.device] = None


class NexLMRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "device": str(NexLMServerState.device)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions" or self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.end_headers()
                return

            prompt = data.get("prompt", "Once upon a time")
            temperature = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 40))
            top_p = float(data.get("top_p", 0.9))
            max_tokens = int(data.get("max_tokens", 50))
            stream = bool(data.get("stream", False))

            model = NexLMServerState.model
            tokenizer = NexLMServerState.tokenizer
            device = NexLMServerState.device

            prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # Stream token-by-token
                for token_id in generate_stream(
                    model, prompt_ids, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p, use_kv_cache=True
                ):
                    tok_text = tokenizer.decode([token_id])
                    event_payload = json.dumps({"token": tok_text})
                    self.wfile.write(f"data: {event_payload}\n\n".encode("utf-8"))
                    self.wfile.flush()

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                out = generate(
                    model, prompt_ids, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p, use_kv_cache=True
                )
                generated_text = tokenizer.decode(out[0].tolist())
                response_payload = {
                    "prompt": prompt,
                    "text": generated_text,
                    "model": "nexlm",
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def create_demo_server(port: int = 8080) -> HTTPServer:
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    NexLMServerState.device = device

    # Initialize or load model and tokenizer
    tokenizer_path = "data/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    else:
        tokenizer = ByteLevelBPETokenizer(vocab_size=256)
        tokenizer.train("Once upon a time there was a puppy.", verbose=False)

    NexLMServerState.tokenizer = tokenizer

    ckpt_path = "checkpoints/best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        config = GPTConfig(vocab_size=tokenizer.vocab_size, max_seq_len=128, d_model=64, n_heads=2, n_layers=2)
        model = GPT(config).to(device)

    model.eval()
    NexLMServerState.model = model

    server = HTTPServer(("127.0.0.1", port), NexLMRequestHandler)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = create_demo_server(port=args.port)
    print(f"🚀 NexLM Streaming Server live at http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

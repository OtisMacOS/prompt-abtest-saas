from flask import Flask, request, jsonify
import os
import json
import time
import uuid
import glob
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation import evaluate
from supabase import create_client

app = Flask(__name__)

# === 加载环境变量 ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# === 初始化 LLM 和数据库 ===
llm_worker = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm_judge = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 读取所有 prompt 版本 ===
PROMPTS = []
for path in glob.glob("prompts/*.md"):
    version = os.path.basename(path).replace(".md", "")
    with open(path, "r") as f:
        template_str = f.read()
        prompt = PromptTemplate.from_template(template_str)
        prompt.metadata = {
            "id": str(uuid.uuid4()),
            "version": version
        }
        PROMPTS.append(prompt)

@app.route('/run_ab_test', methods=['POST'])
def run_ab_test():
    # 支持传入任务，也可以默认读取本地 tasks.json
    tasks = request.json.get("tasks") if request.is_json and "tasks" in request.json else None
    if not tasks:
        with open("tasks.json", "r") as f:
            tasks = json.load(f)

    results = []
    for task in tasks:
        for prompt in PROMPTS:
            input_vars = {"question": task["question"]}
            try:
                t0 = time.time()
                output = llm_worker.predict(prompt.format(**input_vars))
                latency = round((time.time() - t0) * 1000)
                judge = evaluate(
                    prediction=output,
                    reference=task["ground_truth"],
                    input=input_vars,
                    criteria=["relevance", "correctness"],
                    llm=llm_judge
                )
                score = {
                    k: v.score if hasattr(v, "score") else None
                    for k, v in judge.items()
                }
                # 写入数据库
                supabase.table("prompt_runs").insert({
                    "id": str(uuid.uuid4()),
                    "prompt_id": prompt.metadata["id"],
                    "variant": prompt.metadata["version"],
                    "input": json.dumps(task, ensure_ascii=False),
                    "output": output,
                    "latency_ms": latency,
                    "judge_score": json.dumps(score, ensure_ascii=False),
                    "raw_cost": None
                }).execute()
                results.append({
                    "task_id": task["id"],
                    "variant": prompt.metadata["version"],
                    "output": output,
                    "score": score,
                    "latency_ms": latency
                })
            except Exception as e:
                results.append({
                    "task_id": task["id"],
                    "variant": prompt.metadata["version"],
                    "error": str(e)
                })
    return jsonify(results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

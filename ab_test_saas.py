import streamlit as st
import json
import time
import uuid
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation.qa import QAEvalChain

st.title("Prompt AB 测试 SaaS Demo")

# 自定义背景：竖向6条红紫渐变条纹，无模糊
st.markdown(
    """
    <style>
    .stApp {
        background: repeating-linear-gradient(
            to right,
            #ff4d4f 0%, #ff4d4f 16.66%,
            #a259ec 16.66%, #a259ec 33.33%,
            #ff4d4f 33.33%, #ff4d4f 50%,
            #a259ec 50%, #a259ec 66.66%,
            #ff4d4f 66.66%, #ff4d4f 83.33%,
            #a259ec 83.33%, #a259ec 100%
        );
        min-height: 100vh;
        position: fixed;
        width: 100vw;
        z-index: 0;
    }
    /* 让内容区不受背景影响 */
    .main-content {
        position: relative;
        z-index: 1;
        padding: 2rem 2rem 2rem 2rem;
        background: rgba(255,255,255,0.85);
        border-radius: 1.5rem;
        box-shadow: 0 4px 32px rgba(0,0,0,0.08);
        max-width: 700px;
        margin: 2rem auto;
    }
    /* 输入框和按钮样式 */
    .main-content input, .main-content textarea, .main-content select {
        background: #fff !important;
        color: #111 !important;
        font-size: 1.2rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #ccc !important;
        margin-bottom: 1rem !important;
    }
    .main-content label, .main-content span, .main-content div, .main-content p {
        color: #111 !important;
        font-size: 1.1rem !important;
    }
    .main-content button {
        font-size: 1.1rem !important;
        border-radius: 0.5rem !important;
    }
    </style>
    <div class="main-content">
    """,
    unsafe_allow_html=True
)

# 1. 上传任务文件
tasks = None
uploaded_file = st.file_uploader("上传你的 tasks.json 文件", type=["json"])
if uploaded_file:
    tasks = json.load(uploaded_file)
else:
    st.warning("请先上传 tasks.json 文件")
    st.stop()

# 2. 选择 LLM 模型
llm_model = st.selectbox("选择 LLM 模型", ["gpt-3.5-turbo", "gpt-4o", "gpt-4"]) 
judge_model = st.selectbox("选择评审模型", ["gpt-3.5-turbo", "gpt-4o", "gpt-4"]) 

# 3. 输入 API Token
api_key = st.text_input("输入你的 OpenAI API Key", type="password")
if not api_key:
    st.warning("请输入 API Key")
    st.stop()

# 3.1 输入 Base URL
base_url = st.text_input("输入你的 OpenAI Base URL（可选，留空则用官方默认）")

# 4. 上传 prompt 版本
prompt_files = st.file_uploader("上传你的 prompt 版本（支持多选）", type=["md"], accept_multiple_files=True)
if not prompt_files:
    st.warning("请上传至少一个 prompt 版本")
    st.stop()

# 5. 运行
def run_ab_test(tasks, prompt_files, llm_model, judge_model, api_key, base_url):
    PROMPTS = []
    for file in prompt_files:
        version = file.name.replace(".md", "")
        template_str = file.read().decode("utf-8")
        prompt = PromptTemplate.from_template(template_str)
        PROMPTS.append({
            "prompt": prompt,
            "version": version,
            "id": str(uuid.uuid4())
        })

    llm_worker = ChatOpenAI(model=llm_model, temperature=0.7, openai_api_key=api_key, base_url=base_url if base_url else None)
    llm_judge = ChatOpenAI(model=judge_model, temperature=0, openai_api_key=api_key, base_url=base_url if base_url else None)
    qa_eval_chain = QAEvalChain.from_llm(llm_judge)

    results = []
    for task in tasks:
        for prompt_info in PROMPTS:
            prompt = prompt_info["prompt"]
            version = prompt_info["version"]
            input_vars = {"question": task["question"]}
            try:
                t0 = time.time()
                output = llm_worker.predict(prompt.format(**input_vars))
                latency = round((time.time() - t0) * 1000)
                eval_result = qa_eval_chain.evaluate(
                    [{"query": task["question"], "answer": output, "reference": task["ground_truth"]}]
                )
                score = eval_result[0].get("results", eval_result[0].get("result", ""))
                results.append({
                    "task_id": task["id"],
                    "variant": version,
                    "output": output,
                    "score": score,
                    "latency_ms": latency
                })
            except Exception as e:
                st.error(f"任务 {task['id']} 使用 prompt {version} 时出错: {str(e)}")
                results.append({
                    "task_id": task["id"],
                    "variant": version,
                    "output": "N/A",
                    "score": "N/A", 
                    "latency_ms": "N/A",
                    "error": str(e)
                })
    return results

if st.button("开始 AB 测试"):
    with st.spinner("正在运行 AB 测试，请稍候..."):
        results = run_ab_test(tasks, prompt_files, llm_model, judge_model, api_key, base_url)
        st.success("测试完成！")
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("下载结果为 CSV", data=df.to_csv(index=False), file_name="ab_test_results.csv")

# 末尾加：关闭 main-content div
st.markdown("</div>", unsafe_allow_html=True) 
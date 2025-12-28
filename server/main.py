import sys
import os
# 强制添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import sqlite3
import json
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from core.trainer import train_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 数据库初始化 ---
DB_PATH = "tasks.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 创建表（如果不存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT,
            file_path TEXT,
            args TEXT,
            output_path TEXT,
            error TEXT
        )
    ''')
    
    # 检查是否需要迁移（添加新列）
    cursor.execute("PRAGMA table_info(tasks)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'args' not in columns:
        cursor.execute("ALTER TABLE tasks ADD COLUMN args TEXT")
    if 'output_path' not in columns:
        cursor.execute("ALTER TABLE tasks ADD COLUMN output_path TEXT")
        
    conn.commit()
    conn.close()

init_db()

# --- 系统监控 ---
def get_gpu_status():
    status = {}
    
    # 获取 GPU (NVIDIA)
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            # 默认只看第一张卡
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            status.update({
                "gpu_name": name,
                "gpu_memory_used": mem_info.used // 1024**2,
                "gpu_memory_total": mem_info.total // 1024**2,
                "gpu_util": util_info.gpu,
                "gpu_memory_util": util_info.memory
            })
        else:
            status["gpu_error"] = "No NVIDIA GPU found"
        
        pynvml.nvmlShutdown()
        
    except Exception as e:
        status["gpu_error"] = str(e)
        # 填充默认值防止前端报错
        status.update({
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "gpu_util": 0
        })

    return status

@app.get("/system_status")
async def system_status():
    return get_gpu_status()

# --- 全局推理引擎 (单例模式) ---
class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.current_lora_path = None
    
    def load_base_model(self):
        if self.model is None:
            print("⏳ [Server] 正在加载基座模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, 
                device_map="auto", 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ [Server] 基座模型加载完毕")

    def get_response(self, query: str, lora_path: str = None):
        self.load_base_model() # 确保基座已加载
        
        # 情况 1: 用户想用 LoRA 微调模型
        if lora_path:
            # 如果当前挂载的不是这个 LoRA，或者当前是纯基座，就需要切换
            if self.current_lora_path != lora_path:
                print(f"🔄 [Server] 切换到微调模型: {lora_path}")
                try:
                    # 1. 为了防止显存泄露或冲突，先强制重新加载一遍纯净的基座
                    # (虽然稍微慢点，但绝对稳，不会报错)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name, 
                        device_map="auto", 
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    
                    # 2. 挂载 LoRA
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    self.current_lora_path = lora_path
                    
                except Exception as e:
                    return f"❌ 模型加载失败: {str(e)}"
        
        # 情况 2: 用户想用纯基座模型
        else:
            # 如果当前挂着 LoRA，说明需要卸载
            if self.current_lora_path is not None:
                print("🔄 [Server] 切换回基座模型 (卸载 LoRA)")
                try:
                    # 修正：不要用 unload_adapter，直接重载基座最稳妥
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name, 
                        device_map="auto", 
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    self.current_lora_path = None
                except Exception as e:
                    return f"❌ 切换基座失败: {str(e)}"

        # --- 开始生成 ---
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

# 初始化引擎
engine = InferenceEngine()

# --- API 定义 ---
class TrainRequest(BaseModel):
    file_id: str
    args: dict = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    # 确保保存到项目根目录的 data 文件夹
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{file_id}.json")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "file_id": file_id}

@app.post("/train")
async def start_training(req: TrainRequest,
                         background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    # 重新构建 data 路径
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    data_path = os.path.join(data_dir, f"{req.file_id}.json")
    
    # 预计算 output_path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "output", task_id)

    # 写入数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    args_json = json.dumps(req.args) if req.args else "{}"
    cursor.execute(
        "INSERT INTO tasks (task_id, status, file_path, args, output_path) VALUES (?, ?, ?, ?, ?)", 
        (task_id, "running", data_path, args_json, output_path)
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(run_training_background, task_id, data_path, req.args)
    return {"task_id": task_id, "status": "started"}

def run_training_background(task_id, data_path, user_args):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        train_model(task_id, data_path, user_args)
        cursor.execute("UPDATE tasks SET status = ? WHERE task_id = ?", ("success", task_id))
    except Exception as e:
        cursor.execute("UPDATE tasks SET status = ?, error = ? WHERE task_id = ?", ("failed", str(e), task_id))
    finally:
        conn.commit()
        conn.close()

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT status, error, file_path, args, output_path FROM tasks WHERE task_id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        status, error, file_path, args, output_path = row
        response = {
            "task_id": task_id,
            "status": status,
            "file_path": file_path,
            "args": json.loads(args) if args else None,
            "output_path": output_path
        }
        if error:
            response["error"] = error
        return response
    else:
        return {"status": "not_found"}

@app.get("/tasks")
async def list_tasks():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT task_id, status, file_path, args, output_path, error FROM tasks")
    rows = cursor.fetchall()
    conn.close()
    
    tasks_list = []
    for row in rows:
        task_id, status, file_path, args, output_path, error = row
        tasks_list.append({
            "task_id": task_id,
            "status": status,
            "file_path": file_path,
            "args": json.loads(args) if args else None,
            "output_path": output_path,
            "error": error
        })
    return tasks_list

# 新增：聊天接口
class ChatRequest(BaseModel):
    query: str
    task_id: str = None   # 如果为空，就是基座；如果有值，就是微调
    use_lora: bool = False

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    lora_path = None
    if req.use_lora and req.task_id:
        # 构建 LoRA 的绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lora_path = os.path.join(project_root, "output", req.task_id)
        if not os.path.exists(lora_path):
            return {"response": "❌ 错误：找不到该任务的训练结果，请检查Task ID"}
    
    response = engine.get_response(req.query, lora_path)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
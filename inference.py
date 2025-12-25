import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ---------------- 配置区域 ----------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_ROOT = "./output"
# ----------------------------------------

def get_latest_lora_path():
    """自动寻找 output 文件夹里最新的那个模型"""
    if not os.path.exists(OUTPUT_ROOT):
        return None
    
    # 获取所有子文件夹
    all_subdirs = [os.path.join(OUTPUT_ROOT, d) for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]
    
    if not all_subdirs:
        return None
    
    # 按修改时间排序，找最新的
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir

def chat_with_model():
    print("🔍 正在自动寻找最新的训练模型...")
    lora_path = get_latest_lora_path()
    
    if lora_path:
        print(f"✅ 找到了最新模型路径: {lora_path}")
    else:
        print("❌ 在 output 文件夹里没找到任何模型！请先去网页上训练一个任务。")
        return

    print("⏳ 正在加载基座模型 (Qwen2.5-0.5B)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        # 自动检测显卡
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载基座
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
    except Exception as e:
        print(f"❌ 基座模型加载失败，请检查网络。报错: {e}")
        return
    
    print(f"⏳ 正在挂载 LoRA 补丁...")
    try:
        model = PeftModel.from_pretrained(model, lora_path)
        print("🎉 模型加载成功！")
    except Exception as e:
        print(f"❌ LoRA加载失败。请检查 {lora_path} 下是否有 adapter_config.json。报错: {e}")
        return

    print("\n" + "="*30)
    print("🤖 EasyTune 对话终端 (输入 quit 退出)")
    print("="*30)
    
    # 简单的对话历史，让它能记住上下文
    history = [] 

    while True:
        query = input("\n👤 用户: ")
        if query.strip().lower() == "quit":
            break
            
        # 构建 Prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        # 简单的多轮对话拼接（演示用）
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        
        messages.append({"role": "user", "content": query})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=200, # 回复最大长度
                temperature=0.7     # 控制创造性
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"🤖 EasyTune: {response}")
        
        # 记录历史
        history.append((query, response))
        if len(history) > 3: history.pop(0) # 只记最近3轮

if __name__ == "__main__":
    chat_with_model()
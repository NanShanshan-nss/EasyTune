import sys
print("Step 1: 脚本开始运行...")

try:
    from fastapi import FastAPI
    import uvicorn
    print("Step 2: Web库导入成功")
except Exception as e:
    print(f"导入报错: {e}")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    print("Step 3: 准备启动服务器...")
    try:
        # 强制使用 localhost，避免 0.0.0.0 权限问题
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"启动报错: {e}")
# 使用轻量级 Python 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖（XGBoost 和 Pandas 需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录（建议挂载为 Volume）
RUN mkdir -p data/logs data/models data/reports

# 暴露 Streamlit 端口
EXPOSE 8501

# 设置环境变量
ENV RUNTIME_MODE=paper
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# 默认启动命令（启动后台引擎，用户可以通过 docker exec 运行 main.py）
# 或者启动 Dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

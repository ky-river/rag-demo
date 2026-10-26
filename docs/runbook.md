# 本地运行与演示手册

这份手册用于每次重启电脑后的启动、功能验收、演示、关闭和排错。

## 服务地址

| 服务 | 地址 |
| --- | --- |
| 网页前端 | http://127.0.0.1:5501 |
| 后端 API | http://127.0.0.1:8000 |
| 健康检查 | http://127.0.0.1:8000/health |

## 每次启动

先打开 Docker Desktop，等待状态显示为 `Engine running`。

在 VS Code 中打开项目目录，进入 `终端 -> 新建终端`，执行：

```bash
cd ~/Desktop/rag-demo
docker compose up -d --build
docker compose ps
curl -sS http://127.0.0.1:8000/health
open http://127.0.0.1:5501
```

正常运行时应满足：

- `docker compose ps` 中 `backend` 和 `frontend` 均显示 `healthy`。
- `/health` 返回 `status` 为 `ok`。
- `/health` 返回 `index_ready` 为 `true`。
- 网页右上角显示“服务与索引正常”。

如果服务已经启动，只需要执行：

```bash
docker compose up -d
open http://127.0.0.1:5501
```

## 功能验收

### 1. 知识库问答

进入“知识库问答”，输入：

```text
这个项目采用了哪些检索和重排技术？
```

预期结果：

- 返回技术栈、混合检索和可选 Cross-Encoder 重排说明。
- 显示引用来源、文件名和页码。
- 返回约 4 个来源片段。

继续输入：

```text
公司今年的营业收入是多少？
```

预期返回：

```text
当前资料中没有找到答案
```

同时不应返回无关来源。

### 2. 求职 Agent

进入“求职 Agent”，输入：

```text
请制定一个 RAG 学习路线
```

预期结果：

- Agent 路由到 `study_plan`。
- 返回四周学习计划。
- 页面显示“已完成”或对应的 Agent 路由标签。

### 3. 文档上传

进入“文档管理”：

1. 选择一份可读取文字的 PDF。
2. 点击“上传并建立索引”。
3. 等待上传完成并自动刷新索引状态。

预期结果：

- 页面提示上传和索引重建成功。
- 左侧文档数量和片段数量更新。
- 知识库问答能够检索到新文档内容。

限制：

- 仅支持 PDF。
- 单个文件不能超过 10 MB。
- 扫描件需要先进行 OCR。

### 4. 简历与岗位匹配

进入“简历匹配”，上传：

- 简历：`data/AI实习生_个人简历_测试.pdf`
- 岗位 JD：一份内容明确的 Agent/RAG 实习岗位 PDF

点击“开始匹配”，预期结果：

- 显示综合匹配度。
- 显示匹配技能和技能缺口。
- 显示简历修改建议。
- 显示预测面试题。

随后点击：

1. “生成 4 周学习计划”
2. “下载匹配报告”

预期生成完整的四周计划，并下载 `简历岗位匹配报告.txt`。

演示时只使用合成简历和公开或自拟的岗位 JD，不要上传真实个人敏感信息。

## 推荐演示顺序

完整演示控制在 2 到 3 分钟：

1. 展示网页首页和服务状态。
2. 提出一个知识库问题，展示答案、文件名和页码。
3. 提出一个无依据问题，展示拒答和不返回无关来源。
4. 运行一次 Agent 学习计划任务，展示路由结果。
5. 上传简历与 JD，展示匹配度、技能缺口和学习计划。
6. 展示 `/health`、两个 `healthy` 容器和 GitHub Actions。
7. 简要说明 Chroma、BM25、RRF 和可选重排的流程。

详细口播脚本见 `docs/demo-script.md`。

## 停止与再次启动

临时停止：

```bash
docker compose stop
```

再次启动：

```bash
docker compose start
```

停止并删除当前容器：

```bash
docker compose down
```

`docker compose down` 不会删除本项目通过 bind mount 使用的 `data`、
`chroma_db` 和 `sessions.sqlite3`。

## 常见问题

### Docker 未运行

如果提示 `Cannot connect to the Docker daemon`，先打开 Docker Desktop：

```bash
open -a Docker
```

也可以在“应用程序”中手动打开 Docker，等待引擎启动后再执行
`docker compose up -d`。

### 8000 或 5501 端口被占用

检查占用进程：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5501 -sTCP:LISTEN
```

不要直接结束不认识的进程。确认进程属于旧项目后再处理，也可以修改
`docker-compose.yml` 中的宿主机端口。

### 容器不是 healthy

查看后端日志：

```bash
cd ~/Desktop/rag-demo
docker compose logs --tail=100 backend
```

查看全部服务日志：

```bash
docker compose logs --tail=100
```

### 网页状态没有刷新

先刷新浏览器。静态页面仍为旧版本时，在 macOS 浏览器中执行强制刷新：

```text
Command + Shift + R
```

### 模型请求较慢

简历匹配和 Agent 任务需要多次模型调用，通常等待 20 到 60 秒属于正常范围。
如果持续超时，检查 `.env` 中的模型配置、网络连接和后端日志。不要把 `.env`
或 API Key 粘贴到公开仓库、截图或演示视频中。

### 修改代码后页面没有变化

前端通过 `./frontend` 只读挂载到容器，修改 HTML、CSS 或 JavaScript 后刷新浏览器
即可。后端代码或依赖发生变化时重新构建：

```bash
docker compose up -d --build
```

## 提交前检查

```bash
cd ~/Desktop/rag-demo
source .venv/bin/activate
python -m unittest discover -s tests -v
git diff --check
git status --short
```

确认没有 `.env`、密钥、用户 PDF、会话数据库或其他生成文件被加入版本控制。

## 录制前检查

- 使用 1280x720 或更高的浏览器窗口。
- 关闭与演示无关的标签页和通知。
- 不展示 API Key、环境变量、真实简历或私人岗位文件。
- 提前确认两个容器为 `healthy`。
- 提前准备好测试简历和测试 JD。
- 把演示视频控制在 3 分钟以内。

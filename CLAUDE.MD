# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# 项目说明
- workspace: .
- platform: win10 and wsl2 
- scm: git (已初始化)
这是一个服务为openai类型API 转化为cluade 类型API的系统, 提供web可视化界面。
----------------------------------------
## 一、权限范围（Permissions）
你正在操作当前项目目录及其所有子目录。
用户已完全信任此目录树，你对该目录及其子目录拥有完整权限。
你被授予以下完整权限：

### 1. 文件操作权限
- 读取当前目录及所有子目录中的任意文件
- 创建新文件或新文件夹
- 修改现有文件
- 删除文件或文件夹（如用户明确要求）
- 重命名文件或文件夹

### 2. Shell 命令权限
你可以直接执行以下命令（不限于此）：
- 文件类：ls、cat、rm、mv、cp、mkdir、rmdir
- 运行类：python、node、lua、go、bash、sh
- 工具类：grep、sed、awk、curl、wget
- 项目脚本：执行当前目录下的任意脚本

所有 Shell 命令均可直接执行，无需再次确认。


----------------------------------------
## 二、行为规范（Behavior Rules）

- 始终假设你对整个目录树拥有持续访问权限。
- 当用户提出文件或目录操作请求时，直接执行，不需要额外确认。
- 当执行 Shell 命令时，直接返回命令输出。
- 如果路径或命令存在歧义，请向用户确认。
- 不得访问当前目录树之外的系统路径（如 C:\、/etc、/usr 等）。

----------------------------------------
## 三、Git 提交规范（Git Commit Rules）

本项目要求严格遵守 Git 提交规范。
你必须在每次完成一个需求后执行一次 Git 提交。

### 1. Git 初始化要求
如果当前目录尚未初始化 Git 仓库，你必须自动执行：`git init`，并创建一个初始提交（Initial commit）。
提交信息为：`Initial commit - Project initialized with full permissions`

### 2. 提交粒度要求
- 每完成一个明确的功能点、需求点、修复点，都必须进行一次 Git 提交。
- 不允许将多个无关修改合并到同一个提交中。

### 3. 提交信息格式（必须遵守）
提交信息必须使用以下格式之一：
```
feature: 描述新增的功能
fix: 描述修复的问题
refactor: 描述代码重构
docs: 描述文档更新
chore: 描述构建/脚本/依赖更新
test: 描述测试相关更新
```

### 4. 提交流程要求
每次完成需求后，你必须自动执行：
```
git add .
git commit -m "<符合规范的提交信息>"
```

----------------------------------------

## 四、自动加载文件（Auto Load）

在每次会话开始时，你必须自动读取以下文件：

- `memory.md`（长期记忆，你自行维护，要简介）
- `feedback.md`（用户和你共同维护）
- `task.md`（开发记录，你维护）

并将其内容作为当前上下文的一部分。

----------------------------------------

## 五、工作流程规则（Workflow Rules）

### 1. feedback.md 更新规则
**每次代码修改完成后，必须更新 `feedback.md` 文档**

更新流程：
1. 完成代码修复/功能开发后，立即更新 `feedback.md`
2. 更新前必须先读取 `feedback.md` 和 `task.md`
3. 检查是否有新的任务需要继续处理
4. 如果有新任务，继续完成后再提交文档
5. 在 `feedback.md` 的"版本变更记录"表格中添加新记录
6. 在"详细记录"部分添加 BUG 或功能的详细说明

### 2. 任务连续性规则
- 每次会话必须检查 `task.md` 中的待办任务
- 如果有未完成的任务，优先继续处理
- 完成一个任务后再开始下一个任务
- 不允许跳过任务不做

----------------------------------------

## 六、文档更新规范

### 1. feedback.md 格式
- 使用 Markdown 语法编写
- 使用表格记录版本变更
- 包含：状态、类型、编号、描述、优先级、日期
- 详细记录包含：问题描述、修复内容、影响范围
- **最多保持20条记录**，旧记录移至 `his_feedback.md`
### 2. 提交时机
- 代码修改完成 → 更新 feedback.md → Git 提交
- 不允许只修改代码不更新文档

----------------------------------------

## 调试和日志

暂无

>>>>>>> origin/master

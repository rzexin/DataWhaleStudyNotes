# Agent 开发三大 Harness 框架对比分析报告

**调研对象：** DeepAgents（LangChain）、DSH / DeepSeek Harness（DeepSeek）、PiAgent / Pi Coding Agent（Earendil Works）
**报告定位：** 面向工程团队的技术选型与架构决策参考
**核心命题：** 在"模型趋于商品化、Harness 成为产品"的时代，三大代表性 Harness 的架构取舍与能力边界

---

## 目录

1. 背景：为什么是 Harness
2. 框架概览与基本信息
3. 架构设计对比
4. 核心能力矩阵
5. 关键差异深度解析
6. 性能、成本与实测数据
7. 安全模型与沙箱
8. 生态、许可与治理
9. 选型决策指南
10. 风险、局限与趋势判断
11. 参考资料

---

## 1. 背景：为什么是 Harness

### 1.1 概念定义

**Agent Harness（智能体驾驭层）** 是包裹在语言模型之外、把模型的"推理能力"转化为"可持续执行行动"的全部工程组件的统称。如果用一句话概括：

> **Agent = Model（智能）+ Harness（执行）**

Harness 通常涵盖以下九类组件，本报告的对比即围绕它们展开：

| # | 组件 | 解决的问题 |
|---|------|-----------|
| 1 | 工具调用循环（Tool Calling Loop） | 模型 ↔ 工具的多轮往返执行 |
| 2 | 上下文工程（Context Engineering） | 长会话下的消息压缩、截断、缓存 |
| 3 | 任务规划（Planning） | 把目标拆解为可跟踪的子任务 |
| 4 | 文件系统 / 沙箱（Filesystem / Sandbox） | 安全地读写文件、执行代码 |
| 5 | 子代理调度（Subagent Delegation） | 把工作隔离到独立上下文并行执行 |
| 6 | 持久化内存（Persistent Memory） | 跨会话的状态与记忆 |
| 7 | 技能加载（Skills） | 按需注入领域能力（渐进式披露） |
| 8 | 可观测性（Observability） | trace / replay / evaluation |
| 9 | 人机协同（Human-in-the-Loop） | 敏感操作前暂停等待批准 |

### 1.2 时代背景

2024–2026 年间，行业出现了一次清晰的认知迁移：

- **推理模型能力趋同**：当主流 frontier 模型的通用推理能力差距缩小后，差异化从"哪个模型更聪明"转移到"哪个 Harness 更可靠"。
- **同一模型、不同 Harness、结果不同**：Composio 的基准测试显示，同一个 Kimi K3 模型在 8 个不同 Harness 上的观察通过率从 **68% 到 88%**，差距高达 **20 个百分点**。这直接证明了 Harness 层对最终效果的一阶影响。
- **Harness 成为新的竞争焦点**：DeepAgent 的爆发（Claude Code 引领）让整个行业意识到——长时任务的成功率，很大程度上由 Harness 的工程质量决定。

本文选取的三者恰好代表了 Harness 设计的**三种极端哲学**，构成了一个完整的光谱。

---

## 2. 框架概览与基本信息

| 维度 | **DeepAgents** | **DSH（DeepSeek Harness）** | **PiAgent（Pi）** |
|------|---------------|---------------------------|-------------------|
| 出品方 | LangChain | DeepSeek AI | Earendil Works（Mario Zechner，独立开发者） |
| 首次发布 | 2025 年（开源），后续推出 Managed 版本 | 2026 年（v0.1 Developer Preview） | 2025 年（开源，持续迭代） |
| 定位 | Batteries-included 的工业级框架 | "Everything is a Plugin" 的可组合运行时 | Minimal、可自生长的极简主义 Harness |
| 核心口号 | "The batteries-included agent harness" | "Everything is a Plugin" | "Primitives, not features" |
| 语言 / 栈 | Python（主）、TypeScript | Node.js / TypeScript | TypeScript / Node.js |
| 许可证 | MIT（Harness）+ LangSmith 商业托管 | MIT | MIT |
| 底层运行时 | LangGraph | Cordis 元框架 | 自研 + Chord 运行时 |
| 部署形态 | 本地 / LangSmith / 任意 LangGraph 运行时 | 本地 CLI / Web Workbench | CLI / RPC / SDK 嵌入式 |
| 模型支持 | 任何支持 tool calling 的模型 | 任何模型（模型本身即插件） | 15+ providers，支持中途切换 |
| 开源程度 | 完全开源 | 完全开源（Dev Preview） | 完全开源 |
| 上手复杂度 | 中 | 中高 | 低（默认）／高（深度定制） |

### 2.1 三种哲学的一句话概括

- **DeepAgents**：**"工具齐全，托管友好"**——把长时运行 Agent 需要的一切默认装好，让开发者开箱即用。
- **DSH**：**"万物皆插件"**——把 Harness 的每个零件（包括 Agent Loop 本身）都做成可替换的插件。
- **PiAgent**：**"给原语，不给功能"**——默认只给 4 个工具和极简系统提示，其余交给用户按需组装。

---

## 3. 架构设计对比

### 3.1 DeepAgents：基于 LangGraph 的中间件栈

DeepAgents 是一个构建在 LangChain / LangGraph 之上的 **"有主张"（opinionated）** Harness。所谓有主张，是指它默认已经把所有"长时运行 Agent"所需的能力都装好了。

```
┌──────────────────────────────────────────────┐
│  LangGraph Runtime（图运行时）                 │
│  持久化 / 流式 / checkpoint / HITL            │
├──────────────────────────────────────────────┤
│  create_agent（LangChain 最小 Harness）       │
│  基础 tool calling loop                      │
├──────────────────────────────────────────────┤
│  DeepAgents Harness（中间件栈）                │
│  • TodoListMiddleware（任务规划）             │
│  • FilesystemMiddleware（虚拟文件系统）        │
│  • SubAgentMiddleware（子代理委托）           │
│  • SummarizationMiddleware（上下文压缩）      │
│  • MemoryMiddleware（持久化记忆）             │
│  • SkillsMiddleware（技能加载）               │
│  • AnthropicPromptCachingMiddleware（缓存）   │
├──────────────────────────────────────────────┤
│  LangSmith（托管运行时 / 可观测性）             │
│  durable threads / memory / sandboxes /       │
│  channels / schedules / evals / traces        │
└──────────────────────────────────────────────┘
```

**核心设计原则：**

- **Opinionated**：默认内置长时、多步场景的最佳实践。
- **Extensible**：每个中间件都可覆写（`excluded_middleware` / `extra_middleware` / `HarnessProfile`），无须 fork。
- **Model-agnostic**：对接任何支持 tool calling 的模型。
- **Production-ready**：基于 LangGraph 的流式、持久化、checkpointing，配合 LangSmith 提供 tracing / eval / deployment。

**关键中间件一览：**

| Middleware | 功能 |
|-----------|------|
| `TodoListMiddleware` | 提供 `write_todos` / `read_todos` 工具管理任务列表 |
| `FilesystemMiddleware` | 文件操作工具，自动把超大工具输出落盘 |
| `SubAgentMiddleware` | 提供 `task` 工具，把工作委托给子代理 |
| `SummarizationMiddleware` | 上下文超阈值（约 170k tokens）自动摘要 |
| `AnthropicPromptCachingMiddleware` | 为 Anthropic 模型启用 prompt 缓存 |

**存储后端可插拔：** `StateBackend`（临时）、`FilesystemBackend`（本地磁盘）、`StoreBackend`（持久化）、`CompositeBackend`（混合路由）。

### 3.2 DSH：基于 Cordis 的"万物皆插件"

DSH 最大胆之处在于：**连 Agent Loop 本身都被实现为一个插件**。在绝大多数框架里 Agent Loop 是"承重墙"，不可替换；而在 DSH 中，它只是众多插件中的一个。

**Cordis 元框架的五大原语：**

1. **Service**：插件间互相发现并调用的对象（DI 容器）
2. **Plugin**：实现 Capability 的单元
3. **Context**：单次运行的上下文（ctx）
4. **Hook**：在事件上注入的拦截点
5. **Event**：append-only 的事件流，构成 Trajectory 视图

```
┌────────────────────────────────────────────────┐
│           Cordis Kernel（依赖注入 + 挂载）        │
└───────────────────────┬────────────────────────┘
        ┌───────────────┼───────────────┬─────────────────┐
   ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐   ┌───────▼─────┐
   │ Model   │    │ Tools     │   │ Sessions  │   │ Sandbox     │
   │ Adapter │    │ Registry  │   │(Event Log)│   │(Modal/      │
   │(plugin) │    │ (plugin)  │   │ (plugin)  │   │ Daytona/..) │
   └─────────┘    └───────────┘   └───────────┘   └─────────────┘
   ┌─────────┐    ┌───────────┐   ┌─────────────┐
   │ Agent   │    │ Subagents │   │ Web         │
   │ Loop    │    │ (6 种     │   │ Workbench   │
   │(plugin!)│    │  provider)│   │ (plugin)    │
   └─────────┘    └───────────┘   └─────────────┘
```

**插件化范围（几乎覆盖一切）：**

- 模型适配器（DeepSeek、Anthropic、OpenAI、本地…）
- 工具注册表（Shell、Web、Edit、Browser…）
- Session Log（append-only 事件流）
- 沙箱后端（Modal、Daytona、local…）
- **Agent Loop 本身**（Standard、PTC、Minimal、Creator 四种）
- Subagent 提供器（**6 种**：in-process、fork、ACP、Codex、Claude Code…）

**DSH 最突出的差异化能力——跨产品编排：** 同一份工具消费者代码（`tool-subagent`）可以同时调用进程内子代理、Claude Code、Codex、ACP 协议实现。**把代理委托给商业 Agent 产品，从"代码重写"变成"配置选择"。**

### 3.3 PiAgent：极简主义 + 可自生长

Pi 的设计哲学与前两者截然不同：**"Primitives, not Features"（提供原语，而非功能）**。

**四层包结构：**

```
┌─────────────────────────────────────────┐
│  pi （CLI 入口 + TUI）                    │
├─────────────────────────────────────────┤
│  coding-agent （Agent + Tool Loop）      │
├─────────────────────────────────────────┤
│  pi-ai （多 provider 统一 LLM API）       │
├─────────────────────────────────────────┤
│  chord （插件运行时 + RPC + 持久化）       │
└─────────────────────────────────────────┘
```

**Pi 默认只装 4 个工具**（`read`、`write`、`edit`、`bash`），系统提示与工具定义加起来 **不到 1000 tokens**。子代理、计划模式、权限弹窗、MCP、后台 bash、待办列表——其他框架默认装好的功能——Pi **一概不装**，由用户按需扩展。

**唯一"内建"的核心能力是：自生长（Self-modification）。** Pi 可以用 TypeScript 扩展改写自己的 Harness（如注入 `pre-tool-use` hook 写审计日志），运行 `/reload` 后立即生效——这是 Claude Code / Codex 等产品做不到的（它们只能改 `settings.json` 这类预设配置）。

---

## 4. 核心能力矩阵

下表对 Agent Harness 的关键能力逐一比对：
**✓** = 完整内置支持　　**○** = 部分支持 / 需扩展　　**—** = 不支持 / 需完全自建

| 能力维度 | DeepAgents | DSH | Pi |
|---------|:---------:|:---:|:--:|
| 工具调用循环 | ✓（LangGraph 原生） | ✓（Agent Loop 可替换） | ✓（内置最小循环） |
| 任务规划（write_todos） | ✓ TodoListMiddleware | ○（插件实现） | —（需扩展/装包） |
| 虚拟文件系统 | ✓ FilesystemMiddleware（多 backend） | ✓（fs 后端插件） | ○（4 个文件工具） |
| 代码执行沙箱 | ✓（Modal/Daytona/Deno/本地） | ✓（可配置沙箱后端） | —（默认无沙箱，用 bash） |
| 子代理（Subagent） | ✓ SubAgentMiddleware（task 工具） | ✓（**6 种提供器**，含 Claude Code/Codex 桥接） | —（默认无，有包机制） |
| 上下文压缩 | ✓ SummarizationMiddleware（~170k 阈值） | ✓（ctx.compaction 作为插件） | ○（compaction，需扩展） |
| 持久化记忆 | ✓ MemoryMiddleware | ✓（Session Log + Store 插件） | ○（靠文件系统扩展） |
| 技能加载（Skills） | ✓ SkillsMiddleware | ✓（Skills 作为插件） | ✓（按需加载 Markdown skills） |
| Human-in-the-Loop | ✓（LangGraph 原生） | ✓（可扩展 hook / 权限预设） | —（无内置，需扩展） |
| 可观测性 / Trace | ✓（LangSmith 一等公民） | ✓（Trajectory + append-only event log） | ○（JSON 事件流，无内建 UI） |
| Session 恢复 / 分支 | ✓（LangGraph checkpoint） | ✓（resume/fork/search/replay） | ✓（session/export/branch，树状） |
| 多模型 Provider | ✓（任何支持 tool calling） | ✓（模型是插件） | ✓（15+ providers，可中途切换） |
| MCP 支持 | ✓（LangChain 适配器） | ✓（插件） | ○（`mcp.json` 配置 / 扩展） |
| Prompt 缓存 | ✓ AnthropicPromptCachingMiddleware | ○（插件） | —（无显式缓存） |
| SDK 嵌入 | ○（LangGraph 嵌入） | ✓（TS SDK，含 PTC Code Mode） | ✓（原生 TS SDK，in-process） |
| 多语言 RPC | —（Python 主） | ○（基于 Node） | ✓（JSON-RPC over stdio） |
| 自修改（Self-modification） | — | ○（Creator 模式可改运行时） | ✓（让 Pi 自己写扩展 + reload） |
| 端到端托管 | ✓ Managed Deep Agents（LangSmith） | ○（可自部署 / Web Workbench） | —（需自行部署） |
| 生态成熟度 | ✓（LangChain/LangSmith/LangGraph） | ○（快速成长的 dsh-plugin 生态） | ○（npm 包生态） |
| 生产稳定性 | ✓（1.x 稳定） | —（v0.1 Dev Preview，API 可能破坏性变更） | ○（迭代快，个人维护） |

---

## 5. 关键差异深度解析

### 5.1 扩展性哲学：Opinionated vs. Composable vs. Primitive

这是三者最根本的分野。

| 框架 | 哲学 | 默认装什么 | 覆写粒度 |
|------|------|-----------|---------|
| **DeepAgents** | Opinionated | 全套长时运行能力 | 中间件级 |
| **DSH** | Composable | 完整运行时 + 159 个插件 | **插件级（含 Agent Loop）** |
| **Pi** | Primitive | 4 个工具 + <1000 token 提示 | **代码级（可改自身源码）** |

**DeepAgents** 走"**默认调好**"路线：`create_deep_agent()` 一行就返回一个装好了 planning、filesystem、subagent、summarization、prompt caching 的编译图。所有中间件都可通过 `excluded_middleware` / `extra_middleware` 显式覆写，`HarnessProfile` 甚至允许按模型（如 `openai:gpt-5.5`）注册不同的中间件栈。

**DSH** 走"**极致可组合**"路线：不仅中间件可替换，连 Agent Loop 本身也是插件。其配置层用 `cordis.patch.yml` 做声明式覆盖，例如：

```yaml
- id: subagent-claude-code
  config:
    permissionMode: plan
```

**Pi** 走"**原语**"路线：默认功能近乎白板，但配套完整的四层扩展体系——Extensions（TypeScript 模块）、Skills（按需加载的能力包）、Prompt Templates、Themes。可通过 `pi install npm:package` 像安装 npm 包一样安装社区能力。

> **启示：** 扩展性越强 → 灵活性越高，但学习曲线与决策成本也越高。这正是三者的"自由度/负担"权衡曲线。

### 5.2 子代理（Subagent）模型

**DeepAgents**：内置 `task` 工具，主 Agent 可创建临时子 Agent，提供三项保证：
- **Fresh context**：每次调用都是全新实例，上下文互不污染
- **Autonomous execution**：子代理独立运行至完成
- **Single handoff**：只向主 Agent 返回一份最终报告

**DSH**：subagent 作为插件，**共 6 种提供器**：
1. `subagent-claude-code`：委托给 Anthropic Claude Code
2. `subagent-codex`：委托给 OpenAI Codex
3. `subagent-acp`：通过 Agent Client Protocol 与任何 ACP 实现互通
4. `subagent-in-process`：进程内子代理
5. `subagent-fork`：进程内 fork
6. （其余提供器）

> **这是 DSH 最大的亮点：跨产品编排。** 例如外层用便宜的开源模型驱动，把复杂子任务委托给 Claude Code / Codex，并在同一个 session log 中记录完整轨迹。这种"Harness 编排 Harness"的能力在其他框架中罕见。

**Pi**：**默认没有子代理**。用户可自行扩展，或安装第三方包实现——"没有任何内置，就没有任何阻挡"。

### 5.3 上下文工程（Context Engineering）

| 框架 | 策略 | 具体机制 |
|------|------|---------|
| **DeepAgents** | 主动压缩 | `SummarizationMiddleware`（>170k 自动摘要）+ 超大输出落盘 + Anthropic prompt caching |
| **DSH** | 干净接缝 | `ctx.compaction` 作为插件可替换；**模型不调用 compaction**，是清晰的 seam；session log 支持 search/replay |
| **Pi** | 主动做减法 | 极小系统提示 + 4 个工具；`AGENTS.md` 项目级注入；`SYSTEM.md`/`APPEND_SYSTEM.md` 替换或追加提示；Skills 渐进式披露避免打破 prompt 缓存 |

**对比启示：**
- DeepAgents 用"工程化压缩"对抗上下文膨胀。
- DSH 把压缩抽象为可替换的"接缝"，强调机制透明。
- Pi 用"源头减量"——提示越短、工具越少，上下文膨胀越慢。

### 5.4 运行模式（Runtime Modes）

**DSH 的四种 Agent 预设：**

| 模式 | 工具集 | 定位 |
|------|-------|------|
| **Standard** | 完整日常 Agent 环境 | 干活（执行） |
| **PTC**（Programmatic Tool Calling） | 同 Standard，但通过 Code Mode SDK 暴露 | 编排（模型写一段 TS 程序组合多步操作，减少往返） |
| **Minimal** | 仅 2 个工具（持久 bash + `str_replace_editor`） | 评测（作为模型基线） |
| **Creator** | Standard + 运行时自省能力 | 改 Harness（自组合） |

一句话记忆：**Standard 干活、PTC 编排、Minimal 度量、Creator 重塑 Harness。**

**Pi 的四种运行模式：**

| 模式 | 命令 / API | 用途 |
|------|-----------|------|
| Interactive | `pi` | 完整终端 TUI，日常编码 |
| Print / JSON | `pi -p "query"`、`--mode json` | 脚本、CI、事件流 |
| RPC | stdin/stdout 的 JSON 协议 | 非 Node 集成（Go、Python、IDE） |
| SDK | `@earendil-works/pi-coding-agent` | 嵌入自家应用（如 OpenClaw） |

### 5.5 会话持久化与分支

| 框架 | 机制 |
|------|------|
| **DeepAgents** | LangGraph checkpoint（durable threads） |
| **DSH** | Session append-only event log，支持 resume / fork / search / replay，Trajectory 视图 |
| **Pi** | **树状 Session**：可从任一早先节点分支，无需复制扁平 transcript；支持 export / branch |

Pi 的树状会话是独特的：类似 Git 的分支模型，允许"回退到某个点再试另一条路"，而不必复制整条对话历史。

### 5.6 可观测性 / 调试

| 框架 | 能力 | 特点 |
|------|------|------|
| **DeepAgents** | LangSmith：trace / debug / evaluate，一键部署 | 深度集成，但为付费托管 |
| **DSH** | Trajectory 视图 + append-only 事件流，按插件来源过滤 | "不是看清理过的聊天渲染，而是看原始上下文"（DeepInfra 评测） |
| **Pi** | `--mode json` 事件流、session 导出 | 轻量，事件流可被任意外部工具消费 |

---

## 6. 性能、成本与实测数据

> 以下数据来自第三方基准（Composio），**注意其对比条件并非完全对齐**，应作为方向性信号而非绝对结论。

### 6.1 同一模型（DeepSeek V4 Flash）跨 Harness 对比

| Harness | 任务通过率 | 每成功任务成本 | 中位完成时间 |
|---------|-----------|--------------|-------------|
| **Pi Agent** | 20/30（**66.7%**，最高） | **$0.028**（最低） | 132.2 s |
| DeepSeek Harness | 20/30 | $0.028（共享成功任务口径） | 252.1 s |
| Hermes Agent | — | $0.056 | — |
| DeepAgents | 16/30（53.3%） | $0.045 | — |
| OpenCode | — | $0.073 | 较快 |
| Codex | 17/30 | $0.081 | — |
| Claude Code | 16/30（53.3%） | $0.195（最高） | 181.8 s（最快） |

**关键观察：**

1. **Pi 通过率最高（66.7%），成本最低（$0.028），且 token 用量与工具调用次数最少。**
2. **DeepAgents 与 Claude Code 通过率相同（53.3%），但 Claude Code 的每成功任务成本是 DeepAgents 的 4 倍多。**
3. **DeepSeek 的"第一方 Harness 优势"并不明显**——Pi 反而多完成了一个任务。这有力印证了"Harness 质量 > 厂商绑定"。

### 6.2 另一组基准（Kimi K3，8 个 Harness，25 任务）

| Harness | 通过率 | 备注 |
|---------|-------|------|
| Oh My Pi | 22/25（88%，最高） | 质量与成本平衡最佳（$0.52/成功） |
| Kimi Code | 21/25（84%） | — |
| Hermes Agent | 20/25 | 总成本最低（$9.28）、每成功成本最低（$0.46） |
| Codex | 17/25（68%，最低） | — |
| Pi Agent | 通过率与 Grok Build 相同 | **中位运行时间最快、token 最少、工具调用最少（223 次）** |
| Claude Code | — | API 成本最高（$35.37），每成功 $1.96 |

**关键结论：**
- **同一模型跨 Harness 的通过率差距可达 20 个百分点**（68% → 88%）——Harness 是一阶变量。
- **工具调用次数与结果不必然正相关**：Grok Build 调用了 402 次工具，Pi 只调用 223 次，两者通过任务数相同。
- **所有 8 个 Harness 都通过了那 12 个简单任务，也都失败了那 3 个超难任务**——差异集中在中档任务。

### 6.3 成本结构模型

| 框架 | 开源成本 | 生产侧成本 | 主要锁定风险 |
|------|---------|-----------|-------------|
| **DeepAgents** | 免费 | LangSmith 订阅（托管、trace、eval） | LangChain / LangSmith 生态绑定 |
| **DSH** | 免费（MIT） | 自部署成本 + 模型 API 成本；可混搭便宜模型 + 商业子代理 | 无强制供应商锁定（但 Dev Preview 有迁移风险） |
| **Pi** | 免费（MIT） | 仅模型 API 成本（可按价格自由切换 15+ providers） | 无供应商锁定 |

---

## 7. 安全模型与沙箱

| 框架 | 沙箱策略 | 安全哲学 |
|------|---------|---------|
| **DeepAgents** | 多 backend：Modal / Daytona / Deno / 本地 VFS。可 per-thread（会话级）或 per-assistant（跨会话共享）。Agent 可在沙箱内运行，也可在沙箱外通过沙箱作为工具调用 | **在工具/沙箱层强制边界，而非指望模型自我约束**（"trust the LLM" 原则） |
| **DSH** | 沙箱是插件，可替换为 Modal、Daytona 等。**权限模型与执行边界分离**（类似容器运行时把进程权限与执行边界分开） | 明确区分"提示是建议"与"OS 级沙箱是强制"；文档直言系统提示里的"只读文件"是建议而非强制边界 |
| **Pi** | **默认无沙箱，以用户权限运行** | 官方文档明确警示："Pi 默认以你的用户权限运行，没有内置沙箱。在不受信任的代码仓库上使用前，请先阅读安全须知。" 把隔离责任完全交给用户 |

**关键洞察：**

- **DeepAgents** 与 **DSH** 都试图用 OS 层沙箱做**真正隔离**，把安全边界下沉到执行环境。
- **Pi** 将责任完全交给用户——这与它的极简哲学一致，但也意味着**在不受信任环境中使用时必须自建隔离层**（容器、VM、只读文件系统等）。
- 值得注意的是 DSH 的一个设计原则：**"告诉模型只读文件"是建议，不是强制。** 这一认识对任何 Harness 的安全设计都成立——安全必须靠执行边界，而非提示词。

---

## 8. 生态、许可与治理

### 8.1 许可证与开源

| 框架 | 许可证 | 开源完整性 | 商业限制 |
|------|-------|-----------|---------|
| **DeepAgents** | MIT（Harness）+ 商业托管 | 完全开源 | 托管部分为商业服务 |
| **DSH** | MIT | 完全开源（含商业再分发权） | 无 |
| **Pi** | MIT | 完全开源 | 无 |

三者都采用 MIT 许可，这与许多闭源 Agent 产品形成鲜明对比。DSH 官方特别强调 MIT 允许"不受限的商业与个人定制"。

### 8.2 社区与生态

| 维度 | DeepAgents | DSH | Pi |
|------|-----------|-----|-----|
| 文档完备性 | ✓✓（LangChain 文档站） | ✓（官方文档 + Cordis 论文） | ✓（官方文档 + 社区） |
| 学习资源 | ✓✓（LangChain Academy、官方视频） | ✓（第三方评测、博客） | ✓（官方教程、社区博客） |
| 社区规模 | ✓✓（LangChain 庞大社区） | ✓（快速成长，248k+ GitHub stars） | ✓（115k stars、9.4M 月下载） |
| 商业支持 | ✓ LangSmith 托管 | ○（自部署 / Web Workbench） | — |
| 第三方插件 | ✓（LangChain 生态） | ✓（`@deepseek-ai/dsh-plugin-*` + 社区 hub） | ✓（npm 包 + skill.sh 通用） |

**DSH 生态亮点：** `awesome-deepseek-harness` 整理了丰富的社区插件，涵盖 Web UI（dsh-web-ui、dsh-better-sidebar）、TUI（dsh-TUI）、Agent Teams（dsh-agent-teams）、Office 面板（dsh-office）、配额监控（dsh-deepseek-quota）等，并可通过原生 Settings 市场或 `/store` 浏览安装。

**Pi 生态亮点：** Pi 强调"通用技能协议"——其 Skills 可与 `skill.sh` 生态互通，且支持从任意目录拉取 prompts / skills / agents（"Cross Agent" 模式）。

---

## 9. 选型决策指南

### 9.1 选择 DeepAgents 的信号

- ✅ 团队已在用 LangChain / LangGraph
- ✅ 需要开箱即用，不愿从零拼装
- ✅ 需要生产级持久化、可观测性、Human-in-the-Loop
- ✅ 希望用 LangSmith 托管以降低运维负担
- ✅ 工作负载是长时、多步、跨工具的 Agent

**典型场景：** 企业级研究 Agent、SaaS 后端 Agent、数据分析流水线。

### 9.2 选择 DSH 的信号

- ✅ 想研究"everything is a plugin"的极限
- ✅ 需要混搭多个 Agent 产品（Claude Code + Codex + 自研）
- ✅ 想替换 Agent Loop 本身做实验
- ✅ 想用同质化 session log 统一调度多种工具与轨迹
- ✅ 能容忍 Developer Preview 阶段的 API 变化

**典型场景：** AI 研究实验室、跨 Agent 编排的复合工作流、对可审计性有强要求的合规场景。

### 9.3 选择 PiAgent 的信号

- ✅ 偏好终端工作流
- ✅ 想深度定制，甚至让 Agent 改写自己
- ✅ 想把 Agent 嵌入自家产品（SDK 嵌入或 RPC）
- ✅ 能接受"自己装扩展"的认知负担
- ✅ 希望 system prompt 极短、token 成本敏感

**典型场景：** CLI 工具开发者、独立开发者、轻量级编程助手、嵌入式 Agent 产品。

### 9.4 一句话决策树

```
需要托管 + 生产稳定 + 完整工具链        → DeepAgents
需要极致插件化 + 跨产品编排 + 可审计     → DSH
需要极简 + SDK 嵌入 + 多语言 RPC        → PiAgent
多场景并存                             → DSH（其 subagent 提供器可同时挂载 Claude Code / Codex）
```

---

## 10. 风险、局限与趋势判断

### 10.1 各自的风险与局限

| 框架 | 风险 | 局限 |
|------|------|------|
| **DeepAgents** | 强绑定 LangChain 生态，迁移成本不低 | Python 为主，TS 支持实验性 |
| **DSH** | Developer Preview 阶段，存在兼容性破坏性变更 | Node.js 生态，其他语言集成需 RPC；高速演进中，文档与稳定性仍在打磨 |
| **PiAgent** | 个人维护，长期可持续性依赖核心贡献者 | **默认无沙箱、无 HITL、无子代理**——高生产级特性需自行扩展 |

### 10.2 四大共同趋势

1. **Harness 成为产品层**：三者都把 Harness 与 Model 明确解耦，"Agent = Model + Harness" 成为共识。
2. **可观测性即基础设施**：DeepAgents → LangSmith；DSH → Trajectory 视图；Pi → JSON 事件流。append-only event stream 在三者中都是核心。
3. **会话持久化与分支**：从"聊天记录"升级为"Agent OS 级能力"（checkpoint / resume-fork-replay / 树状 branch）。
4. **开源 + 商业二段式**：三者都走 MIT 开源，盈利点在托管 / 服务 / 生态——模型之战正让位于 Harness 之战。

### 10.3 对 Harness 设计者的启示

- **安全必须靠边界，而非提示词**：DSH 明确指出系统提示的"只读"是建议而非强制。
- **极简是竞争力**：Pi 用 4 个工具 + <1000 token 提示，在多项基准上取得最高通过率与最低成本，说明"少即是快"。
- **可替换性是上限**：DSH 把 Agent Loop 也做成插件，把 Harness 的可组合性推到极限，适合研究与复合编排。
- **压缩策略要有干净接缝**：DSH 的 `ctx.compaction` 抽象是良好示范——模型不调用压缩，压缩是独立的可替换机制。

---

## 11. 结语

三大框架代表 Agent Harness 的三种不同哲学，构成一条完整的"自由度光谱"：

| 框架 | 一句话定位 | 决策权归谁 |
|------|-----------|-----------|
| **DeepAgents** | 工具齐全、托管友好的工业级方案 | **框架替你做决策** |
| **DSH** | 万物皆插件、可组合性的极致演绎 | **你控制每一层** |
| **PiAgent** | 原语而非功能、可自生长的极简主义 | **你从最底层自己造** |

在"模型不再是产品，Harness 才是产品"的时代，选型的本质是回答一个更深的问题：

> **你的团队愿意在哪一层做决策？**

- 想让框架做决策 → **DeepAgents**
- 想自己控制每一层 → **DSH**
- 想从最底层自己造 → **PiAgent**

三者不存在绝对优劣，只有与团队成熟度、运维能力、工作负载特征是否匹配。Harness 之战的号角已经吹响——而这，正是 Agent 走向大规模生产的必经之路。

---

## 附录：参考资料

### DeepAgents
1. [Deep Agents overview — Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/overview)
2. [GitHub — langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
3. [What Are LangGraph Deep Agents? The Harness Explained — TrueFoundry](https://www.truefoundry.com/blog/what-are-langgraph-deep-agents-the-harness-explained)
4. [LangChain DeepAgents Explained — KmeansKaran Substack](https://kmeanskaran.substack.com/p/how-agent-harnesses-beat-classic)
5. [How to Build Production-Ready AI Agents with Deep Agents — Milvus Blog](https://milvus.io/blog/how-to-build-productionready-ai-agents-with-deep-agents-and-milvus.md)
6. [Building Production-Ready Deep Agents with LangChain 1.0 — Medium](https://medium.com/data-science-collective/building-deep-agents-with-langchain-1-0s-middleware-architecture-7fdbb3e47123)

### DSH（DeepSeek Harness）
1. [GitHub — deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
2. [DeepSeek Harness — 官方网站](https://www.deepseek.com/en/harness)
3. [DeepSeek Harness Review: Agent Loop & Plugin Architecture — DeepInfra Blog](https://deepinfra.com/blog/deepseek-harness-review)
4. [What Is DeepSeek Harness? The Agent Where Everything Is a Plugin — DataCamp](https://www.datacamp.com/blog/what-is-deepseek-harness)
5. [DeepSeek Harness v0.1 released as open-source developer preview — The Neural Feed](https://theneuralfeed.com/article/deepseek-harness-v0-1-released-as-open-source-developer-preview/v8iJpU72)
6. [DeepSeek Harness Plugin Architecture: Everything Is a Plugin (Cordis) — TeamoRouter](https://teamorouter.com/blogs/dsh-plugin-architecture-cordis)
7. [DeepSeek Harness tool-catalog.md — GitHub](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/tool-catalog.md)
8. [awesome-deepseek-harness — GitHub](https://github.com/0xsline/awesome-deepseek-harness)

### PiAgent（Pi Coding Agent）
1. [GitHub — earendil-works/pi](https://github.com/earendil-works/pi)
2. [Pi — 官方网站](https://pi.dev)
3. [Pi Agent Harness (pi.dev): Minimal Coding Agent by Mario Zechner — explainx.ai](https://explainx.ai/blog/pi-minimal-agent-harness-mario-zechner-guide-2026)
4. [Pi Coding Agent: Install, Models, Setup, and Safety Guide — MorphLLM](https://www.morphllm.com/pi-coding-agent)
5. [Pi Coding Agent: The Only Claude Code Competitor — Agentic Engineer](https://agenticengineer.com/the-only-claude-code-competitor)
6. [Pi – A minimal terminal coding harness — Hacker News](https://news.ycombinator.com/item?id=47143754)

### 横向对比与基准测试
1. [DeepSeek Harness (DSH) vs Pi Agent — Composio](https://composio.dev/content/deepseek-harness-vd-pi-agent)
2. [Finding the Best Harness for DeepSeek V4 Flash — Composio](https://composio.dev/content/best-agent-harness-deepseek-v4-flash)
3. [8 Best AI Agent Harnesses in 2026: Performance, Cost, and Speed Compared — Composio](https://composio.dev/content/best-ai-agent-harnesses)
4. [DeepSeek Harness vs Pi Coding Agent: MCP, Tools and Safety — agiflow.io](https://agiflow.io/blog/deepseek-harness-vs-pi-coding-agent)
5. [Why Pi Is My GOAT Agent Harness — thinhcorner.com](https://thinhcorner.com/writing/pi-agent)

---

*报告基于 2026 年公开资料整理。基准测试数据来自第三方（Composio 等），对比条件并非完全对齐，请作为方向性参考。*

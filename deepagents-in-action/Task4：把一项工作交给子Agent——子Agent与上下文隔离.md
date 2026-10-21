
# 把一项工作交给子Agent——子Agent与上下文隔离

## 学习材料

https://datawhalechina.github.io/deepagents-in-action/chapters/ch05-subagents/

## 核心概念

### 为什么需要子 Agent？

#### （1）上下文膨胀问题

有些子任务非常复杂，需要大量工具调用和中间处理——如果全在主 Agent 里完成，上下文很快就会爆炸。

#### （2）Context Quarantine（上下文隔离）

主Agent根本不需要知道过多细节——它只需要最终的摘要。

![上下文隔离对比：没有子 Agent 时所有中间过程堆积在主 Agent 上下文中；有子 Agent 时中间过程隔离在子 Agent 中，主 Agent 只收到精炼摘要](assets/13-comparison-context-quarantine.png)

### 什么时候用子 Agent？

| 场景                                     | 是否用子 Agent | 理由                                    |
| ---------------------------------------- | -------------- | --------------------------------------- |
| 需要多次搜索和整理的研究任务             | ✅              | 大量中间结果会膨胀主 Agent 上下文       |
| 需要特殊工具或指令的专业任务             | ✅              | 子 Agent 可以有自己的工具集和系统提示词 |
| 需要不同模型能力的任务                   | ✅              | 子 Agent 可以用不同的模型               |
| 需要高层协调的复杂任务                   | ✅              | 主 Agent 专注协调，子 Agent 专注执行    |
| 单步简单查询                             | ❌              | 委派开销大于收益                        |
| 需要主 Agent 看到子 Agent 的完整调用过程 | ❌              | 主 Agent 只收到最终结果                 |

### General-purpose 子 Agent

- ==General-purpose 子 Agent 的作用是**上下文隔离**——它复用主 Agent 的模型和显式工具，使用自己的系统提示词，在独立的上下文中处理委派任务。主 Agent 不需要承受子任务中 10 次搜索带来的上下文膨胀，只需要收到一份精炼的摘要。==

- 即使你不定义任何子 Agent，Deep Agents 也自带一个 **general-purpose 子 Agent**。它**继承主 Agent 显式传入的 `tools`、`model` 和 `skills`**。

- 它使用自己的默认系统提示词（`DEFAULT_SUBAGENT_PROMPT`），并应用当前模型的 Harness Profile 提示词配置，**不会继承主 Agent 传入的 `system_prompt`**

#### （1）禁用 general-purpose 子 Agent

如果你不想让 Agent 拥有 `task` 工具，可以完全禁用子 Agent 机制：

```python
from deepagents import create_deep_agent
from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile

# 两步缺一不可
# 为指定模型注册 Harness Profile。
register_harness_profile(
    key="openai:zai-org/GLM-5.2",
    profile=HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            enabled=False
        )
    ),
)

agent = create_deep_agent(
    model=model,
    subagents=[],  # 不传任何同步子 Agent
)
```

> ⚠️ 不要用 `excluded_middleware` 来排除 `SubAgentMiddleware`——这会直接抛出 `ValueError`。正确做法是通过 `GeneralPurposeSubagentProfile(enabled=False)` 关闭。

#### （2）覆盖 general-purpose 子 Agent

如果你想给默认子 Agent 配置不同的系统提示词、模型或工具，可以用 `name="general-purpose"` 来覆盖：

> **示意片段**：`model`、`internet_search`、`ChatOpenAI` 和 `os` 沿用应用已有定义，这里只展示覆盖字段。

```python
agent = create_deep_agent(
    model=model,  # 主 Agent 复用应用的默认模型
    tools=[internet_search],
    subagents=[
        {
            "name": "general-purpose",  # 覆盖默认
            "description": "通用助手，处理各种委派任务",
            "system_prompt": "你是一个通用助手。",  # 单独配置子 Agent 的提示词
            "tools": [internet_search],
            "model": ChatOpenAI(  # 子 Agent 用更强的模型
                model="zai-org/GLM-5.2",
                api_key=os.environ["SILICONFLOW_API_KEY"],
                base_url="https://api.siliconflow.cn/v1",
            ),
        },
    ],
)
```

#### （3）为 general-purpose 启用 Todo

需要让默认子 Agent 也使用 `write_todos` 时，用同名 spec 显式配置它的中间件：

> **示意片段**：`model` 是应用已配置的 LangChain 模型对象，这里只展示主、子 Agent 分别启用 Todo 的方式。

```python
from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    model=model,
    middleware=[TodoListMiddleware()],  # 主 Agent 的规划能力
    subagents=[
        {
            "name": "general-purpose",  # 覆盖自动添加的默认子 Agent
            "description": "处理需要规划的委派任务",
            "system_prompt": "为委派任务制定计划，执行后返回结果。",
            "middleware": [TodoListMiddleware()],  # 子 Agent 单独启用
        },
    ],
)
```

两处 Todo 配置相互独立：主 Agent 设置的 `tool_description` 等参数不会自动应用到子 Agent，子 Agent 需要单独配置。在默认的隔离模式下，子 Agent 不会自动读取或写回主 Agent 的 `todos`；双方各自维护任务清单。其他声明式子 Agent 也使用同样的配置方式。

### 定义子 Agent方式1️⃣：字典方式

最常用的方式是用**字典**定义子 Agent：

> **示意片段**：本节聚焦子 Agent 字段；`internet_search` 代表应用自行实现的工具，`/skills/...` 代表项目中实际存在的 Skill 目录。

```python
from deepagents import create_deep_agent

# 定义一个研究型子 Agent
research_subagent = {
    "name": "researcher",                # 必填：唯一标识符
    "description": "深入研究特定主题，搜索多个信息源并整理成摘要",  # 必填：主 Agent 靠它决定何时委派
    "system_prompt": """你是一位专业的研究员。你的任务是：
1. 把研究问题拆解为多个搜索查询
2. 用 internet_search 搜索相关信息
3. 整理发现，写成简洁摘要
4. 列出关键发现和信息来源

注意：返回结果控制在 500 字以内，只返回核心发现。""",  # 必填：子 Agent 自己的指令
    "tools": [internet_search],          # 可选，默认继承；显式指定后完全替换（不合并）
    "skills": ["/skills/research/"],     # 可选，不继承主 Agent；指定后独立运行 SkillsMiddleware
}

agent = create_deep_agent(
    model="google_genai:gemini-3.1-pro-preview",
    skills=["/skills/main/"],            # 主 Agent 和 general-purpose 子 Agent 继承此处
    subagents=[research_subagent],       # researcher 只获得 /skills/research/，不获得 /skills/main/
)
```

#### （1）字段说明

| 字段              | 必填 | 继承主 Agent？             | 说明                                                         |
| ----------------- | ---- | -------------------------- | ------------------------------------------------------------ |
| `name`            | ✅    | —                          | 唯一标识符，主 Agent 通过它指定委派给谁；同时作为流式输出和追踪中的 `lc_agent_name` 元数据 |
| `description`     | ✅    | —                          | 描述子 Agent 的能力，主 Agent 靠它决策路由                   |
| `system_prompt`   | ✅    | ❌ 不继承                   | 子 Agent 自己的指令，需要独立定义                            |
| `tools`           | 可选 | ✅ 默认继承，指定后完全替换 | 子 Agent 工具集；不指定则继承主 Agent 的所有工具，指定后完全替换（不合并） |
| `model`           | 可选 | ✅ 默认继承                 | 可指定不同模型，支持 LangChain 对象或 `"provider:model"` 字符串（如 `"openai:gpt-5.4"`） |
| `middleware`      | 可选 | ❌ 不继承                   | 子 Agent 自己的中间件                                        |
| `interrupt_on`    | 可选 | ✅ 默认继承                 | 子 Agent 的人工审批配置，可覆盖主 Agent                      |
| `skills`          | 可选 | ❌ 不继承                   | 子 Agent 自己的 Skills 路径；指定后子 Agent 运行独立的 SkillsMiddleware，技能状态与主 Agent 完全隔离 |
| `response_format` | 可选 | ❌ 不继承                   | 结构化输出 schema（需 `deepagents>=0.5.3`）；设置后主 Agent 收到 JSON 而非自由文本 |
| `permissions`     | 可选 | ✅ 默认继承，指定后完全替换 | 文件系统权限规则；指定后完全替换主 Agent 的权限配置          |

#### （2）关键点

- **子 Agent 的 `system_prompt` 不继承主 Agent**——每个子 Agent 应有自己专属的指令。
- `tools` 默认继承主 Agent 的工具集，但一旦显式指定就会完全替换（不是合并）。
- **`skills` 也不继承**——主 Agent 的 `skills` 只会传给 **general-purpose** 子 Agent，其他子 Agent 需要显式配置自己的 `skills` 路径。

- `permissions` 同样需要特别小心：省略时继承父规则，显式提供后则整体替换而非叠加。如何为只读审计子 Agent 写出闭合的权限集
- **默认 `general-purpose` 和声明式子 Agent 都不会继承主 Agent 的 `TodoListMiddleware`**。需要规划能力时，应在子 Agent 自己的 `middleware` 字段中添加 `TodoListMiddleware(

### 定义子 Agent方式2️⃣：CompiledSubAgent（集成 LangGraph 工作流）

于更复杂的场景，你可以用一个预构建的 LangGraph 图作为子 Agent。这适用于需要**多步骤、有分支逻辑**的工作流：

> **示意片段**：`model`、`statistical_analysis` 和 `generate_chart` 代表应用已有的模型与工具；代码只演示如何把已编译图包装成 `CompiledSubAgent`。

```python
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent

# 用 LangChain 创建一个自定义 Agent 图
custom_graph = create_agent(
    model=model,
    tools=[statistical_analysis, generate_chart],
    system_prompt="你是数据分析专家，擅长统计分析和可视化。",
)

# 包装为 CompiledSubAgent
data_subagent = CompiledSubAgent(
    name="data-analyzer",
    description="执行复杂的数据分析任务，包括统计分析和图表生成",
    runnable=custom_graph,  # 传入编译好的 LangGraph 图
)

agent = create_deep_agent(
    model=model,
    subagents=[data_subagent],
)
```

`CompiledSubAgent` 的要求：传入的 LangGraph 图的 State 中必须包含 `"messages"` 键。

### 字典 vs CompiledSubAgent：怎么选？

| 场景                               | 推荐方式         | 理由                                     |
| ---------------------------------- | ---------------- | ---------------------------------------- |
| 大多数情况                         | 字典方式         | 简单直观，配置灵活                       |
| 子 Agent 需要复杂的多步骤工作流    | CompiledSubAgent | 可以用 LangGraph 的图 API 定义分支、循环 |
| 子 Agent 已经有现成的 LangGraph 图 | CompiledSubAgent | 直接复用，无需重写                       |



## Task4：多Agent协作研究员

### 示例代码

#### （1）入口代码

```python
# 4. 协调者：只负责拆计划、委派和汇总
coordinator_instructions = """你是一位项目协调者。面对复杂研究任务时：
1. 先用 write_todos 制定计划
2. 将数据收集委派给 data-collector，让它返回结构化摘要
3. 把摘要原样交给 data-analyzer，取回 3-5 条关键发现
4. 把发现和目标路径交给 report-writer，由它写入 /reports/<topic>.md
5. 用 read_file 核对成稿，再向用户汇总各子 Agent 的结果

研究、分析和成稿通过这三个子 Agent 完成。
查阅或小幅修改已有报告时，直接使用 read_file 与 edit_file。
临时草稿写入 /scratch/，仅保留在当前会话。"""

agent = create_deep_agent(
    model=model,
    backend=backend,
    checkpointer=MemorySaver(),
    middleware=[TodoListMiddleware()],
    system_prompt=coordinator_instructions,
    subagents=subagents,
)
```

#### （2）关掉默认的 general-purpose 子Agent

```bash
# 关掉默认的 general-purpose子Agent，避免它绕过三个专用子 Agent 自己做完研究。
register_harness_profile(
    "openai",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)
register_harness_profile(
    "anthropic",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)
```

#### （3）三个子Agent定义

```python
# 3. 子 Agent
# SkillsMiddleware 只会扫描 source 目录下的子文件夹，因此每个子 Agent
# 使用单独的 source（如 /skills/collector/），里面只放自己的一个 skill。
subagents = [
    {
        "name": "data-collector",
        "description": "从多个来源收集原始数据，包括网络搜索和 API 调用。需要资料时委派给它，并在任务里写明主题。",
        "system_prompt": "你是数据收集专家。搜索并整理相关数据，返回结构化的数据摘要。开始前先读取并遵循你的技能。",
        "tools": [internet_search, api_call],
        "skills": ["/skills/collector/"],
    },
    {
        "name": "data-analyzer",
        "description": "对收集到的数据进行统计分析，提取关键洞察。把上一步的摘要原样放进任务里再委派。",
        "system_prompt": "你是数据分析专家。分析数据并提取 3-5 个关键发现，控制在 300 字以内。开始前先读取并遵循你的技能。",
        "tools": [statistical_analysis],
        "skills": ["/skills/analyzer/"],
    },
    {
        "name": "report-writer",
        "description": "根据分析结果撰写专业报告并保存到 /reports/。委派时要给出分析结果和目标文件路径。",
        "system_prompt": "你是技术写作专家。根据提供的分析结果撰写清晰、专业的报告。开始前先读取并遵循你的技能。",
        "tools": [format_document],
        "skills": ["/skills/writer/"],
    },
]
```

#### （4）配置组合后端

```python
# 2. 配置 CompositeBackend
# - /reports/ 路由到磁盘上的 reports 目录，研究报告会持久化保存
# - /skills/ 路由到磁盘上的 skills 目录，供各子 Agent 读取自己的技能
# - 其他路径走 StateBackend，只活在当前 thread，适合临时草稿
project_dir = Path(__file__).resolve().parent
reports_dir = project_dir / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/reports/": FilesystemBackend(root_dir=reports_dir, virtual_mode=True),
        # 把虚拟路径 /skills/ 映射到 skills/ 目录
        "/skills/": FilesystemBackend(root_dir=project_dir / "skills", virtual_mode=True),
    },
)
```

#### （5）技能代码

- **目录接口**

```
tree skills/
skills/
├── analyzer
│   └── data-analyzer
│       └── SKILL.md
├── collector
│   └── data-collector
│       └── SKILL.md
└── writer
    └── report-writer
        └── SKILL.md
```

- **技能内容**

```markdown
---
name: data-analyzer
description: 分析已收集的资料时使用。提取少量关键发现。
---

只处理委派内容里已经给出的资料。调用 statistical_analysis 整理条目，再写出 3 到 5 条发现，全文控制在 300 字以内并直接返回。

---
name: data-collector
description: 收集研究资料时使用。先搜索，再补一条 API 记录，返回结构化摘要。
---

用 internet_search 围绕主题搜索 1 到 2 次，再用 api_call 补一条来源。把要点、数字和来源整理成简短摘要后直接返回，交给协调者。

---
name: report-writer
description: 根据分析结果撰写报告并保存到指定路径时使用。
---

只根据委派内容里的分析结果撰写简短报告。先用 format_document 生成 markdown，再用 write_file 保存到任务指定的 /reports/ 路径，最后返回该路径。
```

### 执行结果

```
========== 研究并保存报告 ==========
Agent 回复:
 LangGraph 研究报告已完成，保存至 `/reports/langgraph_summary.md`（131 行，约 1700 字）。

## 核心内容

**LangGraph 核心概念**

LangGraph 是 LangChain 团队推出的 orchestration runtime，首发于 2024-06-27，MIT 许可证。其核心抽象是一张显式的状态图：

- **State / StateGraph** — 贯穿全图的中央状态对象，用 TypedDict 或 Annotated 声明 schema；`compile()` 后返回遵循 Runnable 协议的可运行对象
- **Node** — 签名 `State -> Partial`，普通 Python 函数即可，无需装饰器或继承
- **Edge / Conditional Edge** — 静态连线与动态路由，`add_conditional_edges` 返回 `Send` 可做 Map-Reduce 扇出
- **Reducer** — schema 上声明的合并函数，规定多节点写同一字段时如何合并
- **Checkpoint / interrupt / Command / Time Travel** — 每 super-step 持久化状态，`interrupt()` 冻结、`Command(resume=)` 恢复、`get_state_history()` 回溯
- **Subgraph** — 已编译的图可作为另一图的节点

**三点关键优势**

1. **状态机式运行时 vs 隐式循环** — 传统 `AgentExecutor` 把控制流藏在隐式循环里，难以插入持久化、分支或人工闸门；LangGraph 把 ReAct 式回环变成默认拓扑，可枚举、可检查、可注入。
2. **checkpointing 是 HITL 与 time travel 的公共底座** — 持久执行、人机协同、时间旅行不是三个并列特性，而是同一 checkpoint 序列的三种读法（冻结点 / 向前恢复 / 任意点回溯）。没有 checkpointer，后两者都不成立。
3. **多 Agent 是原语组合而非独立能力** — supervisor/router 只需 subgraph + 条件边，价值在于复杂度被显性化，而非能力更强。

**适用边界**

含循环/分支的工作流、多人审批、跨会话状态、多 Agent 监督、需要断点续跑的长任务适用；纯线性 pipeline、快速原型 demo 不适用。四家独立来源的批评高度一致（过度工程、"非真正 agentic"、不利于快速原型），另有实证缺陷 issue #9106：并行中断在 subgraph 恢复时可能重跑已完成的兄弟节点。

**两处不确定数据已按保守方式处理**：GitHub stars 各来源相差约 20%（34k vs 40,713），报告中表述为「约 3.4 万–4.1 万，量级可靠、精确值不可靠」；v1.0 日期仅有单一二手来源转述 2025-10，标注「官方 release notes 未直接核实」故不写具体日期。此外报告提示了版本迭代节奏异常快的问题，引用 patch号前建议现场核对。

流程上，data-collector 汇总了 6 大板块带 URL 的资料，data-analyzer 提取了 5 条发现并对口径冲突做了可信度评估，report-writer 据此成稿，我已逐行核对。
```

### Langsmith跟踪记录

#### （1）write_todos 制定计划

![image-20261021114724945](assets/image-20261021114724945.png)

![image-20261021114745836](assets/image-20261021114745836.png)

#### （2）委派给数据收集子Agent

![image-20261021115311850](assets/image-20261021115311850.png)

![image-20261021115418954](assets/image-20261021115418954.png)

#### （3）数据收集子Agent根据技能进行数据收集

![image-20261021115649963](assets/image-20261021115649963.png)

![image-20261021115738597](assets/image-20261021115738597.png)

#### （4）完成资料收集后更新TodoList

![image-20261021120114925](assets/image-20261021120114925.png)

#### （5）完成数据收集后，启动数据分析Agent

![image-20261021120330141](assets/image-20261021120330141.png)

![image-20261021120431033](assets/image-20261021120431033.png)

#### （6）分析Agent调用数据分析技能进行采集数据分析

![image-20261021120615195](assets/image-20261021120615195.png)

#### （7）完成数据分析后更新TodoList

![image-20261021120733785](assets/image-20261021120733785.png)

#### （8）完成数据分析后，启动报告撰写Agent

![image-20261021120855251](assets/image-20261021120855251.png)

#### （9）告撰写Agent调用响应技能完成报告撰写

![image-20261021121056844](assets/image-20261021121056844.png)

#### （10）主Agent完成报告检查后，更新TodoList状态

![image-20261021121243725](assets/image-20261021121243725.png)

#### （11）三个子Agent串行调用链路

![image-20261021121651332](assets/image-20261021121651332.png)
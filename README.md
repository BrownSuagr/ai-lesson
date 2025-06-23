# AI-Lesson 人工智能课程学习项目

本项目是一个系统性的人工智能课程学习仓库，涵盖从 Python 编程基础到大型语言模型（LLM）开发的全栈知识体系。适用于希望系统掌握 AI 技术的学习者、学生和开发者。

---

## 📚 项目结构概览

本仓库共包含 10 个阶段，每个阶段聚焦不同的核心技能模块与实战项目：

| 阶段 | 内容 | 标签 |
|------|------|------|
| STAGE-1 | Python 基础编程 | 语法、面向对象 |
| STAGE-2 | Python 进阶编程 | 模块、函数、异常 |
| STAGE-3 | 数据处理与统计分析 | Pandas、Numpy、MySQL |
| STAGE-4 | 机器学习多场景项目实战 | 电商、医疗、金融、推荐系统等 |
| STAGE-5 | 金融风控项目 | 项目管理、风控实战 |
| STAGE-6 | 深度学习 | PyTorch、CNN/RNN/Transformer |
| STAGE-7 | 自然语言处理 | NLP实战、情感分析、命名实体识别 |
| STAGE-8 | 推荐系统构建 | 算法原理 + 实战（待补充） |
| STAGE-9 | 多模态大模型项目 | 跨模态建模（待补充） |
| STAGE-10 | 大模型开发基础与项目 | LLM架构、RAG、Agent等 |

---

## ⚙️ 环境配置指南

推荐使用 `conda` 创建虚拟环境，并通过 Jupyter Notebook 进行交互式学习和开发。

### 1. 创建并激活环境

```bash
conda create -n ai-lesson python=3.10
conda activate ai-lesson
```

### 2. 安装依赖项

```bash
pip install -r requirements.txt
```

如果需要 Jupyter：

```bash
conda install jupyter
```

或者使用 Jupyter Lab：

```bash
conda install -c conda-forge jupyterlab
```

### 3. 启动 Notebook 环境

```bash
nohup jupyter notebook --allow-root &
```

---

## 🧠 学习建议

* 📌 **按阶段逐步深入**：建议从 STAGE-1 开始学习，逐步积累基础与实战经验。
* 💡 **结合 Notebook 与讲义**：项目中提供了丰富的 `.ipynb` 文件和讲义资料，便于理解与复现。
* 🔍 **注重实践与总结**：每个阶段包含了小项目或实战任务，建议动手完成并撰写学习总结。

---

## 🗂️ 示例目录结构

```
├── STAGE-1 Python基础编程
│   └── 第一章 Python基础语法
├── STAGE-2 Python进阶编程
├── STAGE-3 数据处理与统计分析
│   ├── 1-Linux                 # Linux 系统基础命令，用于数据处理环境的命令行熟练操作
│   ├── 2-MySQL                 # 数据库基础操作，包含 SQL 查询、数据表操作等
│   ├── 3-搭建环境               # 安装 Python、Jupyter、依赖包，配置数据科学工作环境
│   ├── 4-NumPy                 # 学习矩阵计算、广播机制、数组操作等基础数值处理
│   ├── 5-Pandas                # 数据清洗、筛选、合并、分组聚合等常用操作
│   ├── 6-数据分析绘图           # 可视化工具使用（如 Matplotlib、Seaborn）进行数据分析展示
│   └── 7-综合案例               # 综合使用上述知识，完成真实案例数据分析项目
├── STAGE-4 机器学习多场景项目实战
│   ├── 业务项目代码与讲义
├── STAGE-5 金融风控项目
├── STAGE-6 深度学习
│   ├── 模型demo（pipeline, RNN, Transformer 等）
│   ├── 图文讲解
│   └── 项目实战 notebook
├── STAGE-7 自然语言处理
├── STAGE-10 大模型开发基础与项目
│   └── RAG、Agent、Chain-of-Thought 多样应用
```

---

## 📄 License

本项目基于 [MIT License](./LICENSE) 开源。

---

## 🙌 鸣谢

本课程内容整理自公开教学资源与实战项目经验，致谢开源社区以及所有贡献者。

---
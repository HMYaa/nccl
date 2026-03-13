# 费曼学习法文档 · 图片目录

本目录用于存放从 **Mermaid** 导出的 PNG/SVG 图片，与 `.claude/` 下的费曼学习法文档配套使用。

## 目录结构建议

```
.claude/figures/
├── README.md           # 本说明
├── ncclCommInitAll/    # ncclCommInitAll 相关图
│   ├── 图1_四阶段总览.png
│   ├── 图2_组语义时序.png
│   ├── 图5_主流程.png
│   └── ...
├── initTransportsRank/
│   └── 图6_子流程.png
└── 其他主题/
```

## 如何从 Mermaid 导出图片

1. 打开 [Mermaid Live Editor](https://mermaid.live/)
2. 从对应费曼学习法文档中复制 ` ```mermaid ` 与 ` ``` ` 之间的代码
3. 粘贴到 Mermaid Live 左侧编辑区
4. 点击 **Actions → PNG** 或 **SVG** 下载
5. 保存到本目录下对应主题的子目录，命名与文档中「图N」一致，例如：`图1_四阶段总览.png`

## 在文档中引用

若 Markdown 环境不支持 Mermaid 渲染，可将导出的图片插入文档，例如：

```markdown
![图1：ncclCommInitAll 四阶段总览](.claude/figures/ncclCommInitAll/图1_四阶段总览.png)
```

相对路径以**文档所在位置**为基准；若文档在 `.claude/` 下，则使用 `.claude/figures/...` 或 `figures/...` 均可（取决于你的根路径设置）。

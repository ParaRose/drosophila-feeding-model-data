# `SEZ → MN9` 三连接组可移植性封顶补强协议（`mn9-portability-final`）

**协议状态**：先期锁定（在运行任何新增敏感性仿真、密度诊断、活动匹配零模型或 44 对照三脑零分布计算之前独立提交；直接回应独立审阅报告 [`XIAOKE_OPUS55_REVIEW_20261124.md`](XIAOKE_OPUS55_REVIEW_20261124.md)）。**本轮绝不引入第四套连接组，严禁触碰既有 5 份锁定结果 JSON。**

---

## 1. 稿件主线重塑与事前锁定双并列主终点（Co-Primary Endpoints）

面向 *PLOS Computational Biology* / *eLife* Reviewed Preprint，将论文核心科学问题明确为**跨连接组可移植性（Portability）**：*在单只果蝇脑连接组上零行为拟合建立的全脑脉冲模型，换入独立重建个体后哪些预测能保留、哪些会失效、失效机制是什么、以及移植时必须通过哪些无标签质控？*

1. **双并列主终点（Co-primary endpoints，全部以三脑 50 Hz 零行为拟合主尺度读出报告）**：
   - **主终点 1（全集排序）**：三套独立重建（`FlyWire v783` `0.275 mV`、`BANC v888` `0.6904 mV`、`MaleCNS v1.0` 完整左半球缩放 `0.2683 mV`）在 `50 Hz` 驱动下的 `all_106`（`14` 阳 / `92` 阴，随机基线 `0.1321`）阶梯 AUPRC。
   - **主终点 2（未发表盲筛迁移）**：上述三套重建在 `50 Hz` 驱动下的 `novel_95`（`Shiu 2024` 既往未发表 `95` 类盲筛子集，`4` 阳 / `91` 阴，随机基线 `0.0421`）阶梯 AUPRC（三脑基线已锁存为 `v783 = 0.5211`、`BANC = 0.1149`、`MaleCNS = 0.0816`，三脑无遗漏并列呈现）。
2. **次级与探索性终点及多重比较校正**：
   - 其余驱动频率（`20 / 100 / 200 Hz`）、子分母（`nondirect`、`nondirect_common_3brains`、`mapped_104`、`isomorphic_102`、`per_ge_0_2`）、静态图基线对决及单类型拆解统一标注为次级/探索性分析，并对跨连接组配对自助法 `p` 值提供 Benjamini–Hochberg（BH-FDR）校正 `q` 值。
3. **归因克制规则**：因每种成像模态、每个性别各仅 `n = 1` 套连接组（个体、性别、电镜物理模态、自动化突触检测器、校对完整度与实验室完全共线），全文删去不可识别的 `"cross-modality / cross-sex"` 因果归因措辞，统一表述为“三套独立连接组重建（two female ssTEM, one male FIB-SEM）”，并明确全部行为 PER 标签测自雌蝇。

---

## 2. 四项靶向补强检验（`mn9_portability_hardening.py` → `mn9_portability_results.json`）

### 检验 A — `MaleCNS` 递质符号规则敏感性（回应 `xiaoke §1.2`）
- **背景**：在 `Shiu 2024` 统一符号规则下，`serotonin` 与 `acetylcholine` 均为 `+1`。`MaleCNS v1.0` 中 `diatom` 的 `20` 胞单细胞 `predicted_nt` 为 `10 serotonin / 8 acetylcholine / 2 unclear`，`TH_VUM` 为 `1 serotonin`，但在主图中因 `consensus_nt == 'unclear'` 而被置零。
- **确定性对照设计（`30` 次重复，同种子规则 `50000/100000/200000 + i`）**：
  1. **`A1a`（`diatom18_thvum_plus1`）**：仅将 `diatom` 中单细胞 `predicted_nt` 为 `+1` 类的 `18` 个神经元及 `TH_VUM`（`1` 胞）赋 `+1`，保持主图其余边及 $W_{\text{syn}}^{\text{intact}} = 0.268284\text{ mV}$ 不变，运行全 `106` 类 `50 / 100 / 200 Hz`。
  2. **`A1b`（`diatom20_thvum_plus1`）**：将全部 `20/20` 个 `diatom` 胞与 `1/1` 个 `TH_VUM` 胞均赋 `+1`（上界敏感性），同权重运行全 `106` 类 `50 / 100 / 200 Hz`。
  3. **`A2`（`all_unclear_predicted_nt`）**：将全图全部 `consensus_nt == 'unclear'` 节点按单细胞 `predicted_nt` 赋号（`acetylcholine/serotonin/dopamine/octopamine/tyramine/histamine → +1`，`gaba/glutamate → −1`，仍 `unclear`/缺失 → `0`）；按更新后完整左 `MN9_L` 有效入突触重算零行为拟合权重 $W_{\text{syn}}^{\text{intact,A2}} = 0.275 \times (5732.5 / S_{\text{in,L}}^{\text{A2}})$，运行全 `106` 类 `50 / 200 Hz` **以及 `SELK (DNg68)` `20 / 50 / 100 / 200 Hz`**（直接检验 `SELK` 前 8 强上行靶点中两个 `ANXXX005` 恢复单细胞递质预测后是否解锁 `MN9`）。
- **报告项**：`diatom`、`TH_VUM` 与 `SELK` 精确放电率、`ANXXX005` 单细胞 `predicted_nt`、全脑均频，以及 `all_106`、`novel_95`、`nondirect_malecns` AUPRC 相对主基准的变化。

### 检验 B — 全网跨脑匹配神经元入突触密度诊断、全网密度缩放仿真与 6 类雄蝇假阳性二态审计（回应 `xiaoke §1.3`）
1. **静态入突触密度审计**：
   - `FlyWire v783 ↔ BANC v888`：在全部单细胞 reviewed matches 及同源 `cell_type` 匹配神经元上，计算有效递质入突触比 $S_{\text{in}}(\text{BANC}) / S_{\text{in}}(\text{v783})$ 的中位数、IQR（25%–75%）与 10% 截尾均值（分全脑、中央脑与 `105` 类 SEZ 靶集报告），对比双侧 `MN9` 比值（`2283.5 / 5732.5 = 0.3983`）。
   - `FlyWire v783 ↔ MaleCNS v1.0`：在全部同源 `flywireType`/`type` 匹配类型上，计算每神经元均值有效入突触比（及每类型总入突触比）$S_{\text{in}}(\text{MaleCNS}) / S_{\text{in}}(\text{v783})$ 的中位数、IQR 与 10% 截尾均值（分全脑、中央脑与 `104` 类 SEZ 靶集报告），对比单神经元完整左 `MN9_L` 比值（`5876 / 5886 = 0.9983`，相对 `v783` 双侧均值 `1.0250`）。
2. **全 106 类与 6 类雄蝇活跃阴性专项注释审计**：
   - 在 `body-annotations-male-cns-v1.0-minconf-0.5.feather` 中逐一提取 `peacock, puddle, spirit, mime, peahen, crab` 及全 `106` 类的 `dimorphism`、`fruDsx`、`statusLabel`、`matchingNotes`、胞数及相对 `v783`/`BANC` 的入/出突触比。
3. **全网密度匹配缩放仿真（`B_density_scale`）**：
   - 以中央脑同源匹配细胞类型每神经元入突触中位比 $r_{\text{med}}^{\text{CB}}$ 定义零行为拟合的全网密度校准权重 $W_{\text{syn}}^{\text{density}} = 0.275\text{ mV} / r_{\text{med}}^{\text{CB}}$（若 `r_med` 明显高于 `1`，则该权重显著低于 `0.2683 mV`）；在 `MaleCNS v1.0` 主图上运行全 `106` 类 `50 / 200 Hz`（`30` 试次），直接检验 `peacock / puddle / spirit / mime / peahen / crab` 六类假阳性是否消退、全脑均频如何变化、以及 `all_106` / `novel_95` / `nondirect_malecns` AUPRC 与跨脑 Spearman $\rho$ 的真实表现。

### 检验 C — `FlyWire v783` `MN9` 保护重连零图的糖锚点检验与活动匹配对照（回应 `xiaoke §1.5`）
使用与 `mn9_transfer_benchmark.py` 完全相同的 `FlyWire v783` 图与种子（`9201..9210`）重现全部 `K = 10` 张 `MN9` 保护重连图（逐边冻结 `468` 条 `MN9` 入边），执行两项实测：
1. **`C1`（糖锚点可达性扫描）**：在全部 `10` 张保护重连图上，用单侧 `21` 个 `100 Hz` 糖受体驱动扫描 $W_{\text{syn}} \in \{0.275, 0.40, 0.55, 0.70, 0.90, 1.20, 1.60, 2.00, 2.50\}\text{ mV}$（`30` 试次），同时记录 `MN9` 均频与全脑均频，判定保护重连图能否在进入全脑异常爆发前触达 `58.5 Hz` 糖锚点。
2. **`C2`（活动匹配零模型，Activity-Matched Null）**：对全部 `10` 张 `MN9` 保护重连图，逐图以确定性二分搜索（$W_{\text{syn}} \in [0.275, 2.50]\text{ mV}$，容差 `±5%`，全脑均值安全上限 `15 Hz`）求解 $W_{\text{syn}}^*$，使该重连图在 `106` 类 `50 Hz` 下的全集平均 `MN9` 放电率匹配真实 `v783` 图均值（`2.0236 Hz`），随后以 `30` 试次完整评测每张活动匹配重连图在 `all_106`、`nondirect_83` 与 `novel_95` 上的 AUPRC 与图级经验胜场（实测检验重连图败北究竟源于静默未标定还是特定多跳拓扑缺失）。

### 检验 D — 全部 `44` 个纯抑制性阴性对照的三脑保守性零分布（回应 `xiaoke §1.6`，纯静态精确计算）
对全部 `44` 个在 `v783`、`BANC v888`、`MaleCNS v1.0` 三脑完整映射的纯抑制性阴性 SEZ 对照，按与 `tentacular / phantom / marge / usnea` 完全相同的定义提取三脑 `hop2_net` 百分位、`hop3_disinh` 百分位、以及三脑 `top15_hop2_ipre` 与 `top15_hop3_i1` 同源类型集合，精确统计全部 `44` 个阴性对照在以下判据下的实测通过数与经验假阳性率 $p_{\text{emp}} = (k+1)/45$：
1. 三脑 `3-hop disinh` 百分位均 `≥ 80%`（及均 `≥ 90%`）；
2. `top15_hop2_ipre` 与 `top15_hop3_i1` 在三脑共享交集大小分布（`≥ 4`、`≥ 5`）及双脑共享集合在第三脑的 `100%` 保留率；
3. 联合判据：（i）`tentacular` 同级标准（三脑 `3-hop ≥ 90%` 且 `2-hop net ≥ 90%` 且三脑共享 `2h ≥ 5`、`3h ≥ 5`）；（ii）`phantom` 同级标准（三脑 `3-hop ≥ 80%` 且三脑共享 `3h ≥ 4`）。
无论零分布通过率高低，正文均严格依据实测 $p_{\text{emp}}$ 措辞，绝不夸大。

---

## 3. 文献核验与交付纪律

1. 在线核查 Bates 等（2026, *Nature* 656）与 Berg 等（2026, *Cell* 189）是否存在同题全脑进食 LIF 跨体横评，并准确核对 Savaş 等（2026, *Nat. Commun.* 17, 3596）关于 ACh 与 Lk 分工的原始结论。
2. 编写并运行 `verify_references.py` 对 `manuscript.md` 扩充后的 **`35–45` 篇**参考文献逐条实时请求 Crossref / arXiv API 校验作者、标题、期刊、年份、卷期页码与 DOI，零幻觉、零未核实引文。
3. 重构 `english_manuscript/manuscript.md`（≤200 词摘要、零加粗数字、零戏剧化修饰）、新增 `english_manuscript/supplementary.md`（含 `figS1.svg` 17×8 Oracle 矩阵、106 类三脑放电率与 102 组 Oracle 导出 TSV 表）、重绘修复全部 `xiaoke` 指出视觉问题的 `fig1.svg`–`fig4.svg`（二次复跑 `0-byte diff`、PDF 逐页目视检查），同步更新 `CROSS_CIRCUIT_SYNTHESIS.md`、`REPRODUCE.md` 并新增 `PROJECT_CLOSURE_MEMO.md`。
4. 全程本地提交于 `research/motor-source-benchmark`，严禁 `git push`，严禁断开在线 `colab` A100 会话 `-s mitotransmap`。

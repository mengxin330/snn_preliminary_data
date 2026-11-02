"""
第一步实验的全部参数。

改参数只改这个文件。
"""

CONFIG = {
    # ==================== 数据生成 ====================
    "rate": 1.0,              # 到达率（事件 / ms）
    "n_events": 400_000,      # 事件数
    "seed": 0,                # 随机种子

    # ==================== 服务时间 ====================
    # 利用率 rho = rate * service_time
    # 0.50 -> rho=0.50, 0.70 -> rho=0.70, 0.85 -> rho=0.85
    "service_times": [0.50, 0.70, 0.85],

    # ==================== 缓冲 ====================
    # 验证组用无限缓冲（这样 M/D/1 公式才成立）
    # 观察组用有限缓冲（看丢弃与峰值的真实行为）
    "capacity_validate": None,
    "capacity_observe": 32,

    # ==================== warm-up 检测 ====================
    "warmup_window": 1000,      # 滚动窗口长度（事件数）
    "warmup_tol": 0.05,         # 平稳判据：尾部窗口均值的极差 < 5%
    "warmup_min_windows": 10,   # 稳态段至少保留的窗口数
    "warmup_max_frac": 0.50,    # warm-up 不得超过总样本的比例

    # ==================== 批次均值法 ====================
    "n_batches": 10,            # 把稳态段分成几批

    # ==================== 判据 ====================
    "max_n_se": 2.0,            # 误差 < 2 个标准误视为通过

    # ==================== 输出 ====================
    "results_dir": "results",
    "make_plots": True,         # 是否画图

    # ==================== 多跳 NoC（对齐 BookSim） ====================
    # 拓扑与路由器
    "mesh_n": 8,                 # 8x8 mesh
    "vc_count": 16,              # BookSim num_vcs
    "vc_depth": 8,               # BookSim vc_buf_size
    "hops_cycles": 3,            # BookSim: routing 0 + vc_alloc 1 + sw_alloc 1 + st 1
    "n_output_ports": 5,         # 北 南 东 西 本地

    # 消息长度（flits）
    "n_end": 1,                  # end-of-message 信号
    "n_payload": 1,              # 数据本身（一个事件）
    "n_header_max": 3,           # 单条 dual-path 消息的 header 地址数上限

    # 每端口 FIFO 容量（flits）。
    # BookSim 的 num_vcs × vc_buf_size = 16 × 8 = 128，但那是为 cache coherence
    # 场景（4 类消息各分 4 个 VC）设计的，不是为"测量缓冲峰值"设计的。
    # 实测（scripts/step0_sim_smoke.py）：在 events_per_source=10000、λ_s=0.03、
    # lrd hurst=0.6 下，去掉容量限制后的真实峰值是 605 flits；
    # 用 128 会把它截断 4.7 倍（峰值饱和在 128 + 0.15% 丢弃）。
    # 因此主 pipeline 显式用 1024（2 的幂），使 "peak buffer occupancy" 这个
    # 目标量不被截断。step2 的验收脚本仍用 128（它验证的是 BookSim 配置下的模拟器）。
    "port_capacity": 1024,

    # 负载（三档，做敏感性检查）
    # 注意：多跳网络的端口负载是"分布"（实测 max/mean ≈ 2.8），
    # 一个平均 rho 无法确定工作点，所以直接用"每源事件率"作控制变量。
    # 实测 λ=0.10 时平均 util 已 0.54、热点 1.50，所以三档取更低的值，
    # 使热点利用率约为 0.30 / 0.53 / 0.75。
    #
    # 主工作点的定案依据（scripts/step0_sim_smoke.py 实测）：
    #   λ=0.03: ρ_avg=0.061, ρ_hot≈0.17, P95 族间比 1.75×, 四族 drop=0
    #   λ=0.05: ρ_avg=0.101, ρ_hot≈0.28, P95 族间比 2.40×, 四族 drop=0
    #           LRD 真实峰值 692 < port_capacity=1024
    # 取 0.05：族间差异更大、仍不饱和、容量不用改。
    "lambda_s": 0.05,                          # 主工作点（事件 / cycle / 源）
    "lambda_s_sensitivity": [0.03, 0.05],      # 敏感性档（验证结论不依赖工作点）
    "lambda_s_targets": [0.02, 0.035, 0.05],   # 旧三档（step2 的负载敏感性仍用）
    "rho_targets": [0.2, 0.6, 0.8],            # 保留备查

    # 每源事件数
    # 三档都已实测（10000 / 5000 / 3000），用来观察 C3 效应量随观测长度的衰减：
    #   n=10000: LRD ΔR²=+2.89%, F=21.7, p=9.1e-50, 但峰值 13% 触顶 cap=1024
    #   n=5000 : LRD ΔR²=+2.13%, F=11.2, p=7.6e-27, 峰值  5.8% 触顶
    #   n=3000 : 待测
    # MFDFA 尺度数：n=10000 为 9 个、5000 为 8 个、3000 为 7 个；
    # 论文 [1] 自身 MFDFA 输入中位数约 1100 个 ISI（只够 6 个尺度），
    # 所以三档都不低于论文的实践水平。
    "events_per_source": 3_000,

    # 连接模型（论文 [1] Table 1）
    "sigma_ref": 10.0,           # 30x30 上的 sigma（归一化单位）
    "mesh_n_ref": 30,
    "alpha_ee": 0.11,            # alpha_{E->E} 中档

    # 目的地采样尺度 σ（核间距单位）。
    # 直接换算 sigma_ref × 8/30 ≈ 2.67 只得到 localization factor 0.144，
    # 低于 ISCAS 2005（Pande et al.）研究的 0.3–0.8 区间。
    # 改为按该文献**下沿**反解（扫描见 src/destinations.py 注释与本次记录）：
    #   σ=1.30 → locality 0.359 | σ=1.40 → 0.329 | **σ=1.50 → 0.301**
    #   σ=1.60 → 0.277 | σ=1.70 → 0.256 | σ=2.00 → 0.206
    # 取 1.50：使 locality 恰好落在文献覆盖的最低档。
    # 这样 σ 有了外部依据（对齐文献的 locality 范围），
    # 而不只是"神经元→核"的尺度换算假设。
    "sigma": 1.5,

    # 扇出（每事件目的地数）—— **逐核固定**，由论文 [1] 的几何出度决定
    #
    # 依据：论文 [1] Methods 的连接公式  P_ij = α · exp(-d_ij² / 2σ²)
    #   fanout_i = α · Σ_{j≠i} exp(-d_ij² / 2σ²)   （核 i 到其余核的期望连接数）
    #
    # 本项目的取值链：
    #   σ = 1.5     —— 使 localization factor = 0.30，对齐 ISCAS 2005 覆盖区间的下沿
    #   α = 0.274   —— **反解**，使平均扇出 = 2.56
    #                  （2.56 是同一公式在 σ=2.67 面积缩放下的原始结果）
    #   结果：均值 2.56、CV 0.256、范围 1.3–3.5、locality 0.30
    #
    # 代价（必须在报告中披露）：α 不是论文的 0.11，而是为同时满足
    # "locality=0.30" 与 "平均扇出=2.56" 反解出的值，偏离 2.5 倍。
    # 若只用论文参数（σ=2.67, α=0.11）则 fanout=2.56 但 locality 仅 0.144；
    # 若用 σ=1.5 配 α=0.11 则 fanout 退化到 1.03（100% 的核扇出 < 2，多播失效）。
    "fanout_distribution": "geometric",   # "geometric"(逐核固定) | "fixed" | "two_point" | "poisson"
    "fanout_alpha": 0.274,
    "fanout_fraction": None,              # 若非空则按节点数比例（Fleury 风格），优先于 alpha
    "fanout_mean": 2.56,                  # 回退值 / 校验目标

    # 多播策略："unicast" | "dual_path"
    "multicast_mode": "dual_path",

    # 是否把到达时刻量化到整数 cycle
    "quantize_arrivals": False,

    # ==================== 特征的时间尺度（一律写成 × mean ISI） ====================
    # 单位说明：多跳段的时间单位是 **cycles**；单队列段（rate/service_time）
    # 的时间单位任意（标为 ms），因为 rho = rate * service_time 与单位无关。
    # 特征侧不引入 Hz / ns —— 所有时间尺度都写成「mean ISI 的倍数」，
    # 于是特征无量纲、跨负载可比。依据：Legendy & Salcman (1985) 的 burst
    # 检测参数本身就是 3 spikes / 0.5·μ / 2·μ。
    # 前两项来自原文；后三项待敏感性检查（见 features 的 docstring）。
    "isi_rel": {
        "burst_merge_ratio": 0.5,      # Legendy & Salcman 1985
        "burst_split_ratio": 2.0,      # Legendy & Salcman 1985
        "fano_window_ratio": 5.0,      # 待敏感性检查
        # 每箱平均事件数 = acf_bin_ratio^-1；取 1.0 使每箱平均 1 个事件，
        # 否则分箱过细会产生大量空箱，lag-1 相关不可靠。
        "acf_bin_ratio": 1.0,          # 待敏感性检查
        "acf_max_lag_ratio": 10.0,     # 待敏感性检查
    },
}

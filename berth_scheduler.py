"""
泊位调度优化程序
使用 PuLP 建立整数规划模型，最小化船舶的总（加权）在港时间，
支持通过 JSON 传入潮汐窗口、依赖关系、优先级权重、靠泊限制、
最小间隔、自定义硬约束等可选条件，并通过 Flask 提供 HTTP API 接口。
"""

from flask import Flask, request, jsonify
import pulp

app = Flask(__name__)


def optimize_berth_scheduling(ships, num_berths,
                              tide_windows=None,
                              dependencies=None,
                              priority_weights=None,
                              berth_restrictions=None,
                              min_gap=0.0,
                              custom_constraints=None,
                              objective="weighted_time"):
    """
    使用 PuLP 建立整数规划模型求解泊位调度问题。

    参数:
        ships: 船舶列表，每艘船为 dict，包含 name, arrival, service
        num_berths: 可用泊位数量
        tide_windows: 各泊位的不可作业（潮汐）区间，形如
            [[[9,11],[15,17]], [], [[12,14]]]，外层下标为泊位号（0-based）
        dependencies: 依赖关系 dict，如 {"C": ["A"], "E": ["B","D"]}，
            表示右侧被依赖船全部离港后，该船才能开始作业
        priority_weights: 优先级权重 dict，如 {"A": 3}；未列出的船默认权重 1
        berth_restrictions: 靠泊限制 dict，如 {"A": [0,1]}（0-based 泊位索引），
            表示该船只能停靠列出的泊位
        min_gap: 同泊位前后两船之间的最小间隔（小时），默认 0（不启用）
        custom_constraints: 自定义硬约束列表，当前支持
            {"type": "deadline", "ship": "A", "time": 16.0}
        objective: 目标函数类型，"weighted_time"（默认，加权总在港时间）
            或 "total_time"（总在港时间，所有权重视为 1）

    返回:
        dict: 包含每艘船的分配结果及总等待时间；
              模型无解时返回 {"status": "infeasible"}
    """
    n = len(ships)
    if n == 0:
        return {"ships": [], "total_wait_time": 0}

    arrivals = [s["arrival"] for s in ships]
    services = [s["service"] for s in ships]
    names = [s["name"] for s in ships]
    # 船舶名 -> 下标 映射，用于解析按名字传入的依赖/限制/自定义约束
    name_to_idx = {name: i for i, name in enumerate(names)}

    # big-M 的取值：足够大以放松非绑定约束
    M = max(arrivals) + sum(services) + 1

    # 创建整数规划问题（目标函数在下方按 objective 参数设置）
    prob = pulp.LpProblem("Berth_Scheduling", pulp.LpMinimize)

    # ====== 决策变量 ======

    # x[i][b]: 二进制变量，船 i 是否分配到泊位 b
    x = [[pulp.LpVariable(f"x_{i}_{b}", cat="Binary") for b in range(num_berths)]
         for i in range(n)]

    # s[i]: 整数变量，船 i 的开始服务时间
    s = [pulp.LpVariable(f"s_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # y[i][j]: 二进制变量，船 i 是否在船 j 之前开始服务
    # 用于 big-M 方法处理同一泊位上的服务不重叠约束
    y = [[pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(n)]
         for i in range(n)]

    # ====== 目标函数（可选模式） ======
    # 在港时间 = 离港 - 到达 = (s_i + svc_i) - arrival_i = 等待 + 作业。
    # 作业时长 svc_i 与调度方案无关（常数），因此最小化 Σ w_i*在港时间
    # 等价于最小化 Σ w_i*(s_i - arrival_i)，目标按加权等待时间书写。
    # - weighted_time（默认）：权重取 priority_weights 中该船的权重，未列出默认 1
    # - total_time：所有权重视为 1，即最小化总在港/总等待时间
    wait_time = [s[i] - arrivals[i] for i in range(n)]
    if objective == "weighted_time" and priority_weights:
        weights = [float(priority_weights.get(name, 1)) for name in names]
        prob += pulp.lpSum(weights[i] * wait_time[i] for i in range(n))
    else:
        prob += pulp.lpSum(wait_time)

    # ====== 约束条件 ======

    # 约束 1：每艘船必须分配且仅分配一个泊位
    for i in range(n):
        prob += pulp.lpSum(x[i][b] for b in range(num_berths)) == 1, f"assign_berth_{i}"

    # 约束 2：开始服务时间不能早于到达时间
    for i in range(n):
        prob += s[i] >= arrivals[i], f"no_early_start_{i}"

    # 约束 3：同一泊位上任意两艘船的服务时间不能重叠（big-M 方法）
    # 对每对船 (i, j) 其中 i < j，以及每个泊位 b：
    #   若 i 和 j 都在泊位 b，且 i 在 j 之前服务 => s[i] + svc[i] + gap <= s[j]
    #   若 i 和 j 都在泊位 b，且 j 在 i 之前服务 => s[j] + svc[j] + gap <= s[i]
    # 使用 big-M 放松非同一泊位或相对顺序相反时的约束
    # 若传入 min_gap > 0（岸桥连续性间隔），则在同泊位前后两船之间追加最小间隔：
    #   数学含义：同泊位上紧邻作业的两船，后船开始时间 >= 前船离港时间 + min_gap
    min_gap_val = float(min_gap or 0)
    for i in range(n):
        for j in range(i + 1, n):
            for b in range(num_berths):
                # i 先于 j 服务，或 i、j 不在同一泊位 b 时约束放松
                prob += (s[i] + services[i] + min_gap_val <= s[j]
                         + M * (1 - y[i][j])
                         + M * (2 - x[i][b] - x[j][b])), \
                    f"overlap_{i}_{j}_{b}_a"
                # j 先于 i 服务，或 i、j 不在同一泊位 b 时约束放松
                prob += (s[j] + services[j] + min_gap_val <= s[i]
                         + M * y[i][j]
                         + M * (2 - x[i][b] - x[j][b])), \
                    f"overlap_{i}_{j}_{b}_b"

    # 约束 4：潮汐约束 —— 分配到泊位 b 的船，作业不得落在该泊位的不可作业区间内
    # 输入 tide_windows[b] = [[t1,t2], ...]：泊位 b 的若干潮汐窗口 [t1, t2)
    # 数学含义：若船 i 分配到泊位 b（x[i][b]=1），其作业区间 [s_i, s_i+svc_i)
    #          必须整体避开每个窗口，即
    #          s_i + svc_i <= t1（窗口开始前完成）或 s_i >= t2（窗口结束后才开始）
    # 引入二进制变量 z 在两种情形中选择；x[i][b]=0 时两式均被 big-M 放松
    for b in range(num_berths):
        windows = tide_windows[b] if tide_windows and b < len(tide_windows) else []
        for k, window in enumerate(windows):
            t1, t2 = window
            for i in range(n):
                z = pulp.LpVariable(f"tide_{i}_{b}_{k}", cat="Binary")
                # 情形 A（z=0）：船在窗口开始前完成作业
                prob += (s[i] + services[i] <= t1
                         + M * (1 - x[i][b]) + M * z), \
                    f"tide_before_{i}_{b}_{k}"
                # 情形 B（z=1）：船在窗口结束后才开始作业
                prob += (s[i] >= t2
                         - M * (1 - x[i][b]) - M * (1 - z)), \
                    f"tide_after_{i}_{b}_{k}"

    # 约束 5：依赖关系 —— 被依赖船全部离港后，依赖它们的船才能开始作业
    # 输入 dependencies = {"C": ["A"], "E": ["B","D"], ...}
    # 数学含义：对每条依赖 j ← i：s_j >= s_i + svc_i
    #          （j 的开始时间不早于 i 的离港时间）
    for ship_name, prereq_names in (dependencies or {}).items():
        j = name_to_idx.get(ship_name)
        if j is None:
            continue  # 船名不在输入船舶列表中，忽略该条依赖
        for prereq_name in prereq_names:
            i = name_to_idx.get(prereq_name)
            if i is None:
                continue
            prob += s[j] >= s[i] + services[i], f"dependency_{i}_to_{j}"

    # 约束 6：靠泊限制 —— 某些船只能停靠指定泊位（0-based 泊位索引）
    # 输入 berth_restrictions = {"A": [0,1], ...}
    # 数学含义：对船 i 未被允许的泊位 b，强制 x[i][b] = 0
    for ship_name, allowed_berths in (berth_restrictions or {}).items():
        i = name_to_idx.get(ship_name)
        if i is None:
            continue
        allowed = set(allowed_berths)
        for b in range(num_berths):
            if b not in allowed:
                prob += x[i][b] == 0, f"berth_restriction_{i}_{b}"

    # 约束 7：自定义硬约束（当前支持 deadline 类型）
    # 输入 custom_constraints = [{"type": "deadline", "ship": "A", "time": 16.0}, ...]
    # deadline 数学含义：船必须在 time 之前离港，即 s_i + svc_i <= time
    for idx, cc in enumerate(custom_constraints or []):
        i = name_to_idx.get(cc.get("ship"))
        if cc.get("type") == "deadline" and i is not None:
            prob += s[i] + services[i] <= cc.get("time"), \
                f"custom_deadline_{i}_{idx}"

    # ====== 求解 ======
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        # 模型无解（如 deadline 过早、依赖成环等）时返回统一状态，而非抛错崩溃
        return {"status": "infeasible"}

    # ====== 提取结果 ======
    result_ships = []
    total_wait = 0

    for i in range(n):
        start_time = int(pulp.value(s[i]))
        end_time = start_time + services[i]
        wait = start_time - arrivals[i]
        total_wait += wait

        # 找出船 i 被分配到哪个泊位
        berth_assigned = None
        for b in range(num_berths):
            if pulp.value(x[i][b]) > 0.5:
                berth_assigned = b
                break

        result_ships.append({
            "name": names[i],
            "arrival": arrivals[i],
            "service": services[i],
            "berth": berth_assigned,
            "start_time": start_time,
            "end_time": end_time,
            "wait_time": wait
        })

    return {
        "ships": result_ships,
        "total_wait_time": total_wait
    }


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    HTTP API：接收 JSON 输入，返回泊位调度优化结果。
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    ships = data.get("ships", [])
    berths = data.get("berths", 1)

    if not ships:
        return jsonify({"error": "At least one ship is required"}), 400

    # 新增可选字段：JSON 中未提供时默认不启用对应约束
    result = optimize_berth_scheduling(
        ships,
        berths,
        tide_windows=data.get("tide_windows"),
        dependencies=data.get("dependencies"),
        priority_weights=data.get("priority_weights"),
        berth_restrictions=data.get("berth_restrictions"),
        min_gap=data.get("min_gap", 0.0),
        custom_constraints=data.get("custom_constraints"),
        objective=data.get("objective", "weighted_time"),
    )
    return jsonify(result)


def _print_result(result):
    """打印调度结果（本地测试用）。"""
    if result.get("status") == "infeasible":
        print("  模型无解 (infeasible)")
        return
    for s in result["ships"]:
        print(f"  船舶 {s['name']}: 泊位 {s['berth']}, "
              f"到达 {s['arrival']}, 开始 {s['start_time']}, "
              f"结束 {s['end_time']}, 等待 {s['wait_time']}")
    print(f"  总等待时间: {result['total_wait_time']}")


if __name__ == "__main__":
    # ====== 本地测试 1：基础功能（无额外约束，行为与原版一致） ======
    print("=" * 50)
    print("本地测试 1：3 艘船，2 个泊位（基础功能）")
    print("=" * 50)

    test_ships = [
        {"name": "A", "arrival": 1, "service": 2},
        {"name": "B", "arrival": 2, "service": 1},
        {"name": "C", "arrival": 3, "service": 3}
    ]
    test_berths = 2

    result = optimize_berth_scheduling(test_ships, test_berths)
    _print_result(result)

    # ====== 本地测试 2：包含全部新增字段的完整 JSON（走 /optimize 接口） ======
    print("\n" + "=" * 50)
    print("本地测试 2：完整 JSON（潮汐/依赖/权重/靠泊限制/间隔/硬约束/目标）")
    print("=" * 50)

    full_payload = {
        "ships": [
            {"name": "A", "arrival": 0, "service": 3},
            {"name": "B", "arrival": 1, "service": 2},
            {"name": "C", "arrival": 2, "service": 2},
            {"name": "D", "arrival": 3, "service": 2},
            {"name": "E", "arrival": 4, "service": 3},
            {"name": "F", "arrival": 5, "service": 2},
            {"name": "G", "arrival": 6, "service": 2}
        ],
        "berths": 3,
        # 岸桥数量：当前模型按固定作业时长求解，此字段暂不参与计算，仅作输入示例
        "berth_cranes": [3, 2, 2],
        # 泊位 0 不可作业区间 [9,11] 和 [15,17]，泊位 1 无，泊位 2 为 [12,14]
        "tide_windows": [[[9, 11], [15, 17]], [], [[12, 14]]],
        "priority_weights": {"A": 3, "B": 2, "C": 1, "D": 2, "E": 1, "F": 0.5, "G": 1},
        # C 需等 A 离港后才能开始；E 需等 B、D 均离港后才能开始
        "dependencies": {"C": ["A"], "E": ["B", "D"]},
        # A 只能停泊位 0/1，F 只能停泊位 1/2
        "berth_restrictions": {"A": [0, 1], "F": [1, 2]},
        # 同泊位前后两船间隔至少 1 小时
        "min_gap": 1.0,
        # A 必须在 16.0 之前离港
        "custom_constraints": [{"type": "deadline", "ship": "A", "time": 16.0}],
        # 目标函数：最小化加权总在港时间
        "objective": "weighted_time"
    }

    client = app.test_client()
    result = client.post("/optimize", json=full_payload).get_json()
    _print_result(result)

    # ====== 本地测试 3：无解场景（应返回 {"status": "infeasible"} 而非崩溃） ======
    print("\n" + "=" * 50)
    print("本地测试 3：无解场景（A 最早 3.0 才能离港，却要求 2.0 前离港）")
    print("=" * 50)

    infeasible_payload = dict(full_payload)
    infeasible_payload["custom_constraints"] = [
        {"type": "deadline", "ship": "A", "time": 2.0}
    ]
    result = client.post("/optimize", json=infeasible_payload).get_json()
    print(f"  返回结果: {result}")

    # ====== 启动 Flask 服务 ======
    print("\nFlask 服务启动在 http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

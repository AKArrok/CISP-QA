from retrieval.rank_features import fold_near_duplicate_pages, rerank_with_features


def test_feature_rerank_penalizes_overview_pages():
    overview = {
        "id": "overview",
        "score": 0.032,
        "source": "信息安全保障_V4.2",
        "title": "知识子域：信息安全保障框架",
        "text": "理解信息安全保障技术框架（IATF）的深度防御核心思想、三个核心要素及四个焦点领域。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "信息安全保障_V4.2",
        "title": "信息保障技术框架（IATF）",
        "text": "核心思想：深度防御。三个要素：人、技术、操作。四个焦点领域：保护网络和基础设施、保护区域边界、保护计算环境、支持性基础设施。",
    }

    ranked = rerank_with_features(
        "IATF纵深防御框架的三个核心和四个保护方面是什么？",
        [overview, answer],
    )

    assert ranked[0]["id"] == "answer"


def test_feature_rerank_keeps_clear_score_winner():
    strong = {"id": "strong", "score": 0.05, "source": "课件", "title": "AES", "text": "AES"}
    weak = {"id": "weak", "score": 0.01, "source": "课件", "title": "AES", "text": "AES 包括 128 192 256"}

    assert rerank_with_features("AES支持哪三种密钥长度？", [strong, weak])[0]["id"] == "strong"


def test_feature_rerank_boosts_composition_answer():
    generic = {
        "id": "generic",
        "score": 0.032,
        "source": "课件",
        "title": "入侵检测系统的作用与功能",
        "text": "入侵检测系统可以监测并分析用户和系统的活动。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "课件",
        "title": "入侵检测系统",
        "text": "组成：事件产生器、事件分析、事件响应、数据库。",
    }

    ranked = rerank_with_features("入侵检测系统通常由哪些组成部分构成？", [generic, answer])

    assert ranked[0]["id"] == "answer"


def test_feature_rerank_boosts_exact_title_phrase():
    generic = {
        "id": "generic",
        "score": 0.032,
        "source": "课件",
        "title": "系统安全工程能力成熟度模型",
        "text": "理解能力成熟度级别的概念。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "课件",
        "title": "能力级别-2级",
        "text": "规划和跟踪级：规划执行、规范化执行、跟踪执行、验证执行。",
    }

    ranked = rerank_with_features("SSE-CMM能力级别2级包含哪些公共特征？", [generic, answer])

    assert ranked[0]["id"] == "answer"


def test_feature_rerank_boosts_eal_level_enumeration():
    concept = {
        "id": "concept",
        "score": 0.032,
        "source": "信息安全评估_V4.2",
        "title": "CC的关键概念",
        "text": "EAL:评估保证级，GB/T 18336.3中预先定义的一些保证包。",
    }
    enumeration = {
        "id": "enumeration",
        "score": 0.031,
        "source": "CISP 4.2版本《信息安全评估》知识点总结",
        "title": "《信息安全评估》知识点串讲",
        "text": "CC 标准概念：评估对象 TOE、保护轮廓 PP、安全目标 ST、EAL（EAL1-EAL7）。",
    }

    ranked = rerank_with_features("CC标准中的评估保证级EAL如何划分？", [concept, enumeration])

    assert ranked[0]["id"] == "enumeration"


def test_feature_rerank_boosts_required_constraint_coverage():
    generic = {
        "id": "generic",
        "score": 0.032,
        "source": "课件",
        "title": "AAA协议",
        "text": "常见AAA协议包括RADIUS协议、TACACS+协议、Diameter协议。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "课件",
        "title": "认证、授权和计费",
        "text": "常见AAA协议包括RADIUS、TACACS+、Diameter，其中Diameter是RADIUS的升级版。",
    }

    ranked = rerank_with_features("AAA常见协议有哪些，RADIUS的升级版是什么？", [generic, answer])

    assert ranked[0]["id"] == "answer"


def test_feature_rerank_boosts_process_detail_over_adjacent_concept():
    concept = {
        "id": "concept",
        "score": 0.032,
        "source": "课件",
        "title": "信息安全风险管理基本过程",
        "text": "四个阶段：背景建立、风险评估、风险处理、批准监督。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "课件",
        "title": "风险评估过程",
        "text": "风险评估过程包括准备、风险要素识别、风险分析、风险结果判定。",
    }

    ranked = rerank_with_features("信息安全风险评估通常包括哪四个阶段？", [concept, answer])

    assert ranked[0]["id"] == "answer"


def test_feature_rerank_boosts_specific_level_detail():
    overview = {
        "id": "overview",
        "score": 0.032,
        "source": "课件",
        "title": "知识子域：系统安全工程",
        "text": "理解SSE-CMM能力成熟度级别的概念，掌握1~5级公共特征。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "课件",
        "title": "能力级别-2级",
        "text": "SSE-CMM能力级别2级是规划和跟踪级，包括规划执行、规范化执行、跟踪执行、验证执行。",
    }

    ranked = rerank_with_features("SSE-CMM能力级别2级包含哪些公共特征？", [overview, answer])

    assert ranked[0]["id"] == "answer"


def test_fold_near_duplicate_pages_only_cross_version_same_title():
    chunks = [
        {"id": "v42", "domain": "信息安全监管", "source": "网络安全监管_v4.2", "title": "网络安全等级保护政策"},
        {"id": "v41", "domain": "信息安全监管", "source": "信息安全监管_v4.1", "title": "网络安全等级保护政策"},
        {"id": "other", "domain": "信息安全监管", "source": "网络安全监管_v4.2", "title": "定级与备案"},
        {"id": "same_file", "domain": "信息安全监管", "source": "网络安全监管_v4.2", "title": "网络安全等级保护政策"},
    ]

    folded = fold_near_duplicate_pages(chunks)

    assert [chunk["id"] for chunk in folded] == ["v42", "other", "same_file"]


def test_fold_near_duplicate_pages_keeps_first_ranked():
    complete_v41 = {
        "id": "v41_full",
        "domain": "信息安全管理",
        "source": "信息安全管理_V4.1",
        "title": "信息安全风险管理基本过程",
        "text": "信息安全风险管理基本过程 | GB/Z 24364《信息安全风险管理指南》 | 四个阶段：背景建立、风险评估、"
                "风险处理、批准监督；两个贯穿：监控审查、沟通咨询。四个阶段覆盖风险管理全过程。",
    }
    complete_v42 = {
        "id": "v42_full",
        "domain": "信息安全管理",
        "source": "信息安全管理_V4.2",
        "title": "信息安全风险管理基本过程",
        "text": "信息安全风险管理基本过程 | GB/Z 24364《信息安全风险管理指南》 | 四个阶段：背景建立、风险评估、"
                "风险处理、批准监督；两个贯穿：监控审查、沟通咨询。按 GB/Z 24364 的风险管理框架展开。",
    }

    # 两个完整页互为等价答案：折叠只保留排名最高者，不做长短取舍。
    folded = fold_near_duplicate_pages([complete_v41, complete_v42])
    assert [chunk["id"] for chunk in folded] == ["v41_full"]

    folded = fold_near_duplicate_pages([complete_v42, complete_v41])
    assert [chunk["id"] for chunk in folded] == ["v42_full"]


def test_fold_swaps_out_stub_page_for_fuller_version():
    stub = {
        "id": "v41_stub",
        "domain": "信息安全管理",
        "source": "信息安全管理_V4.1",
        "title": "信息安全风险管理基本过程",
        "text": "信息安全风险管理基本过程 | GB/Z 24364《信息安全风险管理指南》 | 四个阶段 | 两个贯穿",
    }
    full = {
        "id": "v42_full",
        "domain": "信息安全管理",
        "source": "信息安全管理_V4.2",
        "title": "信息安全风险管理基本过程",
        "text": "信息安全风险管理基本过程 | GB/Z 24364《信息安全风险管理指南》 | 四个阶段：背景建立、风险评估、"
                "风险处理、批准监督；两个贯穿：监控审查、沟通咨询。",
    }

    # 残页即使排名靠前，折叠时也让位给同标题内容完整的版本。
    folded = fold_near_duplicate_pages([stub, full])
    assert [chunk["id"] for chunk in folded] == ["v42_full"]

    # 完整页在先时残页直接被折叠丢弃。
    folded = fold_near_duplicate_pages([full, stub])
    assert [chunk["id"] for chunk in folded] == ["v42_full"]


def test_feature_rerank_downgrades_stub_page():
    stub = {
        "id": "stub",
        "score": 0.032,
        "source": "信息安全管理_V4.1",
        "title": "信息安全风险管理基本过程",
        "text": "信息安全风险管理基本过程 | GB/Z 24364《信息安全风险管理指南》 | 四个阶段 | 两个贯穿",
    }
    full = {
        "id": "full",
        "score": 0.0315,
        "source": "信息安全管理_V4.2",
        "title": "信息安全风险管理基本过程",
        "text": "信息安全风险管理基本过程 | GB/Z 24364《信息安全风险管理指南》 | 四个阶段：背景建立、风险评估、"
                "风险处理、批准监督；两个贯穿：监控审查、沟通咨询。",
    }

    ranked = rerank_with_features(
        "GB/Z 24364信息安全风险管理包括哪四个阶段和哪两个贯穿过程？",
        [stub, full],
    )

    assert ranked[0]["id"] == "full"


def test_fold_never_swaps_two_complete_pages():
    # 两个完整页（均 >80 字）互为等价答案：折叠只看排名，不做长短取舍。
    v41 = {
        "id": "v41_full",
        "domain": "物理与网络通信安全",
        "source": "物理与网络通信安全_V4.1",
        "title": "无线局域网安全协议-WPA、WPA2",
        "text": "802.11i运行通常包括发现、认证、密钥管理和数据加密四个阶段：发现阶段通过信标与探测确定 AP 及"
                "安全能力，认证阶段完成身份验证，密钥管理阶段协商配对密钥，数据加密阶段保护无线数据传输。",
    }
    v42 = {
        "id": "v42_full",
        "domain": "物理与网络通信安全",
        "source": "物理与网络通信安全_V4.2",
        "title": "无线局域网安全协议-WPA、WPA2",
        "text": "无线局域网安全协议-WPA、WPA2 | 802.11i运行阶段：发现阶段、认证阶段、密钥管理阶段、数据加密阶段，"
                "各阶段的交互过程与密钥协商细节。",
    }

    folded = fold_near_duplicate_pages([v41, v42])
    assert [chunk["id"] for chunk in folded] == ["v41_full"]


def test_multi_entity_bonus_prefers_comparison_page():
    comparison = {
        "id": "compare",
        "score": 0.031,
        "source": "软件安全开发_V4.2",
        "title": "各模型比较",
        "text": "SDL文档丰富适合大型企业；CLASP轻量级过程适合小型企业；SAMM开放框架安全知识要求较低。",
    }
    single = {
        "id": "single",
        "score": 0.032,
        "source": "软件安全开发_V4.1",
        "title": "CLASP",
        "text": "CLASP综合的轻量应用安全过程，由30个特定的活动构成的集合。",
    }

    ranked = rerank_with_features(
        "SDL、CLASP和SAMM各自有什么特点，分别适合什么场景？",
        [single, comparison],
    )

    assert ranked[0]["id"] == "compare"


def test_acronym_title_bonus_matches_expansion_in_title():
    from retrieval.rank_features import _acronym_title_bonus

    # 标题没有缩写本身，但含 query 中紧邻缩写的中文全称，应视同命中。
    assert _acronym_title_bonus({"IATF"}, "信息保障技术框架-安全原则与特点",
                                "IATF的信息保障技术框架有哪些安全原则和特点？") == 0.004
    assert _acronym_title_bonus({"IATF"}, "知识子域：信息安全保障框架",
                                "IATF的信息保障技术框架有哪些安全原则和特点？") == 0.0


def test_constraint_bonus_for_grading_process_page():
    workflow = {
        "id": "workflow",
        "score": 0.032,
        "source": "信息安全监管_v4.1",
        "title": "等级保护工作流程",
        "text": "等级保护工作流程分为定级、备案、建设整改、等级测评、监督检查五个环节。",
    }
    answer = {
        "id": "answer",
        "score": 0.031,
        "source": "网络安全监管_v4.2",
        "title": "定级与备案",
        "text": "定级与备案：确定定级对象，综合评定对客体的侵害程度，定级对象的安全保护等级由业务信息安全和系统服务安全等级确定。",
    }

    ranked = rerank_with_features(
        "等级保护定级的一般流程是什么，定级对象的安全保护等级如何确定？",
        [workflow, answer],
    )

    assert ranked[0]["id"] == "answer"


def test_context_continuity_boosts_adjacent_page():
    """多轮追问：上一轮命中页的邻域页小幅加成，赢下近平局。

    两页正文特征对称（与查询词的重叠一致），胜负只由 base 分差与
    邻域加成决定——加成 0.0015 恰好翻转 0.0005 的 base 劣势。
    """
    neighbor = {
        "id": "neighbor",
        "score": 0.0315,
        "source": "业务连续性_V4.2",
        "page": 11,  # 上一轮命中 p9，+2 页的邻域页
        "title": "确定业务优先级",
        "text": "本节给出相应的度量方法与判定依据。",
    }
    unrelated = {
        "id": "unrelated",
        "score": 0.032,
        "source": "业务连续性_V4.2",
        "page": 46,
        "title": "恢复点目标（RPO）",
        "text": "本节给出相应的度量方法与判定依据。",
    }

    ranked = rerank_with_features(
        "业务影响分析和RTO是什么关系？", [unrelated, neighbor],
        context_pages=[("业务连续性_V4.2", 9)],
    )
    assert ranked[0]["id"] == "neighbor"

    # 无上一轮上下文（单轮）时加成不生效，base 分高的 unrelated 胜出
    ranked = rerank_with_features(
        "业务影响分析和RTO是什么关系？", [unrelated, neighbor], context_pages=None,
    )
    assert ranked[0]["id"] == "unrelated"


def test_context_continuity_ignores_far_pages():
    """页码距离超过 3 页不享受邻域加成，防止会话黏死在单一章节。"""
    from retrieval.rank_features import _context_continuity_bonus

    chunk = {"source": "业务连续性_V4.2", "page": 46}
    assert _context_continuity_bonus(chunk, [("业务连续性_V4.2", 9)]) == 0.0
    assert _context_continuity_bonus(chunk, [("业务连续性_V4.2", 45)]) == 0.0015
    # 已读页本身（Δ=0）不加成：重读加成会挤占邻域新页的名额
    assert _context_continuity_bonus(chunk, [("业务连续性_V4.2", 46)]) == 0.0
    assert _context_continuity_bonus(chunk, None) == 0.0

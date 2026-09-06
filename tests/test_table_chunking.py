"""表格感知切片的单元测试：过滤、序列化、语境注入与父子关联。"""
import os
import unittest

import config
from data_ingest.chunking import build_structured_chunks
from data_ingest.tables import (
    body_text_without_tables,
    char_coverage,
    clean_cell,
    is_meaningful_table,
    split_table_markdown,
    to_markdown,
)


def _page(**overrides) -> dict:
    page = {
        "domain": "信息安全评估",
        "source": "测试课件",
        "page": 1,
        "kind": "courseware",
        "text": "风险等级划分\n风险矩阵用于判定风险等级。",
    }
    page.update(overrides)
    return page


class TestQualityFilter(unittest.TestCase):
    def test_real_table_passes(self):
        rows = [
            ["等级", "取值范围", "名称"],
            ["H", "25, 20", "高风险"],
            ["M", "6, 8, 9, 10", "中等风险"],
            ["L", "1, 2, 3, 4, 5", "低风险"],
        ]
        self.assertTrue(is_meaningful_table(rows))

    def test_decorative_boxes_fail_on_filled_cells(self):
        # 版式装饰框：单元格多但几乎全空
        self.assertFalse(is_meaningful_table([[""] * 13 for _ in range(13)]))

    def test_side_by_side_matrix_fails_on_width(self):
        # 并排多表误检：列数超上限即拒绝
        rows = [["a"] * 15 for _ in range(12)]
        self.assertFalse(is_meaningful_table(rows))

    def test_wide_merged_header_diagram_gets_ratio_waiver(self):
        # IP/TCP 报头示意图类：合并单元格多、密度低，列数少时豁免密度要求
        rows = [["版本", "包头长度", "服务类型", "数据包长度", ""],
                ["标识", "", "", "标记", "偏移"],
                ["生存期", "", "协议类型", "包头校验和", ""],
                ["源IP地址", "", "", "", ""],
                ["目的IP地址", "", "", "", ""],
                ["可选项", "", "", "", ""],
                ["用户数据", "", "", "", ""]]
        self.assertTrue(is_meaningful_table(rows))


class TestMarkdown(unittest.TestCase):
    def test_to_markdown_shape_and_cell_cleaning(self):
        md = to_markdown([["类别", "描述"], ["基础环 境类", "\uf06c 窃听\n\uf06c 篡改"]])
        lines = md.splitlines()
        self.assertEqual(lines[0], "| 类别 | 描述 |")
        self.assertTrue(lines[1].startswith("|---"))
        self.assertIn("| 基础环境类 | •窃听•篡改 |", lines[2])

    def test_clean_cell_english_keeps_spaces(self):
        self.assertEqual(clean_cell("Denial of Service"), "Denial of Service")

    def test_char_coverage_detects_dropped_text(self):
        rows = [["GB/T 22239", ""]]
        region = "GB/T 22239.2 第2部分 GB/T 22239.3 第3部分"
        self.assertLess(char_coverage(rows, region), 0.75)
        self.assertGreaterEqual(char_coverage(rows, "GB/T 22239"), 0.75)

    def test_split_repeats_header_line(self):
        md = "| 列1 | 列2 |\n|---|---|\n" + "\n".join(
            f"| 行{i} | 内容内容内容内容内容内容内容 |" for i in range(10)
        )
        parts = split_table_markdown(md, md.splitlines()[0], 120)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertTrue(part.startswith("| 列1 | 列2 |"))
            self.assertLessEqual(len(part), 120)


class TestStructuredTableChunks(unittest.TestCase):
    def test_table_child_shares_parent_and_injects_context(self):
        table_md = "| 等级 | 名称 |\n|---|---|\n| H | 高风险 |\n| L | 低风险 |"
        pages = [_page(
            body_text="风险等级划分\n风险矩阵用于判定风险等级。",
            tables=[{"bbox": [0, 0, 1, 1], "caption": "风险等级对照", "markdown": table_md}],
        )]
        chunks = build_structured_chunks(pages, max_chars=420)
        table_chunks = [c for c in chunks if c["element"] == "table"]
        text_chunks = [c for c in chunks if c["element"] == "text"]
        self.assertEqual(len(table_chunks), 1)
        self.assertTrue(text_chunks)
        table = table_chunks[0]
        # 语境注入：表题 + 知识点标题 + 知识域都进入 embedding 文本
        self.assertIn("【表格】风险等级对照", table["child_text"])
        self.assertIn("知识点：风险等级划分", table["embedding_text"])
        self.assertIn("知识域：信息安全评估", table["embedding_text"])
        # 跨元素关联：表格 Child 与正文 Child 共享同一 Parent，且 Parent 同时含两者
        self.assertEqual(table["parent_id"], text_chunks[0]["parent_id"])
        self.assertIn("风险矩阵用于判定风险等级", table["text"])
        self.assertIn("| H | 高风险 |", table["text"])

    def test_all_table_page_still_produces_chunks(self):
        pages = [_page(
            text="风险等级",
            body_text="风险等级",
            tables=[{"bbox": [0, 0, 1, 1], "caption": "", "markdown": "| a |\n|---|\n| b |"}],
        )]
        chunks = build_structured_chunks(pages)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["element"], "table")

    def test_long_table_splits_with_repeated_header(self):
        rows = "| 等级 | 描述 |\n|---|---|\n" + "\n".join(
            f"| L{i} | {'描述内容' * 10} |" for i in range(12)
        )
        pages = [_page(
            body_text="等级说明",
            tables=[{"bbox": [0, 0, 1, 1], "caption": "等级表", "markdown": rows}],
        )]
        chunks = build_structured_chunks(pages, max_chars=420)
        table_chunks = [c for c in chunks if c["element"] == "table"]
        self.assertGreater(len(table_chunks), 1)
        for chunk in table_chunks:
            self.assertTrue(chunk["child_text"].startswith("【表格】等级表\n| 等级 | 描述 |"))
            self.assertLessEqual(len(chunk["child_text"]), 420)
        # 所有分段仍指向同一个 Parent
        self.assertEqual({c["parent_id"] for c in table_chunks}, {table_chunks[0]["parent_id"]})

    def test_page_without_tables_is_unchanged(self):
        chunks = build_structured_chunks([_page()])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["element"], "text")
        self.assertEqual(chunks[0]["text"], "风险等级划分\n风险矩阵用于判定风险等级。")
        self.assertNotIn("【表格】", chunks[0]["child_text"])


class TestBodyReconstruction(unittest.TestCase):
    def test_blocks_inside_table_bbox_are_dropped(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("pymupdf 未安装")
        body = body_text_without_tables(
            [
                (10, 10, 200, 30, "幻灯片标题", 0, 0),
                (20, 100, 480, 200, "表内单元格文本", 1, 0),
                (20, 260, 480, 300, "表格下方的正文说明", 2, 0),
            ],
            [pymupdf.Rect(15, 95, 485, 205)],
        )
        self.assertNotIn("表内单元格文本", body)
        self.assertIn("幻灯片标题", body)
        self.assertIn("表格下方的正文说明", body)


@unittest.skipUnless(os.path.isdir(config.COURSE_DIR), "本机无课件 PDF，跳过真实文件冒烟")
class TestRealPdfExtraction(unittest.TestCase):
    def test_known_table_page_extracts_with_caption(self):
        import glob

        import pymupdf

        from data_ingest.tables import extract_page_tables
        matches = glob.glob(os.path.join(config.COURSE_DIR, "**", "业务连续性_V4.2.pdf"), recursive=True)
        if not matches:
            self.skipTest("业务连续性_V4.2.pdf 不存在")
        doc = pymupdf.open(matches[0])
        tables = extract_page_tables(doc[30])  # p31 应急预案分类表
        doc.close()
        self.assertTrue(tables)
        markdown = tables[0]["markdown"]
        self.assertIn("基础环境类", markdown.replace(" ", ""))
        self.assertTrue(tables[0]["markdown"].startswith("|"))


if __name__ == "__main__":
    unittest.main()

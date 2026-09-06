"""BM25 数字检索盲区修复的单元测试：个位数保留 + 中文序数等级归一。"""
import unittest

from retrieval.bm25 import BM25Index, normalize_cn_levels, tokenize


class TestTokenizeNumeric(unittest.TestCase):
    def test_single_digit_survives(self):
        # 修复前："赋值为5分" → ['赋值','为','分']，数字被当噪声丢弃
        self.assertIn("5", tokenize("赋值为5分"))
        self.assertIn("3", tokenize("赋值3代表中"))

    def test_single_punctuation_still_dropped(self):
        self.assertEqual(tokenize("，。！？"), [])
        self.assertNotIn(",", tokenize("25,20"))

    def test_multi_digit_numbers_kept(self):
        tokens = tokenize("GB/T 22239 系列")
        self.assertIn("22239", tokens)


class TestCnLevelNormalization(unittest.TestCase):
    def test_three_writing_forms_converge(self):
        # 第三级（文档整 token）/ 三级 / 3级（jieba 拆开）必须产出相同词面
        self.assertEqual(tokenize("第三级"), tokenize("三级"))
        self.assertEqual(tokenize("第三级"), tokenize("3级"))

    def test_compound_numerals(self):
        self.assertEqual(normalize_cn_levels("第十一级"), "11级")
        self.assertEqual(normalize_cn_levels("十五级"), "15级")
        self.assertEqual(normalize_cn_levels("二十级"), "20级")

    def test_confounder_words_untouched(self):
        # "等级/分级/定级" 前面不是数字，不受影响
        self.assertEqual(normalize_cn_levels("等级保护与分级保护"), "等级保护与分级保护")
        self.assertEqual(normalize_cn_levels("定级和备案"), "定级和备案")
        # "统一/万一" 这类含数字的词没有被误转
        self.assertEqual(normalize_cn_levels("统一管理"), "统一管理")
        self.assertEqual(normalize_cn_levels("万一发生"), "万一发生")


class TestBM25NumericBridge(unittest.TestCase):
    def _index(self):
        chunks = [
            {"id": "doc_level", "text": "受侵害的客体是社会秩序公共利益，受到严重损害时定为第三级。",
             "embedding_text": "受侵害的客体是社会秩序公共利益，受到严重损害时定为第三级。"},
            {"id": "doc_freq", "text": "威胁出现频率赋值为5代表很高。",
             "embedding_text": "威胁出现频率赋值为5代表很高。"},
            {"id": "doc_decoy", "text": "机房应当建立门禁系统和视频监控系统。",
             "embedding_text": "机房应当建立门禁系统和视频监控系统。"},
        ]
        return BM25Index.build(chunks)

    def test_query_with_arabic_level_finds_chinese_numeral_doc(self):
        index = self._index()
        results = index.search("社会秩序受严重损害应定为3级", 3)
        self.assertEqual(results[0]["id"], "doc_level")

    def test_query_with_single_digit_finds_value_doc(self):
        index = self._index()
        results = index.search("威胁出现频率赋值为5是什么水平", 3)
        self.assertEqual(results[0]["id"], "doc_freq")


if __name__ == "__main__":
    unittest.main()

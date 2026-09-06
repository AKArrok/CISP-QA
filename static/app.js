/* CISP 备考助手前端：问答（SSE）+ 刷题 + 统计 */
const $ = (id) => document.getElementById(id);
const threadId = "web-" + Date.now();

/* ── 页签切换 ── */
function switchTab(chat) {
  $("chat-panel").classList.toggle("hidden", !chat);
  $("quiz-panel").classList.toggle("hidden", chat);
  $("tab-chat").classList.toggle("active", chat);
  $("tab-quiz").classList.toggle("active", !chat);
  if (!chat) loadStats();
}
$("tab-chat").onclick = () => switchTab(true);
$("tab-quiz").onclick = () => switchTab(false);

/* ══════════ 问答 ══════════ */
function renderMarkdown(text) {
  // 先转义 HTML 防注入，再做有限 Markdown 转换（加粗/行内代码/列表/分隔线/换行）
  let t = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/^---+$/gm, "<hr>");
  t = t.replace(/^[-*] (.+)$/gm, "&bull; $1");
  t = t.replace(/\n/g, "<br>");
  return t;
}

function addBubble(cls, text) {
  const div = document.createElement("div");
  div.className = "bubble " + cls;
  div.textContent = text;
  $("chat-history").appendChild(div);
  $("chat-history").scrollTop = $("chat-history").scrollHeight;
  return div;
}

async function sendChat() {
  const input = $("chat-input");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  addBubble("user", q);
  const bot = addBubble("bot", "");
  $("chat-send").disabled = true;

  const sources = [];
  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, thread_id: threadId }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let full = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "token") {
          full += ev.content;
          // 剥离模型自带的出处行（页面底部统一渲染来源，避免重复）
          bot.innerHTML = renderMarkdown(full.replace(/\n?【出处】[^\n]*$/, ""));
        }
        else if (ev.type === "contexts") {
          ev.contexts.forEach(c => sources.push(`《${c.source}》第${c.page}页`));
        }
        $("chat-history").scrollTop = $("chat-history").scrollHeight;
      }
    }
    bot.innerHTML = renderMarkdown(full.replace(/\n?【出处】[^\n]*$/, ""));
  } catch (e) {
    bot.textContent += "\n[连接出错: " + e.message + "]";
  }
  if (sources.length) {
    const ctx = document.createElement("div");
    ctx.className = "ctx";
    ctx.textContent = "【出处】" + [...new Set(sources)].slice(0, 4).join("；");
    bot.appendChild(ctx);
  }
  $("chat-send").disabled = false;
  $("chat-input").focus();
}
$("chat-send").onclick = sendChat;
$("chat-input").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });

/* ══════════ 刷题 ══════════ */
let currentQuestion = null;
let answered = false;

async function initDomains() {
  try {
    const domains = await (await fetch("/api/domains")).json();
    for (const d of domains) {
      const opt = document.createElement("option");
      opt.value = d; opt.textContent = d;
      $("quiz-domain").appendChild(opt);
    }
  } catch (e) { /* 忽略 */ }
}

async function nextQuestion() {
  const mode = $("quiz-mode").value;
  const domain = $("quiz-domain").value || null;
  $("quiz-next").disabled = true;
  $("quiz-question").classList.remove("hidden");
  $("q-feedback").classList.add("hidden");
  $("q-options").innerHTML = "";
  $("q-stem").textContent = "加载中…";
  answered = false;
  try {
    const resp = await fetch("/quiz/next", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, domain }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    renderQuestion(await resp.json());
  } catch (e) {
    $("q-stem").textContent = "出题失败: " + e.message;
  }
  $("quiz-next").disabled = false;
}

async function choose(letter, el) {
  if (answered || !currentQuestion) return;
  answered = true;
  document.querySelectorAll(".option").forEach(o => o.classList.add("locked"));
  el.classList.add("chosen");
  let result;
  try {
    const resp = await fetch("/quiz/answer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: currentQuestion.id, choice: letter }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    result = await resp.json();
  } catch (e) {
    $("q-stem").textContent += "\n[判分失败: " + e.message + "]";
    return;
  }
  // 正确答案高亮
  for (const [l, text] of Object.entries(currentQuestion.options)) {
    if (result.answer.includes(l)) {
      document.querySelectorAll(".option")[ "ABCD".indexOf(l) ]?.classList.add("right");
    }
  }
  if (!result.correct) el.classList.add("wrong");
  $("q-verdict").textContent = result.correct ? "✅ 回答正确" : "❌ 回答错误";
  $("q-verdict").className = result.correct ? "ok" : "bad";
  $("q-analysis").textContent = "【解析】" + result.analysis;
  $("q-related").innerHTML = (result.related_chunks || []).map(c =>
    `<div class="rel">《${c.source}》第${c.page}页 · ${c.domain}<br>${c.text}</div>`).join("")
    || "<div class='rel'>（未找到关联课件）</div>";
  $("q-feedback").classList.remove("hidden");
}

/* 答题后闭环：把错题带进问答页深挖 */
function deepDive() {
  if (!currentQuestion) return;
  const wrongOpt = document.querySelector(".option.chosen");
  const chosen = wrongOpt ? wrongOpt.textContent.trim().slice(0, 120) : "";
  const q = `关于这道题我答错了，请帮我讲透背后的知识点，并指出我可能混淆的概念：
「${currentQuestion.stem}」
我的选择：${chosen || "（见上）"}
请先解释正确答案为什么对，再逐个分析易混淆的干扰项。`;
  switchTab(true);
  $("chat-input").value = q;
  sendChat();
}

/* 再练一道同类题：以当前题为锚点定向生成 */
async function similarQuestion() {
  if (!currentQuestion) return;
  $("quiz-next").disabled = true;
  $("quiz-question").classList.remove("hidden");
  $("q-feedback").classList.add("hidden");
  $("q-options").innerHTML = "";
  $("q-stem").textContent = "正在围绕同一考点出新题（约需十几秒）…";
  answered = false;
  try {
    const resp = await fetch("/quiz/next", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "weak", domain: currentQuestion.domain,
                             anchor_question_id: currentQuestion.id }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    renderQuestion(await resp.json());
  } catch (e) {
    $("q-stem").textContent = "出题失败: " + e.message;
  }
  $("quiz-next").disabled = false;
}

function renderQuestion(q) {
  currentQuestion = q;
  $("q-domain").textContent = q.domain || "未分类";
  $("q-source").textContent = q.source;
  $("q-stem").textContent = q.stem;
  $("q-options").innerHTML = "";
  for (const [letter, text] of Object.entries(q.options)) {
    const div = document.createElement("div");
    div.className = "option";
    div.textContent = `${letter}. ${text}`;
    div.onclick = () => choose(letter, div);
    $("q-options").appendChild(div);
  }
}

function confLabel(conf) {
  if (conf === null || conf === undefined) return "—";
  return conf < 0.5 ? "低" : conf < 0.8 ? "中" : "高";
}

async function loadStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    if (!s.total_attempts) {
      $("stats-summary").textContent = "还没有答题记录，先来做几道题吧！";
      return;
    }
    let summary = `累计答题 ${s.total_attempts} 道，总正确率 ${(s.total_accuracy * 100).toFixed(1)}%`;
    if (s.weak_domains.length) summary += `；薄弱域：${s.weak_domains.join("、")}`;
    if (s.review_due_domains.length) summary += `；复习到期：${s.review_due_domains.join("、")}`;
    if (s.review_due_cards) summary += `；待复习错题 ${s.review_due_cards} 道`;
    if (s.ask_unverified_domains && s.ask_unverified_domains.length)
      summary += `；你反复提问但还没刷题验证：${s.ask_unverified_domains.join("、")}`;
    $("stats-summary").textContent = summary;
    const tbody = $("stats-table").querySelector("tbody");
    tbody.innerHTML = "";
    for (const d of s.domains) {
      const tr = document.createElement("tr");
      const masteryPct = d.mastery === null || d.mastery === undefined ? "—" : (d.mastery * 100).toFixed(0) + "%";
      const status = d.weak ? "⚠ 薄弱"
        : d.ask_unverified ? "❓ 提问多·未验证"
        : d.insufficient ? "样本不足"
        : d.due_for_review ? "复习到期"
        : "正常";
      const statusCls = d.weak ? "weak" : d.due_for_review ? "review" : "";
      tr.innerHTML = `<td>${d.domain}</td><td>${d.attempts}</td><td>${d.asks || 0}</td>
        <td><span class="bar" style="width:${d.mastery ? d.mastery * 80 : 0}px"></span>${masteryPct}</td>
        <td>${confLabel(d.confidence)}</td>
        <td class="${statusCls}">${status}</td>`;
      tbody.appendChild(tr);
    }
  } catch (e) { /* 忽略 */ }
}

$("quiz-next").onclick = nextQuestion;
$("q-deep-dive").onclick = deepDive;
$("q-similar").onclick = similarQuestion;
initDomains();
loadStats();

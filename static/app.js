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
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "token") bot.textContent += ev.content;
        else if (ev.type === "contexts") {
          ev.contexts.forEach(c => sources.push(`《${c.source}》第${c.page}页`));
        }
        $("chat-history").scrollTop = $("chat-history").scrollHeight;
      }
    }
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
    currentQuestion = await resp.json();
    $("q-domain").textContent = currentQuestion.domain || "未分类";
    $("q-source").textContent = currentQuestion.source;
    $("q-stem").textContent = currentQuestion.stem;
    for (const [letter, text] of Object.entries(currentQuestion.options)) {
      const div = document.createElement("div");
      div.className = "option";
      div.textContent = `${letter}. ${text}`;
      div.onclick = () => choose(letter, div);
      $("q-options").appendChild(div);
    }
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

async function loadStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    $("stats-summary").textContent = s.total_attempts
      ? `累计答题 ${s.total_attempts} 道，总正确率 ${(s.total_accuracy * 100).toFixed(1)}%` +
        (s.weak_domains.length ? `；薄弱域：${s.weak_domains.join("、")}` : "")
      : "还没有答题记录，先来做几道题吧！";
    const tbody = $("stats-table").querySelector("tbody");
    tbody.innerHTML = "";
    for (const d of s.domains) {
      const tr = document.createElement("tr");
      const pct = d.accuracy === null ? "—" : (d.accuracy * 100).toFixed(0) + "%";
      tr.innerHTML = `<td>${d.domain}</td><td>${d.attempts}</td>
        <td><span class="bar" style="width:${d.accuracy ? d.accuracy * 80 : 0}px"></span>${pct}</td>
        <td class="${d.weak ? "weak" : ""}">${d.weak ? "⚠ 薄弱" : "正常"}</td>`;
      tbody.appendChild(tr);
    }
  } catch (e) { /* 忽略 */ }
}

$("quiz-next").onclick = nextQuestion;
initDomains();
loadStats();

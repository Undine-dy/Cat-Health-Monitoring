const state = {
  sessionId: null,
};

const wellnessUserId = document.getElementById("wellnessUserId");
const wellnessFile = document.getElementById("wellnessFile");
const wellnessInput = document.getElementById("wellnessInput");
const wellnessStatus = document.getElementById("wellnessStatus");
const generateReport = document.getElementById("generateReport");
const wellnessResult = document.getElementById("wellnessResult");
const responseText = document.getElementById("responseText");
const overviewText = document.getElementById("overviewText");
const flagList = document.getElementById("flagList");
const reportText = document.getElementById("reportText");
const chatHistory = document.getElementById("chatHistory");
const chatQuestion = document.getElementById("chatQuestion");
const sendChat = document.getElementById("sendChat");
const downloadRecord = document.getElementById("downloadRecord");
const loadProjectSample = document.getElementById("loadProjectSample");
const loadSimpleSample = document.getElementById("loadSimpleSample");

const fallInput = document.getElementById("fallInput");
const fallStatus = document.getElementById("fallStatus");
const detectFall = document.getElementById("detectFall");
const fallResult = document.getElementById("fallResult");
const fallReaction = document.getElementById("fallReaction");
const fallActivity = document.getElementById("fallActivity");
const fallState = document.getElementById("fallState");
const fallDetails = document.getElementById("fallDetails");

const simpleWellnessSample = {
  user_id: "cat_demo_user",
  sleep_hours: 5.8,
  exercise_minutes: 16,
  sedentary_minutes: 390,
  water_ml: 1000,
  stress_score: 0.74,
  resting_heart_rate: 92,
  notes: "最近连续赶进度，凌晨两点后才睡，白天有点心慌。",
};

const simpleFallSample = {
  user_id: "cat_fall_user",
  activity_label: "LAYING",
  timestamp: new Date().toISOString(),
};

function setStatus(element, text, error = false) {
  element.textContent = text;
  element.style.color = error ? "#b42318" : "rgba(44, 36, 28, 0.7)";
}

function safeJsonParse(text) {
  return JSON.parse(text);
}

function renderMarkdown(md) {
  if (!md) return "";
  let html = md
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  // horizontal rules
  html = html.replace(/^---+$/gm, '<hr class="report-hr">');
  // headings
  html = html.replace(/^####\s+(.+)$/gm, '<h4 class="report-h4">$1</h4>');
  html = html.replace(/^###\s+(.+)$/gm, '<h3 class="report-h3">$1</h3>');
  html = html.replace(/^##\s+(.+)$/gm, '<h2 class="report-h2">$1</h2>');
  // bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // list items
  html = html.replace(/^[-*]\s+(.+)$/gm, '<li class="report-li">$1</li>');
  html = html.replace(/((?:<li class="report-li">.*<\/li>\n?)+)/g, '<ul class="report-ul">$1</ul>');
  // numbered list items
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li class="report-li">$1</li>');
  // paragraphs: wrap remaining non-empty, non-tag lines
  html = html.replace(/^(?!<[hulo\-lr])((?!<).+)$/gm, '<p class="report-p">$1</p>');
  // clean up extra blank lines
  html = html.replace(/\n{3,}/g, "\n\n");
  return html;
}

function renderFlags(flags) {
  flagList.innerHTML = "";
  (flags || []).forEach((flag) => {
    const item = document.createElement("span");
    item.className = "flag";
    item.textContent = flag;
    flagList.appendChild(item);
  });
}

function appendChat(role, text) {
  const item = document.createElement("div");
  item.className = `chat-item ${role}`;
  item.textContent = text;
  chatHistory.appendChild(item);
}

function fillSimpleSamples() {
  wellnessUserId.value = simpleWellnessSample.user_id;
  wellnessInput.value = JSON.stringify(simpleWellnessSample, null, 2);
  fallInput.value = JSON.stringify(simpleFallSample, null, 2);
}

async function loadProjectDemo() {
  setStatus(wellnessStatus, "正在载入项目样例...");
  try {
    const response = await fetch("/wellness/demo-user/ID_010");
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "样例加载失败");
    }
    wellnessUserId.value = body.person_id || "ID_010";
    wellnessInput.value = JSON.stringify(body, null, 2);
    setStatus(wellnessStatus, "项目样例已载入");
  } catch (error) {
    setStatus(wellnessStatus, error.message, true);
  }
}

async function handleWellnessFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  wellnessInput.value = text;
  setStatus(wellnessStatus, `已载入文件：${file.name}`);
}

async function submitWellnessReport() {
  let data;
  try {
    data = safeJsonParse(wellnessInput.value);
  } catch (error) {
    setStatus(wellnessStatus, "用户数据 JSON 解析失败", true);
    return;
  }

  const payload = {
    user_id: wellnessUserId.value.trim() || "cat_user",
    data,
  };

  generateReport.disabled = true;
  downloadRecord.disabled = true;
  sendChat.disabled = true;
  setStatus(wellnessStatus, "正在生成报告...");

  try {
    const response = await fetch("/wellness/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "报告生成失败");
    }

    state.sessionId = body.session_id;
    responseText.textContent = body.response || "";
    overviewText.textContent = body.overview || "";
    renderFlags(body.flags || []);
    reportText.innerHTML = renderMarkdown(body.report || "");
    chatHistory.innerHTML = "";
    appendChat("assistant", body.response || "");
    wellnessResult.classList.remove("hidden");
    downloadRecord.disabled = false;
    sendChat.disabled = false;
    setStatus(wellnessStatus, "报告已生成");
  } catch (error) {
    setStatus(wellnessStatus, error.message, true);
  } finally {
    generateReport.disabled = false;
  }
}

async function submitChatQuestion() {
  const question = chatQuestion.value.trim();
  if (!state.sessionId) {
    setStatus(wellnessStatus, "请先生成健康报告", true);
    return;
  }
  if (!question) {
    setStatus(wellnessStatus, "请输入追问内容", true);
    return;
  }

  sendChat.disabled = true;
  appendChat("user", question);
  chatQuestion.value = "";
  setStatus(wellnessStatus, "猫猫正在整理回答...");

  try {
    const response = await fetch("/wellness/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        question,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "追问失败");
    }
    appendChat("assistant", body.answer || "");
    setStatus(wellnessStatus, "已更新回答");
  } catch (error) {
    appendChat("assistant", error.message);
    setStatus(wellnessStatus, error.message, true);
  } finally {
    sendChat.disabled = false;
  }
}

async function downloadMedicalRecord() {
  if (!state.sessionId) {
    setStatus(wellnessStatus, "请先生成健康报告", true);
    return;
  }

  downloadRecord.disabled = true;
  setStatus(wellnessStatus, "正在生成正式记录...");

  try {
    const response = await fetch(`/wellness/medical-record/${state.sessionId}`);
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || "正式记录下载失败");
    }
    const blob = await response.blob();
    const contentDisposition = response.headers.get("Content-Disposition") || "";
    const matched = /filename="?([^"]+)"?/.exec(contentDisposition);
    const filename = matched?.[1] || "official_medical_record.md";
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setStatus(wellnessStatus, "正式记录已下载");
  } catch (error) {
    setStatus(wellnessStatus, error.message, true);
  } finally {
    downloadRecord.disabled = false;
  }
}

async function submitFallDetection() {
  let parsed;
  try {
    parsed = safeJsonParse(fallInput.value);
  } catch (error) {
    setStatus(fallStatus, "动作数据 JSON 解析失败", true);
    return;
  }

  detectFall.disabled = true;
  setStatus(fallStatus, "正在分析动作反应...");

  try {
    const response = await fetch("/wellness/fall-detection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: parsed.user_id || "cat_fall_user",
        motion_data: parsed,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "跌倒检测失败");
    }

    fallReaction.textContent = body.reaction || "";
    fallActivity.textContent = body.predicted_activity || "-";
    fallState.textContent = body.user_state || "-";
    fallDetails.textContent = JSON.stringify(body, null, 2);
    fallResult.classList.remove("hidden");
    setStatus(fallStatus, "动作分析完成");
  } catch (error) {
    setStatus(fallStatus, error.message, true);
  } finally {
    detectFall.disabled = false;
  }
}

loadProjectSample.addEventListener("click", loadProjectDemo);
loadSimpleSample.addEventListener("click", fillSimpleSamples);
wellnessFile.addEventListener("change", handleWellnessFile);
generateReport.addEventListener("click", submitWellnessReport);
sendChat.addEventListener("click", submitChatQuestion);
downloadRecord.addEventListener("click", downloadMedicalRecord);
detectFall.addEventListener("click", submitFallDetection);

fillSimpleSamples();
downloadRecord.disabled = true;
sendChat.disabled = true;

// ================= TAB SWITCHING =================
const tabButtons = document.querySelectorAll(".tab-button");
const tabPanes = document.querySelectorAll(".tab-pane");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const targetId = btn.getAttribute("data-tab");

    tabButtons.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    tabPanes.forEach((p) => p.classList.remove("active"));

    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    const targetPane = document.getElementById(targetId);
    if (targetPane) {
      targetPane.classList.add("active");
    }
  });
});

// ================= UTILITIES =================
function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inCodeBlock = false;
  let codeLines = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
      }
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
    } else if (bullet) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
    }
  }

  if (inCodeBlock) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  closeList();
  return html.join("");
}

function inlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
  value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  value = value.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  value = value.replace(/_([^_]+)_/g, "<em>$1</em>");
  return value;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read image file."));
    reader.readAsDataURL(file);
  });
}

// ================= TAB 1: CROP RECOMMENDATION =================
const recForm = document.getElementById("rec-form");
const recSubmitBtn = document.getElementById("rec-submit-btn");
const recPlaceholder = document.getElementById("rec-placeholder");
const recContent = document.getElementById("rec-content");

const recPresets = {
  "rec-rice": { N: 90, P: 42, K: 43, temperature: 20.87, humidity: 82.0, ph: 6.5, rainfall: 202.9 },
  "rec-maize": { N: 80, P: 45, K: 20, temperature: 23.5, humidity: 65.0, ph: 6.8, rainfall: 70.0 },
  "rec-coffee": { N: 100, P: 30, K: 30, temperature: 25.0, humidity: 60.0, ph: 6.5, rainfall: 160.0 },
  "rec-cotton": { N: 120, P: 50, K: 20, temperature: 26.0, humidity: 80.0, ph: 7.0, rainfall: 80.0 },
};

document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.getAttribute("data-preset");
    if (recPresets[key]) {
      const p = recPresets[key];
      document.getElementById("rec-N").value = p.N;
      document.getElementById("rec-P").value = p.P;
      document.getElementById("rec-K").value = p.K;
      document.getElementById("rec-ph").value = p.ph;
      document.getElementById("rec-temperature").value = p.temperature;
      document.getElementById("rec-humidity").value = p.humidity;
      document.getElementById("rec-rainfall").value = p.rainfall;
      recForm.dispatchEvent(new Event("submit"));
    }
  });
});

recForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  recSubmitBtn.disabled = true;
  recSubmitBtn.innerHTML = `<span>गणना हो रही है... (Running Random Forest)</span>`;

  const payload = {
    N: parseFloat(document.getElementById("rec-N").value),
    P: parseFloat(document.getElementById("rec-P").value),
    K: parseFloat(document.getElementById("rec-K").value),
    ph: parseFloat(document.getElementById("rec-ph").value),
    temperature: parseFloat(document.getElementById("rec-temperature").value),
    humidity: parseFloat(document.getElementById("rec-humidity").value),
    rainfall: parseFloat(document.getElementById("rec-rainfall").value),
  };

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    recPlaceholder.classList.add("hidden");
    recContent.classList.remove("hidden");

    document.getElementById("rec-crop-name").textContent = data.crop.toUpperCase();
    document.getElementById("rec-crop-hi").textContent = data.crop_name_hi || data.crop;
    document.getElementById("rec-season").textContent = data.season || "अनुकूल मौसम";
    document.getElementById("rec-water").textContent = data.water_req || "मध्यम";
    document.getElementById("rec-soil").textContent = data.soil_type || "दोमट मिट्टी";
    document.getElementById("rec-description").textContent = data.description || "फसल के लिए वातावरण उपयुक्त है।";
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    recSubmitBtn.disabled = false;
    recSubmitBtn.innerHTML = `<span class="btn-icon">🌾</span><span>फसल की सिफारिश प्राप्त करें (Predict Crop)</span>`;
  }
});

// ================= TAB 2: DISEASE DETECTION =================
const leafDropzone = document.getElementById("leaf-dropzone");
const leafFileInput = document.getElementById("leaf-file-input");
const dropzoneContent = document.getElementById("dropzone-content");
const previewWrap = document.getElementById("preview-wrap");
const imagePreview = document.getElementById("image-preview");
const removeImgBtn = document.getElementById("remove-img-btn");
const diseaseSubmitBtn = document.getElementById("disease-submit-btn");
const diseasePlaceholder = document.getElementById("disease-placeholder");
const diseaseContent = document.getElementById("disease-content");

let currentLeafDataUrl = null;
let currentLeafFileName = "leaf.jpg";

function displaySelectedImage(dataUrl, fileName) {
  currentLeafDataUrl = dataUrl;
  currentLeafFileName = fileName || "leaf.jpg";
  imagePreview.src = dataUrl;
  dropzoneContent.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  diseaseSubmitBtn.disabled = false;
}

function clearSelectedImage() {
  currentLeafDataUrl = null;
  leafFileInput.value = "";
  imagePreview.src = "";
  previewWrap.classList.add("hidden");
  dropzoneContent.classList.remove("hidden");
  diseaseSubmitBtn.disabled = true;
}

leafFileInput.addEventListener("change", async () => {
  const file = leafFileInput.files[0];
  if (file) {
    const dataUrl = await fileToDataUrl(file);
    displaySelectedImage(dataUrl, file.name);
  }
});

removeImgBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSelectedImage();
});

// Drag and drop
leafDropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  leafDropzone.classList.add("dragover");
});

leafDropzone.addEventListener("dragleave", () => {
  leafDropzone.classList.remove("dragover");
});

leafDropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  leafDropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    const dataUrl = await fileToDataUrl(file);
    displaySelectedImage(dataUrl, file.name);
  }
});

// Sample leaves buttons
document.querySelectorAll("[data-sample]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const samplePath = btn.getAttribute("data-sample");
    try {
      const response = await fetch(samplePath);
      const blob = await response.blob();
      const dataUrl = await fileToDataUrl(blob);
      displaySelectedImage(dataUrl, samplePath.split("/").pop());
      diseaseSubmitBtn.click();
    } catch (err) {
      alert("Could not load sample image: " + err.message);
    }
  });
});

diseaseSubmitBtn.addEventListener("click", async () => {
  if (!currentLeafDataUrl) return;

  diseaseSubmitBtn.disabled = true;
  diseaseSubmitBtn.innerHTML = `<span>सीएनएन मॉडल जांच कर रहा है... (CNN Inferencing)</span>`;

  try {
    const res = await fetch("/api/detect-disease", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: currentLeafDataUrl,
        imageName: currentLeafFileName,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");

    diseasePlaceholder.classList.add("hidden");
    diseaseContent.classList.remove("hidden");

    const statusBox = document.getElementById("disease-status-box");
    const statusIcon = document.getElementById("disease-status-icon");
    const healthStatus = document.getElementById("disease-health-status");
    const plantTitle = document.getElementById("disease-plant-title");
    const conditionTitle = document.getElementById("disease-condition-title");
    const confVal = document.getElementById("disease-confidence-val");
    const confBar = document.getElementById("disease-confidence-bar");
    const treatmentText = document.getElementById("disease-treatment-text");
    const preventionText = document.getElementById("disease-prevention-text");

    if (data.is_healthy) {
      statusBox.className = "disease-status-bar status-healthy";
      statusIcon.textContent = "✅";
      healthStatus.textContent = "स्वस्थ पत्ती (Healthy Leaf)";
    } else {
      statusBox.className = "disease-status-bar status-infected";
      statusIcon.textContent = "⚠️";
      healthStatus.textContent = "रोगग्रस्त (Diseased / Infected)";
    }

    plantTitle.textContent = data.plant;
    conditionTitle.textContent = data.condition;
    confVal.textContent = data.confidence + "%";
    confBar.style.width = Math.min(Math.max(data.confidence, 10), 100) + "%";
    treatmentText.textContent = data.treatment;
    preventionText.textContent = data.prevention;
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    diseaseSubmitBtn.disabled = false;
    diseaseSubmitBtn.innerHTML = `<span class="btn-icon">🔬</span><span>रोग का विश्लेषण करें (Run CNN Diagnosis)</span>`;
  }
});

// ================= TAB 3: PRICE PREDICTION =================
const priceForm = document.getElementById("price-form");
const priceSubmitBtn = document.getElementById("price-submit-btn");
const pricePlaceholder = document.getElementById("price-placeholder");
const priceContent = document.getElementById("price-content");

const pricePresets = {
  "price-wheat": {
    STATE: "Haryana",
    district: "Ambala",
    market: "Ambala City",
    commodity: "Wheat",
    variety: "Other",
    grade: "FAQ",
    min: 2100,
    max: 2350,
    date: "2024-04-15",
  },
  "price-tomato": {
    STATE: "Maharashtra",
    district: "Pune",
    market: "Pune",
    commodity: "Tomato",
    variety: "Local",
    grade: "FAQ",
    min: 1200,
    max: 1800,
    date: "2024-05-10",
  },
  "price-potato": {
    STATE: "Uttar Pradesh",
    district: "Agra",
    market: "Agra",
    commodity: "Potato",
    variety: "Deshi",
    grade: "FAQ",
    min: 900,
    max: 1250,
    date: "2024-03-20",
  },
};

document.querySelectorAll("[data-preset^='price-']").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.getAttribute("data-preset");
    if (pricePresets[key]) {
      const p = pricePresets[key];
      document.getElementById("price-state").value = p.STATE;
      document.getElementById("price-district").value = p.district;
      document.getElementById("price-market").value = p.market;
      document.getElementById("price-commodity").value = p.commodity;
      document.getElementById("price-variety").value = p.variety;
      document.getElementById("price-grade").value = p.grade;
      document.getElementById("price-min").value = p.min;
      document.getElementById("price-max").value = p.max;
      document.getElementById("price-date").value = p.date;
      priceForm.dispatchEvent(new Event("submit"));
    }
  });
});

priceForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  priceSubmitBtn.disabled = true;
  priceSubmitBtn.innerHTML = `<span>कैटबूस्ट गणना कर रहा है... (Forecasting Price)</span>`;

  const dateVal = document.getElementById("price-date").value;
  const d = new Date(dateVal || Date.now());
  const year = d.getFullYear();
  const month = d.getMonth() + 1;
  const day = d.getDate();
  const dayOfWeek = (d.getDay() + 6) % 7; // Monday = 0

  const minPrice = parseFloat(document.getElementById("price-min").value);
  const maxPrice = parseFloat(document.getElementById("price-max").value);

  const payload = {
    STATE: document.getElementById("price-state").value.trim(),
    "District Name": document.getElementById("price-district").value.trim(),
    "Market Name": document.getElementById("price-market").value.trim(),
    Commodity: document.getElementById("price-commodity").value.trim(),
    Variety: document.getElementById("price-variety").value.trim(),
    Grade: document.getElementById("price-grade").value.trim(),
    Min_Price: minPrice,
    Max_Price: maxPrice,
    Year: year,
    Month: month,
    Day: day,
    DayOfWeek: dayOfWeek,
  };

  try {
    const res = await fetch("/api/predict-price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Price prediction failed");

    pricePlaceholder.classList.add("hidden");
    priceContent.classList.remove("hidden");

    const predPrice = data.predicted_price;
    document.getElementById("price-predicted-val").textContent = predPrice.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    document.getElementById("price-disp-min").textContent = "₹" + minPrice.toLocaleString("en-IN");
    document.getElementById("price-disp-max").textContent = "₹" + maxPrice.toLocaleString("en-IN");

    let pct = 50;
    if (maxPrice > minPrice) {
      pct = ((predPrice - minPrice) / (maxPrice - minPrice)) * 100;
      pct = Math.min(Math.max(pct, 5), 95);
    }
    document.getElementById("price-range-marker").style.left = pct + "%";
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    priceSubmitBtn.disabled = false;
    priceSubmitBtn.innerHTML = `<span class="btn-icon">📈</span><span>मंडी भाव का अनुमान लगाएं (Forecast Modal Price)</span>`;
  }
});

// ================= TAB 4: CHAT (AI KRISHI MITRA) =================
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-message-input");
const chatImageInput = document.getElementById("chat-image-input");
const chatAttachment = document.getElementById("chat-attachment");
const chatMessages = document.getElementById("chat-messages");
const chatSendBtn = document.getElementById("chat-send-btn");

let chatSelectedImage = null;

function addChatMessage(text, role) {
  const wrapper = document.createElement("div");
  wrapper.className = `chat-row ${role}-row`;
  wrapper.innerHTML = `<div class="chat-avatar">${role === "user" ? "आप" : "🌾"}</div><div class="chat-bubble"></div>`;
  const bubble = wrapper.querySelector(".chat-bubble");
  if (role === "bot") {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatImageInput.addEventListener("change", () => {
  chatSelectedImage = chatImageInput.files[0] || null;
  chatAttachment.textContent = chatSelectedImage ? `Attached: ${chatSelectedImage.name}` : "";
  chatAttachment.classList.toggle("hidden", !chatSelectedImage);
});

// Quick inquiry chips
document.querySelectorAll(".btn-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const query = chip.getAttribute("data-query");
    chatInput.value = query;
    chatForm.dispatchEvent(new Event("submit"));
  });
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message && !chatSelectedImage) return;

  addChatMessage(message || "कृपया इस पत्ती की जांच करें।", "user");
  chatInput.value = "";
  chatSendBtn.disabled = true;
  chatSendBtn.textContent = "...";

  try {
    const payload = { message: message || "Please detect the disease in this plant image." };
    if (chatSelectedImage) {
      payload.imageName = chatSelectedImage.name;
      payload.image = await fileToDataUrl(chatSelectedImage);
    }
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Chat failed");
    addChatMessage(data.reply, "bot");
  } catch (err) {
    addChatMessage(`त्रुटि (Error): ${err.message}`, "bot");
  } finally {
    chatSelectedImage = null;
    chatImageInput.value = "";
    chatAttachment.classList.add("hidden");
    chatSendBtn.disabled = false;
    chatSendBtn.textContent = "पूछें (Send)";
    chatInput.focus();
  }
});

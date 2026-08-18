const elements = {
  healthBadge: document.querySelector("#healthBadge"),
  healthText: document.querySelector("#healthText"),
  imageSelect: document.querySelector("#imageSelect"),
  loadButton: document.querySelector("#loadButton"),
  fileInput: document.querySelector("#fileInput"),
  dropzone: document.querySelector("#dropzone"),
  foregroundLabel: document.querySelector("#foregroundLabel"),
  backgroundLabel: document.querySelector("#backgroundLabel"),
  candidateSlider: document.querySelector("#candidateSlider"),
  candidateValue: document.querySelector("#candidateValue"),
  scoreRow: document.querySelector("#scoreRow"),
  undoButton: document.querySelector("#undoButton"),
  clearButton: document.querySelector("#clearButton"),
  commitButton: document.querySelector("#commitButton"),
  resetButton: document.querySelector("#resetButton"),
  fillHoles: document.querySelector("#fillHoles"),
  closeKernel: document.querySelector("#closeKernel"),
  closeKernelValue: document.querySelector("#closeKernelValue"),
  erodeIterations: document.querySelector("#erodeIterations"),
  erodeIterationsValue: document.querySelector("#erodeIterationsValue"),
  previewButton: document.querySelector("#previewButton"),
  imageTitle: document.querySelector("#imageTitle"),
  canvas: document.querySelector("#imageCanvas"),
  emptyState: document.querySelector("#emptyState"),
  canvasLoading: document.querySelector("#canvasLoading"),
  loadingText: document.querySelector("#loadingText"),
  statusText: document.querySelector("#statusText"),
  saveButton: document.querySelector("#saveButton"),
  rgbaPreview: document.querySelector("#rgbaPreview"),
  previewPlaceholder: document.querySelector("#previewPlaceholder"),
  toast: document.querySelector("#toast"),
};

const context = elements.canvas.getContext("2d");
if (window.location.protocol === "file:") {
  window.location.replace("http://sam-food/");
}

const state = {
  sessionId: null,
  busy: false,
  image: null,
};

function processOptions() {
  return {
    use_fill_holes: elements.fillHoles.checked,
    close_kernel: Number(elements.closeKernel.value),
    erode_iterations: Number(elements.erodeIterations.value),
  };
}

function labelMode() {
  return document.querySelector('input[name="labelMode"]:checked').value;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(payload?.detail || `请求失败：HTTP ${response.status}`);
  }
  return payload;
}

function setBusy(busy, message = "正在处理") {
  state.busy = busy;
  elements.canvasLoading.classList.toggle("hidden", !busy);
  elements.loadingText.textContent = message;
  document.querySelectorAll("button").forEach((button) => {
    if (busy) {
      button.dataset.wasDisabled = String(button.disabled);
      button.disabled = true;
    } else if (button.dataset.wasDisabled !== undefined) {
      button.disabled = button.dataset.wasDisabled === "true";
      delete button.dataset.wasDisabled;
    }
  });
  if (!busy) {
    setSessionControls(Boolean(state.sessionId));
    elements.loadButton.disabled = false;
  }
}

function setSessionControls(enabled) {
  [
    elements.undoButton,
    elements.clearButton,
    elements.commitButton,
    elements.resetButton,
    elements.previewButton,
    elements.saveButton,
  ].forEach((button) => {
    button.disabled = !enabled;
  });
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.classList.add("hidden");
  }, 4200);
}

function updateScores(scores, currentIndex) {
  elements.scoreRow.replaceChildren();
  if (!scores.length) {
    const message = document.createElement("span");
    message.textContent = "候选分数将在首次点击后显示";
    elements.scoreRow.append(message);
    elements.candidateSlider.disabled = true;
    return;
  }
  scores.forEach((score, index) => {
    const chip = document.createElement("span");
    chip.className = `score-chip${index === currentIndex ? " current" : ""}`;
    chip.textContent = `${index} · ${score.toFixed(4)}`;
    elements.scoreRow.append(chip);
  });
  elements.candidateSlider.disabled = false;
}

function drawImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      elements.canvas.width = image.naturalWidth;
      elements.canvas.height = image.naturalHeight;
      context.clearRect(0, 0, image.naturalWidth, image.naturalHeight);
      context.drawImage(image, 0, 0);
      elements.canvas.classList.add("visible");
      elements.emptyState.classList.add("hidden");
      state.image = image;
      resolve();
    };
    image.onerror = reject;
    image.src = dataUrl;
  });
}

async function applyResponse(payload) {
  state.sessionId = payload.session_id;
  elements.imageTitle.textContent = payload.image_name || "未命名图片";
  elements.statusText.textContent = payload.status || "处理完成";
  elements.candidateSlider.value = String(payload.candidate_index ?? 0);
  elements.candidateValue.textContent = String(payload.candidate_index ?? 0);
  updateScores(payload.scores || [], payload.candidate_index ?? 0);
  if (payload.image) {
    await drawImage(payload.image);
  }
  if (payload.preview) {
    elements.rgbaPreview.src = payload.preview;
    elements.rgbaPreview.classList.add("visible");
    elements.previewPlaceholder.classList.add("hidden");
  }
  setSessionControls(true);
}

async function loadImageList() {
  const payload = await requestJson("/api/images");
  elements.imageSelect.replaceChildren();
  if (!payload.images.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "目录为空，请上传本地图片";
    elements.imageSelect.append(option);
    return;
  }
  payload.images.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    elements.imageSelect.append(option);
  });
}

async function checkHealth() {
  try {
    const payload = await requestJson("/api/health");
    elements.healthBadge.classList.add("online");
    elements.healthBadge.classList.remove("offline");
    elements.healthText.textContent = payload.ok ? `SAM 已就绪 · ${payload.device}` : "SAM 尚未初始化";
  } catch (error) {
    elements.healthBadge.classList.add("offline");
    elements.healthBadge.classList.remove("online");
    elements.healthText.textContent = "模型服务连接失败";
    elements.statusText.textContent = error.message;
  }
}

async function loadSelectedImage() {
  const imageName = elements.imageSelect.value;
  if (!imageName) {
    showToast("请先选择项目图片");
    return;
  }
  setBusy(true, "正在加载图片并计算图像特征");
  try {
    const payload = await requestJson("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_name: imageName }),
    });
    await applyResponse(payload);
  } catch (error) {
    showToast(error.message);
    elements.statusText.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function uploadFile(file) {
  if (!file) {
    showToast("请选择图片文件");
    return;
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  setBusy(true, `正在上传并解析：${file.name}`);
  try {
    const payload = await requestJson("/api/upload", {
      method: "POST",
      body: formData,
    });
    await applyResponse(payload);
    await loadImageList();
    elements.imageSelect.value = payload.image_name;
  } catch (error) {
    showToast(error.message);
    elements.statusText.textContent = error.message;
  } finally {
    elements.fileInput.value = "";
    setBusy(false);
  }
}

async function processAction(endpoint, extra = {}, message = "正在处理") {
  if (!state.sessionId || state.busy) {
    return;
  }
  setBusy(true, message);
  try {
    const payload = await requestJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        ...processOptions(),
        ...extra,
      }),
    });
    await applyResponse(payload);
    return payload;
  } catch (error) {
    showToast(error.message);
    elements.statusText.textContent = error.message;
    return null;
  } finally {
    setBusy(false);
  }
}

elements.loadButton.addEventListener("click", loadSelectedImage);
elements.imageSelect.addEventListener("dblclick", loadSelectedImage);

elements.fileInput.addEventListener("change", () => {
  uploadFile(elements.fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("dragover");
  });
});

elements.dropzone.addEventListener("drop", (event) => {
  uploadFile(event.dataTransfer.files[0]);
});

document.querySelectorAll('input[name="labelMode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    elements.foregroundLabel.classList.toggle("active", labelMode() === "前景点");
    elements.backgroundLabel.classList.toggle("active", labelMode() === "背景点");
  });
});

elements.canvas.addEventListener("click", (event) => {
  if (!state.sessionId || state.busy) {
    return;
  }
  const rect = elements.canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * elements.canvas.width / rect.width;
  const y = (event.clientY - rect.top) * elements.canvas.height / rect.height;
  processAction(
    "/api/click",
    { x, y, label_mode: labelMode() },
    "SAM 正在计算候选 Mask",
  );
});

elements.candidateSlider.addEventListener("input", () => {
  elements.candidateValue.textContent = elements.candidateSlider.value;
});

elements.candidateSlider.addEventListener("change", () => {
  processAction(
    "/api/candidate",
    { candidate_index: Number(elements.candidateSlider.value) },
    "正在切换候选 Mask",
  );
});

elements.undoButton.addEventListener("click", () => processAction("/api/undo", {}, "正在撤销提示点"));
elements.clearButton.addEventListener("click", () => processAction("/api/clear", {}, "正在清空当前提示"));
elements.commitButton.addEventListener("click", () => processAction("/api/commit", {}, "正在合并当前 Mask"));
elements.resetButton.addEventListener("click", () => processAction("/api/reset", {}, "正在重置选择"));
elements.previewButton.addEventListener("click", () => processAction("/api/preview", {}, "正在生成后处理预览"));

elements.saveButton.addEventListener("click", async () => {
  const payload = await processAction("/api/save", {}, "正在保存 Mask、RGBA 和可视化");
  if (payload) {
    showToast(payload.status);
  }
});

elements.closeKernel.addEventListener("input", () => {
  elements.closeKernelValue.textContent = elements.closeKernel.value;
});

elements.erodeIterations.addEventListener("input", () => {
  elements.erodeIterationsValue.textContent = elements.erodeIterations.value;
});

async function bootstrap() {
  setSessionControls(false);
  await Promise.allSettled([checkHealth(), loadImageList()]);
}

bootstrap();

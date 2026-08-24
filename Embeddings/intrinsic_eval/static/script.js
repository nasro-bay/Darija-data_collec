const SIMILARITY_CATEGORIES = window.SIMILARITY_CATEGORIES;
const ANALOGY_CATEGORIES = window.ANALOGY_CATEGORIES;

const ICON_ARROW_SIMILARITY = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 8l-4 4 4 4M17 8l4 4-4 4"/></svg>';
const ICON_ARROW_ANALOGY = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
const ICON_EDIT = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
const ICON_DELETE = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>';

let currentDataset = "word_similarity";
let currentData = [];
let editingId = null; // null = add mode, otherwise editing this row's id

const grid = document.getElementById("cardGrid");
const emptyState = document.getElementById("emptyState");
const categoryFilter = document.getElementById("categoryFilter");
const countBadge = document.getElementById("countBadge");
const toggle = document.getElementById("datasetToggle");
const overlay = document.getElementById("overlay");
const addPanel = document.getElementById("addPanel");
const addForm = document.getElementById("addForm");
const categorySelect = document.getElementById("categorySelect");
const similarityFields = document.getElementById("similarityFields");
const analogyFields = document.getElementById("analogyFields");
const scoreField = document.getElementById("scoreField");
const toast = document.getElementById("toast");
const submitBtn = document.getElementById("submitBtn");
const addPanelTitle = document.getElementById("addPanelTitle");

function categoriesFor(dataset) {
  return dataset === "word_similarity" ? SIMILARITY_CATEGORIES : ANALOGY_CATEGORIES;
}

function scoreColor(score) {
  // low similarity -> red, high similarity -> green (Algerian palette doubles
  // as a natural low/high convention here)
  const r = score < 0.5 ? 200 : Math.round(200 - (score - 0.5) * 2 * 176);
  const g = score < 0.5 ? Math.round(16 + score * 2 * 86) : 102 + Math.round((score - 0.5) * 2 * 18);
  const b = score < 0.5 ? 46 : Math.round(46 - (score - 0.5) * 2 * 22);
  return `rgb(${r}, ${g}, ${b})`;
}

function showToast(message, kind) {
  toast.textContent = message;
  toast.className = "toast show " + (kind || "");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { toast.className = "toast"; }, 2600);
}

function populateCategoryDropdowns() {
  const cats = categoriesFor(currentDataset);

  categoryFilter.innerHTML = '<option value="">All categories</option>' +
    cats.map((c) => `<option value="${c}">${c.replace(/_/g, " ")}</option>`).join("");

  categorySelect.innerHTML = cats.map((c) => `<option value="${c}">${c.replace(/_/g, " ")}</option>`).join("");
}

async function loadData() {
  const res = await fetch(`/api/${currentDataset}`);
  currentData = await res.json();
  renderCards();
}

function renderCards() {
  const filter = categoryFilter.value;
  const rows = filter ? currentData.filter((r) => r.category === filter) : currentData;

  countBadge.textContent = `${rows.length} ${rows.length === 1 ? "entry" : "entries"}`;
  grid.innerHTML = "";
  emptyState.style.display = rows.length ? "none" : "block";

  rows.forEach((row) => {
    const card = document.createElement("div");
    card.className = "entry-card";
    card.dataset.id = row.id;

    const arrowIcon = currentDataset === "word_similarity" ? ICON_ARROW_SIMILARITY : ICON_ARROW_ANALOGY;

    if (currentDataset === "word_similarity") {
      const pct = Math.round(row.score * 100);
      card.innerHTML = `
        <div class="card-actions">
          <button class="icon-btn edit-btn" title="Edit">${ICON_EDIT}</button>
          <button class="icon-btn delete-btn" title="Delete">${ICON_DELETE}</button>
        </div>
        <div class="words">${row.word1} <span class="arrow">${arrowIcon}</span> ${row.word2}</div>
        <span class="category-chip">${row.category.replace(/_/g, " ")}</span>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${pct}%; background:${scoreColor(row.score)}"></div></div>
        <span class="score-value">similarity ${row.score.toFixed(2)}</span>
      `;
    } else {
      card.innerHTML = `
        <div class="card-actions">
          <button class="icon-btn edit-btn" title="Edit">${ICON_EDIT}</button>
          <button class="icon-btn delete-btn" title="Delete">${ICON_DELETE}</button>
        </div>
        <div class="words">${row.word_a} <span class="arrow">${arrowIcon}</span> ${row.word_b}</div>
        <span class="category-chip">${row.category.replace(/_/g, " ")}</span>
      `;
    }

    card.querySelector(".delete-btn").addEventListener("click", () => deleteRow(row.id, card));
    card.querySelector(".edit-btn").addEventListener("click", () => openEditPanel(row));
    grid.appendChild(card);
  });
}

async function deleteRow(id, cardEl) {
  cardEl.classList.add("removing");
  try {
    const res = await fetch(`/api/${currentDataset}/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error("delete failed");
    setTimeout(async () => {
      await loadData();
      showToast("Entry deleted", "error");
    }, 260);
  } catch (e) {
    cardEl.classList.remove("removing");
    showToast("Could not delete entry", "error");
  }
}

function switchDataset(dataset) {
  currentDataset = dataset;
  toggle.classList.toggle("on-analogy", dataset === "analogy_pairs");
  toggle.querySelectorAll(".toggle-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.dataset === dataset);
  });
  scoreField.style.display = dataset === "word_similarity" ? "flex" : "none";
  similarityFields.style.display = dataset === "word_similarity" ? "flex" : "none";
  analogyFields.style.display = dataset === "analogy_pairs" ? "flex" : "none";
  populateCategoryDropdowns();
  loadData();
  positionToggleSlider();
}

const toggleSlider = document.querySelector(".toggle-slider");

function positionToggleSlider() {
  const activeBtn = toggle.querySelector(".toggle-btn.active");
  if (!activeBtn) return;
  // Measure the active button's actual rendered position/size relative
  // to the container -- correct regardless of padding/gap/text-width,
  // unlike a hand-computed 50% CSS split.
  toggleSlider.style.left = `${activeBtn.offsetLeft}px`;
  toggleSlider.style.width = `${activeBtn.offsetWidth}px`;
}

toggle.querySelectorAll(".toggle-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchDataset(btn.dataset.dataset));
});
window.addEventListener("resize", positionToggleSlider);

categoryFilter.addEventListener("change", renderCards);

// --- add / edit panel open/close ---
function openAddPanel() {
  editingId = null;
  addForm.reset();
  addPanelTitle.textContent = currentDataset === "word_similarity" ? "Add word-similarity pair" : "Add analogy pair";
  submitBtn.textContent = "Save entry";
  overlay.classList.add("visible");
  addPanel.classList.add("visible");
}

function openEditPanel(row) {
  editingId = row.id;
  addPanelTitle.textContent = currentDataset === "word_similarity" ? "Edit word-similarity pair" : "Edit analogy pair";
  submitBtn.textContent = "Save changes";

  if (currentDataset === "word_similarity") {
    addForm.word1.value = row.word1;
    addForm.word2.value = row.word2;
    addForm.score.value = row.score;
    addForm.score_range.value = row.score;
  } else {
    addForm.word_a.value = row.word_a;
    addForm.word_b.value = row.word_b;
  }
  addForm.category.value = row.category;
  addForm.note.value = row.note || "";

  overlay.classList.add("visible");
  addPanel.classList.add("visible");
}

function closeAddPanel() {
  overlay.classList.remove("visible");
  addPanel.classList.remove("visible");
  addForm.reset();
  editingId = null;
}
document.getElementById("openAddBtn").addEventListener("click", openAddPanel);
document.getElementById("cancelAddBtn").addEventListener("click", closeAddPanel);
overlay.addEventListener("click", closeAddPanel);

// keep the score slider and number box in sync
const scoreRange = addForm.querySelector('[name="score_range"]');
const scoreNumber = addForm.querySelector('[name="score"]');
scoreRange.addEventListener("input", () => { scoreNumber.value = scoreRange.value; });
scoreNumber.addEventListener("input", () => { scoreRange.value = scoreNumber.value; });

addForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(addForm);
  let payload;
  if (currentDataset === "word_similarity") {
    payload = {
      word1: fd.get("word1"),
      word2: fd.get("word2"),
      category: fd.get("category"),
      score: parseFloat(fd.get("score")),
      note: fd.get("note"),
    };
  } else {
    payload = {
      word_a: fd.get("word_a"),
      word_b: fd.get("word_b"),
      category: fd.get("category"),
      note: fd.get("note"),
    };
  }

  const isEdit = editingId !== null;
  const url = isEdit ? `/api/${currentDataset}/${editingId}` : `/api/${currentDataset}`;
  const method = isEdit ? "PUT" : "POST";

  const res = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json();
  if (!res.ok) {
    showToast(body.error || "Could not save entry", "error");
    return;
  }
  closeAddPanel();
  await loadData();
  showToast(isEdit ? "Entry updated" : "Entry added", "success");
});

populateCategoryDropdowns();
loadData();
// Fonts/layout may still be settling on first paint -- position once
// immediately, then again after fonts finish loading so the slider
// lines up with the button's final rendered width.
positionToggleSlider();
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(positionToggleSlider);
}

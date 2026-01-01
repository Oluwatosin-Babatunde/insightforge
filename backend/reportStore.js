// backend/reportStore.js
const fs = require("fs");
const path = require("path");

const {
  REPORTS_FILE,
  EXPORTS_DIR,
  MAX_REPORTS,
  PURGE_EXPORTS
} = require("./config");

// -----------------------------
// Ensure storage exists
// -----------------------------
function ensureReportsFile() {
  const dir = path.dirname(REPORTS_FILE);

  // Create folder if missing
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // Create empty reports.json if missing
  if (!fs.existsSync(REPORTS_FILE)) {
    fs.writeFileSync(REPORTS_FILE, JSON.stringify([], null, 2), "utf-8");
  }
}

// -----------------------------
// Load all reports
// -----------------------------
function loadReports() {
  ensureReportsFile();
  try {
    const raw = fs.readFileSync(REPORTS_FILE, "utf-8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    console.warn("⚠️ reports.json corrupted. Rebuilding empty store...");
    fs.writeFileSync(REPORTS_FILE, JSON.stringify([], null, 2), "utf-8");
    return [];
  }
}

// -----------------------------
// Save reports list safely
// -----------------------------
function saveReports(reports) {
  ensureReportsFile();
  fs.writeFileSync(REPORTS_FILE, JSON.stringify(reports, null, 2), "utf-8");
}

// -----------------------------
// Delete export files safely
// -----------------------------
function deleteExports(exportObj = {}) {
  if (!exportObj || typeof exportObj !== "object") return;

  const exportPaths = Object.values(exportObj); // markdown, html, pdf

  exportPaths.forEach((relPath) => {
    if (!relPath) return;

    // example: "/exports/report_123.pdf"
    const cleanPath = relPath.replace("/exports/", "");
    const absPath = path.join(EXPORTS_DIR, cleanPath);

    try {
      if (fs.existsSync(absPath)) {
        fs.unlinkSync(absPath);
        console.log("🧹 Deleted export:", absPath);
      }
    } catch (err) {
      console.warn("⚠️ Failed to delete export:", absPath, err.message);
    }
  });
}

// -----------------------------
// Purge Oldest Reports
// -----------------------------
function purgeOldReports(reports) {
  const max = parseInt(MAX_REPORTS || "100", 10);

  if (!max || reports.length <= max) return reports;

  // Sort by createdAt (oldest first)
  const sorted = [...reports].sort(
    (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
  );

  const overflow = sorted.length - max;
  const removed = sorted.slice(0, overflow);
  const kept = sorted.slice(overflow);

  console.log(`🧹 Purging ${removed.length} old reports (limit ${max})...`);

  if (PURGE_EXPORTS === true || PURGE_EXPORTS === "true") {
    removed.forEach((r) => deleteExports(r.exports));
  }

  saveReports(kept);
  return kept;
}

// -----------------------------
// API FUNCTIONS
// -----------------------------
function saveReport(reportObj) {
  let reports = loadReports();

  reports.push(reportObj);
  reports = purgeOldReports(reports);

  saveReports(reports);
  return reportObj;
}

function listReports() {
  const reports = loadReports();

  // Return latest first
  return reports.sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
}

function getReport(id) {
  const reports = loadReports();
  return reports.find((r) => String(r.id) === String(id));
}

// -----------------------------
// Export
// -----------------------------
module.exports = {
  saveReport,
  listReports,
  getReport
};
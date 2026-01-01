// backend/reportStore.js
const fs = require("fs");
const path = require("path");
require("dotenv").config();

// ----------------------
// Config
// ----------------------
const REPORTS_FILE = process.env.REPORTS_FILE
  ? path.resolve(__dirname, process.env.REPORTS_FILE)
  : path.resolve(__dirname, "reports.json");

const EXPORTS_DIR = process.env.EXPORTS_DIR
  ? path.resolve(__dirname, process.env.EXPORTS_DIR)
  : path.resolve(__dirname, "exports");

const MAX_REPORTS = parseInt(process.env.MAX_REPORTS || "100", 10);
const PURGE_EXPORTS = String(process.env.PURGE_EXPORTS || "true") === "true";

// ----------------------
// Ensure store file exists
// ----------------------
function ensureStore() {
  if (!fs.existsSync(REPORTS_FILE)) {
    fs.writeFileSync(REPORTS_FILE, JSON.stringify([]), "utf-8");
  }
}

// ----------------------
// Safe Load
// ----------------------
function loadStore() {
  ensureStore();
  try {
    return JSON.parse(fs.readFileSync(REPORTS_FILE, "utf-8"));
  } catch (err) {
    console.warn("⚠️ reports.json corrupted — resetting:", err.message);
    fs.writeFileSync(REPORTS_FILE, JSON.stringify([]), "utf-8");
    return [];
  }
}

// ----------------------
// Save Store
// ----------------------
function saveStore(data) {
  fs.writeFileSync(REPORTS_FILE, JSON.stringify(data, null, 2), "utf-8");
}

// ----------------------
// Purge helper
// ----------------------
function purgeOldReports(reports) {
  if (reports.length <= MAX_REPORTS) return reports;

  const removed = reports.splice(MAX_REPORTS);

  // ✅ Remove exports of purged reports
  if (PURGE_EXPORTS) {
    removed.forEach(r => {
      if (!r.exports) return;

      const targets = [
        r.exports.markdown,
        r.exports.html,
        r.exports.pdf
      ].filter(Boolean);

      targets.forEach(urlPath => {
        const filename = urlPath.replace("/exports/", "");
        const filePath = path.join(EXPORTS_DIR, filename);

        if (fs.existsSync(filePath)) {
          try {
            fs.unlinkSync(filePath);
          } catch (_) {}
        }
      });
    });
  }

  return reports;
}

// ----------------------
// ✅ Save full report record
// ----------------------
function saveReport(report) {
  const reports = loadStore();
  reports.unshift(report); // newest first

  // Purge beyond MAX_REPORTS
  purgeOldReports(reports);

  saveStore(reports);
}

// ----------------------
// ✅ List reports (summary only)
// ----------------------
function listReports() {
  return loadStore().map(r => ({
    id: r.id,
    fileName: r.fileName,
    intent: r.intent,
    createdAt: r.createdAt,
    sourceType: r.sourceType,
    rows: r.rows,
    cols: r.cols,
    exports: r.exports
  }));
}

// ----------------------
// ✅ Get full report by ID
// ----------------------
function getReport(reportId) {
  const reports = loadStore();
  return reports.find(r => String(r.id) === String(reportId));
}

module.exports = {
  saveReport,
  listReports,
  getReport
};

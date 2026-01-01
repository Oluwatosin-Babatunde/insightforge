// backend/exportService.js
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

// ----------------------
// Load ENV (safe)
// ----------------------
require("dotenv").config();

// ----------------------
// Config
// ----------------------
const EXPORTS_DIR = process.env.EXPORTS_DIR
  ? path.resolve(__dirname, process.env.EXPORTS_DIR)
  : path.resolve(__dirname, "exports");

const REPORTS_FILE = process.env.REPORTS_FILE
  ? path.resolve(__dirname, process.env.REPORTS_FILE)
  : path.resolve(__dirname, "reports.json");

const MAX_REPORTS = parseInt(process.env.MAX_REPORTS || "100", 10);
const PURGE_EXPORTS = String(process.env.PURGE_EXPORTS || "true") === "true";

const MAX_EXPORT_FILES = parseInt(process.env.MAX_EXPORT_FILES || "250", 10);
const MAX_EXPORT_AGE_DAYS = parseInt(process.env.MAX_EXPORT_AGE_DAYS || "30", 10);

// ----------------------
// Ensure folders exist
// ----------------------
function ensureExportsDir() {
  if (!fs.existsSync(EXPORTS_DIR)) fs.mkdirSync(EXPORTS_DIR, { recursive: true });

  // Ensure gitkeep remains
  const gitkeepPath = path.join(EXPORTS_DIR, ".gitkeep");
  if (!fs.existsSync(gitkeepPath)) fs.writeFileSync(gitkeepPath, "", "utf-8");
}

function ensureReportsFile() {
  if (!fs.existsSync(REPORTS_FILE)) {
    fs.writeFileSync(REPORTS_FILE, JSON.stringify([]), "utf-8");
  }
}

// ----------------------
// HTML Wrapper
// ----------------------
function wrapHTML(markdownHTML, title = "InsightForge Report") {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>${title}</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 40px; max-width: 900px; margin: auto; }
    h1,h2,h3 { color: #1e1b4b; }
    p { color: #334155; line-height: 1.6; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 10px; }
    th { background: #f1f5f9; font-weight: bold; }
    img { max-width: 100%; margin: 20px 0; border: 1px solid #eee; border-radius: 12px; }
    footer { margin-top: 40px; font-size: 12px; color: #94a3b8; text-align: center; }
  </style>
</head>
<body>
  ${markdownHTML}
  <footer>
    InsightForge • Open Source Analytics Engine <br/>
    Developed by Oluwatosin Agbaakin © 2026 • MIT License
  </footer>
</body>
</html>`;
}

// ----------------------
// PDF Generator (Safe)
// ----------------------
async function generatePDF(htmlPath, pdfPath) {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"]
  });

  const page = await browser.newPage();
  await page.goto("file://" + htmlPath, { waitUntil: "networkidle0" });

  await page.pdf({
    path: pdfPath,
    format: "A4",
    printBackground: true
  });

  await browser.close();
}

// ----------------------
// Helper: Cleanup Exports
// ----------------------
function cleanupExportsFolder() {
  ensureExportsDir();

  let files = fs
    .readdirSync(EXPORTS_DIR)
    .filter(f => f !== ".gitkeep");

  // 1️⃣ Delete exports older than MAX_EXPORT_AGE_DAYS
  const now = Date.now();
  const maxAgeMs = MAX_EXPORT_AGE_DAYS * 24 * 60 * 60 * 1000;

  files.forEach(file => {
    const filePath = path.join(EXPORTS_DIR, file);
    const stat = fs.statSync(filePath);
    const age = now - stat.mtimeMs;

    if (age > maxAgeMs) {
      try {
        fs.unlinkSync(filePath);
      } catch (_) {}
    }
  });

  // Refresh list after age purge
  files = fs
    .readdirSync(EXPORTS_DIR)
    .filter(f => f !== ".gitkeep");

  // 2️⃣ Enforce MAX_EXPORT_FILES
  if (files.length > MAX_EXPORT_FILES) {
    const sorted = files
      .map(f => ({
        name: f,
        time: fs.statSync(path.join(EXPORTS_DIR, f)).mtimeMs
      }))
      .sort((a, b) => a.time - b.time);

    const toDelete = sorted.slice(0, files.length - MAX_EXPORT_FILES);

    toDelete.forEach(item => {
      try {
        fs.unlinkSync(path.join(EXPORTS_DIR, item.name));
      } catch (_) {}
    });
  }
}

// ----------------------
// Helper: Purge Old Reports + Their Exports
// ----------------------
function purgeOldReports() {
  ensureReportsFile();

  let reports;
  try {
    reports = JSON.parse(fs.readFileSync(REPORTS_FILE, "utf-8"));
  } catch (err) {
    reports = [];
  }

  if (reports.length <= MAX_REPORTS) return;

  // Remove oldest
  const removed = reports.splice(MAX_REPORTS);

  // Save trimmed report store
  fs.writeFileSync(REPORTS_FILE, JSON.stringify(reports, null, 2), "utf-8");

  // Delete exports for removed reports (if enabled)
  if (PURGE_EXPORTS && removed.length) {
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
}

// ----------------------
// ✅ Save Exports Entry
// ----------------------
function saveExports({ reportId, markdownText, markdownHTML }) {
  ensureExportsDir();
  ensureReportsFile();

  // Cleanup old data first
  purgeOldReports();
  cleanupExportsFolder();

  const mdFile = `report_${reportId}.md`;
  const htmlFile = `report_${reportId}.html`;
  const pdfFile = `report_${reportId}.pdf`;

  const mdPath = path.join(EXPORTS_DIR, mdFile);
  const htmlPath = path.join(EXPORTS_DIR, htmlFile);
  const pdfPath = path.join(EXPORTS_DIR, pdfFile);

  fs.writeFileSync(mdPath, markdownText, "utf-8");
  fs.writeFileSync(htmlPath, wrapHTML(markdownHTML), "utf-8");

  // PDF generation should never crash server
  generatePDF(htmlPath, pdfPath).catch(err => {
    console.warn("⚠️ PDF generation failed:", err.message);
  });

  return {
    markdown: `/exports/${mdFile}`,
    html: `/exports/${htmlFile}`,
    pdf: `/exports/${pdfFile}`
  };
}

module.exports = { saveExports };
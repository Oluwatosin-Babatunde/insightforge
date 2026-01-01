// backend/server.js
require("dotenv").config();

const express = require("express");
const cors = require("cors");
const path = require("path");
const multer = require("multer");
const { execFile } = require("child_process");
const fs = require("fs");

const { saveExports } = require("./exportService");
const { saveReport, listReports, getReport } = require("./reportStore");
const { EXPORTS_DIR } = require("./config");

const app = express();

// ----------------------
// Upload config
// ----------------------
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
});

// ----------------------
// Middleware
// ----------------------
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ----------------------
// Static serving
// ----------------------

// Serve frontend
app.use(express.static(path.join(__dirname, "../frontend")));

// Serve exports from configurable directory
// IMPORTANT: this must match whatever EXPORTS_DIR points to
app.use("/exports", express.static(EXPORTS_DIR));

// ----------------------
// Health check
// ----------------------
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    product: "InsightForge",
    version: "1.0",
    engine: "Offline Analysis Engine",
    message: "InsightForge backend is running",
    exportsDir: EXPORTS_DIR,
  });
});

// ----------------------
// Reports API
// ----------------------
app.get("/reports", (req, res) => {
  try {
    const reports = listReports();
    res.json({ status: "success", reports });
  } catch (err) {
    console.error("🔥 Reports list error:", err);
    res.status(500).json({
      status: "error",
      message: "Failed to list reports",
      details: err.message,
    });
  }
});

app.get("/reports/:id", (req, res) => {
  try {
    const report = getReport(req.params.id);
    if (!report) {
      return res.status(404).json({
        status: "error",
        message: "Report not found",
      });
    }
    res.json({ status: "success", report });
  } catch (err) {
    console.error("🔥 Report fetch error:", err);
    res.status(500).json({
      status: "error",
      message: "Failed to fetch report",
      details: err.message,
    });
  }
});

// ----------------------
// File support check
// ----------------------
function isSupported(filename = "") {
  const lower = filename.toLowerCase();
  return (
    lower.endsWith(".csv") ||
    lower.endsWith(".xlsx") ||
    lower.endsWith(".xls") ||
    lower.endsWith(".json")
  );
}

// ----------------------
// Analyze endpoint
// ----------------------
app.post("/analyze", upload.single("file"), (req, res) => {
  const intent = (req.body.context || "").trim();
  const file = req.file;

  if (!file && !intent) {
    return res.status(400).json({
      status: "error",
      message: "Source file or intent required",
    });
  }

  // Intent-only mode
  if (!file) {
    return res.json({
      status: "success",
      analysis:
        "✅ No file uploaded. (Intent-only mode)\n\nNext step: upload CSV/XLSX/JSON to analyze.",
      metadata: "No file uploaded.",
      code: "- No file provided; skipping file analysis",
    });
  }

  // Validate file type
  if (!isSupported(file.originalname)) {
    return res.status(400).json({
      status: "error",
      message: "Unsupported file type. Supported: .csv, .xlsx, .xls, .json",
    });
  }

  // Save to temp file
  const ext = path.extname(file.originalname).toLowerCase();
  const tempPath = path.join(__dirname, `tmp_${Date.now()}${ext}`);

  try {
    fs.writeFileSync(tempPath, file.buffer);
  } catch (err) {
    return res.status(500).json({
      status: "error",
      message: "Failed to save uploaded file",
      details: err.message,
    });
  }

  // Python script path
  const scriptPath = path.join(__dirname, "../python_engine/analyze.py");

  execFile("python3", [scriptPath, tempPath, intent], (err, stdout, stderr) => {
    // Cleanup temp file always
    try {
      fs.unlinkSync(tempPath);
    } catch (_) {}

    if (err) {
      console.error("🔥 Python error:", (stderr || err.message || "").toString());
      return res.status(500).json({
        status: "error",
        message: "Python analysis failed",
        error: (stderr || err.message || "").toString(),
      });
    }

    let parsed;
    try {
      parsed = JSON.parse(stdout);
    } catch (e) {
      console.error("🔥 Failed to parse python JSON output:", stdout);
      return res.status(500).json({
        status: "error",
        message: "Invalid response from Python engine",
        error: e.message,
      });
    }

    // Save exports + report record
    try {
      const reportId = Date.now();

      // Ensure exports dir exists (if user points it elsewhere)
      try {
        if (!fs.existsSync(EXPORTS_DIR)) fs.mkdirSync(EXPORTS_DIR, { recursive: true });
      } catch (_) {}

      const exportsObj = saveExports({
        reportId,
        markdownText: parsed.analysis || "",
        markdownHTML: parsed.analysis || "",
      });

      // Parse metadata to store summary fields safely
      let metaObj = null;
      try {
        metaObj = parsed.metadata ? JSON.parse(parsed.metadata) : null;
      } catch (_) {
        metaObj = null;
      }

      saveReport({
        id: reportId,
        fileName: file.originalname,
        intent: intent || "(none)",
        createdAt: new Date().toISOString(),

        sourceType: metaObj?.source_type || ext.replace(".", ""),
        rows: metaObj?.rows ?? null,
        cols: metaObj?.cols ?? null,

        exports: exportsObj,

        // Save full report content for in-app viewing
        analysis: parsed.analysis || "",
        metadata: parsed.metadata || "",
        code: parsed.code || "",
      });

      return res.json({
        status: "success",
        analysis: parsed.analysis || "",
        metadata: parsed.metadata || "",
        code: parsed.code || "",
        exports: exportsObj,
      });
    } catch (e) {
      console.error("🔥 Export/report save error:", e);
      return res.status(500).json({
        status: "error",
        message: "Failed to save exports/report",
        details: e.message,
      });
    }
  });
});

// ----------------------
// SPA fallback (Express v5 safe)
// ----------------------
app.use((req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/index.html"));
});

// ----------------------
// Global error handler (must be last)
// ----------------------
app.use((err, req, res, next) => {
  console.error("🔥 Backend Error:", err);
  res.status(500).json({
    status: "error",
    message: "Internal server error",
    details: err?.message || "Unknown error",
  });
});

// ----------------------
// Start server
// ----------------------
const PORT = process.env.PORT || 5050;
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log(`📦 Exports served from: ${EXPORTS_DIR}`);
});
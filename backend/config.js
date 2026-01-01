// backend/config.js
const path = require("path");
require("dotenv").config();

// ----------------------
// Default base directory
// ----------------------
const BASE_DIR = path.resolve(__dirname);

// ----------------------
// Configurable paths
// ----------------------
const EXPORTS_DIR = process.env.EXPORTS_DIR
  ? path.resolve(process.env.EXPORTS_DIR)
  : path.join(BASE_DIR, "exports");

const REPORTS_FILE = process.env.REPORTS_FILE
  ? path.resolve(process.env.REPORTS_FILE)
  : path.join(BASE_DIR, "reports.json");

// ----------------------
// Cleanup / purge rules
// ----------------------
const MAX_REPORTS = process.env.MAX_REPORTS
  ? parseInt(process.env.MAX_REPORTS, 10)
  : 100; // default: keep last 100 reports

const PURGE_EXPORTS =
  process.env.PURGE_EXPORTS === "true" ? true : false;

// ----------------------
// Export everything
// ----------------------
module.exports = {
  BASE_DIR,
  EXPORTS_DIR,
  REPORTS_FILE,
  MAX_REPORTS,
  PURGE_EXPORTS,
};

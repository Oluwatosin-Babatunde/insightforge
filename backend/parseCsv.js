const { parse } = require("csv-parse/sync");

function parseCsv(buffer) {
  const text = buffer.toString("utf-8");

  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true
  });

  if (!records.length) {
    return {
      rows: 0,
      columns: [],
      preview: []
    };
  }

  const columns = Object.keys(records[0]);

  return {
    rows: records.length,
    columns,
    preview: records.slice(0, 5)
  };
}

module.exports = { parseCsv };
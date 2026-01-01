# InsightForge (v1.0)

InsightForge is a lightweight AI-powered data profiling and reporting tool.

Upload a CSV or Excel dataset, provide an analysis intent (example: “trend over time”), and InsightForge generates an analyst-style report with:

✅ Executive summary  
✅ Data quality score  
✅ Duplicate + missing value analysis  
✅ Trend + segmentation insights  
✅ Charts + simple English explanations  
✅ Exportable reports (Markdown, HTML, PDF)  

---

## Features

- Upload datasets (CSV / Excel)
- Auto detects schema, numeric vs categorical fields
- Generates AI-enhanced narrative analysis
- Produces visual trend and distribution charts
- Exports reports to:
  - `.md`
  - `.html`
  - `.pdf`
- Auto purges old reports based on config

---

## Project Structure

```

workflow-ai-ops/
├── backend/
├── frontend/
├── python_engine/
├── docs/
├── examples/
└── README.md

````

---

## Requirements

- Node.js v18+ (recommended)
- Python 3.10+ (for report generation)
- pip (Python package manager)

---

## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Oluwatosin-Babatunde/insightforge.git
cd insightforge
````

---

### 2. Backend Setup

```bash
cd backend
npm install
```

✅ Create your local `.env` file:

```bash
cp ../.env.example .env
```

Then edit `.env` and confirm your paths:

```env
PORT=5050
EXPORTS_DIR=./exports
REPORTS_FILE=./reports.json
MAX_REPORTS=100
PURGE_EXPORTS=true
```

---

### 3. Python Setup

Go to the Python engine folder:

```bash
cd ../python_engine
pip install -r requirements.txt
```

✅ If you do not have `requirements.txt` yet, run this manually:

```bash
pip install pandas numpy matplotlib openpyxl
```

---

### 4. Start the Server

From the backend folder:

```bash
cd ../backend
npm run dev
```

Server runs at:

✅ `http://localhost:5050`

---

### 5. Open the Frontend

```bash
cd ../frontend
open index.html
```

---

## Testing with Example Dataset

Example dataset exists at:

```bash
examples/sample_input.csv
```

Upload it via the UI and try these intents:

✅ `trend over time using workDate`
✅ `distribution by department`
✅ `duplicates check`
✅ `missing values breakdown`
✅ `sum amount by category`

---

## Exports

Reports are generated into:

```
backend/exports/
```

and served publicly at:

✅ `http://localhost:5050/exports`

Example:

```
http://localhost:5050/exports/report_1234567890.html
```

---

## Auto Cleanup (Purge Strategy)

InsightForge can automatically delete old reports.

Controlled using `.env`:

| Setting              | Meaning                               |
| -------------------- | ------------------------------------- |
| `MAX_REPORTS=100`    | Keep only the latest 100 reports      |
| `PURGE_EXPORTS=true` | Deletes files when reports are purged |

---

## Git Ignore + Safety

The following files are ignored automatically:

* `.env`
* `backend/exports/*` (except `.gitkeep`)
* `node_modules/`
* runtime logs
* reportStore database file (`backend/reports.json`)

This keeps the repository clean and prevents accidental pushing of generated reports.

---

## Troubleshooting

### Error: Cannot find module 'dotenv'

Fix:

```bash
cd backend
npm install dotenv
```

---

### Python export errors

Ensure dependencies are installed:

```bash
pip install pandas numpy matplotlib openpyxl
```

---

### PDF export not working

This is usually caused by missing dependencies (like wkhtmltopdf or chromium).
For v1.0, markdown and html export works best.
PDF export will be improved in future versions.

---

## Roadmap

This is **v1.0 release**.
We will update the tool based on feedback from users in the coming year.

Planned improvements:

* better PDF export reliability
* smart intent auto-correction
* richer chart caption explanations
* more advanced anomaly detection
* connectors for Google Sheets and APIs

---

## Contributing

Pull requests are welcome.

If you want to propose improvements:

1. Fork the repo
2. Create a new branch
3. Submit a PR

---

## Feedback

If you use InsightForge, please share feedback:

* what worked well
* what was confusing
* what you want next
* need feedback to make it better.

---

## Author

Built by **Oluwatosin Babatunde**
GitHub: `@Oluwatosin-Babatunde`

````

---
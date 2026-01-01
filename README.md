# InsightForge v1.0  
**Offline AI Analytics Engine for CSV / Excel / JSON Data**

InsightForge is a lightweight, offline-first analytics engine that allows users to upload structured datasets and query insights using natural language intents.

It automatically generates:

- Executive Summary  
- KPI Profiling  
- Data Quality Score  
- Duplicate + Missing Value Detection  
- Smart Suggestions (automatic intent prompts)  
- Intent-Based Analyst Answers (with computation)  
- Charts (auto-generated images inside report)  
- Dataset Story (what the dataset likely represents)  
- Final Take (practical business conclusions)  
- Export Reports (Markdown, HTML, PDF)  
- Report History Storage + Retrieval

---

## Why InsightForge?
Most tools require SQL skills or rigid dashboards. InsightForge allows anyone to analyze datasets using natural language.

Example intents:
- `"trend over time"`
- `"distribution by status"`
- `"missing values breakdown"`
- `"duplicates check"`
- `"sum amount by department"`
- `"highest earnings by employee"`
- `"top 10 rows by revenue"`

---

## Features
### Flexible Natural Language Querying
InsightForge detects your intent and responds like a real analyst:
- Explains what it is doing
- Computes answers
- Generates supporting charts

### Offline Charts + Visuals
Charts are embedded as base64 images directly inside the report.

### Export Reports
Reports can be downloaded as:
- `.MD`
- `.HTML`
- `.PDF`

### Persistent Report History
Reports are saved automatically and accessible inside the UI.

### Built for Everyone
Works across thousands of datasets with zero configuration.

---

## Project Structure
```

workflow-ai-ops/
├── backend/             # Node.js Express backend
├── frontend/            # Tailwind single page UI
├── python_engine/       # Python analysis engine
├── docs/                # Documentation + roadmap
├── examples/            # Sample datasets & reports
└── README.md

````

---

## 🛠️ Tech Stack
**Frontend**
- TailwindCSS
- Marked.js
- Font Awesome Icons

**Backend**
- Node.js + Express
- Multer file uploads
- Puppeteer PDF exports
- JSON file-based report store

**Python Engine**
- Pandas + NumPy
- Matplotlib for charts
- Natural language intent engine

---

## Installation & Setup
### Clone repo
```bash
git clone https://github.com/yourusername/insightforge.git
cd insightforge
````

### 2️⃣ Backend install

```bash
cd backend
npm install
```

### 3️⃣ Python dependencies

Make sure Python 3 is installed, then run:

```bash
pip install pandas numpy matplotlib openpyxl
```

### 4️⃣ Setup `.env`

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 5️⃣ Start the backend

```bash
cd backend
npm run dev
```

Now open:
[http://localhost:5050](http://localhost:5050)

---

## Environment Configuration

InsightForge supports full control via `.env`

| Key           | Purpose                          | Default          |
| ------------- | -------------------------------- | ---------------- |
| PORT          | Backend port                     | `5050`           |
| EXPORTS_DIR   | Export output folder             | `./exports`      |
| REPORTS_FILE  | Report store database            | `./reports.json` |
| MAX_REPORTS   | Number of reports saved          | `100`            |
| PURGE_EXPORTS | Remove export files when purging | `true`           |

---

## Cleanup / Purge Strategy

InsightForge automatically purges old reports if they exceed `MAX_REPORTS`.

If `PURGE_EXPORTS=true`, export files (MD/HTML/PDF) are deleted automatically when a report is purged.

---

## Git Ignore Strategy

Generated files are not tracked:

* `/backend/exports/`
* `/backend/reports.json`

A `.gitkeep` file is used to preserve empty folders in GitHub.

---

## Roadmap (v1.1+)

* Multi-file dataset ingestion
* AI-assisted insight summarization (LLM plug-in)
* Interactive charts (Plotly)
* Search inside report history
* Export to PowerPoint
* Cloud deploy option

---

## Author

Developed by **Oluwatosin Agbaakin**
Licensed under the MIT License

---

## Support the Project

If this helped you, star the repo and share it with your community 

```

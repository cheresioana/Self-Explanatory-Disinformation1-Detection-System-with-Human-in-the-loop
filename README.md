# 🧠 Disinformation Narrative Detection App

This Flask app identifies whether a given text aligns with known disinformation narratives. It uses embeddings, a narrative tree structure, and allows expert edits via a web interface.

## 🚀 Features

- Classify text into disinformation narratives
- Visualize and explore narrative hierarchies
- Edit or remove nodes in real time

## 📁 Structure

```
├── app/                  # Flask entry point (app.py)
├── algo/                 # Tree logic and classification
├── LLM/                  # LLMs 
├── utils.py              # Text cleaning utilities
├── results/              # Trained narrative model (JSON)
├── work_data/            # CSVs: train, validation, correction, eval
├── requirements.txt
├── Dockerfile
```

## 🛠️ Usage

First unzip the archives from results and data/work_data folders.

### 🔧 Local

To see the interface:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/app.py
```

To run the algorithms
```bash
python algo/2_create_trees.py
```

### 🐳 Docker

```bash
docker build -t narrative-app .
docker run -p 5000:5000 narrative-app
```

Then visit: [http://localhost:5000](http://localhost:5000)

## 📂 Data

- CSVs in `work_data/`: `train.csv`, `validation.csv`, `correction.csv`, `evaluation.csv`
- Trained structure in: `results/full_result_0.4_change3.json`

## 📬 Endpoints

| Route                 | Method | Function                            |
|----------------------|--------|-------------------------------------|
| `/`                  | GET    | Homepage                            |
| `/process_text`      | POST   | Classify user input                 |
| `/get_internal_structure` | POST | Show narrative structure         |
| `/edit_node`         | POST   | Edit narrative text                 |
| `/mark_item_as_fake` | POST   | Add narrative node                  |
| `/go_to_parent`      | POST   | Navigate up the tree                |
| `/go_to_children`    | POST   | Navigate to child nodes             |
| `/remove_node`       | POST   | Delete a narrative node             |

---

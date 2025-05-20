# Clustering Data Visualization Tool

This application was developed as part of a master's thesis and is available at [https://telegram-clustering-viewer.streamlit.app/](https://telegram-clustering-viewer.streamlit.app/).  
The main goal of the project is to provide users with an insightful view into clustering data and to clearly present the results.

## 🎯 Key Features

- Interactive visualization of clustering results
- Show the clustering results
- Compare different clustering algorithms
- User-friendly interface
- LLM as a Judge

## 🛠️ Technologies Used

- Python, Streamlit
- PostgreSQL

## 🚀 How to run locally:

1. Clone the repository and switch to local branch:

   ```bash
   git clone https://github.com/Zizkai/telegram-clustering-viewer.git
   cd telegram-clustering-viewer
   git checkout local
   ```

2. Install the required packages:
   ```bash
    python -m venv .venv #must be python 3.10 or higher try python3.10 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install -r requirements.txt
   ```
3. Set up the database:
   ```
   docker compose up
   ```
4. Download database backup
   ```bash
   gdown https://drive.google.com/uc\?id\=1TRGHMOuv1xy4oZxCCbgwSSxWBYZmP9gj
   ```
5. Uploade the database backup to the PostgreSQL container:
   ```bash
    cat backup.sql | docker exec -i local-postgres psql -U admin -d data
   ```
6. Run the application:

   ```bash
   .venv/bin/streamlit run app/Home.py
   ```

7. Open your web browser and go to `http://localhost:8501`

8. Enjoy the application!

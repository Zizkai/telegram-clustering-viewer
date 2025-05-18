# Clustering Data Visualization Tool

This application was developed as part of a master's thesis and is available at [XXX](XXX).  
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

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository

   ```

2. Install the required packages:
   ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install .
   ```
3. Set up the database:
   ```
   docker compose up
   ```
4. Download database backup
   ```bash
   gdown https://drive.google.com/uc\?id\=1LndG_YSmcSTSB7PSCtZh0fVIsQSa48_t
   ```
5. Uploade the database backup to the PostgreSQL container:
   ```bash
    cat full_backup_16_5.sql | docker exec -i local-postgres psql -U admin
   ```
6. Run the application:

   ```bash
   streamlit run app/Home.py
   ```

7. Open your web browser and go to `http://localhost:8501`

8. Enjoy the application!

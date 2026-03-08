
DATABASE INTEGRATION UPDATE

1. Import in your Flask app:
   from database_utils import init_db, save_prediction

2. Call init_db() once when app starts.

3. After prediction:
   hash_value = save_prediction(patient_id, disease, confidence)

4. This creates predictions.db file.
   Data will persist even after server restart.

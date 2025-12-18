conda activate "E:\Sagar\Research Work\Conference_research\GraphBuilder-RAG\venv"
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
celery -A workers.tasks worker --loglevel=info --pool=solo
celery -A workers.tasks beat --loglevel=info
streamlit run ui/app.py
python helpers/test_query.py

# Clear Celery queue (purge all pending tasks)
celery -A workers.tasks purge -f

# Inspect Celery queue (check pending tasks)
celery -A workers.tasks inspect active
celery -A workers.tasks inspect reserved

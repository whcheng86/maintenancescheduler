1) Open a terminal in this folder.
2) Authenticate/select your GCP project:
   gcloud init
   gcloud config set project YOUR_PROJECT_ID
3) Deploy to Cloud Run (Singapore):
   gcloud run deploy track-optimizer --source . --region asia-southeast1 --allow-unauthenticated --timeout 900 --memory 2Gi --cpu 2 --concurrency 1
4) Open the Service URL printed by gcloud.

Local test:
  python -m pip install -r requirements.txt
  python main.py
  Open http://localhost:8080

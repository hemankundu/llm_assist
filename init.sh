wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh


dvc remote modify myremote gdrive_client_id "${GOOGLE_CLIENT_ID}"
dvc remote modify myremote gdrive_client_secret "${GOOGLE_CLIENT_SECRET}"

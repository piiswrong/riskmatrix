set -e

cd /home/dp/production

export PYTHONPATH=../

mkdir -p ./logs

echo "Fetching data..."
python3 ./fetch_data.py "today" 10 "./data/all_data_1d.parquet" > "./logs/fetch_data.log"

echo "Running pipeline..."
python3 ./run_prod_pipeline.py "./data/all_data_1d.parquet" "./data/predictions.parquet" --save-options "save_all" > "./logs/pipeline_factor.log"

echo "Executing..."
python3 ./execution.py "./config/execution_prod.yaml" > "./logs/execution.log"


mkdir -p logs_archive
date=`date +%Y%m%d`
mv logs logs_archive/$date

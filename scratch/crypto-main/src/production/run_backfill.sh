set -e
mkdir -p ./logs

export PYTHONPATH=../

#for i in 20241119 20241120 20241121 20241122 20241123 20241124 20241125 20241126 20241127 20241128 20241129 20241130 20241201; do
for i in 20241129 20241130; do
    export DATE=$(date -d "$i -1 day" +%Y%m%d)
    echo "Processing date: $DATE, storing in ./logs/$i"
    mkdir -p ./logs/$i

    # echo "Fetching data..."
    # python3 ./fetch_data.py "today" 10 "./data/all_data_1d.parquet" > "./logs/fetch_data.log" 2>&1

    echo "Running pipeline..."
    uv run ./run_prod_pipeline.py "./data/all_data_1d.parquet" "./data/predictions_$DATE.parquet" --save-options "save_per_day" --date "DATE" > "./logs/$i/pipeline_factor.log" 2>&1

    echo "Executing..."
    uv run ./execution.py "./config/execution_backfill.yaml" > "./logs/$i/execution.log" 2>&1
    # sleep 2
    echo "Done processing date: $DATE"
done

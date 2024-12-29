import argparse
import os
from factor.pipeline_factor import pipeline_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input and output paths')
    
    # Required arguments
    parser.add_argument('input_path', help='Path to input file')
    parser.add_argument('output_path', help='Path to output file')
    
    # Optional arguments
    parser.add_argument('--save-options', 
                    choices=['save_all', 'save_per_day'],
                    required=True,
                    help='Save options: save_all or save_per_day')
    
    parser.add_argument('--date',
                    default=None,
                    help='Date parameter. Use "DATE" to get from environment variable')
    
    parser.add_argument('--backtest',
                    action='store_true',
                    default=False,
                    help='Enable backtest mode')
    
    parser.add_argument('--rolling_day',
                    type=int,
                    default=None,
                    help='Rolling day parameter for training window')

    args = parser.parse_args()
    
    # Process date if it's "DATE"
    if args.date == "DATE":
        args.date = os.getenv("DATE")

    # gubo alpha 1
    factor_combination_list = [
        "amihud",
        "return_skewness",
        "alpha30",
        "alpha36",
        "alpha40",
        "alpha45",
        "ID",
    ]


    print(f"save_options: {args.save_options}, date: {args.date}, backtest: {args.backtest}")
    result = pipeline_main(input_path=args.input_path,
                           output_path=args.output_path,
                           factor_combination_list=factor_combination_list,
                           save_options=args.save_options,
                           date=args.date,
                           backtest_mode=args.backtest,
                           rolling_day=args.rolling_day)

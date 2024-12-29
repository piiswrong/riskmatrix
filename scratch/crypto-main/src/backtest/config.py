import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from .utils import GetDateTimeAsFileName

@dataclass
class BacktestConfig:
    """Configuration class for backtest parameters."""
    # Required input paths
    input_df_path: str
    delist_info_path: str
    trade_factor_name: str
    update_position_time: int

    start_date: datetime = None # Only do backtest after this date
    
    # Trading Constants
    commission: float = 10 / 10000.0
    start_cash: int = 100000
    use_hour_data: int = 0
    debug: int = 0
    eps: float = 1e-6
    
    # Strategy Parameters
    enable_early_cut_loss: int = 0
    cut_loss_pct_threshold: float = 20
    use_factor_pos_ratio: int = 0
    long_trade_rank_ratio: float = 1.0
    short_trade_rank_ratio: float = 0.5
    pos_ret_threshold: float = 3.0
    vol_filter_ratio: float = 30
    trade_rank_num: int = 20
    
    # Trading Parameters
    trade_with_rank: int = -20
    group_num: int = 20
    leverage: float = 1
    long_factor_combination_list: List[int] = None
    
    # Directory configurations
    base_dir: str = str(Path(__file__).parent)
    log_dir: str = str(Path(__file__).parent / "log")
    plot_dir: str = str(Path(__file__).parent / "plot")
    
    # Display configurations
    show_row_num: int = 20
    
    def __post_init__(self):
        """Create necessary directories and set defaults."""
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Set default for long_factor_combination_list if None
        if self.long_factor_combination_list is None:
            self.long_factor_combination_list = [1, 2, 3]
    
    @property
    def log_path(self) -> str:
        """Generate log file path."""
        return os.path.join(self.log_dir, f"factor_backtest_eval_{GetDateTimeAsFileName()}.log")
    
    @property
    def plot_path(self) -> str:
        """Generate plot file path."""
        input_file_name = self.input_df_path.split(r'/')[-1].split('.')[0]
        return os.path.join(self.plot_dir, f"backtest_{GetDateTimeAsFileName()}_{input_file_name}_{self.update_position_time}bar.png")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not os.path.exists(self.input_df_path):
            raise FileNotFoundError(f"input file path does not exist: {self.input_df_path}")
            
        if not os.path.exists(self.delist_info_path):
            print(f"Warning: Delist info path does not exist: {self.delist_info_path}")
        
        # Validate numeric ranges
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
        
        if self.start_cash <= 0:
            raise ValueError("Start cash must be positive")
        
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")
            
        return True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary format."""
        return {
            key: getattr(self, key) 
            for key in self.__annotations__
        }
import numpy as np
import pandas as pd
from utils import logger
from utils.constants import *

LOGGER = logger.setup_logger(__name__)


def run(df):
    market_df = pd.DataFrame([], columns=MODEL_VARIABLES, index=df.index)
    return market_df

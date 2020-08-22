import sys
from app import run

#Allows a CLI use after setup file installed, e.g.: python -m market_app TSLA 2018-01-01
if __name__ == "__main__":
    stock = sys.argv[1]
    start_date = sys.argv[2]
    run(stock, start_date)

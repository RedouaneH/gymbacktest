# gymbacktest

To get the data, run the two following commands in the `data` directory. Make sure to run the trade data command first.

```bash
# Downloads Binance BTCUSDT trades
!wget https://datasets.tardis.dev/v1/binance-futures/trades/2020/02/01/BTCUSDT.csv.gz -O BTCUSDT_trades.csv.gz

# Downloads Binance BTCUSDT order book
!wget https://datasets.tardis.dev/v1/binance-futures/incremental_book_L2/2020/02/01/BTCUSDT.csv.gz -O BTCUSDT_book.csv.gz
```
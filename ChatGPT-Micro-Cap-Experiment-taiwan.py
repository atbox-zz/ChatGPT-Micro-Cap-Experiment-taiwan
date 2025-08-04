"""台灣股票投資組合管理工具

這個腳本處理投資組合持倉、記錄交易，並印出每日結果。
專門針對台灣股票市場進行了調整。
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any, cast
import os

# 共用檔案位置
DATA_DIR = Path(".")
PORTFOLIO_CSV = DATA_DIR / "tw_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "tw_trade_log.csv"


def set_data_dir(data_dir: Path) -> None:
    """更新投資組合和交易記錄的全域路徑。

    Parameters
    ----------
    data_dir:
        儲存 ``tw_portfolio_update.csv`` 和
        ``tw_trade_log.csv`` 的目錄。
    """

    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DATA_DIR = Path(data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "tw_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "tw_trade_log.csv"

# 今天的日期，在記錄中重複使用
today = datetime.today().strftime("%Y-%m-%d")
now = datetime.now()
day = now.weekday()


def format_tw_ticker(ticker: str) -> str:
    """將台灣股票代碼格式化為 yfinance 可識別的格式。
    
    Parameters
    ----------
    ticker : str
        台灣股票代碼 (例如: "2330" 或 "2330.TW")
        
    Returns
    -------
    str
        格式化後的股票代碼 (例如: "2330.TW")
    """
    ticker = ticker.strip().upper()
    if not ticker.endswith('.TW') and not ticker.endswith('.TWO'):
        # 假設是上市股票，加上 .TW
        ticker += '.TW'
    return ticker


def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
) -> tuple[pd.DataFrame, float]:
    """更新每日價格資訊、記錄停損賣出，並提示交易。

    Parameters
    ----------
    portfolio:
        目前持倉，可以是 DataFrame、欄位名稱到列表的對應，或行字典的列表。
        輸入會被標準化為 ``DataFrame``，以便下游程式碼只需處理單一類型。
    cash:
        可用於交易的現金餘額。

    Returns
    -------
    tuple[pd.DataFrame, float]
        更新後的投資組合和現金餘額。
    """

    if isinstance(portfolio, pd.DataFrame):
        portfolio_df = portfolio.copy()
    elif isinstance(portfolio, (dict, list)):
        portfolio_df = pd.DataFrame(portfolio)
    else:  # pragma: no cover - 防禦性類型檢查
        raise TypeError("portfolio 必須是 DataFrame、dict 或 dict 的列表")

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    # 檢查是否為週末（台灣時間）
    if day == 6 or day == 5:
        check = input("""今天是週末，股市沒有開盤。
這會導致程式計算上一個交易日（通常是週五）的資料，並儲存為今天的資料。
您確定要這樣做嗎？要退出請輸入 1： """)
        if check == "1":
            raise SystemError("退出程式...")

    while True:
        action = input(
            f""" 您有 {cash:,.0f} 元現金。
您要記錄手動交易嗎？輸入 'b' 買入、's' 賣出，或按 Enter 繼續： """
        ).strip().lower()
        if action == "b":
            try:
                ticker = input("輸入股票代碼（例如 2330）： ").strip()
                ticker = format_tw_ticker(ticker)
                shares = int(input("輸入股數（張數）： ")) * 1000  # 台股以張為單位，1張=1000股
                buy_price = float(input("輸入買入價格： "))
                stop_loss = float(input("輸入停損價格： "))
                if shares <= 0 or buy_price <= 0 or stop_loss <= 0:
                    raise ValueError
            except ValueError:
                print("輸入無效。取消手動買入。")
            else:
                cash, portfolio_df = log_manual_buy(
                    buy_price, shares, ticker, stop_loss, cash, portfolio_df
                )
            continue
        if action == "s":
            try:
                ticker = input("輸入股票代碼： ").strip()
                ticker = format_tw_ticker(ticker)
                shares = int(input("輸入要賣出的股數（張數）： ")) * 1000
                sell_price = float(input("輸入賣出價格： "))
                if shares <= 0 or sell_price <= 0:
                    raise ValueError
            except ValueError:
                print("輸入無效。取消手動賣出。")
            else:
                cash, portfolio_df = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio_df
                )
            continue
        break

    for _, stock in portfolio_df.iterrows():
        ticker = stock["ticker"]
        shares = int(stock["shares"])
        cost = stock["buy_price"]
        stop = stock["stop_loss"]
        data = yf.Ticker(ticker).history(period="1d")

        if data.empty:
            print(f"{ticker} 無資料")
            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost,
                "Stop Loss": stop,
                "Current Price": "",
                "Total Value": "",
                "PnL": "",
                "Action": "NO DATA",
                "Cash Balance": "",
                "Total Equity": "",
            }
        else:
            low_price = round(float(data["Low"].iloc[-1]), 2)
            close_price = round(float(data["Close"].iloc[-1]), 2)

            if low_price <= stop:
                price = stop
                value = round(price * shares, 2)
                pnl = round((price - cost) * shares, 2)
                action = "SELL - 停損觸發"
                cash += value
                portfolio_df = log_sell(ticker, shares, price, cost, pnl, portfolio_df)
            else:
                price = close_price
                value = round(price * shares, 2)
                pnl = round((price - cost) * shares, 2)
                action = "HOLD"
                total_value += value
                total_pnl += pnl

            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost,
                "Stop Loss": stop,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }

        results.append(row)

    # 加入總計摘要行
    total_row = {
        "Date": today,
        "Ticker": "TOTAL",
        "Shares": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != today]
        print("今天的記錄已存在，不儲存結果到 CSV...")
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(PORTFOLIO_CSV, index=False)
    return portfolio_df, cash


def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """在 ``TRADE_LOG_CSV`` 中記錄停損賣出並移除該股票代碼。"""
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "自動賣出 - 停損觸發",
    }

    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio


def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    tw_portfolio: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """記錄手動買入並附加到投資組合。"""
    check = input(
        f"""您正要買入 {int(shares/1000)} 張 {ticker}，價格 {buy_price}，停損 {stoploss}。
        如果這是錯誤，請輸入 "1"： """
    )
    if check == "1":
        print("返回...")
        return cash, tw_portfolio

    data = yf.download(ticker, period="1d")
    data = cast(pd.DataFrame, data)
    if data.empty:
        print(f"{ticker} 手動買入失敗：無市場資料。")
        return cash, tw_portfolio
    day_high = float(data["High"].iloc[-1].item())
    day_low = float(data["Low"].iloc[-1].item())
    if not (day_low <= buy_price <= day_high):
        print(
            f"{ticker} 在 {buy_price} 手動買入失敗：價格超出今日範圍 {round(day_low, 2)}-{round(day_high, 2)}。"
        )
        return cash, tw_portfolio
    if buy_price * shares > cash:
        print(
            f"{ticker} 手動買入失敗：成本 {buy_price * shares:,.0f} 超過現金餘額 {cash:,.0f}。"
        )
        return cash, tw_portfolio
    pnl = 0.0

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": buy_price,
        "Cost Basis": buy_price * shares,
        "PnL": pnl,
        "Reason": "手動買入 - 新部位",
    }

    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    
    # 如果投資組合尚未包含該股票代碼，創建新行
    mask = tw_portfolio["ticker"] == ticker

    if not mask.any():
        new_trade = {
            "ticker": ticker,
            "shares": shares,
            "stop_loss": stoploss,
            "buy_price": buy_price,
            "cost_basis": buy_price * shares,
        }
        tw_portfolio = pd.concat(
            [tw_portfolio, pd.DataFrame([new_trade])], ignore_index=True
        )
    else:
        row_index = tw_portfolio[mask].index[0]
        current_shares = float(tw_portfolio.at[row_index, "shares"])
        tw_portfolio.at[row_index, "shares"] = current_shares + shares
        current_cost_basis = float(tw_portfolio.at[row_index, "cost_basis"])
        tw_portfolio.at[row_index, "cost_basis"] = shares * buy_price + current_cost_basis
        tw_portfolio.at[row_index, "stop_loss"] = stoploss
    cash = cash - shares * buy_price
    print(f"{ticker} 手動買入完成！")
    return cash, tw_portfolio


def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    tw_portfolio: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """記錄手動賣出並更新投資組合。"""
    reason = input(
        f"""您正要賣出 {int(shares_sold/1000)} 張 {ticker}，價格 {sell_price}。
如果這是錯誤，請輸入 1： """
    )

    if reason == "1":
        print("返回...")
        return cash, tw_portfolio
    if ticker not in tw_portfolio["ticker"].values:
        print(f"{ticker} 手動賣出失敗：投資組合中沒有此股票。")
        return cash, tw_portfolio
    ticker_row = tw_portfolio[tw_portfolio["ticker"] == ticker]

    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(
            f"{ticker} 手動賣出失敗：試圖賣出 {int(shares_sold/1000)} 張，但只持有 {int(total_shares/1000)} 張。"
        )
        return cash, tw_portfolio
    data = yf.download(ticker, period="1d")
    data = cast(pd.DataFrame, data)
    if data.empty:
        print(f"{ticker} 手動賣出失敗：無市場資料。")
        return cash, tw_portfolio
    day_high = float(data["High"].iloc[-1])
    day_low = float(data["Low"].iloc[-1])
    if not (day_low <= sell_price <= day_high):
        print(
            f"{ticker} 在 {sell_price} 手動賣出失敗：價格超出今日範圍 {round(day_low, 2)}-{round(day_high, 2)}。"
        )
        return cash, tw_portfolio
    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = sell_price * shares_sold - cost_basis
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": "",
        "Buy Price": "",
        "Cost Basis": cost_basis,
        "PnL": pnl,
        "Reason": f"手動賣出 - {reason}",
        "Shares Sold": shares_sold,
        "Sell Price": sell_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    if total_shares == shares_sold:
        tw_portfolio = tw_portfolio[tw_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        tw_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        tw_portfolio.at[row_index, "cost_basis"] = (
            tw_portfolio.at[row_index, "shares"]
            * tw_portfolio.at[row_index, "buy_price"]
        )

    cash = cash + shares_sold * sell_price
    print(f"{ticker} 手動賣出完成！")
    return cash, tw_portfolio


def daily_results(tw_portfolio: pd.DataFrame, cash: float) -> None:
    """印出每日價格更新和績效指標。"""
    portfolio_dict: list[dict[str, object]] = tw_portfolio.to_dict(orient="records")

    print(f"{today} 的價格和更新")
    
    # 台灣主要指數和ETF
    benchmark_tickers = [
        {"ticker": "^TWII"},     # 台灣加權指數
        {"ticker": "0050.TW"},   # 元大台灣50
        {"ticker": "006208.TW"}, # 富邦台50
        {"ticker": "00692.TW"},  # 富邦公司治理
    ]
    
    for stock in portfolio_dict + benchmark_tickers:
        ticker = stock["ticker"]
        try:
            data = yf.download(ticker, period="2d", progress=False)
            data = cast(pd.DataFrame, data)
            if data.empty or len(data) < 2:
                print(f"{ticker} 的資料為空或不完整。")
                continue
            price = float(data["Close"].iloc[-1].item())
            last_price = float(data["Close"].iloc[-2].item())

            percent_change = ((price - last_price) / last_price) * 100
            volume = float(data["Volume"].iloc[-1].item())
        except Exception as e:
            print(f"{ticker} 下載失敗。{e} 請檢查網路連線。")
            continue
            
        print(f"{ticker} 收盤價: {price:.2f}")
        print(f"{ticker} 今天成交量: {volume:,.0f}")
        print(f"較前一日變化: {percent_change:.2f}%")
        print("-" * 30)

    if not PORTFOLIO_CSV.exists():
        print("投資組合CSV檔案不存在。")
        return
        
    tw_df = pd.read_csv(PORTFOLIO_CSV)

    # 篩選總計行並取得最新淨值
    tw_totals = tw_df[tw_df["Ticker"] == "TOTAL"].copy()
    if tw_totals.empty:
        print("沒有找到總計資料。")
        return
        
    tw_totals["Date"] = pd.to_datetime(tw_totals["Date"])
    final_date = tw_totals["Date"].max()
    final_value = tw_totals[tw_totals["Date"] == final_date]
    final_equity = float(final_value["Total Equity"].values[0])
    equity_series = tw_totals["Total Equity"].astype(float).reset_index(drop=True)

    # 每日報酬率
    daily_pct = equity_series.pct_change().dropna()

    if len(equity_series) > 1:
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
    else:
        total_return = 0.0

    # 總交易日數
    n_days = len(tw_totals)

    # 無風險報酬率（假設台灣1年期定存利率1.5%）
    rf_annual = 0.015
    rf_period = (1 + rf_annual) ** (n_days / 252) - 1 if n_days > 0 else 0

    if len(daily_pct) > 0:
        # 每日報酬率標準差
        std_daily = daily_pct.std()
        negative_pct = daily_pct[daily_pct < 0]
        negative_std = negative_pct.std() if len(negative_pct) > 0 else 0
        
        # 夏普比率
        sharpe_total = (total_return - rf_period) / (std_daily * np.sqrt(n_days)) if std_daily > 0 and n_days > 0 else 0
        # 索提諾比率
        sortino_total = (total_return - rf_period) / (negative_std * np.sqrt(n_days)) if negative_std > 0 and n_days > 0 else 0
    else:
        sharpe_total = 0
        sortino_total = 0

    # 輸出
    print(f"總夏普比率（{n_days} 天）: {sharpe_total:.4f}")
    print(f"總索提諾比率（{n_days} 天）: {sortino_total:.4f}")
    print(f"最新投資組合淨值: NT${final_equity:,.0f}")
    
    # 取得台灣加權指數資料作為基準
    try:
        # 假設投資組合開始日期
        start_date = tw_totals["Date"].min().strftime("%Y-%m-%d")
        twii = yf.download("^TWII", start=start_date, end=final_date + pd.Timedelta(days=1), progress=False)
        twii = cast(pd.DataFrame, twii)
        twii = twii.reset_index()

        if not twii.empty and len(twii) > 1:
            # 標準化為相同起始金額
            initial_price = twii["Close"].iloc[0].item()
            price_now = twii["Close"].iloc[-1].item()
            initial_equity = equity_series.iloc[0]
            scaling_factor = initial_equity / initial_price
            twii_value = price_now * scaling_factor
            print(f"同期投資台灣加權指數: NT${twii_value:,.0f}")
        else:
            print("無法取得台灣加權指數資料")
    except Exception as e:
        print(f"取得台灣加權指數資料時發生錯誤: {e}")
    
    print(f"今日投資組合: {tw_portfolio}")
    print(f"現金餘額: NT${cash:,.0f}")

    print(
        "這是您今天的更新。您可以進行任何您認為合適的變更（如有必要），\n"
        "但您不能使用深度研究。您可以使用網路並檢查目前價格以尋找潛在買入機會。\n"
        "任何變更都需要您的許可，因為您擁有完全控制權。"
    )


def main(file: str, data_dir: Path | None = None) -> None:
    """執行交易腳本。

    Parameters
    ----------
    file:
        包含歷史投資組合記錄的 CSV 檔案。
    data_dir:
        儲存交易和投資組合 CSV 的目錄。
    """
    tw_portfolio, cash = load_latest_portfolio_state(file)
    if data_dir is not None:
        set_data_dir(data_dir)

    tw_portfolio, cash = process_portfolio(tw_portfolio, cash)
    daily_results(tw_portfolio, cash)

def load_latest_portfolio_state(
    file: str,
) -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """載入最新的投資組合快照和現金餘額。

    Parameters
    ----------
    file:
        包含歷史投資組合記錄的 CSV 檔案。

    Returns
    -------
    tuple[pd.DataFrame | list[dict[str, Any]], float]
        最新持倉的表示（空的 DataFrame 或行字典列表）和相關的現金餘額。
    """

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"找不到檔案 {file}，將創建新的投資組合。")
        df = pd.DataFrame()
    
    if df.empty:
        portfolio = pd.DataFrame([])
        print(
            "投資組合 CSV 為空。返回設定的現金金額來創建投資組合。"
        )
        try:
            cash = float(input("您希望起始現金金額為多少？ "))
        except ValueError:
            raise ValueError(
                "現金無法轉換為浮點數類型。請輸入有效數字。"
            )
        return portfolio, cash
    
    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"])

    latest_date = non_total["Date"].max()

    # 取得最新日期的所有股票代碼
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    latest_tickers.drop(columns=["Date", "Cash Balance", "Total Equity", "Action", "Current Price", "PnL", "Total Value"], inplace=True)
    latest_tickers.rename(columns={"Cost Basis": "buy_price", "Shares": "shares"}, inplace=True)
    latest_tickers['cost_basis'] = latest_tickers['shares'] * latest_tickers['buy_price']
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient='records')
    
    df = df[df["Ticker"] == "TOTAL"]  # 只有總計摘要行
    df["Date"] = pd.to_datetime(df["Date"])
    latest = df.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"])
    return latest_tickers, cash

if __name__ == "__main__":
    main("tw_portfolio_update.csv", Path.cwd())
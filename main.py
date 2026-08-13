import os
import sys
import pandas as pd
import yfinance as yf
from google import genai

TARGET_TICKERS = [
    "7203.T", "8306.T", "6098.T", "6857.T", "5803.T",
    "6981.T", "4062.T", "5801.T", "6920.T", "8136.T"
]

def fetch_stock_data(tickers):
    data_list = []
    print("データ取得中...")
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("currentPrice", info.get("previousClose", "N/A"))
            per = info.get("forwardPE", "N/A")
            pbr = info.get("priceToBook", "N/A")
            div_yield = info.get("dividendYield", 0)
            div_yield_pct = f"{div_yield * 100:.2f}%" if div_yield else "N/A"
            market_cap = info.get("marketCap", 0)
            market_cap_okoku = f"{market_cap / 100000000:.0f}億円" if market_cap else "N/A"
            
            data_text = (
                f"■ 銘柄コード: {symbol}\n"
                f"  現在株価: {price}円 | 予想PER: {per}倍 | PBR: {pbr}倍\n"
                f"  配当利回り: {div_yield_pct} | 時価総額: {market_cap_okoku}\n"
            )
            data_list.append(data_text)
        except Exception as e:
            print(f"警告: {symbol} のデータ取得に失敗しました: {e}")
            
    return "\n".join(data_list)

PROMPT_TEMPLATE = """
あなたは経験豊富な株式アナリストであり、初心者にも分かりやすくアドバイスを行う親切なファイナンシャルアドバイザーです。
以下に与えられた最新のリアルタイム市場数値データを基に、購入推奨度を数字（100点満点）で示して分析・比較し、結果を出力してください。

【プロが提供した最新市場データ】
{raw_data}

# 目的
注目度の高い日本株10社を同一の多角的な基準で比較評価し、初心者にとって最も安全かつ狙い目となる銘柄を特定する。

# 評価基準（合計100点満点）
1. 財務健全性・安全性（25点）
2. 業績・指標の割安度（20点）
3. モメンタム・ニュース性（20点）
4. テクニカル・チャート（20点）
5. プロの動向とリスク感度【信頼度向上チェック】（15点）

# 出力フォーマット
HTMLの <body> タグ内に埋め込みやすいよう、見やすいMarkdown（またはシンプルなHTML装飾）で出力してください。

■ 1. 注目10社 比較スコア一覧表
Markdownテーブル形式で出力してください。

■ 2. トップ2銘柄の売買アクションプラン（初心者向け）
上位1位〜2位の銘柄について、SBI証券で購入する場合の具体的プランを提示してください。
"""

def save_to_html(report_md):
    """分析結果をWeb用HTMLファイルとして保存"""
    # Simple CSSを適用してスマホでも見やすいデザインに整形
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>朝の日本株分析レポート</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/water.css">
    <style>
        body {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; }}
    </style>
</head>
<body>
    <h1>📈 本日の日本株 比較・分析レポート</h1>
    <p>更新日時: <script>document.write(new Date().toLocaleString("ja-JP"));</script></p>
    <hr>
    <div>
        {report_md.replace('\n', '<br>')}
    </div>
</body>
</html>
"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("index.html を保存しました。")

def run_analysis():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が設定されていません。")
        sys.exit(1)
        
    raw_stock_data = fetch_stock_data(TARGET_TICKERS)
    full_prompt = PROMPT_TEMPLATE.format(raw_data=raw_stock_data)
    
    print("Gemini API で分析を実行中...")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_prompt
    )
    
    save_to_html(response.text)

if __name__ == "__main__":
    run_analysis()

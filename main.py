import os
import sys
import pandas as pd
import yfinance as yf
from google import genai
# ---------------------------------------------------------
# 1. 安全なデータ取得対象銘柄（注目・主要銘柄）
# ---------------------------------------------------------
TARGET_TICKERS = [
    "7203.T",  # トヨタ自動車
    "8306.T",  # 三菱UFJフィナンシャル・グループ
    "6098.T",  # リクルートホールディングス
    "6857.T",  # アドバンテスト
    "5803.T",  # フジクラ
    "6981.T",  # 村田製作所
    "4062.T",  # イビデン
    "5801.T",  # 古河電気工業
    "6920.T",  # レーザーテック
    "8136.T",  # サンリオ
]
def fetch_stock_data(tickers):
    """yfinanceを使用して安全かつ規約遵守で指標・財務データを取得"""
    data_list = []
    print("データ取得中...")
    
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 各種数値データの抽出（取得できない場合はN/A）
            name = info.get("shortName", symbol)
            price = info.get("currentPrice", info.get("previousClose", "N/A"))
            per = info.get("forwardPE", "N/A")
            pbr = info.get("priceToBook", "N/A")
            equity_ratio = info.get("debtToEquity", "N/A") # 自己資本・負債指標の補助
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
# ---------------------------------------------------------
# 2. Gemini APIへのプロンプト作成と送信
# ---------------------------------------------------------
PROMPT_TEMPLATE = """
あなたは経験豊富な株式アナリストであり、初心者にも分かりやすくアドバイスを行う親切なファイナンシャルアドバイザーです。
以下に与えられた最新のリアルタイム市場数値データを基に、購入推奨度を数字（100点満点）で示して分析・比較し、結果を出力してください。
【プロが提供した最新市場データ】
{raw_data}
# 目的
注目度の高い日本株10社を同一の多角的な基準で比較評価し、初心者にとって最も安全かつ狙い目となる銘柄を特定する。
# 前提条件
・対象：与えられた日本株式10銘柄
・取引環境：SBI証券（単元株・S株/ミニ株の双方を考慮可）
・想定運用スタイル：短期〜中短期（数日から数週間程度での売却を想定）
# 評価基準（合計100点満点）
各銘柄を以下の5項目で総合的に評価し、合算したスコア（購入推奨度）を算出してください。
1. 財務健全性・安全性（25点）：自己資本・負債比率、財務基盤の強さ、倒産・無配リスクの低さ
2. 業績・指標の割安度（20点）：PER、PBR、配当利回りなどのバリュエーション
3. モメンタム・ニュース性（20点）：昨晩の海外市場影響、直近ニュース、業界動向
4. テクニカル・チャート（20点）：トレンドの向き、過熱感、出来高の推移、下値の硬さ
5. プロの動向とリスク感度【信頼度向上チェック】（15点）：機関投資家の空売り急増リスクの有無、アナリスト目標株価との乖離、為替感度、決算直後の「出尽くし売り」リスクの有無
# 出力フォーマット
■ 1. 注目10社 比較スコア一覧表
以下のMarkdownテーブル形式で出力してください。

| 順位 | 銘柄名（コード） | 購入推奨度 | 財務(25) | 割安度(20) | 材料(20) | チャート(20) | プロ/リスク(15) | 主な推奨ポイント/リスク |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

■ 2. トップ2銘柄の売買アクションプラン（初心者向け）
上位1位〜2位の銘柄について、SBI証券で購入する場合の具体的プランを提示してください。
--------------------------------------------------
【銘柄名（コード）】：
【売買区分】：単元株（100株単位） or S株（1株単位）
【推奨理由（スコアの根拠）】：
【購入想定株価】：〇〇円（指値／成行の推奨）
【購入推奨数量】：〇〇株（初心者のリスク管理に配慮した数量）
【目標株価（売却見込）】：〇〇円（＋〇%）
【損切りライン（撤退基準）】：〇〇円（−〇%）
【売却時期の目安】：〇〇
【注意すべきリスク・補足】：
--------------------------------------------------
# 注意事項
・専門用語を使用する場合は補足説明を入れてください。
・データにない確証のない情報については「未確認」と表記してください。
"""
def run_analysis():
    # GitHub Secrets から環境変数 GEMINI_API_KEY を取得
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が設定されていません。GitHub Secretsを確認してください。")
        sys.exit(1)
        
    # 1. データの安全収集
    raw_stock_data = fetch_stock_data(TARGET_TICKERS)
    
    # 2. プロンプトの組み立て
    full_prompt = PROMPT_TEMPLATE.format(raw_data=raw_stock_data)
    
    # 3. Gemini API (google-genai SDK) による推論
    print("Gemini API で分析を実行中...")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_prompt
    )
    
    # 4. 分析結果の出力
    print("\n==========================================")
    print("        本日の株価分析結果レポート        ")
    print("==========================================\n")
    print(response.text)
if __name__ == "__main__":
    run_analysis()

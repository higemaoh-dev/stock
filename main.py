import os
import sys
import yfinance as yf

# GenAI client: 複数のパッケージ名/呼び出し方に対応するためのインポート処理
try:
    # 一部の配布での import 形式
    from google import genai  # type: ignore
except Exception:
    try:
        # 公式パッケージ名での import 形式
        import google.generativeai as genai  # type: ignore
    except Exception:
        genai = None  # 後でチェック

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
            info = getattr(ticker, "info", {}) or {}
            price = info.get("currentPrice", info.get("previousClose", "N/A"))
            per = info.get("forwardPE", "N/A")
            pbr = info.get("priceToBook", "N/A")

            # dividendYield は None / 0 / 値 の可能性があるため明示的に処理
            div_yield = info.get("dividendYield", None)
            if div_yield is None:
                div_yield_pct = "N/A"
            else:
                # 0 -> "0.00%" と表示される
                div_yield_pct = f"{div_yield * 100:.2f}%"

            market_cap = info.get("marketCap", None)
            if market_cap is None or market_cap == 0:
                market_cap_okoku = "N/A"
            else:
                # 1億円単位に丸めて表示（例: 1234億円）
                market_cap_okoku = f"{market_cap / 100000000:.0f}億円"

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


def _extract_genai_text(resp):
    """
    GenAI の応答オブジェクトはバージョンにより形が異なるため、
    よくある形から生成テキストを取り出すユーティリティ関数。
    見つからなければ None を返す。
    """
    if resp is None:
        return None

    # 1) 直接 .text 属性
    try:
        text = getattr(resp, "text", None)
        if text:
            return text
    except Exception:
        pass

    # 2) candidates (リスト) を探す
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates and len(candidates) > 0:
            c0 = candidates[0]
            if isinstance(c0, dict):
                return c0.get("content") or c0.get("text")
            else:
                return getattr(c0, "content", None) or getattr(c0, "text", None)
    except Exception:
        pass

    # 3) outputs / output の一般的なネスト形式
    try:
        outputs = getattr(resp, "outputs", None) or getattr(resp, "output", None)
        if outputs:
            first = outputs[0]
            if isinstance(first, dict):
                # {'content': [{'type': 'output_text', 'text': '...'}], ...}
                content = first.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") in ("output_text", "text"):
                            return item.get("text") or item.get("content")
                return first.get("text") or first.get("content")
            else:
                return getattr(first, "text", None) or getattr(first, "content", None)
    except Exception:
        pass

    # 4) dict 形式のレスポンス
    try:
        if isinstance(resp, dict):
            if "text" in resp:
                return resp["text"]
            if "candidates" in resp and isinstance(resp["candidates"], list) and len(resp["candidates"]) > 0:
                c0 = resp["candidates"][0]
                if isinstance(c0, dict):
                    return c0.get("content") or c0.get("text")
    except Exception:
        pass

    return None


def run_analysis():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が設定されていません。")
        sys.exit(1)

    if genai is None:
        print("エラー: genai ライブラリが見つかりません。google-generativeai 等をインストールしてください。")
        sys.exit(1)

    raw_stock_data = fetch_stock_data(TARGET_TICKERS)
    full_prompt = PROMPT_TEMPLATE.format(raw_data=raw_stock_data)

    print("GenAI API で分析を実行中...")

    resp = None
    try:
        # 新しいスタイル: genai.Client がある場合
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=api_key)
            try:
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
            except TypeError:
                # contents がリストであることを要求するバージョンもある
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=[full_prompt])
        else:
            # 古い/別のインターフェース: configure -> generate_text/generate を試す
            if hasattr(genai, "configure"):
                genai.configure(api_key=api_key)
            if hasattr(genai, "generate_text"):
                try:
                    resp = genai.generate_text(model="gemini-2.5-flash", prompt=full_prompt)
                except TypeError:
                    resp = genai.generate_text(model="gemini-2.5-flash", text=full_prompt)
            elif hasattr(genai, "generate"):
                resp = genai.generate(model="gemini-2.5-flash", prompt=full_prompt)
            else:
                print("エラー: インストールされている genai のインターフェースが不明です。")
                sys.exit(1)
    except Exception as e:
        print(f"GenAI 呼び出しでエラーが発生しました: {e}")
        sys.exit(1)

    response_text = _extract_genai_text(resp)
    if not response_text:
        print("警告: GenAI の応答からテキストを抽出できませんでした。応答オブジェクトを表示します。")
        print(resp)
        sys.exit(1)

    save_to_html(response_text)


if __name__ == "__main__":
    run_analysis()

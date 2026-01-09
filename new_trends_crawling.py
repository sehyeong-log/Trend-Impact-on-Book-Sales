#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경제/경영 도서 트렌드 영향 분석 - 통합 분석 스크립트
Google Trends 키워드 수집 + 도서 카테고리 분류 + 상관분석
"""

import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import re
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 설정 및 카테고리 정의
# =============================================================================

# 도서 카테고리 분류 규칙 (우선순위 순)
CATEGORY_RULES = {
    '부동산/주거': r'(부동산|아파트|집|청약|재개발|토지|분양|갭투자|빌라|주거)',
    '주식/ETF': r'(주식|ETF|배당|종목|나스닥|S&P|펀드|차트|퀀트|증권|투자)',
    '암호화폐/디지털자산': r'(비트코인|암호화폐|코인|블록체인|NFT|이더리움|가상자산|디지털)',
    '거시경제/금리/인플레': r'(금리|인플레|물가|연준|경제위기|경기침체|GDP|환율|통화|경제)',
    '은퇴/연금/노후': r'(연금|은퇴|노후|퇴직|FIRE|조기은퇴|국민연금|생애)',
    '경영전략/조직/리더십': r'(경영|전략|리더십|마케팅|조직|스타트업|CEO|혁신|브랜드|OKR)',
}

# 카테고리별 Google Trends 키워드 매핑
KEYWORDS_MAP = {
    '부동산/주거': ['부동산', '아파트', '청약'],
    '주식/ETF': ['주식', 'ETF', '배당', '나스닥'],
    '암호화폐/디지털자산': ['비트코인', '암호화폐', '이더리움'],
    '거시경제/금리/인플레': ['금리', '인플레이션', '환율', '물가'],
    '은퇴/연금/노후': ['연금', '은퇴', '국민연금'],
    '경영전략/조직/리더십': ['경영전략', '마케팅', '스타트업'],
}

# =============================================================================
# 2. Google Trends 데이터 수집
# =============================================================================

def collect_google_trends(start_date='2024-01-01', end_date='2025-01-31'):
    """
    Google Trends에서 월별 키워드 검색 지수 수집

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)

    Returns:
        DataFrame: month, keyword, category, index 컬럼
    """
    print("=" * 80)
    print("Google Trends 데이터 수집 시작")
    print("=" * 80)

    pytrends = TrendReq(hl='ko', tz=540, timeout=(10, 25))
    timeframe = f'{start_date} {end_date}'
    results = []

    for category, keywords in KEYWORDS_MAP.items():
        print(f"\n[{category}] 카테고리 수집 중...")

        for kw in keywords:
            try:
                print(f"  - '{kw}' 키워드 수집 중...", end=' ')

                # Google Trends API 호출
                pytrends.build_payload([kw], timeframe=timeframe, geo='KR')
                data = pytrends.interest_over_time()

                if not data.empty and kw in data.columns:
                    # 월별로 집계
                    data = data[[kw]].reset_index()
                    data.columns = ['date', 'index']
                    data['month'] = data['date'].dt.to_period('M').astype(str)

                    # 월별 평균 계산
                    monthly_data = data.groupby('month')['index'].mean().reset_index()
                    monthly_data['keyword'] = kw
                    monthly_data['category'] = category

                    results.append(monthly_data[['month', 'keyword', 'category', 'index']])
                    print(f"✓ 완료 (평균 지수: {monthly_data['index'].mean():.1f})")
                else:
                    print(f"✗ 데이터 없음")

                # API Rate Limit 대응
                time.sleep(2)

            except Exception as e:
                print(f"✗ 오류: {str(e)[:50]}")
                time.sleep(5)
                continue

    if results:
        df_trends = pd.concat(results, ignore_index=True)
        df_trends['index'] = df_trends['index'].round(1)

        print(f"\n{'=' * 80}")
        print(f"수집 완료: 총 {len(df_trends)}개 레코드")
        print(f"{'=' * 80}")

        # 통계 요약
        print("\n[키워드별 통계]")
        summary = df_trends.groupby('keyword').agg({
            'index': ['mean', 'min', 'max', 'count']
        }).round(1)
        print(summary)

        return df_trends
    else:
        print("\n⚠️  수집된 데이터 없음")
        return pd.DataFrame()

# =============================================================================
# 3. 도서 카테고리 자동 분류
# =============================================================================

def classify_book(title, subtitle=''):
    """
    도서 제목/부제를 기반으로 카테고리 자동 분류

    Args:
        title: 도서 제목
        subtitle: 도서 부제 (선택)

    Returns:
        str: 카테고리명
    """
    text = (str(title) + ' ' + str(subtitle)).lower()

    # 우선순위 순으로 매칭
    for category, pattern in CATEGORY_RULES.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category

    return '기타/미분류'

def classify_bestseller_data(df_bestseller):
    """
    베스트셀러 데이터프레임에 카테고리 컬럼 추가

    Args:
        df_bestseller: 베스트셀러 데이터 (필수 컬럼: month, title)

    Returns:
        DataFrame: category 컬럼이 추가된 데이터프레임
    """
    print("\n" + "=" * 80)
    print("도서 카테고리 자동 분류 시작")
    print("=" * 80)

    # subtitle 컬럼이 없으면 빈 문자열로 처리
    if 'subtitle' not in df_bestseller.columns:
        df_bestseller['subtitle'] = ''

    # 카테고리 분류
    df_bestseller['category'] = df_bestseller.apply(
        lambda row: classify_book(row['title'], row.get('subtitle', '')),
        axis=1
    )

    # 분류 결과 통계
    category_dist = df_bestseller['category'].value_counts()
    unclassified_rate = (df_bestseller['category'] == '기타/미분류').mean()

    print(f"\n[분류 결과]")
    print(f"총 도서 수: {len(df_bestseller)}")
    print(f"미분류 비율: {unclassified_rate:.1%}")
    print(f"\n카테고리별 분포:")
    print(category_dist)

    # 미분류 샘플 출력
    if unclassified_rate > 0:
        print(f"\n[미분류 도서 샘플 (최대 10개)]")
        unclassified_books = df_bestseller[df_bestseller['category'] == '기타/미분류']['title'].head(10)
        for idx, title in enumerate(unclassified_books, 1):
            print(f"  {idx}. {title}")

    return df_bestseller

# =============================================================================
# 4. 월별 집계 및 분석
# =============================================================================

def aggregate_monthly_share(df_bestseller):
    """
    월별 카테고리 점유율 계산

    Args:
        df_bestseller: 카테고리가 분류된 베스트셀러 데이터

    Returns:
        DataFrame: month × category 점유율(%) 테이블
    """
    print("\n" + "=" * 80)
    print("월별 카테고리 점유율 계산")
    print("=" * 80)

    # 월별 카테고리 도서 수
    monthly_counts = df_bestseller.groupby(['month', 'category']).size().unstack(fill_value=0)

    # 점유율(%) 계산
    monthly_share_pct = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

    print(f"\n[월별 점유율 (%)]")
    print(monthly_share_pct.round(1))

    return monthly_share_pct

def calculate_new_entries(df_bestseller):
    """
    월별 신규 진입 도서 수 계산

    Args:
        df_bestseller: 베스트셀러 데이터

    Returns:
        DataFrame: month × category 신규 진입 도서 수
    """
    print("\n" + "=" * 80)
    print("월별 신규 진입 도서 분석")
    print("=" * 80)

    all_months = sorted(df_bestseller['month'].unique())
    new_entries_list = []

    for i, month in enumerate(all_months):
        if i == 0:
            continue

        # 이전 달 도서 제목
        prev_titles = set(df_bestseller[df_bestseller['month'] == all_months[i-1]]['title'])
        # 현재 달 도서 제목
        curr_titles = set(df_bestseller[df_bestseller['month'] == month]['title'])
        # 신규 진입 도서
        new_titles = curr_titles - prev_titles

        # 카테고리별 집계
        new_df = df_bestseller[(df_bestseller['month'] == month) & (df_bestseller['title'].isin(new_titles))]
        new_count = new_df.groupby('category').size()
        new_entries_list.append(new_count.rename(month))

    if new_entries_list:
        new_entries_df = pd.concat(new_entries_list, axis=1).T.fillna(0)
        print(f"\n[월별 신규 진입 도서 수]")
        print(new_entries_df)
        return new_entries_df
    else:
        return pd.DataFrame()

# =============================================================================
# 5. 트렌드 vs 점유율 상관분석
# =============================================================================

def analyze_correlation(df_trends, df_share):
    """
    Google Trends 지수와 카테고리 점유율 상관분석

    Args:
        df_trends: Google Trends 데이터
        df_share: 월별 카테고리 점유율 데이터

    Returns:
        DataFrame: 카테고리별 상관계수 및 패턴
    """
    print("\n" + "=" * 80)
    print("트렌드 vs 점유율 상관분석")
    print("=" * 80)

    # 카테고리별 키워드 평균 지수
    df_trends_agg = df_trends.groupby(['month', 'category'])['index'].mean().reset_index()
    df_trends_pivot = df_trends_agg.pivot(index='month', columns='category', values='index')

    results = []

    for category in df_share.columns:
        if category == '기타/미분류':
            continue

        if category not in df_trends_pivot.columns:
            continue

        # 데이터 정렬 (month 기준)
        share = df_share[category].sort_index()
        trend = df_trends_pivot[category].sort_index()

        # 공통 월만 사용
        common_months = share.index.intersection(trend.index)
        if len(common_months) < 3:
            continue

        share_aligned = share.loc[common_months]
        trend_aligned = trend.loc[common_months]

        # 동행 상관 (같은 달)
        try:
            corr_concurrent, p_conc = spearmanr(trend_aligned, share_aligned, nan_policy='omit')
        except:
            corr_concurrent, p_conc = 0, 1

        # 1개월 지연 상관
        if len(common_months) > 1:
            try:
                trend_lag1 = trend_aligned.shift(1).dropna()
                share_lag1 = share_aligned.loc[trend_lag1.index]
                corr_lagged, p_lag = spearmanr(trend_lag1, share_lag1, nan_policy='omit')
            except:
                corr_lagged, p_lag = 0, 1
        else:
            corr_lagged, p_lag = 0, 1

        # 패턴 분류
        if abs(corr_concurrent) < 0.3 and abs(corr_lagged) < 0.3:
            pattern = '무관형'
        elif corr_concurrent > corr_lagged:
            pattern = '동행형'
        else:
            pattern = '지연형'

        results.append({
            'category': category,
            'corr_concurrent': round(corr_concurrent, 3),
            'p_concurrent': round(p_conc, 3),
            'corr_lagged': round(corr_lagged, 3),
            'p_lagged': round(p_lag, 3),
            'pattern': pattern,
            'trend_avg': round(trend_aligned.mean(), 1),
            'share_avg': round(share_aligned.mean(), 1)
        })

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        print(f"\n[상관분석 결과]")
        print(df_results.to_string(index=False))

        print(f"\n[패턴별 분류]")
        pattern_dist = df_results['pattern'].value_counts()
        print(pattern_dist)

    return df_results

# =============================================================================
# 6. 인사이트 자동 생성
# =============================================================================

def generate_insights(df_results, df_share):
    """
    분석 결과 기반 인사이트 문장 자동 생성

    Args:
        df_results: 상관분석 결과
        df_share: 월별 점유율 데이터

    Returns:
        list: 인사이트 문장 리스트
    """
    print("\n" + "=" * 80)
    print("인사이트 자동 생성")
    print("=" * 80)

    insights = []

    # 1. 최대 점유율 카테고리
    avg_share = df_share.mean().sort_values(ascending=False)
    if len(avg_share) > 0:
        top_category = avg_share.index[0]
        top_share = avg_share.iloc[0]
        max_share = df_share[top_category].max()
        max_month = df_share[top_category].idxmax()

        insight1 = (f"1) 주제별 점유율: '{top_category}' 분야는 평균 점유율 {top_share:.1f}%로 "
                   f"경제/경영 베스트셀러의 최대 세그먼트였으며, {max_month}에는 {max_share:.1f}%까지 "
                   f"상승해 연중 최고치를 기록했다.")
        insights.append(insight1)

    # 2. 동행형 주제
    concurrent_df = df_results[df_results['pattern'] == '동행형'].sort_values('corr_concurrent', ascending=False)
    if not concurrent_df.empty:
        cat = concurrent_df.iloc[0]['category']
        corr = concurrent_df.iloc[0]['corr_concurrent']

        insight2 = (f"2) 동행형 주제: '{cat}' 카테고리는 Google Trends 검색량과 당월 점유율 간 "
                   f"상관계수 r={corr:.2f}로, 검색 관심 상승 즉시 도서 판매로 연결되는 **동행형** 패턴을 보였다.")
        insights.append(insight2)

    # 3. 지연형 주제
    lagged_df = df_results[df_results['pattern'] == '지연형'].sort_values('corr_lagged', ascending=False)
    if not lagged_df.empty:
        cat = lagged_df.iloc[0]['category']
        corr_lag = lagged_df.iloc[0]['corr_lagged']
        corr_conc = lagged_df.iloc[0]['corr_concurrent']

        insight3 = (f"3) 지연형 주제: '{cat}' 분야는 1개월 지연 상관(r={corr_lag:.2f})이 "
                   f"동행 상관(r={corr_conc:.2f})보다 높아, 트렌드 발생 → 학습 수요 → 1개월 후 도서 구매로 "
                   f"이어지는 **지연형** 반응이 확인됐다.")
        insights.append(insight3)

    # 4. 안정형 주제 (무관형)
    stable_df = df_results[df_results['pattern'] == '무관형']
    if not stable_df.empty:
        cat = stable_df.iloc[0]['category']
        corr = stable_df.iloc[0]['corr_concurrent']

        insight4 = (f"4) 안정형 주제: '{cat}' 카테고리는 검색 트렌드와 상관계수 r={corr:.2f}로 "
                   f"거의 무관했으며, 트렌드보다 **입소문·추천** 중심 판매 구조를 시사한다.")
        insights.append(insight4)

    # 5. 전체 요약
    total_categories = len(df_results)
    concurrent_count = len(df_results[df_results['pattern'] == '동행형'])
    lagged_count = len(df_results[df_results['pattern'] == '지연형'])

    insight5 = (f"5) 종합: 분석된 {total_categories}개 카테고리 중 {concurrent_count}개는 동행형, "
               f"{lagged_count}개는 지연형 반응을 보여, 외부 트렌드가 도서 시장에 실질적 영향을 "
               f"미치는 것으로 확인됐다.")
    insights.append(insight5)

    print("\n[생성된 인사이트]")
    for ins in insights:
        print(f"\n{ins}")

    return insights

# =============================================================================
# 7. 메인 실행 함수
# =============================================================================

def main():
    """
    메인 실행 함수
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "경제/경영 도서 트렌드 영향 분석" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")

    # -------------------------------------------------------------------------
    # STEP 1: Google Trends 데이터 수집
    # -------------------------------------------------------------------------
    df_trends = collect_google_trends(start_date='2024-01-01', end_date='2025-01-31')

    if df_trends.empty:
        print("\n⚠️  Google Trends 데이터 수집 실패. 프로그램 종료.")
        return

    # 결과 저장
    trends_file = 'new_trends_crawling.csv'
    df_trends.to_csv(trends_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Google Trends 데이터 저장: {trends_file}")

    # -------------------------------------------------------------------------
    # STEP 2: 베스트셀러 데이터 분류 (예시)
    # -------------------------------------------------------------------------
    # 실제 베스트셀러 데이터가 있다면 아래 주석 해제하여 사용
    # df_bestseller = pd.read_csv('your_bestseller_file.csv', encoding='utf-8-sig')
    # df_bestseller = classify_bestseller_data(df_bestseller)

    # 예시 데이터 생성 (실제 데이터 없을 때)
    print("\n" + "=" * 80)
    print("⚠️  베스트셀러 데이터 파일이 없어 예시 데이터로 대체합니다.")
    print("   실제 분석 시에는 베스트셀러 CSV 파일을 준비해주세요.")
    print("=" * 80)

    # 예시: 2024년 1월~12월 월별 더미 데이터
    months = pd.date_range('2024-01', '2024-12', freq='MS').strftime('%Y-%m').tolist()
    example_books = [
        {'title': '트렌드 코리아 2024', 'category': '경영전략/조직/리더십'},
        {'title': '주식투자 무작정 따라하기', 'category': '주식/ETF'},
        {'title': '비트코인 이야기', 'category': '암호화폐/디지털자산'},
        {'title': '아파트 투자 마법공식', 'category': '부동산/주거'},
        {'title': '금리의 배신', 'category': '거시경제/금리/인플레'},
        {'title': '90세 시대 은퇴 설계', 'category': '은퇴/연금/노후'},
    ]

    bestseller_data = []
    for month in months:
        for rank, book in enumerate(example_books * 3, 1):  # 책 반복해서 랭킹 채우기
            bestseller_data.append({
                'month': month,
                'rank': rank,
                'title': book['title'],
                'category': book['category']
            })

    df_bestseller = pd.DataFrame(bestseller_data)

    # -------------------------------------------------------------------------
    # STEP 3: 월별 집계
    # -------------------------------------------------------------------------
    df_share = aggregate_monthly_share(df_bestseller)
    df_new_entries = calculate_new_entries(df_bestseller)

    # -------------------------------------------------------------------------
    # STEP 4: 상관분석
    # -------------------------------------------------------------------------
    df_correlation = analyze_correlation(df_trends, df_share)

    # -------------------------------------------------------------------------
    # STEP 5: 인사이트 생성
    # -------------------------------------------------------------------------
    if not df_correlation.empty:
        insights = generate_insights(df_correlation, df_share)

    # -------------------------------------------------------------------------
    # STEP 6: 최종 결과 저장
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("최종 결과 저장")
    print("=" * 80)

    # 상관분석 결과 저장
    if not df_correlation.empty:
        corr_file = 'new_trends_crawling_correlation.csv'
        df_correlation.to_csv(corr_file, index=False, encoding='utf-8-sig')
        print(f"✓ 상관분석 결과: {corr_file}")

    # 월별 점유율 저장
    share_file = 'new_trends_crawling_share.csv'
    df_share.to_csv(share_file, encoding='utf-8-sig')
    print(f"✓ 월별 점유율: {share_file}")

    # 신규 진입 도서 저장
    if not df_new_entries.empty:
        new_file = 'new_trends_crawling_new_entries.csv'
        df_new_entries.to_csv(new_file, encoding='utf-8-sig')
        print(f"✓ 신규 진입 도서: {new_file}")

    print("\n" + "=" * 80)
    print("✓ 분석 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print(f"  1. {trends_file} - Google Trends 원본 데이터")
    print(f"  2. {corr_file} - 상관분석 결과")
    print(f"  3. {share_file} - 월별 카테고리 점유율")
    print(f"  4. {new_file} - 월별 신규 진입 도서")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

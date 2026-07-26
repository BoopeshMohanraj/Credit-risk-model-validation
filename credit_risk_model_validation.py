"""
Credit Risk Modeling & Model Validation
Altman Z-Score, Merton structural PD, Basel III EL/UL/EC/CVA (Monte Carlo Vasicek),
Kaplan-Meier survival, and SR 11-7 champion-challenger validation (Gini / AUC / KS / PSI).
Data: SEC EDGAR (financials), yfinance (equity), FRED (rates/spreads), 2015-2023.

Exported from Colab notebook. Requires FRED_API_KEY and a writable BASE_DIR.
"""

# ── USER CONFIG — change this to your own path ────────────────
BASE_DIR = '/content/drive/MyDrive/P4_CreditRisk'  # ← update this
# ──────────────────────────────────────────────────────────────


# ============================================================
# P4 PHASE 1 — INSTALL & SETUP
# ============================================================

import subprocess, sys

for pkg in ['yfinance','fredapi','plotly','scipy','statsmodels','kaleido']:
    subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])

import os, time, warnings, requests
import numpy  as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot    as plt
from datetime                  import datetime
from fredapi                   import Fred
from scipy                     import stats
from statsmodels.tsa.stattools import adfuller
from google.colab              import drive, userdata

warnings.filterwarnings('ignore')
pd.set_option('display.float_format','{:.4f}'.format)
pd.set_option('display.max_columns',30)

drive.mount('/content/drive', force_remount=False)

BASE_DIR   = '/content/drive/MyDrive/P4_CreditRisk'
CACHE_DIR  = os.path.join(BASE_DIR,'cache')
OUTPUT_DIR = os.path.join(BASE_DIR,'outputs')
LOG_DIR    = os.path.join(BASE_DIR,'logs')
for d in [BASE_DIR,CACHE_DIR,OUTPUT_DIR,LOG_DIR]:
    os.makedirs(d,exist_ok=True)

# Delete stale empty cache files
for f in ['edgar_financials.csv','financials_clean.csv']:
    fp = os.path.join(CACHE_DIR,f)
    if os.path.exists(fp) and os.path.getsize(fp)<200:
        os.remove(fp)
        print(f'  ✓ Deleted stale: {f}')

print(f'✓ Setup complete')
print(f'  Base  : {BASE_DIR}')
print(f'  Cache : {CACHE_DIR}')
print(f'  Output: {OUTPUT_DIR}')


# ============================================================
# P4 PHASE 1 — FULL DATA PIPELINE
# EDGAR · yfinance · FRED · ETFs · Quality checks · Viz
# ============================================================

# ── Parameter Registry ────────────────────────────────────────────────────────
PARAMS = {
    'start_date':'2015-01-01','end_date':'2023-12-31',
    'train_start':'2015-01-01','train_end':'2020-12-31',
    'test_start' :'2021-01-01','test_end' :'2023-12-31',
    'psi_dev_start':'2018-01-01','psi_dev_end':'2021-12-31',
    'psi_mon_start':'2022-01-01','psi_mon_end':'2023-12-31',
    'svb_quarters':['2022Q1','2022Q2','2022Q3'],
    'merton_T':1.0,'merton_vol_window':252,'merton_solver_tol':1e-8,'merton_max_iter':500,
    'zscore_w1':1.2,'zscore_w2':1.4,'zscore_w3':3.3,'zscore_w4':0.6,'zscore_w5':1.0,
    'zscore_safe':2.99,'zscore_distress':1.81,
    'lgd_senior_unsecured':0.40,'lgd_subordinated':0.75,'recovery_rate':0.60,
    'mc_n_paths':1000,'mc_n_steps':12,
    'vasicek_kappa':0.15,'vasicek_theta':0.035,'vasicek_sigma':0.01,
    'psi_bins':10,'psi_stable':0.10,'psi_monitor':0.25,'psi_epsilon':1e-4,
    'gini_good':0.60,'gini_moderate':0.40,'ks_good':0.40,'ks_moderate':0.20,
    'min_quarters':4,'outlier_zscore_cap':5.0,'return_winsor_pct':0.01,
    'min_price':0.10,'vol_window_min':60,
    'sec_user_agent':'Boopesh Mohanraj boopesh@northeastern.edu','sec_sleep':0.15,
    'fred_series_rf':'DGS1','fred_series_10y':'DGS10',
    'fred_series_spread':'BAMLH0A0HYM2','fred_series_ig':'BAMLC0A0CM',
    'random_seed':42,
}
Z_SAFE     = PARAMS['zscore_safe']
Z_DISTRESS = PARAMS['zscore_distress']
print('✓ Parameters loaded')

# ── Universe ──────────────────────────────────────────────────────────────────
DELISTED_CIK = {
    'SIVB':'0000719739','SBNY':'0001288946','FRC':'0001408278',
    'BBBY':'0000886158','HTZ':'0000047987','CBL':'0000907254',
    'WPG':'0001594686','CHK':'0000895126',
}
DEFAULT_EVENTS = {
    'CHK':'2020-06-28','HTZ':'2020-05-22','BBBY':'2023-04-23',
    'SIVB':'2023-03-10','SBNY':'2023-03-12','FRC':'2023-05-01',
    'CBL':'2020-11-01','WPG':'2021-06-13',
}
UNIVERSE_RECORDS = [
    ('AAPL','Apple','Tech','ig',True),('MSFT','Microsoft','Tech','ig',True),
    ('NVDA','NVIDIA','Tech','ig',True),('GOOGL','Alphabet','Tech','ig',True),
    ('META','Meta Platforms','Tech','ig',True),('AMZN','Amazon','Tech','ig',True),
    ('CRM','Salesforce','Tech','grey',True),('ORCL','Oracle','Tech','grey',True),
    ('JPM','JPMorgan Chase','Banks','ig',True),('BAC','Bank of America','Banks','ig',True),
    ('GS','Goldman Sachs','Banks','ig',True),('WFC','Wells Fargo','Banks','ig',True),
    ('COF','Capital One','Banks','grey',True),('WAL','Western Alliance','Banks','dist',True),
    ('PACW','PacWest Bancorp','Banks','dist',True),('SIVB','SVB Financial','Banks','def',True),
    ('SBNY','Signature Bank','Banks','def',True),('FRC','First Republic','Banks','def',True),
    ('XOM','ExxonMobil','Energy','ig',True),('CVX','Chevron','Energy','ig',True),
    ('DVN','Devon Energy','Energy','ig',True),('HAL','Halliburton','Energy','grey',True),
    ('APA','APA Corp','Energy','grey',True),('OXY','Occidental Petrol','Energy','dist',True),
    ('MRO','Marathon Oil','Energy','dist',True),('CHK','Chesapeake Energy','Energy','def',True),
    ('WMT','Walmart','Retail','ig',True),('COST','Costco','Retail','ig',True),
    ('TGT','Target','Retail','ig',True),('M','Macys','Retail','grey',True),
    ('KSS','Kohls','Retail','grey',True),('DG','Dollar General','Retail','grey',True),
    ('GME','GameStop','Retail','dist',True),('AMC','AMC Entertainment','Retail','dist',True),
    ('BBBY','Bed Bath & Beyond','Retail','def',True),
    ('JNJ','Johnson & Johnson','Healthcare','ig',True),('UNH','UnitedHealth','Healthcare','ig',True),
    ('PFE','Pfizer','Healthcare','ig',True),('ABBV','AbbVie','Healthcare','ig',True),
    ('HCA','HCA Healthcare','Healthcare','grey',True),('THC','Tenet Healthcare','Healthcare','grey',True),
    ('ENVX','Enovis Corp','Healthcare','grey',True),('CLOV','Clover Health','Healthcare','dist',True),
    ('CAT','Caterpillar','Industrial','ig',True),('GE','General Electric','Industrial','ig',True),
    ('BA','Boeing','Industrial','grey',True),('GM','General Motors','Industrial','grey',True),
    ('F','Ford','Industrial','grey',True),('AAL','American Airlines','Industrial','dist',True),
    ('HTZ','Hertz Global','Industrial','def',True),
    ('SPG','Simon Property Grp','Real Estate','ig',True),
    ('CBL','CBL & Associates','Real Estate','def',True),
    ('WPG','Washington Prime','Real Estate','def',True),
    ('VZ','Verizon','Telecom','ig',True),('T','AT&T','Telecom','grey',True),
    ('LUMN','Lumen Technologies','Telecom','dist',True),('DISH','EchoStar/DISH','Telecom','dist',True),
    ('HYG','iShares HY Bond ETF','ETF','etf',False),
    ('LQD','iShares IG Bond ETF','ETF','etf',False),
    ('JNK','SPDR Bloomberg HY ETF','ETF','etf',False),
]
UNIVERSE = pd.DataFrame(UNIVERSE_RECORDS,
                        columns=['ticker','company','sector','tier','in_merton'])
UNIVERSE['is_default']   = UNIVERSE['ticker'].isin(DEFAULT_EVENTS)
UNIVERSE['default_date'] = UNIVERSE['ticker'].map(DEFAULT_EVENTS)
UNIVERSE['cik']          = UNIVERSE['ticker'].map(DELISTED_CIK)
UNIVERSE.index = range(1,len(UNIVERSE)+1)
MERTON_TICKERS  = UNIVERSE[UNIVERSE['in_merton']]['ticker'].tolist()
ETF_TICKERS     = UNIVERSE[~UNIVERSE['in_merton']]['ticker'].tolist()
DEFAULT_TICKERS = UNIVERSE[UNIVERSE['is_default']]['ticker'].tolist()
print(f'✓ Universe: {len(UNIVERSE)} entities · {len(MERTON_TICKERS)} Merton · {len(DEFAULT_TICKERS)} defaults')

# ── SEC EDGAR helpers ─────────────────────────────────────────────────────────
SEC_HEADERS = {
    'User-Agent':PARAMS['sec_user_agent'],
    'Accept-Encoding':'gzip, deflate',
}
CONCEPTS = {
    'total_assets'  :['Assets','TotalAssets'],
    'total_debt'    :['DebtLongtermAndShorttermCombinedAmount',
                      'LongTermDebtAndCapitalLeaseObligations','LongTermDebt',
                      'LongTermDebtNoncurrent','DebtAndCapitalLeaseObligations'],
    'current_assets':['AssetsCurrent','CashAndCashEquivalentsAtCarryingValue'],
    'current_liab'  :['LiabilitiesCurrent','ShortTermBorrowings'],
    'retained_earn' :['RetainedEarningsAccumulatedDeficit','RetainedEarnings'],
    'revenue'       :['Revenues','RevenueFromContractWithCustomerExcludingAssessedTax',
                      'SalesRevenueNet','InterestAndDividendIncomeOperating',
                      'RevenueFromContractWithCustomerIncludingAssessedTax',
                      'HealthCareOrganizationRevenue','OilAndGasRevenue'],
    'ebit'          :['OperatingIncomeLoss',
                      'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
    'net_income'    :['NetIncomeLoss','ProfitLoss'],
    'da'            :['DepreciationDepletionAndAmortization','DepreciationAndAmortization'],
    'cfo'           :['NetCashProvidedByUsedInOperatingActivities'],
    'capex'         :['PaymentsToAcquirePropertyPlantAndEquipment'],
    'total_liab'    :['Liabilities'],
    'shares_out'    :['CommonStockSharesOutstanding'],
}
DEBT_LT = ['LongTermDebt','LongTermDebtNoncurrent','LongTermDebtAndCapitalLeaseObligations']
DEBT_ST = ['ShortTermBorrowings','LongTermDebtCurrent','CommercialPaper']
EBIT_NI = ['NetIncomeLoss']
EBIT_IE = ['InterestExpense','InterestAndDebtExpense']
EBIT_TX = ['IncomeTaxExpenseBenefit','IncomeTaxExpense']

def get_cik_map():
    try:
        r = requests.get('https://www.sec.gov/files/company_tickers.json',
                         headers=SEC_HEADERS,timeout=15)
        m = {v['ticker'].upper():str(v['cik_str']).zfill(10) for v in r.json().values()}
        m.update(DELISTED_CIK)
        m.update({'PACW':'0001102266','SBNY':'0001288946','MRO':'0000101778',
                  'HTZ':'0000047987','DISH':'0001001082','SIVB':'0000719739',
                  'WAL':'0001212545','JPM':'0000019617','GS':'0000886982',
                  'BAC':'0000070858','WFC':'0000072971','COF':'0000927628',
                  'XOM':'0000034088','DVN':'0001090012','HAL':'0000045012',
                  'HCA':'0000860730','WPG':'0001594686','ENVX':'0001628171'})
        return m
    except:
        return {**DELISTED_CIK}

def get_facts(cik):
    r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
                     headers=SEC_HEADERS,timeout=30)
    return r.json() if r.status_code==200 else None

def best_series(us_gaap,candidates):
    best,best_n = {},0
    for c in candidates:
        if c not in us_gaap: continue
        entries = us_gaap[c].get('units',{}).get('USD',
                  us_gaap[c].get('units',{}).get('shares',[]))
        pm = {}
        for e in entries:
            if e.get('form','') not in ('10-K','10-Q'): continue
            end,val,filed = e.get('end',''),e.get('val'),e.get('filed','')
            if not end or val is None: continue
            if end not in pm or filed>pm[end]['filed']:
                pm[end]={'val':float(val),'filed':filed}
        if len(pm)>best_n: best,best_n=pm,len(pm)
    return best

def extract_facts(facts_json,ticker):
    if not facts_json: return []
    us = facts_json.get('facts',{}).get('us-gaap',{})
    if not us: return []
    fm = {f:best_series(us,c) for f,c in CONCEPTS.items()}
    lt=best_series(us,DEBT_LT); st=best_series(us,DEBT_ST)
    if lt and st:
        comp={}
        for p in set(lt)|set(st):
            comp[p]={'val':(lt.get(p,{}).get('val',0) or 0)+(st.get(p,{}).get('val',0) or 0),
                     'filed':max(lt.get(p,{}).get('filed',''),st.get(p,{}).get('filed',''))}
        if len(comp)>len(fm.get('total_debt',{})): fm['total_debt']=comp
    if len(fm.get('ebit',{}))<4:
        ni=best_series(us,EBIT_NI); ie=best_series(us,EBIT_IE); tx=best_series(us,EBIT_TX)
        if ni:
            ce={p:{'val':ni[p]['val']+(ie.get(p,{}).get('val',0) or 0)+(tx.get(p,{}).get('val',0) or 0),
                   'filed':ni[p]['filed']} for p in ni}
            if len(ce)>len(fm.get('ebit',{})): fm['ebit']=ce
    all_p=set()
    for pm in fm.values(): all_p.update(pm.keys())
    if not all_p: return []
    rows=[]
    for p in sorted(all_p):
        row={'ticker':ticker,'period_end':p,'filed_date':None}
        for f in CONCEPTS:
            pm=fm.get(f,{})
            if p in pm: row[f]=pm[p]['val']; row['filed_date']=pm[p]['filed']
            else: row[f]=np.nan
        rows.append(row)
    return rows

def clean_financials(df):
    if len(df)==0: return df
    df=df.copy()
    df['period_end']=pd.to_datetime(df['period_end'],errors='coerce')
    df=df.dropna(subset=['period_end'])
    df['quarter']=df['period_end'].dt.to_period('Q').astype(str)
    df['year']=df['period_end'].dt.year
    df=df[(df['period_end']>=PARAMS['start_date'])&(df['period_end']<=PARAMS['end_date'])]
    dollar_cols=['total_assets','total_debt','current_assets','current_liab',
                 'retained_earn','revenue','ebit','net_income','da','cfo','capex','total_liab']
    for col in dollar_cols:
        if col not in df.columns: continue
        df[col]=pd.to_numeric(df[col],errors='coerce')
        med=df[col].median()
        if pd.notna(med) and abs(med)>1e8: df[col]=df[col]/1e6
    df['filed_date']=pd.to_datetime(df['filed_date'],errors='coerce')
    df=(df.sort_values('filed_date',ascending=False)
          .drop_duplicates(subset=['ticker','quarter'],keep='first')
          .sort_values(['ticker','period_end'])
          .reset_index(drop=True))
    for col in dollar_cols:
        if col not in df.columns: continue
        mu,sig=df[col].mean(),df[col].std()
        if pd.notna(sig) and sig>0:
            df[col]=df[col].clip(lower=mu-PARAMS['outlier_zscore_cap']*sig,
                                  upper=mu+PARAMS['outlier_zscore_cap']*sig)
    return df

# ── EDGAR pipeline ────────────────────────────────────────────────────────────
EDGAR_CACHE = os.path.join(CACHE_DIR,'edgar_financials.csv')
if os.path.exists(EDGAR_CACHE) and os.path.getsize(EDGAR_CACHE)>500:
    print('→ Loading EDGAR cache...')
    financials_raw=pd.read_csv(EDGAR_CACHE)
    print(f'✓ {len(financials_raw):,} rows · {financials_raw["ticker"].nunique()} companies')
else:
    print(f'→ Extracting EDGAR for {len(MERTON_TICKERS)} companies (~5-10 min)...')
    cik_map=get_cik_map()
    all_records,parse_log=[],[]
    for i,row in UNIVERSE[UNIVERSE['in_merton']].iterrows():
        t=row['ticker']; cik=cik_map.get(t)
        print(f'  [{i:02d}/{len(MERTON_TICKERS)}] {t:<6}',end='',flush=True)
        if not cik: print(' ✗ no CIK'); parse_log.append({'ticker':t,'status':'no_cik'}); continue
        try:
            recs=extract_facts(get_facts(cik),t)
            time.sleep(PARAMS['sec_sleep'])
            if recs: all_records.extend(recs); print(f' ✓ {len(recs)}'); parse_log.append({'ticker':t,'status':'ok','n':len(recs)})
            else: print(' ✗ no data'); parse_log.append({'ticker':t,'status':'no_data'})
        except Exception as e: print(f' ✗ {e}'); parse_log.append({'ticker':t,'status':'error'}); time.sleep(1)
    financials_raw=pd.DataFrame(all_records) if all_records else pd.DataFrame()
    financials_raw.to_csv(EDGAR_CACHE,index=False)
    pd.DataFrame(parse_log).to_csv(os.path.join(LOG_DIR,'edgar_parse_log.csv'),index=False)
    n_ok=(pd.DataFrame(parse_log)['status']=='ok').sum()
    print(f'\n✓ EDGAR done — {n_ok}/{len(MERTON_TICKERS)} parsed')

financials_df=clean_financials(financials_raw)
financials_df.to_csv(os.path.join(CACHE_DIR,'financials_clean.csv'),index=False)
print(f'✓ Financials: {len(financials_df):,} rows · {financials_df["ticker"].nunique()} companies')

# ── yfinance ──────────────────────────────────────────────────────────────────
EQUITY_CACHE=os.path.join(CACHE_DIR,'equity_data.csv')
VOL_CACHE=os.path.join(CACHE_DIR,'equity_vol.csv')
MCAP_CACHE=os.path.join(CACHE_DIR,'market_cap.csv')
ALL_TICKERS=MERTON_TICKERS+ETF_TICKERS

if os.path.exists(EQUITY_CACHE) and os.path.getsize(EQUITY_CACHE)>500:
    equity_df=pd.read_csv(EQUITY_CACHE,parse_dates=['Date'],index_col='Date')
    print(f'✓ Equity cache: {equity_df.shape}')
else:
    print('→ Downloading equity data...')
    raw=yf.download(ALL_TICKERS,start=PARAMS['start_date'],end=PARAMS['end_date'],
                    auto_adjust=True,progress=True)
    equity_df=raw['Close'].copy()
    equity_df.to_csv(EQUITY_CACHE)
    print(f'✓ Downloaded: {equity_df.shape}')

returns_df=np.log(equity_df/equity_df.shift(1)).dropna(how='all')
lq=returns_df.quantile(PARAMS['return_winsor_pct'])
hq=returns_df.quantile(1-PARAMS['return_winsor_pct'])
returns_df=returns_df.clip(lower=lq,upper=hq,axis=1)

if os.path.exists(VOL_CACHE) and os.path.getsize(VOL_CACHE)>500:
    vol_df=pd.read_csv(VOL_CACHE,parse_dates=['Date'],index_col='Date')
else:
    vol_df=(returns_df.rolling(window=PARAMS['merton_vol_window'],
                                min_periods=PARAMS['vol_window_min'])
                       .std()*np.sqrt(252))
    vol_df.to_csv(VOL_CACHE)

if os.path.exists(MCAP_CACHE) and os.path.getsize(MCAP_CACHE)>100:
    mcap_df=pd.read_csv(MCAP_CACHE,parse_dates=['date'])
else:
    print('→ Fetching market cap...')
    mcap_records=[]
    for t in MERTON_TICKERS:
        try:
            shares=getattr(yf.Ticker(t).fast_info,'shares',None)
            if shares and t in equity_df.columns:
                for dt,price in equity_df[t].resample('Q').last().items():
                    if pd.notna(price):
                        mcap_records.append({'ticker':t,'date':dt,'market_cap_M':price*shares/1e6})
        except: pass
        time.sleep(0.2)
    mcap_df=pd.DataFrame(mcap_records)
    mcap_df.to_csv(MCAP_CACHE,index=False)
print(f'✓ Equity ready: prices={equity_df.shape} · vol={vol_df.shape}')

# ── FRED ──────────────────────────────────────────────────────────────────────
FRED_CACHE=os.path.join(CACHE_DIR,'fred_rates.csv')
try:
    FRED_API_KEY=userdata.get('FRED_API_KEY')
    print('✓ FRED API key loaded')
except:
    raise ValueError('Add FRED_API_KEY to Colab Secrets (🔑 icon)')

if os.path.exists(FRED_CACHE) and os.path.getsize(FRED_CACHE)>500:
    rf_df=pd.read_csv(FRED_CACHE,parse_dates=['date'],index_col='date')
    print(f'✓ FRED cache: {rf_df.shape}')
else:
    fred=Fred(api_key=FRED_API_KEY)
    series={'rf_1y':PARAMS['fred_series_rf'],'treasury_10y':PARAMS['fred_series_10y'],
            'hy_spread':PARAMS['fred_series_spread'],'ig_spread':PARAMS['fred_series_ig']}
    rf_df=pd.concat([fred.get_series(sid,observation_start=PARAMS['start_date'],
                                      observation_end=PARAMS['end_date']).rename(n)
                     for n,sid in series.items()],axis=1)
    rf_df.index.name='date'; rf_df=rf_df/100.0; rf_df=rf_df.ffill()
    rf_df.to_csv(FRED_CACHE)
    print(f'✓ FRED downloaded: {rf_df.shape}')
print(f'  1Y RF: {rf_df["rf_1y"].dropna().iloc[-1]*100:.2f}%  HY spread: {rf_df["hy_spread"].dropna().iloc[-1]*100:.2f}%')

etf_quarterly=returns_df[ETF_TICKERS].resample('Q').sum()
etf_quarterly.to_csv(os.path.join(CACHE_DIR,'etf_data.csv'))

# ── Save all outputs ──────────────────────────────────────────────────────────
financials_df.to_csv(os.path.join(OUTPUT_DIR,'financials_quarterly.csv'),index=False)
returns_df.to_csv(os.path.join(OUTPUT_DIR,'equity_returns_daily.csv'))
vol_df.to_csv(os.path.join(OUTPUT_DIR,'equity_vol_rolling.csv'))
rf_df.to_csv(os.path.join(OUTPUT_DIR,'fred_rates.csv'))
UNIVERSE.to_csv(os.path.join(OUTPUT_DIR,'universe.csv'))
etf_quarterly.to_csv(os.path.join(OUTPUT_DIR,'etf_quarterly_returns.csv'))
mcap_df.to_csv(os.path.join(CACHE_DIR,'market_cap.csv'),index=False)
pd.DataFrame([{'parameter':k,'value':str(v)} for k,v in PARAMS.items()]
             ).to_csv(os.path.join(OUTPUT_DIR,'parameter_registry.csv'),index=False)

# ── ADF stationarity ──────────────────────────────────────────────────────────
print('\n── Stationarity (ADF) ──')
stat_res=[]
for col in rf_df.columns:
    s=rf_df[col].dropna()
    if len(s)<20: continue
    adf,p,_,_,_,_=adfuller(s,autolag='AIC')
    stat_res.append({'series':col,'adf':round(adf,4),'p':round(p,4),
                     'stationary':p<0.05,'action':'use as-is' if p<0.05 else 'first difference'})
    print(f'  {"✓" if p<0.05 else "⚠"} {col:<20} p={p:.4f}  {"STATIONARY" if p<0.05 else "→ difference"}')
pd.DataFrame(stat_res).to_csv(os.path.join(OUTPUT_DIR,'stationarity_results.csv'),index=False)

print(f'\n✓ Phase 1 complete — all outputs saved to {OUTPUT_DIR}')
print(f'  Equity range: {equity_df.index.min().date()} → {equity_df.index.max().date()}')
print(f'  EDGAR companies: {financials_df["ticker"].nunique()}')


# ============================================================
# P4 PHASE 2 — ALTMAN Z-SCORE + ALL VISUALIZATIONS
# ============================================================

import os, warnings
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy        import stats
from google.colab import drive

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white',
                     'axes.facecolor':'white','axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.2})
drive.mount('/content/drive',force_remount=False)

BASE_DIR   = '/content/drive/MyDrive/P4_CreditRisk'
OUTPUT_DIR = os.path.join(BASE_DIR,'outputs')
CACHE_DIR  = os.path.join(BASE_DIR,'cache')

# Load
financials_df = pd.read_csv(os.path.join(OUTPUT_DIR,'financials_quarterly.csv'))
universe_df   = pd.read_csv(os.path.join(OUTPUT_DIR,'universe.csv'))
equity_df     = pd.read_csv(os.path.join(OUTPUT_DIR,'equity_returns_daily.csv'),
                             parse_dates=['Date'],index_col='Date')
mcap_df       = pd.read_csv(os.path.join(CACHE_DIR,'market_cap.csv'),parse_dates=['date'])

financials_df['period_end']=pd.to_datetime(financials_df['period_end'],errors='coerce')
financials_df['quarter']=financials_df['period_end'].dt.to_period('Q').astype(str)
mcap_df['quarter']=pd.to_datetime(mcap_df['date']).dt.to_period('Q').astype(str)

Z_SAFE,Z_DISTRESS = 2.99,1.81
W1,W2,W3,W4,W5   = 1.2,1.4,3.3,0.6,1.0
MERTON_TICKERS = universe_df[universe_df['in_merton']==True]['ticker'].tolist()
BANK_TICKERS   = universe_df[universe_df['sector']=='Banks']['ticker'].tolist()
REIT_TICKERS   = universe_df[universe_df['sector']=='Real Estate']['ticker'].tolist()
sector_map     = universe_df.set_index('ticker')['sector'].to_dict()
tier_map       = universe_df.set_index('ticker')['tier'].to_dict()

C_SAFE='#2D6A27'; C_GREY='#B07D10'; C_DIST='#C0392B'; C_ACC='#2C3E50'
C_SAFE_BG='#F0F7EC'; C_GREY_BG='#FDF6E3'; C_DIST_BG='#FDF0EF'

print('✓ Data loaded')

# ── BLOCK 1: Compute Z-Scores on complete grid ────────────────────────────────
print('\n→ Computing Z-Scores...')
all_quarters=[str(q) for q in pd.period_range('2015Q1','2023Q4',freq='Q')]
grid=pd.MultiIndex.from_product([MERTON_TICKERS,all_quarters],
                                 names=['ticker','quarter']).to_frame(index=False)

fin=financials_df.copy()
m=fin['ebit'].isna()&fin['net_income'].notna()
fin.loc[m,'ebit']=fin.loc[m,'net_income']+fin.loc[m,'da'].fillna(0)

keep=['ticker','quarter','total_assets','total_liab','current_assets','current_liab',
      'retained_earn','revenue','ebit','cfo','net_income','da']
fin=grid.merge(fin[[c for c in keep if c in fin.columns]],on=['ticker','quarter'],how='left')
fin=fin.sort_values(['ticker','quarter']).reset_index(drop=True)
fin['was_ffilled']=fin['total_assets'].isna()

fill_cols=[c for c in ['total_assets','total_liab','current_assets','current_liab',
                        'retained_earn','revenue','ebit','cfo','net_income','da']
           if c in fin.columns]
fin[fill_cols]=fin.groupby('ticker')[fill_cols].transform(lambda x:x.ffill().bfill())

fin=fin.merge(mcap_df[['ticker','quarter','market_cap_M']],on=['ticker','quarter'],how='left')
fin['market_cap_M']=fin.groupby('ticker')['market_cap_M'].transform(lambda x:x.ffill().bfill())
fin=fin.merge(universe_df[['ticker','sector','tier']],on='ticker',how='left')

fin['book_equity']=(fin['total_assets']-fin['total_liab'].fillna(fin['total_assets']*0.5)
                    ).clip(upper=fin['total_assets']*2)
fin['eq4']=np.where(fin['market_cap_M'].notna()&(fin['market_cap_M']>0),
                    fin['market_cap_M'],fin['book_equity'])

fin['X1_raw']=(fin['current_assets']-fin['current_liab'])/fin['total_assets']
fin['X2_raw']=fin['retained_earn']/fin['total_assets']
fin['X3_raw']=fin['ebit']/fin['total_assets']
fin['X4_raw']=fin['eq4']/fin['total_liab'].replace(0,np.nan)
fin['X5_raw']=fin['revenue']/fin['total_assets']

m5=fin['X5_raw'].isna()&fin['cfo'].notna()
fin.loc[m5,'X5_raw']=fin.loc[m5,'cfo']/fin.loc[m5,'total_assets']

for c,b in [('X1_raw',(-1,1)),('X2_raw',(-2,2)),('X3_raw',(-0.5,0.5)),
            ('X4_raw',(0,5)),('X5_raw',(0,4))]:
    fin[c]=fin[c].clip(*b)

for c in ['X1_raw','X2_raw','X3_raw','X4_raw','X5_raw']:
    fin[c.replace('_raw','_imp')]=fin[c].isna()
fin['n_imp']=fin[['X1_imp','X2_imp','X3_imp','X4_imp','X5_imp']].sum(axis=1)

fin['X1']=fin['X1_raw'].fillna(0)
fin['X2']=fin['X2_raw'].fillna(0)
fin['X3']=fin['X3_raw'].fillna(0)
fin['X4']=fin['X4_raw'].fillna(0)
fin['X5']=fin['X5_raw'].fillna(0)

has_data=fin['total_assets'].notna()&(fin['total_assets']>0)
fin['Z_score']=np.where(has_data,
    W1*fin['X1']+W2*fin['X2']+W3*fin['X3']+W4*fin['X4']+W5*fin['X5'],np.nan)

def classify(z):
    if pd.isna(z): return 'unknown'
    if z>Z_SAFE:   return 'safe'
    if z<Z_DISTRESS: return 'distress'
    return 'grey'

fin['zone']=fin['Z_score'].apply(classify)
fin['bank_flag']=fin['ticker'].isin(BANK_TICKERS)
fin['z_reliable']=(fin['n_imp']<=1)&(~fin['was_ffilled'])

nan_r=fin['Z_score'].isna().mean()*100
print(f'  NaN rate: {nan_r:.1f}%')

svb=fin[(fin['ticker']=='SIVB')&(fin['quarter']=='2022Q3')]
nvda=fin[(fin['ticker']=='NVDA')&(fin['quarter']=='2023Q1')]
if len(svb)>0:  print(f'  SVB Q3 2022 : Z={svb.iloc[0]["Z_score"]:.4f}  {svb.iloc[0]["zone"].upper()}')
if len(nvda)>0: print(f'  NVDA Q1 2023: Z={nvda.iloc[0]["Z_score"]:.4f}  {nvda.iloc[0]["zone"].upper()}')

# Save
cols=['ticker','sector','tier','quarter','X1','X2','X3','X4','X5',
      'Z_score','zone','bank_flag','z_reliable','n_imp','was_ffilled',
      'X1_imp','X2_imp','X3_imp','X4_imp','X5_imp']
zscore_df=fin[cols].copy()
zscore_df.to_csv(os.path.join(OUTPUT_DIR,'zscore_quarterly.csv'),index=False)

latest_q=zscore_df.dropna(subset=['Z_score'])['quarter'].max()
latest=(zscore_df[(zscore_df['quarter']==latest_q)&zscore_df['z_reliable']]
        .sort_values('Z_score',ascending=False).reset_index(drop=True))
latest.to_csv(os.path.join(OUTPUT_DIR,'zscore_latest_ranking.csv'),index=False)
print(f'  {latest_q}: Safe={(latest["zone"]=="safe").sum()} · '
      f'Grey={(latest["zone"]=="grey").sum()} · '
      f'Distress={(latest["zone"]=="distress").sum()}')
print('✓ Z-Scores saved')

# ── BLOCK 2: Heatmap ─────────────────────────────────────────────────────────
print('\n→ Building heatmap...')

pivot_z=zscore_df.pivot_table(index='ticker',columns='quarter',values='Z_score',aggfunc='mean')
q_cols=sorted([c for c in pivot_z.columns if '2015Q1'<=c<='2023Q4'])
pivot_z=pivot_z[q_cols]

mean_z=pivot_z.mean(axis=1)
sdf=pd.DataFrame({'sector':pd.Series(sector_map),'mean_z':mean_z})
sdf=sdf[sdf.index.isin(pivot_z.index)].sort_values(['sector','mean_z'],ascending=[True,False])
pivot_z=pivot_z.reindex(sdf.index)

flag_lkp=zscore_df.set_index(['ticker','quarter'])[
    ['n_imp','was_ffilled','X1_imp','X2_imp','X3_imp','X4_imp','X5_imp']
].to_dict('index')

hover=[]
for t in pivot_z.index:
    rh=[]
    for q in q_cols:
        val=pivot_z.loc[t,q]
        if pd.notna(val):
            zl='SAFE' if val>Z_SAFE else 'DISTRESS' if val<Z_DISTRESS else 'GREY ZONE'
            flg=flag_lkp.get((t,q),{})
            ni=int(flg.get('n_imp',0)); ff=bool(flg.get('was_ffilled',False))
            notes=[]
            if ff: notes.append('ffilled')
            if ni>0:
                imp=[c.replace('_imp','') for c in ['X1_imp','X2_imp','X3_imp','X4_imp','X5_imp']
                     if flg.get(c,False)]
                notes.append(f'{ni} comp set to 0: {",".join(imp)}')
            ns=('<br><i>⚠ '+'  ·  '.join(notes)+'</i>') if notes else ''
            rh.append(f'<b>{t}</b>  ·  {q}<br>Z=<b>{val:.2f}</b>  ·  <b>{zl}</b><br>'
                      f'{sector_map.get(t,"")}  ·  {tier_map.get(t,"")}{ns}')
        else:
            rh.append(f'<b>{t}</b>  ·  {q}<br><i>No EDGAR data</i>')
    hover.append(rh)

shapes,cur_sec=[],None
for i,t in enumerate(pivot_z.index):
    s=sector_map.get(t,'')
    if s!=cur_sec and i>0:
        shapes.append(dict(type='line',x0=-0.5,x1=len(q_cols)-0.5,
                           y0=i-0.5,y1=i-0.5,line=dict(color='white',width=3)))
    cur_sec=s

anns,cur_sec,ss=[],None,0
tlist=list(pivot_z.index)
for i,t in enumerate(tlist):
    s=sector_map.get(t,'')
    if s!=cur_sec:
        if cur_sec:
            anns.append(dict(x=len(q_cols)+0.8,y=(ss+i-1)/2,text=f'<b>{cur_sec}</b>',
                             showarrow=False,xref='x',yref='y',
                             font=dict(size=9,color='#333'),xanchor='left'))
        cur_sec,ss=s,i
if cur_sec:
    anns.append(dict(x=len(q_cols)+0.8,y=(ss+len(tlist)-1)/2,text=f'<b>{cur_sec}</b>',
                     showarrow=False,xref='x',yref='y',
                     font=dict(size=9,color='#333'),xanchor='left'))

z_min,z_max=(-1,6); sp=z_max-z_min
pd_=(Z_DISTRESS-z_min)/sp; ps_=(Z_SAFE-z_min)/sp

colorscale=[
    [0.00,'#7B0000'],[0.10,'#C0392B'],[0.25,'#E8856B'],
    [round(pd_-0.03,3),'#F5C6B8'],
    [round(pd_,3),'#E8B820'],
    [round((pd_+ps_)/2,3),'#D4A010'],
    [round(ps_,3),'#B8860B'],
    [round(ps_+0.04,3),'#D4EDCA'],
    [round(ps_+0.15,3),'#88C870'],
    [round(ps_+0.28,3),'#4A9E36'],
    [1.00,'#2D6A27'],
]

fig_hm=go.Figure(go.Heatmap(
    z=pivot_z.values,x=q_cols,y=pivot_z.index.tolist(),
    text=hover,hovertemplate='%{text}<extra></extra>',
    colorscale=colorscale,zmin=z_min,zmax=z_max,
    colorbar=dict(
        title=dict(text='Z-Score',side='right',font=dict(size=10,color='#333')),
        thickness=14,len=0.75,x=1.13,
        tickvals=[-1,0,Z_DISTRESS,Z_SAFE,4,6],
        ticktext=['−1','0',f'<b>{Z_DISTRESS}</b><br><i>distress</i>',
                  f'<b>{Z_SAFE}</b><br><i>safe</i>','4','6'],
        tickfont=dict(size=8),
    ),
))
fig_hm.update_layout(
    title=dict(
        text=('Altman Z-Score  ·  All Companies × Quarters  |  Phase 2<br>'
              '<sup>'
              '<span style="color:#C0392B">■</span> Distress (Z&lt;1.81)  ·  '
              '<span style="color:#B8860B">■</span> Grey zone (1.81–2.99)  ·  '
              '<span style="color:#2D6A27">■</span> Safe (Z&gt;2.99)  ·  '
              '⚠ hover for data quality flags'
              '</sup>'),
        font=dict(size=13,color=C_ACC),x=0.5,xanchor='center',
    ),
    xaxis=dict(tickangle=-50,tickfont=dict(size=7.5),showgrid=False,domain=[0.0,0.87]),
    yaxis=dict(tickfont=dict(size=8.5),autorange='reversed',showgrid=False),
    height=max(750,len(pivot_z.index)*15),
    margin=dict(l=85,r=170,t=70,b=70),
    shapes=shapes,annotations=anns,
    plot_bgcolor='white',paper_bgcolor='white',
)
fig_hm.show()
fig_hm.write_html(os.path.join(OUTPUT_DIR,'zscore_heatmap.html'))
print('  ✓ Heatmap saved')

# ── BLOCK 3: SVB Timeline ─────────────────────────────────────────────────────
print('\n→ SVB timeline...')

def q2ts(q):
    try: yr,qn=int(q[:4]),int(q[5]); return pd.Timestamp(year=yr,month=(qn-1)*3+1,day=1)
    except: return pd.NaT

zscore_df['ts']=zscore_df['quarter'].apply(q2ts)

bank_peers=['SIVB','FRC','WAL','PACW','JPM','BAC']
peer_cfg={
    'JPM':('#B5D4F4',1.2,'--'),'BAC':('#B5D4F4',1.2,':'),
    'WAL':('#FAC775',1.5,'-.'),'PACW':('#EF9F27',1.5,'-.'),
    'FRC':('#E24B4A',1.8,'--'),
}
fig,ax=plt.subplots(figsize=(12,5.5))
ax.axhspan(-0.5,Z_DISTRESS,color='#FDF0EF',alpha=0.5,zorder=0)
ax.axhspan(Z_DISTRESS,Z_SAFE,color='#FDF6E3',alpha=0.5,zorder=0)
ax.axhspan(Z_SAFE,3.0,color='#F0F7EC',alpha=0.5,zorder=0)
ax.axhline(Z_DISTRESS,color=C_DIST,lw=1.0,linestyle='--',alpha=0.7)
ax.axhline(Z_SAFE,color=C_SAFE,lw=1.0,linestyle='--',alpha=0.7)
ax.text(pd.Timestamp('2015-04-01'),Z_DISTRESS-0.12,'DISTRESS ZONE',
        fontsize=7.5,color=C_DIST,alpha=0.8,fontstyle='italic')
ax.text(pd.Timestamp('2015-04-01'),Z_DISTRESS+0.06,'GREY ZONE',
        fontsize=7.5,color=C_GREY,alpha=0.8,fontstyle='italic')

for t,(color,lw,ls) in peer_cfg.items():
    d=zscore_df[zscore_df['ticker']==t].dropna(subset=['Z_score']).sort_values('ts')
    if len(d)>0: ax.plot(d['ts'],d['Z_score'],color=color,lw=lw,linestyle=ls,
                         alpha=0.8,label=t,zorder=2)

svb_d=zscore_df[zscore_df['ticker']=='SIVB'].dropna(subset=['Z_score']).sort_values('ts')
if len(svb_d)>0:
    ax.plot(svb_d['ts'],svb_d['Z_score'],color='#A32D2D',lw=2.8,
            marker='o',markersize=5,label='SIVB (SVB)',zorder=5)
    q3=svb_d[svb_d['quarter']=='2022Q3']
    if len(q3)>0:
        ax.annotate(f'Q3 2022  Z={q3["Z_score"].values[0]:.2f}',
                    xy=(q3['ts'].values[0],q3['Z_score'].values[0]),
                    xytext=(20,-28),textcoords='offset points',fontsize=8,
                    color='#A32D2D',fontweight='bold',
                    arrowprops=dict(arrowstyle='->',color='#A32D2D',lw=1.2))

for ts,lbl,color,pos in [
    (pd.Timestamp('2020-03-01'),'COVID\nshock','#2471A3','top'),
    (pd.Timestamp('2022-03-01'),'Fed hike\ncycle','#7D6608','top'),
    (pd.Timestamp('2023-03-10'),'SVB\ncollapse','#A32D2D','top'),
]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.8)
    ax.text(ts+pd.Timedelta(days=20),2.25,lbl,fontsize=7.5,color=color,alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))

ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax.set_ylim(-0.4,2.6)
ax.set_xlabel('Date',fontsize=9); ax.set_ylabel('Altman Z-Score',fontsize=9)
ax.set_title('SVB Z-Score Deterioration vs Bank Peers  |  Phase 2',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'All bank Z-Scores structurally low (no current/non-current split) — '
        'chart shows relative deterioration within distress zone',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor='#ddd',ncol=3)
ax.grid(True,alpha=0.2,linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'zscore_svb_timeline.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ SVB timeline saved')

# ── BLOCK 4: Zone distribution ────────────────────────────────────────────────
print('\n→ Zone distribution...')
zone_time=(zscore_df[zscore_df['z_reliable']==True]
           .groupby(['quarter','zone']).size().unstack(fill_value=0).reset_index())
for z in ['safe','grey','distress']:
    if z not in zone_time.columns: zone_time[z]=0
zone_time['ts']=zone_time['quarter'].apply(q2ts)
zone_time=zone_time.dropna(subset=['ts']).sort_values('ts')
zone_time=zone_time[zone_time['ts']>=pd.Timestamp('2015-01-01')]

fig,ax=plt.subplots(figsize=(13,5))
ax.stackplot(zone_time['ts'].values,
             zone_time['distress'],zone_time['grey'],zone_time['safe'],
             labels=['Distress (Z<1.81)','Grey Zone (1.81–2.99)','Safe (Z>2.99)'],
             colors=[C_DIST,C_GREY,C_SAFE],alpha=0.75)
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID\nshock','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike\ncycle','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'Bank\ncrisis','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.5,linestyle='--',alpha=0.7,zorder=5)
    ax.text(ts+pd.Timedelta(days=20),ax.get_ylim()[1]*0.95,lbl,
            fontsize=8,color=color,va='top',
            bbox=dict(boxstyle='round,pad=0.25',fc='white',ec=color,alpha=0.8))
ax.set_ylabel('Number of companies',fontsize=9)
ax.set_xlabel('Date',fontsize=9)
ax.set_title('Universe Zone Distribution Over Time  |  Phase 2',
             fontsize=11,fontweight='500',color=C_ACC,pad=10)
ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'zscore_zone_distribution.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Zone distribution saved')

# ── BLOCK 5: Ranking lollipop ─────────────────────────────────────────────────
print('\n→ Ranking chart...')
ranked=latest[latest['Z_score'].notna()].copy()
plot_r=pd.concat([ranked.tail(15).sort_values('Z_score'),
                  ranked.head(15)]).drop_duplicates().sort_values('Z_score')
zc_map={'safe':C_SAFE,'grey':C_GREY,'distress':C_DIST,'unknown':'#95A5A6'}
dot_c=[zc_map.get(z,'#95A5A6') for z in plot_r['zone']]
y_pos=np.arange(len(plot_r))

fig,ax=plt.subplots(figsize=(11,max(8,len(plot_r)*0.38)))
fig.subplots_adjust(top=0.88)
ax.axvspan(-4,Z_DISTRESS,color='#FDF0EF',alpha=0.4,zorder=0)
ax.axvspan(Z_DISTRESS,Z_SAFE,color='#FDF6E3',alpha=0.4,zorder=0)
ax.axvspan(Z_SAFE,8,color='#F0F7EC',alpha=0.4,zorder=0)
ax.axvline(Z_DISTRESS,color=C_DIST,lw=1.2,linestyle='--',alpha=0.8,zorder=2)
ax.axvline(Z_SAFE,color=C_SAFE,lw=1.2,linestyle='--',alpha=0.8,zorder=2)
for i,(_,row) in enumerate(plot_r.iterrows()):
    c=zc_map.get(row['zone'],'#95A5A6')
    ax.plot([0,row['Z_score']],[i,i],color=c,lw=1.5,alpha=0.5,zorder=1)
ax.scatter(plot_r['Z_score'].values,y_pos,c=dot_c,s=80,zorder=4,
           edgecolors='white',linewidths=0.8)
for i,(_,row) in enumerate(plot_r.iterrows()):
    off=0.12 if row['Z_score']>=0 else -0.12
    ha='left' if row['Z_score']>=0 else 'right'
    ax.text(row['Z_score']+off,i,f'{row["Z_score"]:.2f}',
            va='center',ha=ha,fontsize=7.5,color=zc_map.get(row['zone'],'#555'))
smap=universe_df.set_index('ticker')['sector'].to_dict()
for i,(_,row) in enumerate(plot_r.iterrows()):
    ax.text(7.8,i,smap.get(row['ticker'],''),va='center',ha='right',fontsize=6.5,color='#888')
ax.set_yticks(y_pos); ax.set_yticklabels(plot_r['ticker'],fontsize=9)
ax.set_xlim(-4,8); ax.set_ylim(-0.8,len(plot_r)-0.2)
ax.set_xlabel('Altman Z-Score',fontsize=9)
ax.set_title(f'Z-Score Ranking — {latest_q}  |  Phase 2\n'
             'Top & bottom 15  ·  ★ = bank (X1/X5 structurally unreliable)',
             fontsize=11,fontweight='500',color=C_ACC,pad=14)
ax.grid(axis='x',alpha=0.2,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'zscore_ranking.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Ranking saved')

# ── BLOCK 6: Z-Score vs Forward Return ───────────────────────────────────────
print('\n→ Z-Score vs forward return...')
ret_annual={}
for ticker in MERTON_TICKERS:
    if ticker not in equity_df.columns: continue
    prices=equity_df[ticker].dropna()
    if len(prices)<252: continue
    qp=prices.resample('Q').last()
    for i in range(len(qp)-4):
        ql=qp.index[i].to_period('Q').strftime('%YQ%q')
        p0,p1=qp.iloc[i],qp.iloc[i+4]
        if pd.notna(p0) and pd.notna(p1) and p0>0:
            ret_annual[(ticker,ql)]=(p1/p0-1)*100

sc_rows=[]
for _,row in zscore_df.iterrows():
    key=(row['ticker'],row['quarter'])
    if key in ret_annual and pd.notna(row['Z_score']):
        sc_rows.append({'Z_score':row['Z_score'],'fwd':ret_annual[key],
                        'zone':row['zone'],'sector':row['sector']})
sc=pd.DataFrame(sc_rows)
sc=sc[sc['Z_score'].between(-3,7)&sc['fwd'].between(-100,200)]

fig,axes=plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={'width_ratios':[1.4,1]})
ax=axes[0]
for zone,label,color,marker in [('distress','Distress (<1.81)',C_DIST,'o'),
                                   ('grey','Grey Zone',C_GREY,'s'),
                                   ('safe','Safe (>2.99)',C_SAFE,'o')]:
    sub=sc[sc['zone']==zone]
    ax.scatter(sub['Z_score'],sub['fwd'],c=color,s=35,alpha=0.4,
               marker=marker,label=f'{label} (n={len(sub)})',edgecolors='none')
sc_cl=sc.dropna()
if len(sc_cl)>30:
    m,b,r,p,_=stats.linregress(sc_cl['Z_score'],sc_cl['fwd'])
    xl=np.linspace(-3,7,100)
    ax.plot(xl,m*xl+b,'k--',lw=1.5,alpha=0.7,label=f'OLS  R={r:.2f}, p={p:.2f}')
ax.axvline(Z_DISTRESS,color=C_DIST,lw=1,linestyle=':',alpha=0.7)
ax.axvline(Z_SAFE,color=C_SAFE,lw=1,linestyle=':',alpha=0.7)
ax.axhline(0,color='#888',lw=1,alpha=0.5)
ax.axvspan(-3.5,Z_DISTRESS,color='#FDF0EF',alpha=0.3,zorder=0)
ax.axvspan(Z_DISTRESS,Z_SAFE,color='#FDF6E3',alpha=0.3,zorder=0)
ax.axvspan(Z_SAFE,7.5,color='#F0F7EC',alpha=0.3,zorder=0)
ax.set_xlim(-3.5,7.5); ax.set_ylim(-110,210)
ax.set_xlabel('Altman Z-Score',fontsize=9)
ax.set_ylabel('1-Year Forward Return (%)',fontsize=9)
ax.set_title('Z-Score vs 1-Year Forward Return',fontsize=10,fontweight='500',color=C_ACC)
ax.legend(fontsize=8,framealpha=0.9,edgecolor='#ddd')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2=axes[1]
zone_order=['distress','grey','safe']
data_bz=[sc[sc['zone']==z]['fwd'].dropna().values for z in zone_order]
bps=ax2.boxplot(data_bz,patch_artist=True,notch=False,
                medianprops=dict(color='white',linewidth=2.5),
                whiskerprops=dict(linewidth=1.2,color='#888'),
                capprops=dict(linewidth=1.2,color='#888'),
                flierprops=dict(marker='o',markersize=3,alpha=0.35,markeredgewidth=0))
for patch,color in zip(bps['boxes'],[C_DIST,C_GREY,C_SAFE]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax2.axhline(0,color='#888',lw=1,alpha=0.5,linestyle='--')
ax2.set_xticks([1,2,3]); ax2.set_xticklabels(['Distress','Grey Zone','Safe'],fontsize=9)
ax2.set_ylabel('1-Year Forward Return (%)',fontsize=9)
ax2.set_title('Return Distribution by Zone',fontsize=10,fontweight='500',color=C_ACC)
for i,vals in enumerate(data_bz):
    if len(vals)>0:
        med=np.median(vals)
        ax2.text(i+1,med+3,f'{med:.1f}%',ha='center',va='bottom',fontsize=8,
                 fontweight='bold',color=[C_DIST,C_GREY,C_SAFE][i])
ax2.text(0.5,-0.18,f'R={r:.2f} (p={p:.2f}) — flat relationship is a documented limitation\n'
         'of Altman Z-Score outside manufacturing sector (Altman 1968)',
         transform=ax2.transAxes,ha='center',fontsize=7,color='#666',style='italic')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.suptitle('Phase 2 · Z-Score Predictive Validity',fontsize=11,fontweight='500',color=C_ACC,y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'zscore_vs_return.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Return scatter saved')

# ── BLOCK 7: Component breakdown ─────────────────────────────────────────────
print('\n→ Component breakdown...')
bank_ex=['JPM','BAC','GS','COF','WAL','WFC','SIVB','SBNY','FRC','PACW']
show_t=(latest[latest['zone'].isin(['distress','grey'])&
               ~latest['ticker'].isin(bank_ex)&
               latest['Z_score'].notna()]
        .sort_values('Z_score')['ticker'].tolist())[:14]

if show_t:
    W=[W1,W2,W3,W4,W5]; C=['X1','X2','X3','X4','X5']
    CL=['1.2×X1\n(Work.Cap)','1.4×X2\n(Ret.Earn)','3.3×X3\n(EBIT)',
        '0.6×X4\n(Equity)','1.0×X5\n(Revenue)']
    CC=['#3498DB','#1ABC9C','#E74C3C','#9B59B6','#E67E22']
    n=len(show_t); width=0.15

    fig,(ax_m,ax_z)=plt.subplots(2,1,figsize=(max(13,n*1.0),9),
                                   gridspec_kw={'height_ratios':[3.5,1.2],'hspace':0.05},
                                   sharex=True)
    x=np.arange(n)
    for k,(col,w,lbl,color) in enumerate(zip(C,W,CL,CC)):
        vals=[]
        for t in show_t:
            row=latest[latest['ticker']==t]
            v=float(row[col].values[0])*w if len(row)>0 and pd.notna(row[col].values[0]) else 0.0
            if col=='X4': v=np.clip(v,-2.0,2.5)
            vals.append(v)
        ax_m.bar(x+k*width,vals,width,label=lbl,color=color,alpha=0.82,
                 edgecolor='white',linewidth=0.4)

    ax_m.axhline(0,color='#666',lw=0.8,alpha=0.5)
    ax_m.set_ylabel('Weighted component contribution',fontsize=9)
    ax_m.set_ylim(-1.2,3.0)
    ax_m.set_title('Z-Score Component Breakdown — Distress & Grey Zone  |  Phase 2',
                   fontsize=11,fontweight='500',color=C_ACC,pad=12)
    ax_m.text(0.5,1.012,'Banks excluded  ·  X4 capped at 2.5 for readability',
              transform=ax_m.transAxes,ha='center',fontsize=7.8,color='#666',style='italic')
    ax_m.legend(loc='upper right',fontsize=8,ncol=3,framealpha=0.92,edgecolor='#ddd')
    ax_m.grid(axis='y',alpha=0.2,linestyle='--')
    ax_m.spines['top'].set_visible(False); ax_m.spines['right'].set_visible(False)

    z_vals=[float(latest[latest['ticker']==t]['Z_score'].values[0])
            if len(latest[latest['ticker']==t])>0 else 0 for t in show_t]
    barz=[C_DIST if z<Z_DISTRESS else C_GREY for z in z_vals]
    bars=ax_z.bar(x+2*width,z_vals,width*5,color=barz,alpha=0.75,edgecolor='white',linewidth=0.5)
    ax_z.axhline(Z_DISTRESS,color=C_DIST,lw=1.3,linestyle='--',alpha=0.85)
    ax_z.axhline(Z_SAFE,color=C_GREY,lw=1.0,linestyle=':',alpha=0.7)
    ax_z.text(n-0.3,Z_DISTRESS+0.07,f'Distress {Z_DISTRESS}',
              fontsize=7.5,color=C_DIST,ha='right')
    for bar,zv in zip(bars,z_vals):
        ax_z.text(bar.get_x()+bar.get_width()/2,max(zv,0)+0.05,f'{zv:.2f}',
                  ha='center',va='bottom',fontsize=7.5,fontweight='bold',
                  color=C_DIST if zv<Z_DISTRESS else C_GREY)
    ax_z.set_xticks(x+2*width); ax_z.set_xticklabels(show_t,fontsize=9)
    ax_z.set_ylabel('Z-Score\ntotal',fontsize=8.5)
    ax_z.set_ylim(-0.15,3.5); ax_z.set_xlim(-0.4,n-0.2)
    ax_z.grid(axis='y',alpha=0.2,linestyle='--')
    ax_z.spines['top'].set_visible(False); ax_z.spines['right'].set_visible(False)
    dp=plt.Rectangle((0,0),1,1,fc=C_DIST,alpha=0.75)
    gp=plt.Rectangle((0,0),1,1,fc=C_GREY,alpha=0.75)
    ax_z.legend([dp,gp],['Distress','Grey zone'],fontsize=8,loc='upper left',
                framealpha=0.9,edgecolor='#ddd')
    plt.savefig(os.path.join(OUTPUT_DIR,'zscore_component_breakdown.png'),
                dpi=160,bbox_inches='tight',facecolor='white')
    plt.show(); print('  ✓ Component breakdown saved')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'PHASE 2 COMPLETE')
print(f'{"="*55}')
for f in ['zscore_heatmap.html','zscore_svb_timeline.png',
          'zscore_zone_distribution.png','zscore_ranking.png',
          'zscore_vs_return.png','zscore_component_breakdown.png']:
    path=os.path.join(OUTPUT_DIR,f)
    if os.path.exists(path):
        print(f'  ✓ {f:<45} {os.path.getsize(path)/1024:6.1f} KB')
print(f'\n  SR 11-7: Z-Score = CHALLENGER · Merton = CHAMPION')
print(f'  → Ready for Phase 3: Merton Structural Credit Risk Model')
print(f'{"="*55}')


# ============================================================
# PHASE 3 — COMPLETE (One Cell)
# Merton Structural Credit Risk Model + All Visualizations
# ============================================================

import os, warnings
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy               import stats, optimize
from scipy.stats         import norm
from google.colab        import drive

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':0.2,'grid.linestyle':'--',
})

drive.mount('/content/drive', force_remount=False)
BASE_DIR   = '/content/drive/MyDrive/P4_CreditRisk'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CACHE_DIR  = os.path.join(BASE_DIR, 'cache')

# ── Load ──────────────────────────────────────────────────────────────────────
universe_df   = pd.read_csv(os.path.join(OUTPUT_DIR, 'universe.csv'))
zscore_df     = pd.read_csv(os.path.join(OUTPUT_DIR, 'zscore_quarterly.csv'))
rf_df         = pd.read_csv(os.path.join(OUTPUT_DIR, 'fred_rates.csv'),
                             parse_dates=['date'], index_col='date')
params_df     = pd.read_csv(os.path.join(OUTPUT_DIR, 'parameter_registry.csv'))

PARAMS = dict(zip(params_df['parameter'], params_df['value']))
T      = float(PARAMS['merton_T'])

MERTON_TICKERS = universe_df[universe_df['in_merton']==True]['ticker'].tolist()
BANK_TICKERS   = universe_df[universe_df['sector']=='Banks']['ticker'].tolist()
REIT_TICKERS   = universe_df[universe_df['sector']=='Real Estate']['ticker'].tolist()
sector_map     = universe_df.set_index('ticker')['sector'].to_dict()
tier_map       = universe_df.set_index('ticker')['tier'].to_dict()

DEFAULT_DATES = {
    'CHK':'2020-06-28','HTZ':'2020-05-22','BBBY':'2023-04-23',
    'SIVB':'2023-03-10','SBNY':'2023-03-12','FRC':'2023-05-01',
    'CBL':'2020-11-01','WPG':'2021-06-13',
}

C_SAFE='#2D6A27'; C_GREY='#B07D10'; C_DIST='#C0392B'; C_ACC='#2C3E50'

def q2ts(q):
    try:
        yr,qn=int(q[:4]),int(q[5])
        return pd.Timestamp(year=yr,month=(qn-1)*3+1,day=1)
    except: return pd.NaT

def is_post_default(ticker, quarter):
    if ticker not in DEFAULT_DATES: return False
    return q2ts(quarter) > pd.Timestamp(DEFAULT_DATES[ticker])

print('✓ Data loaded')

# ── Load pre-built complete input grid ────────────────────────────────────────
print('\n→ Loading Merton inputs...')
merton_input = pd.read_csv(os.path.join(CACHE_DIR, 'merton_input_complete.csv'))
n_valid = (merton_input['market_cap_M'].notna() &
           merton_input['sigma_E'].notna() &
           merton_input['D'].notna() &
           merton_input['rf_rate'].notna() &
           (merton_input['market_cap_M']>0) &
           (merton_input['D']>0)).sum()
print(f'  Input rows : {len(merton_input):,}')
print(f'  Valid rows : {n_valid:,}')

# ── Merton Solver ─────────────────────────────────────────────────────────────
def d1_d2(A, D, r, sA, T):
    if A<=0 or sA<=0 or D<=0: return np.nan, np.nan
    try:
        d1 = (np.log(A/D)+(r+0.5*sA**2)*T)/(sA*np.sqrt(T))
        return d1, d1-sA*np.sqrt(T)
    except: return np.nan, np.nan

def merton_eqs(x, E, sE, D, r, T):
    A,sA = x
    if A<=0 or sA<=0: return [1e6,1e6]
    d1,d2 = d1_d2(A,D,r,sA,T)
    if np.isnan(d1): return [1e6,1e6]
    return [A*norm.cdf(d1)-D*np.exp(-r*T)*norm.cdf(d2)-E,
            norm.cdf(d1)*sA*A-sE*E]

def solve_merton(E, sE, D, r, T=1.0, prior_sA=None):
    if any(pd.isna(v) for v in [E,sE,D,r]): return None
    if E<=0 or D<=0: return None
    r=max(r,0.0001); sE=max(sE,0.01)
    guesses=[[E+D,sE*E/(E+D)],[E+D*0.9,sE*0.8],[E+D*1.1,sE*1.2],
             [E*1.5,sE*0.6],[max(E+D,1),min(sE*2,1.5)]]
    if prior_sA and prior_sA>0:
        guesses.insert(0,[E+D,prior_sA])
    best,best_err=None,np.inf
    for x0 in guesses:
        try:
            sol,_,ier,_=optimize.fsolve(merton_eqs,x0,
                args=(E,sE,D,r,T),full_output=True,xtol=1e-10,maxfev=2000)
            A_s,sA_s=sol
            if A_s<=0 or sA_s<=0: continue
            d1,d2=d1_d2(A_s,D,r,sA_s,T)
            if np.isnan(d1): continue
            E_imp=A_s*norm.cdf(d1)-D*np.exp(-r*T)*norm.cdf(d2)
            err=abs(E_imp-E)/max(E,1e-6)
            if err<best_err: best_err=err; best=(A_s,sA_s,d1,d2,err)
        except: continue
    if best is None: return None
    A_s,sA_s,d1,d2,err=best
    PD=norm.cdf(-d2)
    try:
        cs_arg=norm.cdf(d2)+(A_s/(D*np.exp(-r*T)))*norm.cdf(-d1)
        cs=max(-(1/T)*np.log(max(cs_arg,1e-10)),0.0)
    except: cs=np.nan
    LGD=float(PARAMS['lgd_senior_unsecured'])
    return {'A':A_s,'sigma_A':sA_s,'d1':d1,'d2':d2,'PD':PD,'D2D':d2,
            'credit_spread':cs,'EL':PD*LGD*D,'leverage':D/A_s,
            'conv_error':err,'converged':err<0.01}

# ── Run solver ────────────────────────────────────────────────────────────────
print('\n→ Running Merton solver...')
results,prior_sA=[],{}
n_solved=n_conv=n_fail=0

for _,row in merton_input.iterrows():
    ticker=row['ticker']; quarter=row['quarter']
    E=row['market_cap_M']; sE=row['sigma_E']
    D=row['D']; r=row['rf_rate']
    sol=solve_merton(E,sE,D,r,T=T,prior_sA=prior_sA.get(ticker))
    if sol is None: n_fail+=1; continue
    n_solved+=1
    if sol['converged']: n_conv+=1
    prior_sA[ticker]=sol['sigma_A']
    results.append({
        'ticker':ticker,'quarter':quarter,
        'sector':sector_map.get(ticker,''),'tier':tier_map.get(ticker,''),
        'E_market_cap':E,'D_debt':D,'rf_rate':r,'sigma_E':sE,
        'A_assets':sol['A'],'sigma_A':sol['sigma_A'],
        'd1':sol['d1'],'d2':sol['d2'],'PD':sol['PD'],'D2D':sol['D2D'],
        'credit_spread':sol['credit_spread'],'EL':sol['EL'],
        'leverage':sol['leverage'],'conv_error':sol['conv_error'],
        'converged':sol['converged'],
        'is_bank':ticker in BANK_TICKERS,'is_reit':ticker in REIT_TICKERS,
        'D_method':'total_liab' if ticker in BANK_TICKERS+REIT_TICKERS else 'total_debt',
    })

if not results: raise RuntimeError('No solutions')
merton_df=pd.DataFrame(results)
merton_df['ts']=merton_df['quarter'].apply(q2ts)
merton_df['post_default']=merton_df.apply(
    lambda r: is_post_default(r['ticker'],r['quarter']),axis=1)

print(f'  Solved    : {n_solved:,} / {len(merton_input):,}')
print(f'  Converged : {n_conv:,} ({n_conv/max(n_solved,1)*100:.1f}%)')
print(f'  Companies : {merton_df["ticker"].nunique()}')

# ── Validation ────────────────────────────────────────────────────────────────
print('\n── VALIDATION ──')
svb=merton_df[(merton_df['ticker']=='SIVB')&(merton_df['quarter']=='2022Q3')]
if len(svb)>0:
    r=svb.iloc[0]
    print(f'  SVB Q3 2022: PD={r["PD"]*100:.2f}%  D2D={r["D2D"]:.3f}  '
          f'Converged={r["converged"]}')
nvda=merton_df[(merton_df['ticker']=='NVDA')&(merton_df['quarter']=='2023Q1')]
if len(nvda)>0:
    r=nvda.iloc[0]
    print(f'  NVDA Q1 2023: PD={r["PD"]*100:.4f}%  D2D={r["D2D"]:.3f}')
print(f'\n  SVB 2022 deterioration:')
for _,r in (merton_df[(merton_df['ticker']=='SIVB')&
    (merton_df['quarter'].isin(['2022Q1','2022Q2','2022Q3']))]
    .sort_values('quarter').iterrows()):
    print(f'    {r["quarter"]}: PD={r["PD"]*100:.2f}%  D2D={r["D2D"]:.3f}')

# ── Save ──────────────────────────────────────────────────────────────────────
merton_df.drop(columns=['ts']).to_csv(
    os.path.join(OUTPUT_DIR,'merton_results.csv'),index=False)
(merton_df.groupby(['ticker','sector','tier'])
          .agg(n=('PD','count'),mean_PD=('PD','mean'),max_PD=('PD','max'),
               mean_D2D=('D2D','mean'),min_D2D=('D2D','min'),
               mean_spread=('credit_spread','mean'),pct_conv=('converged','mean'))
          .round(6).reset_index()
          .to_csv(os.path.join(OUTPUT_DIR,'merton_summary.csv'),index=False))
print(f'\n✓ Saved: {len(merton_df):,} rows  {merton_df["ticker"].nunique()} companies')

latest_q_m=merton_df[~merton_df['post_default']]['quarter'].max()
print(f'\n  Top 10 PD — {latest_q_m} (post-default excluded):')
top_pd=(merton_df[(merton_df['quarter']==latest_q_m)&~merton_df['post_default']]
        .nlargest(10,'PD')[['ticker','sector','PD','D2D','credit_spread']])
for _,r in top_pd.iterrows():
    cs=f'{r["credit_spread"]*100:.2f}%' if pd.notna(r['credit_spread']) else 'N/A'
    print(f'    {r["ticker"]:<6} {r["sector"]:<14} '
          f'PD={r["PD"]*100:.2f}%  D2D={r["D2D"]:.3f}  spread={cs}')

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── VIZ 1: SVB D2D + PD timeline ─────────────────────────────────────────────
print('\n→ [1/6] SVB D2D timeline...')
peer_style={
    'JPM':('#B5D4F4',1.2,'--','JPMorgan (IG)'),
    'BAC':('#85B7EB',1.2,':','Bank of America (IG)'),
    'GS' :('#5B9BD5',1.2,'--','Goldman Sachs (IG)'),
    'WAL':('#FAC775',1.6,'-.','Western Alliance'),
    'FRC':('#E24B4A',1.8,'--','First Republic'),
}
fig,(ax,ax2)=plt.subplots(2,1,figsize=(13,10),
    gridspec_kw={'height_ratios':[1.6,1],'hspace':0.35})
ax.axhspan(-10,0,color='#FDF0EF',alpha=0.3,zorder=0)
ax.axhline(0,color=C_DIST,lw=1.5,linestyle='--',alpha=0.8,
           label='Default threshold (D2D=0)')
for t,(color,lw,ls,label) in peer_style.items():
    d=merton_df[merton_df['ticker']==t].dropna(subset=['D2D']).sort_values('ts')
    if len(d)>0:
        ax.plot(d['ts'],d['D2D'],color=color,lw=lw,linestyle=ls,
                alpha=0.8,label=label,zorder=3)
svb=merton_df[merton_df['ticker']=='SIVB'].sort_values('ts')
if len(svb)>0:
    ax.plot(svb['ts'],svb['D2D'],color='#A32D2D',lw=3.0,
            marker='o',markersize=4,label='SVB Financial (SIVB)',zorder=6)
    q3=svb[svb['quarter']=='2022Q3']
    if len(q3)>0:
        ax.annotate(
            f'Q3 2022\nD2D={q3["D2D"].values[0]:.2f}\nPD={q3["PD"].values[0]*100:.1f}%',
            xy=(q3['ts'].values[0],q3['D2D'].values[0]),
            xytext=(30,20),textcoords='offset points',fontsize=8,
            color='#A32D2D',fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#A32D2D',lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#A32D2D',alpha=0.9))
bank_d2d=merton_df[merton_df['ticker'].isin(list(peer_style.keys())+['SIVB'])]['D2D'].dropna()
ylim_top=min(bank_d2d.quantile(0.95),12) if len(bank_d2d)>0 else 8
ax.set_ylim(-2,ylim_top)
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID shock','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike cycle','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'SVB collapse','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.8)
    ax.text(ts+pd.Timedelta(days=25),ylim_top*0.88,lbl,fontsize=7.5,color=color,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))
ax.set_ylabel('Distance to Default (d2)',fontsize=9)
ax.set_title('Merton Model  ·  Distance-to-Default: SVB vs Bank Peers  |  Phase 3',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'D2D = σ above default point  ·  D2D→0 signals imminent risk  ·  '
        'Banks use total_liab as D (Basel III)',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor='#ddd',ncol=3)
ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax.grid(True,alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
if len(svb)>0:
    ax2.fill_between(svb['ts'],svb['PD']*100,alpha=0.3,color='#A32D2D')
    ax2.plot(svb['ts'],svb['PD']*100,color='#A32D2D',lw=2.0,marker='o',markersize=4)
    ax2.set_ylabel('Merton PD (%)',fontsize=9)
    ax2.set_title('SVB Probability of Default  N(−d2)',fontsize=10,fontweight='500',color=C_ACC)
    ax2.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
    for ts,lbl,color in [(pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                          (pd.Timestamp('2023-03-10'),'Collapse','#A32D2D')]:
        ax2.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.8)
        ax2.text(ts+pd.Timedelta(days=25),ax2.get_ylim()[1]*0.75,lbl,fontsize=7.5,color=color)
    ax2.grid(True,alpha=0.2); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR,'merton_svb_d2d.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ SVB D2D saved')

# ── VIZ 2: PD Heatmap ────────────────────────────────────────────────────────
print('\n→ [2/6] PD heatmap...')
pivot_pd=merton_df.pivot_table(index='ticker',columns='quarter',values='PD',aggfunc='mean')
q_cols=sorted([c for c in pivot_pd.columns if '2015Q1'<=c<='2023Q4'])
pivot_pd=pivot_pd[q_cols]
mean_pd=pivot_pd.mean(axis=1)
sdf=pd.DataFrame({'sector':pd.Series(sector_map),'mean_pd':mean_pd})
sdf=sdf[sdf.index.isin(pivot_pd.index)].sort_values(['sector','mean_pd'],ascending=[True,False])
pivot_pd=pivot_pd.reindex(sdf.index)
hover_pd=[]
for t in pivot_pd.index:
    rh=[]
    for q in q_cols:
        val=pivot_pd.loc[t,q]
        if pd.notna(val):
            rd=merton_df[(merton_df['ticker']==t)&(merton_df['quarter']==q)]
            d2d=rd['D2D'].values[0] if len(rd)>0 else np.nan
            cs=rd['credit_spread'].values[0] if len(rd)>0 else np.nan
            conv=bool(rd['converged'].values[0]) if len(rd)>0 else False
            cs_str=f'{cs*100:.2f}%' if pd.notna(cs) else 'N/A'
            rh.append(f'<b>{t}</b>  ·  {q}<br>PD: <b>{val*100:.2f}%</b><br>'
                      f'D2D: {d2d:.3f}<br>Spread: {cs_str}<br>'
                      f'{sector_map.get(t,"")}  ·  {tier_map.get(t,"")}<br>'
                      f'{"✓ converged" if conv else "⚠ not converged"}')
        else:
            rh.append(f'<b>{t}</b>  ·  {q}<br><i>No data</i>')
    hover_pd.append(rh)
shapes_pd,cur_sec=[],None
for i,t in enumerate(pivot_pd.index):
    s=sector_map.get(t,'')
    if s!=cur_sec and i>0:
        shapes_pd.append(dict(type='line',x0=-0.5,x1=len(q_cols)-0.5,
                               y0=i-0.5,y1=i-0.5,line=dict(color='white',width=3)))
    cur_sec=s
anns_pd,cur_sec,ss=[],None,0
tlist=list(pivot_pd.index)
for i,t in enumerate(tlist):
    s=sector_map.get(t,'')
    if s!=cur_sec:
        if cur_sec:
            anns_pd.append(dict(x=len(q_cols)+0.8,y=(ss+i-1)/2,text=f'<b>{cur_sec}</b>',
                                showarrow=False,xref='x',yref='y',
                                font=dict(size=9,color='#333'),xanchor='left'))
        cur_sec,ss=s,i
if cur_sec:
    anns_pd.append(dict(x=len(q_cols)+0.8,y=(ss+len(tlist)-1)/2,text=f'<b>{cur_sec}</b>',
                        showarrow=False,xref='x',yref='y',
                        font=dict(size=9,color='#333'),xanchor='left'))
fig_pd=go.Figure(go.Heatmap(
    z=pivot_pd.values*100,x=q_cols,y=pivot_pd.index.tolist(),
    text=hover_pd,hovertemplate='%{text}<extra></extra>',
    colorscale=[[0,'#F0F7EC'],[0.05,'#D4EDCA'],[0.15,'#88C870'],
                [0.30,'#4A9E36'],[0.50,'#E8856B'],[0.70,'#C0392B'],
                [0.85,'#8B0000'],[1.00,'#4A0000']],
    zmin=0,zmax=50,
    colorbar=dict(title=dict(text='PD (%)',side='right',font=dict(size=10)),
                  thickness=14,len=0.75,x=1.13,
                  tickvals=[0,5,10,20,30,50],
                  ticktext=['0%','5%','10%','20%','30%','50%+'],tickfont=dict(size=8))))
fig_pd.update_layout(
    title=dict(text='Merton PD  ·  All Companies × Quarters  |  Phase 3<br>'
               '<sup>PD=N(−d2)  ·  Green=low risk  ·  Red=high risk  ·  Hover for D2D and spread</sup>',
               font=dict(size=13,color=C_ACC),x=0.5,xanchor='center'),
    xaxis=dict(tickangle=-50,tickfont=dict(size=7.5),showgrid=False,domain=[0,0.87]),
    yaxis=dict(tickfont=dict(size=8.5),autorange='reversed',showgrid=False),
    height=max(750,len(pivot_pd.index)*15),margin=dict(l=85,r=170,t=70,b=70),
    shapes=shapes_pd,annotations=anns_pd,plot_bgcolor='white',paper_bgcolor='white')
fig_pd.show()
fig_pd.write_html(os.path.join(OUTPUT_DIR,'merton_pd_heatmap.html'))
print('  ✓ PD heatmap saved')

# ── VIZ 3: Spread comparison ──────────────────────────────────────────────────
print('\n→ [3/6] Spread comparison...')
rf_q2=rf_df[['hy_spread','ig_spread']].resample('Q').mean()
rf_q2.index=rf_q2.index.to_period('Q').astype(str)
tier_spreads={}
for tier in ['ig','grey','dist','def']:
    ticks=universe_df[(universe_df['tier']==tier)&(universe_df['in_merton']==True)]['ticker'].tolist()
    sub=merton_df[merton_df['ticker'].isin(ticks)&~merton_df['post_default']]
    if len(sub)>0:
        tier_spreads[tier]=sub.groupby('quarter')['credit_spread'].median().rename(f'spread_{tier}')
spread_df=pd.concat(tier_spreads.values(),axis=1).join(rf_q2)
spread_df['ts']=[q2ts(q) for q in spread_df.index]
spread_df=spread_df.dropna(subset=['ts']).sort_values('ts')
fig,ax=plt.subplots(figsize=(13,5.5))
for col,label,color,lw,ls in [
    ('spread_ig','Merton — IG',C_SAFE,2.0,'-'),
    ('spread_grey','Merton — Grey zone',C_GREY,1.8,'--'),
    ('spread_dist','Merton — Distressed',C_DIST,1.8,'-.'),
    ('spread_def','Merton — Default',  '#7B0000',2.0,':'),
]:
    if col in spread_df.columns:
        ax.plot(spread_df['ts'],spread_df[col].clip(upper=0.25)*100,
                color=color,lw=lw,linestyle=ls,label=label,alpha=0.85)
if 'hy_spread' in spread_df.columns:
    ax.plot(spread_df['ts'],spread_df['hy_spread']*100,
            color='#2C3E50',lw=2.0,alpha=0.6,label='Market HY OAS (FRED)')
if 'ig_spread' in spread_df.columns:
    ax.plot(spread_df['ts'],spread_df['ig_spread']*100,
            color='#7F8C8D',lw=1.5,alpha=0.6,label='Market IG OAS (FRED)')
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID shock','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'Bank crisis','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.7)
    ax.text(ts+pd.Timedelta(days=20),ax.get_ylim()[1]*0.85 if ax.get_ylim()[1]>0 else 2,
            lbl,fontsize=7.5,color=color,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))
ax.set_ylabel('Credit Spread (%)',fontsize=9)
ax.set_xlabel('Date',fontsize=9)
ax.set_ylim(0,25)
ax.set_title('Merton Model-Implied Spreads vs Market Spreads  |  Phase 3',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'Post-default excluded  ·  Spreads capped at 25%  ·  '
        'Merton tiers vs FRED HY/IG OAS benchmarks',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor='#ddd',ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'merton_spread_comparison.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Spread comparison saved')

# ── VIZ 4: PD vs Z-Score ─────────────────────────────────────────────────────
print('\n→ [4/6] PD vs Z-Score...')
pd_vs_z=merton_df[~merton_df['post_default']][
    ['ticker','quarter','PD','D2D','sector','tier']].merge(
    zscore_df[['ticker','quarter','Z_score','zone']],on=['ticker','quarter'],how='inner')
pd_vs_z=pd_vs_z[pd_vs_z['PD'].notna()&pd_vs_z['Z_score'].notna()&
                 pd_vs_z['PD'].between(0,0.5)&pd_vs_z['Z_score'].between(-1,7)]
fig,ax=plt.subplots(figsize=(12,7))
for zone,label,color,marker,size,alpha in [
    ('safe',    'Safe (Z>2.99)',     C_SAFE,'o',28,0.30),
    ('grey',    'Grey zone',          C_GREY,'s',28,0.40),
    ('distress','Distress (Z<1.81)', C_DIST,'^',38,0.30),
]:
    sub=pd_vs_z[pd_vs_z['zone']==zone]
    ax.scatter(sub['Z_score'],sub['PD']*100,c=color,s=size,alpha=alpha,
               marker=marker,label=f'{label} (n={len(sub)})',edgecolors='none')
for t,lbl in [('SIVB','SVB'),('CHK','CHK'),('BBBY','BBBY'),('HTZ','HTZ')]:
    sub=pd_vs_z[pd_vs_z['ticker']==t]
    if len(sub)==0: continue
    last=sub.nlargest(1,'PD').iloc[0]
    ax.scatter(last['Z_score'],last['PD']*100,c='#7B0000',s=130,marker='*',
               zorder=6,edgecolors='white',linewidths=0.5)
    ax.annotate(lbl,xy=(last['Z_score'],last['PD']*100),
                xytext=(7,5),textcoords='offset points',
                fontsize=8,color='#7B0000',fontweight='bold')
r_val=0
clean=pd_vs_z[pd_vs_z['PD']<0.15].dropna(subset=['Z_score','PD'])
if len(clean)>50:
    m,b,r_val,p_val,_=stats.linregress(clean['Z_score'],clean['PD']*100)
    xl=np.linspace(-1,7,100)
    ax.plot(xl,m*xl+b,'k--',lw=1.5,alpha=0.6,
            label=f'OLS (PD<15%)  R={r_val:.2f}, p={p_val:.3f}')
    print(f'  OLS: R={r_val:.3f}, p={p_val:.4f}')
ax.axvspan(-1.5,1.81,color='#FDF0EF',alpha=0.2,zorder=0)
ax.axvspan(1.81,2.99,color='#FDF6E3',alpha=0.2,zorder=0)
ax.axvspan(2.99,7.5, color='#F0F7EC',alpha=0.2,zorder=0)
ax.axvline(1.81,color=C_DIST,lw=1.2,linestyle='--',alpha=0.6)
ax.axvline(2.99,color=C_SAFE,lw=1.2,linestyle='--',alpha=0.6)
ax.axhline(5,color='gray',lw=0.8,linestyle=':',alpha=0.5)
ax.text(6.8,5.4,'PD=5%',fontsize=7.5,color='gray',ha='right')
for x,lbl,c in [(0.9,'DISTRESS\nZONE',C_DIST),(2.4,'GREY\nZONE',C_GREY),(5.0,'SAFE\nZONE',C_SAFE)]:
    ax.text(x,51,lbl,ha='center',fontsize=7.5,color=c,fontstyle='italic',va='top')
ax.set_xlabel('Altman Z-Score (Challenger)',fontsize=9)
ax.set_ylabel('Merton PD % (Champion)',fontsize=9)
ax.set_xlim(-1.5,7.5); ax.set_ylim(-1,53)
ax.set_title('Merton PD vs Altman Z-Score  ·  Champion vs Challenger Preview  |  Phase 3',
             fontsize=11,fontweight='500',color=C_ACC,pad=14)
ax.text(0.5,1.005,'★ = peak stress before default  ·  Post-default excluded  ·  '
        'Negative correlation expected (high Z → low PD)',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper right',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'merton_pd_vs_zscore.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ PD vs Z-Score saved')

# ── VIZ 5: D2D ranking ───────────────────────────────────────────────────────
print('\n→ [5/6] D2D ranking...')
clean_m=(merton_df[(merton_df['quarter']==latest_q_m)&~merton_df['post_default']]
         .sort_values('D2D',ascending=False).reset_index(drop=True))
top15=clean_m.head(15)
bot15=clean_m.tail(15).sort_values('D2D')
plot_d=pd.concat([bot15,top15]).drop_duplicates('ticker').sort_values('D2D')
y_pos=np.arange(len(plot_d))
bar_c=[C_DIST if d<1 else C_GREY if d<2 else C_SAFE for d in plot_d['D2D']]
fig,ax=plt.subplots(figsize=(13,max(9,len(plot_d)*0.42)))
ax.axvspan(-3,0,color='#FDF0EF',alpha=0.35,zorder=0)
ax.axvspan(0,1, color='#FEF9EC',alpha=0.30,zorder=0)
ax.axvspan(1,2, color='#FDF6E3',alpha=0.30,zorder=0)
ax.axvspan(2,20,color='#F0F7EC',alpha=0.30,zorder=0)
ax.axvline(0,color=C_DIST,lw=1.8,linestyle='--',alpha=0.85,zorder=2)
ax.axvline(1,color=C_GREY,lw=1.2,linestyle='--',alpha=0.65,zorder=2)
ax.axvline(2,color=C_SAFE,lw=1.2,linestyle='--',alpha=0.65,zorder=2)
n=len(plot_d); label_y=n-0.7
for x,lbl,c in [(-1.3,'DEFAULT\nZONE',C_DIST),(0.05,'HIGH\nRISK','#C87000'),
                  (1.05,'WATCH',C_GREY),(2.1,'SAFE',C_SAFE)]:
    ax.text(x,label_y,lbl,fontsize=7.5,color=c,fontweight='bold',va='top',ha='left')
for i,(_,row) in enumerate(plot_d.iterrows()):
    c=C_DIST if row['D2D']<1 else C_GREY if row['D2D']<2 else C_SAFE
    ax.plot([0,row['D2D']],[i,i],color=c,lw=1.5,alpha=0.40,zorder=1)
ax.scatter(plot_d['D2D'].values,y_pos,c=bar_c,s=85,zorder=4,
           edgecolors='white',linewidths=0.8)
for i,(_,row) in enumerate(plot_d.iterrows()):
    c=C_DIST if row['D2D']<1 else C_GREY if row['D2D']<2 else C_SAFE
    x_off=0.15 if row['D2D']>=0 else -0.15
    ax.text(row['D2D']+x_off,i,f'{row["D2D"]:.2f}',
            va='center',ha='left' if row['D2D']>=0 else 'right',
            fontsize=7.5,color=c)
    ax.text(-0.4,i,f'{row["PD"]*100:.1f}%',va='center',ha='right',fontsize=7.5,color='#666')
    ax.text(19.5,i,sector_map.get(row['ticker'],''),va='center',ha='right',fontsize=7,color='#999')
ax.text(-0.4,n-0.3,'PD',va='bottom',ha='right',fontsize=8,color='#555',fontweight='bold')
ax.set_yticks(y_pos); ax.set_yticklabels(plot_d['ticker'],fontsize=9.5,fontweight='500')
ax.set_xlim(-1.8,20); ax.set_ylim(-0.8,n-0.2)
ax.set_xlabel('Distance to Default (d2)',fontsize=9)
ax.set_title(f'Merton D2D Ranking — {latest_q_m}  |  Phase 3\n'
             'Top & bottom companies  ·  Left=Merton PD  ·  '
             'D2D=σ above default point  ·  Post-default excluded',
             fontsize=10.5,fontweight='500',color=C_ACC,pad=12)
ax.grid(axis='x',alpha=0.2,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'merton_d2d_ranking.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ D2D ranking saved')

# ── VIZ 6: Diagnostics ───────────────────────────────────────────────────────
print('\n→ [6/6] Diagnostics...')
vol_comp=merton_df[merton_df['converged']==True].copy()
vol_comp=vol_comp[vol_comp['sigma_E'].between(0.05,1.5)&vol_comp['sigma_A'].between(0.005,1.0)]
sec_colors={'Tech':'#185FA5','Banks':'#854F0B','Energy':'#3B6D11','Healthcare':'#7F77DD',
            'Industrial':'#888780','Real Estate':'#D85A30','Retail':'#D4537E','Telecom':'#1D9E75'}
fig,axes=plt.subplots(1,2,figsize=(15,6.5))
ax=axes[0]
for sec,grp in vol_comp.groupby('sector'):
    ax.scatter(grp['sigma_E'],grp['sigma_A'],c=sec_colors.get(sec,'#888'),
               s=16,alpha=0.30,label=sec,edgecolors='none')
xl=np.linspace(0,1.2,100)
ax.plot(xl,xl,'k--',lw=1.0,alpha=0.4,label='sA=sE (zero leverage)')
ax.plot(xl,xl*0.5,'gray',lw=0.8,linestyle=':',alpha=0.4,label='sA=0.5×sE')
svb_v=vol_comp[(vol_comp['ticker']=='SIVB')&(vol_comp['quarter']=='2022Q3')]
if len(svb_v)>0:
    r=svb_v.iloc[0]
    ax.scatter(r['sigma_E'],r['sigma_A'],c='#A32D2D',s=160,marker='*',
               zorder=8,edgecolors='white',linewidths=0.5)
    ax.annotate('SVB Q3 2022',xy=(r['sigma_E'],r['sigma_A']),
                xytext=(8,8),textcoords='offset points',fontsize=8,
                color='#A32D2D',fontweight='bold',
                arrowprops=dict(arrowstyle='->',color='#A32D2D'))
ax.set_xlabel('Equity Volatility (σ_E)',fontsize=9)
ax.set_ylabel('Asset Volatility (σ_A)',fontsize=9)
ax.set_title('Asset Vol vs Equity Vol  (Merton de-levering)',
             fontsize=10,fontweight='500',color=C_ACC)
ax.legend(loc='upper left',fontsize=7.5,framealpha=0.9,edgecolor='#ddd',ncol=2)
ax.set_xlim(0,1.2); ax.set_ylim(0,0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2=axes[1]
tier_order=['ig','grey','dist','def']
tier_labels=['Investment\nGrade','Grey\nZone','Distressed','Default/\nFailed']
tier_colors=[C_SAFE,C_GREY,C_DIST,'#7B0000']
# Use pre-default only for cleaner boxplot
data_d2d=[merton_df[(merton_df['tier']==t)&~merton_df['post_default']]['D2D'].dropna().values
          for t in tier_order]
bps=ax2.boxplot(data_d2d,patch_artist=True,notch=False,
                medianprops=dict(color='white',linewidth=2.5),
                whiskerprops=dict(linewidth=1.2,color='#888'),
                capprops=dict(linewidth=1.2,color='#888'),
                flierprops=dict(marker='o',markersize=2,alpha=0.2,markeredgewidth=0),
                showfliers=False)   # hide extreme outliers for readability
for patch,color in zip(bps['boxes'],tier_colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax2.axhline(0,color=C_DIST,lw=1.5,linestyle='--',alpha=0.8,label='Default threshold')
ax2.axhline(1,color=C_GREY,lw=1.0,linestyle=':',alpha=0.6)
ax2.set_xticks([1,2,3,4]); ax2.set_xticklabels(tier_labels,fontsize=9)
ax2.set_ylabel('Distance to Default',fontsize=9)
ax2.set_title('D2D Distribution by Risk Tier\n(Post-default excluded, outliers hidden)',
              fontsize=10,fontweight='500',color=C_ACC)
for i,(vals,color) in enumerate(zip(data_d2d,tier_colors)):
    if len(vals)>0:
        med=np.median(vals)
        ax2.text(i+1,med+0.2,f'{med:.2f}',ha='center',va='bottom',
                 fontsize=8,fontweight='bold',color=color)
ax2.legend(fontsize=8,framealpha=0.9,edgecolor='#ddd')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.suptitle('Phase 3  ·  Merton Model Diagnostics',
             fontsize=12,fontweight='500',color=C_ACC,y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'merton_vol_d2d_diagnostics.png'),
            dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Diagnostics saved')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'PHASE 3 COMPLETE')
print(f'{"="*60}')
print(f'  Companies   : {merton_df["ticker"].nunique()}')
print(f'  Obs         : {len(merton_df):,}')
print(f'  Conv rate   : {merton_df["converged"].mean()*100:.1f}%')
print(f'  Post-default: {merton_df["post_default"].sum()} flagged')
for f in ['merton_results.csv','merton_summary.csv','merton_svb_d2d.png',
          'merton_pd_heatmap.html','merton_spread_comparison.png',
          'merton_pd_vs_zscore.png','merton_d2d_ranking.png',
          'merton_vol_d2d_diagnostics.png']:
    path=os.path.join(OUTPUT_DIR,f)
    if os.path.exists(path):
        print(f'  ✓ {f:<45} {os.path.getsize(path)/1024:6.1f} KB')
print(f'\n  SR 11-7: CHAMPION=Merton ✓  CHALLENGER=Z-Score ✓')
print(f'  Ready for Phase 4: Basel III PD/LGD/EAD/CVA')
print(f'{"="*60}')


# ============================================================
# PHASE 4 — COMPLETE (One Cell — Final)
# Basel III PD/LGD/EAD/CVA + Monte Carlo CVA
# 7 Visualizations
# ============================================================

import os, warnings
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats         import norm
from google.colab        import drive

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':0.2,'grid.linestyle':'--',
})

drive.mount('/content/drive', force_remount=False)
BASE_DIR   = '/content/drive/MyDrive/P4_CreditRisk'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CACHE_DIR  = os.path.join(BASE_DIR, 'cache')

# ── Load ──────────────────────────────────────────────────────────────────────
merton_df   = pd.read_csv(os.path.join(OUTPUT_DIR, 'merton_results.csv'))
universe_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'universe.csv'))
params_df   = pd.read_csv(os.path.join(OUTPUT_DIR, 'parameter_registry.csv'))
rf_df       = pd.read_csv(os.path.join(OUTPUT_DIR, 'fred_rates.csv'),
                           parse_dates=['date'], index_col='date')

PARAMS     = dict(zip(params_df['parameter'], params_df['value']))
sector_map = universe_df.set_index('ticker')['sector'].to_dict()
tier_map   = universe_df.set_index('ticker')['tier'].to_dict()

C_SAFE='#2D6A27'; C_GREY='#B07D10'; C_DIST='#C0392B'; C_ACC='#2C3E50'

DEFAULT_DATES = {
    'CHK':'2020-06-28','HTZ':'2020-05-22','BBBY':'2023-04-23',
    'SIVB':'2023-03-10','SBNY':'2023-03-12','FRC':'2023-05-01',
    'CBL':'2020-11-01','WPG':'2021-06-13',
}

def q2ts(q):
    try:
        yr,qn=int(q[:4]),int(q[5])
        return pd.Timestamp(year=yr,month=(qn-1)*3+1,day=1)
    except: return pd.NaT

def is_post(ticker, quarter):
    if ticker not in DEFAULT_DATES: return False
    return q2ts(quarter) > pd.Timestamp(DEFAULT_DATES[ticker])

merton_df['ts']          = merton_df['quarter'].apply(q2ts)
merton_df['post_default']= merton_df.apply(
    lambda r: is_post(r['ticker'],r['quarter']), axis=1)

df = merton_df[~merton_df['post_default']].copy()
print('✓ Data loaded')
print(f'  Pre-default obs: {len(df):,}  Companies: {df["ticker"].nunique()}')


# ══════════════════════════════════════════════════════════════════════════════
# BLOCKS 1-6: Compute all Basel III metrics
# ══════════════════════════════════════════════════════════════════════════════

# LGD
LGD_BASE={'ig':0.35,'grey':0.45,'dist':0.60,'def':0.75,'etf':0.00}
SECTOR_LGD_ADJ={'Tech':0.90,'Banks':1.10,'Energy':0.85,'Healthcare':0.95,
                'Industrial':0.85,'Real Estate':0.75,'Retail':1.05,'Telecom':0.90}
df['LGD']         = df.apply(
    lambda r: min(LGD_BASE.get(r['tier'],0.45)*SECTOR_LGD_ADJ.get(r['sector'],1.0),0.95),axis=1)
df['LGD_downturn']= (df['LGD']*1.25).clip(upper=0.95)

# EAD
df['EAD']         = df['D_debt'].copy()
df['EAD_adjusted']= df['EAD']*(1+df['rf_rate']*0.25)

# EL
df['EL']          = df['PD']*df['LGD']*df['EAD']
df['EL_downturn'] = df['PD']*df['LGD_downturn']*df['EAD_adjusted']
df['EL_pct']      = df['PD']*df['LGD']

# UL + EC
MIN_PD=1e-6; CONF=0.999
def asset_corr(pd_v):
    pd_v=max(pd_v,MIN_PD)
    e=(1-np.exp(-50*pd_v))/(1-np.exp(-50))
    return 0.12*e+0.24*(1-e)
def ec(ead,lgd,pd_v,rho):
    pd_v=max(pd_v,MIN_PD)
    if pd_v>=1 or rho>=1: return 0.0
    try:
        wcdr=norm.cdf(np.sqrt(1/(1-rho))*norm.ppf(pd_v)+np.sqrt(rho/(1-rho))*norm.ppf(CONF))
        return ead*lgd*max(wcdr-pd_v,0.0)
    except: return 0.0
df['asset_corr']=df['PD'].apply(asset_corr)
df['UL']=df['EAD']*df['LGD']*np.sqrt(df['PD']*(1-df['PD']))
df['EC']=df.apply(lambda r: ec(r['EAD'],r['LGD'],r['PD'],r['asset_corr']),axis=1)
df['EC_pct']=df['EC']/df['EAD'].replace(0,np.nan)

# Monte Carlo CVA (Vasicek)
kappa=float(PARAMS['vasicek_kappa']); theta=float(PARAMS['vasicek_theta'])
sigma_r=0.03; T_cva=1.0
N_PATHS=int(PARAMS['mc_n_paths']); N_STEPS=int(PARAMS['mc_n_steps'])
dt=T_cva/N_STEPS; r0=rf_df['rf_1y'].dropna().iloc[-1]; MIN_LAM=1e-6

rng=np.random.default_rng(42)
r_paths=np.zeros((N_PATHS,N_STEPS+1)); r_paths[:,0]=r0
for s in range(N_STEPS):
    dW=rng.standard_normal(N_PATHS)*np.sqrt(dt)
    r_paths[:,s+1]=np.maximum(r_paths[:,s]+kappa*(theta-r_paths[:,s])*dt+sigma_r*dW,0.001)
DF_mean=np.exp(-np.cumsum(r_paths[:,1:],axis=1)*dt).mean(axis=0)
t_grid=np.linspace(dt,T_cva,N_STEPS)

def cva_fn(pd_a,lgd,ead):
    pd_a=max(pd_a,MIN_PD)
    if pd_a>=1: return lgd*ead
    try:
        lam=max(-np.log(1-pd_a),MIN_LAM)
        dpd=np.diff(1-np.exp(-lam*t_grid),prepend=0)
        return max(lgd*ead*np.sum(DF_mean*dpd),0.0)
    except: return 0.0

df['CVA']    =df.apply(lambda r: cva_fn(r['PD'],r['LGD'],r['EAD']),axis=1)
df['CVA_pct']=df['CVA']/df['EAD'].replace(0,np.nan)
mkt_pd=df[df['tier']=='ig']['PD'].mean()
df['DVA'] =df['CVA']*(1-mkt_pd)
df['BCVA']=df['CVA']-df['DVA']

# RWA
def pd_rw(p):
    if p<0.001: return 0.20
    if p<0.005: return 0.50
    if p<0.020: return 1.00
    if p<0.100: return 1.50
    return 2.50
df['risk_weight'] =df['PD'].apply(pd_rw)
df['RWA']         =df['EAD']*df['risk_weight']
df['reg_capital'] =df['RWA']*0.08
df['tier1_capital']=df['RWA']*0.06
df['EC_vs_reg']   =df['EC']/df['reg_capital'].replace(0,np.nan)

latest_q=df['quarter'].max()
pf=df[df['quarter']==latest_q]
print(f'\n  Portfolio ({latest_q}):')
print(f'    EAD=${pf["EAD"].sum()/1e3:,.1f}B  '
      f'EL=${pf["EL"].sum()/1e3:,.3f}B  '
      f'EC=${pf["EC"].sum()/1e3:,.2f}B')
print(f'    RWA=${pf["RWA"].sum()/1e3:,.1f}B  '
      f'RegCap=${pf["reg_capital"].sum()/1e3:,.2f}B  '
      f'CVA=${pf["CVA"].sum()/1e3:,.3f}B')

# Save
save_cols=['ticker','quarter','sector','tier','PD','LGD','LGD_downturn',
           'EAD','EAD_adjusted','EL','EL_downturn','EL_pct','UL','asset_corr',
           'EC','EC_pct','CVA','CVA_pct','DVA','BCVA',
           'risk_weight','RWA','reg_capital','tier1_capital','EC_vs_reg']
df[save_cols].to_csv(os.path.join(OUTPUT_DIR,'basel_results.csv'),index=False)
(df.groupby(['ticker','sector','tier'])
   .agg(mean_PD=('PD','mean'),mean_LGD=('LGD','mean'),
        mean_EL_pct=('EL_pct','mean'),mean_CVA=('CVA','mean'),
        mean_EC_pct=('EC_pct','mean'),mean_RWA=('RWA','mean'))
   .round(6).reset_index()
   .to_csv(os.path.join(OUTPUT_DIR,'basel_summary.csv'),index=False))
print(f'✓ Saved: {len(df):,} rows')


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── VIZ 1: EL Heatmap ────────────────────────────────────────────────────────
print('\n→ [1/7] EL heatmap...')
pivot_el=df.pivot_table(index='ticker',columns='quarter',values='EL_pct',aggfunc='mean')
q_cols=sorted([c for c in pivot_el.columns if '2015Q1'<=c<='2023Q4'])
pivot_el=pivot_el[q_cols]
sdf=pd.DataFrame({'sector':pd.Series(sector_map),'v':pivot_el.mean(axis=1)})
sdf=sdf[sdf.index.isin(pivot_el.index)].sort_values(['sector','v'],ascending=[True,False])
pivot_el=pivot_el.reindex(sdf.index)
hover_el=[]
for t in pivot_el.index:
    rh=[]
    for q in q_cols:
        val=pivot_el.loc[t,q]
        if pd.notna(val):
            sub=df[(df['ticker']==t)&(df['quarter']==q)]
            if len(sub)>0:
                r=sub.iloc[0]
                rh.append(f'<b>{t}</b>  ·  {q}<br>'
                          f'EL: <b>{val*100:.3f}%</b><br>'
                          f'PD={r["PD"]*100:.2f}%  LGD={r["LGD"]:.2f}<br>'
                          f'EL=${r["EL"]:,.0f}M<br>'
                          f'{sector_map.get(t,"")}  ·  {tier_map.get(t,"")}')
            else: rh.append(f'<b>{t}</b>  ·  {q}<br>EL={val*100:.3f}%')
        else: rh.append(f'<b>{t}</b>  ·  {q}<br><i>No data</i>')
    hover_el.append(rh)
shapes_el,cur_sec=[],None
for i,t in enumerate(pivot_el.index):
    s=sector_map.get(t,'')
    if s!=cur_sec and i>0:
        shapes_el.append(dict(type='line',x0=-0.5,x1=len(q_cols)-0.5,
                               y0=i-0.5,y1=i-0.5,line=dict(color='white',width=3)))
    cur_sec=s
anns_el,cur_sec,ss=[],None,0
tlist=list(pivot_el.index)
for i,t in enumerate(tlist):
    s=sector_map.get(t,'')
    if s!=cur_sec:
        if cur_sec:
            anns_el.append(dict(x=len(q_cols)+0.8,y=(ss+i-1)/2,text=f'<b>{cur_sec}</b>',
                                showarrow=False,xref='x',yref='y',
                                font=dict(size=9,color='#333'),xanchor='left'))
        cur_sec,ss=s,i
if cur_sec:
    anns_el.append(dict(x=len(q_cols)+0.8,y=(ss+len(tlist)-1)/2,text=f'<b>{cur_sec}</b>',
                        showarrow=False,xref='x',yref='y',
                        font=dict(size=9,color='#333'),xanchor='left'))
fig_el=go.Figure(go.Heatmap(
    z=pivot_el.values*100,x=q_cols,y=pivot_el.index.tolist(),
    text=hover_el,hovertemplate='%{text}<extra></extra>',
    colorscale=[[0,'#F0F7EC'],[0.05,'#D4EDCA'],[0.15,'#88C870'],
                [0.35,'#E8856B'],[0.60,'#C0392B'],[0.85,'#8B0000'],[1.00,'#4A0000']],
    zmin=0,zmax=5,
    colorbar=dict(title=dict(text='EL Rate (%)',side='right',font=dict(size=10)),
                  thickness=14,len=0.75,x=1.13,
                  tickvals=[0,0.5,1,2,3,5],
                  ticktext=['0%','0.5%','1%','2%','3%','5%+'],
                  tickfont=dict(size=8))))
fig_el.update_layout(
    title=dict(text='Expected Loss Rate  ·  All Companies × Quarters  |  Phase 4<br>'
               '<sup>EL%=PD×LGD  ·  Green=low  ·  Red=high  ·  Hover for details</sup>',
               font=dict(size=13,color=C_ACC),x=0.5,xanchor='center'),
    xaxis=dict(tickangle=-50,tickfont=dict(size=7.5),showgrid=False,domain=[0,0.87]),
    yaxis=dict(tickfont=dict(size=8.5),autorange='reversed',showgrid=False),
    height=max(800,len(pivot_el.index)*15),margin=dict(l=85,r=170,t=70,b=70),
    shapes=shapes_el,annotations=anns_el,plot_bgcolor='white',paper_bgcolor='white')
fig_el.show()
fig_el.write_html(os.path.join(OUTPUT_DIR,'basel_el_heatmap.html'))
print('  ✓ EL heatmap saved')


# ── VIZ 2: EL/UL/EC by sector — log scale ────────────────────────────────────
print('\n→ [2/7] EL/UL/EC by sector...')
sec_g=(pf.groupby('sector')
         .agg(total_EL=('EL','sum'),total_UL=('UL','sum'),
              total_EC=('EC','sum'),total_EAD=('EAD','sum'))
         .reset_index())
sec_g['EL_pct']=sec_g['total_EL']/sec_g['total_EAD']*100
sec_g['UL_pct']=sec_g['total_UL']/sec_g['total_EAD']*100
sec_g['EC_pct']=sec_g['total_EC']/sec_g['total_EAD']*100
sec_g=sec_g.sort_values('EL_pct',ascending=True)

fig,ax=plt.subplots(figsize=(13,6.5))
y=np.arange(len(sec_g)); w=0.22
# Clip to MIN so log scale works
EL_plot=sec_g['EL_pct'].clip(lower=0.0001)
UL_plot=sec_g['UL_pct'].clip(lower=0.0001)
EC_plot=sec_g['EC_pct'].clip(lower=0.0001)
b1=ax.barh(y+w,EL_plot,w,color=C_DIST,alpha=0.82,label='EL Rate (%)')
b2=ax.barh(y,  UL_plot,w,color=C_GREY,alpha=0.82,label='UL Rate (%)')
b3=ax.barh(y-w,EC_plot,w,color=C_ACC, alpha=0.82,label='EC Rate (%)')
for bars,actual in [(b1,sec_g['EL_pct']),(b2,sec_g['UL_pct']),(b3,sec_g['EC_pct'])]:
    for bar,v in zip(bars,actual):
        bw=bar.get_width()
        lbl=f'{v:.4f}%' if v<0.01 else f'{v:.3f}%' if v<0.1 else f'{v:.2f}%'
        ax.text(bw*1.1,bar.get_y()+bar.get_height()/2,
                lbl,va='center',fontsize=6.8,color='#444')
ax.set_xscale('log')
ax.set_xlim(0.00005,30)
ax.set_yticks(y); ax.set_yticklabels(sec_g['sector'],fontsize=9.5)
ax.set_xlabel('% of EAD  (log scale)',fontsize=9)
ax.set_title(f'EL / UL / EC by Sector — {latest_q}  |  Phase 4',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,
        'Log scale — all sectors visible  ·  '
        'Real Estate / Tech / Energy near-zero EL = low PD (correct, not missing)',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='lower right',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.xaxis.grid(True,which='both',alpha=0.2,linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'basel_el_ul_ec.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ EL/UL/EC saved')


# ── VIZ 3: Portfolio EL over time ────────────────────────────────────────────
print('\n→ [3/7] Portfolio EL over time...')
port_t=(df.groupby('quarter')
          .agg(total_EL=('EL','sum'),total_EAD=('EAD','sum'),
               total_CVA=('CVA','sum'))
          .reset_index())
port_t['EL_rate'] =port_t['total_EL'] /port_t['total_EAD']*100
port_t['CVA_rate']=port_t['total_CVA']/port_t['total_EAD']*100
port_t['ts']=port_t['quarter'].apply(q2ts)
port_t=port_t.sort_values('ts')

fig,(ax,ax2)=plt.subplots(2,1,figsize=(13,9),
    gridspec_kw={'height_ratios':[1.6,1],'hspace':0.35})
ax.fill_between(port_t['ts'],port_t['EL_rate'],alpha=0.20,color=C_DIST)
ax.plot(port_t['ts'],port_t['EL_rate'],color=C_DIST,lw=2.2,label='Portfolio EL Rate (%)')
ax2r=ax.twinx()
ax2r.plot(port_t['ts'],port_t['total_EL']/1e3,color='#7B0000',
          lw=1.5,linestyle='--',alpha=0.7,label='Total EL ($B)')
ax2r.set_ylabel('Total EL ($B)',fontsize=9,color='#7B0000')
ax2r.tick_params(axis='y',labelcolor='#7B0000')
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID shock','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'Bank crisis','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.8)
    ax.text(ts+pd.Timedelta(days=25),ax.get_ylim()[1]*0.88 if ax.get_ylim()[1]>0 else 0.05,
            lbl,fontsize=7.5,color=color,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))
ax.set_ylabel('EL Rate (% of EAD)',fontsize=9)
ax.set_title('Portfolio Expected Loss Over Time  |  Phase 4',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,
        'EL=PD×LGD×EAD  ·  Model captures COVID shock, Fed hike cycle, bank crisis',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
lines1,lbl1=ax.get_legend_handles_labels()
lines2,lbl2=ax2r.get_legend_handles_labels()
ax.legend(lines1+lines2,lbl1+lbl2,loc='upper left',fontsize=8.5,
          framealpha=0.9,edgecolor='#ddd')
ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax.spines['top'].set_visible(False)
ax2.stackplot(port_t['ts'],port_t['EL_rate'],port_t['CVA_rate'],
              labels=['EL Rate','CVA Rate'],colors=[C_DIST,'#7B0000'],alpha=0.65)
ax2.set_ylabel('Rate (% of EAD)',fontsize=9)
ax2.set_title('EL + CVA Stack Over Time',fontsize=10,fontweight='500',color=C_ACC)
ax2.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax2.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor='#ddd')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR,'basel_el_over_time.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Portfolio EL over time saved')


# ── VIZ 4: CVA timeline ───────────────────────────────────────────────────────
print('\n→ [4/7] CVA timeline...')
cva_groups={
    'Banks (IG)'      :['JPM','BAC','GS','WFC'],
    'Banks (Stressed)':['SIVB','WAL','PACW'],
    'Tech (IG)'       :['AAPL','MSFT','NVDA','GOOGL'],
    'Distressed'      :['CHK','AMC','BBBY','LUMN'],
}
cva_colors={'Banks (IG)':'#2471A3','Banks (Stressed)':'#E74C3C',
            'Tech (IG)':'#2D6A27','Distressed':'#7B0000'}

fig,(ax,ax2)=plt.subplots(2,1,figsize=(13,10),
    gridspec_kw={'height_ratios':[1.6,1],'hspace':0.35})
for group,tickers in cva_groups.items():
    grp=(df[df['ticker'].isin(tickers)]
         .groupby('quarter')[['CVA','EAD']].sum().reset_index())
    grp['CVA_pct']=grp['CVA']/grp['EAD']*100
    grp['ts']=grp['quarter'].apply(q2ts)
    ax.plot(grp.sort_values('ts')['ts'],grp.sort_values('ts')['CVA_pct'],
            color=cva_colors[group],lw=2.0,label=group,alpha=0.85)
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID shock','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'SVB collapse','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.7)
    ax.text(ts+pd.Timedelta(days=25),ax.get_ylim()[1]*0.88 if ax.get_ylim()[1]>0 else 0.1,
            lbl,fontsize=7.5,color=color,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))
ax.set_ylabel('CVA (% of EAD)',fontsize=9)
ax.set_title('CVA by Group — Monte Carlo Vasicek  |  Phase 4',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,
        f'CVA=(1-RR)×∫DF(t)×dPD(t)  ·  Vasicek κ={kappa} θ={theta} σ={sigma_r}  ·  '
        f'{N_PATHS:,} paths',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# SVB CVA bars — cap y-axis at 15,000M, annotate the spike
svb_pre=df[df['ticker']=='SIVB'].sort_values('ts')
if len(svb_pre)>0:
    bar_c=[C_DIST if q>='2022Q1' else C_SAFE for q in svb_pre['quarter']]
    ax2.bar(svb_pre['ts'],svb_pre['CVA'].clip(upper=15000),
            width=70,color=bar_c,alpha=0.78)
    # Annotate bars that were clipped
    clipped=svb_pre[svb_pre['CVA']>15000]
    for _,r in clipped.iterrows():
        ax2.text(r['ts'],15000*1.02,f'${r["CVA"]/1000:,.0f}B↑',
                 ha='center',fontsize=7,color='#7B0000',fontweight='bold')
    # Annotate Q3 2022
    q3=svb_pre[svb_pre['quarter']=='2022Q3']
    if len(q3)>0:
        ax2.annotate(f'Q3 2022\n${q3["CVA"].values[0]:,.0f}M',
                     xy=(q3['ts'].values[0],q3['CVA'].values[0]),
                     xytext=(30,15),textcoords='offset points',
                     fontsize=8,color='#A32D2D',fontweight='bold',
                     arrowprops=dict(arrowstyle='->',color='#A32D2D',lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#A32D2D',alpha=0.9))
    ax2.set_ylabel('SVB CVA ($M)',fontsize=9)
    ax2.set_ylim(0,16500)
    ax2.set_title('SVB CVA — Quarterly ($M)  ·  Y-axis capped at $15,000M',
                  fontsize=10,fontweight='500',color=C_ACC)
    ax2.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
    for ts,lbl,color in [(pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                          (pd.Timestamp('2023-03-10'),'Collapse','#A32D2D')]:
        ax2.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.8)
        ax2.text(ts+pd.Timedelta(days=25),14000,lbl,fontsize=7.5,color=color)
    gp=plt.Rectangle((0,0),1,1,fc=C_SAFE,alpha=0.78)
    rp=plt.Rectangle((0,0),1,1,fc=C_DIST,alpha=0.78)
    ax2.legend([gp,rp],['Pre-2022','2022+ (stressed)'],
               fontsize=8,loc='upper left',framealpha=0.9,edgecolor='#ddd')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR,'basel_cva_timeline.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ CVA timeline saved')


# ── VIZ 5: PD → LGD → EL waterfall ──────────────────────────────────────────
print('\n→ [5/7] EL waterfall...')
top20=(df[df['quarter']==latest_q].nlargest(20,'EL').sort_values('EL',ascending=True))
fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
for ax_,col,label,cfn,is_pct in [
    (axes[0],'PD',    'PD (%)',      lambda v:C_DIST if v>0.05 else C_GREY if v>0.01 else C_SAFE,True),
    (axes[1],'LGD',   'LGD',         lambda v:C_DIST if v>0.55 else C_GREY if v>0.40 else C_SAFE,False),
    (axes[2],'EL_pct','EL Rate (%)', lambda v:C_DIST if v>0.02 else C_GREY if v>0.005 else C_SAFE,True),
]:
    vals=top20[col].values*(100 if is_pct else 1)
    raw =top20[col].values
    ax_.barh(np.arange(len(top20)),vals,
             color=[cfn(v) for v in raw],alpha=0.78,
             edgecolor='white',linewidth=0.5)
    for i,v in enumerate(vals):
        if v>0:
            ax_.text(v+vals.max()*0.01,i,
                     f'{v:.2f}{"%" if is_pct else ""}',
                     va='center',fontsize=7.5,color='#444')
    ax_.set_xlabel(label,fontsize=9)
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
    ax_.grid(axis='x',alpha=0.2,linestyle='--')
axes[0].set_yticks(np.arange(len(top20)))
axes[0].set_yticklabels(top20['ticker'],fontsize=9)
fig.suptitle(f'PD → LGD → EL Rate  ·  Top 20 by EL — {latest_q}  |  Phase 4',
             fontsize=11,fontweight='500',color=C_ACC,y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'basel_pd_lgd_el.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ EL waterfall saved')


# ── VIZ 6: EC vs Reg Capital ─────────────────────────────────────────────────
print('\n→ [6/7] EC vs Reg Capital...')
pf2=df[df['quarter']==latest_q].copy()
pf2['EC_rate'] =pf2['EC'] /pf2['EAD']*100
pf2['reg_rate']=pf2['reg_capital']/pf2['EAD']*100
pf2=pf2[(pf2['EC_rate']>0)|(pf2['reg_rate']>0)]
fig,ax=plt.subplots(figsize=(11,8))
for tier,(color,marker,size,alpha) in [('ig',(C_SAFE,'o',55,0.65)),
                                         ('grey',(C_GREY,'s',55,0.75)),
                                         ('dist',(C_DIST,'^',70,0.80))]:
    sub=pf2[pf2['tier']==tier]
    if len(sub)==0: continue
    ax.scatter(sub['reg_rate'],sub['EC_rate'],c=color,s=size,alpha=alpha,
               marker=marker,label=f'{tier.upper()} (n={len(sub)})',
               edgecolors='white',linewidths=0.6)
max_val=min(max(pf2['EC_rate'].max(),pf2['reg_rate'].max())*1.15,50)
xl=np.linspace(0,max_val,100)
ax.plot(xl,xl,'k--',lw=1.5,alpha=0.5,label='EC = Reg Capital')
ax.fill_between(xl,xl,max_val,alpha=0.06,color='#C0392B')
ax.fill_between(xl,0,xl,alpha=0.06,color='#2D6A27')
ax.text(max_val*0.75,max_val*0.92,'EC > Reg Capital\n(undercapitalized)',
        fontsize=8,color=C_DIST,ha='center',style='italic')
ax.text(max_val*0.22,max_val*0.08,'EC < Reg Capital\n(overcapitalized)',
        fontsize=8,color=C_SAFE,ha='center',style='italic')
for _,r in pf2.nlargest(8,'EC_rate').iterrows():
    if r['EC_rate']>0.5:
        ax.annotate(r['ticker'],xy=(r['reg_rate'],r['EC_rate']),
                    xytext=(6,4),textcoords='offset points',
                    fontsize=7.5,color='#444',fontweight='bold')
ax.set_xlabel('Regulatory Capital Rate (% of EAD)',fontsize=9)
ax.set_ylabel('Economic Capital Rate (% of EAD)',fontsize=9)
ax.set_xlim(0,max_val); ax.set_ylim(0,max_val)
ax.set_title(f'Economic Capital vs Regulatory Capital — {latest_q}  |  Phase 4',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'Reg capital=8%×RWA (Basel III standardised)  ·  EC=AIRB (99.9% VaR)',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'basel_ec_vs_reg.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ EC vs Reg Capital saved')


# ── VIZ 7: Monte Carlo CVA distribution ──────────────────────────────────────
print('\n→ [7/7] Monte Carlo CVA...')
svb_q3=df[(df['ticker']=='SIVB')&(df['quarter']=='2022Q3')]
if len(svb_q3)>0:
    r=svb_q3.iloc[0]
    lam=max(-np.log(1-max(r['PD'],MIN_PD)),MIN_LAM)
    cva_paths=[]
    for p in range(N_PATHS):
        df_p=np.exp(-np.cumsum(r_paths[p,1:])*dt)
        dpd=np.diff(1-np.exp(-lam*t_grid),prepend=0)
        cva_paths.append(max(r['LGD']*r['EAD']*np.sum(df_p*dpd),0.0))
    cva_paths=np.array(cva_paths)
    cva_mean=cva_paths.mean()
    cva_p95=np.percentile(cva_paths,95)
    cva_p05=np.percentile(cva_paths,5)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    ax=axes[0]
    ax.hist(cva_paths,bins=50,color=C_DIST,alpha=0.72,edgecolor='white',linewidth=0.4)
    ax.axvline(cva_mean,color='#2C3E50',lw=2.2,linestyle='-',
               label=f'Mean  ${cva_mean:,.0f}M')
    ax.axvline(cva_p95,color='#7B0000',lw=1.8,linestyle='--',
               label=f'95th  ${cva_p95:,.0f}M')
    ax.axvline(cva_p05,color=C_SAFE,lw=1.8,linestyle='--',
               label=f'5th   ${cva_p05:,.0f}M')
    ax.set_xlabel('CVA ($M)',fontsize=9)
    ax.set_ylabel('Frequency',fontsize=9)
    ax.set_title(f'CVA Distribution — SVB Q3 2022\n'
                 f'({N_PATHS:,} paths  ·  σ=${cva_paths.std():,.0f}M)',
                 fontsize=10,fontweight='500',color=C_ACC)
    ax.legend(fontsize=8,framealpha=0.9,edgecolor='#ddd')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax2=axes[1]
    t_full=np.linspace(0,T_cva,N_STEPS+1)
    for p in np.random.choice(N_PATHS,size=150,replace=False):
        ax2.plot(t_full,r_paths[p]*100,color='#2471A3',alpha=0.06,lw=0.8)
    ax2.plot(t_full,r_paths.mean(axis=0)*100,color='#2C3E50',lw=2.5,label='Mean path')
    ax2.axhline(theta*100,color=C_GREY,lw=1.5,linestyle='--',
                label=f'Long-run mean θ={theta*100:.1f}%')
    ax2.fill_between(t_full,
                     np.percentile(r_paths,5,axis=0)*100,
                     np.percentile(r_paths,95,axis=0)*100,
                     alpha=0.15,color='#2471A3',label='5th–95th pct')
    ax2.set_xlabel('Time (years)',fontsize=9)
    ax2.set_ylabel('Interest Rate (%)',fontsize=9)
    ax2.set_title(f'Vasicek Rate Paths  ·  σ={sigma_r}\n'
                  f'κ={kappa}  θ={theta}  r₀={r0:.3f}',
                  fontsize=10,fontweight='500',color=C_ACC)
    ax2.legend(fontsize=8,framealpha=0.9,edgecolor='#ddd')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.suptitle('Phase 4  ·  Monte Carlo CVA — SVB Q3 2022',
                 fontsize=12,fontweight='500',color=C_ACC,y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'basel_mc_cva.png'),dpi=160,bbox_inches='tight')
    plt.show()
    print(f'  ✓ MC CVA saved  '
          f'Mean=${cva_mean:,.0f}M  P5=${cva_p05:,.0f}M  '
          f'P95=${cva_p95:,.0f}M  Spread=${cva_p95-cva_p05:,.0f}M')


# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'PHASE 4 COMPLETE — BASEL III PD/LGD/EAD/CVA')
print(f'{"="*60}')
pf=df[df['quarter']==latest_q]
print(f'  Companies : {df["ticker"].nunique()}  |  Obs: {len(df):,}')
print(f'  EAD       : ${pf["EAD"].sum()/1e3:,.1f}B')
print(f'  EL        : ${pf["EL"].sum()/1e3:,.3f}B  ({pf["EL"].sum()/pf["EAD"].sum()*100:.3f}%)')
print(f'  EC        : ${pf["EC"].sum()/1e3:,.2f}B')
print(f'  RWA       : ${pf["RWA"].sum()/1e3:,.1f}B')
print(f'  Reg Cap   : ${pf["reg_capital"].sum()/1e3:,.2f}B')
print(f'  CVA       : ${pf["CVA"].sum()/1e3:,.3f}B')
for f in ['basel_results.csv','basel_summary.csv','basel_el_heatmap.html',
          'basel_el_ul_ec.png','basel_el_over_time.png','basel_cva_timeline.png',
          'basel_pd_lgd_el.png','basel_ec_vs_reg.png','basel_mc_cva.png']:
    path=os.path.join(OUTPUT_DIR,f)
    if os.path.exists(path):
        print(f'  ✓ {f:<45} {os.path.getsize(path)/1024:6.1f} KB')
print(f'\n  Basel III: EL ✓  UL ✓  EC ✓  CVA ✓  RWA ✓  EL-over-time ✓')
print(f'  → Ready for Phase 5: Kaplan-Meier + SR 11-7 Validation')
print(f'{"="*60}')


# ============================================================
# PHASE 5 — FIX: test_lr index + complete run
# ============================================================

import os, warnings
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy               import stats
from scipy.stats         import norm
from sklearn.metrics     import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from google.colab        import drive

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':0.2,'grid.linestyle':'--',
})

drive.mount('/content/drive', force_remount=False)
BASE_DIR   = '/content/drive/MyDrive/P4_CreditRisk'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

merton_df   = pd.read_csv(os.path.join(OUTPUT_DIR, 'merton_results.csv'))
zscore_df   = pd.read_csv(os.path.join(OUTPUT_DIR, 'zscore_quarterly.csv'))
universe_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'universe.csv'))
params_df   = pd.read_csv(os.path.join(OUTPUT_DIR, 'parameter_registry.csv'))

PARAMS     = dict(zip(params_df['parameter'], params_df['value']))
sector_map = universe_df.set_index('ticker')['sector'].to_dict()
tier_map   = universe_df.set_index('ticker')['tier'].to_dict()

C_SAFE='#2D6A27'; C_GREY='#B07D10'; C_DIST='#C0392B'; C_ACC='#2C3E50'
C_BLUE='#2471A3'; C_PURP='#7D3C98'

DEFAULT_DATES = {
    'CHK':'2020-06-28','HTZ':'2020-05-22','BBBY':'2023-04-23',
    'SIVB':'2023-03-10','SBNY':'2023-03-12','FRC':'2023-05-01',
    'CBL':'2020-11-01','WPG':'2021-06-13',
}
DEFAULT_TICKERS = list(DEFAULT_DATES.keys())

def q2ts(q):
    try:
        yr,qn=int(q[:4]),int(q[5])
        return pd.Timestamp(year=yr,month=(qn-1)*3+1,day=1)
    except: return pd.NaT

def is_post(ticker, quarter):
    if ticker not in DEFAULT_DATES: return False
    return q2ts(quarter) > pd.Timestamp(DEFAULT_DATES[ticker])

merton_df['ts']          = merton_df['quarter'].apply(q2ts)
merton_df['post_default']= merton_df.apply(
    lambda r: is_post(r['ticker'],r['quarter']),axis=1)
merton_df['defaulted']   = merton_df['ticker'].isin(DEFAULT_TICKERS).astype(int)

print('✓ Data loaded')

# ── BLOCK 1: Labeled dataset ──────────────────────────────────────────────────
merged = merton_df[['ticker','quarter','ts','PD','D2D','post_default',
                     'defaulted']].merge(
    zscore_df[['ticker','quarter','Z_score','zone',
               'X1','X2','X3','X4','X5']],
    on=['ticker','quarter'], how='inner')

def build_default_label(df_in):
    df_in = df_in.sort_values(['ticker','quarter']).copy()
    records = []
    for ticker in df_in['ticker'].unique():
        sub = df_in[df_in['ticker']==ticker].copy()
        default_date = pd.Timestamp(DEFAULT_DATES[ticker]) \
                       if ticker in DEFAULT_DATES else None
        for i, row in sub.iterrows():
            obs_ts = row['ts']
            if pd.isna(obs_ts): continue
            if default_date is not None:
                horizon = obs_ts + pd.DateOffset(months=12)
                label = 1 if (obs_ts < default_date <= horizon) else 0
            else:
                label = 0
            records.append({**row.to_dict(), 'default_label': label})
    return pd.DataFrame(records)

labeled = build_default_label(merged)
labeled = labeled[labeled['Z_score'].notna() & labeled['PD'].notna()]
labeled = labeled[~labeled['post_default']].reset_index(drop=True)

TRAIN_END  = '2020Q4'
TEST_START = '2021Q1'
train = labeled[labeled['quarter'] <= TRAIN_END].reset_index(drop=True)
test  = labeled[labeled['quarter'] >= TEST_START].reset_index(drop=True)

print(f'  Train: {len(train):,}  defaults={train["default_label"].sum()}')
print(f'  Test : {len(test):,}   defaults={test["default_label"].sum()}')

# ── BLOCK 2: Kaplan-Meier ─────────────────────────────────────────────────────
km_data = []
for ticker in merton_df['ticker'].unique():
    sub = merton_df[merton_df['ticker']==ticker].sort_values('quarter')
    if len(sub)==0: continue
    obs_start = sub['ts'].min(); obs_end = sub['ts'].max()
    if ticker in DEFAULT_DATES:
        default_ts = pd.Timestamp(DEFAULT_DATES[ticker])
        time_to_event = max(round((default_ts-obs_start).days/91.25),1)
        event = 1
    else:
        time_to_event = max(round((obs_end-obs_start).days/91.25),1)
        event = 0
    km_data.append({'ticker':ticker,'time':time_to_event,'event':event,
                    'tier':tier_map.get(ticker,'ig'),
                    'sector':sector_map.get(ticker,'')})

km_df = pd.DataFrame(km_data)

def kaplan_meier(times, events):
    times=np.array(times); events=np.array(events)
    unique_t=np.sort(np.unique(times[events==1]))
    S=1.0; curve=[(0,1.0)]; n=len(times)
    for t in unique_t:
        at_risk=(times>=t).sum()
        d=((times==t)&(events==1)).sum()
        if at_risk>0: S*=(1-d/at_risk)
        curve.append((t,S))
    return zip(*curve)

km_tiers={}
for tier in ['ig','grey','dist','def']:
    sub=km_df[km_df['tier']==tier]
    if len(sub)>=2:
        t_km,s_km=kaplan_meier(sub['time'].values,sub['event'].values)
        km_tiers[tier]=(list(t_km),list(s_km),len(sub),sub['event'].sum())

print(f'\n  KM: {km_df["event"].sum()} events, {(km_df["event"]==0).sum()} censored')

# ── BLOCK 3: Logistic Regression ─────────────────────────────────────────────
FEATURES = ['X1','X2','X3','X4','X5','Z_score']

# KEY FIX: keep ticker+quarter in train_lr and test_lr
train_lr = train[['ticker','quarter'] + FEATURES + ['default_label']].dropna().reset_index(drop=True)
test_lr  = test[['ticker','quarter']  + FEATURES + ['default_label']].dropna().reset_index(drop=True)

X_tr = train_lr[FEATURES].values; y_tr = train_lr['default_label'].values
X_te = test_lr[FEATURES].values;  y_te = test_lr['default_label'].values

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = LogisticRegression(C=0.1,class_weight='balanced',max_iter=1000,random_state=42)
lr.fit(X_tr_sc, y_tr)

train_lr['LR_PD'] = lr.predict_proba(X_tr_sc)[:,1]
test_lr['LR_PD']  = lr.predict_proba(X_te_sc)[:,1]

print(f'  LR trained: {len(train_lr):,} obs')
print(f'  Coefficients:')
for f,c in zip(FEATURES,lr.coef_[0]):
    print(f'    {f:<12} {c:>+.4f}')

# ── BLOCK 4: Gini / AUC / KS ─────────────────────────────────────────────────
print('\n  SR 11-7 Validation (OOS 2021-2023):')

# Merge all models — ticker+quarter now available in test_lr
test_merged = test.merge(
    test_lr[['ticker','quarter','LR_PD']],
    on=['ticker','quarter'], how='left'
).dropna(subset=['PD','Z_score','default_label'])

y_val = test_merged['default_label'].values
print(f'  Test obs: {len(test_merged):,}  defaults: {y_val.sum()}')

GINI_GOOD=float(PARAMS['gini_good']); GINI_MOD=float(PARAMS['gini_moderate'])
KS_GOOD  =float(PARAMS['ks_good']);   KS_MOD  =float(PARAMS['ks_moderate'])

def gini_auc_ks(y_true, scores, name, flip=False):
    if flip: scores=-scores
    if len(np.unique(y_true))<2:
        print(f'  ⚠ {name}: only one class in test set')
        return {'model':name,'auc':np.nan,'gini':np.nan,'ks':np.nan,
                'fpr':np.array([0,1]),'tpr':np.array([0,1])}
    fpr,tpr,_=roc_curve(y_true,scores)
    auc =roc_auc_score(y_true,scores)
    gini=2*auc-1; ks=np.max(tpr-fpr)
    g_r='GOOD' if gini>=GINI_GOOD else 'MODERATE' if gini>=GINI_MOD else 'POOR'
    k_r='GOOD' if ks>=KS_GOOD     else 'MODERATE' if ks>=KS_MOD     else 'POOR'
    print(f'  {name:<35} AUC={auc:.4f}  '
          f'Gini={gini:.4f}[{g_r}]  KS={ks:.4f}[{k_r}]')
    return {'model':name,'auc':auc,'gini':gini,'ks':ks,'fpr':fpr,'tpr':tpr}

results_val={}
results_val['Merton']=gini_auc_ks(y_val,test_merged['PD'].values,
                                   'CHAMPION: Merton PD')
results_val['ZScore']=gini_auc_ks(y_val,test_merged['Z_score'].values,
                                   'CHALLENGER 1: Altman Z-Score',flip=True)
if test_merged['LR_PD'].notna().sum()>10:
    results_val['LogReg']=gini_auc_ks(y_val,test_merged['LR_PD'].fillna(0).values,
                                       'CHALLENGER 2: Logistic Reg')
results_val['D2D']=gini_auc_ks(y_val,test_merged['D2D'].values,
                                'DIAGNOSTIC: Merton D2D',flip=True)

# ── BLOCK 5: PSI ─────────────────────────────────────────────────────────────
PSI_BINS=int(PARAMS['psi_bins']); PSI_STABLE=float(PARAMS['psi_stable'])
PSI_MONITOR=float(PARAMS['psi_monitor']); PSI_EPS=float(PARAMS['psi_epsilon'])

def compute_psi(dev,mon,n=10,eps=1e-4):
    bins=np.percentile(dev,np.linspace(0,100,n+1))
    bins[0]-=1e-6; bins[-1]+=1e-6
    dp=np.histogram(dev,bins=bins)[0]; mp=np.histogram(mon,bins=bins)[0]
    dp=dp/dp.sum()+eps; mp=mp/mp.sum()+eps
    return np.sum((mp-dp)*np.log(mp/dp)),bins,dp,mp

def psi_rating(p):
    return 'STABLE' if p<PSI_STABLE else 'MONITOR' if p<PSI_MONITOR else 'UNSTABLE'

dev_df=merton_df[(merton_df['quarter']>='2018Q1')&
                  (merton_df['quarter']<='2021Q4')&~merton_df['post_default']]
mon_df=merton_df[(merton_df['quarter']>='2022Q1')&
                  (merton_df['quarter']<='2023Q4')&~merton_df['post_default']]

psi_m,bins_m,dp_m,mp_m=compute_psi(dev_df['PD'].dropna().values,
                                     mon_df['PD'].dropna().values,PSI_BINS,PSI_EPS)
z_dev=zscore_df[(zscore_df['quarter']>='2018Q1')&
                 (zscore_df['quarter']<='2021Q4')]['Z_score'].dropna().values
z_mon=zscore_df[(zscore_df['quarter']>='2022Q1')&
                 (zscore_df['quarter']<='2023Q4')]['Z_score'].dropna().values
psi_z,bins_z,dp_z,mp_z=compute_psi(np.clip(z_dev,-1,7),np.clip(z_mon,-1,7),
                                     PSI_BINS,PSI_EPS)

print(f'\n  PSI — Merton={psi_m:.4f}[{psi_rating(psi_m)}]  '
      f'Z-Score={psi_z:.4f}[{psi_rating(psi_z)}]')

# Save
val_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ['fpr','tpr']}
                      for r in results_val.values()])
val_df.to_csv(os.path.join(OUTPUT_DIR,'sr117_validation.csv'),index=False)
km_df.to_csv(os.path.join(OUTPUT_DIR,'km_survival_data.csv'),index=False)
pd.DataFrame([{'model':'Merton_PD','psi':psi_m,'rating':psi_rating(psi_m)},
              {'model':'Altman_Z', 'psi':psi_z,'rating':psi_rating(psi_z)}]
             ).to_csv(os.path.join(OUTPUT_DIR,'psi_results.csv'),index=False)
print(f'✓ All outputs saved')

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── VIZ 1: Kaplan-Meier ───────────────────────────────────────────────────────
print('\n→ [1/6] Kaplan-Meier...')
tier_km_cfg={
    'ig'  :(C_SAFE,'Investment Grade','-',2.2),
    'grey':(C_GREY,'Grey Zone','--',2.0),
    'dist':(C_DIST,'Distressed','-.', 2.0),
    'def' :('#7B0000','Default/Failed',':',2.2),
}
fig,(ax,ax2)=plt.subplots(1,2,figsize=(14,6))
for tier,(color,label,ls,lw) in tier_km_cfg.items():
    if tier not in km_tiers: continue
    ts_km,sv_km,n,d=km_tiers[tier]
    ts_s=[0]+[t for t in ts_km for _ in (0,1)][:-1]
    sv_s=[s for s in sv_km for _ in (0,1)]
    ax.plot(ts_s,sv_s,color=color,lw=lw,linestyle=ls,
            label=f'{label} (n={n}, events={d})',alpha=0.88)
    if tier!='def':
        ax.scatter([list(ts_km)[-1]],[list(sv_km)[-1]],
                   marker='+',color=color,s=80,zorder=5,alpha=0.7)
ax.axhline(0.5,color='gray',lw=0.8,linestyle=':',alpha=0.6)
ax.text(1,0.51,'50% survival',fontsize=7.5,color='gray')
ax.set_xlabel('Time (quarters)',fontsize=9)
ax.set_ylabel('Survival Probability S(t)',fontsize=9)
ax.set_xlim(-0.5,38); ax.set_ylim(-0.02,1.05)
ax.set_title('Kaplan-Meier Survival Curves by Risk Tier  |  Phase 5',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'S(t)=prob of surviving beyond t quarters  ·  +=censored  ·  Step estimator',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='lower left',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

default_km=km_df[km_df['event']==1].sort_values('time')
ax2.barh(np.arange(len(default_km)),default_km['time'],
         color=[tier_km_cfg.get(t,(C_DIST,'','',1))[0] for t in default_km['tier']],
         alpha=0.78,edgecolor='white',linewidth=0.5)
for i,(_,r) in enumerate(default_km.iterrows()):
    ax2.text(r['time']+0.3,i,f'{r["time"]:.0f}Q',va='center',fontsize=8,color='#444')
ax2.set_yticks(np.arange(len(default_km)))
ax2.set_yticklabels(default_km['ticker'],fontsize=9)
ax2.set_xlabel('Quarters to Default',fontsize=9)
ax2.set_title('Time to Default — All 8 Events',fontsize=10,fontweight='500',color=C_ACC)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.grid(axis='x',alpha=0.2,linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'km_survival_curves.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ KM saved')

# ── VIZ 2: ROC curves ────────────────────────────────────────────────────────
print('\n→ [2/6] ROC curves...')
model_roc_cfg={
    'Merton':(C_DIST,'-',2.5,'CHAMPION: Merton PD'),
    'ZScore':(C_BLUE,'--',2.0,'CHALLENGER 1: Altman Z-Score'),
    'LogReg':(C_PURP,'-.',2.0,'CHALLENGER 2: Logistic Reg'),
    'D2D'   :(C_GREY,':',1.5,'DIAGNOSTIC: Merton D2D'),
}
fig,ax=plt.subplots(figsize=(9,8))
ax.plot([0,1],[0,1],'k--',lw=1.0,alpha=0.4,label='Random (AUC=0.50)')
for name,(color,ls,lw,label) in model_roc_cfg.items():
    if name not in results_val: continue
    r=results_val[name]
    if np.isnan(r.get('auc',np.nan)): continue
    auc=r['auc']; gini=r['gini']; ks=r['ks']
    fpr=r['fpr']; tpr=r['tpr']
    ax.plot(fpr,tpr,color=color,lw=lw,linestyle=ls,alpha=0.88,
            label=f'{label}\nAUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}')
    ks_i=np.argmax(tpr-fpr)
    ax.scatter(fpr[ks_i],tpr[ks_i],color=color,s=80,zorder=6,edgecolors='white',lw=0.8)
    ax.plot([fpr[ks_i],fpr[ks_i]],[fpr[ks_i],tpr[ks_i]],
            color=color,lw=1.0,linestyle=':',alpha=0.6)
for y_pos,txt,fc,ec in [
    (0.28,f'Gini≥{GINI_GOOD}: GOOD','#F0F7EC',C_SAFE),
    (0.20,f'Gini≥{GINI_MOD}: MODERATE','#FDF6E3',C_GREY),
    (0.12,f'Gini<{GINI_MOD}: POOR','#FDF0EF',C_DIST),
]:
    ax.text(0.72,y_pos,txt,fontsize=8,color=ec,fontweight='bold',
            bbox=dict(boxstyle='round',fc=fc,ec=ec,alpha=0.8))
ax.set_xlabel('False Positive Rate',fontsize=9)
ax.set_ylabel('True Positive Rate',fontsize=9)
ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
ax.set_title('ROC — SR 11-7 Champion vs Challenger  |  Phase 5\nOOS Test: 2021–2023',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.005,'● = KS point  ·  OOS test period 2021Q1–2023Q4',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='lower right',fontsize=8,framealpha=0.92,edgecolor='#ddd')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'sr117_roc_curves.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ ROC saved')

# ── VIZ 3: Gini/AUC/KS bar ───────────────────────────────────────────────────
print('\n→ [3/6] Gini/AUC/KS bars...')
val_plot=val_df.dropna()
model_labels_s={
    'CHAMPION: Merton PD'       :'CHAMPION\nMerton PD',
    'CHALLENGER 1: Altman Z-Score':'CHALLENGER 1\nZ-Score',
    'CHALLENGER 2: Logistic Reg':'CHALLENGER 2\nLogistic',
    'DIAGNOSTIC: Merton D2D'    :'DIAGNOSTIC\nD2D',
}
model_colors_s={
    'CHAMPION: Merton PD'       :C_DIST,
    'CHALLENGER 1: Altman Z-Score':C_BLUE,
    'CHALLENGER 2: Logistic Reg':C_PURP,
    'DIAGNOSTIC: Merton D2D'    :C_GREY,
}
fig,axes=plt.subplots(1,3,figsize=(14,5.5),sharey=True)
for ax_,(col,title,tg,tm) in zip(axes,[
    ('gini','Gini Coefficient',GINI_GOOD,GINI_MOD),
    ('auc', 'AUC-ROC',0.80,0.70),
    ('ks',  'KS Statistic',KS_GOOD,KS_MOD),
]):
    x=np.arange(len(val_plot))
    bc=[model_colors_s.get(m,C_ACC) for m in val_plot['model']]
    bars=ax_.bar(x,val_plot[col],color=bc,alpha=0.80,edgecolor='white',linewidth=0.5,width=0.6)
    ax_.axhline(tg,color=C_SAFE,lw=1.5,linestyle='--',alpha=0.8,label=f'Good≥{tg}')
    ax_.axhline(tm,color=C_GREY,lw=1.2,linestyle=':',alpha=0.8,label=f'Moderate≥{tm}')
    for bar,v in zip(bars,val_plot[col]):
        if pd.notna(v):
            ax_.text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.3f}',
                     ha='center',fontsize=8,fontweight='bold',color='#333')
    ax_.set_xticks(x)
    ax_.set_xticklabels([model_labels_s.get(m,m) for m in val_plot['model']],fontsize=7.5)
    ax_.set_title(title,fontsize=10,fontweight='500',color=C_ACC)
    ax_.set_ylim(0,1.05)
    ax_.legend(fontsize=7.5,loc='upper right',framealpha=0.9,edgecolor='#ddd')
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
fig.suptitle('SR 11-7 Champion vs Challenger  |  Phase 5\nOOS Performance 2021–2023',
             fontsize=12,fontweight='500',color=C_ACC,y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'sr117_gini_ks.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Gini/KS bars saved')

# ── VIZ 4: PSI ───────────────────────────────────────────────────────────────
print('\n→ [4/6] PSI monitoring...')
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
bin_labels=[f'B{i+1}' for i in range(PSI_BINS)]
for ax_,(pv,dp,mp,ttl,col) in [
    (axes[0],(psi_m,dp_m,mp_m,'Merton PD — Dev vs Mon',C_DIST)),
    (axes[1],(psi_z,dp_z,mp_z,'Altman Z — Dev vs Mon',C_BLUE)),
]:
    psi_v,dev_p,mon_p,title,color=pv,dp,mp,ttl,col
    x=np.arange(PSI_BINS); w=0.35
    ax_.bar(x-w/2,dev_p,w,color=C_ACC,alpha=0.75,label='Dev 2018–2021')
    ax_.bar(x+w/2,mon_p,w,color=color,alpha=0.75,label='Mon 2022–2023')
    rating=psi_rating(psi_v)
    bc=C_SAFE if rating=='STABLE' else C_GREY if rating=='MONITOR' else C_DIST
    ax_.text(0.98,0.96,f'PSI={psi_v:.4f}\n{rating}',
             transform=ax_.transAxes,ha='right',va='top',fontsize=9,
             fontweight='bold',color='white',
             bbox=dict(boxstyle='round,pad=0.4',fc=bc,alpha=0.9))
    ax_.set_xticks(np.arange(PSI_BINS))
    ax_.set_xticklabels(bin_labels,fontsize=8)
    ax_.set_xlabel('Score Bin (decile)',fontsize=9)
    ax_.set_ylabel('Proportion',fontsize=9)
    ax_.set_title(title,fontsize=10,fontweight='500',color=C_ACC)
    ax_.legend(fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
fig.suptitle(f'PSI Model Monitoring  |  Phase 5\n'
             f'Stable<{PSI_STABLE}  Monitor<{PSI_MONITOR}  Unstable≥{PSI_MONITOR}',
             fontsize=12,fontweight='500',color=C_ACC,y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'psi_monitoring.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ PSI saved')

# ── VIZ 5: Calibration ───────────────────────────────────────────────────────
print('\n→ [5/6] Calibration...')
fig,axes=plt.subplots(1,3,figsize=(15,5.5))
calib_models=[
    (test_merged,'PD','CHAMPION: Merton PD',C_DIST,False),
    (test_merged,'Z_score','CHALLENGER 1: Z-Score',C_BLUE,True),
    (test_merged,'LR_PD','CHALLENGER 2: LogReg',C_PURP,False),
]
for ax_,(data,sc,title,color,flip) in zip(axes,calib_models):
    d=data.dropna(subset=[sc,'default_label']).copy()
    if len(d)<20: ax_.text(0.5,0.5,'Insufficient data',ha='center',transform=ax_.transAxes); continue
    scores=d[sc].values
    if flip: scores=-scores
    try:
        d['bin']=pd.qcut(scores,min(10,d[sc].nunique()),labels=False,duplicates='drop')
    except: continue
    calib=(d.groupby('bin').agg(mean_score=(sc,'mean'),
                                 actual_rate=('default_label','mean'),
                                 n=('default_label','count')).reset_index())
    ax_.scatter(calib['mean_score'],calib['actual_rate']*100,
                s=calib['n']*3,color=color,alpha=0.75,
                edgecolors='white',linewidths=0.8,zorder=4)
    if len(calib)>3:
        m,b,r,p,_=stats.linregress(calib['mean_score'],calib['actual_rate']*100)
        xl=np.linspace(calib['mean_score'].min(),calib['mean_score'].max(),100)
        ax_.plot(xl,m*xl+b,color=color,lw=1.5,linestyle='--',alpha=0.7)
        ax_.text(0.05,0.90,f'R={r:.3f}',transform=ax_.transAxes,fontsize=8,color='#555')
    ax_.set_xlabel(sc,fontsize=9); ax_.set_ylabel('Actual Default Rate (%)',fontsize=9)
    ax_.set_title(title,fontsize=9.5,fontweight='500',color=C_ACC)
    ax_.text(0.05,0.80,f'n={len(d):,}',transform=ax_.transAxes,fontsize=8,color='#666')
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
fig.suptitle('Calibration — Score vs Actual Default Rate  |  Phase 5\n'
             'Test 2021–2023  ·  Bubble size = observation count',
             fontsize=11,fontweight='500',color=C_ACC,y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'sr117_calibration.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Calibration saved')

# ── VIZ 6: Default event timeline ────────────────────────────────────────────
print('\n→ [6/6] Default event timeline...')
fig,ax=plt.subplots(figsize=(13,5))
ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2021-01-01'),
           color='#D6EAF8',alpha=0.3,label='COVID period')
ax.axvspan(pd.Timestamp('2022-01-01'),pd.Timestamp('2024-01-01'),
           color='#FADBD8',alpha=0.3,label='Rate hike + bank stress')
default_info={
    'CHK' :('2020-06-28','Energy',   0.30),
    'CBL' :('2020-11-01','Real Estate',0.50),
    'HTZ' :('2020-05-22','Industrial',0.70),
    'WPG' :('2021-06-13','Real Estate',0.20),
    'SIVB':('2023-03-10','Banks',    0.80),
    'SBNY':('2023-03-12','Banks',    0.60),
    'FRC' :('2023-05-01','Banks',    0.40),
    'BBBY':('2023-04-23','Retail',   0.15),
}
sec_col={'Energy':'#3B6D11','Real Estate':'#D85A30','Industrial':'#888780',
         'Banks':'#854F0B','Retail':'#D4537E'}
for ticker,(date_str,sector,y_pos) in default_info.items():
    ts=pd.Timestamp(date_str)
    color=sec_col.get(sector,C_DIST)
    ax.scatter(ts,y_pos,s=200,color=color,zorder=6,edgecolors='white',linewidths=1.5)
    ax.axvline(ts,color=color,lw=1.0,linestyle=':',alpha=0.5)
    ax.text(ts+pd.Timedelta(days=8),y_pos+0.03,ticker,
            fontsize=8.5,fontweight='bold',color=color)
    ax.text(ts+pd.Timedelta(days=8),y_pos-0.05,sector,
            fontsize=7,color=color,style='italic')
    q=f'{pd.Timestamp(date_str).year}Q{(pd.Timestamp(date_str).month-1)//3+1}'
    sub=merton_df[(merton_df['ticker']==ticker)&(merton_df['quarter']==q)]
    if len(sub)>0:
        pd_val=sub['PD'].values[0]*100
        ax.text(ts-pd.Timedelta(days=5),y_pos-0.09,
                f'PD={pd_val:.1f}%',fontsize=7,color=color,ha='right')
ax.set_xlim(pd.Timestamp('2019-01-01'),pd.Timestamp('2024-01-01'))
ax.set_ylim(0,1.0); ax.set_yticks([]); ax.set_xlabel('Date',fontsize=9)
ax.set_title('Default Event Timeline — 8 Events  |  Phase 5',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'Merton PD at default quarter  ·  Two clusters: COVID 2020 + Banking crisis 2023',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
handles=[mpatches.Patch(fc=c,label=s,alpha=0.8) for s,c in sec_col.items()]
handles+=[mpatches.Patch(fc='#D6EAF8',label='COVID period',alpha=0.5),
          mpatches.Patch(fc='#FADBD8',label='Rate hike stress',alpha=0.5)]
ax.legend(handles=handles,loc='upper left',fontsize=8,
          framealpha=0.9,edgecolor='#ddd',ncol=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'default_event_timeline.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Timeline saved')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'PHASE 5 COMPLETE — SR 11-7 CHAMPION-CHALLENGER')
print(f'{"="*60}')
print(f'\n  SR 11-7 Rankings (OOS 2021-2023):')
for _,row in val_df.iterrows():
    g=row['gini']; k=row['ks']
    if pd.isna(g): continue
    g_r='GOOD' if g>=GINI_GOOD else 'MODERATE' if g>=GINI_MOD else 'POOR'
    k_r='GOOD' if k>=KS_GOOD   else 'MODERATE' if k>=KS_MOD   else 'POOR'
    print(f'  {row["model"]:<38} Gini={g:.4f}[{g_r}]  KS={k:.4f}[{k_r}]')
print(f'\n  PSI: Merton={psi_m:.4f}[{psi_rating(psi_m)}]  '
      f'Z-Score={psi_z:.4f}[{psi_rating(psi_z)}]')
print(f'\n  KM: 8 default events  ·  def tier median survival=26Q')
for f in ['km_survival_curves.png','sr117_roc_curves.png','sr117_gini_ks.png',
          'psi_monitoring.png','sr117_calibration.png','default_event_timeline.png']:
    path=os.path.join(OUTPUT_DIR,f)
    if os.path.exists(path):
        print(f'  ✓ {f:<45} {os.path.getsize(path)/1024:6.1f} KB')
print(f'\n  Champion: Merton PD (structural model)')
print(f'  → Ready for Phase 6: Portfolio Application')
print(f'{"="*60}')


# ============================================================
# PHASE 6 — FIX: Remove fund name from all charts
# ============================================================

import os, warnings
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from google.colab import drive

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.facecolor':'white',
    'axes.facecolor':'white','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':0.2,'grid.linestyle':'--',
})

drive.mount('/content/drive', force_remount=False)
BASE_DIR   = '/content/drive/MyDrive/P4_CreditRisk'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

scorecard  = pd.read_csv(os.path.join(OUTPUT_DIR, 'credit_scorecard.csv'))
port_q     = pd.read_csv(os.path.join(OUTPUT_DIR, 'portfolio_risk_quarterly.csv'))
universe_df= pd.read_csv(os.path.join(OUTPUT_DIR, 'universe.csv'))

sector_map = universe_df.set_index('ticker')['sector'].to_dict()
tier_map   = universe_df.set_index('ticker')['tier'].to_dict()

C_SAFE='#2D6A27'; C_GREY='#B07D10'; C_DIST='#C0392B'; C_ACC='#2C3E50'

def q2ts(q):
    try:
        yr,qn=int(q[:4]),int(q[5])
        return pd.Timestamp(year=yr,month=(qn-1)*3+1,day=1)
    except: return pd.NaT

scorecard['ts']  = scorecard['quarter'].apply(q2ts)
port_q['ts']     = port_q['quarter'].apply(q2ts)
port_q           = port_q.sort_values('ts')
latest_q         = scorecard['quarter'].max()
lq               = scorecard[scorecard['quarter']==latest_q]
lp               = port_q[port_q['quarter']==latest_q].iloc[0]

alert_colors={'CRITICAL':'#7B0000','HIGH':C_DIST,'MEDIUM':C_GREY,
              'LOW':'#88C870','CLEAR':C_SAFE}

sec_colors_p={
    'Banks':'#854F0B','Energy':'#3B6D11','Retail':'#D4537E',
    'Telecom':'#1D9E75','Tech':'#185FA5','Healthcare':'#7F77DD',
    'Industrial':'#888780','Real Estate':'#D85A30',
}

print('✓ Data loaded — rebuilding charts without fund name...')

# ── VIZ 1: Scorecard heatmap ──────────────────────────────────────────────────
print('\n→ [1/7] Scorecard heatmap...')

pivot_sc=scorecard.pivot_table(
    index='ticker',columns='quarter',values='composite_score',aggfunc='mean')
q_cols=sorted([c for c in pivot_sc.columns if '2015Q1'<=c<='2023Q4'])
pivot_sc=pivot_sc[q_cols]
sdf=pd.DataFrame({'sector':pd.Series(sector_map),'v':pivot_sc.mean(axis=1)})
sdf=sdf[sdf.index.isin(pivot_sc.index)].sort_values(['sector','v'],ascending=[True,False])
pivot_sc=pivot_sc.reindex(sdf.index)

hover_sc=[]
for t in pivot_sc.index:
    rh=[]
    for q in q_cols:
        val=pivot_sc.loc[t,q]
        if pd.notna(val):
            sub=scorecard[(scorecard['ticker']==t)&(scorecard['quarter']==q)]
            if len(sub)>0:
                r=sub.iloc[0]
                pd_str=f'{r["PD"]*100:.2f}%' if pd.notna(r.get("PD")) else 'N/A'
                d2d_str=f'{r["D2D"]:.3f}'    if pd.notna(r.get("D2D")) else 'N/A'
                z_str=f'{r["Z_score"]:.2f}'  if pd.notna(r.get("Z_score")) else 'N/A'
                rh.append(f'<b>{t}</b>  ·  {q}<br>'
                          f'Score: <b>{val:.1f}</b>  [{r.get("traffic_light","N/A")}]<br>'
                          f'PD={pd_str}  D2D={d2d_str}<br>'
                          f'Rating={r.get("internal_rating","N/A")}  Z={z_str}<br>'
                          f'EWS: {r.get("ews_flags","N/A")}<br>'
                          f'{sector_map.get(t,"")}  ·  {tier_map.get(t,"")}')
            else: rh.append(f'<b>{t}</b>  ·  {q}<br>Score={val:.1f}')
        else: rh.append(f'<b>{t}</b>  ·  {q}<br><i>No data</i>')
    hover_sc.append(rh)

shapes_sc,cur_sec=[],None
for i,t in enumerate(pivot_sc.index):
    s=sector_map.get(t,'')
    if s!=cur_sec and i>0:
        shapes_sc.append(dict(type='line',x0=-0.5,x1=len(q_cols)-0.5,
                               y0=i-0.5,y1=i-0.5,line=dict(color='white',width=3)))
    cur_sec=s

anns_sc,cur_sec,ss=[],None,0
tlist=list(pivot_sc.index)
for i,t in enumerate(tlist):
    s=sector_map.get(t,'')
    if s!=cur_sec:
        if cur_sec:
            anns_sc.append(dict(x=len(q_cols)+0.8,y=(ss+i-1)/2,
                                text=f'<b>{cur_sec}</b>',showarrow=False,
                                xref='x',yref='y',font=dict(size=9,color='#333'),
                                xanchor='left'))
        cur_sec,ss=s,i
if cur_sec:
    anns_sc.append(dict(x=len(q_cols)+0.8,y=(ss+len(tlist)-1)/2,
                        text=f'<b>{cur_sec}</b>',showarrow=False,
                        xref='x',yref='y',font=dict(size=9,color='#333'),
                        xanchor='left'))

fig_sc=go.Figure(go.Heatmap(
    z=pivot_sc.values,x=q_cols,y=pivot_sc.index.tolist(),
    text=hover_sc,hovertemplate='%{text}<extra></extra>',
    colorscale=[
        [0.00,'#1A6B2A'],[0.15,'#2D6A27'],[0.25,'#88C870'],
        [0.40,'#E8B820'],[0.55,'#E8856B'],[0.70,'#C0392B'],
        [0.85,'#8B0000'],[1.00,'#4A0000'],
    ],
    zmin=0,zmax=100,
    colorbar=dict(
        title=dict(text='Risk Score',side='right',font=dict(size=10)),
        thickness=14,len=0.75,x=1.13,
        tickvals=[0,25,50,75,100],
        ticktext=['0<br><i>GREEN</i>','25<br><i>AMBER</i>',
                  '50<br><i>RED</i>','75<br><i>CRITICAL</i>','100'],
        tickfont=dict(size=8)),
))
fig_sc.update_layout(
    title=dict(
        text=('Credit Risk Scorecard  ·  All Companies × Quarters  |  Phase 6<br>'
              '<sup>'
              '<span style="color:#2D6A27">■</span> GREEN (0-25)  ·  '
              '<span style="color:#E8B820">■</span> AMBER (25-50)  ·  '
              '<span style="color:#C0392B">■</span> RED (50-75)  ·  '
              '<span style="color:#4A0000">■</span> CRITICAL (75-100)  ·  '
              'Hover for full detail</sup>'),
        font=dict(size=13,color=C_ACC),x=0.5,xanchor='center'),
    xaxis=dict(tickangle=-50,tickfont=dict(size=7.5),showgrid=False,domain=[0,0.87]),
    yaxis=dict(tickfont=dict(size=8.5),autorange='reversed',showgrid=False),
    height=max(820,len(pivot_sc.index)*15),
    margin=dict(l=85,r=180,t=80,b=70),
    shapes=shapes_sc,annotations=anns_sc,
    plot_bgcolor='white',paper_bgcolor='white')
fig_sc.show()
fig_sc.write_html(os.path.join(OUTPUT_DIR,'p6_scorecard_heatmap.html'))
print('  ✓ Scorecard heatmap saved')


# ── VIZ 2: Portfolio risk over time ──────────────────────────────────────────
print('\n→ [2/7] Portfolio risk over time...')

fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
fig.suptitle('Portfolio Credit Risk Over Time  |  Phase 6',
             fontsize=12,fontweight='500',color=C_ACC,y=1.01)

ts_vec=port_q['ts'].values

ax=axes[0,0]
ax.fill_between(ts_vec,port_q['mean_PD']*100,alpha=0.20,color=C_DIST)
ax.plot(ts_vec,port_q['mean_PD']*100,color=C_DIST,lw=2.0,label='Mean PD')
ax.plot(ts_vec,port_q['median_PD']*100,color='#2471A3',lw=1.5,
        linestyle='--',label='Median PD')
ax.set_ylabel('PD (%)',fontsize=9)
ax.set_title('Portfolio Mean & Median PD',fontsize=10,fontweight='500',color=C_ACC)
ax.legend(fontsize=8,framealpha=0.9)

ax=axes[0,1]
ax.fill_between(ts_vec,port_q['mean_D2D'],alpha=0.15,color='#2471A3')
ax.plot(ts_vec,port_q['mean_D2D'],color='#2471A3',lw=2.0)
ax.axhline(1.5,color=C_DIST,lw=1.2,linestyle='--',alpha=0.7,label='D2D=1.5 (warning)')
ax.axhline(0,  color='#7B0000',lw=1.0,linestyle=':',alpha=0.7,label='D2D=0 (default)')
ax.set_ylabel('Distance to Default',fontsize=9)
ax.set_title('Portfolio Mean D2D',fontsize=10,fontweight='500',color=C_ACC)
ax.legend(fontsize=8,framealpha=0.9)

ax=axes[1,0]
ax.fill_between(ts_vec,port_q['EL_rate'],alpha=0.20,color=C_DIST)
ax.plot(ts_vec,port_q['EL_rate'],color=C_DIST,lw=2.0,label='EL Rate (%)')
ax2r=ax.twinx()
ax2r.bar(ts_vec,port_q['n_distressed'],width=60,alpha=0.35,
         color='#7B0000',label='# Distressed')
ax2r.set_ylabel('# Distressed',fontsize=8,color='#7B0000')
ax2r.tick_params(axis='y',labelcolor='#7B0000')
ax.set_ylabel('EL Rate (% of EAD)',fontsize=9)
ax.set_title('EL Rate + Distressed Count',fontsize=10,fontweight='500',color=C_ACC)
lines1,lbl1=ax.get_legend_handles_labels()
lines2,lbl2=ax2r.get_legend_handles_labels()
ax.legend(lines1+lines2,lbl1+lbl2,fontsize=8,framealpha=0.9)

ax=axes[1,1]
ax.stackplot(ts_vec,
             port_q['n_green'],port_q['n_amber'],
             port_q['n_red'],port_q['n_critical'],
             labels=['GREEN','AMBER','RED','CRITICAL'],
             colors=[C_SAFE,C_GREY,C_DIST,'#7B0000'],alpha=0.75)
ax.set_ylabel('Number of companies',fontsize=9)
ax.set_title('Traffic Light Distribution',fontsize=10,fontweight='500',color=C_ACC)
ax.legend(fontsize=8,framealpha=0.9,loc='upper left')

for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'SVB','#A32D2D')]:
    for ax_ in axes.flat:
        ax_.axvline(ts,color=color,lw=1.0,linestyle=':',alpha=0.7)

for ax_ in axes.flat:
    ax_.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'p6_portfolio_risk_time.png'),
            dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Portfolio risk over time saved')


# ── VIZ 3: Migration matrix ───────────────────────────────────────────────────
print('\n→ [3/7] Migration matrix...')
mig_df=pd.read_csv(os.path.join(OUTPUT_DIR,'credit_migration.csv'))
import pandas as pd
RATING_ORDER=['AAA','AA','A','BBB','BB','B','CCC','D','NR']
ratings_used=[r for r in RATING_ORDER if r in mig_df['from'].values
              or r in mig_df['to'].values]
ratings_used=[r for r in RATING_ORDER if r in ratings_used and r!='NR']
tm=pd.DataFrame(0.0,index=ratings_used,columns=ratings_used)
for fr in ratings_used:
    sub=mig_df[mig_df['from']==fr]; total=len(sub)
    if total==0: continue
    for tr in ratings_used:
        tm.loc[fr,tr]=(sub['to']==tr).sum()/total*100

fig,ax=plt.subplots(figsize=(9,7))
im=ax.imshow(tm.values,cmap='RdYlGn_r',vmin=0,vmax=100,aspect='auto')
plt.colorbar(im,ax=ax,label='Transition Probability (%)',shrink=0.8)
for i in range(len(ratings_used)):
    for j in range(len(ratings_used)):
        v=tm.values[i,j]
        if v>0:
            ax.text(j,i,f'{v:.1f}%',ha='center',va='center',
                    fontsize=8.5,color='white' if v>60 else 'black',
                    fontweight='bold')
ax.set_xticks(range(len(ratings_used))); ax.set_yticks(range(len(ratings_used)))
ax.set_xticklabels(ratings_used,fontsize=10); ax.set_yticklabels(ratings_used,fontsize=10)
ax.set_xlabel('Rating at Year End (To)',fontsize=9)
ax.set_ylabel('Rating at Year Start (From)',fontsize=9)
ax.set_title('Credit Migration Matrix — Annual Transitions  |  Phase 6\n'
             'Averaged 2016–2023  ·  Internal ratings from Merton PD',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
for i in range(len(ratings_used)):
    ax.add_patch(plt.Rectangle((i-0.5,i-0.5),1,1,
                                fill=False,edgecolor='#2C3E50',lw=2.5))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'p6_migration_matrix.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Migration matrix saved')


# ── VIZ 4: EWS dashboard ─────────────────────────────────────────────────────
print('\n→ [4/7] EWS dashboard...')
lq_plot=scorecard[scorecard['quarter']==latest_q].sort_values(
    'composite_score',ascending=False).head(30)
fig,axes=plt.subplots(1,2,figsize=(15,8))
ax=axes[0]
bc=[alert_colors.get(a,C_SAFE) for a in lq_plot['ews_alert']]
y=np.arange(len(lq_plot))
bars=ax.barh(y,lq_plot['composite_score'],color=bc,alpha=0.80,
             edgecolor='white',linewidth=0.5)
ax.axvline(25,color=C_GREY,lw=1.2,linestyle='--',alpha=0.7,label='AMBER threshold')
ax.axvline(50,color=C_DIST,lw=1.2,linestyle='--',alpha=0.7,label='RED threshold')
ax.axvline(75,color='#7B0000',lw=1.2,linestyle='--',alpha=0.7,label='CRITICAL threshold')
for bar,score,alert in zip(bars,lq_plot['composite_score'],lq_plot['ews_alert']):
    ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
            f'{score:.1f}  [{alert}]',va='center',fontsize=7.5,color='#333')
ax.set_yticks(y); ax.set_yticklabels(lq_plot['ticker'],fontsize=9)
ax.set_xlabel('Composite Risk Score (0-100)',fontsize=9)
ax.set_title(f'Composite Risk Score — {latest_q}  |  Top 30',
             fontsize=10,fontweight='500',color=C_ACC)
ax.legend(fontsize=8,framealpha=0.9,edgecolor='#ddd'); ax.set_xlim(0,115)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2=axes[1]
for _,r in lq_plot.iterrows():
    color=alert_colors.get(r['ews_alert'],C_SAFE)
    size=max(r['ews_count']*150,50)
    ax2.scatter(r['PD']*100,r['D2D'],s=size,c=color,alpha=0.75,
                edgecolors='white',linewidths=0.8,zorder=4)
    if r['ews_count']>=2:
        ax2.annotate(r['ticker'],xy=(r['PD']*100,r['D2D']),
                     xytext=(5,3),textcoords='offset points',
                     fontsize=7.5,color='#333',fontweight='bold')
ax2.axhline(1.5,color=C_DIST,lw=1.2,linestyle='--',alpha=0.7,label='D2D=1.5 warning')
ax2.axhline(0,  color='#7B0000',lw=1.0,linestyle=':',alpha=0.7,label='D2D=0 default')
ax2.axvline(5,  color=C_DIST,lw=1.2,linestyle='--',alpha=0.7,label='PD=5% threshold')
ax2.set_xlabel('Merton PD (%)',fontsize=9)
ax2.set_ylabel('Distance to Default (D2D)',fontsize=9)
ax2.set_title(f'PD vs D2D Risk Map — {latest_q}\nBubble size = EWS flag count',
              fontsize=10,fontweight='500',color=C_ACC)
legend_elements=[mpatches.Patch(fc='#7B0000',label='CRITICAL'),
                 mpatches.Patch(fc=C_DIST,label='HIGH'),
                 mpatches.Patch(fc=C_GREY,label='MEDIUM'),
                 mpatches.Patch(fc=C_SAFE,label='LOW/CLEAR')]
ax2.legend(handles=legend_elements,fontsize=8,framealpha=0.9,edgecolor='#ddd')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'p6_ews_dashboard.png'),dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ EWS dashboard saved')


# ── VIZ 5: Sector concentration ───────────────────────────────────────────────
print('\n→ [5/7] Sector concentration...')
sector_ts=(scorecard.groupby(['quarter','sector'])
           .agg(mean_PD=('PD','mean'),total_EL=('EL','sum'),
                total_EAD=('EAD','sum'))
           .reset_index())
sector_ts['EL_rate']=sector_ts['total_EL']/sector_ts['total_EAD']*100
sector_ts['ts']=sector_ts['quarter'].apply(q2ts)
sector_ts=sector_ts.sort_values('ts')
sectors_plot=['Banks','Energy','Retail','Telecom','Tech',
              'Healthcare','Industrial','Real Estate']

fig,axes=plt.subplots(2,1,figsize=(13,10),
                      gridspec_kw={'height_ratios':[1.4,1],'hspace':0.3})
ax=axes[0]
for sec in sectors_plot:
    sub=sector_ts[sector_ts['sector']==sec]
    if len(sub)>0:
        ax.plot(sub['ts'],sub['mean_PD']*100,
                color=sec_colors_p.get(sec,'#888'),lw=1.8,label=sec,alpha=0.85)
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'Bank crisis','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.7)
    ax.text(ts+pd.Timedelta(days=25),ax.get_ylim()[1]*0.88 if ax.get_ylim()[1]>0 else 1,
            lbl,fontsize=7.5,color=color,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))
ax.set_ylabel('Mean PD (%)',fontsize=9)
ax.set_title('Sector PD Over Time  |  Phase 6',fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.legend(loc='upper left',fontsize=8,framealpha=0.9,edgecolor='#ddd',ncol=4)
ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2=axes[1]
pivot_el_sec=sector_ts.pivot_table(
    index='ts',columns='sector',values='EL_rate',aggfunc='mean').fillna(0)
secs_avail=[s for s in sectors_plot if s in pivot_el_sec.columns]
ax2.stackplot(pivot_el_sec.index,
              [pivot_el_sec[s] for s in secs_avail],
              labels=secs_avail,
              colors=[sec_colors_p.get(s,'#888') for s in secs_avail],alpha=0.70)
ax2.set_ylabel('EL Rate (% of EAD)',fontsize=9)
ax2.set_title('Portfolio EL Rate by Sector — Stacked',
              fontsize=10,fontweight='500',color=C_ACC)
ax2.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax2.legend(loc='upper left',fontsize=7.5,framealpha=0.9,edgecolor='#ddd',ncol=4)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR,'p6_sector_concentration.png'),
            dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Sector concentration saved')


# ── VIZ 6: Rating distribution ────────────────────────────────────────────────
print('\n→ [6/7] Rating distribution...')
RATING_ORDER=['AAA','AA','A','BBB','BB','B','CCC','D','NR']
rating_ts=(scorecard.groupby(['quarter','internal_rating'])
           .size().unstack(fill_value=0).reset_index())
rating_ts['ts']=rating_ts['quarter'].apply(q2ts)
rating_ts=rating_ts.sort_values('ts')
invest_grade=['AAA','AA','A','BBB']; sub_grade=['BB','B']; distressed=['CCC','D']
for grp in invest_grade+sub_grade+distressed:
    if grp not in rating_ts.columns: rating_ts[grp]=0
rating_ts['IG_count'] =rating_ts[invest_grade].sum(axis=1)
rating_ts['SG_count'] =rating_ts[sub_grade].sum(axis=1)
rating_ts['DIST_count']=rating_ts[distressed].sum(axis=1)

fig,ax=plt.subplots(figsize=(13,5.5))
ax.stackplot(rating_ts['ts'],
             rating_ts['IG_count'],rating_ts['SG_count'],rating_ts['DIST_count'],
             labels=['Investment Grade (AAA-BBB)',
                     'Sub-Investment Grade (BB-B)',
                     'Distressed (CCC-D)'],
             colors=[C_SAFE,C_GREY,C_DIST],alpha=0.75)
for ts,lbl,color in [(pd.Timestamp('2020-03-01'),'COVID','#2471A3'),
                      (pd.Timestamp('2022-03-01'),'Fed hike','#7D6608'),
                      (pd.Timestamp('2023-03-10'),'Bank crisis','#A32D2D')]:
    ax.axvline(ts,color=color,lw=1.2,linestyle=':',alpha=0.8)
    ax.text(ts+pd.Timedelta(days=25),ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 40,
            lbl,fontsize=7.5,color=color,
            bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,alpha=0.7))
ax.set_ylabel('Number of companies',fontsize=9)
ax.set_xlabel('Date',fontsize=9)
ax.set_title('Internal Credit Rating Distribution Over Time  |  Phase 6',
             fontsize=11,fontweight='500',color=C_ACC,pad=12)
ax.text(0.5,1.01,'Internal ratings based on Merton PD  ·  '
        'COVID 2020 and banking crisis 2023 visible as downgrades',
        transform=ax.transAxes,ha='center',fontsize=7.5,color='#666',style='italic')
ax.legend(loc='upper left',fontsize=8.5,framealpha=0.9,edgecolor='#ddd')
ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2024-03-01'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'p6_rating_distribution.png'),
            dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Rating distribution saved')


# ── VIZ 7: Fund risk report (no fund name) ────────────────────────────────────
print('\n→ [7/7] Risk report...')

basel_df=pd.read_csv(os.path.join(OUTPUT_DIR,'basel_results.csv'))
latest_basel=basel_df[basel_df['quarter']==latest_q]
el_sec=(latest_basel.groupby('sector')
        .agg(total_EL=('EL','sum'),total_EAD=('EAD','sum'))
        .reset_index().sort_values('total_EL',ascending=True))

top5=scorecard[scorecard['quarter']==latest_q].nlargest(5,'composite_score')

fig=plt.figure(figsize=(16,10))
fig.suptitle(f'Credit Risk Report  |  {latest_q}  |  Phase 6',
             fontsize=13,fontweight='600',color=C_ACC,y=1.01)

gs=fig.add_gridspec(2,3,hspace=0.45,wspace=0.35)

# Table
ax1=fig.add_subplot(gs[0,:2])
col_labels=['Ticker','Sector','PD%','D2D','Rating','Score','Alert']
table_data=[]
for _,r in top5.iterrows():
    d2d_str=f'{r["D2D"]:.3f}' if pd.notna(r.get('D2D')) else 'N/A'
    table_data.append([r['ticker'],r['sector'],
                        f'{r["PD"]*100:.2f}%',d2d_str,
                        r['internal_rating'],
                        f'{r["composite_score"]:.1f}',
                        r['ews_alert']])
table=ax1.table(cellText=table_data,colLabels=col_labels,
                cellLoc='center',loc='center',bbox=[0,0,1,1])
table.auto_set_font_size(False); table.set_fontsize(9)
for (row,col),cell in table.get_celld().items():
    if row==0:
        cell.set_facecolor(C_ACC); cell.set_text_props(color='white',fontweight='bold')
    elif row%2==0: cell.set_facecolor('#F5F5F5')
    if col==6 and row>0:
        alert=table_data[row-1][6]
        cell.set_facecolor(
            '#7B0000' if alert=='CRITICAL' else C_DIST if alert=='HIGH'
            else '#E8B820' if alert=='MEDIUM' else '#F0F7EC')
        cell.set_text_props(color='white' if alert in ['CRITICAL','HIGH'] else 'black')
ax1.axis('off')
ax1.set_title('Top 5 Highest Risk Companies',fontsize=10,fontweight='500',
              color=C_ACC,pad=8)

# KPIs
ax2=fig.add_subplot(gs[0,2])
ax2.axis('off')
kpis=[
    ('Portfolio EAD',   f'${lp["total_EAD"]/1e3:,.1f}B'),
    ('Expected Loss',   f'${lp["total_EL"]/1e3:.3f}B  ({lp["EL_rate"]:.3f}%)'),
    ('Economic Capital',f'${lp["total_EC"]/1e3:.2f}B'),
    ('CVA Exposure',    f'${lp["total_CVA"]/1e3:.3f}B'),
    ('Mean PD',         f'{lp["mean_PD"]*100:.3f}%'),
    ('Mean D2D',        f'{lp["mean_D2D"]:.3f}'),
    ('GREEN / AMBER',   f'{lp["n_green"]:.0f} / {lp["n_amber"]:.0f}'),
    ('RED / CRITICAL',  f'{lp["n_red"]:.0f} / {lp["n_critical"]:.0f}'),
]
y_s=0.95
for label,value in kpis:
    ax2.text(0.02,y_s,f'{label}:',fontsize=9,color='#555',
             transform=ax2.transAxes,va='top')
    ax2.text(0.98,y_s,value,fontsize=9,fontweight='bold',color=C_ACC,
             transform=ax2.transAxes,va='top',ha='right')
    y_s-=0.115
ax2.set_title('Portfolio KPIs',fontsize=10,fontweight='500',color=C_ACC,pad=8)
ax2.add_patch(plt.Rectangle((0,0),1,1,transform=ax2.transAxes,
                              fill=True,facecolor='#F8F9FA',
                              edgecolor='#DEE2E6',linewidth=1))

# Pie
ax3=fig.add_subplot(gs[1,0])
tl_counts=[lp['n_green'],lp['n_amber'],lp['n_red'],lp['n_critical']]
tl_labels=[f'GREEN\n({lp["n_green"]:.0f})',f'AMBER\n({lp["n_amber"]:.0f})',
            f'RED\n({lp["n_red"]:.0f})',f'CRITICAL\n({lp["n_critical"]:.0f})']
wedges,_,autotexts=ax3.pie(
    tl_counts,labels=tl_labels,colors=[C_SAFE,C_GREY,C_DIST,'#7B0000'],
    autopct='%1.0f%%',startangle=90,textprops={'fontsize':8},
    wedgeprops={'edgecolor':'white','linewidth':2})
for at in autotexts:
    at.set_color('white'); at.set_fontweight('bold')
ax3.set_title(f'Traffic Light — {latest_q}',fontsize=10,fontweight='500',color=C_ACC)

# EL trend
ax4=fig.add_subplot(gs[1,1])
recent=port_q.tail(8)
ax4.bar(range(len(recent)),recent['EL_rate'],
        color=[C_DIST if r>0.05 else C_GREY if r>0.01 else C_SAFE
               for r in recent['EL_rate']],
        alpha=0.80,edgecolor='white',linewidth=0.5)
ax4.set_xticks(range(len(recent)))
ax4.set_xticklabels(recent['quarter'],rotation=45,fontsize=7.5)
ax4.set_ylabel('EL Rate (%)',fontsize=9)
ax4.set_title('EL Rate — Last 8 Quarters',fontsize=10,fontweight='500',color=C_ACC)
for i,v in enumerate(recent['EL_rate']):
    ax4.text(i,v+0.0005,f'{v:.3f}%',ha='center',fontsize=7,color='#444')
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

# Sector EL
ax5=fig.add_subplot(gs[1,2])
bar_c=[sec_colors_p.get(s,'#888') for s in el_sec['sector']]
ax5.barh(range(len(el_sec)),el_sec['total_EL'],
         color=bar_c,alpha=0.80,edgecolor='white',linewidth=0.5)
ax5.set_yticks(range(len(el_sec)))
ax5.set_yticklabels(el_sec['sector'],fontsize=8.5)
ax5.set_xlabel('Total EL ($M)',fontsize=9)
ax5.set_title(f'EL by Sector — {latest_q}',fontsize=10,fontweight='500',color=C_ACC)
for i,v in enumerate(el_sec['total_EL']):
    if v>0:
        ax5.text(v+10,i,f'${v:,.0f}M',va='center',fontsize=7.5,color='#444')
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

plt.savefig(os.path.join(OUTPUT_DIR,'p6_fund_risk_report.png'),
            dpi=160,bbox_inches='tight')
plt.show(); print('  ✓ Risk report saved')

print(f'\n{"="*55}')
print(f'ALL CHARTS REBUILT — fund name removed from all')
print(f'{"="*55}')
for f in ['p6_scorecard_heatmap.html','p6_portfolio_risk_time.png',
          'p6_migration_matrix.png','p6_ews_dashboard.png',
          'p6_sector_concentration.png','p6_rating_distribution.png',
          'p6_fund_risk_report.png']:
    path=os.path.join(OUTPUT_DIR,f)
    if os.path.exists(path):
        print(f'  ✓ {f:<45} {os.path.getsize(path)/1024:6.1f} KB')
print(f'\n  ALL 6 PHASES COMPLETE')
print(f'  → Project P4 Credit Risk Modeling — DONE')


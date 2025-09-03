
#!/usr/bin/env python3
# Inventory simulation using (s, S) policy with service level targets.
from pathlib import Path
import argparse, pandas as pd, numpy as np
from math import sqrt
from scipy.stats import norm

def calc_safety_stock(daily_std, lead_time_days, service_level):
    z = norm.ppf(service_level)
    return z * daily_std * np.sqrt(lead_time_days)

def run_sim(sales_df, inv_params):
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    last_date = sales_df['date'].max()
    recent = sales_df[sales_df['date'] > last_date - pd.Timedelta(days=180)]
    agg = recent.groupby(['store_id','sku_id'])['sales_qty'].agg(['mean','std']).reset_index().fillna(0.1)
    df = inv_params.merge(agg, on='sku_id', how='left').fillna({'mean':2.0,'std':1.0})
    df['s'] = (df['mean']*df['lead_time_days'] + df.apply(lambda r: calc_safety_stock(r['std'], r['lead_time_days'], r['service_level']), axis=1)).round()
    df['S'] = (df['s'] + np.maximum(10, (2*df['mean']).round())).round()
    kpis = []
    for sku, g in recent.groupby('sku_id'):
        demand_series = g.groupby('date')['sales_qty'].sum().reindex(pd.date_range(last_date - pd.Timedelta(days=89), last_date, freq='D'), fill_value=0)
        row = df[df['sku_id']==sku].iloc[0]
        inv = row['S']; s, S = row['s'], row['S']
        orders = 0; stockouts = 0; onhand_hist = []
        lt = int(row['lead_time_days'])
        pipeline = [0]*lt
        for dqty in demand_series.values:
            arriving = pipeline.pop(0) if pipeline else 0
            inv += arriving
            if dqty > inv:
                stockouts += (dqty - inv); inv = 0
            else:
                inv -= dqty
            onhand_hist.append(inv)
            if inv <= s:
                order_qty = int(S - inv); orders += 1
                if lt>0:
                    while len(pipeline)<lt: pipeline.append(0)
                    pipeline[-1] += order_qty
                else:
                    inv += order_qty
        kpis.append({'sku_id': sku, 'avg_onhand': float(np.mean(onhand_hist)), 'stockouts_units': int(stockouts), 'orders_placed': int(orders)})
    return pd.DataFrame(kpis)

def main(args):
    sales = pd.read_csv(args.sales_csv)
    inv = pd.read_csv(args.inv_csv)
    kpis = run_sim(sales, inv)
    Path('outputs').mkdir(exist_ok=True, parents=True)
    kpis.to_csv(args.output_csv, index=False)
    print(kpis.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sales_csv', default='data/retail_sales.csv')
    parser.add_argument('--inv_csv', default='data/inventory_params.csv')
    parser.add_argument('--output_csv', default='outputs/inventory_sim_kpis.csv')
    args = parser.parse_args()
    main(args)

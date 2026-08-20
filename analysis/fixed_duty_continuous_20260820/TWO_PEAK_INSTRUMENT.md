# Synthetic SE3-shaped two-peak tariff instrument

Status: **synthetic software instrument pending a frozen real tariff series**.
It is not an observed tariff and its outputs are not research results.

## Construction

`two_peak_se3_synthetic_h26.csv` is an explicit hourly, temporal tariff in
model-currency/kWh:

- hours 00–05: deep overnight trough;
- hours 07–11: broad morning peak, centered on hour 09;
- hours 12–15: inter-peak shoulder;
- hours 16–21: broader, higher evening peak, centered on hour 19;
- hours 22–23: decline toward the next overnight trough;
- hours 24–26: exact repetition of hours 00–02 for the model's 26-hour
  horizon.

The unscaled 24-hour shape multipliers are:

```text
0.45, 0.38, 0.32, 0.28, 0.30, 0.42,
0.70, 1.05, 1.35, 1.45, 1.25, 1.05,
0.90, 0.82, 0.85, 0.95, 1.15, 1.40,
1.65, 1.75, 1.55, 1.20, 0.85, 0.60
```

They are multiplied by `0.10501985002205559`, making the first 24 hours'
arithmetic mean exactly `0.0992`, equal to the tracked flat instrument. This
holds average price fixed while changing only intraday shape. The resulting
minimum is `0.029405558006176`, the maximum is `0.183784737538597`, and the
max/min ratio is `6.25`.

## Provenance and limits

The geometry is deliberately stylized rather than fitted to a selected day:
no historical observations or price levels were copied. It is shaped after
the commonly observed Nordic/SE3 intraday pattern of lower overnight prices,
a morning demand ramp, and a stronger evening peak.

Market provenance for that interpretation:

1. [Nord Pool day-ahead price calculation](https://www.nordpoolgroup.com/en/trading/Day-ahead-trading/Price-calculation/)
   states that bidding-area prices are calculated for each delivery hour from
   hourly purchase and sell orders, with congestion producing distinct area
   prices.
2. [Nord Pool day-ahead prices](https://data.nordpoolgroup.com/auction/day-ahead/prices?aggregation=DeliveryPeriod&currency=EUR&deliveryAreas=SE3&deliveryDate=latest)
   is the official delivery-period portal for the SE3 bidding area.
3. [ENTSO-E SE3 market description](https://entsoemcp.com/day-ahead/se-3)
   identifies Sweden 3/Stockholm (`SE_3`, EIC `10Y1001A1001A46L`) as an
   hourly day-ahead series sourced from the ENTSO-E Transparency Platform.

Sources were consulted on 2026-08-20. They establish market, bidding-zone, and
hourly-series provenance; they do **not** make this constructed curve observed
data. A publication claim requires replacing this instrument with a frozen,
hashed SE3 series and documented date-selection rule.

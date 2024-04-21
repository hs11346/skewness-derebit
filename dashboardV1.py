import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import scipy.interpolate as interpolate
from scipy.stats import norm
import streamlit as st


start = time.time()
async def call_api(msg):
    async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
        await websocket.send(msg)
        while websocket.open:
            response = await websocket.recv()
            json_par = json.loads(response)
            return (json_par)


def N(z):
    """ Normal cumulative density function

    :param z: point at which cumulative density is calculated
    :return: cumulative density under normal curve
    """
    return norm.cdf(z)


def call_delta(S, K, r, t, vol):
    """ Black-Scholes call delta

    :param S: underlying
    :param K: strike price
    :param r: rate
    :param t: time to expiration
    :param vol: volatility
    :return: call delta
    """
    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S / K) + (r + 0.5 * vol ** 2.0) * t)

    return N(d1)


def put_delta(S, K, r, t, vol):
    """ Black-Scholes put delta

    :param S: underlying
    :param K: strike price
    :param r: rate
    :param t: time to expiration
    :param vol: volatility
    :return: put delta
    """
    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S / K) + (r + 0.5 * vol ** 2.0) * t)

    return N(d1) - 1.0


def calculate_delta_from_derebit(contract):
    if contract.cp == 'C':
        return call_delta(contract.underlying_price, contract.strike, contract.interest_rate, contract.ttm,
                          contract.mark_iv / 100)
    elif contract.cp == 'P':
        return put_delta(contract.underlying_price, contract.strike, contract.interest_rate, contract.ttm,
                         contract.mark_iv / 100)
def plotting(final, calls_spline, graph1, graph2):
    fig, ax1 = plt.subplots()

    # Plot the first line using the primary y-axis
    ax1.plot(final.time, final['1 week skewness'], 'g-', label='Skewness')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Skewness', color='g')

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Plot the second line using the secondary y-axis
    ax2.plot(final.time, final['BTC price'], 'b-', label='BTC price')
    ax2.set_ylabel('BTC price', color='b')

    # Add a legend
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)

    graph1.pyplot(fig)
    calls['fitted_iv'] = calls.apply(lambda row: calls_spline.ev(row.cal_delta, row.ttm).item(), axis=1)
    size = (10, 24)
    plt.rcParams.update({'font.size': 8})
    fig2 = plt.figure(figsize=size)
    i = 1
    plt.tight_layout()
    for expiry in calls.sort_values(by='ttm').expiry.unique():
        plt.subplot(5, 2, i)
        tmp = calls[calls.expiry == expiry]
        plt.title("IV smile against delta at " + expiry)
        plt.scatter(tmp.cal_delta, tmp.mark_iv, label='actual')
        plt.plot(tmp.sort_values(by='cal_delta').cal_delta, tmp.sort_values(by='cal_delta').fitted_iv,
                 label='fitted', color='orange')
        plt.axvline(0.25)
        plt.axvline(0.75)
        plt.legend()
        i = i + 1
    graph2.pyplot(fig2)
    time.sleep(15)

def main():
    msg = \
        {
            "jsonrpc": "2.0",
            "id": 7538,
            "method": "public/get_book_summary_by_currency",
            "params": {
                "currency": "BTC",
                "kind": "option"
            }
        }
    st.title("Skewness Indicator from Derebit")
    st.markdown("This dashboard scraps Skewness data from Derebit API directly, with latency of around 1 second.")
    st.markdown("Most of the delay in the dashboard comes from the visualisation")
    tab1, tab2 = st.tabs(["Skew", "Fitting"])
    with tab1:
        st.header("Skewness")
        graph1 = st.empty()
    with tab2:
        st.header("Fitting in Call Options")
        graph2 = st.empty()

    final = pd.DataFrame()
    while True:
        response = asyncio.run(call_api(json.dumps(msg)))
        # print("Request sent at " + str(datetime.fromtimestamp(response['usIn'] / 1000000)))
        # print("Data received at " + str(datetime.fromtimestamp(response['usOut'] / 1000000)))
        # print("Request took (in seconds) " + str(response['usDiff'] / 1000000))
        df = pd.json_normalize(response['result'])
        df[["currency", "expiry", "strike", "cp"]] = df['instrument_name'].str.split('-', expand=True)

        df['strike'] = df['strike'].astype(int)
        df.dropna(how='all', inplace=True)
        df['ttm'] = None
        for i, row in df.iterrows():
            date_str = row.expiry
            date = datetime.strptime(date_str, '%d%b%y')
            target_time = datetime.combine(date.date(), datetime.strptime('08:00', '%H:%M').time())
            ttm = target_time - datetime.utcnow()
            df.at[i, 'ttm'] = (ttm.total_seconds() / 86400) / 365



        df['cal_delta'] = df.apply(calculate_delta_from_derebit, axis = 1)
        calls = df[(df.cp == 'C') & (df.ttm > 3/365)] # get options with TTM > 3 days
        puts = df[(df.cp == 'P') & (df.ttm > 3/365)] # get options with TTM > 3 days

        cx = np.array(calls.cal_delta)
        cy = np.array(calls.ttm)
        cz = np.array(calls.mark_iv)

        px = np.array(puts.cal_delta)
        py = np.array(puts.ttm)
        pz = np.array(puts.mark_iv)

        kx, ky = 3, 4  # spline order
        assert len(cx) >= (kx+1)*(ky+1)
        assert len(px) >= (kx+1)*(ky+1)
        calls_spline = interpolate.SmoothBivariateSpline(cx,cy,cz, kx=kx, ky=ky)
        puts_spline = interpolate.SmoothBivariateSpline(px,py,pz, kx=kx, ky=ky)
        tmp_xy = [[round(x, 2), y/365] for x in np.arange(0.2, 0.8, 0.05) for y in range(3, 181)]
        call_interpolate = pd.DataFrame(tmp_xy, columns = ['delta','ttm'])
        call_interpolate['iv'] = call_interpolate.apply(lambda row: calls_spline.ev(row.delta, row.ttm).item(), axis = 1)

        tmp_xy = [[round(x, 2), y/365] for x in np.arange(-0.8, -0.2, 0.05) for y in range(3, 181)]
        put_interpolate = pd.DataFrame(tmp_xy, columns = ['delta','ttm'])
        put_interpolate['iv'] = put_interpolate.apply(lambda row: puts_spline.ev(row.delta, row.ttm).item(), axis = 1)
        #print(puts_spline.ev(-0.25, 7/365).item() - calls_spline.ev(0.25, 7/365).item())
        final = pd.concat([final, pd.DataFrame([{"time": datetime.fromtimestamp(response['usOut'] / 1000000),
                                                 "1 week skewness": puts_spline.ev(-0.25,
                                                                                   7 / 365).item() - calls_spline.ev(
                                                     0.25, 7 / 365).item(),
                                                 "BTC price" : df.underlying_price.iloc[0]
                                                 }])],
                  axis = 0, ignore_index = True)
        final = final.tail(30)
        plotting(final, calls_spline, graph1, graph2)
if __name__ == '__main__':
    main()

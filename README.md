**Exploring SPY Expected Moves for Options Trading**

- Built an interactive Python dashboard that lets users see the probability of SPY expiring above any selected strike.

- Integrated live market data (SPY & VIX) and visualizations with Plotly for dynamic expected-move analysis.

**Note**:
1. **Volatility σ**: VIX as the proxy value for SPY options pricing.

2. **Probability Analysis**: The probability of expiring above one strike price on a certain day is measured by option deltas. e.g. for the options with strike price 677 expire on Oct. 31, the probability of expiring above 677 is 42.4%. 

3. Useful for **vertical spread trading**: For example, consider a long call spread 665/670 with a premium of 3, making the breakeven 668. This chart allows users to see the probability of SPY expiring above 668—that is, the probability of profit. With the max profit and max loss, we can also calculate the expected profit for this vertical spread.

4. **Next steps**: I plan to add more features to automatically calculate option prices, max loss/gain, breakeven points, and expected profit. Happy to connect and discuss!

<img width="2676" height="1420" alt="spy em" src="https://github.com/user-attachments/assets/e34811b5-de2c-4543-8423-31bb19b1f2eb" />
<img width="2806" height="966" alt="spy" src="https://github.com/user-attachments/assets/80dd1d18-f53b-4c9e-9a54-c3c1cad510a3" />


# TVLIndicator
A cryptocurrency indicator using defi TVL 

Fits a rolling linear model mapping the TVL to the closing price. The indicator is the difference between the actual close price and TVL predicted close price. This difference is volatility normalized by dividing by the average true range. The indicator will take on high magnitude values when the price deviates from the TVL. 

This script was created for this youtube video:
https://www.youtube.com/watch?v=9W5mczpSboE

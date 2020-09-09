# Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to
# predict the closing stock price of the stock you would like by using the past 90 day stock price.
# Import the libraries
from stock_predict import Predict


help = '''
start - Predict stock price.
quit - Exit program.
'''

start = 'Start predicting stocks!'
idk = "I don't recognize that command. Please enter another command or type 'help' for assistance"

command = ""
while True:
    command = input(
        "What would you like to do? For assistance type 'help.' ").lower()
    if command == "start":
        Predict.stock_predict()
    elif command == 'quit':
        break
    elif command == 'help':
        print(help)
    else:
        print(idk)

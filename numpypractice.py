import pandas as pd
import numpy as np




data = [[1, '2015-01-01', 10], [2, '2015-01-02', 25], [3, '2015-01-03', 20], [4, '2015-01-04', 30]]
weather = pd.DataFrame(data, columns=['id', 'recordDate', 'temperature']).astype({'id':'Int64', 'recordDate':'datetime64[ns]', 'temperature':'Int64'})

def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:
    ##print(weather)

    print(weather[((weather['temperature'].diff()>0) & (weather['recordDate'].diff().dt.days==1))][['id']])
    return weather

#rising_temperature(weather)



data = [[1, 2, '2016-03-01', 5], [1, 2, '2016-05-02', 6], [2, 3, '2017-06-25', 1], [3, 1, '2016-03-02', 0], [3, 4, '2018-07-03', 5]]
activity = pd.DataFrame(data, columns=['player_id', 'device_id', 'event_date', 'games_played']).astype({'player_id':'Int64', 'device_id':'Int64', 'event_date':'datetime64[ns]', 'games_played':'Int64'})


def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    first_login = activity.groupby('player_id', as_index=False).event_date.min()
    return first_login.rename(
        columns={
            'event_date': 'first_login'
        }
    )
game_analysis(activity)
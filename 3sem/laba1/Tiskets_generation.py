import faker
import pandas as pd
import tkinter as tk
from datetime import datetime
from dateutil.relativedelta import relativedelta

import utils

fake = faker.Faker('ru.RU')

initial_time = datetime(2024, 10, 1, 0, 0)
end_time = initial_time + relativedelta(days=1)

train_dictionary_df = pd.read_excel('../data/trainsbook.xlsx')

ticket_data=[]
cards_array=[]

root = tk.Tk()
app = utils.BankInputUI(root)
root.mainloop()
sys_prob, bank_prob = app.get_sys_prob(),app.get_bank_prob()

for n in range(1,len(train_dictionary_df)):
    train = utils.Route(n,train_dictionary_df,initial_time, ticket_data, sys_prob, bank_prob)
    while (train.position < end_time):
        train.swap(ticket_data,sys_prob, bank_prob)

# Создание DataFrame из списка данных
df = pd.DataFrame(ticket_data)
#Запись в Excel таблицу
df.to_excel('../data/Tickets.xlsx', index=False)
df.to_excel('../data/Tickets_anon.xlsx', index=False)

app.show_info('Данные сгенерированы, программа отработала штатно. Датасет сохранен в Tickets.xlsx')

















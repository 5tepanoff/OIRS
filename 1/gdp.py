import pandas
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # Стиль графиков

FILE_PATH = 'gdp.xls'
df = pandas.read_excel(FILE_PATH)

# Удаляем лишние столбцы. Иначе не сможем нарисовать график
df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', '2019'], axis=1, inplace=True)
df.set_index('Country Name').loc['United States'].plot()
df.set_index('Country Name').loc['Japan'].plot()
df.set_index('Country Name').loc['China'].plot()
df.set_index('Country Name').loc['Germany'].plot()
df.set_index('Country Name').loc['France'].plot()
plt.legend() # Каждая линия графика подписана
plt.show()

# сумма по годам для всех стран
df.set_index('Country Name').sum().plot()
plt.show()

# ## два столбца - года и сумма значений по всем странам
# a = df.set_index('Country Name').sum()
# print(a)


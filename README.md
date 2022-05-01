# Categorization-of-products-in-the-catalog-by-name
Categorization of products in the catalog by name

Решение тестового задания от [KazanExpress](https://kazanexpress.ru/) 🛒

## Задача

В маркетплейс каждый день поступает множество новых товаров и каждый из них необходимо отнести в определенную категорию в  дереве категорий. На это тратится много сил и времени, поэтому мы хотим научиться предсказывать категорию на основе названий и параметров товаров. 

### **Формат входных данных**

Данные расположены по ссылке: https://drive.google.com/drive/folders/194JOoKDZCkmpBglf7Fs7hlzk5xXJSYgI?usp=sharing .

Имеются три файла: 

**categories_tree.csv** - файл с деревом категорий на маркетплейсе. У каждой категории есть id, заголовок и parent_id, по которому можно восстановить полный путь категории.

Допустим у категории `2642` заголовок "Мелкие инструменты", а путь в дереве категорий - `10016->10072->10690->2642`. Если заменить id категорий в этом пути на заголовки, то получим следующее дерево:

- Строительство и ремонт
    - Ручной инструмент и оснастка
        - Столярно-слесарные инструменты
            - Мелкие инструменты

**train.parquet** - файл с товарами на маркетплейсе. 
У каждого товара есть:

- *id* - идентификатор товара
- *title - заголовок*
- *short_description - краткое описание*
- *name_value_characteristics - название:значение* характеристики товара, может быть несколько для одного товара и для одной характеристик. Пример: `name1: value1 | value2 | valueN_1 / name2: value1 | value2 | valueN_2 / nameK: value1 | value2 | valueN_K`
- *rating - средний рейтинг товара*
- *feedback_quantity - количество отзывов по товару*
- *category_id - категория товара(таргет)*

**test.parquet** - файл идентичный **train.parquet**, но без реального *category_id*, именно его вам и предстоит предсказать.


### Метрики

Предсказывать листовую категорию товара может показаться тривиальной задачей, однако также важен путь к листовой категории. Для неверно предсказанных листовых категорий мы хотим добавлять скор, если были угаданы предшествующие родительские категории на разных уровнях.

Для честной оценки многоуровневой классификации мы будем использовать видоизмененную метрику **F1**, взвешенную на размер выборки класса(при подсчете учитываются только листовые категории):

![Untitled](https://github.com/Irinas5555/Categorization-of-products-in-the-catalog-by-name/blob/main/Untitled.png)

**Pi -** это набор, состоящий из классов, предсказанных для каждого сэмпла i и соотвествующих классов-предков

**Ti -** это набор, состоящий из истинных классов тестового сэмпла i и соотвествующих классов-предков

**hP** - иерархический precision

**hR** - иерархический recall

**hF** - иерархический F1 

Проще понять подсчет данных метрик поможет следующий пример и [статья](https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/DMKD-J-2010-Silla.pdf)**.**

Пример расчета **F1** score для листовой категории **“Шины”.**
![Untitled](https://github.com/Irinas5555/Categorization-of-products-in-the-catalog-by-name/blob/main/Screenshot_2022-04-08_at_12.05.04.png)

### Решение

Для решения задачи использовалась предобученная нейронная сеть ("cointegrated/rubert-tiny2") с последующим дообучением весов под конкретную задачу классификации. 
- models.py содержит классы для формирования датасета и модели на torch,
- utils.py содержит необходимые функции для обучения модели и расчета метрик,
- best_model.pt - сохраненная лучшая модель.

В итоговом решении использован Local Classifier per Node (LCN), то есть  обучение происходило только с учетом конечных (листовых) категорий товаров.
В ходе экспериментов пробовала обучать multi-head nets, которые учитывали категории более высоких уровней, но данные подходы не дали увеличения качества классификации.

Возможные пути дальнейшего улучшения модели:
 - использование других, более сложных предобученных трансформеров (при наличии достаточных вычислительный ресурсов),
 - попытаться построить ансамбль с предсказаниями других типов моделей (например, catboost classifier).

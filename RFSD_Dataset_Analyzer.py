import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Tuple
from enum import Enum

# НАСТРОЙКИ ДЛЯ ПОЛНОГО ВЫВОДА В ФАЙЛ
pl.Config.set_tbl_width_chars(1000)  # Ширина таблицы
pl.Config.set_tbl_cols(50)     # Максимальное количество колонок для показа
pl.Config.set_tbl_rows(100)    # Количество строк для показа
pl.Config.set_fmt_str_lengths(100)

# Настройка размеров шрифтов для всех графиков
plt.rcParams.update({
    'font.size': 6,          # Основной размер шрифта
    'axes.titlesize': 10,    # Размер заголовков графиков
    'axes.labelsize': 6,     # Размер подписей осей
    'xtick.labelsize': 6,    # Размер подписей по оси X
    'ytick.labelsize': 6,    # Размер подписей по оси Y
    'legend.fontsize': 6,    # Размер шрифта в легенде
    'figure.titlesize': 10   # Размер заголовка figure
})

class CompanySize(Enum):
    MICRO = "Микро"
    SMALL = "Малые" 
    MEDIUM = "Средние"
    LARGE = "Крупные"
    ALL = "Все"

class DataConfig:
    """Класс конфигурации для фильтрации датасета"""
    
    def __init__(
        self,
        company_size: CompanySize = CompanySize.ALL,
        exclude_outliers: bool = True,
        exclude_zero_revenue: bool = True,
        fill_missing_values: bool = True
    ):
        self.company_size = company_size
        self.exclude_outliers = exclude_outliers
        self.exclude_zero_revenue = exclude_zero_revenue
        self.fill_missing_values = fill_missing_values
    
    def get_size_filter(self) -> Optional[pl.Expr]:
        """Возвращает выражение для фильтрации по размеру компании"""
        if self.company_size == CompanySize.ALL:
            return None
            
        size_filters = {
            CompanySize.MICRO: pl.col("line_2110") < 120_000_000,
            CompanySize.SMALL: (pl.col("line_2110") >= 120_000_000) & (pl.col("line_2110") < 800_000_000),
            CompanySize.MEDIUM: (pl.col("line_2110") >= 800_000_000) & (pl.col("line_2110") < 2_000_000_000),
            CompanySize.LARGE: pl.col("line_2110") >= 2_000_000_000
        }
        
        return size_filters[self.company_size]

class RFSDAnalyzer:
    """Анализатор датасета RFSD"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.df = None
        
        # Определение колонок для загрузки
        self.service_columns = [
            "inn", "outlier"
        ]
        
        self.analysis_columns = [
            "line_2110", "line_2400", "line_2200", "line_1600", "line_1300",
            "line_1250", "line_1230", "line_1210", "line_1520", "line_1400",
            "line_2120", "line_2210", "line_2220", "line_4121", "line_4100",
            "line_1370", "line_1180", "line_1150", "line_1240", "line_1510"
        ]
        
        self.column_mapping = {
            # Служебные колонки
            "inn": "ИНН",
            "outlier": "Аномалия",
            
            # Финансовые показатели
            "line_2110": "Выручка",
            "line_2400": "Чистая_прибыль",
            "line_2200": "Прибыль_от_продаж",
            "line_1600": "Активы_всего",
            "line_1300": "Капитал_и_резервы",
            "line_1250": "Денежные_средства",
            "line_1230": "Дебиторская_задолженность",
            "line_1210": "Запасы",
            "line_1520": "Кредиторская_задолженность", 
            "line_1400": "Долгосрочные_обязательства",
            "line_2120": "Себестоимость",
            "line_2210": "Коммерческие_расходы",
            "line_2220": "Управленческие_расходы",
            "line_4121": "Денежный_поток_от_продаж",
            "line_4100": "Денежный_поток_операционный",
            "line_1370": "Нераспределенная_прибыль",
            "line_1180": "Отложенные_налоговые_активы",
            "line_1150": "Основные_средства",
            "line_1240": "Финансовые_вложения", 
            "line_1510": "Заемные_средства"
        }
    
    def load_data(self, limit: int = 1000) -> pl.DataFrame:
        """Загрузка данных с фильтрацией и лимитом"""
        
        # Базовые фильтры
        filters = []
        
        if self.config.exclude_outliers:
            filters.append(pl.col("outlier") == False)
            
        if self.config.exclude_zero_revenue:
            filters.append(pl.col("line_2110").is_not_null())
            filters.append(pl.col("line_2110") > 0)
        
        # Фильтр по размеру компании
        size_filter = self.config.get_size_filter()
        if size_filter is not None:
            filters.append(size_filter)
        
        # Комбинируем все фильтры
        combined_filter = None
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
        
        # Загрузка данных
        all_columns = self.service_columns + self.analysis_columns
        
        try:
            if combined_filter is not None:
                self.df = pl.read_parquet(
                    'RFSD/year=2024/part-0.parquet',
                    columns=all_columns
                ).filter(combined_filter).head(limit)
            else:
                self.df = pl.read_parquet(
                    'RFSD/year=2024/part-0.parquet',
                    columns=all_columns
                ).head(limit)
                
            print(f"Успешно загружено {len(self.df)} записей")
            return self.df
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None
    
    def rename_columns(self) -> pl.DataFrame:
        """Переименование колонок в человеко-понятные имена"""
        if self.df is None:
            print("Данные не загружены. Сначала выполните load_data()")
            return None
        
        self.df = self.df.rename(self.column_mapping)
        print("Колонки успешно переименованы")
        return self.df
    
    def print_descriptive_stats(self):
        """Вывод описательной статистики"""
        if self.df is None:
            print("Данные не загружены")
            return
        
        print("=" * 60)
        print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА ВЫБОРКИ")
        print("=" * 60)
        
        # Базовая информация о датафрейме
        print(f"Размер выборки: {len(self.df)} записей")
        print(f"Количество колонок: {len(self.df.columns)}")
        
        # Статистика по числовым колонкам (финансовые показатели)
        financial_columns = [col for col in self.df.columns 
                           if col not in ['ИНН', 'Аномалия']]
        
        if financial_columns:
            stats = self.df.select(financial_columns).describe()
            print("\nСтатистика по финансовым показателям (в рублях):")
            print(stats)
    
    def fill_missing_values(self):
        """Заполнение пропущенных значений"""
        if self.df is None:
            print("Данные не загружены")
            return
        
        if not self.config.fill_missing_values:
            print("Заполнение пропусков отключено в конфигурации")
            return
        
        print("\nЗаполнение пропущенных значений...")
        
        # Анализ пропусков до заполнения
        missing_before = self.df.null_count()
        print("Пропуски до заполнения:")
        
        for col_name in missing_before.columns:
            count = missing_before[col_name][0]
            if count > 0:
                percentage = (count / len(self.df)) * 100
                print(f"  {col_name}: {count} пропусков ({percentage:.1f}%)")
        
        # Стратегии заполнения для разных типов колонок
        fill_expressions = []
        
        for col in self.df.columns:
            if col == 'ИНН':
                # ИНН - заполняем 'Не указано'
                fill_expressions.append(pl.col(col).fill_null('Не указано'))
            elif col == 'Аномалия':
                # Булева колонка - заполняем False
                fill_expressions.append(pl.col(col).fill_null(False))
            else:
                # Числовые колонки - заполняем медианой
                median_val = self.df[col].median()
                if median_val is not None:
                    fill_expressions.append(pl.col(col).fill_null(median_val))
                else:
                    # Если медиана не определена (все значения null), заполняем 0
                    fill_expressions.append(pl.col(col).fill_null(0))
        
        # Применяем все выражения заполнения
        if fill_expressions:
            self.df = self.df.with_columns(fill_expressions)
        
        # Проверяем результат
        missing_after = self.df.null_count()
        remaining_missing = sum(missing_after.row(0))
        
        if remaining_missing == 0:
            print("✅ Все пропуски успешно заполнены")
        else:
            print(f"⚠️ Осталось {remaining_missing} пропусков после заполнения")
    
    def analyze_selected_columns(self):
        """Анализ выбранных колонок: Выручка, Чистая прибыль, Активы"""
        if self.df is None:
            print("Данные не загружены")
            return
        
        selected_columns = ['Выручка', 'Чистая_прибыль', 'Активы_всего']
        
        print("\n" + "=" * 60)
        print("АНАЛИЗ ВЫБРАННЫХ КОЛОНОК")
        print("=" * 60)
        
        # 1. Построение гистограмм
        self._plot_histograms(selected_columns)
        
        # 2. Анализ выбросов и аномалий
        self._analyze_outliers(selected_columns)
    
    def _plot_histograms(self, columns: List[str]):
        """Построение гистограмм для выбранных колонок"""
        print("\n1. ПОСТРОЕНИЕ ГИСТОГРАММ")
        
        # Конвертируем в pandas для seaborn
        df_pandas = self.df.select(columns).to_pandas()
        
        # Создаем subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, col in enumerate(columns):
            # Логарифмируем данные для лучшей визуализации (исключая нули и отрицательные)
            data = df_pandas[col]
            log_data = np.log10(np.abs(data) + 1) * np.sign(data)  # Сохраняем знак для прибыли
            
            # Строим гистограмму
            sns.histplot(data=log_data, ax=axes[i], bins=30, kde=True)
            axes[i].set_title(f'Распределение {col} (лог. шкала)', fontsize=10)
            axes[i].set_xlabel(f'log10({col})', fontsize=6)
            axes[i].set_ylabel('Частота', fontsize=6)
            axes[i].tick_params(axis='both', which='major', labelsize=6)
            
            # Добавляем вертикальную линию для медианы
            median_val = log_data.median()
            axes[i].axvline(median_val, color='red', linestyle='--', 
                           label=f'Медиана: {median_val:.2f}')
            axes[i].legend(fontsize=6)
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительно: гистограммы в оригинальном масштабе (ограниченные)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, col in enumerate(columns):
            data = df_pandas[col]
            
            # Ограничиваем диапазон для лучшей визуализации (95-й перцентиль)
            p95 = data.quantile(0.95)
            filtered_data = data[data <= p95]
            
            sns.histplot(data=filtered_data, ax=axes[i], bins=30, kde=True)
            axes[i].set_title(f'Распределение {col} (до 95-го перцентиля)', fontsize=10)
            axes[i].set_xlabel(f'{col} (руб)', fontsize=6)
            axes[i].set_ylabel('Частота', fontsize=6)
            axes[i].tick_params(axis='both', which='major', labelsize=6)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_outliers(self, columns: List[str]):
        """Анализ выбросов для выбранных колонок"""
        print("\n2. АНАЛИЗ ВЫБРОСОВ И АНОМАЛИЙ")
        
        for col in columns:
            print(f"\n--- Анализ {col} ---")
            data = self.df[col]
            
            # Базовая статистика
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Считаем выбросы
            outliers_low = self.df.filter(pl.col(col) < lower_bound).height
            outliers_high = self.df.filter(pl.col(col) > upper_bound).height
            total_outliers = outliers_low + outliers_high
            
            print(f"Границы выбросов (IQR): [{lower_bound:,.0f}, {upper_bound:,.0f}]")
            print(f"Выбросы (нижние): {outliers_low} ({outliers_low/len(data)*100:.1f}%)")
            print(f"Выбросы (верхние): {outliers_high} ({outliers_high/len(data)*100:.1f}%)")
            print(f"Всего выбросов: {total_outliers} ({total_outliers/len(data)*100:.1f}%)")
            
            # Анализ экстремальных значений
            if outliers_high > 0:
                top_5 = self.df.select(['ИНН', col]).sort(col, descending=True).head(5)
                print(f"Топ-5 максимальных значений {col}:")
                for row in top_5.rows():
                    inn, value = row
                    print(f"  ИНН {inn}: {value:,.0f} руб.")
            
            # Логические аномалии (специфичные для каждого показателя)
            if col == 'Выручка':
                negative_revenue = self.df.filter(pl.col(col) < 0).height
                print(f"Отрицательная выручка: {negative_revenue} компаний")
            
            elif col == 'Чистая_прибыль':
                # Прибыль/убыток больше выручки
                profit_exceeds_revenue = self.df.filter(
                    (pl.col(col).abs() > pl.col('Выручка')) & 
                    (pl.col('Выручка') > 0)
                ).height
                print(f"Прибыль/убыток > выручки: {profit_exceeds_revenue} компаний")
                
                # Экстремальная рентабельность (>1000% или <-1000%)
                extreme_profitability = self.df.filter(
                    (pl.col('Выручка') > 0) &
                    ((pl.col(col) / pl.col('Выручка')).abs() > 10)
                ).height
                print(f"Экстремальная рентабельность (>1000% или <-1000%): {extreme_profitability} компаний")
            
            elif col == 'Активы_всего':
                # Нулевые активы при наличии выручки
                zero_assets_with_revenue = self.df.filter(
                    (pl.col(col) == 0) & 
                    (pl.col('Выручка') > 0)
                ).height
                print(f"Нулевые активы при наличии выручки: {zero_assets_with_revenue} компаний")
        
        # Визуализация выбросов с помощью boxplot
        print("\n3. ВИЗУАЛИЗАЦИЯ ВЫБРОСОВ (Boxplot)")
        
        df_pandas = self.df.select(columns).to_pandas()
        
        # Boxplot в логарифмической шкале
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, col in enumerate(columns):
            data = df_pandas[col]
            log_data = np.log10(np.abs(data) + 1) * np.sign(data)
            
            sns.boxplot(y=log_data, ax=axes[i])
            axes[i].set_title(f'Выбросы {col} (лог. шкала)', fontsize=10)
            axes[i].set_ylabel(f'log10({col})', fontsize=6)
            axes[i].tick_params(axis='both', which='major', labelsize=6)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_correlations(self):
        """Анализ корреляций между всеми финансовыми колонками"""
        if self.df is None:
            print("Данные не загружены")
            return
        
        print("\n" + "=" * 60)
        print("АНАЛИЗ КОРРЕЛЯЦИЙ МЕЖДУ ФИНАНСОВЫМИ ПОКАЗАТЕЛЯМИ")
        print("=" * 60)
        
        # 1. Вычисляем матрицу корреляции
        self._compute_correlation_matrix()
        
        # 2. Выбираем наиболее коррелирующие пары
        top_correlations = self._find_top_correlations()
        
        # 3. Строим диаграммы рассеивания
        self._plot_correlation_scatterplots(top_correlations)
    
    def _compute_correlation_matrix(self):
        """Вычисление и визуализация матрицы корреляции"""
        print("\n1. МАТРИЦА КОРРЕЛЯЦИИ")
        
        # Выбираем только финансовые колонки (исключаем служебные)
        financial_columns = [col for col in self.df.columns 
                           if col not in ['ИНН', 'Аномалия']]
        
        # Конвертируем в pandas для вычисления корреляции
        df_financial = self.df.select(financial_columns)
        df_pandas = df_financial.to_pandas()
        
        # Вычисляем матрицу корреляции
        correlation_matrix = df_pandas.corr()
        
        # Визуализируем тепловую карту
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Маска для верхнего треугольника
        
        heatmap = sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt=".2f", 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={"shrink": .8},
                   annot_kws={"size": 6})  # Размер шрифта для аннотаций
        
        plt.title('Матрица корреляции финансовых показателей', fontsize=10, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        
        # Настройка colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)
        
        plt.tight_layout()
        plt.show()
        
        # Сохраняем матрицу для дальнейшего использования
        self.correlation_matrix = correlation_matrix
        self.financial_columns = financial_columns
    
    def _find_top_correlations(self, n_pairs: int = 5) -> List[Tuple[str, str, float]]:
        """Нахождение пар колонок с наибольшей корреляцией"""
        print("\n2. НАИБОЛЕЕ КОРРЕЛИРУЮЩИЕ ПАРЫ ПОКАЗАТЕЛЕЙ")
        
        correlation_pairs = []
        
        # Собираем все пары корреляций (исключая диагональ)
        for i in range(len(self.financial_columns)):
            for j in range(i + 1, len(self.financial_columns)):
                col1 = self.financial_columns[i]
                col2 = self.financial_columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                # Исключаем NaN значения
                if not np.isnan(corr_value):
                    correlation_pairs.append((col1, col2, abs(corr_value), corr_value))
        
        # Сортируем по абсолютному значению корреляции
        correlation_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Выбираем топ пар
        top_pairs = correlation_pairs[:n_pairs]
        
        print(f"Топ-{n_pairs} наиболее коррелирующих пар:")
        for i, (col1, col2, abs_corr, corr) in enumerate(top_pairs, 1):
            direction = "положительная" if corr > 0 else "отрицательная"
            print(f"{i}. {col1} ↔ {col2}: {corr:.3f} ({direction})")
        
        return [(col1, col2, corr) for col1, col2, _, corr in top_pairs]
    
    def _plot_correlation_scatterplots(self, top_correlations: List[Tuple[str, str, float]]):
        """Построение диаграмм рассеивания для коррелирующих пар"""
        print("\n3. ДИАГРАММЫ РАССЕИВАНИЯ ДЛЯ КОРРЕЛИРУЮЩИХ ПАР")
        
        df_pandas = self.df.select(self.financial_columns).to_pandas()
        
        # Создаем subplot для scatter plots
        n_pairs = len(top_correlations)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (col1, col2, corr) in enumerate(top_correlations):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Логарифмируем данные для лучшей визуализации
            x_data = np.log10(np.abs(df_pandas[col1]) + 1) * np.sign(df_pandas[col1])
            y_data = np.log10(np.abs(df_pandas[col2]) + 1) * np.sign(df_pandas[col2])
            
            # Убираем бесконечные значения, которые могут появиться при логарифмировании
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_data_clean = x_data[valid_mask]
            y_data_clean = y_data[valid_mask]
            
            # Простой scatter plot без вычисления плотности
            scatter = ax.scatter(x_data_clean, y_data_clean, alpha=0.6, s=30, 
                               color='blue', edgecolors='white', linewidth=0.5)
            
            # Настройки графика
            ax.set_xlabel(f'log10({col1})', fontsize=6)
            ax.set_ylabel(f'log10({col2})', fontsize=6)
            ax.set_title(f'{col1} vs {col2}\nКорреляция: {corr:.3f}', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            # Добавляем линию тренда только если есть достаточно точек
            if len(x_data_clean) > 2:
                try:
                    z_coef = np.polyfit(x_data_clean, y_data_clean, 1)
                    p = np.poly1d(z_coef)
                    
                    # Создаем точки для линии тренда
                    x_trend = np.linspace(x_data_clean.min(), x_data_clean.max(), 100)
                    y_trend = p(x_trend)
                    
                    ax.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2,
                           label=f'y = {z_coef[0]:.2f}x + {z_coef[1]:.2f}')
                    ax.legend(fontsize=6)
                except Exception as e:
                    print(f"Не удалось построить линию тренда для {col1} vs {col2}: {e}")
        
        # Удаляем лишние subplots
        for idx in range(n_pairs, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    def get_dataframe(self) -> pl.DataFrame:
        """Возвращает текущий датафрейм"""
        return self.df

# Демонстрация использования
if __name__ == "__main__":
    # Создаем конфигурацию
    config = DataConfig(
        company_size=CompanySize.MICRO,
        exclude_outliers=False,
        exclude_zero_revenue=True, 
        fill_missing_values=False
    )
    
    # Создаем анализатор
    analyzer = RFSDAnalyzer(config)
    
    # 1. Загружаем данные
    print("1. ЗАГРУЗКА ДАННЫХ")
    analyzer.load_data(limit=1000)
    
    # 2. Переименовываем колонки
    print("\n2. ПЕРЕИМЕНОВАНИЕ КОЛОНОК")
    analyzer.rename_columns()
    
    # 3. Выводим статистику
    print("\n3. СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    analyzer.print_descriptive_stats()
    
    # 4. Заполняем пропуски
    print("\n4. ОБРАБОТКА ПРОПУСКОВ")
    analyzer.fill_missing_values()
    
    # 5. Анализ выбранных колонок
    print("\n5. АНАЛИЗ КЛЮЧЕВЫХ ПОКАЗАТЕЛЕЙ")
    analyzer.analyze_selected_columns()
    
    # 6. Анализ корреляций
    print("\n6. АНАЛИЗ КОРРЕЛЯЦИЙ")
    analyzer.analyze_correlations()
    
    # Финальная информация
    if analyzer.df is not None:
        print(f"\nФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        print(f"Размер датафрейма: {len(analyzer.df)} строк {len(analyzer.df.columns)} колонок")